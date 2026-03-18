# GraphRAG SDK 2.0 — Ingestion: Embedding-Clustered LLM Resolution
# Clusters entities by embedding cosine similarity, then uses LLM to decide
# which cluster members are true duplicates. Handles alias variants like
# "Angeline" / "Angeline S. Hall" / "Angeline Hall" that exact-name dedup misses.

from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any

import numpy as np

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    GraphData,
    GraphNode,
    GraphRelationship,
    ResolutionResult,
)
from graphrag_sdk.core.providers import Embedder, LLMInterface
from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy

logger = logging.getLogger(__name__)

_CLUSTER_DEDUP_PROMPT = (
    "You are an entity resolution expert for a knowledge graph.\n\n"
    "Below is a cluster of entities that have similar names. Your job is to decide "
    "which entities refer to the SAME real-world thing and should be merged.\n\n"
    "## Entities in this cluster\n"
    "{entity_list}\n\n"
    "## Instructions\n"
    "- Group the entities that are the SAME thing (aliases, abbreviations, "
    "or name variants of the same entity).\n"
    "- For each group, pick the BEST canonical name — prefer the most complete, "
    "properly capitalized form.\n"
    "- Entities that are DIFFERENT things must remain separate even if names are "
    "similar (e.g., 'Don Carlo' and 'Dan Carlo' may be different characters).\n"
    "- Use the descriptions to help decide — entities about different topics are "
    "different even if names overlap.\n\n"
    "## Output Format\n"
    "Return one line per merge group. Each line: the canonical name, then a pipe, "
    "then comma-separated aliases.\n"
    "Only include groups with 2+ members (skip singletons).\n\n"
    "Example:\n"
    "Gen. Ulysses S. Grant | Grant, Gen. Grant, U.S. Grant\n"
    "Angeline Stickney Hall | Angeline, Angeline Hall, Angeline S. Hall\n\n"
    "If NO entities in this cluster should be merged, return: NONE\n"
)

_DESC_SUMMARY_PROMPT = (
    "Summarise the following descriptions of the entity '{entity_name}' "
    "into a single concise description (max {max_tokens} tokens).\n\n"
    "Descriptions:\n{descriptions}\n\n"
    "Summary:"
)


class EmbeddingClusterResolution(ResolutionStrategy):
    """Deduplicate entities using embedding clustering + LLM verification.

    Three-phase approach:
    1. **Exact-name dedup** — same (normalized_name, label) → merge directly.
    2. **Embedding clustering** — embed entity names, group by cosine
       similarity above ``cluster_threshold``.
    3. **LLM verification** — send each cluster to LLM in batch, which decides
       true duplicates vs. false positives. Merges confirmed duplicates.

    Args:
        llm: LLM provider for dedup decisions and description summarization.
        embedder: Embedder for entity name similarity.
        cluster_threshold: Cosine similarity threshold for clustering candidates
            (default 0.85). Lower than standard fuzzy merge (0.95) because the
            LLM makes the final call.
        max_cluster_size: Maximum entities per LLM cluster prompt (default 15).
        force_summary_threshold: Number of descriptions triggering LLM summary.
        max_summary_tokens: Token budget for description summaries.
    """

    def __init__(
        self,
        llm: LLMInterface,
        embedder: Embedder,
        *,
        cluster_threshold: float = 0.85,
        max_cluster_size: int = 15,
        force_summary_threshold: int = 3,
        max_summary_tokens: int = 500,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.cluster_threshold = cluster_threshold
        self.max_cluster_size = max_cluster_size
        self.force_summary_threshold = force_summary_threshold
        self.max_summary_tokens = max_summary_tokens

    async def resolve(
        self,
        graph_data: GraphData,
        ctx: Context,
    ) -> ResolutionResult:
        ctx.log(
            f"EmbeddingClusterResolution: resolving {len(graph_data.nodes)} nodes, "
            f"{len(graph_data.relationships)} rels"
        )

        # ── Phase 1: Exact-name dedup ──
        id_remap, deduplicated_nodes, merged_count = self._exact_name_dedup(
            graph_data.nodes, ctx,
        )

        # ── Phase 2+3: Embedding cluster + LLM verification ──
        cluster_remap = await self._embedding_cluster_dedup(
            deduplicated_nodes, ctx,
        )
        if cluster_remap:
            # Remove merged nodes and update remap
            surviving_ids = set()
            for dup_id, surv_id in cluster_remap.items():
                surviving_ids.add(surv_id)
            final_nodes = [
                n for n in deduplicated_nodes if n.id not in cluster_remap
            ]
            id_remap.update(cluster_remap)
            merged_count += len(cluster_remap)
            deduplicated_nodes = final_nodes

        # ── Remap relationship endpoints and deduplicate ──
        deduplicated_rels = self._remap_relationships(
            graph_data.relationships, id_remap,
        )

        ctx.log(
            f"EmbeddingClusterResolution complete: {len(deduplicated_nodes)} nodes "
            f"({merged_count} merged), {len(deduplicated_rels)} rels"
        )
        return ResolutionResult(
            nodes=deduplicated_nodes,
            relationships=deduplicated_rels,
            merged_count=merged_count,
        )

    # ──────────────────────────────────────────────────────────────
    # Phase 1: Exact-name dedup (same as DescriptionMergeResolution)
    # ──────────────────────────────────────────────────────────────

    def _exact_name_dedup(
        self,
        nodes: list[GraphNode],
        ctx: Context,
    ) -> tuple[dict[str, str], list[GraphNode], int]:
        """Group by (normalized_name, label), merge descriptions."""
        groups: dict[tuple[str, str], list[GraphNode]] = defaultdict(list)
        for node in nodes:
            name = node.properties.get("name", node.id)
            key = (str(name).strip().lower(), node.label)
            groups[key].append(node)

        id_remap: dict[str, str] = {}
        deduplicated: list[GraphNode] = []
        merged_count = 0

        for _key, group_nodes in groups.items():
            survivor = group_nodes[0]
            deduplicated.append(survivor)

            if len(group_nodes) == 1:
                continue

            # Merge descriptions and source_chunk_ids
            descriptions: list[str] = []
            all_source_ids: list[str] = []
            for n in group_nodes:
                desc = n.properties.get("description", "")
                if desc:
                    descriptions.append(str(desc))
                src_ids = n.properties.get("source_chunk_ids", [])
                if isinstance(src_ids, list):
                    for sid in src_ids:
                        if sid not in all_source_ids:
                            all_source_ids.append(sid)

            if descriptions:
                survivor.properties["description"] = " | ".join(descriptions)
            if all_source_ids:
                survivor.properties["source_chunk_ids"] = all_source_ids

            # Prefer properly-capitalized name
            for n in group_nodes[1:]:
                n_name = n.properties.get("name", "")
                s_name = survivor.properties.get("name", "")
                if n_name and s_name and n_name[0].isupper() and not s_name[0].isupper():
                    survivor.properties["name"] = n_name

            for duplicate in group_nodes[1:]:
                for prop_key, value in duplicate.properties.items():
                    if prop_key not in survivor.properties:
                        survivor.properties[prop_key] = value
                id_remap[duplicate.id] = survivor.id
                merged_count += 1

        ctx.log(f"Phase 1 (exact-name): {merged_count} merges")
        return id_remap, deduplicated, merged_count

    # ──────────────────────────────────────────────────────────────
    # Phase 2+3: Embedding cluster + LLM verification
    # ──────────────────────────────────────────────────────────────

    async def _embedding_cluster_dedup(
        self,
        nodes: list[GraphNode],
        ctx: Context,
    ) -> dict[str, str]:
        """Cluster by embedding similarity, verify with LLM, return remap."""
        if len(nodes) < 2:
            return {}

        # Group by label — only cluster within same label
        by_label: dict[str, list[GraphNode]] = defaultdict(list)
        for n in nodes:
            by_label[n.label].append(n)

        all_remap: dict[str, str] = {}

        for label, label_nodes in by_label.items():
            if len(label_nodes) < 2:
                continue

            remap = await self._cluster_label_group(label, label_nodes, ctx)
            all_remap.update(remap)

        return all_remap

    async def _cluster_label_group(
        self,
        label: str,
        nodes: list[GraphNode],
        ctx: Context,
    ) -> dict[str, str]:
        """Cluster + LLM verify for a single label group."""
        names = [str(n.properties.get("name", n.id)) for n in nodes]

        try:
            vectors = await self.embedder.aembed_documents(names)
        except Exception as e:
            ctx.log(
                f"Embedding failed for label '{label}' ({len(nodes)} entities): {e}",
                logging.WARNING,
            )
            return {}

        # Filter out failed embeddings
        valid_indices = [i for i, v in enumerate(vectors) if v]
        if len(valid_indices) < 2:
            return {}

        valid_nodes = [nodes[i] for i in valid_indices]
        valid_names = [names[i] for i in valid_indices]
        mat = np.array([vectors[i] for i in valid_indices], dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat_normed = mat / norms

        # Union-Find clustering
        n = len(valid_nodes)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Block-wise similarity to handle large groups
        BLOCK_SIZE = 500
        for i_start in range(0, n, BLOCK_SIZE):
            i_end = min(i_start + BLOCK_SIZE, n)
            block = mat_normed[i_start:i_end]
            remaining = mat_normed[i_start:]
            sim_block = block @ remaining.T
            local_rows, local_cols = np.where(sim_block >= self.cluster_threshold)
            for lr, lc in zip(local_rows.tolist(), local_cols.tolist()):
                gi = i_start + lr
                gj = i_start + lc
                if gj > gi:
                    union(gi, gj)

        # Build clusters
        clusters: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(i)

        # Filter to multi-member clusters
        multi_clusters = [
            members for members in clusters.values() if len(members) >= 2
        ]

        if not multi_clusters:
            return {}

        ctx.log(
            f"Label '{label}': {len(multi_clusters)} candidate clusters "
            f"({sum(len(c) for c in multi_clusters)} entities)"
        )

        # ── Phase 3: LLM verification ──
        # Build prompts for each cluster
        prompts: list[str] = []
        cluster_nodes_list: list[list[int]] = []

        for members in multi_clusters:
            # Cap cluster size for prompt
            capped = members[: self.max_cluster_size]
            entity_lines = []
            for idx in capped:
                node = valid_nodes[idx]
                name = valid_names[idx]
                desc = str(node.properties.get("description", ""))[:200]
                entity_lines.append(f"- {name}: {desc}" if desc else f"- {name}")

            prompt = _CLUSTER_DEDUP_PROMPT.format(
                entity_list="\n".join(entity_lines)
            )
            prompts.append(prompt)
            cluster_nodes_list.append(capped)

        # Batch LLM calls
        batch_results = await self.llm.abatch_invoke(prompts)

        # Parse LLM responses and build remap
        remap: dict[str, str] = {}
        for item in batch_results:
            if not item.ok or item.response is None:
                continue
            cluster_members = cluster_nodes_list[item.index]
            merge_groups = self._parse_dedup_response(
                item.response.content,
                cluster_members,
                valid_nodes,
                valid_names,
            )
            for survivor_idx, duplicate_indices in merge_groups:
                survivor = valid_nodes[survivor_idx]
                for dup_idx in duplicate_indices:
                    dup = valid_nodes[dup_idx]
                    remap[dup.id] = survivor.id
                    # Merge properties into survivor
                    self._merge_node_properties(survivor, dup)

        if remap:
            ctx.log(f"Label '{label}': LLM confirmed {len(remap)} merges")

        return remap

    def _parse_dedup_response(
        self,
        content: str,
        cluster_members: list[int],
        all_nodes: list[GraphNode],
        all_names: list[str],
    ) -> list[tuple[int, list[int]]]:
        """Parse LLM dedup response into merge groups.

        Returns list of (survivor_idx, [duplicate_indices]).
        """
        content = content.strip()
        if content.upper() == "NONE" or not content:
            return []

        # Build name→index lookup for this cluster
        name_to_idx: dict[str, int] = {}
        for idx in cluster_members:
            name_to_idx[all_names[idx].strip().lower()] = idx

        merge_groups: list[tuple[int, list[int]]] = []

        for line in content.split("\n"):
            line = line.strip()
            if not line or line.upper() == "NONE":
                continue

            # Parse "Canonical Name | alias1, alias2, alias3"
            if "|" not in line:
                continue

            parts = line.split("|", 1)
            canonical_raw = parts[0].strip()
            aliases_raw = parts[1].strip()

            # Find the canonical entity
            canonical_key = canonical_raw.lower()
            canonical_idx = name_to_idx.get(canonical_key)

            # If canonical name doesn't match directly, try fuzzy match
            if canonical_idx is None:
                for name_key, idx in name_to_idx.items():
                    if canonical_key in name_key or name_key in canonical_key:
                        canonical_idx = idx
                        break

            if canonical_idx is None:
                continue

            # Parse aliases
            duplicate_indices: list[int] = []
            for alias in aliases_raw.split(","):
                alias = alias.strip()
                if not alias:
                    continue
                alias_key = alias.lower()
                alias_idx = name_to_idx.get(alias_key)
                if alias_idx is None:
                    # Try fuzzy
                    for name_key, idx in name_to_idx.items():
                        if alias_key in name_key or name_key in alias_key:
                            alias_idx = idx
                            break
                if alias_idx is not None and alias_idx != canonical_idx:
                    duplicate_indices.append(alias_idx)

            if duplicate_indices:
                # Update the survivor's name to the canonical form
                all_nodes[canonical_idx].properties["name"] = canonical_raw
                merge_groups.append((canonical_idx, duplicate_indices))

        return merge_groups

    @staticmethod
    def _merge_node_properties(survivor: GraphNode, duplicate: GraphNode) -> None:
        """Merge duplicate's properties into survivor."""
        # Merge descriptions
        surv_desc = str(survivor.properties.get("description", ""))
        dup_desc = str(duplicate.properties.get("description", ""))
        if dup_desc and dup_desc not in surv_desc:
            if surv_desc:
                survivor.properties["description"] = f"{surv_desc} | {dup_desc}"
            else:
                survivor.properties["description"] = dup_desc

        # Merge source_chunk_ids
        surv_ids = survivor.properties.get("source_chunk_ids", [])
        dup_ids = duplicate.properties.get("source_chunk_ids", [])
        if isinstance(surv_ids, list) and isinstance(dup_ids, list):
            for sid in dup_ids:
                if sid not in surv_ids:
                    surv_ids.append(sid)
            survivor.properties["source_chunk_ids"] = surv_ids

        # Copy other properties not already present
        for key, value in duplicate.properties.items():
            if key not in ("name", "description", "source_chunk_ids"):
                if key not in survivor.properties:
                    survivor.properties[key] = value

    @staticmethod
    def _remap_relationships(
        relationships: list[GraphRelationship],
        id_remap: dict[str, str],
    ) -> list[GraphRelationship]:
        """Remap relationship endpoints and deduplicate."""
        deduplicated: list[GraphRelationship] = []
        seen: set[tuple[str, str, str]] = set()

        for rel in relationships:
            start = id_remap.get(rel.start_node_id, rel.start_node_id)
            end = id_remap.get(rel.end_node_id, rel.end_node_id)

            # Skip self-loops created by merging
            if start == end:
                continue

            rel_key = (start, rel.type, end)
            if rel_key not in seen:
                seen.add(rel_key)
                deduplicated.append(
                    GraphRelationship(
                        start_node_id=start,
                        end_node_id=end,
                        type=rel.type,
                        properties=rel.properties,
                    )
                )

        return deduplicated
