# GraphRAG SDK 2.0 — Ingestion: Semantic Resolution
# Groups entities by (normalized_name, label) to prevent cross-type merges,
# then uses embeddings for near-duplicate detection within same-label groups.

from __future__ import annotations

import logging
from collections import defaultdict

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

_SUMMARY_PROMPT = (
    "Summarise the following descriptions of the entity '{entity_name}' "
    "into a single concise description (max {max_tokens} tokens).\n\n"
    "Descriptions:\n{descriptions}\n\n"
    "Summary:"
)


class SemanticResolution(ResolutionStrategy):
    """Deduplicate entities by (normalized_name, label) grouping with
    optional embedding-based near-duplicate detection.

    Key improvement over DescriptionMergeResolution: groups by
    ``(name.lower().strip(), label)`` so entities with the same name but
    different labels (e.g., Person "Paris" vs Location "Paris") are kept
    separate.

    Optionally uses an embedder to detect near-duplicates within the same
    label group (e.g., "J.R.R. Tolkien" ≈ "Tolkien") above a similarity
    threshold.

    Args:
        llm: LLM provider (used for description summarisation).
        embedder: Embedder for near-duplicate detection (optional).
        similarity_threshold: Cosine similarity threshold for fuzzy merge
            (default: 0.95 — very conservative).
        force_summary_threshold: Number of descriptions that triggers LLM
            summary instead of simple concatenation.
        max_summary_tokens: Token budget hint for the LLM prompt.
    """

    def __init__(
        self,
        llm: LLMInterface | None = None,
        embedder: Embedder | None = None,
        *,
        similarity_threshold: float = 0.95,
        force_summary_threshold: int = 3,
        max_summary_tokens: int = 500,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.force_summary_threshold = force_summary_threshold
        self.max_summary_tokens = max_summary_tokens

    async def resolve(
        self,
        graph_data: GraphData,
        ctx: Context,
    ) -> ResolutionResult:
        ctx.log(
            f"SemanticResolution: resolving {len(graph_data.nodes)} nodes, "
            f"{len(graph_data.relationships)} rels"
        )

        # Phase 1: Group by (normalized_name, label) — prevents cross-type merges
        groups: dict[tuple[str, str], list[GraphNode]] = defaultdict(list)
        for node in graph_data.nodes:
            name = node.properties.get("name", node.id)
            key = (str(name).strip().lower(), node.label)
            groups[key].append(node)

        # Phase 2: Build ID remap from exact-name groups
        id_remap: dict[str, str] = {}
        deduplicated_nodes: list[GraphNode] = []
        merged_count = 0

        # Collect groups needing LLM summary
        summary_requests: list[tuple[int, str, list[str]]] = []
        group_data: list[tuple[GraphNode, list[GraphNode], list[str], list[str]]] = []

        for _key, nodes in groups.items():
            survivor = nodes[0]
            deduplicated_nodes.append(survivor)

            if len(nodes) == 1:
                group_data.append((survivor, nodes, [], []))
                continue

            descriptions: list[str] = []
            all_source_ids: list[str] = []
            for n in nodes:
                desc = n.properties.get("description", "")
                if desc:
                    descriptions.append(str(desc))
                src_ids = n.properties.get("source_chunk_ids", [])
                if isinstance(src_ids, list):
                    for sid in src_ids:
                        if sid not in all_source_ids:
                            all_source_ids.append(sid)

            group_data.append((survivor, nodes, descriptions, all_source_ids))

            if (
                descriptions
                and len(descriptions) >= self.force_summary_threshold
                and self.llm is not None
            ):
                entity_name = str(survivor.properties.get("name", survivor.id))
                summary_requests.append(
                    (len(group_data) - 1, entity_name, descriptions)
                )

        # Batch LLM summaries
        summary_results: dict[int, str] = {}
        if summary_requests and self.llm is not None:
            prompts = [
                _SUMMARY_PROMPT.format(
                    entity_name=entity_name,
                    max_tokens=self.max_summary_tokens,
                    descriptions="\n".join(f"- {d}" for d in descs),
                )
                for _, entity_name, descs in summary_requests
            ]
            batch_results = await self.llm.abatch_invoke(prompts)
            for item in batch_results:
                group_idx, entity_name, descs = summary_requests[item.index]
                if item.ok:
                    summary_results[group_idx] = item.response.content.strip()
                else:
                    summary_results[group_idx] = " | ".join(descs)

        # Apply merges
        for gi, (survivor, nodes, descriptions, all_source_ids) in enumerate(group_data):
            if len(nodes) == 1:
                continue

            if descriptions:
                if gi in summary_results:
                    survivor.properties["description"] = summary_results[gi]
                else:
                    survivor.properties["description"] = " | ".join(descriptions)

            if all_source_ids:
                survivor.properties["source_chunk_ids"] = all_source_ids

            for duplicate in nodes[1:]:
                for key, value in duplicate.properties.items():
                    if key not in survivor.properties:
                        survivor.properties[key] = value
                id_remap[duplicate.id] = survivor.id
                merged_count += 1

        # Phase 3: Optional embedding-based fuzzy merge within same label
        if self.embedder and len(deduplicated_nodes) >= 2:
            fuzzy_remap = await self._fuzzy_merge(deduplicated_nodes, ctx)
            if fuzzy_remap:
                # Apply fuzzy merges
                final_nodes: list[GraphNode] = []
                for node in deduplicated_nodes:
                    if node.id not in fuzzy_remap:
                        final_nodes.append(node)
                    else:
                        merged_count += 1
                id_remap.update(fuzzy_remap)
                deduplicated_nodes = final_nodes

        # Remap relationship endpoints and deduplicate
        deduplicated_rels: list[GraphRelationship] = []
        seen_rels: set[tuple[str, str, str]] = set()

        for rel in graph_data.relationships:
            start = id_remap.get(rel.start_node_id, rel.start_node_id)
            end = id_remap.get(rel.end_node_id, rel.end_node_id)
            rel_key = (start, rel.type, end)

            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                deduplicated_rels.append(
                    GraphRelationship(
                        start_node_id=start,
                        end_node_id=end,
                        type=rel.type,
                        properties=rel.properties,
                    )
                )

        ctx.log(
            f"SemanticResolution complete: {len(deduplicated_nodes)} nodes "
            f"({merged_count} merged), {len(deduplicated_rels)} rels"
        )
        return ResolutionResult(
            nodes=deduplicated_nodes,
            relationships=deduplicated_rels,
            merged_count=merged_count,
        )

    async def _fuzzy_merge(
        self,
        nodes: list[GraphNode],
        ctx: Context,
    ) -> dict[str, str]:
        """Detect near-duplicate entities within same-label groups via embeddings.

        Returns a remap dict: duplicate_id → survivor_id.
        """
        # Group by label
        by_label: dict[str, list[GraphNode]] = defaultdict(list)
        for n in nodes:
            by_label[n.label].append(n)

        remap: dict[str, str] = {}

        for label, label_nodes in by_label.items():
            if len(label_nodes) < 2:
                continue

            names = [n.properties.get("name", n.id) for n in label_nodes]
            try:
                vectors = await self.embedder.aembed_documents(
                    [str(name) for name in names]
                )
            except Exception:
                continue

            # Filter out failed embeddings
            valid = [
                (i, node, vec)
                for i, (node, vec) in enumerate(zip(label_nodes, vectors))
                if vec is not None
            ]
            if len(valid) < 2:
                continue

            indices, valid_nodes, vecs = zip(*valid)
            mat = np.array(vecs, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat_normed = mat / norms

            # Find pairs above threshold (upper triangle only)
            # Use Union-Find to cluster
            parent: dict[int, int] = {i: i for i in range(len(valid_nodes))}

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x: int, y: int) -> None:
                px, py = find(x), find(y)
                if px != py:
                    parent[py] = px

            n = len(valid_nodes)
            BLOCK_SIZE = 500
            for i_start in range(0, n, BLOCK_SIZE):
                i_end = min(i_start + BLOCK_SIZE, n)
                block = mat_normed[i_start:i_end]
                remaining = mat_normed[i_start:]
                sim_block = block @ remaining.T
                local_rows, local_cols = np.where(sim_block >= self.similarity_threshold)
                for lr, lc in zip(local_rows.tolist(), local_cols.tolist()):
                    gi = i_start + lr
                    gj = i_start + lc
                    if gj > gi:
                        union(gi, gj)

            # Build clusters
            clusters: dict[int, list[int]] = defaultdict(list)
            for i in range(n):
                clusters[find(i)].append(i)

            for root, members in clusters.items():
                if len(members) <= 1:
                    continue
                survivor = valid_nodes[members[0]]
                for mi in members[1:]:
                    dup = valid_nodes[mi]
                    remap[dup.id] = survivor.id
                    # Merge properties
                    for key, value in dup.properties.items():
                        if key not in survivor.properties:
                            survivor.properties[key] = value

            if remap:
                ctx.log(
                    f"Fuzzy merge on label '{label}': "
                    f"{len(remap)} duplicates found"
                )

        return remap
