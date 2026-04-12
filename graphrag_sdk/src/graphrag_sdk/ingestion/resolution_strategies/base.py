# GraphRAG SDK — Ingestion: Resolution Strategy ABC
# Pattern: Strategy — different deduplication approaches implement this interface.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, GraphNode, GraphRelationship, ResolutionResult

if TYPE_CHECKING:
    from graphrag_sdk.core.providers import LLMInterface


_SUMMARY_PROMPT = (
    "Summarise the following descriptions of the entity '{entity_name}' "
    "into a single concise description (max {max_tokens} tokens).\n\n"
    "Descriptions:\n{descriptions}\n\n"
    "Summary:"
)

_CROSS_LABEL_PROMPT = (
    "Entities with the name \"{name}\" exist under different types in the "
    "knowledge graph:\n\n"
    "{entries}\n\n"
    "Do ALL of these refer to the same real-world entity?\n"
    "If YES, which single type is most accurate?\n\n"
    "Answer with exactly: YES <chosen_type> or NO\n"
    "Then on a new line give a brief reason (max 20 words).\n\n"
    "Answer:"
)


async def exact_match_merge(
    nodes: list[GraphNode],
    llm: LLMInterface | None,
    *,
    force_summary_threshold: int = 3,
    max_summary_tokens: int = 500,
    cross_label_merge: bool = False,
) -> tuple[list[GraphNode], dict[str, str], int]:
    """Phase 1: group nodes by (normalized_name, label) and merge exact duplicates.

    Merges description strings (optionally via LLM batch summarisation) and
    accumulates source_chunk_ids. Properties from duplicates are copied to the
    survivor only when the key is absent.

    Returns:
        (deduplicated_nodes, id_remap, merged_count)
    """
    groups: dict[tuple[str, str], list[GraphNode]] = defaultdict(list)
    for node in nodes:
        name = node.properties.get("name", node.id)
        key = (str(name).strip().lower(), node.label)
        groups[key].append(node)

    id_remap: dict[str, str] = {}
    deduplicated_nodes: list[GraphNode] = []
    merged_count = 0

    summary_requests: list[tuple[int, str, list[str]]] = []
    group_data: list[tuple[GraphNode, list[GraphNode], list[str], list[str]]] = []

    for _key, group_nodes in groups.items():
        survivor = group_nodes[0]
        deduplicated_nodes.append(survivor)

        if len(group_nodes) == 1:
            group_data.append((survivor, group_nodes, [], []))
            continue

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

        group_data.append((survivor, group_nodes, descriptions, all_source_ids))

        if descriptions and len(descriptions) >= force_summary_threshold and llm is not None:
            entity_name = str(survivor.properties.get("name", survivor.id))
            summary_requests.append((len(group_data) - 1, entity_name, descriptions))

    # ── Cross-label detection ───────────────────────────────────────
    # Find same-name entities under different labels (e.g. "FalkorDB"
    # as Technology vs Organisation vs Unknown).
    cross_label_requests: list[tuple[int, str, list[GraphNode]]] = []
    if cross_label_merge and llm is not None:
        name_to_survivors: dict[str, list[GraphNode]] = defaultdict(list)
        for node in deduplicated_nodes:
            norm = str(node.properties.get("name", node.id)).strip().lower()
            name_to_survivors[norm].append(node)

        for norm_name, candidates in name_to_survivors.items():
            if len(candidates) < 2:
                continue
            labels = {c.label for c in candidates}
            if len(labels) < 2:
                continue
            # One prompt per cross-label group; index offset after summaries
            cross_label_requests.append((
                len(summary_requests) + len(cross_label_requests),
                str(candidates[0].properties.get("name", norm_name)),
                candidates,
            ))

    # ── Single LLM batch call (summaries + cross-label verifications) ─
    prompts: list[str] = [
        _SUMMARY_PROMPT.format(
            entity_name=en,
            max_tokens=max_summary_tokens,
            descriptions="\n".join(f"- {d}" for d in descs),
        )
        for _, en, descs in summary_requests
    ]
    for _, name, candidates in cross_label_requests:
        entries = "\n".join(
            f"- Type: {c.label}, Description: "
            f"{c.properties.get('description', '(no description)')}"
            for c in candidates
        )
        prompts.append(_CROSS_LABEL_PROMPT.format(name=name, entries=entries))

    batch_results: list = []
    summary_results: dict[int, str] = {}
    if prompts and llm is not None:
        batch_results = await llm.abatch_invoke(prompts)

    # Process summary results
    n_summaries = len(summary_requests)
    for item in batch_results:
        if item.index >= n_summaries:
            continue  # cross-label result, handled below
        group_idx, _, descs = summary_requests[item.index]
        if item.ok:
            summary_results[group_idx] = item.response.content.strip()
        else:
            summary_results[group_idx] = " | ".join(descs)

    # ── Apply same-label merges (unchanged logic) ────────────────────
    for gi, (survivor, group_nodes, descriptions, all_source_ids) in enumerate(group_data):
        if len(group_nodes) == 1:
            continue
        if descriptions:
            survivor.properties["description"] = (
                summary_results[gi] if gi in summary_results else " | ".join(descriptions)
            )
        if all_source_ids:
            survivor.properties["source_chunk_ids"] = all_source_ids
        for duplicate in group_nodes[1:]:
            for key, value in duplicate.properties.items():
                if key not in survivor.properties:
                    survivor.properties[key] = value
            id_remap[duplicate.id] = survivor.id
            merged_count += 1

    # ── Apply cross-label merges ─────────────────────────────────────
    if cross_label_requests and batch_results:
        result_map = {item.index: item for item in batch_results}
        merged_away: set[str] = set()

        for prompt_idx, _name, candidates in cross_label_requests:
            item = result_map.get(prompt_idx)
            if item is None or not item.ok:
                continue
            first_line = item.response.content.strip().split("\n")[0].strip()
            if not first_line.upper().startswith("YES"):
                continue

            # Extract canonical type from "YES Technology"
            parts = first_line.split(None, 1)
            canonical = parts[1].strip() if len(parts) >= 2 else ""

            # Find survivor: node whose label matches canonical (case-insensitive)
            survivor = None
            for c in candidates:
                if c.label.lower() == canonical.lower():
                    survivor = c
                    break
            if survivor is None:
                survivor = candidates[0]  # fallback: first node

            # Merge losers into survivor
            for loser in candidates:
                if loser.id == survivor.id or loser.id in merged_away:
                    continue
                # Merge descriptions
                surv_desc = str(survivor.properties.get("description", ""))
                loser_desc = str(loser.properties.get("description", ""))
                if loser_desc and loser_desc not in surv_desc:
                    survivor.properties["description"] = (
                        f"{surv_desc} | {loser_desc}" if surv_desc else loser_desc
                    )
                # Merge source_chunk_ids
                surv_src = survivor.properties.get("source_chunk_ids", [])
                if not isinstance(surv_src, list):
                    surv_src = []
                for sid in loser.properties.get("source_chunk_ids", []):
                    if sid not in surv_src:
                        surv_src.append(sid)
                survivor.properties["source_chunk_ids"] = surv_src
                # Copy any unique properties
                for key, value in loser.properties.items():
                    if key not in survivor.properties:
                        survivor.properties[key] = value

                id_remap[loser.id] = survivor.id
                merged_away.add(loser.id)
                merged_count += 1

        # Remove merged nodes from deduplicated_nodes
        if merged_away:
            deduplicated_nodes = [n for n in deduplicated_nodes if n.id not in merged_away]

        # Resolve transitive remap chains (Phase 1 mapped X→B, cross-label B→A → X→A)
        for k in list(id_remap.keys()):
            target = id_remap[k]
            seen = {k}
            while target in id_remap and target not in seen:
                seen.add(target)
                target = id_remap[target]
            id_remap[k] = target

    return deduplicated_nodes, id_remap, merged_count


def remap_relationships(
    relationships: list[GraphRelationship],
    id_remap: dict[str, str],
) -> list[GraphRelationship]:
    """Remap relationship endpoints using id_remap and deduplicate."""
    deduplicated_rels: list[GraphRelationship] = []
    seen_rels: set[tuple[str, str, str]] = set()
    for rel in relationships:
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
    return deduplicated_rels


class ResolutionStrategy(ABC):
    """Abstract base class for entity resolution (deduplication) strategies.

    A resolution strategy receives extracted graph data and merges
    duplicate entities, producing a deduplicated result.

    Example::

        class VectorFuzzyResolution(ResolutionStrategy):
            async def resolve(self, graph_data, ctx):
                # Use embeddings to find near-duplicate entities
                ...
    """

    @abstractmethod
    async def resolve(
        self,
        graph_data: GraphData,
        ctx: Context,
    ) -> ResolutionResult:
        """Resolve duplicate entities in the extracted graph data.

        Args:
            graph_data: Extracted nodes and relationships.
            ctx: Execution context.

        Returns:
            ResolutionResult with deduplicated data and merge statistics.
        """
        ...
