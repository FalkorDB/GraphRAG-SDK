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

_SUMMARY_WITH_TYPE_PROMPT = (
    "Summarise the following descriptions of the entity '{entity_name}' "
    "into a single concise description (max {max_tokens} tokens).\n\n"
    "The descriptions come from entities classified under different types: "
    "{types}. Pick the single most accurate type.\n\n"
    "Descriptions:\n{descriptions}\n\n"
    "Answer on the first line with ONLY the chosen type, then on the next "
    "line provide the summary.\n\n"
    "Type:\n"
)


def _pick_canonical_label(nodes: list[GraphNode]) -> str:
    """Heuristic label selection: most frequent non-Unknown label wins."""
    counts: dict[str, int] = defaultdict(int)
    for n in nodes:
        counts[n.label] += 1
    # Prefer any specific type over "Unknown"
    candidates = {k: v for k, v in counts.items() if k != "Unknown"}
    if not candidates:
        return "Unknown"
    return max(candidates, key=lambda k: candidates[k])


async def exact_match_merge(
    nodes: list[GraphNode],
    llm: LLMInterface | None,
    *,
    force_summary_threshold: int = 3,
    max_summary_tokens: int = 500,
    cross_label_merge: bool = False,
) -> tuple[list[GraphNode], dict[str, str], int]:
    """Phase 1: group nodes by normalized name and merge exact duplicates.

    When *cross_label_merge* is False (default), groups by
    ``(normalized_name, label)`` — only same-type duplicates merge.

    When *cross_label_merge* is True, groups by ``(normalized_name,)`` so
    that same-name entities under different labels also merge. For groups
    with mixed labels and enough descriptions, the summary prompt is
    enhanced to also select the canonical type. For smaller groups, a
    heuristic picks the most frequent non-Unknown label.

    Returns:
        (deduplicated_nodes, id_remap, merged_count)
    """
    groups: dict[tuple, list[GraphNode]] = defaultdict(list)
    for node in nodes:
        name = node.properties.get("name", node.id)
        norm = str(name).strip().lower()
        key = (norm,) if cross_label_merge else (norm, node.label)
        groups[key].append(node)

    id_remap: dict[str, str] = {}
    deduplicated_nodes: list[GraphNode] = []
    merged_count = 0

    summary_requests: list[tuple[int, str, list[str]]] = []
    group_data: list[tuple[GraphNode, list[GraphNode], list[str], list[str]]] = []
    # Track which group_data indices are mixed-label (need type selection)
    mixed_label_groups: dict[int, list[GraphNode]] = {}

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
        gi = len(group_data) - 1

        # Check if this group has mixed labels
        is_mixed = len({n.label for n in group_nodes}) >= 2

        if is_mixed and not descriptions:
            # Mixed labels but no descriptions — use heuristic for label
            survivor.label = _pick_canonical_label(group_nodes)
        elif is_mixed and descriptions and len(descriptions) < force_summary_threshold:
            # Mixed labels, few descriptions — use heuristic for label, pipe-join descs
            survivor.label = _pick_canonical_label(group_nodes)
        elif is_mixed and descriptions and len(descriptions) >= force_summary_threshold and llm is not None:
            # Mixed labels, enough descriptions — enhanced summary prompt picks type
            mixed_label_groups[gi] = group_nodes
            entity_name = str(survivor.properties.get("name", survivor.id))
            summary_requests.append((gi, entity_name, descriptions))
        elif descriptions and len(descriptions) >= force_summary_threshold and llm is not None:
            # Same-label group, enough descriptions — standard summary
            entity_name = str(survivor.properties.get("name", survivor.id))
            summary_requests.append((gi, entity_name, descriptions))

    # ── Single LLM batch call ────────────────────────────────────────
    prompts: list[str] = []
    for gi_ref, en, descs in summary_requests:
        if gi_ref in mixed_label_groups:
            types = ", ".join(sorted({n.label for n in mixed_label_groups[gi_ref]}))
            prompts.append(_SUMMARY_WITH_TYPE_PROMPT.format(
                entity_name=en,
                max_tokens=max_summary_tokens,
                types=types,
                descriptions="\n".join(f"- {d}" for d in descs),
            ))
        else:
            prompts.append(_SUMMARY_PROMPT.format(
                entity_name=en,
                max_tokens=max_summary_tokens,
                descriptions="\n".join(f"- {d}" for d in descs),
            ))

    batch_results: list = []
    summary_results: dict[int, str] = {}
    type_results: dict[int, str] = {}
    if prompts and llm is not None:
        batch_results = await llm.abatch_invoke(prompts)

    for item in batch_results:
        group_idx, _, descs = summary_requests[item.index]
        if not item.ok:
            summary_results[group_idx] = " | ".join(descs)
            continue
        content = item.response.content.strip()
        if group_idx in mixed_label_groups:
            # First line = chosen type, rest = summary
            lines = content.split("\n", 1)
            type_results[group_idx] = lines[0].strip()
            summary_results[group_idx] = lines[1].strip() if len(lines) > 1 else " | ".join(descs)
        else:
            summary_results[group_idx] = content

    # ── Apply merges ─────────────────────────────────────────────────
    for gi, (survivor, group_nodes, descriptions, all_source_ids) in enumerate(group_data):
        if len(group_nodes) == 1:
            continue
        if descriptions:
            survivor.properties["description"] = (
                summary_results[gi] if gi in summary_results else " | ".join(descriptions)
            )
        if all_source_ids:
            survivor.properties["source_chunk_ids"] = all_source_ids
        # Apply LLM-chosen canonical label for mixed-label groups
        if gi in type_results:
            chosen = type_results[gi]
            valid_labels = {n.label for n in group_nodes}
            # Validate LLM returned a label that exists in the group
            matched = next((l for l in valid_labels if l.lower() == chosen.lower()), None)
            survivor.label = matched if matched else _pick_canonical_label(group_nodes)
        for duplicate in group_nodes[1:]:
            for key, value in duplicate.properties.items():
                if key not in survivor.properties:
                    survivor.properties[key] = value
            id_remap[duplicate.id] = survivor.id
            merged_count += 1

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
