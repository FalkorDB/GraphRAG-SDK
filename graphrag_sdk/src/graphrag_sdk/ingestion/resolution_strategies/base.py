# GraphRAG SDK 2.0 — Ingestion: Resolution Strategy ABC
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


async def exact_match_merge(
    nodes: list[GraphNode],
    llm: LLMInterface | None,
    *,
    force_summary_threshold: int = 3,
    max_summary_tokens: int = 500,
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

    # Batch LLM description summaries
    summary_results: dict[int, str] = {}
    if summary_requests and llm is not None:
        prompts = [
            _SUMMARY_PROMPT.format(
                entity_name=en,
                max_tokens=max_summary_tokens,
                descriptions="\n".join(f"- {d}" for d in descs),
            )
            for _, en, descs in summary_requests
        ]
        batch_results = await llm.abatch_invoke(prompts)
        for item in batch_results:
            group_idx, _, descs = summary_requests[item.index]
            if item.ok:
                summary_results[group_idx] = item.response.content.strip()
            else:
                summary_results[group_idx] = " | ".join(descs)

    # Apply merges
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
