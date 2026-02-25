# GraphRAG SDK 2.0 — Ingestion: Description Merge Resolution
# Deduplicates entities by normalized name and merges their descriptions.
# Uses LLM summarisation when the number of descriptions exceeds a threshold.

from __future__ import annotations

import logging
from collections import defaultdict

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    GraphData,
    GraphNode,
    GraphRelationship,
    ResolutionResult,
)
from graphrag_sdk.core.providers import LLMInterface
from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = (
    "Summarise the following descriptions of the entity '{entity_name}' "
    "into a single concise description (max {max_tokens} tokens).\n\n"
    "Descriptions:\n{descriptions}\n\n"
    "Summary:"
)


class DescriptionMergeResolution(ResolutionStrategy):
    """Deduplicate entities by normalized name with description merging.

    Groups nodes by normalised name (lowercase, stripped). For groups of
    multiple nodes:
    - Below ``force_summary_threshold``: concatenates descriptions with ``" | "``.
    - At or above threshold: uses LLM to summarise all descriptions.

    Follows the same ID-remap pattern as ``ExactMatchResolution``.

    Args:
        llm: LLM provider (used for summarisation above threshold).
        force_summary_threshold: Number of descriptions that triggers LLM summary.
        max_summary_tokens: Token budget hint passed to the LLM prompt.
    """

    def __init__(
        self,
        llm: LLMInterface | None = None,
        force_summary_threshold: int = 3,
        max_summary_tokens: int = 500,
    ) -> None:
        self.llm = llm
        self.force_summary_threshold = force_summary_threshold
        self.max_summary_tokens = max_summary_tokens

    async def resolve(
        self,
        graph_data: GraphData,
        ctx: Context,
    ) -> ResolutionResult:
        ctx.log(
            f"Resolving duplicates by description merge "
            f"({len(graph_data.nodes)} nodes, {len(graph_data.relationships)} rels)"
        )

        # Group nodes by normalised name
        groups: dict[str, list[GraphNode]] = defaultdict(list)
        for node in graph_data.nodes:
            name = node.properties.get("name", node.id)
            key = str(name).strip().lower()
            groups[key].append(node)

        # Build ID remap and deduplicated nodes
        id_remap: dict[str, str] = {}
        deduplicated_nodes: list[GraphNode] = []
        merged_count = 0

        # First pass: collect groups and identify those needing LLM summary
        # Each entry: (group_index, entity_name, descriptions)
        summary_requests: list[tuple[int, str, list[str]]] = []
        group_data: list[tuple[GraphNode, list[GraphNode], list[str], list[str]]] = []

        for _key, nodes in groups.items():
            survivor = nodes[0]
            deduplicated_nodes.append(survivor)

            if len(nodes) == 1:
                group_data.append((survivor, nodes, [], []))
                continue

            # Collect descriptions
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

            # Check if this group needs LLM summary
            if (
                descriptions
                and len(descriptions) >= self.force_summary_threshold
                and self.llm is not None
            ):
                entity_name = str(survivor.properties.get("name", survivor.id))
                summary_requests.append(
                    (len(group_data) - 1, entity_name, descriptions)
                )

        # Batch all LLM summary calls at once
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
                    ctx.log(
                        f"LLM summary failed for '{entity_name}': {item.error}",
                        logging.WARNING,
                    )
                    summary_results[group_idx] = " | ".join(descs)

        # Second pass: apply merged descriptions and build id_remap
        for gi, (survivor, nodes, descriptions, all_source_ids) in enumerate(group_data):
            if len(nodes) == 1:
                continue

            if descriptions:
                if gi in summary_results:
                    survivor.properties["description"] = summary_results[gi]
                else:
                    # Below threshold — simple concatenation
                    survivor.properties["description"] = " | ".join(descriptions)

            if all_source_ids:
                survivor.properties["source_chunk_ids"] = all_source_ids

            for duplicate in nodes[1:]:
                for key, value in duplicate.properties.items():
                    if key not in survivor.properties:
                        survivor.properties[key] = value
                id_remap[duplicate.id] = survivor.id
                merged_count += 1

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
            f"Resolution complete: {len(deduplicated_nodes)} nodes "
            f"({merged_count} merged), {len(deduplicated_rels)} rels"
        )
        return ResolutionResult(
            nodes=deduplicated_nodes,
            relationships=deduplicated_rels,
            merged_count=merged_count,
        )
