# GraphRAG SDK — Ingestion: Exact Match Resolution
# Origin: Neo4j SinglePropertyExactMatchResolver — simplified.

from __future__ import annotations

import logging
from collections import defaultdict

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    GraphData,
    GraphNode,
    ResolutionResult,
)
from graphrag_sdk.ingestion.resolution_strategies.base import (
    ResolutionStrategy,
    remap_relationships,
)

logger = logging.getLogger(__name__)


class ExactMatchResolution(ResolutionStrategy):
    """Deduplicate entities by exact property match.

    Groups nodes with the same (label, resolve_property) pair and
    keeps the first occurrence, remapping all relationships to point
    to the surviving node.

    Args:
        resolve_property: The property to match on (default: "id").
    """

    def __init__(self, resolve_property: str = "id") -> None:
        self.resolve_property = resolve_property

    async def resolve(
        self,
        graph_data: GraphData,
        ctx: Context,
    ) -> ResolutionResult:
        ctx.log(
            f"Resolving duplicates by exact match on '{self.resolve_property}' "
            f"({len(graph_data.nodes)} nodes, {len(graph_data.relationships)} rels)"
        )

        # Group nodes by (label, resolve_property_value)
        groups: dict[tuple[str, str], list[GraphNode]] = defaultdict(list)
        for node in graph_data.nodes:
            key_value = node.properties.get(self.resolve_property, node.id)
            groups[(node.label, str(key_value))].append(node)

        # Build ID remap: duplicate_id → surviving_id
        id_remap: dict[str, str] = {}
        deduplicated_nodes: list[GraphNode] = []
        merged_count = 0

        for (_label, _prop), nodes in groups.items():
            survivor = nodes[0]
            deduplicated_nodes.append(survivor)

            # Merge properties from duplicates into the survivor
            for duplicate in nodes[1:]:
                for key, value in duplicate.properties.items():
                    if key not in survivor.properties:
                        survivor.properties[key] = value
                id_remap[duplicate.id] = survivor.id
                merged_count += 1

        # Remap relationship endpoints. This was an inline copy of
        # ``remap_relationships``; the two drifted out of sync on the
        # dedup key, so both carried the same fact-collapsing bug and
        # each had to be found separately. Call the shared helper.
        deduplicated_rels = remap_relationships(graph_data.relationships, id_remap)

        ctx.log(
            f"Resolution complete: {len(deduplicated_nodes)} nodes "
            f"({merged_count} merged), {len(deduplicated_rels)} rels"
        )
        return ResolutionResult(
            nodes=deduplicated_nodes,
            relationships=deduplicated_rels,
            merged_count=merged_count,
            remap=id_remap,
        )
