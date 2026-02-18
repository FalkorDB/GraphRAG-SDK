# GraphRAG SDK 2.0 — Utils: Graph Visualization
# Debugging and visualization utilities for the knowledge graph.

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GraphVisualizer:
    """Graph visualization and debugging utility.

    Generates human-readable representations of the knowledge graph
    for debugging and development purposes.

    Args:
        connection: FalkorDB connection instance.
    """

    def __init__(self, connection: Any) -> None:
        self._conn = connection

    async def get_stats(self) -> dict[str, Any]:
        """Get basic graph statistics.

        Returns:
            Dict with node_count, relationship_count, labels, and relationship_types.
        """
        stats: dict[str, Any] = {}

        try:
            # Node count
            result = await self._conn.query("MATCH (n) RETURN count(n)")
            stats["node_count"] = result.result_set[0][0] if result.result_set else 0

            # Relationship count
            result = await self._conn.query("MATCH ()-[r]->() RETURN count(r)")
            stats["relationship_count"] = result.result_set[0][0] if result.result_set else 0

            # Labels
            result = await self._conn.query(
                "MATCH (n) RETURN DISTINCT labels(n) AS lbl, count(n) AS cnt"
            )
            stats["labels"] = {
                str(row[0]): row[1] for row in (result.result_set or [])
            }

            # Relationship types
            result = await self._conn.query(
                "MATCH ()-[r]->() RETURN DISTINCT type(r) AS t, count(r) AS cnt"
            )
            stats["relationship_types"] = {
                row[0]: row[1] for row in (result.result_set or [])
            }

        except Exception as exc:
            logger.warning(f"Failed to get graph stats: {exc}")

        return stats

    async def describe(self) -> str:
        """Generate a human-readable description of the graph.

        Returns:
            Multi-line string describing the graph contents.
        """
        stats = await self.get_stats()

        lines = [
            "=== Knowledge Graph Summary ===",
            f"Nodes:         {stats.get('node_count', '?')}",
            f"Relationships: {stats.get('relationship_count', '?')}",
            "",
            "Labels:",
        ]
        for label, count in stats.get("labels", {}).items():
            lines.append(f"  {label}: {count}")

        lines.append("")
        lines.append("Relationship Types:")
        for rel_type, count in stats.get("relationship_types", {}).items():
            lines.append(f"  {rel_type}: {count}")

        return "\n".join(lines)

    async def sample_nodes(self, label: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        """Sample nodes from the graph for inspection.

        Args:
            label: Optional label filter.
            limit: Maximum number of nodes to return.

        Returns:
            List of node dicts.
        """
        if label:
            query = f"MATCH (n:`{label}`) RETURN n LIMIT $limit"
        else:
            query = "MATCH (n) RETURN n LIMIT $limit"

        try:
            result = await self._conn.query(query, {"limit": limit})
            return [{"node": row[0]} for row in (result.result_set or [])]
        except Exception as exc:
            logger.warning(f"Failed to sample nodes: {exc}")
            return []
