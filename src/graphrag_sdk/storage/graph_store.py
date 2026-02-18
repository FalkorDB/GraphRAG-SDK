# GraphRAG SDK 2.0 — Storage: Graph Store
# Pattern: Repository — all Cypher operations behind clean methods.
# Origin: Neo4j batched upserts + User design for storage separation.
#
# Strategies never write raw Cypher — they call graph_store methods.

from __future__ import annotations

import logging
from typing import Any

from graphrag_sdk.core.connection import FalkorDBConnection
from graphrag_sdk.core.exceptions import DatabaseError
from graphrag_sdk.core.models import GraphNode, GraphRelationship

logger = logging.getLogger(__name__)


class GraphStore:
    """Unified graph data access layer for FalkorDB.

    Provides:
    - Batched node upserts (MERGE)
    - Batched relationship upserts (MERGE)
    - Connected entity queries (for retrieval)
    - Schema introspection

    All operations use parameterised queries to prevent injection.

    Args:
        connection: FalkorDB connection instance.

    Example::

        store = GraphStore(connection)
        await store.upsert_nodes([node1, node2])
        await store.upsert_relationships([rel1])
    """

    def __init__(self, connection: FalkorDBConnection) -> None:
        self._conn = connection

    # ── Write Operations ─────────────────────────────────────────

    _BATCH_SIZE = 500
    _STRUCTURAL_LABELS = frozenset({"Chunk", "Document", "Fact"})
    _REL_LABEL_HINTS: dict[str, tuple[str, str]] = {
        "PART_OF": ("Document", "Chunk"),
        "NEXT_CHUNK": ("Chunk", "Chunk"),
        "MENTIONED_IN": ("__Entity__", "Chunk"),
        "SYNONYM": ("__Entity__", "__Entity__"),
    }

    async def upsert_nodes(self, nodes: list[GraphNode]) -> int:
        """Batch upsert nodes using UNWIND, grouped by label.

        Extracted entity nodes get ``__Entity__`` as an additional label.
        Structural nodes (Chunk, Document, Fact) do NOT get ``__Entity__``.

        Args:
            nodes: List of nodes to upsert.

        Returns:
            Number of nodes upserted.
        """
        if not nodes:
            return 0

        # Group nodes by label
        by_label: dict[str, list[GraphNode]] = {}
        for node in nodes:
            by_label.setdefault(node.label, []).append(node)

        count = 0
        for label, group in by_label.items():
            is_entity = label not in self._STRUCTURAL_LABELS
            # Filter out nodes with None or empty id (bad LLM extraction)
            group = [n for n in group if n.id is not None and str(n.id).strip()]
            if not group:
                continue
            # Process in batches
            for start in range(0, len(group), self._BATCH_SIZE):
                batch = group[start : start + self._BATCH_SIZE]
                batch_data = [
                    {"id": n.id, "properties": self._clean_properties(n.properties)}
                    for n in batch
                ]
                query = (
                    f"UNWIND $batch AS item "
                    f"MERGE (n:`{label}` {{id: item.id}}) "
                    f"SET n += item.properties"
                )
                if is_entity:
                    query += " SET n:__Entity__"
                try:
                    await self._conn.query(query, {"batch": batch_data})
                    count += len(batch)
                except Exception as exc:
                    logger.warning(
                        f"Batch upsert failed for {label} ({len(batch)} nodes), "
                        f"falling back to individual: {exc}"
                    )
                    # Per-item fallback
                    for node in batch:
                        q = (
                            f"MERGE (n:`{node.label}` {{id: $id}}) "
                            f"SET n += $properties"
                        )
                        if is_entity:
                            q += " SET n:__Entity__"
                        params = {
                            "id": node.id,
                            "properties": self._clean_properties(node.properties),
                        }
                        try:
                            await self._conn.query(q, params)
                            count += 1
                        except Exception as inner_exc:
                            logger.warning(f"Failed to upsert node {node.id}: {inner_exc}")
                            raise DatabaseError(f"Node upsert failed: {inner_exc}") from inner_exc

        logger.debug(f"Upserted {count} nodes")
        return count

    async def upsert_relationships(self, relationships: list[GraphRelationship]) -> int:
        """Batch upsert relationships using UNWIND, grouped by type.

        Args:
            relationships: List of relationships to upsert.

        Returns:
            Number of relationships upserted.
        """
        if not relationships:
            return 0

        # Group relationships by type
        by_type: dict[str, list[GraphRelationship]] = {}
        for rel in relationships:
            by_type.setdefault(rel.type, []).append(rel)

        count = 0
        for rel_type, group in by_type.items():
            for start in range(0, len(group), self._BATCH_SIZE):
                batch = group[start : start + self._BATCH_SIZE]
                batch_data = [
                    {
                        "start_id": r.start_node_id,
                        "end_id": r.end_node_id,
                        "properties": self._clean_properties(r.properties),
                    }
                    for r in batch
                ]
                src_label, tgt_label = self._REL_LABEL_HINTS.get(
                    rel_type, ("__Entity__", "__Entity__")
                )
                query = (
                    f"UNWIND $batch AS item "
                    f"MATCH (a:`{src_label}` {{id: item.start_id}}), "
                    f"(b:`{tgt_label}` {{id: item.end_id}}) "
                    f"MERGE (a)-[r:`{rel_type}`]->(b) "
                    f"SET r += item.properties"
                )
                try:
                    await self._conn.query(query, {"batch": batch_data})
                    count += len(batch)
                except Exception as exc:
                    logger.warning(
                        f"Batch upsert failed for [{rel_type}] ({len(batch)} rels), "
                        f"falling back to individual: {exc}"
                    )
                    # Per-item fallback
                    for rel in batch:
                        fb_src, fb_tgt = self._REL_LABEL_HINTS.get(
                            rel.type, ("__Entity__", "__Entity__")
                        )
                        q = (
                            f"MATCH (a:`{fb_src}` {{id: $start_id}}), "
                            f"(b:`{fb_tgt}` {{id: $end_id}}) "
                            f"MERGE (a)-[r:`{rel.type}`]->(b) "
                            f"SET r += $properties"
                        )
                        params = {
                            "start_id": rel.start_node_id,
                            "end_id": rel.end_node_id,
                            "properties": self._clean_properties(rel.properties),
                        }
                        try:
                            await self._conn.query(q, params)
                            count += 1
                        except Exception as inner_exc:
                            logger.warning(
                                f"Failed to upsert relationship "
                                f"{rel.start_node_id}-[{rel.type}]->{rel.end_node_id}: {inner_exc}"
                            )

        logger.debug(f"Upserted {count} relationships")
        return count

    # ── Read Operations ──────────────────────────────────────────

    async def get_connected_entities(
        self,
        chunk_id: str,
        max_hops: int = 1,
    ) -> list[dict[str, Any]]:
        """Get entities connected to a chunk within N hops.

        Used by retrieval strategies for graph-augmented context.

        Args:
            chunk_id: ID of the chunk node.
            max_hops: Maximum traversal depth (default: 1).

        Returns:
            List of entity dicts with id, label, properties.
        """
        query = (
            f"MATCH (c {{id: $chunk_id}})-[*1..{max_hops}]-(e:__Entity__) "
            f"RETURN DISTINCT e.id AS id, labels(e) AS labels, properties(e) AS props "
            f"LIMIT 50"
        )
        try:
            result = await self._conn.query(query, {"chunk_id": chunk_id})
            entities: list[dict[str, Any]] = []
            for row in result.result_set:
                entities.append({
                    "id": row[0],
                    "labels": row[1],
                    "properties": row[2] if len(row) > 2 else {},
                })
            return entities
        except Exception as exc:
            logger.warning(f"Failed to get entities for chunk {chunk_id}: {exc}")
            return []

    async def query_raw(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a raw Cypher query.

        Escape hatch for advanced use cases. Prefer specific methods
        for standard operations.
        """
        return await self._conn.query(cypher, params)

    # ── Statistics ────────────────────────────────────────────────

    async def get_statistics(self) -> dict[str, Any]:
        """Query basic graph statistics.

        Returns:
            Dict with node_count, edge_count, entity_types,
            relationship_types, graph_density, fact_node_count,
            synonym_edge_count, mention_edge_count.
        """
        stats: dict[str, Any] = {}

        r = await self._conn.query("MATCH (n) RETURN count(n)")
        stats["node_count"] = r.result_set[0][0] if r.result_set else 0

        r = await self._conn.query("MATCH ()-[r]->() RETURN count(r)")
        stats["edge_count"] = r.result_set[0][0] if r.result_set else 0

        r = await self._conn.query("CALL db.labels()")
        stats["entity_types"] = [row[0] for row in r.result_set] if r.result_set else []

        r = await self._conn.query("CALL db.relationshipTypes()")
        stats["relationship_types"] = [row[0] for row in r.result_set] if r.result_set else []

        n = stats["node_count"]
        stats["graph_density"] = stats["edge_count"] / n if n > 0 else 0

        for label, key in [
            ("Fact", "fact_node_count"),
            ("SYNONYM", "synonym_edge_count"),
            ("MENTIONED_IN", "mention_edge_count"),
        ]:
            try:
                if key.endswith("node_count"):
                    r = await self._conn.query(f"MATCH (f:{label}) RETURN count(f)")
                else:
                    r = await self._conn.query(
                        f"MATCH ()-[r:{label}]->() RETURN count(r)"
                    )
                stats[key] = r.result_set[0][0] if r.result_set else 0
            except Exception:
                stats[key] = 0

        return stats

    # ── Cleanup ──────────────────────────────────────────────────

    async def delete_all(self) -> None:
        """Delete all nodes and relationships. Use with caution."""
        await self._conn.query("MATCH (n) DETACH DELETE n")
        logger.info("Deleted all graph data")

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _clean_properties(props: dict[str, Any]) -> dict[str, Any]:
        """Remove None values and non-serialisable types from properties."""
        cleaned: dict[str, Any] = {}
        for key, value in props.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, list):
                # FalkorDB supports lists of primitives
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned
