# GraphRAG SDK — Storage: Graph Store
# Pattern: Repository — all Cypher operations behind clean methods.
# Origin: Neo4j batched upserts + User design for storage separation.
#
# Strategies never write raw Cypher — they call graph_store methods.

from __future__ import annotations

import json
import logging
import unicodedata
from typing import Any

from graphrag_sdk.core.connection import FalkorDBConnection
from graphrag_sdk.core.exceptions import DatabaseError
from graphrag_sdk.core.models import GraphNode, GraphRelationship
from graphrag_sdk.utils.cypher import sanitize_cypher_label

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
    _STRUCTURAL_LABELS = frozenset({"Chunk", "Document"})
    _REL_LABEL_HINTS: dict[str, tuple[str, str]] = {
        "PART_OF": ("Document", "Chunk"),
        "NEXT_CHUNK": ("Chunk", "Chunk"),
        "MENTIONED_IN": ("__Entity__", "Chunk"),
        "RELATES": ("__Entity__", "__Entity__"),
    }

    async def upsert_nodes(self, nodes: list[GraphNode]) -> int:
        """Batch upsert nodes using UNWIND, grouped by label.

        Extracted entity nodes get ``__Entity__`` as an additional label.
        Structural nodes (Chunk, Document) do NOT get ``__Entity__``.

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
            safe_label = sanitize_cypher_label(label)
            is_entity = label not in self._STRUCTURAL_LABELS
            # Filter out nodes with None or empty id (bad LLM extraction)
            group = [
                n
                for n in group
                if n.id is not None and self._clean_identifier(n.id).strip()
            ]
            if not group:
                continue
            # Process in batches
            for start in range(0, len(group), self._BATCH_SIZE):
                batch = group[start : start + self._BATCH_SIZE]
                batch_data = [
                    {
                        "id": self._clean_identifier(n.id),
                        "properties": self._clean_properties(n.properties),
                    }
                    for n in batch
                ]
                query = (
                    f"UNWIND $batch AS item "
                    f"MERGE (n:`{safe_label}` {{id: item.id}}) "
                    f"SET n += item.properties"
                )
                if is_entity:
                    query += " SET n:__Entity__"
                try:
                    await self._conn.query(query, {"batch": batch_data})
                    count += len(batch)
                except Exception as exc:
                    logger.warning(
                        f"Batch upsert failed for {safe_label} ({len(batch)} nodes), "
                        f"falling back to individual: {exc}"
                    )
                    # Per-item fallback
                    for node in batch:
                        safe_node_label = sanitize_cypher_label(node.label)
                        q = f"MERGE (n:`{safe_node_label}` {{id: $id}}) SET n += $properties"
                        if is_entity:
                            q += " SET n:__Entity__"
                        params = {
                            "id": self._clean_identifier(node.id),
                            "properties": self._clean_properties(node.properties),
                        }
                        try:
                            await self._conn.query(q, params)
                            count += 1
                        except Exception as inner_exc:
                            logger.warning(
                                f"Failed to upsert node {node.id} "
                                f"(label={safe_node_label}): {inner_exc}"
                            )
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
            safe_rel_type = sanitize_cypher_label(rel_type)
            group = [
                rel
                for rel in group
                if self._clean_identifier(rel.start_node_id).strip()
                and self._clean_identifier(rel.end_node_id).strip()
            ]
            if not group:
                continue
            for start in range(0, len(group), self._BATCH_SIZE):
                batch = group[start : start + self._BATCH_SIZE]
                batch_data = [
                    {
                        "start_id": self._clean_identifier(r.start_node_id),
                        "end_id": self._clean_identifier(r.end_node_id),
                        "properties": self._clean_properties(r.properties),
                    }
                    for r in batch
                ]
                src_label, tgt_label = self._REL_LABEL_HINTS.get(
                    rel_type, ("__Entity__", "__Entity__")
                )
                safe_src = sanitize_cypher_label(src_label)
                safe_tgt = sanitize_cypher_label(tgt_label)
                query = (
                    f"UNWIND $batch AS item "
                    f"MATCH (a:`{safe_src}` {{id: item.start_id}}), "
                    f"(b:`{safe_tgt}` {{id: item.end_id}}) "
                    f"MERGE (a)-[r:`{safe_rel_type}`]->(b) "
                    f"SET r += item.properties"
                )
                try:
                    await self._conn.query(query, {"batch": batch_data})
                    count += len(batch)
                except Exception as exc:
                    logger.warning(
                        f"Batch upsert failed for [{safe_rel_type}] "
                        f"({safe_src}→{safe_tgt}, {len(batch)} rels), "
                        f"falling back to individual: {exc}"
                    )
                    # Per-item fallback
                    for rel in batch:
                        fb_src, fb_tgt = self._REL_LABEL_HINTS.get(
                            rel.type, ("__Entity__", "__Entity__")
                        )
                        safe_fb_src = sanitize_cypher_label(fb_src)
                        safe_fb_tgt = sanitize_cypher_label(fb_tgt)
                        safe_fb_rel = sanitize_cypher_label(rel.type)
                        q = (
                            f"MATCH (a:`{safe_fb_src}` {{id: $start_id}}), "
                            f"(b:`{safe_fb_tgt}` {{id: $end_id}}) "
                            f"MERGE (a)-[r:`{safe_fb_rel}`]->(b) "
                            f"SET r += $properties"
                        )
                        params = {
                            "start_id": self._clean_identifier(rel.start_node_id),
                            "end_id": self._clean_identifier(rel.end_node_id),
                            "properties": self._clean_properties(rel.properties),
                        }
                        try:
                            await self._conn.query(q, params)
                            count += 1
                        except Exception as inner_exc:
                            logger.warning(
                                f"Failed to upsert relationship "
                                f"{rel.start_node_id}-[{safe_fb_rel}]->{rel.end_node_id} "
                                f"({safe_fb_src}→{safe_fb_tgt}): {inner_exc}"
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
        max_hops = max(1, min(max_hops, 5))
        query = (
            f"MATCH (c {{id: $chunk_id}})-[*1..{max_hops}]-(e:__Entity__) "
            f"RETURN DISTINCT e.id AS id, labels(e) AS labels, properties(e) AS props "
            f"LIMIT 50"
        )
        try:
            result = await self._conn.query(query, {"chunk_id": chunk_id})
            entities: list[dict[str, Any]] = []
            for row in result.result_set:
                entities.append(
                    {
                        "id": row[0],
                        "labels": row[1],
                        "properties": row[2] if len(row) > 2 else {},
                    }
                )
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
            relationship_types, graph_density,
            embedded_relationship_count,
            mention_edge_count, relates_edge_count.
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

        # Count RELATES edges with embeddings
        try:
            r = await self._conn.query(
                "MATCH ()-[r:RELATES]->() WHERE r.embedding IS NOT NULL RETURN count(r)"
            )
            stats["embedded_relationship_count"] = r.result_set[0][0] if r.result_set else 0
        except Exception:
            logger.debug("Failed to count embedded relationships", exc_info=True)
            stats["embedded_relationship_count"] = 0

        for label, key in [
            ("MENTIONED_IN", "mention_edge_count"),
            ("RELATES", "relates_edge_count"),
        ]:
            try:
                r = await self._conn.query(f"MATCH ()-[r:{label}]->() RETURN count(r)")
                stats[key] = r.result_set[0][0] if r.result_set else 0
            except Exception:
                logger.debug("Failed to count %s edges", label, exc_info=True)
                stats[key] = 0

        return stats

    # ── Cleanup ──────────────────────────────────────────────────

    async def delete_all(self) -> None:
        """Delete all nodes and relationships. Use with caution.

        Uses ``GRAPH.DELETE`` via the connection for speed on large graphs.
        Falls back to ``MATCH (n) DETACH DELETE n`` if ``delete_graph`` is
        not available.
        """
        try:
            await self._conn.delete_graph()
        except Exception:
            logger.debug("GRAPH.DELETE failed, falling back to DETACH DELETE", exc_info=True)
            await self._conn.query("MATCH (n) DETACH DELETE n")
        logger.info("Deleted all graph data")

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _sanitize_string(value: str) -> str:
        """Strip control chars that can break FalkorDB CYPHER param parsing."""
        return "".join(
            ch
            for ch in value
            if ch in "\t\n\r" or unicodedata.category(ch) != "Cc"
        )

    @classmethod
    def _clean_identifier(cls, value: Any) -> str:
        """Normalise IDs used in Cypher params."""
        return cls._sanitize_string(str(value))

    @staticmethod
    def _clean_properties(props: dict[str, Any]) -> dict[str, Any]:
        """Remove None values and non-serialisable types from properties.

        - Lists: filter items to primitives only, drop None/empty.
        - Dicts: serialise to JSON string.
        """
        cleaned: dict[str, Any] = {}
        for key, value in props.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = (
                    GraphStore._sanitize_string(value) if isinstance(value, str) else value
                )
            elif isinstance(value, list):
                # FalkorDB supports lists of primitives — filter items
                filtered: list[str | int | float | bool] = []
                for item in value:
                    if not isinstance(item, (str, int, float, bool)):
                        continue
                    if isinstance(item, str):
                        filtered.append(GraphStore._sanitize_string(item))
                    else:
                        filtered.append(item)
                if filtered:
                    cleaned[key] = filtered
            elif isinstance(value, dict):
                cleaned[key] = json.dumps(value)
            else:
                cleaned[key] = str(value)
        return cleaned
