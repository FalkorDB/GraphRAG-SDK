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
            # Sanitize IDs once and filter out None / empty (bad LLM extraction)
            pre_filter = len(group)
            cleaned_group: list[tuple[GraphNode, str]] = [
                (n, self._clean_identifier(n.id)) for n in group if n.id is not None
            ]
            cleaned_group = [(n, cid) for n, cid in cleaned_group if cid.strip()]
            dropped = pre_filter - len(cleaned_group)
            if dropped:
                logger.warning(
                    "Dropped %d %s node(s) with empty id after sanitization",
                    dropped,
                    safe_label,
                )
            if not cleaned_group:
                continue
            # Process in batches
            for start in range(0, len(cleaned_group), self._BATCH_SIZE):
                batch = cleaned_group[start : start + self._BATCH_SIZE]
                batch_data = [
                    {
                        "id": cid,
                        "properties": self._clean_properties(n.properties),
                    }
                    for n, cid in batch
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
                    for node, cid in batch:
                        safe_node_label = sanitize_cypher_label(node.label)
                        q = f"MERGE (n:`{safe_node_label}` {{id: $id}}) SET n += $properties"
                        if is_entity:
                            q += " SET n:__Entity__"
                        params = {
                            "id": cid,
                            "properties": self._clean_properties(node.properties),
                        }
                        try:
                            await self._conn.query(q, params)
                            count += 1
                        except Exception as inner_exc:
                            logger.warning(
                                f"Failed to upsert node {cid} "
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
            # Sanitize IDs once and filter out empty endpoints
            pre_filter = len(group)
            cleaned_group: list[tuple[GraphRelationship, str, str]] = [
                (
                    rel,
                    self._clean_identifier(rel.start_node_id),
                    self._clean_identifier(rel.end_node_id),
                )
                for rel in group
            ]
            cleaned_group = [
                (rel, sid, eid) for rel, sid, eid in cleaned_group if sid.strip() and eid.strip()
            ]
            dropped = pre_filter - len(cleaned_group)
            if dropped:
                logger.warning(
                    "Dropped %d [%s] relationship(s) with empty endpoint id after sanitization",
                    dropped,
                    safe_rel_type,
                )
            if not cleaned_group:
                continue
            for start in range(0, len(cleaned_group), self._BATCH_SIZE):
                batch = cleaned_group[start : start + self._BATCH_SIZE]
                batch_data = [
                    {
                        "start_id": sid,
                        "end_id": eid,
                        "properties": self._clean_properties(r.properties),
                    }
                    for r, sid, eid in batch
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
                    for rel, sid, eid in batch:
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
                            "start_id": sid,
                            "end_id": eid,
                            "properties": self._clean_properties(rel.properties),
                        }
                        try:
                            await self._conn.query(q, params)
                            count += 1
                        except Exception as inner_exc:
                            logger.warning(
                                f"Failed to upsert relationship "
                                f"{sid}-[{safe_fb_rel}]->{eid} "
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

    # ── Document Lifecycle (incremental ingestion support) ──────
    # Used by GraphRAG.update() / delete_document() / apply_changes().
    # Cypher stays here per the repository pattern — facade calls these.

    async def get_document_record(self, document_id: str) -> dict[str, Any] | None:
        """Return ``{"path", "content_hash"}`` for a Document, or None.

        Pre-1.1.0 Document nodes lack ``content_hash`` — callers should
        treat a returned ``None`` hash as "always run the full update path".
        """
        result = await self._conn.query(
            "MATCH (d:Document {id: $id}) RETURN d.path AS path, "
            "d.content_hash AS content_hash LIMIT 1",
            {"id": document_id},
        )
        if not result.result_set:
            return None
        row = result.result_set[0]
        return {"path": row[0], "content_hash": row[1]}

    async def get_document_entity_candidates(self, document_id: str) -> list[str]:
        """Return ids of entities mentioned in this document's chunks.

        Captured before deletion so orphan cleanup can be scoped to just
        these entities — never global. Entities still mentioned by chunks
        of *other* documents will be filtered out by ``delete_orphan_entities``.
        """
        result = await self._conn.query(
            "MATCH (e:__Entity__)-[:MENTIONED_IN]->(:Chunk)<-[:PART_OF]-"
            "(:Document {id: $id}) RETURN DISTINCT e.id AS eid",
            {"id": document_id},
        )
        return [row[0] for row in result.result_set] if result.result_set else []

    async def find_pending(
        self, document_id: str
    ) -> tuple[str, str, str | None] | None:
        """Look up any ``__pending__`` Document for this id and report its
        commit state.

        Returns ``None`` if no pending exists for ``document_id``. Otherwise
        returns ``(state, pending_id, content_hash)`` where ``state`` is one
        of:

        - ``"COMMITTED"``  — pending has ``ready_to_commit=true``. The next
          step is rollforward; the pending must NOT be deleted.
        - ``"WRITTEN"``    — pending exists, no marker. The new content
          was written but commit was never reached. Safe to roll back
          (delete the pending and its chunks); the live document is
          untouched.

        State semantics are documented in detail in
        ``GraphRAG.update``'s docstring. The marker is the load-bearing
        commit point — anything that can read it is the source of truth
        for "did the prior attempt cross the commit boundary?".
        """
        prefix = f"{document_id}__pending__"
        result = await self._conn.query(
            "MATCH (p:Document) WHERE p.id STARTS WITH $prefix "
            "RETURN p.id AS pid, p.ready_to_commit AS rtc, "
            "p.content_hash AS hash ORDER BY p.id LIMIT 1",
            {"prefix": prefix},
        )
        if not result.result_set:
            return None
        row = result.result_set[0]
        pid, rtc, hash_ = row[0], row[1], row[2]
        state = "COMMITTED" if rtc else "WRITTEN"
        return state, pid, hash_

    async def mark_pending_committed(self, pending_id: str) -> int:
        """Set ``ready_to_commit=true`` on the pending Document — the
        commit point for the state-machine cutover.

        This single-property write is the load-bearing transition: it
        MUST commit before any destructive Cypher touches the live
        document. Once it commits, recovery on crash is rollforward
        (replay rollforward_cutover); before it commits, recovery is
        rollback (delete pending).

        Returns the number of Document nodes updated (0 if the pending
        id was not found, 1 in the normal path).
        """
        result = await self._conn.query(
            "MATCH (p:Document {id: $pid}) "
            "SET p.ready_to_commit = true RETURN count(p) AS n",
            {"pid": pending_id},
        )
        return result.result_set[0][0] if result.result_set else 0

    async def cleanup_pending_documents(self, document_id: str) -> int:
        """Delete UN-COMMITTED leftover pending Document(s) for this id.

        Pending nodes that have ``ready_to_commit=true`` are NEVER
        deleted by this method — those represent a crashed-mid-cutover
        state and the new content must be rolled forward, not discarded.

        Used at the start of ``update()`` whenever ``find_pending``
        reports state ``"WRITTEN"`` (rollback path).

        Returns the number of pending Document nodes removed.
        """
        prefix = f"{document_id}__pending__"
        result = await self._conn.query(
            "MATCH (p:Document) "
            "WHERE p.id STARTS WITH $prefix "
            "AND (p.ready_to_commit IS NULL OR p.ready_to_commit = false) "
            "OPTIONAL MATCH (p)-[:PART_OF]->(c:Chunk) "
            "DETACH DELETE p, c "
            "RETURN count(DISTINCT p) AS n",
            {"prefix": prefix},
        )
        return result.result_set[0][0] if result.result_set else 0

    async def rollforward_cutover(
        self,
        pending_id: str,
        real_id: str,
        path: str,
        content_hash: str,
    ) -> int:
        """Replay the cutover from a (possibly partial) COMMITTED state
        to FINAL. Idempotent — every operation is safe to re-run.

        Sequence:
          1. Delete live document's chunks (idempotent: already-gone is a no-op).
          2. Delete live document node (idempotent).
          3. Rename pending → canonical id, write fresh metadata, clear
             ``ready_to_commit``. Single Cypher SET — atomic at the
             per-statement level FalkorDB does guarantee.

        After step 3 the document is in FINAL state. If the process
        crashes between any two steps, the next call's ``find_pending``
        still reports COMMITTED (because the pending node and its
        marker survive until step 3 completes) and this method replays.

        Returns the number of chunks removed from the (possibly already
        empty) live document.
        """
        # 1. Delete live chunks (idempotent — empty result if already gone)
        r = await self._conn.query(
            "MATCH (:Document {id: $id})-[:PART_OF]->(c:Chunk) "
            "DETACH DELETE c RETURN count(c) AS n",
            {"id": real_id},
        )
        chunks_removed = r.result_set[0][0] if r.result_set else 0

        # 2. Delete live Document node (idempotent — no rows if already gone)
        await self._conn.query(
            "MATCH (d:Document {id: $id}) DETACH DELETE d",
            {"id": real_id},
        )

        # 3. Promote pending → canonical id and clear the commit marker.
        # Setting ready_to_commit = NULL effectively removes it; on a
        # replay where the rename already happened, this MATCH finds
        # nothing and the SET is a no-op.
        await self._conn.query(
            "MATCH (p:Document {id: $pending_id}) "
            "SET p.id = $real_id, p.path = $path, "
            "p.content_hash = $hash, p.ready_to_commit = NULL",
            {
                "pending_id": pending_id,
                "real_id": real_id,
                "path": path,
                "hash": content_hash,
            },
        )
        return chunks_removed

    async def delete_document_chunks_and_node(self, document_id: str) -> int:
        """Delete all chunks for a document and the Document node itself.

        Used by ``delete_document()``. Returns the number of chunks removed.
        Caller is responsible for orphan-entity cleanup (we don't snapshot
        candidates here so callers can do it independently).
        """
        r = await self._conn.query(
            "MATCH (:Document {id: $id})-[:PART_OF]->(c:Chunk) "
            "DETACH DELETE c RETURN count(c) AS n",
            {"id": document_id},
        )
        chunks_removed = r.result_set[0][0] if r.result_set else 0
        await self._conn.query(
            "MATCH (d:Document {id: $id}) DETACH DELETE d",
            {"id": document_id},
        )
        return chunks_removed

    async def delete_orphan_entities(self, candidate_ids: list[str]) -> int:
        """Delete entities from ``candidate_ids`` that no longer have any
        ``MENTIONED_IN`` edge to a chunk.

        Scoped to the candidate list — never touches entities outside it,
        so a document update can never orphan an entity belonging to
        another document. ``DETACH DELETE`` cascades incident ``RELATES``
        edges automatically; FalkorDB drops vector-index entries for
        deleted nodes/edges.
        """
        if not candidate_ids:
            return 0
        deleted = 0
        # Batch to keep param size bounded.
        for start in range(0, len(candidate_ids), self._BATCH_SIZE):
            batch = candidate_ids[start : start + self._BATCH_SIZE]
            r = await self._conn.query(
                "UNWIND $ids AS eid "
                "MATCH (e:__Entity__ {id: eid}) "
                "WHERE NOT (e)-[:MENTIONED_IN]->(:Chunk) "
                "DETACH DELETE e RETURN count(e) AS n",
                {"ids": batch},
            )
            deleted += r.result_set[0][0] if r.result_set else 0
        return deleted

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _sanitize_string(value: str) -> str:
        """Strip control chars that can break FalkorDB CYPHER param parsing."""
        return "".join(ch for ch in value if ch in "\t\n\r" or unicodedata.category(ch) != "Cc")

    @classmethod
    def _clean_identifier(cls, value: Any) -> str:
        """Normalise IDs used in Cypher params."""
        return cls._sanitize_string(str(value))

    @classmethod
    def _clean_properties(cls, props: dict[str, Any]) -> dict[str, Any]:
        """Remove None values and non-serialisable types from properties.

        - Strings: strip control characters.
        - Lists: filter items to primitives only, strip control chars from strings.
        - Dicts: serialise to JSON string.
        - Other: convert to string and strip control characters.
        """
        cleaned: dict[str, Any] = {}
        for key, value in props.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = cls._sanitize_string(value) if isinstance(value, str) else value
            elif isinstance(value, list):
                # FalkorDB supports lists of primitives — filter items
                filtered: list[str | int | float | bool] = []
                for item in value:
                    if not isinstance(item, (str, int, float, bool)):
                        continue
                    if isinstance(item, str):
                        filtered.append(cls._sanitize_string(item))
                    else:
                        filtered.append(item)
                if filtered:
                    cleaned[key] = filtered
            elif isinstance(value, dict):
                cleaned[key] = json.dumps(value)
            else:
                cleaned[key] = cls._sanitize_string(str(value))
        return cleaned
