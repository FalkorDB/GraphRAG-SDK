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
from graphrag_sdk.core.models import DocumentRecord, GraphNode, GraphRelationship
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
                # RELATES edges carry ``source_chunk_ids`` provenance. A
                # plain ``SET r += item.properties`` on MERGE-found edges
                # overwrites the survivor's list with the incoming write's,
                # silently destroying earlier docs' contribution and
                # breaking ``delete_stale_relationships`` (a later delete
                # of the new doc would drop the edge even though earlier
                # docs still support it). Special-case RELATES to UNION
                # the lists — same idiom as the deduplicator remap path.
                if rel_type == "RELATES":
                    query = (
                        f"UNWIND $batch AS item "
                        f"MATCH (a:`{safe_src}` {{id: item.start_id}}), "
                        f"(b:`{safe_tgt}` {{id: item.end_id}}) "
                        f"MERGE (a)-[r:`{safe_rel_type}`]->(b) "
                        f"WITH r, item, "
                        f"     coalesce(r.source_chunk_ids, []) AS old, "
                        f"     coalesce(item.properties.source_chunk_ids, []) AS contrib "
                        f"SET r += item.properties "
                        f"SET r.source_chunk_ids = old + [c IN contrib WHERE NOT c IN old]"
                    )
                else:
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
                        # Mirror the batch-path RELATES special case so
                        # the per-item fallback doesn't silently regress
                        # to overwriting source_chunk_ids when the batch
                        # query fails (e.g. transient Cypher error).
                        if rel.type == "RELATES":
                            q = (
                                f"MATCH (a:`{safe_fb_src}` {{id: $start_id}}), "
                                f"(b:`{safe_fb_tgt}` {{id: $end_id}}) "
                                f"MERGE (a)-[r:`{safe_fb_rel}`]->(b) "
                                f"WITH r, $properties AS props, "
                                f"     coalesce(r.source_chunk_ids, []) AS old, "
                                f"     coalesce($properties.source_chunk_ids, []) AS contrib "
                                f"SET r += props "
                                f"SET r.source_chunk_ids = old + [c IN contrib WHERE NOT c IN old]"
                            )
                        else:
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

    async def get_document_record(self, document_id: str) -> DocumentRecord | None:
        """Return persisted Document state (``path``, ``content_hash``)
        as a typed ``DocumentRecord``, or ``None`` if no such Document.

        Pre-1.1.0 Document nodes lack ``content_hash`` — the field will
        be ``None`` on those records and callers should treat that as
        "always run the full update path" (no stored hash to compare).
        """
        result = await self._conn.query(
            "MATCH (d:Document {id: $id}) RETURN d.path AS path, "
            "d.content_hash AS content_hash LIMIT 1",
            {"id": document_id},
        )
        if not result.result_set:
            return None
        row = result.result_set[0]
        return DocumentRecord(path=row[0], content_hash=row[1])

    async def get_document_entity_candidates(self, document_id: str) -> list[str]:
        """Return ids of entities mentioned in this document's chunks.

        Captured before deletion so orphan cleanup can be scoped to just
        these entities — never global. Entities still mentioned by chunks
        of *other* documents will be filtered out by ``delete_orphan_entities``.

        Returns the full DISTINCT id set in one round-trip with no
        ``LIMIT`` — the assumption is that a single document's entity
        cardinality is small (typically <1000) relative to the global
        graph. Documents with millions of distinct entities are out of
        scope for this design and would need a streaming/batched
        variant.
        """
        result = await self._conn.query(
            "MATCH (e:__Entity__)-[:MENTIONED_IN]->(:Chunk)<-[:PART_OF]-"
            "(:Document {id: $id}) RETURN DISTINCT e.id AS eid",
            {"id": document_id},
        )
        return [row[0] for row in result.result_set] if result.result_set else []

    async def find_pending(self, document_id: str) -> tuple[str, str, str | None] | None:
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

        Implementation note — we issue TWO queries instead of one
        ``ORDER BY p.id LIMIT 1`` because lexicographic order on the
        pending suffix (``__pending__<8-hex>``) is essentially random.
        Under compounded crashes a graph can briefly hold both a
        WRITTEN and a COMMITTED pending for the same id; if the
        WRITTEN one sorted first, the caller would take the rollback
        branch and ``cleanup_pending_documents`` would skip the
        COMMITTED one (it refuses to touch ``ready_to_commit=true``)
        — but the freshly-started replacement update would also start
        a NEW pending, and on the *next* call the stale COMMITTED
        would replay over the just-written live data. Querying for
        COMMITTED explicitly first eliminates the ordering hazard.
        """
        prefix = f"{document_id}__pending__"
        # Phase 1: prefer COMMITTED pendings — they MUST be rolled
        # forward; never silently replaced by a WRITTEN sibling.
        committed = await self._conn.query(
            "MATCH (p:Document) WHERE p.id STARTS WITH $prefix "
            "AND p.ready_to_commit = true "
            "RETURN p.id AS pid, p.content_hash AS hash LIMIT 1",
            {"prefix": prefix},
        )
        if committed.result_set:
            row = committed.result_set[0]
            return ("COMMITTED", row[0], row[1])
        # Phase 2: fall back to any non-committed pending.
        written = await self._conn.query(
            "MATCH (p:Document) WHERE p.id STARTS WITH $prefix "
            "AND (p.ready_to_commit IS NULL OR p.ready_to_commit = false) "
            "RETURN p.id AS pid, p.content_hash AS hash LIMIT 1",
            {"prefix": prefix},
        )
        if not written.result_set:
            return None
        row = written.result_set[0]
        return ("WRITTEN", row[0], row[1])

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
            "MATCH (p:Document {id: $pid}) SET p.ready_to_commit = true RETURN count(p) AS n",
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

        **Limitation — entities introduced by the killed pending are not
        garbage-collected here.** If the pending's pipeline ran far
        enough to write entity nodes (step 7 of ``IngestionPipeline``)
        before the crash, those entities live on in the global
        ``__Entity__`` namespace. They are typically harmless: a
        subsequent successful ``update()`` for the same document will
        snapshot them as candidates and ``delete_orphan_entities`` will
        sweep any that no longer have ``MENTIONED_IN`` edges. We don't
        sweep here because doing so would require either tracking which
        entities the pending introduced (no snapshot mechanism today)
        or a global O(graph) orphan scan — neither acceptable on the
        hot path.
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
          0. Precondition: pending_id must still exist. On a successful
             replay (steps 1-2 ran, step 3 didn't) the pending is still
             present because step 3 is what renames it away. If the
             pending is missing here, something is wrong (concurrent
             cleanup, manual delete, or upstream bug) and we refuse to
             proceed rather than silently destroy the live document.
          1. Delete live document's chunks (idempotent: already-gone is a no-op).
          2. Delete live document node (idempotent).
          3. Rename pending → canonical id, write fresh metadata, clear
             ``ready_to_commit``. Single Cypher statement — atomic at
             the per-statement level FalkorDB does guarantee.

        After step 3 the document is in FINAL state. If the process
        crashes between any two steps, the next call's ``find_pending``
        still reports COMMITTED (because the pending node and its
        marker survive until step 3 completes) and this method replays.

        Returns the number of chunks removed from the (possibly already
        empty) live document.

        Raises:
            DatabaseError: if the precondition fails (pending missing).
        """
        # 0. Precondition check — abort before any destructive op if the
        # pending we'd promote isn't there. Without this guard a misuse
        # of the method (or a concurrent cleanup) would delete the live
        # document and then no-op the rename, leaving the graph empty.
        precheck = await self._conn.query(
            "MATCH (p:Document {id: $pid}) RETURN count(p) AS n",
            {"pid": pending_id},
        )
        if not precheck.result_set or precheck.result_set[0][0] != 1:
            from graphrag_sdk.core.exceptions import DatabaseError

            raise DatabaseError(
                f"rollforward_cutover: pending Document '{pending_id}' not found; "
                "refusing to delete live document. The pending may have been "
                "concurrently deleted, or this method was called with a stale id."
            )

        # 1-2. Delete live chunks + Document node. Same Cypher as
        # delete_document_chunks_and_node — share the helper so the
        # delete pattern lives in one place.
        chunks_removed = await self.delete_document_chunks_and_node(real_id)

        # 3. Promote pending → canonical id and remove the commit marker.
        # ``REMOVE p.ready_to_commit`` is the idiomatic way to drop a
        # property; on a replay where the rename already happened this
        # MATCH finds nothing and the whole statement is a no-op.
        await self._conn.query(
            "MATCH (p:Document {id: $pending_id}) "
            "SET p.id = $real_id, p.path = $path, p.content_hash = $hash "
            "REMOVE p.ready_to_commit",
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

        Used by ``rollforward_cutover`` — the live document is replaced
        wholesale, so removing both in one shot is correct.

        ``delete_document()`` uses ``delete_document_chunks`` and
        ``delete_document_node`` separately so the doc node can stay
        alive (carrying recovery state) during orphan cleanup, then be
        removed once cleanup completes.
        """
        chunks_removed = await self.delete_document_chunks(document_id)
        await self.delete_document_node(document_id)
        return chunks_removed

    async def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks linked to a Document via ``PART_OF``.

        Leaves the Document node intact. Idempotent — re-running on a
        document whose chunks are already gone is a no-op that returns 0.
        """
        r = await self._conn.query(
            "MATCH (:Document {id: $id})-[:PART_OF]->(c:Chunk) "
            "DETACH DELETE c RETURN count(c) AS n",
            {"id": document_id},
        )
        return r.result_set[0][0] if r.result_set else 0

    async def delete_document_node(self, document_id: str) -> None:
        """Delete the Document node itself. Idempotent."""
        await self._conn.query(
            "MATCH (d:Document {id: $id}) DETACH DELETE d",
            {"id": document_id},
        )

    async def get_document_chunk_ids(self, document_id: str) -> list[str]:
        """Snapshot the chunk ids belonging to a document.

        Captured alongside ``get_document_entity_candidates`` before a
        destructive operation; the pair feeds ``delete_stale_relationships``
        after cutover so RELATES facts whose only provenance was these
        chunks can be removed.

        Like the entity-candidate snapshot, this returns the full id set
        in one round trip with no LIMIT — documents with millions of
        chunks are out of scope for incremental update.
        """
        result = await self._conn.query(
            "MATCH (:Document {id: $id})-[:PART_OF]->(c:Chunk) RETURN c.id AS cid",
            {"id": document_id},
        )
        return [row[0] for row in result.result_set] if result.result_set else []

    async def set_pending_cleanup_state(
        self,
        document_id: str,
        candidate_ids: list[str],
        old_chunk_ids: list[str],
    ) -> None:
        """Persist post-cutover cleanup inputs onto a Document node so
        the cleanup is recoverable across a crash.

        Writes two properties:

        - ``cleanup_candidates`` — entity ids to orphan-check
        - ``cleanup_old_chunk_ids`` — chunks the cutover will delete,
          needed by ``delete_stale_relationships`` to identify RELATES
          edges whose only provenance was the doc's old chunks.

        MUST be called before ``mark_pending_committed`` in the update
        flow, or before ``mark_document_pending_delete`` in the delete
        flow. A crash before this write leaves the live doc intact
        (rollback is safe); a crash after this write but before the
        commit marker is also safe (the pending will be discarded along
        with its half-written cleanup state).
        """
        await self._conn.query(
            "MATCH (d:Document {id: $id}) "
            "SET d.cleanup_candidates = $candidates, "
            "    d.cleanup_old_chunk_ids = $chunks",
            {
                "id": document_id,
                "candidates": candidate_ids,
                "chunks": old_chunk_ids,
            },
        )

    async def get_cleanup_state(self, document_id: str) -> tuple[list[str], list[str]] | None:
        """Read persisted cleanup state off a Document node.

        Returns ``(candidate_ids, old_chunk_ids)`` if any cleanup state
        is set, or ``None`` if the document has neither property. Both
        lists are returned as ``[]`` when one property is set and the
        other isn't (a corruption edge that the cleanup still handles
        safely — empty lists are no-ops).
        """
        result = await self._conn.query(
            "MATCH (d:Document {id: $id}) "
            "RETURN d.cleanup_candidates AS cands, "
            "       d.cleanup_old_chunk_ids AS chunks LIMIT 1",
            {"id": document_id},
        )
        if not result.result_set:
            return None
        row = result.result_set[0]
        if row[0] is None and row[1] is None:
            return None
        return (row[0] or [], row[1] or [])

    async def clear_cleanup_state(self, document_id: str) -> None:
        """Remove cleanup-state properties — last step of the
        post-cutover cleanup. Idempotent."""
        await self._conn.query(
            "MATCH (d:Document {id: $id}) REMOVE d.cleanup_candidates, d.cleanup_old_chunk_ids",
            {"id": document_id},
        )

    async def mark_document_pending_delete(
        self,
        document_id: str,
        candidate_ids: list[str],
        chunk_ids: list[str],
    ) -> int:
        """Single atomic write that marks a live Document as undergoing
        deletion AND persists the cleanup inputs — the commit point for
        the delete-side state machine.

        Sets ``pending_delete=true`` plus ``cleanup_candidates`` and
        ``cleanup_old_chunk_ids`` in one Cypher statement. After this
        returns, recovery is rollforward (finish the delete sequence);
        before it returns, recovery is no-op (the live doc is intact).

        Returns the number of Document nodes updated. Callers should
        check for ``== 1`` and refuse to proceed otherwise — a 0
        indicates the doc vanished between the existence check and the
        commit, and pressing on would silently delete nothing.
        """
        result = await self._conn.query(
            "MATCH (d:Document {id: $id}) "
            "SET d.pending_delete = true, "
            "    d.cleanup_candidates = $candidates, "
            "    d.cleanup_old_chunk_ids = $chunks "
            "RETURN count(d) AS n",
            {
                "id": document_id,
                "candidates": candidate_ids,
                "chunks": chunk_ids,
            },
        )
        return result.result_set[0][0] if result.result_set else 0

    async def has_pending_delete(self, document_id: str) -> bool:
        """Return True iff this Document has ``pending_delete=true`` set
        — i.e. a prior ``delete_document()`` call crossed its commit
        point but never finished. Recovery should resume the delete
        before any other operation touches this id.
        """
        result = await self._conn.query(
            "MATCH (d:Document {id: $id}) WHERE d.pending_delete = true "
            "RETURN count(d) AS n LIMIT 1",
            {"id": document_id},
        )
        if not result.result_set:
            return False
        return result.result_set[0][0] > 0

    async def delete_stale_relationships(
        self,
        candidate_ids: list[str],
        old_chunk_ids: list[str],
    ) -> int:
        """Strip ``old_chunk_ids`` from RELATES edges' ``source_chunk_ids``
        provenance lists for edges incident to ``candidate_ids``; delete
        edges whose list becomes empty.

        Scoped to ``candidate_ids`` — never globally scans the graph,
        mirroring the discipline of ``delete_orphan_entities``. The
        ``source_chunk_ids`` property is written by the extractor (see
        ``ingestion/extraction_strategies/graph_extraction.py``); this
        method is its post-cutover counterpart, garbage-collecting facts
        no longer supported by any current document's chunks.

        Idempotent: chunk ids already stripped from a list don't appear
        a second time, so re-runs are no-ops.

        Why both inputs must be non-empty: with no candidates there's
        nothing to scope to (refuses global scan), and with no old
        chunks there's nothing to subtract (returns 0 trivially).
        """
        if not candidate_ids or not old_chunk_ids:
            return 0
        deleted = 0
        for start in range(0, len(candidate_ids), self._BATCH_SIZE):
            batch = candidate_ids[start : start + self._BATCH_SIZE]
            # Two-query implementation. FalkorDB's planner quirks bite
            # both ``WITH DISTINCT r, <expr> AS remaining ... SET r.X``
            # (SET silently no-ops) and ``... SET ... WITH DISTINCT ...
            # DELETE r`` (the SET is undone or skipped). Splitting into
            # two queries — strip-then-delete — avoids both issues and
            # is easier to reason about. Both are idempotent: re-running
            # is a no-op once the list shrinks to the stable set.
            await self._conn.query(
                "UNWIND $ids AS eid "
                "MATCH (e:__Entity__ {id: eid})-[r:RELATES]-(:__Entity__) "
                "WHERE r.source_chunk_ids IS NOT NULL "
                "AND any(c IN r.source_chunk_ids WHERE c IN $old_chunks) "
                "WITH r, "
                "  [c IN r.source_chunk_ids WHERE NOT c IN $old_chunks] AS remaining "
                "SET r.source_chunk_ids = remaining",
                {"ids": batch, "old_chunks": old_chunk_ids},
            )
            # Delete edges whose provenance went empty. Scoped to the
            # same candidate set so we still avoid a global scan, and
            # we use DISTINCT to handle the undirected-match fan-out
            # without errors from double-deleting the same row.
            r = await self._conn.query(
                "UNWIND $ids AS eid "
                "MATCH (e:__Entity__ {id: eid})-[r:RELATES]-(:__Entity__) "
                "WHERE r.source_chunk_ids IS NOT NULL "
                "AND size(r.source_chunk_ids) = 0 "
                "WITH DISTINCT r "
                "DELETE r "
                "RETURN count(r) AS n",
                {"ids": batch},
            )
            deleted += r.result_set[0][0] if r.result_set else 0
        return deleted

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

    # ── Ontology evolution (data-graph migration primitives) ────
    #
    # The ``GraphRAG`` evolution methods orchestrate ontology-graph writes
    # (via ``OntologyStore``) with data-graph writes (via the helpers
    # below). All are idempotent — re-running on already-converged state
    # is a no-op that returns count=0.

    _RETYPE_COERCERS: dict[str, str] = {
        "INTEGER": "toInteger",
        "FLOAT": "toFloat",
        "STRING": "toString",
        "BOOLEAN": "toBoolean",
    }

    async def rename_label(self, old: str, new: str) -> int:
        """Move every node from ``:old`` to ``:new``.

        Idempotent: nodes already moved no longer match the source label
        so the second call's MATCH yields zero. Preserves any other
        labels on the node (e.g. ``__Entity__``).

        Returns the number of nodes relabelled.
        """
        safe_old = sanitize_cypher_label(old)
        safe_new = sanitize_cypher_label(new)
        if safe_old == safe_new:
            return 0
        r = await self._conn.query(
            f"MATCH (n:`{safe_old}`) SET n:`{safe_new}` REMOVE n:`{safe_old}` RETURN count(n) AS n",
        )
        return r.result_set[0][0] if r.result_set else 0

    async def rename_node_property(self, label: str, old: str, new: str) -> int:
        """Rename property ``old`` to ``new`` on every node carrying ``label``.

        Property keys cannot be parameterised in Cypher, so the names are
        sanitised through the same routine used for labels (which allows
        only ``[A-Za-z_][A-Za-z0-9_]*``) before being interpolated.

        Returns the number of nodes whose property was renamed.
        """
        safe_label = sanitize_cypher_label(label)
        safe_old = sanitize_cypher_label(old)
        safe_new = sanitize_cypher_label(new)
        if safe_old == safe_new:
            return 0
        r = await self._conn.query(
            f"MATCH (n:`{safe_label}`) WHERE n.`{safe_old}` IS NOT NULL "
            f"SET n.`{safe_new}` = n.`{safe_old}` "
            f"REMOVE n.`{safe_old}` "
            f"RETURN count(n) AS n",
        )
        return r.result_set[0][0] if r.result_set else 0

    async def drop_node_property(self, label: str, prop: str) -> int:
        """Remove ``prop`` from every node carrying ``label``.

        Returns the number of nodes touched.
        """
        safe_label = sanitize_cypher_label(label)
        safe_prop = sanitize_cypher_label(prop)
        r = await self._conn.query(
            f"MATCH (n:`{safe_label}`) WHERE n.`{safe_prop}` IS NOT NULL "
            f"REMOVE n.`{safe_prop}` "
            f"RETURN count(n) AS n",
        )
        return r.result_set[0][0] if r.result_set else 0

    async def delete_nodes_by_label(self, label: str) -> int:
        """``DETACH DELETE`` every node with this label.

        Returns the count of nodes removed.
        """
        safe_label = sanitize_cypher_label(label)
        # Count first, then delete — FalkorDB's count-after-DELETE returns
        # post-delete cardinality (zero) for a non-aggregated path.
        count_r = await self._conn.query(
            f"MATCH (n:`{safe_label}`) RETURN count(n) AS n",
        )
        count = count_r.result_set[0][0] if count_r.result_set else 0
        if count == 0:
            return 0
        await self._conn.query(f"MATCH (n:`{safe_label}`) DETACH DELETE n")
        return count

    async def rename_relation_type(self, old: str, new: str) -> int:
        """Recreate every ``[:old]`` edge as ``[:new]``, preserving
        endpoints and properties.

        FalkorDB cannot mutate a relationship's type in place, so the
        operation is recreate-and-delete. Logs a warning when the edge
        count exceeds 10k because the rebuild is O(edges).

        Returns the number of edges relabelled.
        """
        safe_old = sanitize_cypher_label(old)
        safe_new = sanitize_cypher_label(new)
        if safe_old == safe_new:
            return 0
        count_r = await self._conn.query(f"MATCH ()-[r:`{safe_old}`]->() RETURN count(r) AS n")
        count = count_r.result_set[0][0] if count_r.result_set else 0
        if count > 10_000:
            logger.warning(
                "rename_relation_type: %d [%s] edges will be recreated as "
                "[%s] (expensive); FalkorDB lacks in-place type rewrite.",
                count,
                safe_old,
                safe_new,
            )
        if count == 0:
            return 0
        # Two-phase rebuild: collect via id pairs, then re-MERGE.
        await self._conn.query(
            f"MATCH (a)-[r:`{safe_old}`]->(b) "
            f"WITH a, b, r, properties(r) AS props "
            f"CREATE (a)-[r2:`{safe_new}`]->(b) "
            f"SET r2 = props "
            f"DELETE r"
        )
        return count

    async def delete_relations_by_type(self, rel_type: str) -> int:
        """``DELETE`` every edge of type ``rel_type``. Returns the count."""
        safe = sanitize_cypher_label(rel_type)
        count_r = await self._conn.query(f"MATCH ()-[r:`{safe}`]->() RETURN count(r) AS n")
        count = count_r.result_set[0][0] if count_r.result_set else 0
        if count == 0:
            return 0
        await self._conn.query(f"MATCH ()-[r:`{safe}`]->() DELETE r")
        return count

    async def delete_relations_by_pattern(
        self, rel_type: str, src_label: str, tgt_label: str
    ) -> int:
        """Delete edges of ``rel_type`` only when source/target labels match.

        Used for ``drop_relation_pattern``: other patterns of the same
        relation type (with different endpoint labels) are preserved.
        """
        safe_rel = sanitize_cypher_label(rel_type)
        safe_src = sanitize_cypher_label(src_label)
        safe_tgt = sanitize_cypher_label(tgt_label)
        count_r = await self._conn.query(
            f"MATCH (s:`{safe_src}`)-[r:`{safe_rel}`]->(t:`{safe_tgt}`) RETURN count(r) AS n"
        )
        count = count_r.result_set[0][0] if count_r.result_set else 0
        if count == 0:
            return 0
        await self._conn.query(
            f"MATCH (s:`{safe_src}`)-[r:`{safe_rel}`]->(t:`{safe_tgt}`) DELETE r"
        )
        return count

    async def coerce_node_property(
        self, label: str, prop: str, target_type: str
    ) -> tuple[int, int]:
        """Coerce values of ``label.prop`` to ``target_type`` via Cypher.

        Cypher's ``toInteger`` / ``toFloat`` / ``toString`` / ``toBoolean``
        return ``null`` for unconvertible values. We split the work into
        two passes:
        1. Convert and write back where the coerced value is non-null.
        2. ``REMOVE`` the property on rows that produced null (the value
           was unconvertible — better dropped than left mistyped).

        Returns ``(coerced, dropped)`` counts. ``LIST`` and ``DATE``
        target types are out of scope here (LIST is structural; DATE
        needs a parser per format) — use ``backfill_attribute_semantic``
        for LLM-assisted coercion of those.
        """
        normalized = (target_type or "STRING").strip().upper()
        if normalized not in self._RETYPE_COERCERS:
            raise ValueError(
                f"coerce_node_property: cannot mechanically coerce to "
                f"{normalized!r}. Mechanical coercion supports "
                f"{sorted(self._RETYPE_COERCERS)}. Use "
                f"backfill_attribute_semantic for LLM-assisted coercion."
            )
        coercer = self._RETYPE_COERCERS[normalized]
        safe_label = sanitize_cypher_label(label)
        safe_prop = sanitize_cypher_label(prop)
        # Pass 1 — coerce non-null convertible values.
        r1 = await self._conn.query(
            f"MATCH (n:`{safe_label}`) "
            f"WHERE n.`{safe_prop}` IS NOT NULL "
            f"AND {coercer}(n.`{safe_prop}`) IS NOT NULL "
            f"SET n.`{safe_prop}` = {coercer}(n.`{safe_prop}`) "
            f"RETURN count(n) AS n"
        )
        coerced = r1.result_set[0][0] if r1.result_set else 0
        # Pass 2 — drop unconvertible. We have to compare via the coercer
        # again rather than caching from pass 1 because FalkorDB doesn't
        # carry "did pass 1 touch this row" state across statements.
        r2 = await self._conn.query(
            f"MATCH (n:`{safe_label}`) "
            f"WHERE n.`{safe_prop}` IS NOT NULL "
            f"AND {coercer}(n.`{safe_prop}`) IS NULL "
            f"REMOVE n.`{safe_prop}` "
            f"RETURN count(n) AS n"
        )
        dropped = r2.result_set[0][0] if r2.result_set else 0
        return (coerced, dropped)

    # ── Chunk-scoped backfill helpers ───────────────────────────

    async def count_chunks_marked_with_op(self, op_id: str) -> int:
        """Count chunks that already carry the given backfill ``op_id``.

        Used to populate ``BackfillResult.chunks_skipped`` — on a fresh
        run this is 0, on an idempotent rerun it equals the work the
        previous run completed.
        """
        r = await self._conn.query(
            "MATCH (c:Chunk) WHERE $op IN coalesce(c.extracted_ops, []) RETURN count(c) AS n",
            {"op": op_id},
        )
        return r.result_set[0][0] if r.result_set else 0

    async def mark_chunk_extracted(self, chunk_id: str, op_id: str) -> None:
        """Append ``op_id`` to a chunk's ``extracted_ops`` list (idempotent).

        Used by ``BackfillExecutor`` to record that a chunk has been
        successfully processed for a given backfill operation, so a
        resumed or repeated run skips it.
        """
        await self._conn.query(
            "MATCH (c:Chunk {id: $id}) "
            "WITH c, coalesce(c.extracted_ops, []) AS ops "
            "WHERE NOT $op IN ops "
            "SET c.extracted_ops = ops + [$op]",
            {"id": chunk_id, "op": op_id},
        )

    async def list_chunks_for_attribute_backfill(
        self,
        owner_label: str,
        prop_name: str,
        *,
        op_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return chunks mentioning entities of ``owner_label`` that
        are missing ``prop_name``, excluding already-processed chunks.

        Each row:
            {"chunk_id": str, "chunk_text": str,
             "entities": [{"id": str, "name": str | None}, ...]}
        """
        safe_label = sanitize_cypher_label(owner_label)
        safe_prop = sanitize_cypher_label(prop_name)
        limit_clause = f" LIMIT {int(limit)}" if limit else ""
        result = await self._conn.query(
            f"MATCH (e:`{safe_label}`)-[:MENTIONED_IN]->(c:Chunk) "
            f"WHERE e.`{safe_prop}` IS NULL "
            "AND NOT $op IN coalesce(c.extracted_ops, []) "
            "WITH c, collect(DISTINCT {id: e.id, name: e.name}) AS ents "
            f"RETURN c.id AS chunk_id, c.text AS chunk_text, ents{limit_clause}",
            {"op": op_id},
        )
        rows: list[dict[str, Any]] = []
        for row in result.result_set or []:
            rows.append(
                {
                    "chunk_id": row[0],
                    "chunk_text": row[1],
                    "entities": row[2] or [],
                }
            )
        return rows

    async def list_chunks_for_entity_backfill(
        self,
        *,
        op_id: str,
        chunk_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return chunks (entire corpus, or a constrained ``chunk_ids``
        scope) that have not been marked with ``op_id`` yet.

        Each row: ``{"chunk_id": str, "chunk_text": str}``.
        """
        limit_clause = f" LIMIT {int(limit)}" if limit else ""
        if chunk_ids is None:
            result = await self._conn.query(
                "MATCH (c:Chunk) "
                "WHERE NOT $op IN coalesce(c.extracted_ops, []) "
                f"RETURN c.id AS chunk_id, c.text AS chunk_text{limit_clause}",
                {"op": op_id},
            )
        else:
            result = await self._conn.query(
                "UNWIND $ids AS cid "
                "MATCH (c:Chunk {id: cid}) "
                "WHERE NOT $op IN coalesce(c.extracted_ops, []) "
                f"RETURN c.id AS chunk_id, c.text AS chunk_text{limit_clause}",
                {"op": op_id, "ids": chunk_ids},
            )
        return [{"chunk_id": row[0], "chunk_text": row[1]} for row in (result.result_set or [])]

    async def list_chunks_for_relation_pattern_backfill(
        self,
        src_label: str,
        tgt_label: str,
        *,
        op_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return chunks where at least one ``src_label`` entity and one
        ``tgt_label`` entity co-occur (both have ``MENTIONED_IN`` to the
        same chunk), excluding already-processed chunks.

        Each row carries the candidate pairs so the LLM can answer
        per-pair without re-scanning.
        """
        safe_src = sanitize_cypher_label(src_label)
        safe_tgt = sanitize_cypher_label(tgt_label)
        limit_clause = f" LIMIT {int(limit)}" if limit else ""
        result = await self._conn.query(
            f"MATCH (s:`{safe_src}`)-[:MENTIONED_IN]->(c:Chunk) "
            f"MATCH (t:`{safe_tgt}`)-[:MENTIONED_IN]->(c) "
            "WHERE s <> t "
            "AND NOT $op IN coalesce(c.extracted_ops, []) "
            "WITH c, collect(DISTINCT {src_id: s.id, src_name: s.name, "
            "tgt_id: t.id, tgt_name: t.name}) AS pairs "
            f"RETURN c.id AS chunk_id, c.text AS chunk_text, pairs{limit_clause}",
            {"op": op_id},
        )
        return [
            {"chunk_id": row[0], "chunk_text": row[1], "pairs": row[2] or []}
            for row in (result.result_set or [])
        ]

    async def list_node_values_for_semantic_coerce(
        self,
        label: str,
        prop: str,
    ) -> list[tuple[str, Any]]:
        """Return ``(node_id, current_value)`` for every node carrying a
        non-null value for ``label.prop`` — input to LLM-assisted
        coercion (``backfill_attribute_semantic``).
        """
        safe_label = sanitize_cypher_label(label)
        safe_prop = sanitize_cypher_label(prop)
        result = await self._conn.query(
            f"MATCH (n:`{safe_label}`) WHERE n.`{safe_prop}` IS NOT NULL "
            f"RETURN n.id AS id, n.`{safe_prop}` AS value"
        )
        return [(row[0], row[1]) for row in (result.result_set or [])]

    async def set_node_property_by_id(
        self,
        label: str,
        node_id: str,
        prop: str,
        value: Any,
    ) -> None:
        """SET a single property on a node identified by ``label`` + ``id``.

        Used by backfill merge fns. ``value=None`` removes the property.
        """
        safe_label = sanitize_cypher_label(label)
        safe_prop = sanitize_cypher_label(prop)
        if value is None:
            await self._conn.query(
                f"MATCH (n:`{safe_label}` {{id: $id}}) REMOVE n.`{safe_prop}`",
                {"id": node_id},
            )
        else:
            await self._conn.query(
                f"MATCH (n:`{safe_label}` {{id: $id}}) SET n.`{safe_prop}` = $value",
                {"id": node_id, "value": value},
            )

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
            if isinstance(value, str | int | float | bool):
                cleaned[key] = cls._sanitize_string(value) if isinstance(value, str) else value
            elif isinstance(value, list):
                # FalkorDB supports lists of primitives — filter items
                filtered: list[str | int | float | bool] = []
                for item in value:
                    if not isinstance(item, str | int | float | bool):
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
