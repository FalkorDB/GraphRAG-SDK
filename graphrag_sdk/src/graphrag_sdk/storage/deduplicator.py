# GraphRAG SDK 2.0 — Storage: Entity Deduplicator
# Two-phase entity deduplication: exact name match + optional fuzzy embedding.
# Preserves label-aware grouping to prevent cross-type merging.

from __future__ import annotations

import logging
from typing import Any

from graphrag_sdk.core.providers import Embedder

logger = logging.getLogger(__name__)

# Cypher queries for remapping edges from a duplicate to a survivor entity.
_REMAP_QUERIES = [
    # Outgoing RELATES from duplicate
    "MATCH (dup:__Entity__ {id: $dup_id})-[r:RELATES]->(b:__Entity__) "
    "WHERE b.id <> $survivor_id "
    "MERGE (s:__Entity__ {id: $survivor_id})-[nr:RELATES]->(b) "
    "SET nr += properties(r) "
    "DELETE r",
    # Incoming RELATES to duplicate
    "MATCH (a:__Entity__)-[r:RELATES]->(dup:__Entity__ {id: $dup_id}) "
    "WHERE a.id <> $survivor_id "
    "MERGE (a)-[nr:RELATES]->(s:__Entity__ {id: $survivor_id}) "
    "SET nr += properties(r) "
    "DELETE r",
    # MENTIONED_IN edges
    "MATCH (dup:__Entity__ {id: $dup_id})-[r:MENTIONED_IN]->(c:Chunk) "
    "MERGE (s:__Entity__ {id: $survivor_id})-[:MENTIONED_IN]->(c) "
    "DELETE r",
]


class EntityDeduplicator:
    """Two-phase entity deduplication engine.

    Phase 1 (always): Exact name match — groups entities by
    ``(normalized_name, label)`` to prevent cross-type merging,
    keeps the one with the longest description, remaps all
    RELATES and MENTIONED_IN edges, deletes duplicates.

    Phase 2 (optional): Fuzzy embedding match — embeds entity
    names, finds near-duplicates by cosine similarity, merges
    those too.

    Args:
        graph_store: Graph data access object with ``query_raw()`` method.
        embedder: Embedding provider for fuzzy dedup.
    """

    def __init__(self, graph_store: Any, embedder: Embedder) -> None:
        self._graph = graph_store
        self._embedder = embedder

    async def deduplicate(
        self,
        *,
        fuzzy: bool = False,
        similarity_threshold: float = 0.9,
        batch_size: int = 500,
    ) -> int:
        """Run deduplication and return total number of duplicates merged."""
        total = await self._deduplicate_exact(batch_size)

        if fuzzy:
            total += await self._deduplicate_fuzzy(batch_size, similarity_threshold)

        logger.info(f"EntityDeduplicator total: {total} duplicates merged")
        return total

    # ── Phase 1: Exact name match ──

    async def _deduplicate_exact(self, batch_size: int) -> int:
        entities = await self._fetch_all_entities(batch_size)
        if len(entities) < 2:
            logger.info("EntityDeduplicator: fewer than 2 entities, nothing to dedup")
            return 0

        # Group by (normalized name, label) to prevent cross-type merging.
        groups: dict[tuple[str, str], list[dict]] = {}
        for ent in entities:
            norm = ent["name"].strip().lower()
            label = ent.get("label", "").strip().lower()
            groups.setdefault((norm, label), []).append(ent)

        merged = 0
        for (_norm_name, _label), group in groups.items():
            if len(group) < 2:
                continue

            # Survivor: longest description
            group.sort(key=lambda e: len(e["description"]), reverse=True)
            survivor = group[0]
            duplicates = group[1:]

            for dup in duplicates:
                await self._remap_entity_edges(dup["id"], survivor["id"])
                try:
                    await self._graph.query_raw(
                        "MATCH (e:__Entity__ {id: $dup_id}) DETACH DELETE e",
                        {"dup_id": dup["id"]},
                    )
                    merged += 1
                except Exception as exc:
                    logger.warning(f"Failed to delete duplicate entity {dup['id']}: {exc}")

        logger.info(f"EntityDeduplicator phase 1 (exact): merged {merged} duplicates")
        return merged

    # ── Phase 2: Fuzzy embedding match ──

    async def _deduplicate_fuzzy(
        self, batch_size: int, similarity_threshold: float
    ) -> int:
        import numpy as np

        # Re-fetch surviving entities
        offset = 0
        all_ids: list[str] = []
        all_names: list[str] = []
        while True:
            result = await self._graph.query_raw(
                "MATCH (e:__Entity__) "
                "RETURN e.id AS id, e.name AS name "
                "SKIP $offset LIMIT $limit",
                {"offset": offset, "limit": batch_size},
            )
            if not result.result_set:
                break
            for row in result.result_set:
                all_ids.append(row[0])
                all_names.append(row[1] if len(row) > 1 and row[1] else str(row[0]))
            offset += batch_size

        if len(all_ids) < 2:
            return 0

        raw_vectors = await self._embedder.aembed_documents(all_names)
        valid = [
            (eid, name, vec)
            for eid, name, vec in zip(all_ids, all_names, raw_vectors)
            if vec
        ]
        if len(valid) < 2:
            return 0

        v_ids, _v_names, vectors = zip(*valid)
        v_ids = list(v_ids)

        mat = np.array(vectors, dtype=np.float32)
        norms_arr = np.linalg.norm(mat, axis=1, keepdims=True)
        norms_arr[norms_arr == 0] = 1.0
        mat_normed = mat / norms_arr

        # Find pairs above threshold (block-wise to avoid OOM)
        BLOCK_SIZE = 1000
        n = len(v_ids)
        merged_set: set[str] = set()
        merged_count = 0

        for i_start in range(0, n, BLOCK_SIZE):
            block = mat_normed[i_start:min(i_start + BLOCK_SIZE, n)]
            remaining = mat_normed[i_start:]
            sim_block = block @ remaining.T
            local_rows, local_cols = np.where(sim_block >= similarity_threshold)
            for lr, lc in zip(local_rows.tolist(), local_cols.tolist()):
                gi = i_start + lr
                gj = i_start + lc
                if gj > gi and v_ids[gj] not in merged_set:
                    survivor_id = v_ids[gi]
                    dup_id = v_ids[gj]
                    merged_set.add(dup_id)

                    await self._remap_entity_edges(dup_id, survivor_id)
                    try:
                        await self._graph.query_raw(
                            "MATCH (e:__Entity__ {id: $dup_id}) DETACH DELETE e",
                            {"dup_id": dup_id},
                        )
                        merged_count += 1
                    except Exception:
                        pass

        logger.info(
            f"EntityDeduplicator phase 2 (fuzzy): merged {merged_count} additional duplicates"
        )
        return merged_count

    # ── Helpers ──

    async def _fetch_all_entities(self, batch_size: int) -> list[dict]:
        """Fetch all entities in batches, including their primary label."""
        offset = 0
        entities: list[dict] = []
        while True:
            result = await self._graph.query_raw(
                "MATCH (e:__Entity__) "
                "RETURN e.id AS id, e.name AS name, e.description AS desc, "
                "HEAD([l IN labels(e) WHERE l <> '__Entity__']) AS label "
                "SKIP $offset LIMIT $limit",
                {"offset": offset, "limit": batch_size},
            )
            if not result.result_set:
                break
            for row in result.result_set:
                entities.append({
                    "id": row[0],
                    "name": row[1] if len(row) > 1 and row[1] else str(row[0]),
                    "description": row[2] if len(row) > 2 and row[2] else "",
                    "label": row[3] if len(row) > 3 and row[3] else "",
                })
            offset += batch_size
        return entities

    async def _remap_entity_edges(self, dup_id: str, survivor_id: str) -> None:
        """Remap all RELATES and MENTIONED_IN edges from duplicate to survivor."""
        params = {"dup_id": dup_id, "survivor_id": survivor_id}
        for query in _REMAP_QUERIES:
            try:
                await self._graph.query_raw(query, params)
            except Exception as exc:
                logger.warning(f"Edge remap failed for {dup_id} -> {survivor_id}: {exc}")
