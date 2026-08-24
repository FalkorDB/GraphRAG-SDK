# GraphRAG SDK — Storage: Entity Deduplicator
# Two-phase entity deduplication: exact name match + optional fuzzy embedding.
# Preserves label-aware grouping to prevent cross-type merging.

from __future__ import annotations

import logging
from typing import Any

from graphrag_sdk.core.providers import Embedder

logger = logging.getLogger(__name__)

# Pagination safety net — well above any realistic graph size at typical
# batch sizes (10_000 * 500 = 5M entities). Trips only on a pathological
# server bug that keeps returning the same page.
_MAX_PAGINATION_ITERATIONS = 10_000

# Cypher queries for remapping edges from a duplicate to a survivor entity.
#
# The RELATES variants union ``source_chunk_ids`` rather than letting
# ``SET nr += properties(r)`` overwrite it. Without that union, a dedup
# merge would silently strip the survivor edge's provenance, breaking
# ``GraphStore.delete_stale_relationships`` on the next update/delete:
# RELATES facts originally contributed by chunks now belonging to the
# survivor would look unrooted and could be wrongly deleted (or, more
# commonly, wrongly retained because their list shrank to the dup's
# contribution alone).
#
# Two properties of these queries are load-bearing and easy to undo by
# "simplifying" them:
#
# 1. The survivor is bound by its own ``MATCH`` before the ``MERGE``.
#    Writing the survivor inline — ``MERGE (s:__Entity__ {id: $survivor_id})
#    -[nr:RELATES]->(b)`` — leaves ``s`` unbound, and MERGE creates the
#    *entire* pattern when it fails to match. That silently forks the
#    survivor into a second, type-label-less ``__Entity__`` node carrying
#    the same id: the ghost takes the remapped edges while the real
#    survivor is left with none. There is no uniqueness constraint on
#    ``__Entity__.id`` to catch it.
#
# 2. The RELATES ``MERGE`` is keyed on ``rel_type``. Without that key it
#    matches *any* RELATES between the pair, so a survivor's ``WORKS_AT``
#    and a duplicate's ``FOUNDED`` collapse onto one edge — ``SET nr +=
#    properties(r)`` then overwrites ``rel_type`` and ``fact``, destroying
#    one fact and leaving its ``source_chunk_ids`` attached to the other,
#    so a chunk vouches for a fact it never asserted.
#
# ``coalesce(r.rel_type, '')`` keeps the key non-null: FalkorDB rejects a
# MERGE keyed on a null property outright ("Cannot merge node using null
# property value"), which would abort the whole remap for any RELATES edge
# written without a ``rel_type`` — leaving the duplicate un-merged.
_REMAP_QUERIES = [
    # Outgoing RELATES from duplicate.
    "MATCH (dup:__Entity__ {id: $dup_id})-[r:RELATES]->(b:__Entity__) "
    "WHERE b.id <> $survivor_id "
    "MATCH (s:__Entity__ {id: $survivor_id}) "
    "MERGE (s)-[nr:RELATES {rel_type: coalesce(r.rel_type, '')}]->(b) "
    "WITH r, nr, "
    "     coalesce(nr.source_chunk_ids, []) AS old, "
    "     coalesce(r.source_chunk_ids, []) AS contrib "
    "SET nr += properties(r) "
    "SET nr.source_chunk_ids = old + [c IN contrib WHERE NOT c IN old] "
    "DELETE r",
    # Incoming RELATES to duplicate.
    "MATCH (a:__Entity__)-[r:RELATES]->(dup:__Entity__ {id: $dup_id}) "
    "WHERE a.id <> $survivor_id "
    "MATCH (s:__Entity__ {id: $survivor_id}) "
    "MERGE (a)-[nr:RELATES {rel_type: coalesce(r.rel_type, '')}]->(s) "
    "WITH r, nr, "
    "     coalesce(nr.source_chunk_ids, []) AS old, "
    "     coalesce(r.source_chunk_ids, []) AS contrib "
    "SET nr += properties(r) "
    "SET nr.source_chunk_ids = old + [c IN contrib WHERE NOT c IN old] "
    "DELETE r",
    # MENTIONED_IN edges — no source_chunk_ids on these, plain remap.
    "MATCH (dup:__Entity__ {id: $dup_id})-[r:MENTIONED_IN]->(c:Chunk) "
    "MATCH (s:__Entity__ {id: $survivor_id}) "
    "MERGE (s)-[:MENTIONED_IN]->(c) "
    "DELETE r",
]


def _survivor_rank(entity: dict) -> tuple[int, int, int]:
    """Rank candidates so the most reproducible identity survives a merge.

    Ordered by:

    1. **Derived from a declared key.** A structured node's id comes from a key
       the mapping declared, so the next ingest of that source recomputes the
       same id. A node extracted from prose is keyed on a surface form the model
       happened to produce. If the prose node survives, the keyed id is gone, and
       re-ingesting the table recreates it as a *second* node: measured as two
       ``E-1`` people, one titled "Engineer" and one "engineer". Keeping the
       reproducible id is what makes re-ingest idempotent after resolution.
    2. **Real over placeholder.** A stub was created by a foreign key and knows
       only an id and a name.
    3. **Longest description**, the original rule, which still decides between
       two nodes of the same provenance.

    ``is_stub`` is the marker because only a mapped source writes it: ``None``
    means the node came from extraction.
    """
    is_stub = entity.get("is_stub")
    return (
        0 if is_stub is None else 1,
        0 if is_stub else 1,
        len(entity.get("description") or ""),
    )


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
        # Names found under more than one label on the last run: usually the
        # fingerprint of an ingest-order mistake. See _report_cross_label_names.
        self.cross_label_names: dict[str, list[str]] = {}
        self._declared_labels: set[str] = set()

    async def deduplicate(
        self,
        *,
        fuzzy: bool = False,
        similarity_threshold: float = 0.9,
        batch_size: int = 500,
        declared_labels: set[str] | None = None,
    ) -> int:
        """Run deduplication and return total number of duplicates merged.

        ``declared_labels`` are labels a structured mapping declared. They are
        treated as authoritative about type, which lets the one safe kind of
        cross-label merge happen: see :meth:`_adopt_into_declared_labels`.
        """
        self._declared_labels = {label.strip().lower() for label in (declared_labels or set())}
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

            group.sort(key=_survivor_rank, reverse=True)
            survivor = group[0]
            duplicates = group[1:]

            for dup in duplicates:
                if not await self._remap_entity_edges(dup["id"], survivor["id"]):
                    logger.warning(f"Skipping deletion of {dup['id']} — edge remap incomplete")
                    continue
                await self._carry_properties(dup["id"], survivor["id"])
                try:
                    await self._graph.query_raw(
                        "MATCH (e:__Entity__ {id: $dup_id}) DETACH DELETE e",
                        {"dup_id": dup["id"]},
                    )
                    merged += 1
                except Exception as exc:
                    logger.warning(f"Failed to delete duplicate entity {dup['id']}: {exc}")

        merged += await self._adopt_into_declared_labels(groups)
        logger.info(f"EntityDeduplicator phase 1 (exact): merged {merged} duplicates")
        self._report_cross_label_names(groups)
        return merged

    # ── Phase 2: Fuzzy embedding match ──

    async def _deduplicate_fuzzy(self, batch_size: int, similarity_threshold: float) -> int:
        import numpy as np

        # Re-fetch surviving entities (with labels for cross-type guard)
        offset = 0
        all_ids: list[str] = []
        all_names: list[str] = []
        all_labels: list[str] = []
        rank_by_id: dict[str, tuple[int, int, int]] = {}
        for _ in range(_MAX_PAGINATION_ITERATIONS):
            result = await self._graph.query_raw(
                "MATCH (e:__Entity__) "
                "RETURN e.id AS id, e.name AS name, "
                "HEAD([l IN labels(e) WHERE l <> '__Entity__']) AS label, "
                "e.is_stub AS is_stub, e.description AS desc "
                "SKIP $offset LIMIT $limit",
                {"offset": offset, "limit": batch_size},
            )
            if not result.result_set:
                break
            for row in result.result_set:
                all_ids.append(row[0])
                all_names.append(row[1] if len(row) > 1 and row[1] else str(row[0]))
                all_labels.append(row[2] if len(row) > 2 and row[2] else "")
                rank_by_id[row[0]] = _survivor_rank(
                    {
                        "is_stub": row[3] if len(row) > 3 else None,
                        "description": row[4] if len(row) > 4 else "",
                    }
                )
            offset += batch_size
        else:
            logger.error(
                "Pagination exceeded %d iterations in _deduplicate_fuzzy — aborting",
                _MAX_PAGINATION_ITERATIONS,
            )

        if len(all_ids) < 2:
            return 0

        raw_vectors = await self._embedder.aembed_documents(all_names)
        valid = [
            (eid, name, label, vec)
            for eid, name, label, vec in zip(all_ids, all_names, all_labels, raw_vectors)
            if vec
        ]
        if len(valid) < 2:
            return 0

        v_ids, _v_names, v_labels, vectors = zip(*valid)
        v_ids = list(v_ids)
        v_labels = list(v_labels)

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
            block = mat_normed[i_start : min(i_start + BLOCK_SIZE, n)]
            remaining = mat_normed[i_start:]
            sim_block = block @ remaining.T
            local_rows, local_cols = np.where(sim_block >= similarity_threshold)
            for lr, lc in zip(local_rows.tolist(), local_cols.tolist()):
                gi = i_start + lr
                gj = i_start + lc
                if (
                    gj > gi
                    and v_ids[gi] not in merged_set
                    and v_ids[gj] not in merged_set
                    and v_labels[gi] == v_labels[gj]  # prevent cross-type merging
                ):
                    survivor_id = v_ids[gi]
                    dup_id = v_ids[gj]
                    # Array order is arbitrary here, so apply the same rule as
                    # the exact phase rather than keeping whichever came first.
                    if rank_by_id.get(dup_id, (0, 0, 0)) > rank_by_id.get(survivor_id, (0, 0, 0)):
                        survivor_id, dup_id = dup_id, survivor_id
                    merged_set.add(dup_id)

                    if not await self._remap_entity_edges(dup_id, survivor_id):
                        continue
                    await self._carry_properties(dup_id, survivor_id)
                    try:
                        await self._graph.query_raw(
                            "MATCH (e:__Entity__ {id: $dup_id}) DETACH DELETE e",
                            {"dup_id": dup_id},
                        )
                        merged_count += 1
                    except Exception:
                        logger.debug("Failed to delete duplicate entity %s", dup_id, exc_info=True)

        logger.info(
            f"EntityDeduplicator phase 2 (fuzzy): merged {merged_count} additional duplicates"
        )
        return merged_count

    # ── Helpers ──

    async def _adopt_into_declared_labels(self, groups: dict[tuple[str, str], list[dict]]) -> int:
        """Merge extracted entities into the declared entity of the same name.

        Matching on name *and* label is what keeps "Apple" the company apart from
        "Apple" the fruit, and that guard stays. But it also blocked the case it
        was never meant to catch: a mapping *declares* that "Carbon Farming" is a
        ``MitigationPractice``, while an extractor reading prose only *guessed*
        ``Concept`` from a built-in list that did not yet contain the real label.
        Those are the same thing described by two sources, one of which knows.

        So exactly one cross-label merge is allowed: when a name exists under one
        label a mapping declared and one or more labels nothing declared, the
        declared one survives and absorbs the rest. A declared type beats an
        inferred type, the same rule that already governs declared *columns*.

        Left alone, and reported instead:

        - two declared labels sharing a name, which is a real modelling conflict
          rather than a guess to correct
        - names under only undeclared labels, which is the Apple case

        This is why the graph no longer depends on the order sources arrive in.
        Measured on the same files, prose first: 0 merged before, 5 after.
        """
        if not self._declared_labels:
            return 0

        by_name: dict[str, list[str]] = {}
        for norm_name, label in groups:
            by_name.setdefault(norm_name, []).append(label)

        merged = 0
        for norm_name, labels in by_name.items():
            if len(labels) < 2:
                continue
            declared = [label for label in labels if label in self._declared_labels]
            inferred = [label for label in labels if label not in self._declared_labels]
            if len(declared) != 1 or not inferred:
                continue

            survivors = groups[(norm_name, declared[0])]
            if not survivors:
                continue
            survivor = survivors[0]
            for label in inferred:
                for duplicate in groups[(norm_name, label)]:
                    if duplicate["id"] == survivor["id"]:
                        continue
                    if not await self._remap_entity_edges(duplicate["id"], survivor["id"]):
                        logger.warning(
                            "Skipping deletion of %s — edge remap incomplete", duplicate["id"]
                        )
                        continue
                    await self._carry_properties(duplicate["id"], survivor["id"])
                    try:
                        await self._graph.query_raw(
                            "MATCH (e:__Entity__ {id: $dup_id}) DETACH DELETE e",
                            {"dup_id": duplicate["id"]},
                        )
                        merged += 1
                        logger.info(
                            "Adopted %r from inferred label %r into declared label %r",
                            survivor["name"],
                            label,
                            declared[0],
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to delete %s during label adoption: %s",
                            duplicate["id"],
                            exc,
                        )
                # Consumed, so the leftover report does not name it.
                groups.pop((norm_name, label), None)
        return merged

    def _report_cross_label_names(self, groups: dict[tuple[str, str], list[dict]]) -> None:
        """Say when two entities share a name but not a label.

        Matching on name *and* label is what stops "Apple" the company merging
        with "Apple" the fruit, so this is never merged automatically. But it is
        also the exact fingerprint of an ordering mistake: a document read before
        a mapping was declared has its entities labelled with a built-in guess,
        and the same name arriving later from a table under a declared label can
        no longer join it. That case is silent otherwise — the caller sees
        ``entities_deduplicated=0`` and nothing else.

        Reporting it costs one pass over grouping that already happened.
        """
        labels_by_name: dict[str, set[str]] = {}
        for norm_name, label in groups:
            labels_by_name.setdefault(norm_name, set()).add(label)
        collisions = {
            name: sorted(labels) for name, labels in labels_by_name.items() if len(labels) > 1
        }
        self.cross_label_names = collisions
        if not collisions:
            return
        sample = "; ".join(
            f"{name!r} as {' and '.join(labels)}" for name, labels in sorted(collisions.items())[:3]
        )
        logger.warning(
            "%d name(s) exist under more than one label and were NOT merged, which "
            "is usually an ingest-order problem: a document read before a mapping "
            "was declared gets its entities labelled by guesswork, and a table "
            "declaring the same name under its own label can no longer join them. "
            "Declare mappings up front (GraphRAG(mappings=[...]) or "
            "declare_mapping()) and the extractor uses the declared label. "
            "Examples: %s",
            len(collisions),
            sample,
        )

    async def _fetch_all_entities(self, batch_size: int) -> list[dict]:
        """Fetch all entities in batches, including their primary label."""
        offset = 0
        entities: list[dict] = []
        for _ in range(_MAX_PAGINATION_ITERATIONS):
            result = await self._graph.query_raw(
                "MATCH (e:__Entity__) "
                "RETURN e.id AS id, e.name AS name, e.description AS desc, "
                "HEAD([l IN labels(e) WHERE l <> '__Entity__']) AS label, "
                "e.is_stub AS is_stub "
                "SKIP $offset LIMIT $limit",
                {"offset": offset, "limit": batch_size},
            )
            if not result.result_set:
                break
            for row in result.result_set:
                entities.append(
                    {
                        "id": row[0],
                        "name": row[1] if len(row) > 1 and row[1] else str(row[0]),
                        "description": row[2] if len(row) > 2 and row[2] else "",
                        "label": row[3] if len(row) > 3 and row[3] else "",
                        # Only a mapped source writes is_stub, so its presence
                        # marks an id derived from a declared key.
                        "is_stub": row[4] if len(row) > 4 else None,
                    }
                )
            offset += batch_size
        else:
            logger.error(
                "Pagination exceeded %d iterations in _fetch_all_entities — aborting",
                _MAX_PAGINATION_ITERATIONS,
            )
        return entities

    # Written by the system, never carried across from a duplicate.
    _NEVER_CARRY = frozenset({"id", "embedding"})

    async def _carry_properties(self, dup_id: str, survivor_id: str) -> int:
        """Move the duplicate's own properties onto the survivor before deleting it.

        The remap migrates edges only, so ``DETACH DELETE`` would otherwise take
        the duplicate's properties with it. That silently loses whatever only the
        duplicate knew: the ``description`` entity vector search embeds, and every
        typed value a structured source supplied.

        keep_existing: a value already on the survivor always wins, so a merge
        can never overwrite what the survivor knew.
        """
        try:
            res = await self._graph.query_raw(
                "MATCH (k:__Entity__ {id: $survivor_id}), (d:__Entity__ {id: $dup_id}) "
                "RETURN properties(k), properties(d)",
                {"survivor_id": survivor_id, "dup_id": dup_id},
            )
        except Exception as exc:
            logger.warning(f"Could not read properties for {dup_id} -> {survivor_id}: {exc}")
            return 0
        if not res.result_set:
            return 0
        keep_props, dup_props = res.result_set[0][0] or {}, res.result_set[0][1] or {}
        carry = {
            key: value
            for key, value in dup_props.items()
            if key not in self._NEVER_CARRY
            and value is not None
            and keep_props.get(key) in (None, "", [])
        }
        if not carry:
            return 0
        try:
            await self._graph.query_raw(
                "MATCH (k:__Entity__ {id: $survivor_id}) SET k += $carry",
                {"survivor_id": survivor_id, "carry": carry},
            )
        except Exception as exc:
            logger.warning(f"Property carry failed for {dup_id} -> {survivor_id}: {exc}")
            return 0
        return len(carry)

    async def _remap_entity_edges(self, dup_id: str, survivor_id: str) -> bool:
        """Remap all RELATES and MENTIONED_IN edges from duplicate to survivor.

        Returns:
            True if all remaps succeeded, False if any failed.
        """
        params = {"dup_id": dup_id, "survivor_id": survivor_id}
        ok = True
        for query in _REMAP_QUERIES:
            try:
                await self._graph.query_raw(query, params)
            except Exception as exc:
                logger.warning(f"Edge remap failed for {dup_id} -> {survivor_id}: {exc}")
                ok = False
        return ok
