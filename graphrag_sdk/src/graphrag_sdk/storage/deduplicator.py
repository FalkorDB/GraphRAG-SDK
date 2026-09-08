# GraphRAG SDK — Storage: Entity Deduplicator
# Finalize-time entity deduplication across the whole graph: exact name match,
# optional fuzzy embedding, and the LLM-judged cross-document phase
# (``judge_dedup.LLMJudgeDeduplicator``: similarity nominates, the LLM decides,
# a second shuffled pass must agree).

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from graphrag_sdk.core.providers import Embedder, LLMInterface
from graphrag_sdk.storage.judge_dedup import (
    LLMJudgeDeduplicator,
    merge_description,
    merge_description_list,
)

logger = logging.getLogger(__name__)

# Pagination safety net — well above any realistic graph size at typical
# batch sizes (10_000 * 500 = 5M entities). Trips only on a pathological
# server bug that keeps returning the same page.
_MAX_PAGINATION_ITERATIONS = 10_000

# Only English articles. Stripping foreign ones turned ``Los Angeles`` into
# ``angeles`` and ``Le Mans`` into ``mans`` -- collisions, not variants.
_LEADING_ARTICLE = re.compile(r"^(the|a|an)\s+")

# Words that carry no signal when forming an acronym.
_ACRONYM_STOPWORDS = frozenset({"of", "the", "and", "for", "de", "la", "del", "at", "in", "on"})


def normalize_entity_name(name: str) -> str:
    """Fold accents, punctuation and a leading English article for grouping.

    Dots inside a token are removed rather than turned into spaces, so ``A.I.``
    stays ``ai`` instead of becoming the single letter ``i``. A name with no
    ASCII fold at all (non-Latin script) keeps its case-folded original form.

    Deliberately does NOT strip generational suffixes: ``Elias Whitford, Jr.``
    and ``Elias Whitford`` are a father and a son, and merging them is a
    correctness bug rather than a cleanup.
    """
    original = re.sub(r"\s+", " ", str(name or "")).strip()
    s = unicodedata.normalize("NFKD", original).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    s = re.sub(r"(?<=\w)\.(?=\w|$)", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    cleaned = _LEADING_ARTICLE.sub("", s).strip() or s
    if cleaned:
        return cleaned
    # Names written entirely in a non-Latin script (東京, القاهرة) have no ASCII
    # fold. Returning "" would put every such name of a label in one group and
    # merge them all; fall back to the case-folded original instead.
    return original.casefold()


def _initials(name: str) -> str:
    words = [w for w in normalize_entity_name(name).split() if w not in _ACRONYM_STOPWORDS]
    return "".join(w[0] for w in words if w)


def is_acronym_of(short: str, long: str) -> bool:
    """Is ``short`` a plausible acronym of the multi-word ``long``?

    This is the only rule that can see ``AIHS`` = ``Ashford Island Historical
    Society``: their character similarity is 0.11 and their embedding similarity
    sits below every threshold worth using, so neither fuzzy matching nor
    vectors recover that pair.

    Kept strict on purpose (2-6 alphabetic characters; the long form must have
    at least two significant words). A looser version merges arbitrary
    unrelated short strings.
    """
    s = normalize_entity_name(short).replace(" ", "")
    if not (2 <= len(s) <= 6) or not s.isalpha():
        return False
    if len([w for w in normalize_entity_name(long).split() if w not in _ACRONYM_STOPWORDS]) < 2:
        return False
    return s == _initials(long)


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
# The survivor is bound with its own MATCH before every MERGE. This is a
# correctness requirement, not a style choice. Cypher MERGE on a *path* is
# all-or-nothing: with ``MERGE (s:__Entity__ {id: $survivor_id})-[nr]->(b)``
# the variable ``s`` is unbound, so when that relationship does not yet exist
# FalkorDB creates the whole pattern -- including a brand new, nameless
# ``__Entity__`` node carrying only ``id``. The remapped edge then attaches to
# that empty duplicate instead of the real survivor, so a "deduplication" run
# both invents entities and silently loses relationships. Reproduced minimally:
# 3 entities in, 4 out, "Nodes created: 1", two nodes sharing one id.
_REMAP_QUERIES = [
    # Outgoing RELATES from duplicate.
    "MATCH (dup:__Entity__ {id: $dup_id})-[r:RELATES]->(b:__Entity__) "
    "WHERE b.id <> $survivor_id "
    "MATCH (s:__Entity__ {id: $survivor_id}) "
    "MERGE (s)-[nr:RELATES]->(b) "
    "WITH r, nr, "
    "     coalesce(nr.source_chunk_ids, []) AS old, "
    "     coalesce(r.source_chunk_ids, []) AS contrib "
    "SET nr += properties(r) "
    "SET nr.source_chunk_ids = old + [c IN contrib WHERE NOT c IN old] "
    "DELETE r",
    # Incoming RELATES to duplicate.
    #
    # The ``MATCH (dup) WITH dup`` prefix is load-bearing, not style. Written
    # as a single pattern starting from ``(a:__Entity__)``, FalkorDB's planner
    # anchors on ``a`` and produces "Node By Label Scan | (a:__Entity__)" —
    # a full scan of every entity in the graph, once per merged duplicate.
    # That is why merge throughput fell 61.9 -> 10.3 merges/s between a 1K and
    # a 50K-node graph. The WITH barrier forces "Node By Index Scan |
    # (dup:__Entity__)" and the incoming remap gets ~7.5x faster at 20K nodes.
    # Reversing the arrow instead (``(dup)<-[r]-(a)``) does NOT help; the
    # planner still chooses the label scan.
    "MATCH (dup:__Entity__ {id: $dup_id}) "
    "WITH dup "
    "MATCH (a:__Entity__)-[r:RELATES]->(dup) "
    "WHERE a.id <> $survivor_id "
    "MATCH (s:__Entity__ {id: $survivor_id}) "
    "MERGE (a)-[nr:RELATES]->(s) "
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


class EntityDeduplicator:
    """Finalize-time entity deduplication engine.

    Phase 1 (always): Exact name match — groups entities by
    ``(normalized_name, label)`` to prevent cross-type merging,
    keeps the one with the longest description, remaps all
    RELATES and MENTIONED_IN edges, deletes duplicates.

    Phase 2 (optional): Fuzzy embedding match — embeds entity
    names, finds near-duplicates by cosine similarity, merges
    those too.

    Phase 3 (optional, ``judge_llm``): LLM-judged cross-document dedup over
    **every** entity in the graph at once — the ingest-time resolver only
    ever sees one document, so ``Airbus`` in one file and ``Airbus SE`` in
    another can never meet there; here they do. Name and description
    embeddings plus "A's name appears in B's description" nominate candidate
    pairs; candidates form small dense sets; the judge LLM partitions each
    set; a second pass over shuffled sets must agree. Agreed pairs are merged
    (survivor gains every member's label, ``" | "``-joined description and
    ``aliases``); disagreements are written as ``SAME_AS`` edges instead.

    Args:
        graph_store: Graph data access object with ``query_raw()`` method.
        embedder: Embedding provider for fuzzy dedup.
    """

    def __init__(self, graph_store: Any, embedder: Embedder) -> None:
        self._graph = graph_store
        self._embedder = embedder
        self.last_judge_stats: dict[str, int] = {}

    async def deduplicate(
        self,
        *,
        fuzzy: bool = False,
        # 0.95, not 0.9. Measured on the benchmark corpus: at 0.90 name-embedding
        # matching drops precision to 0.739 and wrongly merges 2 of 33
        # deliberate hard-negative pairs, for no recall the cheap tiers do not
        # already reach. At 0.95 precision is 0.938 with zero hard negatives
        # merged. The old 0.9 was an unswept guess, and it also disagreed with
        # the 0.95 used by LLMVerifiedResolution for the same job.
        similarity_threshold: float = 0.95,
        batch_size: int = 500,
        judge_llm: LLMInterface | None = None,
        judge_vote: bool = True,
    ) -> int:
        """Run deduplication and return total number of duplicates merged.

        Args:
            fuzzy: legacy name-embedding phase (merges on cosine alone; off by default).
            similarity_threshold: cosine floor for the fuzzy phase.
            batch_size: entities per query page.
            judge_llm: when given, run the LLM-judged cross-document phase after
                exact matching (see class docstring). Stats of the last run are
                in ``self.last_judge_stats``.
            judge_vote: require the second-pass agreement (default). ``False``
                halves LLM cost at roughly 3x the wrong-merge rate.
        """
        total = await self._deduplicate_exact(batch_size)

        if fuzzy:
            total += await self._deduplicate_fuzzy(batch_size, similarity_threshold)

        if judge_llm is not None:
            judge = LLMJudgeDeduplicator(
                self._graph,
                self._embedder,
                judge_llm,
                self._remap_entity_edges,
                vote=judge_vote,
            )
            self.last_judge_stats = await judge.deduplicate(batch_size)
            total += self.last_judge_stats["merged"]

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
            norm = normalize_entity_name(ent["name"])
            label = ent.get("label", "").strip().lower()
            groups.setdefault((norm, label), []).append(ent)

        merged_groups = self._merge_acronym_groups(list(groups.values()))

        merged = 0
        for group in merged_groups:
            if len(group) < 2:
                continue

            # Survivor: longest description
            group.sort(key=lambda e: len(e["description"]), reverse=True)
            survivor = group[0]
            duplicates = group[1:]

            absorbed: list[dict] = []

            for dup in duplicates:
                if not await self._remap_entity_edges(dup["id"], survivor["id"]):
                    logger.warning(f"Skipping deletion of {dup['id']} — edge remap incomplete")
                    continue
                try:
                    await self._graph.query_raw(
                        "MATCH (e:__Entity__ {id: $dup_id}) DETACH DELETE e",
                        {"dup_id": dup["id"]},
                    )
                    merged += 1
                    absorbed.append(dup)
                except Exception as exc:
                    logger.warning(f"Failed to delete duplicate entity {dup['id']}: {exc}")

            # The duplicate's description dies with the node, so anything it
            # said that the survivor did not is lost outright. Concatenated
            # with " | ", matching the judge phase's survivor rule, so both
            # mechanisms leave the same shape behind; the absorbed names are
            # kept in ``aliases`` for the same reason. Only duplicates actually
            # deleted are absorbed — a failed remap leaves its node in place.
            if absorbed:
                await self._write_merged_properties(survivor, absorbed)

        logger.info(f"EntityDeduplicator phase 1 (exact): merged {merged} duplicates")
        return merged

    async def _write_merged_properties(self, survivor: dict, absorbed: list[dict]) -> None:
        """Survivor keeps every member's description: ``descriptions`` is the
        list (no LLM summary, nothing dropped), ``description`` the ' | ' join,
        and ``aliases`` keeps the absorbed names that differ from the survivor's."""
        desc_list = merge_description_list([survivor, *absorbed])
        new_desc = merge_description(desc_list)
        aliases = [d["name"] for d in absorbed if d["name"] and d["name"] != survivor["name"]]
        try:
            await self._graph.query_raw(
                "MATCH (s:__Entity__ {id: $id}) "
                "SET s.description = $desc, s.descriptions = $descs, "
                "s.aliases = [x IN coalesce(s.aliases, []) + $aliases | x]",
                {"id": survivor["id"], "desc": new_desc, "descs": desc_list, "aliases": aliases},
            )
            survivor["description"] = new_desc
            survivor["descriptions"] = desc_list
        except Exception as exc:
            logger.warning(f"Failed to update merged properties on {survivor['id']}: {exc}")

    @staticmethod
    def _merge_acronym_groups(groups: list[list[dict]]) -> list[list[dict]]:
        """Union groups where one name is an acronym of another.

        Runs on the ``(normalised name, label)`` groups so an acronym attaches to
        an already-complete group. Uses union-find rather than pairwise merging
        so the result does not depend on iteration order.

        Two guards, because this fold deletes nodes without an LLM looking:
        the short and long form must share at least one label, and a short form
        that spells the initials of two or more long forms (``US`` = ``United
        States`` = ``Universal Studios``) is left alone rather than used as a
        hub that collapses unrelated entities. Ambiguous cases are the LLM
        judge's job.
        """
        parent = list(range(len(groups)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        names = [g[0]["name"] for g in groups]
        labels = [{(e.get("label") or "").strip().lower() for e in g} - {""} for g in groups]
        shorts = [
            i for i, n in enumerate(names) if len(normalize_entity_name(n).replace(" ", "")) <= 6
        ]
        longs = [i for i, n in enumerate(names) if len(normalize_entity_name(n).split()) >= 2]
        for i in shorts:
            matches = [
                j
                for j in longs
                if i != j and labels[i] & labels[j] and is_acronym_of(names[i], names[j])
            ]
            if len(matches) == 1:
                union(i, matches[0])
            elif len(matches) > 1:
                logger.info(
                    "Acronym %r matches %d long forms (%s); leaving unmerged",
                    names[i],
                    len(matches),
                    ", ".join(names[j] for j in matches),
                )

        combined: dict[int, list[dict]] = {}
        for i, g in enumerate(groups):
            combined.setdefault(find(i), []).extend(g)
        return list(combined.values())

    # ── Phase 2: Fuzzy embedding match ──

    async def _deduplicate_fuzzy(self, batch_size: int, similarity_threshold: float) -> int:
        import numpy as np

        # Re-fetch surviving entities (with labels for cross-type guard)
        offset = 0
        all_ids: list[str] = []
        all_names: list[str] = []
        all_labels: list[str] = []
        for _ in range(_MAX_PAGINATION_ITERATIONS):
            result = await self._graph.query_raw(
                "MATCH (e:__Entity__) "
                "RETURN e.id AS id, e.name AS name, "
                "HEAD([l IN labels(e) WHERE l <> '__Entity__']) AS label, "
                "e.descriptions AS descriptions "
                "SKIP $offset LIMIT $limit",
                {"offset": offset, "limit": batch_size},
            )
            if not result.result_set:
                break
            for row in result.result_set:
                all_ids.append(row[0])
                all_names.append(row[1] if len(row) > 1 and row[1] else str(row[0]))
                all_labels.append(row[2] if len(row) > 2 and row[2] else "")
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
                    merged_set.add(dup_id)

                    if not await self._remap_entity_edges(dup_id, survivor_id):
                        continue
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

    async def _fetch_all_entities(self, batch_size: int) -> list[dict]:
        """Fetch all entities in batches, including their primary label."""
        offset = 0
        entities: list[dict] = []
        for _ in range(_MAX_PAGINATION_ITERATIONS):
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
                entities.append(
                    {
                        "id": row[0],
                        "name": row[1] if len(row) > 1 and row[1] else str(row[0]),
                        "description": row[2] if len(row) > 2 and row[2] else "",
                        "label": row[3] if len(row) > 3 and row[3] else "",
                        "descriptions": list(row[4]) if len(row) > 4 and row[4] else [],
                    }
                )
            offset += batch_size
        else:
            logger.error(
                "Pagination exceeded %d iterations in _fetch_all_entities — aborting",
                _MAX_PAGINATION_ITERATIONS,
            )
        return entities

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
