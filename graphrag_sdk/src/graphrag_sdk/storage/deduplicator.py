# GraphRAG SDK — Storage: Entity Deduplicator
# Two-phase entity deduplication: exact name match + optional fuzzy embedding.
# Preserves label-aware grouping to prevent cross-type merging.

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from graphrag_sdk.core.providers import Embedder

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

# A short form of fewer than 3 letters is not evidence of anything: with a
# floor of 2, ``Bo`` spelled the initials of ``Ben Ottoson`` and the pair was
# merged with no LLM in the loop.
_MIN_ACRONYM_LEN = 3
_MAX_ACRONYM_LEN = 6


def normalize_entity_name(name: str) -> str:
    """Fold accents, punctuation and a leading English article for grouping.

    Accents are folded by dropping the combining marks of the NFKD form
    (``Jardín`` = ``Jardin``); letters and digits of every other script are
    kept. Folding to ASCII instead threw those letters away, so ``Отдел 5``
    and ``Кабинет 5`` both became ``5`` and ``東京, Japan`` and ``大阪, Japan``
    both became ``japan`` -- one of each pair was then deleted.

    Dots inside a token are removed rather than turned into spaces, so ``A.I.``
    stays ``ai`` instead of becoming the single letter ``i``.

    Deliberately does NOT strip generational suffixes: ``Elias Whitford, Jr.``
    and ``Elias Whitford`` are a father and a son, and merging them is a
    correctness bug rather than a cleanup.
    """
    original = re.sub(r"\s+", " ", str(name or "")).strip()
    s = unicodedata.normalize("NFKD", original)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    s = re.sub(r"(?<=\w)\.(?=\w|$)", "", s)
    s = re.sub(r"[\W_]+", " ", s).strip()
    cleaned = _LEADING_ARTICLE.sub("", s).strip() or s
    # A name made only of punctuation has nothing left; keep it distinct from
    # every other such name rather than pooling them under "".
    return cleaned or original.casefold()


def _acronym_key(normalized: str) -> str | None:
    """The letters of a plausible acronym, or ``None`` if the name is not one."""
    s = normalized.replace(" ", "")
    if not (_MIN_ACRONYM_LEN <= len(s) <= _MAX_ACRONYM_LEN):
        return None
    if not (s.isascii() and s.isalpha()):
        return None
    return s


def _expansion_key(normalized: str) -> str | None:
    """The initials of a multi-word name, or ``None`` if it has too few words."""
    words = [w for w in normalized.split() if w not in _ACRONYM_STOPWORDS]
    if len(words) < 2:
        return None
    return "".join(w[0] for w in words)


def is_acronym_of(short: str, long: str) -> bool:
    """Is ``short`` a plausible acronym of the multi-word ``long``?

    This is the only rule that can see ``AIHS`` = ``Ashford Island Historical
    Society``: their character similarity is 0.11 and their embedding similarity
    sits below every threshold worth using, so neither fuzzy matching nor
    vectors recover that pair.

    Kept strict on purpose (3-6 ASCII letters; the long form must have at
    least two significant words). A looser version merges arbitrary unrelated
    short strings.
    """
    key = _acronym_key(normalize_entity_name(short))
    if key is None:
        return False
    return key == _expansion_key(normalize_entity_name(long))


def _survivor_rank(entity: dict) -> tuple[bool, int]:
    """Sort key: higher ranks survive a merge.

    The long form outranks an acronym regardless of description length. The
    survivor's *name* is what ``backfill_entity_embeddings`` embeds, so keeping
    ``AIHS`` over ``Ashford Island Historical Society`` would leave the entity
    vector-indexed as an opaque string. Among names of the same kind the
    longest description wins.
    """
    normalized = entity.get("norm") or normalize_entity_name(entity.get("name") or "")
    return (_acronym_key(normalized) is None, len(entity.get("description") or ""))


def _merge_description(current: str, absorbed: str) -> str:
    """Join two descriptions with ``" | "``, keeping each segment once.

    Segments are compared individually, not whole strings: once a survivor
    holds ``"a lighthouse | first lit in 1871"``, absorbing a node described as
    ``"first lit in 1871"`` must not append it again. Across incremental
    finalize cycles that repetition compounds on hub entities.
    """
    segments: list[str] = []
    for text in (current, absorbed):
        for seg in (s.strip() for s in str(text or "").split(" | ")):
            if seg and seg not in segments:
                segments.append(seg)
    return " | ".join(segments)


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

# Folds the duplicate into the survivor and deletes it in ONE statement, so a
# failed write cannot leave the survivor un-updated with the duplicate already
# gone. Binding the survivor first also closes a race: ``_fetch_all_entities``
# snapshots the entity list up front, and a concurrent ``delete_document()``
# can remove a survivor mid-loop. The remap queries then match nothing, and a
# bare ``DETACH DELETE dup`` would destroy every edge the duplicate still
# holds. Here a missing survivor yields zero rows, nothing is deleted, and the
# empty result tells the caller the merge did not happen.
#
# Node-level ``source_chunk_ids`` are unioned like the RELATES ones: they feed
# ``CachedChunkExtraction`` through ``get_entities_for_chunks``, and deleting
# the duplicate outright dropped its half of the provenance. The CASE keeps a
# survivor with no provenance property from acquiring an empty list.
_ABSORB_QUERY_HEAD = (
    "MATCH (s:__Entity__ {id: $survivor_id}) "
    "MATCH (dup:__Entity__ {id: $dup_id}) "
    "WITH s, dup, "
    "     coalesce(s.source_chunk_ids, []) AS old, "
    "     coalesce(dup.source_chunk_ids, []) AS contrib "
    "SET s.source_chunk_ids = CASE WHEN size(contrib) = 0 THEN s.source_chunk_ids "
    "    ELSE old + [c IN contrib WHERE NOT c IN old] END"
)
_ABSORB_QUERY_TAIL = "WITH s, dup DETACH DELETE dup RETURN s.id"


class EntityDeduplicator:
    """Two-phase entity deduplication engine.

    Phase 1 (always): Exact name match — groups entities by
    ``(normalized_name, label)`` to prevent cross-type merging, folds an
    acronym into its unique same-label long form, keeps the long form (or,
    among like names, the one with the longest description), remaps all
    RELATES and MENTIONED_IN edges, deletes duplicates.

    Phase 2 (optional): Fuzzy embedding match — embeds entity
    names, finds near-duplicates by cosine similarity, merges
    those too.

    Every merge preserves what the duplicate carried: its description is
    joined onto the survivor's with ``" | "``, its name is recorded in the
    survivor's ``aliases`` list when it differs, and its node-level
    ``source_chunk_ids`` are unioned into the survivor's.

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
        # 0.95, not 0.9. Measured on the benchmark corpus: at 0.90 name-embedding
        # matching drops precision to 0.739 and wrongly merges 2 of 33
        # deliberate hard-negative pairs, for no recall the cheap tiers do not
        # already reach. At 0.95 precision is 0.938 with zero hard negatives
        # merged. The old 0.9 was an unswept guess, and it also disagreed with
        # the 0.95 used by LLMVerifiedResolution for the same job.
        similarity_threshold: float = 0.95,
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
            norm = normalize_entity_name(ent["name"])
            ent["norm"] = norm
            label = ent.get("label", "").strip().lower()
            groups.setdefault((norm, label), []).append(ent)

        merged_groups = self._merge_acronym_groups(list(groups.values()))

        merged = 0
        for group in merged_groups:
            if len(group) < 2:
                continue

            group.sort(key=_survivor_rank, reverse=True)
            survivor = group[0]
            for dup in group[1:]:
                if await self._absorb(survivor, dup):
                    merged += 1

        logger.info(f"EntityDeduplicator phase 1 (exact): merged {merged} duplicates")
        return merged

    async def _absorb(self, survivor: dict, dup: dict) -> bool:
        """Remap ``dup``'s edges onto ``survivor``, fold its data in, delete it.

        Returns ``True`` only when the duplicate was actually deleted. The
        in-memory ``survivor`` is updated so a group's later duplicates build
        on the merged description and aliases.
        """
        if not await self._remap_entity_edges(dup["id"], survivor["id"]):
            logger.warning(f"Skipping deletion of {dup['id']} — edge remap incomplete")
            return False

        sets: list[str] = []
        params: dict[str, Any] = {"survivor_id": survivor["id"], "dup_id": dup["id"]}

        # Concatenated with " | ", matching LLMVerifiedResolution's survivor
        # rule, so both mechanisms leave the same shape behind.
        description = _merge_description(
            survivor.get("description") or "", dup.get("description") or ""
        )
        if description != (survivor.get("description") or ""):
            sets.append("s.description = $desc")
            params["desc"] = description

        aliases = self._merged_aliases(survivor, dup)
        if aliases != list(survivor.get("aliases") or []):
            sets.append("s.aliases = $aliases")
            params["aliases"] = aliases

        query = _ABSORB_QUERY_HEAD
        if sets:
            query += " WITH s, dup SET " + ", ".join(sets)
        query += " " + _ABSORB_QUERY_TAIL

        try:
            result = await self._graph.query_raw(query, params)
        except Exception as exc:
            logger.warning(
                f"Failed to merge duplicate entity {dup['id']} into {survivor['id']}: {exc}"
            )
            return False
        if not (getattr(result, "result_set", None) or []):
            logger.warning(
                "Survivor %s no longer exists; duplicate %s left in place",
                survivor["id"],
                dup["id"],
            )
            return False

        survivor["description"] = description
        survivor["aliases"] = aliases
        return True

    @staticmethod
    def _merged_aliases(survivor: dict, dup: dict) -> list[str]:
        """The survivor's aliases plus the duplicate's name and aliases.

        A name that normalises to the survivor's own is a spelling variant,
        not an alias worth keeping; an acronym or a fuzzy-matched variant is.
        """
        survivor_norm = survivor.get("norm") or normalize_entity_name(survivor.get("name") or "")
        aliases = [a for a in (survivor.get("aliases") or []) if isinstance(a, str)]
        for candidate in [dup.get("name") or "", *(dup.get("aliases") or [])]:
            if not isinstance(candidate, str) or not candidate:
                continue
            if normalize_entity_name(candidate) == survivor_norm or candidate in aliases:
                continue
            aliases.append(candidate)
        return aliases

    @staticmethod
    def _merge_acronym_groups(groups: list[list[dict]]) -> list[list[dict]]:
        """Union groups where one name is an acronym of another.

        Runs on the ``(normalised name, label)`` groups so an acronym attaches to
        an already-complete group. Uses union-find rather than pairwise merging
        so the result does not depend on iteration order.

        Long forms are bucketed by their initials, so each short form costs one
        dict lookup: each name is normalised once and the fold is linear in the
        number of groups. The pairwise version re-normalised both names for
        every (short, long) pair -- ~51 µs each, 20 s for 400k pairs -- which is
        quadratic inside ``finalize()``.

        Two guards, because this fold deletes nodes without an LLM looking:
        the short and long form must share at least one label, and a short form
        that spells the initials of two or more long forms (``ABC`` = ``American
        Broadcasting Company`` = ``Australian Broadcasting Corporation``) is
        left alone rather than used as a hub that collapses unrelated entities.
        Ambiguous cases are the LLM judge's job.
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
        norms = [g[0].get("norm") or normalize_entity_name(n) for g, n in zip(groups, names)]
        labels = [{(e.get("label") or "").strip().lower() for e in g} - {""} for g in groups]

        by_initials: dict[str, list[int]] = {}
        for j, norm in enumerate(norms):
            initials = _expansion_key(norm)
            if initials is not None:
                by_initials.setdefault(initials, []).append(j)

        for i, norm in enumerate(norms):
            key = _acronym_key(norm)
            if key is None:
                continue
            matches = [j for j in by_initials.get(key, ()) if i != j and labels[i] & labels[j]]
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

        # Re-fetch surviving entities (with labels for the cross-type guard).
        entities = await self._fetch_all_entities(batch_size)
        if len(entities) < 2:
            return 0

        raw_vectors = await self._embedder.aembed_documents([e["name"] for e in entities])
        valid = [(ent, vec) for ent, vec in zip(entities, raw_vectors) if vec]
        if len(valid) < 2:
            return 0

        v_entities = [ent for ent, _ in valid]
        v_ids = [ent["id"] for ent in v_entities]
        v_labels = [ent["label"] for ent in v_entities]
        vectors = [vec for _, vec in valid]

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
                    merged_set.add(v_ids[gj])
                    if await self._absorb(v_entities[gi], v_entities[gj]):
                        merged_count += 1

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
                "HEAD([l IN labels(e) WHERE l <> '__Entity__']) AS label, "
                "e.aliases AS aliases "
                "SKIP $offset LIMIT $limit",
                {"offset": offset, "limit": batch_size},
            )
            if not result.result_set:
                break
            for row in result.result_set:
                aliases = row[4] if len(row) > 4 and isinstance(row[4], list) else []
                entities.append(
                    {
                        "id": row[0],
                        "name": row[1] if len(row) > 1 and row[1] else str(row[0]),
                        "description": row[2] if len(row) > 2 and row[2] else "",
                        "label": row[3] if len(row) > 3 and row[3] else "",
                        "aliases": [a for a in aliases if isinstance(a, str)],
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
