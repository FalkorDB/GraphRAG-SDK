# GraphRAG SDK — Storage: Entity Deduplicator
# Entity deduplication over the whole graph: exact name match, optional fuzzy
# embedding, and optionally the caller's resolution strategy — the one ingest()
# runs within a document — judging pairs across documents and tables.
# Preserves label-aware grouping to prevent cross-type merging.

from __future__ import annotations

import logging
import re
import unicodedata
from typing import TYPE_CHECKING, Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, GraphNode, GraphRelationship
from graphrag_sdk.core.providers import Embedder
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import DEFAULT_ENTITY_TYPES
from graphrag_sdk.ingestion.resolution_strategies.base import (
    RESOLUTION_ASK_PAIRS,
    RESOLUTION_DISTINCT_IDS,
    RESOLUTION_REJECTED_PAIRS,
    RESOLUTION_SKIP_PAIRS,
)
from graphrag_sdk.storage.identity import NearMiss, canonical_key, find_near_misses

if TYPE_CHECKING:
    from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy

#: Edge between two entities a resolver judged to be two things. Read back on the
#: next run so the pair is neither asked about again nor merged by a threshold,
#: and dropped with either node, so a re-extracted entity is judged afresh.
DISTINCT_FROM = "DISTINCT_FROM"

#: Labels every extractor always has. A declared label in this set can never have
#: been a guess the extractor was missing, so it is never a target for adoption.
_BUILTIN_LABELS: frozenset[str] = frozenset(t.strip().lower() for t in DEFAULT_ENTITY_TYPES)

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
#
# Two properties of these queries are load-bearing and easy to undo by
# "simplifying" them:
#
# 1. The survivor is bound by its own ``MATCH`` before the ``MERGE``.
#    Cypher MERGE on a *path* is all-or-nothing: writing the survivor inline
#    — ``MERGE (s:__Entity__ {id: $survivor_id})-[nr:RELATES]->(b)`` — leaves
#    ``s`` unbound, so when that relationship does not yet exist FalkorDB
#    creates the whole pattern, including a brand new, nameless
#    ``__Entity__`` node carrying only ``id``. The remapped edge attaches to
#    that ghost instead of the real survivor, so a "deduplication" run both
#    invents entities and silently loses relationships. Reproduced minimally:
#    3 entities in, 4 out, "Nodes created: 1", two nodes sharing one id.
#    There is no uniqueness constraint on ``__Entity__.id`` to catch it.
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


def _string_list(value: Any) -> list[str]:
    """The strings in a list-valued property; anything else counts as empty."""
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def union_chunk_ids(existing: Any, incoming: Any) -> list[str]:
    """``existing`` followed by every id in ``incoming`` it lacks, order kept.

    The Python side of the union the RELATES queries above do in Cypher, for
    the places that read both nodes' properties before writing — a merge's
    property carry. Anything that is not a list of strings counts as empty.
    """
    old, new = _string_list(existing), _string_list(incoming)
    return old + [c for c in new if c not in old]


def properties_to_carry(
    keep: dict[str, Any], dup: dict[str, Any], *, never: frozenset[str]
) -> dict[str, Any]:
    """What a merge copies from the node being deleted onto the one that stays.

    keep_existing: a value already on the survivor always wins, and an empty
    string or list counts as absent. ``source_chunk_ids`` is the one property
    both sides hold at once — the survivor was mentioned wherever either was —
    so it becomes the union. ``never`` names what stays behind whatever the
    survivor lacks; ``id`` and ``embedding`` always do.

    ``description`` is the exception to keep_existing: the two are joined with
    ``" | "`` (:func:`_merge_description`), each segment kept once. The
    survivor is chosen for the reproducibility of its identity — a table's key,
    then how much of the graph points at it — not for what it knows, and the
    node that lost on those grounds is often the one the PDF described.
    Measured as a two-line "from the PDF" description dropped for a stub's
    one-liner. Safe to change: the entity embedding is of the name, not of this.
    """
    carry = {
        key: value
        for key, value in dup.items()
        if key not in never and value is not None and keep.get(key) in (None, "", [])
    }
    description = _merge_description(
        str(keep.get("description") or ""), str(dup.get("description") or "")
    )
    if description != str(keep.get("description") or ""):
        carry["description"] = description
    else:
        carry.pop("description", None)
    provenance = union_chunk_ids(keep.get("source_chunk_ids"), dup.get("source_chunk_ids"))
    if provenance and provenance != keep.get("source_chunk_ids"):
        carry["source_chunk_ids"] = provenance
    return carry


def _keep_declared_identities_apart(
    survivor: dict[str, Any], duplicates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Drop candidates a mapping already said are a *different* thing.

    Grouping is by display name, and two rows of the same table can share one:
    two people called John Smith, keyed ``E-1`` and ``E-7``. Merging them deletes
    one — measured as a five-row export arriving as four people, with the loss
    reported as a successful dedup. A mapping that declares a key is asserting
    identity, so two mapped nodes with different ids are two things and the name
    they happen to share is not evidence against that.

    An extracted node has no declared key, so it stays mergeable: that is the
    join this whole path exists for.
    """
    if survivor.get("is_stub") is None:
        return duplicates
    kept: list[dict[str, Any]] = []
    for dup in duplicates:
        if dup.get("is_stub") is not None and dup.get("id") != survivor.get("id"):
            # Rows that share a display name are keyed apart at write time, and
            # rows whose names merely look alike were never one id. Either way
            # two mapped ids are two rows, and a name is not evidence otherwise.
            logger.info(
                "Not merging %s into %s: both were written from a declared key, "
                "so the shared name %r is two different rows",
                dup.get("id"),
                survivor.get("id"),
                survivor.get("name"),
            )
            continue
        kept.append(dup)
    return kept


def _mentions_two_rows_could_own(
    survivor: dict[str, Any], group: list[dict[str, Any]]
) -> list[NearMiss]:
    """Extracted nodes in ``group`` that cannot be given to one row.

    Two keyed rows share the name and a passage mentions it: the passage fits
    both rows equally, and the survivor rank picks one by degree, description
    length or id -- none of which say which row the passage meant. Measured as a
    note about one Alice Smith attaching to the other, key and columns included.
    An extracted node in that group stays its own node and is reported, paired
    with each row it could be, for a resolver or the user to decide.

    Empty when fewer than two distinct keyed ids are in the group: with one row
    the mention has exactly one place to go, which is the join the exact phase
    exists for.
    """
    rows = {ent["id"]: ent for ent in group if ent.get("is_stub") is not None}
    if len(rows) < 2:
        return []
    label = survivor.get("label", "")
    reason = f"same name as {len(rows)} keyed rows; which one is not decidable from the name"
    return [
        NearMiss(
            label=label,
            name_a=mention.get("name", ""),
            name_b=row.get("name", ""),
            id_a=mention["id"],
            id_b=row["id"],
            reason=reason,
            bridges_a_declared_source=True,
        )
        for mention in group
        if mention.get("is_stub") is None
        for row in rows.values()
    ]


def _clusters(
    remap: dict[str, str], by_id: dict[str, dict[str, Any]]
) -> list[list[dict[str, Any]]]:
    """Groups of entities a resolver's remap says are one thing, one label each.

    ``remap`` is ``duplicate id -> survivor id`` and may chain (``a -> b``,
    ``b -> c``), so ids are followed to their root. A group is then split by
    label: a resolver may merge "Apple" the company into "Apple" the fruit, and
    this class never does — phase 1 reports such a pair instead. Ids the graph
    does not hold are ignored.
    """

    def root(entity_id: str) -> str:
        seen = {entity_id}
        while entity_id in remap and remap[entity_id] != entity_id:
            entity_id = remap[entity_id]
            if entity_id in seen:
                break
            seen.add(entity_id)
        return entity_id

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entity_id in set(remap) | set(remap.values()):
        entity = by_id.get(entity_id)
        if entity is None:
            continue
        key = (root(entity_id), (entity.get("label") or "").strip().lower())
        grouped.setdefault(key, []).append(entity)
    return [members for members in grouped.values() if len(members) > 1]


def _survivor_rank(entity: dict[str, Any]) -> tuple[int, int, int, int, int, int, str]:
    """Rank candidates so the most reproducible identity survives a merge.

    Ordered by:

    1. **Written from a table.** A structured node carries ``entity_key``, the
       value its row is keyed by, and the next ingest of that source finds it by
       that key whatever the name became. A node extracted from prose is keyed on
       a surface form the model happened to produce. If the prose node survives,
       the key is gone with the other, and re-ingesting the table recreates the
       row as a *second* node: measured as two ``E-1`` people, one titled
       "Engineer" and one "engineer". Keeping the keyed node is what makes
       re-ingest idempotent after resolution.
    2. **Real over placeholder.** A stub was created by a foreign key and knows
       only an id and a name.
    3. **Long form over acronym.** The survivor's *name* is what
       ``backfill_entity_embeddings`` embeds, so keeping ``AIHS`` over
       ``Ashford Island Historical Society`` would leave the entity
       vector-indexed as an opaque string, whatever else recommended it.
    4. **Best connected.** Between two nodes of the same provenance, the one the
       rest of the graph points at keeps its name. Measured: a resolver judged
       ``Austria`` and ``Republik Österreich`` one country — correctly, from a
       single legislative citation — and the longer-description rule below then
       renamed the node fifty-six table rows and seventy passages pointed at
       after the one nobody did. Every later ``WHERE c.name CONTAINS 'Austria'``
       found nothing. Degree is ``RELATES`` plus ``MENTIONED_IN``; a node fetched
       without one ranks as if it had none.
    5. **Longest description**, the original rule, which still decides between
       two nodes nothing else separates.
    6. **Longest name.** Grouping is by canonical key, so a group holds different
       spellings of one name and the survivor's is the one the graph keeps.
       Length picks the written-out form — "Globex Limited" over "Globex Ltd",
       "Acme Corporation" over "Acme Corp".
    7. **Id**, purely to make the outcome deterministic. Entities are fetched
       with no ``ORDER BY`` and :meth:`list.sort` is stable, so without a total
       ordering a tie resolves to whatever order the server happened to return
       and the same data can settle on a different display name from run to run.
       Under exact-equality grouping every name in a group was identical, so this
       could not be observed; under canonical grouping it can.

    ``is_stub`` is the marker because only a mapped source writes it: ``None``
    means the node came from extraction.
    """
    is_stub = entity.get("is_stub")
    normalized = entity.get("norm") or normalize_entity_name(entity.get("name") or "")
    return (
        0 if is_stub is None else 1,
        0 if is_stub else 1,
        0 if _acronym_key(normalized) is not None else 1,
        int(entity.get("degree") or 0),
        len(entity.get("description") or ""),
        len(entity.get("name") or ""),
        str(entity.get("id") or ""),
    )


# Edges that count toward a node's degree when picking a merge survivor: the
# ones a question can reach it by. ``DISTINCT_FROM`` is bookkeeping, not reach.
_DEGREE_EXPR = "size((e)-[:RELATES]-()) + size((e)-[:MENTIONED_IN]->())"


class EntityDeduplicator:
    """Entity deduplication engine over the whole graph.

    Phase 1 (always): Exact name match — groups entities by
    ``(canonical_name, label)`` to prevent cross-type merging, folds an
    acronym into its unique same-label long form, keeps the survivor
    :func:`_survivor_rank` picks (a table row over a mention, the long form
    over an acronym, then the best-connected and best-described), remaps all
    RELATES and MENTIONED_IN edges, deletes duplicates.

    Phase 2 (optional): Fuzzy embedding match — embeds entity
    names, finds near-duplicates by cosine similarity, merges
    those too.

    Phase 3 (optional): The caller's resolution strategy — the same
    one ``ingest()`` runs within a document — is shown the whole graph
    and asked which surviving entities are one thing. It decides
    identity; the merge follows this class's rules, so a table row
    always survives a mention of it. See :meth:`_deduplicate_with_resolver`.

    Every merge preserves what the duplicate carried: its description is
    joined onto the survivor's with ``" | "``, its name is recorded in the
    survivor's ``aliases`` list when it differs, its node-level
    ``source_chunk_ids`` are unioned into the survivor's, and any property
    the survivor lacks is copied over. See :meth:`_absorb`.

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
        # Pairs the last run judged probably-the-same and deliberately left alone.
        self.near_misses: list[NearMiss] = []
        # Extracted nodes the exact phase would not fold into one of several rows
        # sharing their name. Folded into near_misses by _report_near_misses, and
        # held apart through the later phases (see _undecidable_pairs).
        self._ambiguous_mentions: list[NearMiss] = []
        # Merges the resolver decided on the last run, as "label 'dup' -> 'survivor'".
        self.resolved_pairs: list[str] = []
        # Pairs the resolver judged distinct on the last run, as "label 'a' | 'b'".
        self.rejected_pairs: list[str] = []

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
        declared_labels: set[str] | None = None,
        resolver: ResolutionStrategy | None = None,
    ) -> int:
        """Run deduplication and return total number of duplicates merged.

        ``declared_labels`` are labels a structured mapping declared. They are
        treated as authoritative about type, which lets the one safe kind of
        cross-label merge happen: see :meth:`_adopt_into_declared_labels`.

        ``resolver`` is a resolution strategy to judge the pairs no rule can:
        the same kind ``ingest()`` accepts, here applied across the whole graph.
        Without one, only spelling variants merge and the rest is reported.
        """
        self._declared_labels = {label.strip().lower() for label in (declared_labels or set())}
        self.resolved_pairs = []
        self.rejected_pairs = []
        self._ambiguous_mentions = []
        total = await self._deduplicate_exact(batch_size)

        if fuzzy:
            total += await self._deduplicate_fuzzy(batch_size, similarity_threshold)

        if resolver is not None:
            total += await self._deduplicate_with_resolver(resolver, batch_size)

        # What is left over that probably should not be. Reported, never merged.
        await self._report_near_misses(batch_size)

        logger.info(f"EntityDeduplicator total: {total} duplicates merged")
        return total

    @property
    def _undecidable_pairs(self) -> set[frozenset[str]]:
        """``(mention, row)`` pairs the exact phase found no way to decide.

        A mention whose name two keyed rows share fits both equally, and the
        later phases would otherwise settle it on the one evidence phase 1 had
        already rejected: identical names embed identically, so a cosine cut or
        a resolver's hard-merge band folds the mention into whichever row ranks
        higher. Measured as default ``finalize()`` attaching a passage's facts
        and provenance to one of two Alice Smiths that ``resolve=False`` had
        correctly left alone. These pairs are skipped in every phase and
        reported instead; a name shared by two rows is not disambiguated by
        looking at the name harder.
        """
        return {frozenset((miss.id_a, miss.id_b)) for miss in self._ambiguous_mentions}

    async def _report_near_misses(self, batch_size: int) -> None:
        """Record surviving entities that probably denote one thing.

        The merge is exact string equality on the display name, so two sources
        spelling a name differently leave two nodes and nothing says so: an HR
        export's "Maya Ellison" and a board review's "M. Ellison" end up as two
        people, one holding her age and the other what she did, and every question
        needing both comes back wrong while looking answered.

        Found here and **not** merged. Merging on a guess is destructive and
        irreversible; two nodes are recoverable, and a user told which pairs look
        wrong can fix the source, which fixes it for good. Deciding a pair needs
        judgement this class does not have — it holds no LLM — so it reports,
        and a pair a resolver has already decided against is not a guess any
        more and is left out.
        """
        try:
            survivors = await self._fetch_all_entities(batch_size)
            decided = await self._fetch_distinct_pairs()
            alive = {ent["id"] for ent in survivors}
            # A mention left between two rows first: it is an exact name match
            # that did not merge, which is the finding most worth a look. One a
            # later phase (the resolver) did settle is no longer open.
            open_mentions = [
                miss
                for miss in self._ambiguous_mentions
                if miss.id_a in alive and miss.id_b in alive
            ]
            self.near_misses = [
                miss
                for miss in open_mentions + find_near_misses(survivors)
                if frozenset((miss.id_a, miss.id_b)) not in decided
            ]
        except Exception:
            # A report is never worth failing a finalize over.
            logger.debug("Near-miss detection failed", exc_info=True)
            self.near_misses = []
            return

        if not self.near_misses:
            return
        bridging = sum(1 for m in self.near_misses if m.bridges_a_declared_source)
        logger.warning(
            "%d name(s) look like one thing under a single label and did not merge%s. "
            "Reported, not merged — merging on a guess cannot be undone. First: %s",
            len(self.near_misses),
            f", {bridging} joining a declared source to extracted text" if bridging else "",
            "; ".join(str(m) for m in self.near_misses[:3]),
        )

    # ── Phase 1: Exact name match ──

    async def _deduplicate_exact(self, batch_size: int) -> int:
        entities = await self._fetch_all_entities(batch_size)
        if len(entities) < 2:
            logger.info("EntityDeduplicator: fewer than 2 entities, nothing to dedup")
            return 0

        # Group by (canonical name, label): the label keeps types from merging,
        # the canonical key lets two spellings of one name land in one group.
        # Exact lowercase equality was the old key, and it left "Globex Ltd" and
        # "Globex Limited" as two organizations — one holding the address, the
        # other the revenue. See identity.canonical_key for what it does and does
        # not unify, and why word order is preserved.
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for ent in entities:
            key = canonical_key(ent["name"]) or ent["name"].strip().lower()
            # The acronym fold and the alias rule read this lighter form.
            ent["norm"] = normalize_entity_name(ent["name"])
            label = ent.get("label", "").strip().lower()
            groups.setdefault((key, label), []).append(ent)

        # An acronym and its unique same-label long form are one group too.
        merged_groups = self._merge_acronym_groups(list(groups.values()))

        merged = 0
        deleted: set[str] = set()
        for group in merged_groups:
            if len(group) < 2:
                continue

            group.sort(key=_survivor_rank, reverse=True)
            survivor = group[0]
            duplicates = _keep_declared_identities_apart(survivor, group[1:])
            ambiguous = _mentions_two_rows_could_own(survivor, group)
            if ambiguous:
                # The survivor is one of the rows, and the rank chose it for
                # reasons that say nothing about which row a mention meant.
                self._ambiguous_mentions.extend(ambiguous)
                duplicates = [dup for dup in duplicates if dup.get("is_stub") is not None]
                logger.info(
                    "Not merging %d mention(s) of %r into one of %d keyed rows: the "
                    "name fits every row equally. Reported in probable_duplicates",
                    len({m.id_a for m in ambiguous}),
                    survivor.get("name"),
                    len({m.id_b for m in ambiguous}),
                )

            for dup in duplicates:
                if await self._absorb(survivor, dup):
                    merged += 1
                    deleted.add(dup["id"])

        # What follows works from the groups too, and a node deleted above would
        # otherwise be "adopted" a second time — a MATCH on nothing succeeds, and
        # the count reports a merge that never happened.
        for group in groups.values():
            group[:] = [ent for ent in group if ent["id"] not in deleted]

        merged += await self._adopt_into_declared_labels(groups)
        logger.info(f"EntityDeduplicator phase 1 (exact): merged {merged} duplicates")
        self._report_cross_label_names(groups)
        return merged

    async def _absorb(self, survivor: dict[str, Any], dup: dict[str, Any]) -> bool:
        """Remap ``dup``'s edges onto ``survivor``, fold its data in, delete it.

        The remap migrates edges only, so ``DETACH DELETE`` would otherwise take
        the duplicate's properties with it. That silently loses whatever only
        the duplicate knew: the description that came from the PDF, every typed
        value a structured source supplied, and ``is_stub`` — a survivor that
        absorbed a table's row is now the node that row re-syncs to. Both nodes
        are read back first; the policy for what crosses over is
        :func:`properties_to_carry`, with the description joined and the
        duplicate's name recorded as an alias.

        The survivor's update and the duplicate's deletion are one statement, so
        a failure between them cannot leave the text gone and the node still
        there, and a survivor that no longer exists (deleted by an earlier merge
        in the same run) is detected rather than silently written past.

        Returns ``True`` only when the duplicate was actually deleted. The
        in-memory ``survivor`` is updated so a group's later duplicates build
        on the merged description and aliases.
        """
        if not await self._remap_entity_edges(dup["id"], survivor["id"]):
            logger.warning(f"Skipping deletion of {dup['id']} — edge remap incomplete")
            return False

        keep_props, dup_props = await self._read_properties(survivor["id"], dup["id"])
        if keep_props is not None and dup_props is not None:
            # The graph is the truth about what each node holds; the fetched row
            # carries only the columns the phases rank on.
            survivor["description"] = keep_props.get("description") or ""
            survivor["aliases"] = _string_list(keep_props.get("aliases"))
            dup = {
                **dup,
                "description": dup_props.get("description") or "",
                "aliases": _string_list(dup_props.get("aliases")),
            }

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

        carry: dict[str, Any] = {}
        if keep_props is not None and dup_props is not None:
            carry = properties_to_carry(keep_props, dup_props, never=self._NEVER_CARRY)
            # Handled above, or by the absorb query itself.
            for handled in ("description", "aliases", "source_chunk_ids"):
                carry.pop(handled, None)
        if carry:
            sets.append("s += $carry")
            params["carry"] = carry

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
        if "is_stub" in carry:
            survivor["is_stub"] = carry["is_stub"]
        return True

    async def _read_properties(
        self, survivor_id: str, dup_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | tuple[None, None]:
        """Both nodes' properties, or ``(None, None)`` if they could not be read.

        A failed read is not a failed merge: the edges are already remapped and
        the absorb query still joins what the fetched rows know.
        """
        try:
            res = await self._graph.query_raw(
                "MATCH (k:__Entity__ {id: $survivor_id}), (d:__Entity__ {id: $dup_id}) "
                "RETURN properties(k), properties(d)",
                {"survivor_id": survivor_id, "dup_id": dup_id},
            )
        except Exception as exc:
            logger.warning(f"Could not read properties for {dup_id} -> {survivor_id}: {exc}")
            return None, None
        rows = getattr(res, "result_set", None) or []
        if not rows or len(rows[0]) < 2:
            return None, None
        return rows[0][0] or {}, rows[0][1] or {}

    @staticmethod
    def _merged_aliases(survivor: dict[str, Any], dup: dict[str, Any]) -> list[str]:
        """The survivor's aliases plus the duplicate's name and aliases.

        A name that normalises to the survivor's own is a spelling variant,
        not an alias worth keeping; an acronym or a fuzzy-matched variant is.
        """
        survivor_norm = survivor.get("norm") or normalize_entity_name(survivor.get("name") or "")
        aliases = _string_list(survivor.get("aliases"))
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

        # The same refusals the exact phase makes. Two keyed nodes are two rows
        # whatever their names embed like, a pair the resolver already judged
        # distinct is not re-decided by a cosine score, and a mention two rows
        # could own is not handed to either by one.
        keyed = {ent["id"] for ent in entities if ent.get("is_stub") is not None}
        decided = await self._fetch_distinct_pairs() | self._undecidable_pairs

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
                    and not (v_ids[gi] in keyed and v_ids[gj] in keyed)
                    and frozenset((v_ids[gi], v_ids[gj])) not in decided
                ):
                    survivor, dup = v_entities[gi], v_entities[gj]
                    # Array order is arbitrary here, so apply the same rule as
                    # the exact phase rather than keeping whichever came first.
                    if _survivor_rank(dup) > _survivor_rank(survivor):
                        survivor, dup = dup, survivor
                    merged_set.add(dup["id"])
                    if await self._absorb(survivor, dup):
                        merged_count += 1

        logger.info(
            f"EntityDeduplicator phase 2 (fuzzy): merged {merged_count} additional duplicates"
        )
        return merged_count

    # ── Phase 3: The caller's resolver, over the whole graph ──

    async def _deduplicate_with_resolver(
        self, resolver: ResolutionStrategy, batch_size: int
    ) -> int:
        """Let the ingest-time resolver judge the whole graph, then merge by our rules.

        A resolution strategy sees the entities of one document at a time, so
        the pair it was built to decide — "Ms. Raman" in a market note against
        "Priya Raman" in the HR export — is one it never meets: the two arrive in
        different calls, and by the time both exist nothing asks. This presents
        every surviving entity to the same strategy as one ``GraphData``, with
        the RELATES edges between them, so a resolver that reads neighbours reads
        the whole graph's.

        The split of responsibilities is deliberate. The resolver decides
        *identity*: which ids denote one thing. This class decides *how to
        merge*, with the rules that make a table's row the anchor of its entity:

        - :func:`_survivor_rank` keeps the node whose id came from a declared
          key, so the merged entity is the one the next re-sync of the table
          finds again;
        - :func:`_keep_declared_identities_apart` refuses to merge two rows of a
          table into each other, whatever they are called — a mapping with a key
          has already said they are two things;
        - :meth:`_absorb` keeps every value the survivor holds, so a typed
          value a table supplied is never overwritten by what prose said;
        - labels never merge: the resolver's own cross-label merges are dropped,
          as phase 1 already reports those rather than deciding them.

        A structured node has no ``description``; its evidence is the typed
        values the table signed onto it, which are rendered as one so the
        resolver has something to judge against the document's sentence.
        """
        # Nothing without a name can be judged to be anything, so a nameless
        # entity — a fact row keyed on a reading id — is not shown to the resolver.
        entities = [e for e in await self._fetch_all_entities(batch_size) if e.get("name")]
        if len(entities) < 2:
            return 0
        by_id = {entity["id"]: entity for entity in entities}
        await self._describe_structured_entities(by_id)

        nodes = []
        for entity in entities:
            properties: dict[str, Any] = {"name": entity["name"]}
            if entity.get("description"):
                properties["description"] = entity["description"]
            label = entity["label"] or "__Entity__"
            nodes.append(GraphNode(id=entity["id"], label=label, properties=properties))
        relationships = await self._fetch_relates_edges(batch_size)
        graph_data = GraphData(nodes=nodes, relationships=relationships)

        # What this class knows and the resolver cannot: the pairs a previous run
        # already decided against, the pairs phase 1 found undecidable, and the
        # pairs a name rule says look like one thing — "M. Ellison" beside "Maya
        # Ellison" — which an embedding cut tuned for one document's spellings
        # would not surface.
        decided = await self._fetch_distinct_pairs()
        undecidable = self._undecidable_pairs
        # Two keyed nodes are two rows, and :func:`_keep_declared_identities_apart`
        # would refuse the merge anyway; saying so up front saves the resolver a
        # call per pair, and for a table of n rows that is most of the calls.
        keyed = {e["id"] for e in entities if e.get("is_stub") is not None}
        near = {
            frozenset((miss.id_a, miss.id_b))
            for miss in find_near_misses(entities)
            if frozenset((miss.id_a, miss.id_b)) not in decided
            and frozenset((miss.id_a, miss.id_b)) not in undecidable
            and not {miss.id_a, miss.id_b} <= keyed
        }
        ctx = Context(
            metadata={
                RESOLUTION_SKIP_PAIRS: decided | undecidable,
                RESOLUTION_DISTINCT_IDS: keyed,
                RESOLUTION_ASK_PAIRS: near,
            }
        )

        try:
            result = await resolver.resolve(graph_data, ctx)
        except Exception as exc:
            logger.warning("Resolver %s failed over the graph: %s", type(resolver).__name__, exc)
            return 0

        rejected = set(ctx.metadata.get(RESOLUTION_REJECTED_PAIRS) or ()) - decided
        await self._remember_distinct(rejected, by_id, type(resolver).__name__)

        if not result.remap:
            logger.info("EntityDeduplicator phase 3 (resolver): nothing to merge")
            return 0

        merged = 0
        for members in _clusters(result.remap, by_id):
            members.sort(key=_survivor_rank, reverse=True)
            survivor = members[0]
            duplicates = _keep_declared_identities_apart(survivor, members[1:])
            duplicates = self._keep_undecidable_mentions_apart(survivor, duplicates, members)
            for dup in duplicates:
                if not await self._absorb(survivor, dup):
                    continue
                merged += 1
                pair = f"{survivor['label']} {dup['name']!r} -> {survivor['name']!r}"
                self.resolved_pairs.append(pair)
                logger.info("Resolver merged %s", pair)

        logger.info(f"EntityDeduplicator phase 3 (resolver): merged {merged} duplicates")
        return merged

    def _keep_undecidable_mentions_apart(
        self,
        survivor: dict[str, Any],
        duplicates: list[dict[str, Any]],
        members: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop mentions the resolver put on a row without saying which row.

        The hints in ``ctx.metadata`` are advisory, and a strategy is free to
        ignore them; the refusal has to hold here too. A mention is kept apart
        when phase 1 found it between two rows, or when the cluster itself holds
        two keyed rows and the mention — the same ambiguity, found by a resolver
        that clustered all three. What remains is a mention the resolver paired
        with exactly one row, which is a decision, not a coin toss.
        """
        undecidable = self._undecidable_pairs
        if not undecidable and len({m["id"] for m in members if m.get("is_stub") is not None}) < 2:
            return duplicates
        held = {miss.id_a for miss in _mentions_two_rows_could_own(survivor, members)}
        kept: list[dict[str, Any]] = []
        for dup in duplicates:
            if dup["id"] in held or frozenset((dup["id"], survivor["id"])) in undecidable:
                logger.info(
                    "Not merging %s into %s: the name %r belongs to more than one keyed "
                    "row and nothing says which. Reported in probable_duplicates",
                    dup["id"],
                    survivor["id"],
                    dup.get("name"),
                )
                continue
            kept.append(dup)
        return kept

    async def _describe_structured_entities(self, by_id: dict[str, dict[str, Any]]) -> None:
        """Give each structured entity a description made of its signed values.

        ``employees__employee_id: E-3, employees__age: 39``
        becomes ``employees: employee_id E-3, age 39`` — the
        table named, because that is where the facts came from, and readable
        enough for a resolver's prompt to set beside a sentence of prose.
        Entities that already have a description, and every extracted entity,
        are left as they are.
        """
        wanted = [
            eid
            for eid, entity in by_id.items()
            if entity.get("is_stub") is not None and not entity.get("description")
        ]
        if not wanted:
            return
        try:
            result = await self._graph.query_raw(
                "MATCH (e:__Entity__) WHERE e.id IN $ids RETURN e.id, properties(e)",
                {"ids": wanted},
            )
        except Exception:
            logger.debug("Could not read structured entity properties", exc_info=True)
            return
        for row in result.result_set or []:
            props = row[1] or {}
            by_table: dict[str, list[str]] = {}
            for key, value in props.items():
                table, sep, prop = key.partition("__")
                if not sep or not prop or value in (None, "", []):
                    continue
                by_table.setdefault(table, []).append(f"{prop} {value}")
            if by_table:
                by_id[row[0]]["description"] = "; ".join(
                    f"{table}: {', '.join(facts)}" for table, facts in sorted(by_table.items())
                )

    async def _fetch_relates_edges(self, batch_size: int) -> list[GraphRelationship]:
        """Every RELATES edge between entities, as the resolver's neighbourhood."""
        offset = 0
        edges: list[GraphRelationship] = []
        for _ in range(_MAX_PAGINATION_ITERATIONS):
            result = await self._graph.query_raw(
                "MATCH (a:__Entity__)-[r:RELATES]->(b:__Entity__) "
                "RETURN a.id, b.id, coalesce(r.rel_type, 'RELATES') "
                "SKIP $offset LIMIT $limit",
                {"offset": offset, "limit": batch_size},
            )
            if not result.result_set:
                break
            for start, end, rel_type in result.result_set:
                edges.append(GraphRelationship(start_node_id=start, end_node_id=end, type=rel_type))
            offset += batch_size
        else:
            logger.error(
                "Pagination exceeded %d iterations in _fetch_relates_edges — aborting",
                _MAX_PAGINATION_ITERATIONS,
            )
        return edges

    async def _fetch_distinct_pairs(self) -> set[frozenset[str]]:
        """Pairs a resolver judged to be two things on an earlier run."""
        try:
            result = await self._graph.query_raw(
                f"MATCH (a:__Entity__)-[:{DISTINCT_FROM}]-(b:__Entity__) RETURN a.id, b.id"
            )
        except Exception:
            logger.debug("Could not read %s edges", DISTINCT_FROM, exc_info=True)
            return set()
        return {frozenset((a, b)) for a, b in result.result_set or [] if a != b}

    async def _remember_distinct(
        self, pairs: set[frozenset[str]], by_id: dict[str, dict[str, Any]], decided_by: str
    ) -> None:
        """Write the resolver's NO answers to the graph, so they are not asked again.

        An edge rather than a property: it goes when either node goes, which is
        exactly when the answer stops applying — a document deleted, or re-read
        so its entity is extracted afresh — and it is never copied onto a
        survivor by a merge, where it would name a node that no longer exists.
        """
        rows = []
        for pair in pairs:
            ids = sorted(pair)
            if len(ids) == 2 and all(node_id in by_id for node_id in ids):
                rows.append({"a": ids[0], "b": ids[1]})
                label = by_id[ids[0]].get("label") or ""
                self.rejected_pairs.append(
                    f"{label} {by_id[ids[0]]['name']!r} | {by_id[ids[1]]['name']!r}"
                )
        if not rows:
            return
        try:
            await self._graph.query_raw(
                "UNWIND $rows AS row "
                "MATCH (a:__Entity__ {id: row.a}), (b:__Entity__ {id: row.b}) "
                f"MERGE (a)-[d:{DISTINCT_FROM}]->(b) SET d.decided_by = $decided_by",
                {"rows": rows, "decided_by": decided_by},
            )
        except Exception:
            logger.warning(
                "Could not record %d %s pair(s)", len(rows), DISTINCT_FROM, exc_info=True
            )
            self.rejected_pairs = []
            return
        logger.info(
            "Resolver %s judged %d pair(s) distinct; remembered so they are not asked again",
            decided_by,
            len(rows),
        )

    # ── Helpers ──

    async def _adopt_into_declared_labels(
        self, groups: dict[tuple[str, str], list[dict[str, Any]]]
    ) -> int:
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

        The guess can only have happened for a label the extractor did not have.
        Every built-in type — Person, Organization, Product, Location — is always
        on the extractor's list, so a name it filed under ``Product`` while a
        mapping declared it ``Organization`` is not a fallback: it had
        ``Organization`` available and chose otherwise. That is Apple the fruit
        beside Apple the company, and adopting it deleted the fruit. So adoption
        is restricted to declared labels **outside** the built-in set — the only
        labels an extractor could have been missing.

        Left alone, and reported instead:

        - two declared labels sharing a name, which is a real modelling conflict
          rather than a guess to correct
        - a declared built-in label beside any other label: two things that
          share a name, one of which the extractor deliberately typed differently
        - names under only undeclared labels

        Residual: a *custom* declared label beside a built-in one for a genuinely
        different thing ("Supplier" Apple beside "Product" Apple) is still adopted.
        Distinguishing that from the ordering mistake would need to know whether
        the label existed at extraction time, which is not recorded.

        Measured on the same files, prose first: 0 merged before, 5 after.
        """
        if not self._declared_labels:
            return 0
        adoptable = self._declared_labels - _BUILTIN_LABELS
        if not adoptable:
            return 0

        by_name: dict[str, list[str]] = {}
        for norm_name, label in groups:
            by_name.setdefault(norm_name, []).append(label)

        merged = 0
        for norm_name, labels in by_name.items():
            if len(labels) < 2:
                continue
            declared = [label for label in labels if label in adoptable]
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
                    if await self._absorb(survivor, duplicate):
                        merged += 1
                        logger.info(
                            "Adopted %r from inferred label %r into declared label %r",
                            survivor["name"],
                            label,
                            declared[0],
                        )
                # Consumed, so the leftover report does not name it.
                groups.pop((norm_name, label), None)
        return merged

    def _report_cross_label_names(
        self, groups: dict[tuple[str, str], list[dict[str, Any]]]
    ) -> None:
        """Say when two entities share a name but not a label.

        Matching on name *and* label is what stops "Apple" the company merging
        with "Apple" the fruit, so this is never merged automatically. But it is
        also the exact fingerprint of an ordering mistake: a document read before
        a mapping was declared has its entities labelled with a built-in guess,
        and the same name arriving later from a table under a declared label can
        no longer join it. That case is silent otherwise — the caller sees
        ``entities_deduplicated=0`` and nothing else.

        Reporting it costs one pass over grouping that already happened.

        Reported under a name the user can actually search for. The grouping key
        is canonical — lower-cased, suffixes dropped — so keying the report on it
        would tell someone whose file says "Acme Corp" to go looking for
        "acme corporation", a string that appears nowhere in their data.
        """
        labels_by_name: dict[str, set[str]] = {}
        display_name: dict[str, str] = {}
        for (group_key, label), members in groups.items():
            labels_by_name.setdefault(group_key, set()).add(label)
            for member in members:
                name = member.get("name")
                if not name:
                    continue
                # Longest spelling, then lexicographic. Group iteration order
                # follows the unordered fetch, so picking the first would show a
                # different spelling from one run to the next.
                current = display_name.get(group_key)
                if current is None or (-len(name), name) < (-len(current), current):
                    display_name[group_key] = name
        collisions = {
            display_name.get(group_key, group_key): sorted(labels)
            for group_key, labels in labels_by_name.items()
            if len(labels) > 1
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
            "Put the mapping in the ontology you pass to GraphRAG("
            "ontology=Ontology(tables=[TableMapping(...)])) so the label is "
            "declared before anything is extracted. Examples: %s",
            len(collisions),
            sample,
        )

    async def _fetch_all_entities(self, batch_size: int) -> list[dict[str, Any]]:
        """Fetch all entities in batches, including their primary label."""
        offset = 0
        entities: list[dict[str, Any]] = []
        for _ in range(_MAX_PAGINATION_ITERATIONS):
            result = await self._graph.query_raw(
                "MATCH (e:__Entity__) "
                "RETURN e.id AS id, e.name AS name, e.description AS desc, "
                "HEAD([l IN labels(e) WHERE l <> '__Entity__']) AS label, "
                f"e.aliases AS aliases, e.is_stub AS is_stub, {_DEGREE_EXPR} AS degree "
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
                        # Only a table write sets is_stub (False for a row, True
                        # for a placeholder), so its presence marks a keyed node.
                        "is_stub": row[5] if len(row) > 5 else None,
                        "degree": row[6] if len(row) > 6 else 0,
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
