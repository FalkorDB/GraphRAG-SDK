# GraphRAG SDK — Storage: Entity Deduplicator
# Incremental, crash-safe entity deduplication. A run checks only the entities
# whose check is pending — no name vector yet, or listed in a ``pending``
# system record (``system_store``) — against the whole graph reached through
# the entity vector index, never by reading every entity. Nothing is written
# on an entity to mark it: ingest writes entities without a vector, and a run
# lists them before it gives them one. Phases: exact name match and the
# LLM-judged cross-document phase (``judge_dedup.LLMJudgeDeduplicator``: the
# vector index nominates, the LLM decides). Every merge of the run is decided
# first, written as a sealed plan, and applied one atomic query per merge
# (``merge_query``), so a crash at any point resumes where it stopped.
# Preserves label-aware grouping to prevent cross-type merging.

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from graphrag_sdk.core.exceptions import DatabaseError
from graphrag_sdk.core.models import RESERVED_NODE_LABELS
from graphrag_sdk.core.providers import Embedder, LLMInterface
from graphrag_sdk.core.tables import RESERVED_PROPERTY_NAMES
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import DEFAULT_ENTITY_TYPES
from graphrag_sdk.ingestion.resolution_strategies.base import description_list
from graphrag_sdk.storage.identity import NearMiss, canonical_key, find_near_misses
from graphrag_sdk.storage.judge_dedup import (
    NEIGHBOUR_GATE,
    NEIGHBOUR_K,
    AttributeSchema,
    Hit,
    LLMJudgeDeduplicator,
    NameNeighbourIndex,
    UnionFind,
    Verdicts,
    pick_group_name,
)
from graphrag_sdk.storage.merge_query import (
    APPLIED,
    CHANGED,
    SKIP_QUERY,
    STALE,
    MergeRead,
    absorb_query,
    handled,
    outcome,
    read_for_merge,
    rename_query,
    snapshot_params,
)
from graphrag_sdk.storage.system_store import (
    ABSORB,
    ADOPT,
    EXACT,
    HEARTBEAT_INTERVAL_S,
    JUDGE,
    MAX_JUDGED_ATTEMPTS,
    RENAME,
    Decisions,
    OwnershipLostError,
    PendingPage,
    Plan,
    PlanOp,
    SystemStore,
    Unresolved,
)
from graphrag_sdk.utils.cypher import sanitize_cypher_label

#: Labels every extractor always has. A declared label in this set can never have
#: been a guess the extractor was missing, so it is never a target for adoption.
_BUILTIN_LABELS: frozenset[str] = frozenset(t.strip().lower() for t in DEFAULT_ENTITY_TYPES)

logger = logging.getLogger(__name__)

# Pagination safety net — well above any realistic graph size at typical
# batch sizes (10_000 * 500 = 5M entities). Trips only on a pathological
# server bug that keeps returning the same page.
_MAX_PAGINATION_ITERATIONS = 10_000

# Who holds a dedup lock, for whoever finds it held.
_OWNER = f"{platform.node()}:{os.getpid()}"
# FalkorDB before 4.20.0 lost vector-index entries on writes to embedded nodes.
_MIN_FALKORDB = (4, 20, 0)
# Name vectors per write, and reads of a merge before it is recorded changed.
_VECTOR_BATCH = 500
_STALE_READS = 3

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

# Labels the graph gives its own structure; never stamped onto an entity by a
# merge, where they would surface in every ``MATCH (:Document)`` query.
_STRUCTURAL_LABELS = RESERVED_NODE_LABELS | {"__Entity__"}

# Absorbed labels are recorded here as well as set as Cypher labels, in the
# ``" | "``-joined form ingest-time resolution and ``GraphStore`` use, so a
# survivor's later merge carries them on and its primary label stays fixed.
MERGED_LABELS_KEY = "merged_labels"


def label_for_merge(label: Any) -> str | None:
    """The label as it may be set on a survivor, or ``None`` if it may not be.

    The contract is the repository's, not a stricter one of this module's:
    :func:`sanitize_cypher_label` decides what the write path can quote, so
    ``Legal Entity``, ``Org-Unit`` and ``Ünïcode`` are as valid here as they are
    in a mapping. What is refused is a label the sanitiser would have to
    *change* (a stray backtick — the graph would then hold a name nobody
    declared), a structural label, and one containing ``|`` — the character
    that separates labels in the ``merged_labels`` record, so ``A|B`` written
    there would read back as two labels (see :func:`split_merged_labels`).
    """
    if not isinstance(label, str):
        return None
    try:
        safe = sanitize_cypher_label(label)
    except ValueError:
        return None
    if safe != label.strip() or safe in _STRUCTURAL_LABELS or "|" in safe:
        return None
    return safe


def split_merged_labels(value: Any) -> list[str]:
    """The labels recorded in a ``merged_labels`` property, in order.

    Split on the exact ``" | "`` every writer joins with, not on a bare ``|``:
    a label such as ``A|B`` recorded by an older run must come back as the one
    label it was, not as ``A`` and ``B``, which would then be filtered out of
    the primary-label choice as if they had been absorbed.
    """
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(" | ") if part.strip()]


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


def _merge_description_members(current: list[str], absorbed: list[str]) -> list[str]:
    """The merged member descriptions of two nodes, each kept once, the
    survivor's first.

    Members are compared individually, not whole strings: once a survivor
    holds ``["a lighthouse", "first lit in 1871"]``, absorbing a node described
    as ``"first lit in 1871"`` must not append it again. Across incremental
    finalize cycles that repetition compounds on hub entities.
    """
    segments: list[str] = []
    for seg in (str(s or "").strip() for s in (*current, *absorbed)):
        if seg and seg not in segments:
            segments.append(seg)
    return segments


def _description_members(props: dict[str, Any]) -> list[str]:
    """A node's member descriptions, by the rule every merge path shares.

    :func:`description_list`: the stored ``descriptions`` list where the node
    has one; otherwise its ``description`` as **one** member, deliberately not
    split on ``" | "`` — a fresh node's single description may contain the
    separator (``"CEO | founder of Acme"``), and splitting it would invent
    members that were never written and count them toward
    ``force_summary_threshold``. The one addition: an empty stored list does
    not hide a description the node does hold — the text is still a member,
    because a merge that misread it would delete the only other copy.
    """
    members = description_list(props)
    if members:
        return members
    single = str(props.get("description") or "").strip()
    return [single] if single else []


def _merge_description_segments(current: str, absorbed: str) -> list[str]:
    """:func:`_merge_description_members` over two bare description strings,
    each split on ``" | "``.

    For the callers that hold only the joined text (:func:`properties_to_carry`
    reads whatever properties a node has, and a reconcile writes ``description``
    alone); the merge that also writes the ``descriptions`` list goes through
    :func:`_description_members` so it never invents a member.
    """
    return _merge_description_members(
        str(current or "").split(" | "), str(absorbed or "").split(" | ")
    )


def _merge_description(current: str, absorbed: str) -> str:
    """:func:`_merge_description_segments`, joined with ``" | "``."""
    return " | ".join(_merge_description_segments(current, absorbed))


# Cypher queries for remapping edges from a duplicate to a survivor entity, used
# by ``GraphStore`` when a re-synced table row folds into the node already at
# its new id. A dedup merge moves every relationship type in one query instead
# (``merge_query``).
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
    with each row it could be, for the user to decide.

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
# ones a question can reach it by.
_DEGREE_EXPR = "size((e)-[:RELATES]-()) + size((e)-[:MENTIONED_IN]->())"

#: Nearest names (vector index) each new entity's exact-name match is looked
#: for among: identical and canonical spellings embed as each other's nearest
#: neighbours, so no lookup key has to be stored on the graph.
_EXACT_NEIGHBOUR_K = 10
#: How far that lookup widens for a new entity whose nearest names are all
#: exact-phase matches (twenty rows called "John Smith"): the k-th neighbour
#: being a match says the next one may be too, so k grows tenfold up to this.
_EXACT_NEIGHBOUR_K_MAX = 1000

# The projection every entity read uses, applied to a bound ``e``.
_ENTITY_RETURN = (
    "RETURN e.id AS id, e.name AS name, e.description AS desc, "
    "HEAD([l IN labels(e) WHERE l <> '__Entity__']) AS label, "
    f"e.aliases AS aliases, e.is_stub AS is_stub, {_DEGREE_EXPR} AS degree, "
    "[l IN labels(e) WHERE l <> '__Entity__'] AS labels, "
    f"e.{MERGED_LABELS_KEY} AS merged_labels, e.embedding IS NOT NULL AS has_vector"
)


def _entity_from_row(row: list[Any]) -> dict[str, Any]:
    """One ``_ENTITY_RETURN`` row as the dict every phase works on."""
    aliases = row[4] if len(row) > 4 and isinstance(row[4], list) else []
    labels = _string_list(row[7]) if len(row) > 7 else []
    absorbed = split_merged_labels(row[8]) if len(row) > 8 else []
    label = _primary_label(labels, absorbed) or (row[3] if len(row) > 3 and row[3] else "")
    return {
        "id": row[0],
        "name": row[1] if len(row) > 1 and row[1] else str(row[0]),
        "has_name": bool(len(row) > 1 and row[1] and str(row[1]).strip()),
        "has_vector": bool(row[9]) if len(row) > 9 else True,
        "description": row[2] if len(row) > 2 and row[2] else "",
        "label": label,
        "labels": labels or ([label] if label else []),
        "aliases": [a for a in aliases if isinstance(a, str)],
        # Only a table write sets is_stub (False for a row, True
        # for a placeholder), so its presence marks a keyed node.
        "is_stub": row[5] if len(row) > 5 else None,
        "degree": row[6] if len(row) > 6 else 0,
    }


@dataclass
class DedupReport:
    """What the last run checked and what it left for a later one."""

    #: Pending entities whose check completed this run.
    checked: int = 0
    #: Entities still pending afterwards: a failed or held check, or a run
    #: without the judge.
    pending: int = 0
    #: Entities no longer retried after ``MAX_JUDGED_ATTEMPTS`` judged runs.
    unresolved: dict[str, Unresolved] = field(default_factory=dict)
    #: Planned merges skipped because an entity changed during the run.
    changed: int = 0
    #: Merges applied from an interrupted run's sealed plan.
    resumed: int = 0
    #: Every value a merge kept out of a property because the other node's
    #: won: ``"Tim Cook: born kept 1960 (judge), dropped 1961 from Timothy Cook"``.
    #: The dropped value is on the survivor, in ``<property>__dropped``.
    collisions: list[str] = field(default_factory=list)


@dataclass
class _Run:
    """What one run has decided so far. Nothing in it is written until the
    plan is sealed; a crash before that loses only judging work."""

    judged: bool
    unresolved: dict[str, Unresolved] = field(default_factory=dict)
    #: Ids in the pending records when the run loaded them.
    listed: set[str] = field(default_factory=set)
    #: This run's query points (``_entity_from_row`` rows).
    points: list[dict[str, Any]] = field(default_factory=list)
    #: Listed ids no longer in the graph.
    gone: set[str] = field(default_factory=set)
    #: Points with no completed nearest-name lookup, and why.
    unsearched: dict[str, str] = field(default_factory=dict)
    hits: dict[str, list[Hit]] = field(default_factory=dict)
    #: Rows the merge decisions are ranked and named from, by id.
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Decided absorbs, ``dup id -> (survivor id, policy)``, in decision order.
    into: dict[str, tuple[str, str]] = field(default_factory=dict)
    renames: dict[str, str] = field(default_factory=dict)
    #: Judged survivor id -> the judge's KEEP picks for its group,
    #: ``{node property: value}``.
    keeps: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Listed ids the claim charged one judged attempt.
    charged: set[str] = field(default_factory=set)
    #: Ids in a set that got no usable verdict; ids in a group such a set
    #: could still change.
    failed: set[str] = field(default_factory=set)
    held: set[str] = field(default_factory=set)
    #: Unresolved id -> names of the entities it was kept from being judged with.
    blocked: dict[str, set[str]] = field(default_factory=dict)

    def resolve(self, eid: str) -> str:
        """The entity ``eid`` ends up in once the decided absorbs are applied."""
        while eid in self.into:
            eid = self.into[eid][0]
        return eid

    def absorb(self, dup: dict[str, Any], survivor: dict[str, Any], policy: str) -> None:
        self.into[dup["id"]] = (survivor["id"], policy)
        self.rows.setdefault(dup["id"], dup)
        self.rows.setdefault(survivor["id"], survivor)
        _fold_for_ranking(survivor, dup)


def _fold_for_ranking(survivor: dict[str, Any], dup: dict[str, Any]) -> None:
    """What a planned absorb will give the survivor, as far as a later
    decision of the same run ranks on it: the connections, the description,
    the aliases and a declared key. The graph is written only by the plan."""
    survivor["degree"] = int(survivor.get("degree") or 0) + int(dup.get("degree") or 0)
    survivor["description"] = _merge_description(
        str(survivor.get("description") or ""), str(dup.get("description") or "")
    )
    survivor["aliases"] = EntityDeduplicator._merged_aliases(survivor, dup)
    if survivor.get("is_stub") is None and dup.get("is_stub") is not None:
        survivor["is_stub"] = dup["is_stub"]


async def _require_falkordb(graph: Any) -> None:
    """Refuse FalkorDB before 4.20: it degraded the vector index on every
    write to a node holding a vector (fixed by FalkorDB PR #2197), and a dedup
    run writes vectors and merges into vector-indexed nodes. Measured on
    4.18: ``queryNodes`` returned 2-3 of 6 neighbours after one such write."""
    probe = getattr(graph, "server_version", None)
    if probe is None:
        return
    version = await probe()
    if version is None:
        logger.warning("Could not read the FalkorDB version; entity dedup needs 4.20 or later")
    elif version < _MIN_FALKORDB:
        found = ".".join(str(v) for v in version)
        raise DatabaseError(
            f"Entity deduplication needs FalkorDB 4.20 or later; this server is {found}. "
            "Earlier versions lose vector-index entries on every write to an embedded "
            "node, and dedup writes to them on every run."
        )


class EntityDeduplicator:
    """Incremental, crash-safe deduplication of the entities whose check is
    pending, against the rest of the graph reached through the entity vector
    index.

    **Pending** means: no name vector, or listed in a ``pending`` record (see
    ``system_store``). Ingest writes entities without a vector, so a new
    entity is pending without anything written on it. A run:

    0. takes the dedup lock and finishes any sealed plan an interrupted run
       left;
    1. lists every entity without a vector in a pending record, *then* writes
       its vector — never the other way round, so no entity looks checked
       before it is;
    2. reads each pending entity's nearest names from the vector index;
    3. decides every merge before writing any. Exact phase: entities grouped by
       ``(canonical_name, label)`` (an acronym joins its unique same-label long
       form), the survivor :func:`_survivor_rank` picks (a table row over a
       mention, the long form over an acronym, then the best-connected and
       best-described); with a judge, a group of extracted mentions only is put
       to it instead. Then the LLM-judged phase (``judge_llm``): candidate sets
       from the nearest names, partitioned by the judge, unioned across sets
       but never through a pair it — or a declared key — kept apart. A group a
       failed set could still change is held, not merged;
    4. writes all of it as a merge plan, seals it, and applies it: one atomic
       query per merge (``merge_query``), each ticking itself off in the plan.
       A crash resumes from the ticks; nothing is merged twice or half;
    5. clears the entities whose check completed, and releases the lock.

    Every merge preserves what the duplicate carried: its description is
    joined onto the survivor's with ``" | "`` (and kept as a member of the
    ``descriptions`` list), its name is recorded in the survivor's ``aliases``
    when it differs, its node-level ``source_chunk_ids`` are unioned into the
    survivor's, every relationship of every type moves to the survivor, and any
    property the survivor lacks is copied over. A judged merge also gives the
    survivor the duplicate's labels and may give it the judge's best name.

    Args:
        graph_store: Graph data access object with ``query_raw()`` method.
        embedder: Embedding provider for entity names.
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
        # Stats of the last LLM-judged phase; {"skipped_reason": ...} when the
        # phase was asked for and did not run.
        self.last_judge_stats: dict[str, int | str] = {}
        self.last_report = DedupReport()
        # This run's query points, and every entity it looked at (them plus
        # their nearest names).
        self._pending_ids: list[str] = []
        self._run_entity_ids: list[str] = []
        # Every nearest name the index returned for them, whether or not the
        # exact phase could group it: the near-miss report reads these too.
        self._near_ids: list[str] = []
        # Check every entity, not only the pending ones (see deduplicate()).
        self._dedup_all = False
        # Same-name pairs the exact phase handed to the judge instead of merging.
        self._judge_pairs: list[tuple[str, str]] = []
        # Name vectors this run wrote, on nodes still in the graph at the end.
        self.names_embedded = 0
        self._name_embedded_ids: set[str] = set()
        # The ontology's types and attributes, as the judge sees them.
        self._schema = AttributeSchema()

    async def deduplicate(
        self,
        *,
        batch_size: int = 500,
        declared_labels: set[str] | None = None,
        judge_llm: LLMInterface | None = None,
        dedup_all: bool = False,
        release_unjudged: bool = False,
        ontology: Any = None,
    ) -> int:
        """Run deduplication and return the number of duplicates merged.

        ``ontology``, when given, is the schema the judge sees: each entity
        shows the values of the attributes its types declare, under a header
        listing those types' attributes, and the judge may choose, per
        attribute its group's members disagree on, whose value the merged
        entity keeps (``KEEP``). See :meth:`_fold`.

        ``declared_labels`` are labels a structured mapping declared. They are
        treated as authoritative about type, which lets the one safe kind of
        cross-label merge happen: see :meth:`_plan_adoption`.

        ``judge_llm``, when given, runs the LLM-judged cross-document phase
        after the exact phase (see class docstring). Stats of the last run are
        in ``self.last_judge_stats``, and what it left pending in
        ``self.last_report``. Without it, only spelling variants merge and the
        pending entities stay pending — the judge still owes them a check —
        unless ``release_unjudged=True`` marks them checked.

        A run with nothing pending does no work. ``dedup_all=True`` checks
        every entity instead, and retries the unresolved ones.

        Raises:
            DedupRunActiveError: another live run holds this graph's dedup lock.
            DatabaseError: FalkorDB is older than 4.20, or a write failed. A
                failure after the plan was sealed leaves the plan for the next
                run, which finishes it before anything else.
        """
        self._declared_labels = {label.strip().lower() for label in (declared_labels or set())}
        self._ambiguous_mentions = []
        self.near_misses = []
        self.cross_label_names = {}
        self.last_judge_stats = {}
        self.last_report = DedupReport()
        self._pending_ids = []
        self._run_entity_ids = []
        self._near_ids = []
        self._dedup_all = dedup_all
        self._judge_pairs = []
        self.names_embedded = 0
        self._name_embedded_ids = set()
        self._schema = AttributeSchema.from_ontology(ontology)
        await _require_falkordb(self._graph)

        store = SystemStore(self._graph)
        await store.ensure_indexes()
        if await store.acquire(owner=_OWNER):
            logger.warning(
                "Took over the dedup lock of a run that stopped without releasing it; "
                "its plan, if it sealed one, is finished first"
            )
        beat = asyncio.create_task(self._keep_alive(store))
        merged = 0
        try:
            merged = self.last_report.resumed = await self._recover(store)
            run = await self._start(store, judged=judge_llm is not None, batch_size=batch_size)
            await self._exact_phase(run, batch_size, defer_to_judge=judge_llm is not None)
            if judge_llm is not None:
                await self._judge_phase(store, run, judge_llm, batch_size)
            ops, decisions = self._plan(run, release_unjudged=release_unjudged)
            if ops or _any_decision(decisions):
                await store.set_blocked_with(
                    {eid: " | ".join(sorted(names)) for eid, names in run.blocked.items()}
                )
                await store.write_plan(ops, decisions)
                plan = await store.sealed_plan()
                if plan is not None:
                    merged += await self._finish(store, plan)
            await self._summarise(store)
        finally:
            beat.cancel()
            await asyncio.gather(beat, return_exceptions=True)
            await self._let_go(store)

        # What is left over that probably should not be. Reported, never merged.
        await self._report_near_misses(batch_size)
        await self._count_names_embedded()
        logger.info(f"EntityDeduplicator total: {merged} duplicates merged")
        return merged

    # ── the lock ──

    @staticmethod
    async def _keep_alive(store: SystemStore) -> None:
        """Refresh the lock's heartbeat while the run works; a judge round
        can take minutes, and a silent lock looks abandoned after
        ``LOCK_TTL_MS``. Stops once ownership is lost: every write checks it
        anyway, so the run cannot change anything after that."""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)
            try:
                await store.heartbeat()
            except OwnershipLostError:
                logger.warning(
                    "This dedup run lost its lock to another run; it writes nothing more"
                )
                return
            except Exception:
                logger.debug("Dedup heartbeat failed; retrying", exc_info=True)

    @staticmethod
    async def _let_go(store: SystemStore) -> None:
        """Release the lock — unless a sealed plan is still unfinished, after
        an error. Then the run record stays, expired: delete calls keep leaving
        ``deleted`` records for the plan to check, and the next run takes the
        lock at once and finishes the plan."""
        if store.run_id is None:
            return
        try:
            unfinished = await store.sealed_plan() is not None
        except Exception:
            logger.warning("Could not check for an unfinished merge plan", exc_info=True)
            unfinished = True
        if unfinished:
            await store.expire()
        else:
            await store.release()

    # ── 0. recovery ──

    async def _recover(self, store: SystemStore) -> int:
        """Finish the sealed plan an interrupted run left, from its ticks; drop
        an unsealed one (nothing in it was applied) and every ``deleted``
        record, which only the plans they were left for read."""
        merged = 0
        plan = await store.sealed_plan()
        if plan is not None:
            logger.warning(
                "Resuming the merge plan of an interrupted dedup run: %d operation(s) left",
                sum(len(ops) - nxt for ops, nxt in zip(plan.ops, plan.nexts)),
            )
            merged = await self._finish(store, plan)
        else:
            await store.drop_plans()
        await store.clear_deleted()
        return merged

    async def _finish(self, store: SystemStore, plan: Plan) -> int:
        merged = await self._apply(store, plan)
        # Cleared from the plan as the apply left it: the operations it found
        # changed decide which groups are checked again.
        await self._clear(store, await store.sealed_plan() or plan)
        await store.drop_plans()
        await store.clear_deleted()
        return merged

    # ── 1. publish and load ──

    async def _start(self, store: SystemStore, *, judged: bool, batch_size: int) -> _Run:
        run = _Run(judged=judged)
        run.unresolved = await store.unresolved()
        if self._dedup_all and run.unresolved:
            # Listed again first: a crash in between leaves them pending, which
            # is checked; the other order could leave them in neither.
            await store.add_pending(list(run.unresolved))
            await store.drop_unresolved(run.unresolved)
            run.unresolved = {}
        elif not self._dedup_all:
            await self._give_up(store, run)
        await self._publish(store, run)

        pages = await store.pending_pages()
        run.listed = {eid for page in pages for eid in page.ids} - run.unresolved.keys()
        if self._dedup_all:
            entities = await self._fetch_all_entities(batch_size)
        else:
            found = await self._fetch_entities_by_ids(sorted(run.listed), batch_size)
            entities = list(found.values())
            run.gone = run.listed - found.keys()
        run.points = [ent for ent in entities if ent["id"] not in run.unresolved]
        for ent in run.points:
            run.rows[ent["id"]] = ent
            if not ent["has_vector"]:
                run.unsearched[ent["id"]] = "no name vector"
        if len(run.unsearched) < len(run.points):
            # The index may still be building over vectors written before this
            # run (a fresh index on a filled graph); a search would fail.
            await NameNeighbourIndex(self._graph, self._embedder).ensure_vector_index(None)
        return run

    async def _give_up(self, store: SystemStore, run: _Run) -> None:
        """Move entities with ``MAX_JUDGED_ATTEMPTS`` judged runs and no
        usable verdict to the unresolved records: reported, left out of every
        set, never merged automatically."""
        moved: dict[str, Unresolved] = {}
        rewritten: list[PendingPage] = []
        for page in await store.pending_pages():
            keep = [
                i
                for i, eid in enumerate(page.ids)
                if eid not in run.unresolved and page.attempts[i] < MAX_JUDGED_ATTEMPTS
            ]
            if len(keep) == len(page.ids):
                continue
            for i, eid in enumerate(page.ids):
                if i not in keep and eid not in run.unresolved:
                    moved[eid] = Unresolved(
                        page.attempts[i], page.reasons[i] or "no usable verdict"
                    )
            rewritten.append(
                PendingPage(
                    page.page,
                    [page.ids[i] for i in keep],
                    [page.attempts[i] for i in keep],
                    [page.reasons[i] for i in keep],
                    page.claimed_run,
                )
            )
        if moved:
            await store.add_unresolved(moved)
            run.unresolved.update(moved)
            logger.warning(
                "%d entities got no usable dedup verdict in %d judged runs and are no longer "
                "retried (finalize(dedup_all=True) retries them): %s",
                len(moved),
                MAX_JUDGED_ATTEMPTS,
                ", ".join(sorted(moved)[:10]),
            )
        await store.write_pending(rewritten)

    async def _publish(self, store: SystemStore, run: _Run) -> None:
        """Give every entity without a name vector one, listing it as pending
        first.

        The record is written before the vector: an entity with a vector and
        no record would look checked, so a crash in between must leave a
        record without a vector (embedded under it next run), never the
        reverse. A nameless entity — a fact row keyed on a reading id — gets
        the vector of its id, as retrieval embeds it, and is never listed: the
        judge cannot judge a name it does not have.
        """
        index = NameNeighbourIndex(self._graph, self._embedder)
        listed = {eid for page in await store.pending_pages() for eid in page.ids}
        index_ready = False
        async for window in index.unembedded():
            fresh = [
                eid
                for eid, name in window
                if name and eid not in listed and eid not in run.unresolved
            ]
            await store.add_pending(fresh)
            listed.update(fresh)
            for k in range(0, len(window), _VECTOR_BATCH):
                part = window[k : k + _VECTOR_BATCH]
                try:
                    vectors = await self._embedder.aembed_documents(
                        [str(name or eid) for eid, name in part]
                    )
                except Exception as exc:
                    # Listed without a vector: embedded and checked next run.
                    logger.warning("Could not embed %d entity names: %s", len(part), exc)
                    continue
                items = [
                    {"id": eid, "v": list(vector)}
                    for (eid, _), vector in zip(part, vectors)
                    if vector is not None and len(vector)
                ]
                if not items:
                    continue
                if not index_ready:
                    # Before the first vector: written into a ready index, a
                    # vector is searchable at once; an index built over vectors
                    # already written takes minutes to answer.
                    await index.ensure_vector_index(len(items[0]["v"]))
                    index_ready = True
                await self._write_vectors(store, items)

    async def _write_vectors(self, store: SystemStore, items: list[dict[str, Any]]) -> None:
        rows = await store.fenced(
            "UNWIND $items AS it "
            "OPTIONAL MATCH (e:__Entity__ {id: it.id}) WHERE owned AND e.embedding IS NULL "
            "FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END | "
            "SET e.embedding = vecf32(it.v)) "
            "RETURN owned, collect(DISTINCT e.id)",
            {"items": items},
        )
        self._name_embedded_ids.update(rows[0][1] or [])

    # ── 2-3. the exact phase ──

    async def _exact_phase(self, run: _Run, batch_size: int, *, defer_to_judge: bool) -> None:
        """Decide the exact-name merges: the pending entities, plus their
        nearest names already in the graph (vector index), which is where an
        existing spelling of the same name is. Two old entities were folded
        (or left apart) by an earlier run."""
        pending = run.points
        self._pending_ids = [ent["id"] for ent in pending]
        if not pending:
            logger.info("EntityDeduplicator: no entities pending a dedup check")
            return
        entities = list(pending)
        if not self._dedup_all:
            entities += await self._fetch_name_neighbours(run, batch_size)
        for ent in entities:
            run.rows.setdefault(ent["id"], ent)
        self._run_entity_ids = [ent["id"] for ent in entities]
        if len(entities) < 2:
            return

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
        ambiguous_acronyms: list[tuple[list[dict], list[dict]]] = []
        merged_groups = self._merge_acronym_groups(
            list(groups.values()), ambiguous_out=ambiguous_acronyms
        )
        if defer_to_judge:
            # An acronym that spells several long forms is not folded on its
            # own; the judge, which sees descriptions, can still decide it.
            for short, long_ in ambiguous_acronyms:
                self._judge_pairs.extend(
                    (a["id"], b["id"])
                    for a in short
                    for b in long_
                    if a.get("is_stub") is None and b.get("is_stub") is None
                )

        decided = 0
        for group in merged_groups:
            if len(group) < 2:
                continue
            if defer_to_judge and all(ent.get("is_stub") is None for ent in group):
                # Extracted mentions only: the same spelling is evidence, not
                # proof. The judge sees their descriptions and decides; a table
                # row (keyed) still absorbs its mentions here by the rules below.
                ids = [ent["id"] for ent in group]
                self._judge_pairs.extend((a, b) for i, a in enumerate(ids) for b in ids[i + 1 :])
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
                run.absorb(dup, survivor, EXACT)
                decided += 1

        # What follows works from the groups too, and a node already decided
        # above would otherwise be "adopted" a second time.
        for group in groups.values():
            group[:] = [ent for ent in group if ent["id"] not in run.into]

        decided += self._plan_adoption(run, groups)
        logger.info(f"EntityDeduplicator phase 1 (exact): {decided} merges decided")
        self._report_cross_label_names(groups)

    async def _fetch_name_neighbours(self, run: _Run, batch_size: int) -> list[dict[str, Any]]:
        """Entities already in the graph nearest by name to the pending ones,
        read through the ``__Entity__.embedding`` vector index from the vectors
        stored on them. Only a neighbour the exact phase could group with a
        pending entity is read in full — the same canonical name, or an acronym
        / long-form pair; the rest are the judge's business. A failed lookup
        leaves its entities pending for the next run."""
        pending = run.points
        keys: set[str] = set()
        acronyms: set[str] = set()
        initials: set[str] = set()
        for ent in pending:
            norm = normalize_entity_name(ent["name"])
            keys.add(canonical_key(ent["name"]) or ent["name"].strip().lower())
            if (acr := _acronym_key(norm)) is not None:
                acronyms.add(acr)
            if (exp := _expansion_key(norm)) is not None:
                initials.add(exp)

        def could_group(name: Any) -> bool:
            if not isinstance(name, str) or not name:
                return False
            norm = normalize_entity_name(name)
            return (
                (canonical_key(name) or name.strip().lower()) in keys
                or _acronym_key(norm) in initials
                or _expansion_key(norm) in acronyms
            )

        index = NameNeighbourIndex(self._graph, self._embedder)
        ids = [ent["id"] for ent in pending if ent["id"] not in run.unsearched]
        k = _EXACT_NEIGHBOUR_K
        hits, failed = await index.discover(ids, k)
        run.unsearched.update(dict.fromkeys(failed, "nearest-name lookup failed"))
        # A pending entity whose k nearest names are all matches may have more
        # beyond them (twenty rows called "John Smith"): ask again, wider, for
        # those entities only.
        while k < _EXACT_NEIGHBOUR_K_MAX:
            full = [
                eid
                for eid, found in hits.items()
                if len(found) >= k and all(could_group(name) for _, _, name in found)
            ]
            if not full:
                break
            k = min(k * 10, _EXACT_NEIGHBOUR_K_MAX)
            wider, failed = await index.discover(full, k)
            hits.update(wider)
            if failed:
                logger.warning("Exact phase: a widened lookup failed for %d entities", len(failed))
        run.hits = hits
        pending_ids = {ent["id"] for ent in pending}
        near = [nid for found in hits.values() for nid, _, _ in found if nid not in pending_ids]
        self._near_ids = list(dict.fromkeys(near))
        wanted = [
            nid
            for found in hits.values()
            for nid, _, name in found
            if nid not in pending_ids and could_group(name)
        ]
        return list((await self._fetch_entities_by_ids(wanted, batch_size)).values())

    # ── 3. the LLM-judged phase ──

    async def _judge_phase(
        self, store: SystemStore, run: _Run, judge_llm: LLMInterface, batch_size: int
    ) -> None:
        """Let the judge decide identity for the pending entities; decide the
        merges by our rules.

        The judge (:class:`LLMJudgeDeduplicator`) partitions small sets of
        each pending entity and its nearest names. What it hands back is
        *identity*; *how to merge* stays here — the survivor is the one
        :func:`_survivor_rank` picks, two keyed rows never fold into each
        other, and a mention two rows could own stays apart. It works on the
        graph as the exact phase's decisions will leave it: an entity those
        absorb is represented by its survivor.
        """
        judge = LLMJudgeDeduplicator(self._graph, self._embedder, judge_llm)
        stats = self.last_judge_stats = {
            "entities": 0,
            "neighbours": 0,
            "candidates": 0,
            "sets": 0,
            "llm_calls": 0,
            "llm_failed": 0,
            "agreed_pairs": 0,
            "merged": 0,
        }
        points = [
            ent
            for ent in run.points
            if ent["id"] not in run.into and ent["id"] not in run.unsearched and ent["has_name"]
        ]
        missing = [ent["id"] for ent in points if ent["id"] not in run.hits]
        if missing:
            found, failed = await judge.discover(missing, NEIGHBOUR_K)
            run.hits.update(found)
            run.unsearched.update(dict.fromkeys(failed, "nearest-name lookup failed"))
            points = [ent for ent in points if ent["id"] not in failed]
        stats["entities"] = len(points)

        neighbours: dict[str, list[tuple[str, float]]] = {}
        for ent in points:
            picked: dict[str, float] = {}
            for nid, sim, _ in run.hits.get(ent["id"], []):
                if sim < NEIGHBOUR_GATE or len(picked) == NEIGHBOUR_K:
                    break
                nid = run.resolve(nid)
                if nid == ent["id"] or nid in picked:
                    continue
                if nid in run.unresolved:
                    # Never judged with anything until dedup_all retries it;
                    # who it was kept from is on its record.
                    run.blocked.setdefault(nid, set()).add(str(ent["name"]))
                    continue
                picked[nid] = sim
            neighbours[ent["id"]] = list(picked.items())
        extra = [
            (run.resolve(a), run.resolve(b))
            for a, b in self._judge_pairs
            if run.resolve(a) not in run.unresolved and run.resolve(b) not in run.unresolved
        ]
        point_ids = {ent["id"] for ent in points}
        wanted = {nid for found in neighbours.values() for nid, _ in found}
        wanted |= {i for pair in extra for i in pair}
        rows = {
            row["id"]: row
            for row in await judge.fetch_by_ids(sorted(wanted - point_ids), batch_size)
        }
        skip = self._undecidable_pairs

        def protected(a: dict[str, Any], b: dict[str, Any]) -> bool:
            # A pair phase 1 found undecidable, and any two rows written from a
            # declared key: two mapped ids are two rows, whatever their names.
            return frozenset((a["id"], b["id"])) in skip or (
                a.get("is_stub") is not None and b.get("is_stub") is not None
            )

        ents, sets, candidates = judge.candidate_sets(
            points, neighbours, rows, extra_pairs=extra, protected=protected
        )
        stats["neighbours"] = len(ents) - len(points)
        stats["candidates"] = candidates
        stats["sets"] = len(sets)
        if not sets:
            return
        if self._schema.keys:
            shown = sorted({i for members in sets for i in members})
            stored = await judge.attribute_values(
                [ents[i]["id"] for i in shown], self._schema.keys, batch_size
            )
            for i in shown:
                held = self._schema.values(ents[i], stored.get(ents[i]["id"], {}))
                ents[i]["values"] = [(f.shown, value) for f, value in held]
                ents[i]["attrs"] = {f.shown.casefold(): (f.key, value) for f, value in held}
        run.charged = {ents[i]["id"] for members in sets for i in members} & run.listed
        await self._claim(store, run)
        verdicts = await judge.judge(sets, ents, self._schema)
        stats["llm_calls"] = verdicts.calls
        stats["llm_failed"] = verdicts.failed_calls
        await self._resolve(run, ents, verdicts, protected, batch_size)

    async def _claim(self, store: SystemStore, run: _Run) -> None:
        """Charge one judged attempt to every listed entity in a set, before
        the judge is asked. A run that crashes keeps the charge — each judging
        run that starts is one attempt — so an entity whose sets crash the
        process every time still ends up unresolved, reason ``interrupted``."""
        claimed: list[PendingPage] = []
        for page in await store.pending_pages():
            charged = [i for i, eid in enumerate(page.ids) if eid in run.charged]
            if not charged or page.claimed_run == store.run_id:
                continue
            attempts, reasons = list(page.attempts), list(page.reasons)
            for i in charged:
                attempts[i] += 1
                reasons[i] = "interrupted"
            claimed.append(PendingPage(page.page, page.ids, attempts, reasons, store.run_id))
        await store.write_pending(claimed)

    async def _resolve(
        self,
        run: _Run,
        ents: list[dict[str, Any]],
        verdicts: Verdicts,
        protected: Callable[[dict[str, Any], dict[str, Any]], bool],
        batch_size: int,
    ) -> None:
        """Union the judge's agreements over the whole run into groups, then
        decide each group's merges.

        Transitive on purpose: "AAPL" = "Apple Inc." in one set and "Apple
        Inc." = "Apple" in another is one company; restricting unions to a
        single set left such chains split (recall 0.66 vs 0.90 on the labeled
        set, same precision). A union is refused when it would put into one
        group a pair the judge put apart in a set, or a protected pair (two
        keyed rows, an undecidable mention). A group that contains a member of
        a set without a usable verdict is held: that verdict could still change
        it, so it is not merged, and its pending members are checked again.
        """

        def apart(i: int, j: int) -> bool:
            return (min(i, j), max(i, j)) in verdicts.different or protected(ents[i], ents[j])

        agreed = sorted(pair for pair in verdicts.same if not apart(*pair))
        self.last_judge_stats["agreed_pairs"] = len(agreed)
        uf = UnionFind(range(len(ents)))
        members: dict[Hashable, set[int]] = {i: {i} for i in range(len(ents))}
        for a, b in agreed:
            ra, rb = uf.find(a), uf.find(b)
            if ra == rb:
                continue
            if any(apart(x, y) for x in members[ra] for y in members[rb]):
                logger.info(
                    "judge dedup: not grouping %s with %s — it would join a pair already "
                    "decided distinct",
                    ents[a]["id"],
                    ents[b]["id"],
                )
                continue
            joined = members.pop(ra) | members.pop(rb)
            uf.union(a, b)
            members[uf.find(a)] = joined

        run.failed = {ents[i]["id"] for i in verdicts.failed}
        groups = [sorted(group) for group in members.values() if len(group) > 1]
        held = [group for group in groups if verdicts.failed.intersection(group)]
        for group in held:
            run.held.update(ents[i]["id"] for i in group)
        merging = [group for group in groups if not verdicts.failed.intersection(group)]
        # Ranked on full rows: the judge's carry no degree. A row this run's
        # exact phase already decided on carries what its absorbs will add.
        ids = [ents[i]["id"] for group in merging for i in group]
        fetched = await self._fetch_entities_by_ids(
            [eid for eid in ids if eid not in run.rows], batch_size
        )
        skip = self._undecidable_pairs
        for group in merging:
            known = [run.rows.get(ents[i]["id"]) or fetched.get(ents[i]["id"]) for i in group]
            name = pick_group_name(group, verdicts.names, ents)
            self._plan_judged_group(
                run,
                [row for row in known if row is not None],
                [ents[i] for i in group],
                name,
                skip,
                keep=_picked_values(group, verdicts.keeps, ents),
            )

    def _plan_judged_group(
        self,
        run: _Run,
        known: list[dict[str, Any]],
        members: list[dict[str, Any]],
        preferred_name: str | None,
        skip: set[frozenset[str]],
        keep: dict[str, Any] | None = None,
    ) -> None:
        """Decide the merges of one group the judge agreed on.

        ``keep`` is the judge's choice, per attribute the members disagree on,
        of the value the merged entity keeps; it travels with the group's
        absorbs in the plan and is applied by :meth:`_fold`.

        ``preferred_name`` is the judge's best name for the group (one of its
        members' names or aliases). The survivor is still picked by
        :func:`_survivor_rank` — that decides which node, edges and id remain —
        but it takes this name, and its own moves into ``aliases``: "AAPL" can
        win on connections and still read "Apple Inc.". A table row keeps the
        name its table declared.
        """
        if len(known) < 2:
            return
        known.sort(key=_survivor_rank, reverse=True)
        survivor = known[0]
        duplicates = _keep_declared_identities_apart(survivor, known[1:])
        duplicates = self._keep_undecidable_mentions_apart(survivor, duplicates, members)
        absorbed: list[str] = []
        for dup in duplicates:
            # Transitivity can put a pair the model never saw as a candidate into
            # one group; a pair phase 1 found undecidable still holds — against
            # the survivor and against everything already folded into it.
            if any(frozenset((dup["id"], other)) in skip for other in (survivor["id"], *absorbed)):
                logger.info(
                    "Not merging %s into %s: the pair was already found undecidable",
                    dup["id"],
                    survivor["id"],
                )
                continue
            run.absorb(dup, survivor, JUDGE)
            absorbed.append(dup["id"])
            logger.info(
                "Judge decided %s %r -> %r", survivor.get("label"), dup["name"], survivor["name"]
            )
        self.last_judge_stats["merged"] = int(self.last_judge_stats.get("merged", 0)) + len(
            absorbed
        )
        if absorbed and keep:
            run.keeps[survivor["id"]] = keep
        if (
            absorbed
            and preferred_name
            and survivor.get("is_stub") is None
            and preferred_name.strip() != str(survivor.get("name") or "").strip()
        ):
            run.renames[survivor["id"]] = preferred_name.strip()

    def _plan_adoption(self, run: _Run, groups: dict[tuple[str, str], list[dict[str, Any]]]) -> int:
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

        The adopted node's labels do not travel (policy ``adopt``): its primary
        is the guess this pass exists to correct, and the labels it absorbed in
        earlier merges (``merged_labels``, extra Cypher labels) are guesses of
        the same provenance — stamping ``Practice`` on the declared entity while
        dropping ``Concept`` would treat two identical guesses oppositely.

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

        decided = 0
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
                    run.absorb(duplicate, survivor, ADOPT)
                    decided += 1
                    logger.info(
                        "Adopting %r from inferred label %r into declared label %r",
                        survivor["name"],
                        label,
                        declared[0],
                    )
                # Consumed, so the leftover report does not name it.
                groups.pop((norm_name, label), None)
        return decided

    # ── 4. the plan ──

    def _plan(self, run: _Run, *, release_unjudged: bool) -> tuple[list[PlanOp], Decisions]:
        """The run's merges as plan operations, and what each pending entity's
        check came to.

        A planned group is every entity one survivor ends up holding. Its
        absorbs run in dependency order — exact, then adoption, then judged,
        so an entity is complete before it is itself absorbed — with the old
        entities' first; a rename is the group's last operation.
        """
        uf = UnionFind()
        for dup, (survivor, _) in run.into.items():
            uf.union(survivor, dup)
        grouped: dict[Hashable, list[str]] = {}
        for dup in run.into:
            grouped.setdefault(uf.find(dup), []).append(dup)
        # A rename follows a judged absorb, so its survivor is in a group.
        renames = {uf.find(survivor): (survivor, name) for survivor, name in run.renames.items()}
        tier = {EXACT: 0, ADOPT: 1, JUDGE: 2}
        ops: list[PlanOp] = []
        for gid, (root, dups) in enumerate(grouped.items()):
            for dup in sorted(dups, key=lambda d: (tier[run.into[d][1]], d in run.listed)):
                survivor, policy = run.into[dup]
                ops.append(
                    PlanOp(
                        ABSORB,
                        gid,
                        survivor,
                        run.rows[survivor]["name"],
                        dup=dup,
                        dup_name=run.rows[dup]["name"],
                        policy=policy,
                        keep=(
                            json.dumps(run.keeps[survivor], default=str)
                            if policy == JUDGE and survivor in run.keeps
                            else ""
                        ),
                    )
                )
            if root in renames:
                survivor, name = renames[root]
                ops.append(
                    PlanOp(RENAME, gid, survivor, run.rows[survivor]["name"], rename_to=name)
                )

        decisions = Decisions()
        handoff: set[str] = set()

        def retry(eid: str, why: str) -> None:
            decisions.retry.append(eid)
            decisions.reasons.append(why)

        for eid in sorted(run.listed):
            if eid in run.gone:
                decisions.clear.append(eid)
            elif eid in run.unsearched:
                retry(eid, run.unsearched[eid])
            elif run.judged:
                if eid in run.failed:
                    retry(eid, "no usable verdict")
                elif eid in run.held:
                    # Held only because of a neighbour's failure: not its own
                    # failed attempt, so the claim's charge is given back.
                    retry(eid, "held with a set that got no verdict")
                    if eid in run.charged:
                        decisions.refund.append(eid)
                else:
                    decisions.clear.append(eid)
            elif release_unjudged:
                decisions.clear.append(eid)
            elif eid in run.into:
                # Absorbed by name, its judged check still owed: the survivor
                # holding it now owes it instead.
                decisions.clear.append(eid)
                handoff.add(run.resolve(eid))
        if run.judged:
            # A dedup_all run checks entities no record lists; one whose check
            # did not complete becomes pending.
            handoff |= {
                ent["id"]
                for ent in run.points
                if ent["id"] not in run.listed
                and (
                    ent["id"] in run.failed or ent["id"] in run.held or ent["id"] in run.unsearched
                )
            }
        decisions.handoff = sorted(handoff - run.listed)
        return ops, decisions

    # ── 4. apply ──

    async def _apply(self, store: SystemStore, plan: Plan) -> int:
        """Apply a sealed plan from each page's pointer; returns the absorbs
        applied by this call. Each operation is one query that ticks itself
        off, so a crash, a lost reply or a takeover resumes exactly here."""
        merged = 0
        # A rename names the group its judged absorbs formed; with none of them
        # applied (every one found its entities changed) there is no such group.
        judged: set[int] = set()
        for page, (ops, nxt) in enumerate(zip(plan.ops, plan.nexts)):
            judged.update(
                op.group
                for i, op in enumerate(ops[:nxt])
                if op.op == ABSORB and op.policy == JUDGE and i not in plan.changed[page]
            )
        for page, ops in enumerate(plan.ops):
            for i in range(plan.nexts[page], len(ops)):
                op = ops[i]
                if op.op == ABSORB:
                    state = await self._apply_absorb(store, plan.plan_run, page, i, op)
                    if state == APPLIED:
                        merged += 1
                        self._name_embedded_ids.discard(op.dup)
                        if op.policy == JUDGE:
                            judged.add(op.group)
                else:
                    state = await self._apply_rename(
                        store, plan.plan_run, page, i, op, apply=op.group in judged
                    )
                if state == CHANGED:
                    self.last_report.changed += 1
                    logger.info(
                        "Planned %s of %s skipped: an entity changed during the run; "
                        "its group is checked again next run",
                        op.op,
                        op.dup or op.survivor,
                    )
        return merged

    async def _apply_absorb(
        self, store: SystemStore, plan_run: str, page: int, i: int, op: PlanOp
    ) -> str:
        """One absorb: read both entities, generate the merge query from what
        the duplicate holds, run it. A read that went out of date before the
        write (a new relationship type, ingest writing either node) writes
        nothing and is read again; one that keeps doing so is recorded as
        changed, so the group is decided again next run."""
        where = {
            "plan_run": plan_run,
            "page": page,
            "op": i,
            "dup": op.dup,
            "dup_name": op.dup_name,
            "survivor": op.survivor,
            "survivor_name": op.survivor_name,
        }
        for _ in range(_STALE_READS):
            read = await read_for_merge(self._graph, op.survivor, op.dup)
            if read is None:
                break  # an entity is gone: recorded as changed below
            fold = self._fold(op, read)
            rows = await store.fenced(
                absorb_query(read.kinds, fold.sets, fold.labels),
                {**fold.params, **where, "handled": handled(read.kinds)},
            )
            state = outcome(rows[0], i)
            if state == APPLIED:
                self.last_report.collisions.extend(
                    f"{op.survivor_name}: {c.key} kept {c.kept!r}"
                    f"{' (judge)' if c.by_judge else ''}, dropped {c.dropped!r} from {op.dup_name}"
                    for c in fold.collisions
                )
            if state != STALE:
                return state
        return outcome((await store.fenced(SKIP_QUERY, where))[0], i)

    def _fold(self, op: PlanOp, read: MergeRead) -> Fold:
        """What an absorb writes onto the survivor, from the two nodes as read:
        SET items, labels to add, parameters, and the property collisions.

        The policy decides the labels. Every absorb carries the labels the
        duplicate holds beyond its own primary — the ones earlier merges gave
        it, on the graph or on record in ``merged_labels``. A judged absorb
        also unions the duplicate's own labels (an ``Organization`` in one file
        and the same ``Company`` in another end up as one node carrying both);
        an adoption carries none, since the duplicate's primary is the guess it
        corrects and what it absorbed is of the same provenance. The union is
        written to the survivor's ``merged_labels`` too: the phases read one
        primary label per node, so a survivor that later loses to another node
        would be deleted with its extra labels unread; the property travels.
        Labels are admitted by :func:`label_for_merge`; anything else is
        dropped with a warning.

        The description is joined (:func:`_merge_description_members`, the
        rule every merge path shares), the duplicate's name and aliases become
        the survivor's aliases, and any property the survivor lacks is copied
        (:func:`properties_to_carry`). Node provenance is unioned by the query
        itself.

        A property both nodes hold with different values is a **collision**,
        and no value is lost to it: the one that stays is the judge's choice
        (``KEEP``, on a judged absorb) or else the survivor's, and the other is
        appended to ``<property>__dropped`` on the survivor — a list, unioned
        with both nodes' earlier ones, so a later merge adds to the history.
        Two list values are unioned instead. The SDK's own properties never
        collide: each has its own rule above.
        """
        keep, dup = read.keep, read.dup
        recorded = split_merged_labels(keep.get(MERGED_LABELS_KEY))
        primary = _primary_label(read.keep_labels, recorded)
        dup_absorbed = split_merged_labels(dup.get(MERGED_LABELS_KEY))
        dup_primary = _primary_label(read.dup_labels, dup_absorbed)
        wanted: list[str] = [dup_primary] if op.policy == JUDGE and dup_primary else []
        if op.policy != ADOPT:
            wanted += [lab for lab in read.dup_labels if lab != dup_primary]
            wanted += dup_absorbed
        held = set(read.keep_labels) | set(recorded) | {primary}
        labels: list[str] = []
        for lab in wanted:
            safe = label_for_merge(lab)
            if safe is None:
                logger.warning("Not adding label %r to %s: not a usable label", lab, op.survivor)
                continue
            if safe not in held and safe not in labels:
                labels.append(safe)
        merged_labels = [lab for lab in (*recorded, *labels) if lab != primary]

        sets: list[str] = []
        params: dict[str, Any] = {}
        # Joined with " | ", matching LLMVerifiedResolution's survivor rule, so
        # both mechanisms leave the same shape behind — and counted by the same
        # rule (``description_list``). Both forms are written together:
        # ``description_list`` prefers the list wherever it is present, so a
        # survivor updated without it would silently report fewer members than
        # its own ``description`` holds.
        segments = _merge_description_members(_description_members(keep), _description_members(dup))
        description = " | ".join(segments)
        if description != str(keep.get("description") or "") or segments != _string_list(
            keep.get("descriptions")
        ):
            sets += ["s.description = $desc", "s.descriptions = $descs"]
            params.update(desc=description, descs=segments)

        current = _string_list(keep.get("aliases"))
        aliases = self._merged_aliases(
            {"name": keep.get("name") or "", "aliases": current},
            {"name": dup.get("name") or "", "aliases": _string_list(dup.get("aliases"))},
        )
        if aliases != current:
            sets.append("s.aliases = $aliases")
            params["aliases"] = aliases

        carry = properties_to_carry(keep, dup, never=self._NEVER_CARRY)
        # Handled above, by the collisions below, or by the absorb query itself.
        for key in [k for k in carry if k in _FOLDED_KEYS or k.endswith(DROPPED_SUFFIX)]:
            carry.pop(key)
        if carry:
            sets.append("s += $carry")
        if merged_labels != recorded:
            sets.append(f"s.{MERGED_LABELS_KEY} = $merged_labels")
            params["merged_labels"] = " | ".join(merged_labels)

        override, collisions, seen = _collisions(keep, dup, json.loads(op.keep or "{}"))
        if override:
            sets.append("s += $override")
            params["override"] = override
        return Fold(sets, labels, {**params, **snapshot_params(read, carry, seen)}, collisions)

    async def _apply_rename(
        self, store: SystemStore, plan_run: str, page: int, i: int, op: PlanOp, *, apply: bool
    ) -> str:
        """Give a merged survivor the judge's best name; the old one becomes an
        alias. The id does not change. The name vector is the old name's, so it
        is replaced with the new name's; if that cannot be embedded it is
        removed, and the next run embeds the survivor and checks it again."""
        vector: list[float] = []
        if apply:
            try:
                vector = list((await self._embedder.aembed_documents([op.rename_to]))[0] or [])
            except Exception:
                logger.debug("Could not embed renamed survivor %s", op.survivor, exc_info=True)
        rows = await store.fenced(
            rename_query(with_vector=bool(vector)),
            {
                "plan_run": plan_run,
                "page": page,
                "op": i,
                "survivor": op.survivor,
                "survivor_name": op.survivor_name,
                "name": op.rename_to,
                "vector": vector,
                "apply": apply,
            },
        )
        state = outcome(rows[0], i)
        if state == APPLIED and apply:
            logger.info("Judge named merged %r -> %r", op.survivor_name, op.rename_to)
            if vector:
                self._name_embedded_ids.add(op.survivor)
        return state

    # ── 5. clear ──

    async def _clear(self, store: SystemStore, plan: Plan) -> None:
        """Apply the plan's decisions to the pending records, one page per
        query, each ticked in the same query — so a replay cannot clear or
        refund twice.

        A group in which any operation found its entities changed is decided
        again: its members' checks did not complete after all.
        """
        members: dict[int, set[str]] = defaultdict(set)
        changed_groups: set[int] = set()
        for page, ops in enumerate(plan.ops):
            for i, op in enumerate(ops):
                members[op.group].update(eid for eid in (op.survivor, op.dup) if eid)
                if i in plan.changed[page]:
                    changed_groups.add(op.group)
        recheck = set().union(*(members[g] for g in changed_groups))
        for page, (decided, done) in enumerate(zip(plan.decisions, plan.decision_done)):
            if done:
                continue
            records = await store.pending_pages()
            updates, cleared = _decide_records(records, decided, recheck)
            if await store.apply_decisions(plan.plan_run, page, updates):
                self.last_report.checked += cleared
            if decided.handoff and not records:
                # Every record was cleared before this page, so none could take
                # the hand-off in the same query.
                await store.add_pending(decided.handoff)

    async def _summarise(self, store: SystemStore) -> None:
        pages = await store.pending_pages()
        self.last_report.pending = len({eid for page in pages for eid in page.ids})
        self.last_report.unresolved = await store.unresolved()
        if self.last_report.pending:
            logger.warning(
                "%d entities are still pending a dedup check (a failed judge call or lookup, "
                "a group held for one, a merge an update changed, or a run without the "
                "judge); the next finalize() checks them",
                self.last_report.pending,
            )

    # ── Helpers ──

    @property
    def _undecidable_pairs(self) -> set[frozenset[str]]:
        """``(mention, row)`` pairs the exact phase found no way to decide.

        A mention whose name two keyed rows share fits both equally, and the
        judge would otherwise settle it on the one evidence phase 1 had already
        rejected, folding the mention into whichever row ranks higher. These
        pairs are kept out of the judge phase and reported instead; a name
        shared by two rows is not disambiguated by looking at the name harder.
        """
        return {frozenset((miss.id_a, miss.id_b)) for miss in self._ambiguous_mentions}

    async def _report_near_misses(self, batch_size: int) -> None:
        """Record surviving entities that probably denote one thing.

        Grouping is by canonical name, so two sources spelling a name in ways
        it does not unify leave two nodes, unless the judge merges them: an HR
        export's "Maya Ellison" and a board review's "M. Ellison" end up as two
        people, one holding her age and the other what she did, and every question
        needing both comes back wrong while looking answered.

        Found here and **not** merged. Merging on a guess is destructive and
        irreversible; two nodes are recoverable, and a user told which pairs look
        wrong can fix the source, which fixes it for good. A pair the exact
        phase cannot decide and the judge did not merge is reported.
        """
        try:
            # The entities this run looked at and every nearest name the index
            # returned for them (an existing "Maya Ellison" beside a new
            # "M. Ellison" is no exact match, but is a near miss), as they are
            # now. A pair of two old entities was reported by the run that
            # checked them, so only pairs touching a new entity are kept.
            ids = list(dict.fromkeys(self._run_entity_ids + self._near_ids))
            survivors = list((await self._fetch_entities_by_ids(ids, batch_size)).values())
            alive = {ent["id"] for ent in survivors}
            # A mention left between two rows first: it is an exact name match
            # that did not merge, which is the finding most worth a look. One the
            # judge did settle is no longer open.
            open_mentions = [
                miss
                for miss in self._ambiguous_mentions
                if miss.id_a in alive and miss.id_b in alive
            ]
            found = find_near_misses(survivors)
            if not self._dedup_all:
                new = set(self._pending_ids)
                found = [m for m in found if m.id_a in new or m.id_b in new]
            self.near_misses = open_mentions + found
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

    @staticmethod
    def _merged_aliases(survivor: dict[str, Any], dup: dict[str, Any]) -> list[str]:
        """The survivor's aliases plus the duplicate's name and aliases.

        A name that normalises to the survivor's own is a spelling variant,
        not an alias worth keeping; an acronym or a judged variant is.
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
    def _merge_acronym_groups(
        groups: list[list[dict]],
        ambiguous_out: list[tuple[list[dict], list[dict]]] | None = None,
    ) -> list[list[dict]]:
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
        Ambiguous cases are the LLM judge's job: with ``ambiguous_out`` they are
        appended there as ``(short-form group, long-form group)`` pairs.
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
                if ambiguous_out is not None:
                    ambiguous_out.extend((groups[i], groups[j]) for j in matches)
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

    def _keep_undecidable_mentions_apart(
        self,
        survivor: dict[str, Any],
        duplicates: list[dict[str, Any]],
        members: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop mentions the judge put on a row without saying which row.

        A mention is kept apart when phase 1 found it between two rows, or when
        the group itself holds two keyed rows and the mention — the same
        ambiguity, reached by a judge that grouped all three. What remains is a
        mention paired with exactly one row, which is a decision, not a coin
        toss.
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
        """Fetch all entities in batches, including their primary label.

        A node that absorbed another type carries several labels, and FalkorDB
        does not promise the order ``labels(e)`` returns them in; taking the
        first would let an ``Organization`` that absorbed a ``Company`` report
        either on a later run, and the label is a grouping key in the exact
        phase. The primary label is the first one *not* recorded in
        ``merged_labels`` — the ones a merge added — so it stays what the node
        was written as.
        """
        # Keyset paging (ORDER BY id, WHERE id > last), with the degree computed
        # only for the page: SKIP re-walked and re-counted every skipped row,
        # which made a full read quadratic in the number of entities.
        last = ""
        entities: list[dict[str, Any]] = []
        for _ in range(_MAX_PAGINATION_ITERATIONS):
            result = await self._graph.query_raw(
                "MATCH (e:__Entity__) WHERE e.id > $last "
                f"WITH e ORDER BY e.id LIMIT $limit {_ENTITY_RETURN}",
                {"last": last, "limit": batch_size},
            )
            rows = result.result_set or []
            entities.extend(_entity_from_row(row) for row in rows)
            if len(rows) < batch_size:
                break
            last = rows[-1][0]
        else:
            logger.error(
                "Pagination exceeded %d iterations in _fetch_all_entities — aborting",
                _MAX_PAGINATION_ITERATIONS,
            )
        return entities

    async def _fetch_entities_by_ids(
        self, ids: Iterable[str], batch_size: int = 500
    ) -> dict[str, dict[str, Any]]:
        """The current rows of ``ids`` that still exist, by id."""
        wanted = list(dict.fromkeys(ids))
        out: dict[str, dict[str, Any]] = {}
        for k in range(0, len(wanted), batch_size):
            result = await self._graph.query_raw(
                f"UNWIND $ids AS id MATCH (e:__Entity__ {{id: id}}) WITH e {_ENTITY_RETURN}",
                {"ids": wanted[k : k + batch_size]},
            )
            for row in result.result_set or []:
                ent = _entity_from_row(row)
                out[ent["id"]] = ent
        return out

    async def _count_names_embedded(self) -> None:
        """Name vectors this run wrote that are still in the graph: a vector
        written to a node a merge then deleted is not one the graph holds."""
        ids = list(self._name_embedded_ids)
        found = 0
        try:
            for k in range(0, len(ids), 10_000):
                result = await self._graph.query_raw(
                    "UNWIND $ids AS id MATCH (e:__Entity__ {id: id}) RETURN count(e)",
                    {"ids": ids[k : k + 10_000]},
                )
                found += int((result.result_set or [[0]])[0][0] or 0)
        except Exception:
            logger.debug("Could not count the name vectors written", exc_info=True)
            found = len(ids)
        self.names_embedded = found

    # Written by the system, never carried across from a duplicate.
    _NEVER_CARRY = frozenset({"id", "embedding"})


#: Where a merge keeps the values a collision displaced: ``born__dropped``.
DROPPED_SUFFIX = "__dropped"
# Properties a merge folds by a rule of its own, never by carry.
_FOLDED_KEYS = frozenset(
    {"description", "descriptions", "aliases", "source_chunk_ids", MERGED_LABELS_KEY}
)
# Properties the SDK owns; none collides — the survivor's stays, and one the
# survivor lacks is carried (``is_stub``, ``entity_key``: it now holds the row).
_OWN_KEYS = RESERVED_PROPERTY_NAMES | _FOLDED_KEYS


class Collision(NamedTuple):
    key: str
    kept: Any
    dropped: Any
    by_judge: bool


class Fold(NamedTuple):
    """What one absorb writes: see :meth:`EntityDeduplicator._fold`."""

    sets: list[str]
    labels: list[str]
    params: dict[str, Any]
    collisions: list[Collision]


def _canon(value: Any) -> str:
    """A value's comparable form: equal values compare equal whatever their
    container type (a pick stored as JSON in the plan, a value read back)."""
    return json.dumps(value, sort_keys=True, default=str)


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else ([] if value is None else [value])


def _union(*lists: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for item in (x for values in lists for x in values):
        if _canon(item) not in seen:
            seen.add(_canon(item))
            out.append(item)
    return out


def _collisions(
    keep: dict[str, Any], dup: dict[str, Any], picked: dict[str, Any]
) -> tuple[dict[str, Any], list[Collision], list[str]]:
    """The writes one absorb's property collisions need: ``(override, the
    collisions, every property read for them)`` — see
    :meth:`EntityDeduplicator._fold`. ``picked`` is the judge's KEEP for the
    group: a value it chose that is neither node's leaves the survivor's."""
    override: dict[str, Any] = {}
    collisions: list[Collision] = []
    lost: dict[str, list[Any]] = defaultdict(list)
    for key in sorted(set(keep) & set(dup)):
        if key in _OWN_KEYS or key.endswith(DROPPED_SUFFIX):
            continue
        mine, theirs = keep[key], dup[key]
        if mine in (None, "", []) or theirs in (None, "", []) or _canon(mine) == _canon(theirs):
            continue
        if isinstance(mine, list) and isinstance(theirs, list):
            override[key] = _union(mine, theirs)
            continue
        choice = _canon(picked[key]) if key in picked else None
        if choice == _canon(theirs):
            override[key] = theirs
            collisions.append(Collision(key, theirs, mine, by_judge=True))
            lost[key + DROPPED_SUFFIX].append(mine)
        else:
            collisions.append(Collision(key, mine, theirs, by_judge=choice == _canon(mine)))
            lost[key + DROPPED_SUFFIX].append(theirs)
    histories = {k for k in (*keep, *dup) if k.endswith(DROPPED_SUFFIX)} | set(lost)
    for key in sorted(histories):
        merged = _union(_as_list(keep.get(key)), _as_list(dup.get(key)), lost.get(key, []))
        if merged != _as_list(keep.get(key)):
            override[key] = merged
    seen = sorted(set(override) | {c.key for c in collisions})
    return override, collisions, seen


def _picked_values(
    group: list[int], keeps: dict[frozenset[int], dict[str, int]], ents: list[dict[str, Any]]
) -> dict[str, Any]:
    """The values the judge chose for a group, ``{node property: value}``:
    from every judged sub-group, a pick naming an attribute the chosen member
    shows. Two picks of different values for one property cancel out."""
    members = set(group)
    chosen: dict[str, Any] = {}
    torn: set[str] = set()
    for sub, picks in keeps.items():
        if not sub <= members:
            continue
        for attr, i in picks.items():
            hit = (ents[i].get("attrs") or {}).get(attr)
            if hit is None:
                continue
            key, value = hit
            if key in chosen and _canon(chosen[key]) != _canon(value):
                torn.add(key)
            chosen.setdefault(key, value)
    return {k: v for k, v in chosen.items() if k not in torn}


def _any_decision(decisions: Decisions) -> bool:
    return any(getattr(decisions, col) for col in ("clear", "retry", "refund", "handoff"))


def _decide_records(
    records: list[PendingPage], decided: Decisions, recheck: set[str]
) -> tuple[list[PendingPage], int]:
    """The pending records as one decision page leaves them, and how many
    entities it clears. Only records it changes are returned. ``recheck`` are
    the members of groups with a changed operation: kept pending whatever the
    page says."""
    clear = set(decided.clear) - recheck
    reasons = dict(zip(decided.retry, decided.reasons))
    reasons.update(dict.fromkeys(set(decided.clear) & recheck, "changed during the run"))
    refund = set(decided.refund)
    updates: list[PendingPage] = []
    cleared = 0
    for rec in records:
        ids: list[str] = []
        attempts: list[int] = []
        why: list[str] = []
        touched = False
        for eid, tries, reason in zip(rec.ids, rec.attempts, rec.reasons):
            if eid in clear:
                touched = True
                cleared += 1
                continue
            if eid in refund:
                tries, touched = max(0, tries - 1), True
            if eid in reasons:
                reason, touched = reasons[eid], True
            ids.append(eid)
            attempts.append(tries)
            why.append(reason)
        if touched:
            updates.append(PendingPage(rec.page, ids, attempts, why, rec.claimed_run))
    listed = {eid for rec in records for eid in rec.ids}
    handoff = [eid for eid in decided.handoff if eid not in listed]
    if handoff and records:
        last = next((u for u in updates if u.page == records[-1].page), None)
        if last is None:
            rec = records[-1]
            last = PendingPage(rec.page, list(rec.ids), list(rec.attempts), list(rec.reasons))
            updates.append(last)
        last.ids += handoff
        last.attempts += [0] * len(handoff)
        last.reasons += [""] * len(handoff)
    return updates, cleared


def _primary_label(labels: list[str], absorbed: list[str]) -> str:
    """The label a node was written as: the first one not recorded as
    absorbed by a merge. FalkorDB does not promise the order ``labels(e)``
    returns them in; with every label on record as absorbed there is no
    primary to prefer, and the smallest is at least the same one every run."""
    if not labels:
        return ""
    return next((lab for lab in labels if lab not in absorbed), min(labels))
