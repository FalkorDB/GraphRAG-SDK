# GraphRAG SDK — Storage: one merge, one query
#
# A merge of a duplicate entity into its survivor moves every relationship the
# duplicate has, folds its properties into the survivor, deletes it and ticks
# the run's merge plan — all in ONE query. FalkorDB runs one write at a time per
# graph, so nothing lands between moving the edges and deleting the duplicate:
# an edge ingest adds a moment earlier is moved, one it adds a moment later
# finds no duplicate to attach to, and a merge is never half applied.
#
# Cypher cannot create a relationship whose type is a variable, so the query is
# generated per merge from the relationship types (and directions) the
# duplicate has when it is read. The query checks that the duplicate has no
# type it was not generated for, and that the properties the fold was computed
# from are unchanged; otherwise nothing is written and the caller reads again.
# Measured on FalkorDB 4.20.4: 120 edges of five kinds, both directions, moved
# in one 61 ms query.
#
# Edge keys mirror the write path (``GraphStore.upsert_relationships``):
# ``RELATES`` is keyed on ``rel_type`` with ``source_chunk_ids`` unioned, every
# other type on its endpoints, with ``SET r += properties``. An edge between the
# duplicate and the survivor (or a duplicate's self-loop) would become a
# self-loop on the survivor and is dropped.
#
# Two pitfalls of the queries this replaces still apply: the survivor is bound
# before any ``MERGE`` (a MERGE on an unbound path creates a nameless ghost
# entity), and every pattern expands from the bound duplicate (anchored on a
# labelled far end, the planner scans the whole label).

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from graphrag_sdk.core.exceptions import GraphRAGError
from graphrag_sdk.storage.system_store import PLAN, PROCESSING_LABEL
from graphrag_sdk.utils.cypher import sanitize_cypher_label

OUT, IN = "out", "in"

#: What a plan operation did.
APPLIED = "applied"  # written and ticked
DONE = "done"  # ticked by an earlier attempt: a replay, nothing written
CHANGED = "changed"  # an entity is gone, renamed or deleted since judging: recorded, ticked
STALE = "stale"  # the read is out of date: nothing written, read again


class PlanStateError(GraphRAGError):
    """A plan page is missing, or an operation ran before its turn. Never
    treated as "done": it means the plan or the caller is inconsistent."""


# The duplicate's relationship types, each once, by direction. Each pattern
# needs its own relationship variable: FalkorDB 4.20 binds a name reused across
# two pattern comprehensions of one projection to the first, and the second
# list comes back empty.
_DISTINCT_TYPES = (
    "reduce(a = [], t IN [{pattern} | type({var})] | CASE WHEN t IN a THEN a ELSE a + [t] END)"
)

_READ_QUERY = (
    "MATCH (k:__Entity__ {id: $survivor}), (d:__Entity__ {id: $dup}) "
    "RETURN properties(k), properties(d), labels(k), labels(d), "
    f"{_DISTINCT_TYPES.format(pattern='(d)-[ro]->()', var='ro')}, "
    f"{_DISTINCT_TYPES.format(pattern='(d)<-[ri]-()', var='ri')} LIMIT 1"
)


@dataclass
class MergeRead:
    """Both entities as one merge reads them. Labels exclude ``__Entity__``."""

    keep: dict[str, Any]
    dup: dict[str, Any]
    keep_labels: list[str]
    dup_labels: list[str]
    kinds: list[tuple[str, str]]


async def read_for_merge(graph: Any, survivor_id: str, dup_id: str) -> MergeRead | None:
    """Read what :func:`absorb_query` is generated from; ``None`` if either
    entity is not there. Raises what the read raises."""
    result = await graph.query_raw(_READ_QUERY, {"survivor": survivor_id, "dup": dup_id})
    rows = result.result_set or []
    if not rows:
        return None
    keep, dup, keep_labels, dup_labels, out_types, in_types = rows[0]
    return MergeRead(
        keep=keep or {},
        dup=dup or {},
        keep_labels=[lab for lab in keep_labels or [] if lab != "__Entity__"],
        dup_labels=[lab for lab in dup_labels or [] if lab != "__Entity__"],
        kinds=edge_kinds(out_types or [], in_types or []),
    )


def edge_kinds(out_types: Iterable[str], in_types: Iterable[str]) -> list[tuple[str, str]]:
    """``(type, direction)`` pairs, each once, in a stable order."""
    return sorted({(t, OUT) for t in out_types if t} | {(t, IN) for t in in_types if t})


def handled(kinds: list[tuple[str, str]]) -> list[str]:
    """The ``$handled`` parameter of :func:`absorb_query` for ``kinds``."""
    return [f"{t}:{d}" for t, d in kinds]


def snapshot_params(read: MergeRead, carry: dict[str, Any]) -> dict[str, Any]:
    """The preconditions :func:`absorb_query` checks: the fields the fold was
    computed from, as read. ``carry`` is what the fold copies from the
    duplicate onto properties the survivor lacked."""
    params: dict[str, Any] = {"carry": carry}
    for side, props in (("s", read.keep), ("d", read.dup)):
        for param, key, empty in (
            ("desc", "description", ""),
            ("descs", "descriptions", []),
            ("aliases", "aliases", []),
            ("merged", "merged_labels", ""),
        ):
            value = props.get(key)
            params[f"{side}_{param}"] = empty if value is None else value
    return params


def outcome(row: list[Any], op: int) -> str:
    """:data:`APPLIED`, :data:`DONE`, :data:`CHANGED` or :data:`STALE` from a
    plan query's row (ownership is checked by ``SystemStore.fenced``)."""
    _owned, plan_found, nxt, my_turn, present, fresh, unhandled = row
    if not plan_found:
        raise PlanStateError(f"The merge plan page of operation {op} is missing")
    if not my_turn:
        if int(nxt) > op:
            return DONE
        raise PlanStateError(f"Operation {op} ran before its turn (the page is at {nxt})")
    if not present:
        return CHANGED
    return APPLIED if fresh and not unhandled else STALE


# ── the queries ──────────────────────────────────────────────────
#
# Each continues from ``SystemStore``'s fence (``WITH _run, owned``) and
# returns ``owned, plan_found, next, my_turn, present, fresh, unhandled``.
# ``next`` is the pointer as found, before this query moves it.

# The plan page, and whether this operation is the next one due.
_TURN = (
    f"OPTIONAL MATCH (p:{PROCESSING_LABEL} {{kind: '{PLAN}', plan_run: $plan_run, page: $page}}) "
    "WHERE owned "
    "WITH owned, p, p IS NOT NULL AS plan_found, p.next AS next "
    "WITH owned, p, plan_found, next, (plan_found AND next = $op) AS my_turn "
)
_TICKED = "owned, p, plan_found, next, my_turn"

# Record ``changed`` and tick when the entities are not there; tick when applied.
_TICK = (
    "FOREACH (_ IN CASE WHEN my_turn AND NOT present THEN [1] ELSE [] END | "
    "SET p.changed = p.changed + [$op]) "
    "FOREACH (_ IN CASE WHEN my_turn AND (NOT present OR ok) THEN [1] ELSE [] END | "
    "SET p.next = $op + 1) "
)
_RETURN = "RETURN owned, plan_found, next, my_turn, present, fresh, unhandled"

# Only ``deleted`` records carry ``eid``, so the lookup is one index seek. Each
# entity is aggregated to one node: there is no uniqueness constraint on
# ``__Entity__.id``, and no write below may run once per copy.
_ABSORB_HEAD = (
    _TURN + f"OPTIONAL MATCH (g:{PROCESSING_LABEL}) WHERE my_turn AND g.eid IN [$dup, $survivor] "
    f"WITH {_TICKED}, count(g) = 0 AS kept "
    "OPTIONAL MATCH (d:__Entity__ {id: $dup}) WHERE my_turn AND kept AND d.name = $dup_name "
    f"WITH {_TICKED}, head(collect(d)) AS d "
    "OPTIONAL MATCH (s:__Entity__ {id: $survivor}) "
    "WHERE d IS NOT NULL AND s.name = $survivor_name AND s <> d "
    f"WITH {_TICKED}, d, head(collect(s)) AS s "
    f"WITH {_TICKED}, d, s, (d IS NOT NULL AND s IS NOT NULL) AS present "
    # The fields the fold was computed from, unchanged since the read: an
    # ingest write in between would otherwise be overwritten by a stale fold.
    f"WITH {_TICKED}, d, s, present, present "
    "AND coalesce(s.description, '') = $s_desc AND coalesce(s.descriptions, []) = $s_descs "
    "AND coalesce(s.aliases, []) = $s_aliases AND coalesce(s.merged_labels, '') = $s_merged "
    "AND coalesce(d.description, '') = $d_desc AND coalesce(d.descriptions, []) = $d_descs "
    "AND coalesce(d.aliases, []) = $d_aliases AND coalesce(d.merged_labels, '') = $d_merged "
    "AND all(k IN keys($carry) WHERE d[k] = $carry[k] "
    "AND (s[k] IS NULL OR s[k] = '' OR s[k] = [])) AS fresh "
    # Every relationship type the duplicate has must have a block below. The
    # parentheses matter: ``a + CASE ... END IN l`` parses as ``a + (... IN l)``.
    "OPTIONAL MATCH (d)-[x]-() WHERE present "
    f"WITH {_TICKED}, d, s, present, fresh, "
    "sum(CASE WHEN x IS NULL THEN 0 "
    "WHEN (type(x) + CASE WHEN startNode(x) = d THEN ':out' ELSE ':in' END) IN $handled THEN 0 "
    "ELSE 1 END) AS unhandled "
    f"WITH {_TICKED}, d, s, present, fresh, unhandled, "
    "(present AND fresh AND unhandled = 0) AS ok "
)
_CARRIED = f"{_TICKED}, d, s, present, fresh, unhandled, ok"


def _move_block(rel_type: str, direction: str) -> str:
    """Move the duplicate's ``rel_type`` edges of one direction to the survivor."""
    safe = sanitize_cypher_label(rel_type)
    key = " {rel_type: coalesce(e.rel_type, '')}" if safe == "RELATES" else ""
    if direction == OUT:
        old, new, existing = (
            f"(d)-[e:`{safe}`]->(b)",
            f"(s)-[nr:`{safe}`{key}]->(b)",
            f"(s)-[ex:`{safe}`{key}]->(b)",
        )
    else:
        old, new, existing = (
            f"(b)-[e:`{safe}`]->(d)",
            f"(b)-[nr:`{safe}`{key}]->(s)",
            f"(b)-[ex:`{safe}`{key}]->(s)",
        )
    if key:
        # Provenance is unioned, not overwritten: the survivor's edge keeps the
        # chunks that already vouched for it (``delete_stale_relationships``).
        # Read before the MERGE, which inside FOREACH cannot be followed by WITH.
        read_old = (
            f"OPTIONAL MATCH {existing} "
            f"WITH {_CARRIED}, e, b, coalesce(ex.source_chunk_ids, []) AS old "
        )
        update = (
            "SET nr += properties(e) "
            "SET nr.source_chunk_ids = old + "
            "[c IN coalesce(e.source_chunk_ids, []) WHERE NOT c IN old]"
        )
    else:
        read_old, update = "", "SET nr += properties(e)"
    return (
        f"OPTIONAL MATCH {old} WHERE ok AND b <> s AND b <> d "
        f"WITH {_CARRIED}, collect([e, b]) AS rows "
        "UNWIND CASE WHEN size(rows) = 0 THEN [null] ELSE rows END AS row "
        f"WITH {_CARRIED}, row[0] AS e, row[1] AS b "
        f"{read_old}"
        "FOREACH (_ IN CASE WHEN ok AND e IS NOT NULL THEN [1] ELSE [] END | "
        f"MERGE {new} {update}) "
        f"WITH {_CARRIED}, count(*) AS _moved "
    )


def absorb_query(kinds: list[tuple[str, str]], sets: list[str], labels: list[str]) -> str:
    """The fenced query that applies one ``absorb`` operation.

    ``kinds`` are the duplicate's relationship types by direction
    (:attr:`MergeRead.kinds`). ``sets`` are ``SET`` items on ``s`` computed
    from the read, using the caller's own parameters; ``labels`` are labels to
    add to ``s``, already admitted by ``label_for_merge``. The node-level
    ``source_chunk_ids`` union is done here, from the values at write time.

    Parameters, besides the caller's: ``plan_run``, ``page``, ``op``, ``dup``,
    ``dup_name``, ``survivor``, ``survivor_name``, ``handled`` (from
    :func:`handled`) and :func:`snapshot_params`.
    """
    fold = [
        # A survivor that had no provenance does not acquire an empty list.
        "SET s.source_chunk_ids = CASE WHEN size(coalesce(d.source_chunk_ids, [])) = 0 "
        "THEN s.source_chunk_ids ELSE coalesce(s.source_chunk_ids, []) + "
        "[c IN d.source_chunk_ids WHERE NOT c IN coalesce(s.source_chunk_ids, [])] END",
        *(f"SET {item}" for item in sets),
        *(f"SET s:`{sanitize_cypher_label(label)}`" for label in labels),
    ]
    return (
        _ABSORB_HEAD
        + "".join(_move_block(t, d) for t, d in kinds)
        + f"FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | {' '.join(fold)}) "
        + _TICK
        + "FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | DETACH DELETE d) "
        + _RETURN
    )


def rename_query(*, with_vector: bool) -> str:
    """The fenced query that applies one ``rename`` operation.

    The survivor takes the judge's name ``$name`` and its old name joins its
    aliases (the new one leaves them). Its name vector is replaced by
    ``$vector``, or removed when the new name could not be embedded — the
    entity is then pending again, and the next run embeds and checks it.
    ``$apply`` is false when none of the group's absorbs was applied: the
    operation is then ticked without renaming anything.
    """
    vector = "SET s.embedding = vecf32($vector)" if with_vector else "SET s.embedding = null"
    return (
        _TURN + f"OPTIONAL MATCH (g:{PROCESSING_LABEL}) WHERE my_turn AND g.eid = $survivor "
        f"WITH {_TICKED}, count(g) = 0 AS kept "
        "OPTIONAL MATCH (s:__Entity__ {id: $survivor}) "
        "WHERE my_turn AND kept AND s.name = $survivor_name "
        f"WITH {_TICKED}, head(collect(s)) AS s "
        f"WITH {_TICKED}, s, s IS NOT NULL AS present, true AS fresh, 0 AS unhandled, true AS ok "
        "FOREACH (_ IN CASE WHEN present AND $apply THEN [1] ELSE [] END | "
        "SET s.aliases = reduce(acc = [], a IN coalesce(s.aliases, []) + [s.name] | "
        "CASE WHEN a = $name OR a IN acc THEN acc ELSE acc + [a] END) "
        f"SET s.name = $name {vector}) " + _TICK + _RETURN
    )


#: Tick an operation as ``changed`` without applying it: an absorb whose read
#: kept going out of date. Its group is judged again next run.
SKIP_QUERY = (
    _TURN
    + f"WITH {_TICKED}, false AS present, true AS fresh, 0 AS unhandled, false AS ok "
    + _TICK
    + _RETURN
)
