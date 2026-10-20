# GraphRAG SDK — Storage: SDK system nodes
#
# The SDK keeps its own bookkeeping in the data graph, as a small fixed set of
# system nodes under two labels:
#
#   ``__GraphRAGConfig__``      what the data *is* (embedding model, vector size);
#                               it describes the vectors stored beside it, so a copy
#                               or restore of the graph must carry it.
#   ``__GraphRAGProcessing__``  work in progress, told apart by ``kind``.
#
# The processing records live beside the data, not in a graph of their own,
# because a dedup write must check that its run still owns the lock *in the same
# query* that changes entities, and a FalkorDB query cannot span two graphs. One
# writer per graph is what makes that check-and-write atomic.
#
# Rules every system node follows: no edges (traversals and retrieval never reach
# it), no data label and no vector (no index but its own holds it), small, and
# hidden from statistics and from the schema shown to an LLM (``is_system_label``).
#
# Records of ``__GraphRAGProcessing__`` (one ``kind`` each):
#
#   run          the dedup lock: ``run_id`` (current owner), ``owner``,
#                ``started_at``, ``heartbeat_at``. Taken over once stale.
#   pending      entities whose vector is written but whose dedup check has not
#                finished: ``ids[]``, ``attempts[]``, ``reasons[]``, ``claimed_run``.
#   unresolved   entities that failed three judged attempts: left out of grouping,
#                reported, retried only by ``dedup_all``; ``blocked_with[]``.
#   deleted      one per entity an SDK path deleted or re-keyed while a run
#                record exists, so a planned merge never touches a replacement.
#                The only record with ``eid``: a lookup by it is one index seek.
#   plan_header  ``plan_run`` (the run that wrote the plan), ``pages``, ``sealed``.
#   plan         merge operations of one page, with the ``next`` pointer each
#                operation advances in the same query as its effect.
#   plan_ids     the run's decisions per entity (``clear``/``retry``/``refund``/
#                ``handoff``), applied to the pending records once, behind its
#                own ``next``.

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from graphrag_sdk.core.exceptions import GraphRAGError

logger = logging.getLogger(__name__)

CONFIG_LABEL = "__GraphRAGConfig__"
PROCESSING_LABEL = "__GraphRAGProcessing__"
SYSTEM_LABEL_PREFIX = "__GraphRAG"

#: A lock whose heartbeat is older than this belongs to a run that died; the
#: next run takes it over. Well above the slowest judge call, so a live run
#: never looks dead between two heartbeats.
LOCK_TTL_MS = 600_000
HEARTBEAT_INTERVAL_S = 30.0
#: Ids (or operations) per record. A record is one node property write, so this
#: keeps every write well under the 1 s default query timeout.
RECORD_PAGE = 10_000
#: Judged runs an entity may take part in without a usable verdict before it is
#: unresolved: reported and never merged automatically.
MAX_JUDGED_ATTEMPTS = 3

RUN, PENDING, UNRESOLVED, DELETED = "run", "pending", "unresolved", "deleted"
PLAN_HEADER, PLAN, PLAN_IDS = "plan_header", "plan", "plan_ids"

_GUARD = "FOREACH (_ IN CASE WHEN owned THEN [1] ELSE [] END | "


def is_system_label(label: Any) -> bool:
    """Whether ``label`` belongs to an SDK system node, never to user data.

    ``__Entity__`` is a data label (every entity carries it), so the test is the
    ``__GraphRAG`` prefix, not a leading double underscore.
    """
    return isinstance(label, str) and label.startswith(SYSTEM_LABEL_PREFIX)


class DedupRunActiveError(GraphRAGError):
    """Another live ``finalize()`` holds the dedup lock of this graph."""


class OwnershipLostError(GraphRAGError):
    """This run no longer owns the dedup lock: another run took it over."""


# The fence every write of a run starts with. It always yields one row while a
# run record exists, with ``owned`` saying whether it is this run's; a write
# then guards each change on ``owned`` (inside ``FOREACH``, because FalkorDB
# fails the whole query on a SET to a node an OPTIONAL MATCH did not find) and
# returns ``owned`` first. No row at all means no run record: not owned.
FENCE = (
    f"MATCH (_run:{PROCESSING_LABEL} {{kind: '{RUN}'}}) "
    "WITH _run, _run.run_id = $owner_run AS owned "
)


def deleted_record_clause(ids: str) -> str:
    """Cypher appended to an SDK query that deletes or re-keys entities.

    ``ids`` is a Cypher expression for the list of entity ids the query removes.
    While a dedup run record exists — a live run, or a crashed one whose sealed
    plan still waits — each id gets a ``deleted`` record *in the same query as
    the delete*. A planned merge checks for these, so it never acts on an entity
    that was removed, or on a different entity re-created under the same id, after
    it was judged. The clause keeps the row count and needs no lock.
    """
    return (
        f"WITH * OPTIONAL MATCH (_dedup:{PROCESSING_LABEL} {{kind: '{RUN}'}}) "
        f"FOREACH (_eid IN CASE WHEN _dedup IS NULL THEN [] ELSE {ids} END | "
        f"CREATE (:{PROCESSING_LABEL} {{kind: '{DELETED}', eid: _eid}})) "
    )


@dataclass
class PendingPage:
    """One ``pending`` record: entities and their judged attempts so far."""

    page: int
    ids: list[str]
    attempts: list[int]
    reasons: list[str]
    claimed_run: str | None = None


@dataclass
class Unresolved:
    """An entity left out of dedup after :data:`MAX_JUDGED_ATTEMPTS` runs."""

    attempts: int
    reason: str
    blocked_with: str = ""


#: Plan operations. An ``absorb`` folds ``dup`` into ``survivor``; a ``rename``
#: gives ``survivor`` the name ``rename_to``, after its group's absorbs.
ABSORB, RENAME = "absorb", "rename"
#: How an absorb treats labels: ``exact`` (a spelling of the same name — only
#: the labels the duplicate itself absorbed carry over), ``judge`` (the
#: survivor also gains the duplicate's own labels), ``adopt`` (an inferred label
#: corrected by a declared one — no label carries over).
EXACT, JUDGE, ADOPT = "exact", "judge", "adopt"


@dataclass
class PlanOp:
    """One merge-plan operation. ``group`` numbers the planned group the
    operation belongs to: when any of a group's operations finds its entities
    changed, every member of the group is checked again. The names are the ones
    the entities were judged under; an operation whose entity no longer carries
    its name is not applied."""

    op: str
    group: int
    survivor: str
    survivor_name: str
    dup: str = ""
    dup_name: str = ""
    policy: str = EXACT
    rename_to: str = ""
    #: A judged absorb's ``KEEP`` picks, JSON: ``{node property: value}`` the
    #: judge chose for its group where the members' values differ.
    keep: str = ""


_PLAN_COLUMNS = (
    "op",
    "group",
    "survivor",
    "survivor_name",
    "dup",
    "dup_name",
    "policy",
    "rename_to",
    "keep",
)


@dataclass
class Decisions:
    """One ``plan_ids`` page: what the run decided for each pending entity.

    ``clear`` leave pending; ``retry`` stay, with ``reasons`` (aligned) as the
    last failure; ``refund`` stay and get back the attempt the claim charged
    (they were held only because of a neighbour); ``handoff`` become pending
    (a survivor that absorbed an entity whose check is still owed).
    """

    clear: list[str] = field(default_factory=list)
    retry: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    refund: list[str] = field(default_factory=list)
    handoff: list[str] = field(default_factory=list)


_DECISION_COLUMNS = ("clear", "retry", "reasons", "refund", "handoff")


@dataclass
class Plan:
    """A sealed plan as stored: its operation pages and decision pages."""

    plan_run: str
    ops: list[list[PlanOp]] = field(default_factory=list)
    nexts: list[int] = field(default_factory=list)
    changed: list[list[int]] = field(default_factory=list)
    decisions: list[Decisions] = field(default_factory=list)
    decision_done: list[bool] = field(default_factory=list)


def _pages(items: list[Any], size: int = RECORD_PAGE) -> list[list[Any]]:
    return [items[k : k + size] for k in range(0, len(items), size)] or [[]]


class SystemStore:
    """Reads and writes of the ``__GraphRAGProcessing__`` records of one graph.

    A run takes the lock with :meth:`acquire`, writes only through fenced
    queries while it holds it, and gives it up with :meth:`release`. Every fenced
    write raises :class:`OwnershipLostError` once another run has taken over.

    Args:
        graph: object with ``query_raw(cypher, params)`` — the ``GraphStore``.
        ttl_ms: heartbeat age after which a lock counts as abandoned.
    """

    def __init__(self, graph: Any, *, ttl_ms: int = LOCK_TTL_MS) -> None:
        self._graph = graph
        self._ttl_ms = int(ttl_ms)
        self.run_id: str | None = None

    # ── indexes ──────────────────────────────────────────────────

    async def ensure_indexes(self) -> None:
        """Range indexes on ``kind`` and ``eid``.

        Without the ``kind`` index every fenced write finds the run record by
        scanning all system nodes — measured at 4.6 ms per write with 100k
        ``deleted`` records, against 0.3 ms with it. ``eid`` makes the
        deleted-record check one lookup per planned merge.
        """
        try:
            rows = await self._graph.query_raw(
                "CALL db.indexes() YIELD label, properties WHERE label = $label RETURN properties",
                {"label": PROCESSING_LABEL},
            )
            have = {p for (props,) in (rows.result_set or []) for p in (props or [])}
        except Exception:
            logger.debug("Could not read the system-node indexes", exc_info=True)
            have = set()
        for prop in ("kind", "eid"):
            if prop in have:
                continue
            try:
                await self._graph.query_raw(
                    f"CREATE INDEX FOR (n:{PROCESSING_LABEL}) ON (n.{prop})"
                )
            except Exception as exc:
                if "already indexed" not in str(exc).lower():
                    logger.warning("Index on %s.%s not created: %s", PROCESSING_LABEL, prop, exc)

    # ── the lock ─────────────────────────────────────────────────

    async def acquire(self, owner: str = "") -> str | None:
        """Take the dedup lock, or take over one whose heartbeat is stale.

        One query: FalkorDB serialises writes to a graph, so two runs starting
        at once cannot both win. Returns the previous owner's run id when an
        abandoned lock was taken over (its plan, if any, is now this run's to
        finish), else ``None``.

        Raises:
            DedupRunActiveError: a live run holds the lock.
        """
        run_id = uuid4().hex
        rows = await self._graph.query_raw(
            f"MERGE (r:{PROCESSING_LABEL} {{kind: '{RUN}'}}) "
            "ON CREATE SET r.run_id = $run, r.owner = $owner, "
            "r.started_at = timestamp(), r.heartbeat_at = timestamp() "
            "WITH r, r.run_id AS previous "
            "WHERE previous = $run OR r.heartbeat_at < timestamp() - $ttl "
            "SET r.run_id = $run, r.owner = $owner, "
            "r.started_at = timestamp(), r.heartbeat_at = timestamp() "
            "RETURN previous",
            {"run": run_id, "owner": owner, "ttl": self._ttl_ms},
        )
        result = rows.result_set or []
        if not result:
            raise DedupRunActiveError(
                "Another finalize() is deduplicating this graph; wait for it to finish. "
                f"A run that died releases the lock after {self._ttl_ms // 1000} s."
            )
        self.run_id = run_id
        previous = result[0][0]
        return previous if previous != run_id else None

    async def heartbeat(self) -> None:
        """Refresh the lock's heartbeat. Raises once ownership is lost."""
        await self.fenced(f"{_GUARD}SET _run.heartbeat_at = timestamp()) RETURN owned")

    async def release(self) -> None:
        """Give up the lock. A run that already lost it releases nothing."""
        if self.run_id is None:
            return
        try:
            await self.fenced(f"{_GUARD}DELETE _run) RETURN owned")
        except OwnershipLostError:
            logger.warning("Dedup run %s had already lost its lock", self.run_id)
        finally:
            self.run_id = None

    async def expire(self) -> None:
        """Give up the lock but keep its record, aged past the TTL: for a run
        that stops with a sealed plan unfinished. While the record exists,
        delete calls leave ``deleted`` records the plan checks, and the next
        run takes the lock at once and finishes the plan."""
        if self.run_id is None:
            return
        try:
            await self.fenced(f"{_GUARD}SET _run.heartbeat_at = 0) RETURN owned")
        except OwnershipLostError:
            logger.warning("Dedup run %s had already lost its lock", self.run_id)
        except Exception:
            logger.warning("Could not expire the dedup lock; it expires on its own", exc_info=True)
        finally:
            self.run_id = None

    async def fenced(self, tail: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        """Run ``FENCE + tail`` and check ownership.

        ``tail`` continues from ``WITH _run, owned``, guards every change on
        ``owned`` and returns ``owned`` as its first column.

        Raises:
            OwnershipLostError: no run record, or another run owns it.
        """
        if self.run_id is None:
            raise OwnershipLostError("No dedup lock is held")
        result = await self._graph.query_raw(
            FENCE + tail, {**(params or {}), "owner_run": self.run_id}
        )
        rows = result.result_set or []
        if not rows or rows[0][0] is not True:
            raise OwnershipLostError(f"Dedup run {self.run_id} lost its lock to another run")
        return rows

    # ── pending and unresolved ───────────────────────────────────

    async def pending_pages(self) -> list[PendingPage]:
        """Every ``pending`` record, by page."""
        result = await self._graph.query_raw(
            f"MATCH (p:{PROCESSING_LABEL} {{kind: '{PENDING}'}}) "
            "RETURN p.page, p.ids, p.attempts, p.reasons, p.claimed_run ORDER BY p.page"
        )
        return [
            PendingPage(
                page=int(page),
                ids=list(ids or []),
                attempts=[int(a) for a in (attempts or [])],
                reasons=list(reasons or []),
                claimed_run=claimed,
            )
            for page, ids, attempts, reasons, claimed in result.result_set or []
        ]

    async def add_pending(self, ids: list[str]) -> None:
        """List ``ids`` as pending, ``RECORD_PAGE`` per record, attempts 0.

        Called *before* their vectors are written: an entity with a vector and
        no record would look checked, so the record must exist first."""
        for chunk in _pages(ids):
            if not chunk:
                continue
            await self.fenced(
                f"OPTIONAL MATCH (p:{PROCESSING_LABEL} {{kind: '{PENDING}'}}) "
                f"WITH owned, coalesce(max(p.page), -1) + 1 AS page {_GUARD}"
                f"CREATE (:{PROCESSING_LABEL} {{kind: '{PENDING}', page: page, ids: $ids, "
                "attempts: $zeros, reasons: $blank})) RETURN owned",
                {"ids": chunk, "zeros": [0] * len(chunk), "blank": [""] * len(chunk)},
            )

    async def write_pending(self, pages: Iterable[PendingPage]) -> None:
        """Replace the lists of existing ``pending`` records; a page left with
        no ids is deleted. Used by the claim; the run's final clearing goes
        through :meth:`apply_decisions`, which is replay-safe."""
        for rec in pages:
            await self.fenced(
                f"OPTIONAL MATCH (p:{PROCESSING_LABEL} {{kind: '{PENDING}', page: $page}}) "
                "WITH owned, p, owned AND p IS NOT NULL AS found "
                "FOREACH (_ IN CASE WHEN found AND size($ids) > 0 THEN [1] ELSE [] END | "
                "SET p.ids = $ids, p.attempts = $attempts, p.reasons = $reasons, "
                "p.claimed_run = $claimed) "
                "FOREACH (_ IN CASE WHEN found AND size($ids) = 0 THEN [1] ELSE [] END | "
                "DELETE p) RETURN owned",
                {
                    "page": rec.page,
                    "ids": rec.ids,
                    "attempts": rec.attempts,
                    "reasons": rec.reasons,
                    "claimed": rec.claimed_run,
                },
            )

    async def unresolved(self) -> dict[str, Unresolved]:
        """Every entity of the ``unresolved`` records, by id."""
        result = await self._graph.query_raw(
            f"MATCH (u:{PROCESSING_LABEL} {{kind: '{UNRESOLVED}'}}) "
            "RETURN u.ids, u.attempts, u.reasons, u.blocked_with"
        )
        out: dict[str, Unresolved] = {}
        for ids, attempts, reasons, blocked in result.result_set or []:
            for eid, tries, reason, with_ in zip(
                ids or [], attempts or [], reasons or [], blocked or []
            ):
                out[eid] = Unresolved(int(tries or 0), str(reason or ""), str(with_ or ""))
        return out

    async def add_unresolved(self, entities: dict[str, Unresolved]) -> None:
        """Record entities that are no longer retried automatically."""
        ids = list(entities)
        for chunk in _pages(ids):
            if not chunk:
                continue
            await self.fenced(
                f"{_GUARD}CREATE (:{PROCESSING_LABEL} {{kind: '{UNRESOLVED}', ids: $ids, "
                "attempts: $attempts, reasons: $reasons, blocked_with: $blocked})) RETURN owned",
                {
                    "ids": chunk,
                    "attempts": [entities[i].attempts for i in chunk],
                    "reasons": [entities[i].reason for i in chunk],
                    "blocked": [entities[i].blocked_with for i in chunk],
                },
            )

    async def set_blocked_with(self, blocked: dict[str, str]) -> None:
        """Replace ``blocked_with`` of the unresolved entities in ``blocked``:
        the entities, ``" | "``-joined, each was kept from being judged with."""
        if not blocked:
            return
        await self.fenced(
            f"OPTIONAL MATCH (u:{PROCESSING_LABEL} {{kind: '{UNRESOLVED}'}}) "
            "WITH owned, collect(u) AS recs "
            "FOREACH (u IN CASE WHEN owned THEN recs ELSE [] END | "
            "SET u.blocked_with = [i IN range(0, size(u.ids) - 1) | "
            "coalesce($blocked[u.ids[i]], u.blocked_with[i])]) RETURN owned",
            {"blocked": blocked},
        )

    async def drop_unresolved(self, ids: Iterable[str]) -> None:
        """Remove ``ids`` from the unresolved records (``dedup_all`` retries them)."""
        drop = list(ids)
        if not drop:
            return
        keep = "[i IN range(0, size(u.ids) - 1) WHERE NOT u.ids[i] IN $drop | u.{col}[i]]"
        await self.fenced(
            f"OPTIONAL MATCH (u:{PROCESSING_LABEL} {{kind: '{UNRESOLVED}'}}) "
            "WITH owned, collect(u) AS recs "
            "FOREACH (u IN CASE WHEN owned THEN recs ELSE [] END | "
            f"SET u.attempts = {keep.format(col='attempts')}, "
            f"u.reasons = {keep.format(col='reasons')}, "
            f"u.blocked_with = {keep.format(col='blocked_with')} "
            "SET u.ids = [i IN u.ids WHERE NOT i IN $drop]) "
            "WITH owned, recs "
            "FOREACH (u IN CASE WHEN owned THEN [x IN recs WHERE size(x.ids) = 0] ELSE [] END | "
            "DELETE u) RETURN owned",
            {"drop": drop},
        )

    # ── deleted ──────────────────────────────────────────────────

    async def deleted_among(self, ids: Iterable[str]) -> set[str]:
        """Which of ``ids`` an SDK path deleted or re-keyed during this run."""
        wanted = list(dict.fromkeys(ids))
        out: set[str] = set()
        for k in range(0, len(wanted), RECORD_PAGE):
            result = await self._graph.query_raw(
                f"UNWIND $ids AS id MATCH (d:{PROCESSING_LABEL} {{eid: id}}) RETURN DISTINCT d.eid",
                {"ids": wanted[k : k + RECORD_PAGE]},
            )
            out.update(row[0] for row in result.result_set or [])
        return out

    async def clear_deleted(self) -> None:
        """Drop every ``deleted`` record: they only mean something to a plan."""
        await self._drop_kinds([DELETED])

    async def _drop_kinds(self, kinds: list[str]) -> None:
        while True:
            rows = await self.fenced(
                f"OPTIONAL MATCH (p:{PROCESSING_LABEL}) WHERE p.kind IN $kinds "
                "WITH owned, p LIMIT $limit "
                "WITH owned, collect(p) AS gone "
                "FOREACH (p IN CASE WHEN owned THEN gone ELSE [] END | DELETE p) "
                "RETURN owned, size(gone)",
                {"kinds": kinds, "limit": RECORD_PAGE},
            )
            if int(rows[0][1] or 0) < RECORD_PAGE:
                return

    # ── the merge plan ───────────────────────────────────────────

    async def write_plan(self, ops: list[PlanOp], decisions: Decisions) -> str:
        """Write this run's plan and seal it; returns its ``plan_run``.

        Nothing has been merged when this is called. Every page is written
        before the header is sealed, so a crash part-way leaves an unsealed
        plan, which the next run deletes and replaces; a sealed plan is applied
        to completion by whichever run finds it."""
        plan_run = self.run_id
        op_pages = _pages(ops)
        for page, chunk in enumerate(op_pages):
            columns = {col: [getattr(o, col) for o in chunk] for col in _PLAN_COLUMNS}
            await self.fenced(
                f"{_GUARD}CREATE (:{PROCESSING_LABEL} {{kind: '{PLAN}', plan_run: $plan_run, "
                "page: $page, next: 0, changed: [], "
                + ", ".join(f"{col}: ${col}" for col in _PLAN_COLUMNS)
                + "})) RETURN owned",
                {"plan_run": plan_run, "page": page, **columns},
            )
        # ``retry`` and ``reasons`` page together: they are aligned.
        retry = _pages(list(zip(decisions.retry, decisions.reasons)))
        columns = {
            "clear": _pages(decisions.clear),
            "retry": [[eid for eid, _ in chunk] for chunk in retry],
            "reasons": [[why for _, why in chunk] for chunk in retry],
            "refund": _pages(decisions.refund),
            "handoff": _pages(decisions.handoff),
        }
        n_dec = max(len(pages) for pages in columns.values())
        for page in range(n_dec):
            lists = {
                col: pages[page] if page < len(pages) else [] for col, pages in columns.items()
            }
            await self.fenced(
                f"{_GUARD}CREATE (:{PROCESSING_LABEL} {{kind: '{PLAN_IDS}', plan_run: $plan_run, "
                "page: $page, next: 0, "
                + ", ".join(f"{col}: ${col}" for col in _DECISION_COLUMNS)
                + "})) RETURN owned",
                {"plan_run": plan_run, "page": page, **lists},
            )
        await self.fenced(
            f"{_GUARD}CREATE (:{PROCESSING_LABEL} {{kind: '{PLAN_HEADER}', plan_run: $plan_run, "
            "pages: $pages, decision_pages: $dpages, sealed: true})) RETURN owned",
            {"plan_run": plan_run, "pages": len(op_pages), "dpages": n_dec},
        )
        return plan_run

    async def sealed_plan(self) -> Plan | None:
        """The sealed plan left by an earlier run (or this one), if any."""
        head = await self._graph.query_raw(
            f"MATCH (h:{PROCESSING_LABEL} {{kind: '{PLAN_HEADER}'}}) WHERE h.sealed = true "
            "RETURN h.plan_run ORDER BY h.plan_run LIMIT 1"
        )
        rows = head.result_set or []
        if not rows:
            return None
        plan = Plan(plan_run=rows[0][0])
        pages = await self._graph.query_raw(
            f"MATCH (p:{PROCESSING_LABEL} {{kind: '{PLAN}', plan_run: $plan_run}}) "
            "RETURN p.next, p.changed, "
            + ", ".join(f"p.{col}" for col in _PLAN_COLUMNS)
            + " ORDER BY p.page",
            {"plan_run": plan.plan_run},
        )
        for nxt, changed, *columns in pages.result_set or []:
            plan.nexts.append(int(nxt or 0))
            plan.changed.append([int(op) for op in changed or []])
            plan.ops.append(
                [
                    PlanOp(**dict(zip(_PLAN_COLUMNS, values)))
                    for values in zip(*(c or [] for c in columns))
                ]
            )
        dec = await self._graph.query_raw(
            f"MATCH (p:{PROCESSING_LABEL} {{kind: '{PLAN_IDS}', plan_run: $plan_run}}) "
            "RETURN p.next, "
            + ", ".join(f"p.{col}" for col in _DECISION_COLUMNS)
            + " ORDER BY p.page",
            {"plan_run": plan.plan_run},
        )
        for nxt, *columns in dec.result_set or []:
            plan.decision_done.append(int(nxt or 0) > 0)
            plan.decisions.append(
                Decisions(**{col: list(c or []) for col, c in zip(_DECISION_COLUMNS, columns)})
            )
        return plan

    async def apply_decisions(self, plan_run: str, page: int, updates: list[PendingPage]) -> bool:
        """Apply one decision page to the pending records, and tick it, in one
        query — so a crash and a replay can never clear or refund twice.

        ``updates`` are the pending records as they must read afterwards
        (computed by the caller from the records and this page's lists); a
        record left with no ids is deleted. Returns ``False`` when the page was
        already applied."""
        rows = await self.fenced(
            f"OPTIONAL MATCH (d:{PROCESSING_LABEL} "
            f"{{kind: '{PLAN_IDS}', plan_run: $plan_run, page: $dpage}}) WHERE owned "
            "WITH owned, d, d IS NOT NULL AND d.next = 0 AS my_turn "
            f"OPTIONAL MATCH (p:{PROCESSING_LABEL} {{kind: '{PENDING}'}}) "
            "WHERE my_turn AND p.page IN $pages "
            "WITH owned, d, my_turn, collect(p) AS recs "
            "FOREACH (p IN recs | "
            "SET p.ids = $ids[toString(p.page)], p.attempts = $attempts[toString(p.page)], "
            "p.reasons = $reasons[toString(p.page)]) "
            "FOREACH (p IN [x IN recs WHERE size($ids[toString(x.page)]) = 0] | DELETE p) "
            "FOREACH (_ IN CASE WHEN my_turn THEN [1] ELSE [] END | SET d.next = 1) "
            "RETURN owned, my_turn",
            {
                "plan_run": plan_run,
                "dpage": page,
                "pages": [u.page for u in updates],
                "ids": {str(u.page): u.ids for u in updates},
                "attempts": {str(u.page): u.attempts for u in updates},
                "reasons": {str(u.page): u.reasons for u in updates},
            },
        )
        return bool(rows[0][1])

    async def drop_plans(self) -> None:
        """Delete every plan record: an applied plan, or an unsealed one a crash
        before sealing left (nothing in it was applied)."""
        await self._drop_kinds([PLAN_HEADER, PLAN, PLAN_IDS])
