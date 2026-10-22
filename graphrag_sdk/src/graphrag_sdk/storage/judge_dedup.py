# GraphRAG SDK — Storage: judged cross-document deduplication
#
# "Similarity nominates, the judge decides." Runs at finalize() time, after the
# exact-name phase, over the entities whose dedup check is pending (see
# ``deduplicator``) — never over the whole graph:
#
#   1. nominate candidates with the graph's own ``__Entity__`` vector index:
#      each pending entity's nearest names among ALL entities, searched from
#      the name vector stored on it (the same vector retrieval uses). Cost is
#      per pending entity, not per graph size.
#   2. group nominated pairs into small dense sets (cap 8)
#   3. ask the judge — any ``DedupJudge``; ``LLMJudge`` by default — ONCE to
#      partition each set into same-referent groups. Its answer is checked
#      here, whatever the judge: a set whose answer is not a clean partition
#      is asked again.
#   4. hand the verdicts back. This module decides *identity* only and never
#      writes a merge: ``EntityDeduplicator`` resolves the verdicts of the
#      whole run into a sealed merge plan and applies it under its own rules
#      (survivor rank, property carry, provenance union, descriptions list,
#      aliases, the label union).
#
# Pairs the caller has already settled — a mention two keyed rows could
# equally own, and any two rows written from a declared key (``is_stub``) — are
# dropped before the judge is asked and never shown to it as one set.

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from itertools import combinations
from numbers import Integral
from typing import Any

from graphrag_sdk.core.models import (
    JudgeAttribute,
    JudgeEntity,
    JudgeGroup,
    JudgeSet,
    JudgeType,
    SetVerdict,
)
from graphrag_sdk.storage.dedup_judges.base import DedupJudge

logger = logging.getLogger(__name__)

#: A nearest name: ``(entity id, cosine similarity, name)``.
Hit = tuple[str, float, Any]

# Index-nominated neighbours per pending entity, and the cosine floor for them.
# Measured on the labeled set: k=5 found 92-96 % of true pairs; k=10 doubled
# the judge calls for no F1 gain.
NEIGHBOUR_K = 5
NEIGHBOUR_GATE = 0.5
# Pending entities per batched vector-index query, and the most neighbour rows
# one query may return: a wide lookup (k=1000) asks for fewer entities at once,
# so every query stays far under FalkorDB's 1 s default timeout.
KNN_BATCH = 250
KNN_ROWS_PER_QUERY = 2_500
#: A set without a usable answer is asked again this many times (``retry=True``).
REASKS = 2
#: Internal node ids per window of the scan for entities without a name vector.
SCAN_WINDOW = 10_000
#: How long a run waits for a vector index FalkorDB is still building.
INDEX_READY_TIMEOUT_S = 300.0
GROUP_CAP = 8
MIN_DENSITY = 0.5


# ── pure helpers (unit-testable without a graph) ────────────────────────────


class UnionFind:
    """Union-find over hashable items, with path halving. An item first seen
    by :meth:`find` is its own root."""

    def __init__(self, items: Iterable[Hashable] = ()) -> None:
        self.p: dict[Hashable, Hashable] = {x: x for x in items}

    def find(self, x: Hashable) -> Hashable:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: Hashable, b: Hashable) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[rb] = ra
        return True


def make_groups_dense(
    edges: dict[tuple[int, int], float],
    cap: int = GROUP_CAP,
    min_density: float = MIN_DENSITY,
    keep_apart: Callable[[int, int], bool] | None = None,
) -> list[list[int]]:
    """Connected components under a size cap that must stay dense: more than
    ``min_density`` of all member pairs are nominated edges. The bound is
    exclusive so a chain A-B-C-D where only neighbours are similar — three
    edges of six possible, exactly 0.5 — cannot form; a real alias cluster
    can. Edges left straddling two groups are kept as 2-member sets, so two
    sets may share a member but never a pair.

    ``keep_apart(i, j)`` names pairs the caller protects (two keyed rows, a
    mention two rows could own); two components are not joined when that
    would put such a pair in one set, so the judge is never shown — and never
    asked about — such a pair.
    """
    nodes = sorted({x for e in edges for x in e})
    uf = UnionFind(nodes)
    mem = {x: {x} for x in nodes}
    eset = set(edges)
    for (a, b), _ in sorted(edges.items(), key=lambda kv: -kv[1]):
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        m = mem[ra] | mem[rb]
        if len(m) > cap:
            continue
        if keep_apart is not None and any(keep_apart(x, y) for x in mem[ra] for y in mem[rb]):
            continue
        ml = sorted(m)
        inner = sum(
            1 for i in range(len(ml)) for j in range(i + 1, len(ml)) if (ml[i], ml[j]) in eset
        )
        if inner / (len(ml) * (len(ml) - 1) / 2) <= min_density:
            continue
        uf.union(a, b)
        r = uf.find(a)
        mem[r] = m
        for x in (ra, rb):
            if x != r:
                mem.pop(x, None)
    groups: dict[int, list[int]] = defaultdict(list)
    for x in nodes:
        groups[uf.find(x)].append(x)
    sets = [sorted(g) for g in groups.values() if len(g) >= 2]
    sets += [sorted(e) for e in edges if uf.find(e[0]) != uf.find(e[1])]
    return sets


def pick_group_name(
    group: list[int], names: dict[frozenset[int], str], ents: list[dict]
) -> str | None:
    """The judge's name for a merged group: the one given to the largest
    judged sub-group, kept only if it is one of the members' own names or
    aliases (returned in that exact spelling) — never a name the judge made up."""
    members = set(group)
    proposals = sorted(
        ((len(sub), name) for sub, name in names.items() if sub <= members), reverse=True
    )
    known: dict[str, str] = {}
    for i in group:
        for n in [ents[i].get("name"), *(ents[i].get("aliases") or [])]:
            if isinstance(n, str) and n.strip():
                known.setdefault(n.strip().casefold(), n.strip())
    for _, name in proposals:
        hit = known.get(name.casefold())
        if hit:
            return hit
    return None


# ── the schema the judge sees ───────────────────────────────────────────────


@dataclass(frozen=True)
class AttributeField:
    """One declared attribute of one type, as the judge sees it."""

    #: The node property its value is stored under: the attribute's own name
    #: for an extracted value, the table's signed name (``hr__role``) for a
    #: column a mapping writes.
    key: str
    #: Its name in the prompt: ``role``, or ``role (hr)`` for a signed column.
    shown: str
    type: str = "STRING"
    description: str = ""


@dataclass
class AttributeSchema:
    """The ontology's entity types and their attributes, for the judge.

    Built once per run from the ontology; empty without one, and then the
    judge sees what it saw before attributes were shown.
    """

    #: label -> (type description, its attributes).
    types: dict[str, tuple[str, list[AttributeField]]] = field(default_factory=dict)

    @classmethod
    def from_ontology(cls, ontology: Any) -> AttributeSchema:
        if ontology is None:
            return cls()
        from graphrag_sdk.core.tables import RESERVED_PROPERTY_NAMES

        tables: dict[str, list[Any]] = defaultdict(list)
        for mapping in getattr(ontology, "tables", None) or []:
            tables[mapping.label].append(mapping)
        types: dict[str, tuple[str, list[AttributeField]]] = {}
        for entity in getattr(ontology, "entities", None) or []:
            fields: list[AttributeField] = []
            for attr in entity.properties:
                if attr.name in RESERVED_PROPERTY_NAMES:
                    continue
                desc = attr.description or ""
                fields.append(AttributeField(attr.name, attr.name, attr.type, desc))
                for mapping in tables.get(entity.label, []):
                    if attr.name in mapping.typed_properties:
                        fields.append(
                            AttributeField(
                                mapping.signed_name(attr.name),
                                f"{attr.name} ({mapping.signature})",
                                attr.type,
                                desc,
                            )
                        )
            types[entity.label] = (entity.description or "", fields)
        return cls(types)

    @property
    def keys(self) -> list[str]:
        """Every node property a declared attribute is stored under."""
        return sorted({f.key for _, fields in self.types.values() for f in fields})

    def fields(self, labels: Iterable[str]) -> list[AttributeField]:
        """The attributes of every type in ``labels``, each node property once:
        a node merged under two types has the attributes of both."""
        out: dict[str, AttributeField] = {}
        for label in labels:
            for f in self.types.get(label, ("", []))[1]:
                out.setdefault(f.key, f)
        return list(out.values())

    def judge_types(self, labels: Iterable[str]) -> list[JudgeType]:
        """The declared types among ``labels``, with their attributes, as the
        judge sees them."""
        out: list[JudgeType] = []
        for label in sorted(set(labels)):
            desc, fields = self.types.get(label, ("", []))
            if fields:
                attrs = [
                    JudgeAttribute(name=f.shown, type=f.type, description=f.description)
                    for f in fields
                ]
                out.append(JudgeType(label=label, description=desc, attributes=attrs))
        return out

    def values(
        self, ent: dict[str, Any], stored: dict[str, Any]
    ) -> list[tuple[AttributeField, Any]]:
        """``(attribute, value)`` of each declared attribute ``ent`` holds."""
        return [
            (f, stored[f.key])
            for f in self.fields(ent.get("labels") or [])
            if stored.get(f.key) not in (None, "", [])
        ]


# ── the graph reads ─────────────────────────────────────────────────────────


class NameNeighbourIndex:
    """Entity reads the dedup phases share, all through indexes: rows by id,
    the entities without a name vector (by node-id window), and each entity's
    nearest names from the ``__Entity__.embedding`` vector index, searched from
    the vector stored on it. Nothing here reads every entity.

    Args:
        graph_store: object with ``query_raw(cypher, params)``.
    """

    def __init__(self, graph_store: Any) -> None:
        self._graph = graph_store

    _ENTITY_FIELDS = (
        "{v}.id, {v}.name, {v}.description, "
        "[l IN labels({v}) WHERE l <> '__Entity__'], {v}.aliases, "
        "{v}.descriptions, {v}.is_stub"
    )

    @staticmethod
    def _entity_from_row(row: list[Any]) -> dict | None:
        """A judge row, or ``None`` for a nameless node. A node without a name
        — a fact row keyed on a reading id — cannot be judged to be anything,
        and manufacturing a name from its id would let ``e-1`` nominate against
        a prose ``E-1``."""
        eid, name, desc, labels, aliases, descs, is_stub = (list(row) + [None] * 7)[:7]
        if not name or not str(name).strip():
            return None
        return {
            "id": eid,
            "name": name,
            "description": desc or "",
            "descriptions": list(descs) if descs else [],
            "labels": list(labels or []),
            "label": (labels or [""])[0] if labels else "",
            "aliases": list(aliases or []),
            "is_stub": is_stub,
        }

    async def fetch_by_ids(self, ids: list[str], batch_size: int) -> list[dict]:
        """Named entities among ``ids`` that still exist, in ``ids`` order."""
        found: dict[str, dict] = {}
        for k in range(0, len(ids), batch_size):
            r = await self._graph.query_raw(
                "UNWIND $ids AS id MATCH (e:__Entity__ {id: id}) "
                f"RETURN {self._ENTITY_FIELDS.format(v='e')}",
                {"ids": ids[k : k + batch_size]},
            )
            for row in r.result_set or []:
                ent = self._entity_from_row(row)
                if ent is not None:
                    found[ent["id"]] = ent
        return [found[i] for i in ids if i in found]

    async def attribute_values(
        self, ids: list[str], keys: list[str], batch_size: int
    ) -> dict[str, dict[str, Any]]:
        """``{id: {node property: value}}`` of the properties in ``keys`` each
        entity holds."""
        out: dict[str, dict[str, Any]] = {}
        if not keys:
            return out
        for k in range(0, len(ids), batch_size):
            r = await self._graph.query_raw(
                "UNWIND $ids AS id MATCH (e:__Entity__ {id: id}) "
                "RETURN e.id, [k IN $keys WHERE e[k] IS NOT NULL | [k, e[k]]]",
                {"ids": ids[k : k + batch_size], "keys": keys},
            )
            for eid, pairs in r.result_set or []:
                out.setdefault(eid, {}).update((key, value) for key, value in pairs or [])
        return out

    async def unembedded(self) -> AsyncIterator[list[tuple[str, str | None]]]:
        """``(id, name)`` of every entity without a name vector, one node-id
        window at a time.

        A window is a range scan of internal node ids — no sort, so a page
        costs the same at the end of a large graph as at its start (1M
        entities in 1.1 s, against 3.2 s for the *first* ``ORDER BY e.id``
        page). Ids are reused after deletes, so windows do not follow creation
        order; an entity created behind the scan is found by the next run.
        """
        r = await self._graph.query_raw("MATCH (e:__Entity__) RETURN max(ID(e))")
        top = (r.result_set or [[None]])[0][0]
        if top is None:
            return
        for lo in range(0, int(top) + 1, SCAN_WINDOW):
            r = await self._graph.query_raw(
                "MATCH (e:__Entity__) WHERE ID(e) >= $lo AND ID(e) < $hi "
                "AND e.embedding IS NULL RETURN e.id, e.name",
                {"lo": lo, "hi": lo + SCAN_WINDOW},
            )
            rows = [(row[0], row[1]) for row in r.result_set or [] if row and row[0]]
            if rows:
                yield rows

    async def ensure_vector_index(self, dim: int | None) -> None:
        """The ``__Entity__.embedding`` vector index the neighbour query needs,
        ready to answer. ``ingest()`` creates it; here it may not exist yet on
        a graph written another way, and is created for vectors of ``dim``
        (``None``: only waited for). Looked up first, so an existing index is
        not re-created (the connection would log that refusal as an error).

        FalkorDB builds an index over existing vectors in the background, and
        a query until it is ``OPERATIONAL`` fails ("Invalid arguments") — over
        100k vectors that took more than two minutes — so this waits for it,
        up to ``INDEX_READY_TIMEOUT_S``. Vectors written into a ready index are
        searchable at once.
        """
        deadline = time.monotonic() + INDEX_READY_TIMEOUT_S
        created = False
        while True:
            try:
                r = await self._graph.query_raw(
                    "CALL db.indexes() YIELD label, types, status "
                    "WHERE label = '__Entity__' RETURN types, status"
                )
                rows = getattr(r, "result_set", None) or []
            except Exception as exc:
                logger.debug("judge dedup: could not read the indexes: %s", exc)
                return
            vector = [
                str(status or "")
                for types, status in rows
                if "VECTOR" in str((types or {}).get("embedding", ""))
            ]
            if vector and all(s.upper() == "OPERATIONAL" for s in vector):
                return
            if not vector and dim is None:
                return
            if not vector and not created:
                created = True
                try:
                    await self._graph.query_raw(
                        "CREATE VECTOR INDEX FOR (n:__Entity__) ON (n.embedding) "
                        f"OPTIONS {{dimension:{int(dim)}, similarityFunction:'cosine'}}"
                    )
                except Exception as exc:
                    if "already indexed" not in str(exc).lower():
                        logger.warning("judge dedup: vector index not created: %s", exc)
                        return
                continue
            if time.monotonic() > deadline:
                logger.warning(
                    "The entity vector index is still being built after %d s; nearest-name "
                    "lookups may fail, and the entities they miss stay pending",
                    int(INDEX_READY_TIMEOUT_S),
                )
                return
            await asyncio.sleep(1.0)

    async def discover(self, ids: list[str], k: int) -> tuple[dict[str, list[Hit]], set[str]]:
        """Each entity's ``k`` nearest names, as ``{id: [(neighbour id,
        cosine, neighbour name)]}``, nearest first; and the ids whose search
        failed.

        Searched from the vector stored on the entity, so nothing is embedded
        or sent. One row per entity — every neighbour collected into it — so a
        wide lookup stays far under FalkorDB's 10,000-row result cap; an empty
        list is a completed search that found nothing. The index returns the
        entity itself as its own nearest name, hence ``k + 1``.
        """
        found: dict[str, list[Hit]] = {}
        failed: set[str] = set()
        per_query = max(1, min(KNN_BATCH, KNN_ROWS_PER_QUERY // (k + 1)))
        for b in range(0, len(ids), per_query):
            batch = ids[b : b + per_query]
            try:
                r = await self._graph.query_raw(
                    "UNWIND $ids AS id MATCH (e:__Entity__ {id: id}) "
                    "WHERE e.embedding IS NOT NULL "
                    "CALL db.idx.vector.queryNodes('__Entity__', 'embedding', $k, e.embedding) "
                    "YIELD node, score "
                    "RETURN e.id, "
                    "collect(CASE WHEN node.id <> e.id THEN [node.id, score, node.name] END)",
                    {"ids": batch, "k": k + 1},
                )
            except Exception as exc:
                logger.warning("Nearest-name lookup failed for %d entities: %s", len(batch), exc)
                failed.update(batch)
                continue
            for eid, hits in r.result_set or []:
                # The index reports cosine distance.
                found.setdefault(eid, []).extend(
                    (nid, 1.0 - float(score), name) for nid, score, name in hits or [] if nid
                )
        for hits in found.values():
            hits.sort(key=lambda hit: -hit[1])
        return found, failed


# ── the judge ───────────────────────────────────────────────────────────────


@dataclass
class Verdicts:
    """What the judge decided about one run's sets, by entity index.

    ``same`` pairs were put in one group; ``different`` pairs were in one set
    and put apart. ``names`` is the judge's name per agreed group, ``keeps``
    its picks (attribute, casefolded -> the entity whose value is right).
    ``failed`` are the members of the sets that got no usable answer after
    ``REASKS`` further attempts each; ``unanswered`` counts those sets.
    ``calls`` counts the judge's calls, ``failed_calls`` those that answered
    nothing: they raised, or returned a list of another length. ``usage`` is
    what the judge reports it spent on these sets (see
    :meth:`DedupJudge.usage`).
    """

    same: set[tuple[int, int]] = field(default_factory=set)
    different: set[tuple[int, int]] = field(default_factory=set)
    names: dict[frozenset[int], str] = field(default_factory=dict)
    keeps: dict[frozenset[int], dict[str, int]] = field(default_factory=dict)
    failed: set[int] = field(default_factory=set)
    unanswered: int = 0
    calls: int = 0
    failed_calls: int = 0
    usage: dict[str, int] = field(default_factory=dict)

    def record(self, members: list[int], groups: list[JudgeGroup]) -> None:
        """Add one set's clean verdict; ``members`` are its entity indices, by
        position. A member no group lists is a group of its own."""
        parts = [[members[m] for m in group.members] for group in groups]
        grouped = {i for part in parts for i in part}
        partition = parts + [[i] for i in members if i not in grouped]
        for g, part in enumerate(partition):
            self.same.update((min(a, b), max(a, b)) for a, b in combinations(part, 2))
            for other in partition[g + 1 :]:
                self.different.update((min(a, b), max(a, b)) for a in part for b in other)
        for group, part in zip(groups, parts):
            key = frozenset(part)
            if group.name and group.name.strip():
                self.names[key] = group.name.strip()
            picks = {
                attr.strip().casefold(): members[m]
                for attr, m in group.keep.items()
                if m in group.members
            }
            if picks:
                self.keeps[key] = picks


def candidate_sets(
    pending: list[dict],
    neighbours: dict[str, list[tuple[str, float]]],
    rows: dict[str, dict],
    *,
    extra_pairs: list[tuple[str, str]],
    protected: Callable[[dict, dict], bool],
) -> tuple[list[dict], list[list[int]], int]:
    """``(entities, sets, candidate pairs)`` for one run.

    ``entities`` are the pending rows, then every neighbour row (from
    ``rows``) they nominated; sets index into it. ``extra_pairs`` are id
    pairs put to the judge whatever their similarity — the same-name pairs
    the exact phase declined to merge on spelling alone. A ``protected``
    pair is never a candidate and never shown in one set.
    """
    ents: list[dict] = list(pending)
    index = {e["id"]: i for i, e in enumerate(ents)}

    def at(eid: str) -> int | None:
        if eid not in index:
            if eid not in rows:
                return None
            index[eid] = len(ents)
            ents.append(rows[eid])
        return index[eid]

    edges: dict[tuple[int, int], float] = {}
    for ent in pending:
        i = index[ent["id"]]
        for nid, sim in neighbours.get(ent["id"], []):
            j = at(nid)
            if j is not None and j != i:
                a, b = min(i, j), max(i, j)
                edges[(a, b)] = max(sim, edges.get((a, b), -1.0))
    for x, y in extra_pairs:
        i, j = at(x), at(y)
        if i is not None and j is not None and i != j:
            edges[(min(i, j), max(i, j))] = 1.0

    def protected_idx(i: int, j: int) -> bool:
        return protected(ents[i], ents[j])

    edges = {p: v for p, v in edges.items() if not protected_idx(*p)}
    if not edges:
        return ents, [], 0
    return ents, make_groups_dense(edges, keep_apart=protected_idx), len(edges)


def judge_sets(sets: list[list[int]], ents: list[dict], schema: AttributeSchema) -> list[JudgeSet]:
    """The sets as the judge receives them: each member's name, labels, other
    names, description and declared attribute values (``ents[i]["values"]``),
    with the declared types of the set's members."""
    shown: dict[int, JudgeEntity] = {}
    out: list[JudgeSet] = []
    for members in sets:
        for i in members:
            if i not in shown:
                shown[i] = _judge_entity(ents[i])
        labels = {label for i in members for label in shown[i].labels}
        out.append(JudgeSet(entities=[shown[i] for i in members], types=schema.judge_types(labels)))
    return out


def _judge_entity(ent: dict[str, Any]) -> JudgeEntity:
    """One entity row as the judge sees it: its other names without its own,
    and the declared attribute values it holds."""
    name = str(ent.get("name") or "")
    aliases = dict.fromkeys(a.strip() for a in ent.get("aliases") or [] if isinstance(a, str))
    return JudgeEntity(
        name=name,
        labels=[str(label) for label in ent.get("labels") or [ent.get("label")] if label],
        aliases=[a for a in aliases if a and a != name.strip()],
        description=str(ent.get("description") or ""),
        attributes=dict(ent.get("values") or []),
    )


async def ask_judge(
    judge: DedupJudge, sets: list[list[int]], ents: list[dict], schema: AttributeSchema
) -> Verdicts:
    """Ask ``judge`` about every set and check each answer; a set without a
    usable one is asked again (``retry=True``), up to ``REASKS`` times.
    Nothing is written."""
    asked = judge_sets(sets, ents, schema)
    verdicts = Verdicts()
    before = counted_usage(judge)
    todo = list(range(len(sets)))
    for attempt in range(1 + REASKS):
        if not todo:
            break
        verdicts.calls += 1
        try:
            answers = await judge.judge([asked[s] for s in todo], retry=attempt > 0)
            if len(answers) != len(todo):
                raise ValueError(f"{len(answers)} verdicts for {len(todo)} sets")
        except Exception as exc:
            logger.warning("judge dedup: the judge answered none of %d sets: %s", len(todo), exc)
            verdicts.failed_calls += 1
            answers = [None] * len(todo)
        unanswered: list[int] = []
        for s, answer in zip(todo, answers):
            groups = _usable(answer, len(sets[s]))
            if groups is None:
                unanswered.append(s)
            else:
                verdicts.record(sets[s], groups)
        todo = unanswered
    for s in todo:
        verdicts.failed.update(sets[s])
    verdicts.unanswered = len(todo)
    after = counted_usage(judge, warn=True)
    verdicts.usage = {key: n - before.get(key, 0) for key, n in after.items()}
    return verdicts


def counted_usage(judge: DedupJudge, *, warn: bool = False) -> dict[str, int]:
    """A copy of ``judge.usage()``, whole numbers only: ``judge_stats`` holds
    counts, and a fractional value (``cost_usd``) made ``FinalizeResult``
    raise at the end of ``finalize()``. A copy, because a judge may return
    its live counters, which would read as nothing spent. ``warn`` logs what
    is left out."""
    try:
        usage = judge.usage()
    except Exception:
        logger.warning(
            "judge dedup: the judge's usage() failed; its counts are left out", exc_info=True
        )
        return {}
    if not isinstance(usage, Mapping):
        return {}
    kept = {
        k: int(v) for k, v in usage.items() if isinstance(v, Integral) and not isinstance(v, bool)
    }
    if warn and len(kept) < len(usage):
        logger.warning(
            "judge dedup: usage() values that are not whole numbers are left out of "
            "judge_stats: %s",
            ", ".join(sorted(set(usage) - set(kept))),
        )
    return kept


def _usable(answer: Any, size: int) -> list[JudgeGroup] | None:
    """The groups of ``answer`` when it is a clean verdict about a set of
    ``size`` members, else ``None``.

    Groups that overlap — ``1, 2`` and ``1, 3`` — would merge 2 and 3 through
    1 though the judge never put them together; a member the set does not
    have, or a one-member group carrying a name or ``KEEP`` pick (the name or
    values of a merged entity, given to an entity left alone), says the judge
    lost track of the set. Each makes the whole set unanswered, so it is
    asked again rather than half-read. A one-member group with neither says
    what leaving the member out says.
    """
    if answer is None:
        return None
    if not isinstance(answer, SetVerdict):
        logger.warning("judge dedup: a %s is not a verdict", type(answer).__name__)
        return None
    groups: list[JudgeGroup] = []
    seen: set[int] = set()
    for group in answer.groups:
        members = set(group.members)
        why = None
        if len(members) < len(group.members):
            why = "a member listed twice"
        elif not all(0 <= m < size for m in members):
            why = "a member number the set does not have"
        elif members & seen:
            why = "overlapping groups"
        elif len(members) == 1 and (group.name or group.keep):
            why = "a merged name given to a one-member group"
        if why is not None:
            logger.warning("judge dedup: a set answered with %s has no verdict", why)
            return None
        seen |= members
        if len(members) > 1:
            groups.append(group)
    return groups
