"""A dedup run's decisions and merges, simulated in memory.

:func:`simulate` drives the production code over rows of entities — the exact
phase, the declared-label adoption, the judge's candidate sets and their
resolution, the plan — then applies each planned operation the way its query
does: :meth:`EntityDeduplicator._fold` computes the write from the two nodes as
they stand, and the harness applies it to in-memory nodes, so a later merge of
the same group builds on an earlier one. No database: the merge query itself is
covered on a real FalkorDB by ``test_dedup_records``, and the whole run by
``test_dedup_run``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from graphrag_sdk.core.models import JudgeGroup, SetVerdict
from graphrag_sdk.storage.dedup_judges import DedupJudge, LLMJudge
from graphrag_sdk.storage.deduplicator import EntityDeduplicator, _entity_from_row, _Run
from graphrag_sdk.storage.judge_dedup import AttributeSchema
from graphrag_sdk.storage.merge_query import MergeRead
from graphrag_sdk.storage.system_store import ABSORB, Decisions, PlanOp


@dataclass
class Write:
    """One applied absorb: the plan operation and what ``_fold`` wrote."""

    op: PlanOp
    sets: list[str]
    labels: list[str]
    params: dict[str, Any]
    collisions: list[Any] = field(default_factory=list)


@dataclass
class Sim:
    merged: int
    writes: list[Write]
    renames: list[PlanOp]
    nodes: dict[str, dict[str, Any]]
    ops: list[PlanOp]
    decisions: Decisions
    dedup: EntityDeduplicator
    run: _Run
    prompts: list[str] = field(default_factory=list)

    @property
    def groups(self) -> list[list[str]]:
        """Each planned group's members, sorted, survivor or not."""
        by: dict[int, set[str]] = {}
        for op in self.ops:
            by.setdefault(op.group, set()).update(x for x in (op.survivor, op.dup) if x)
        return sorted(sorted(g) for g in by.values())


def _node(row: list[Any], stored: dict[str, Any]) -> dict[str, Any]:
    eid, name, desc, label = (list(row) + [None] * 4)[:4]
    aliases = row[4] if len(row) > 4 else None
    is_stub = row[5] if len(row) > 5 else None
    labels = list(label) if isinstance(label, list | tuple) else ([label] if label else [])
    props: dict[str, Any] = {"id": eid}
    if name is not None:
        props["name"] = name
    if desc:
        props["description"] = desc
    if aliases:
        props["aliases"] = list(aliases)
    if is_stub is not None:
        props["is_stub"] = is_stub
    props.update(stored.get("props", {}))
    vector = row[7] if len(row) > 7 else stored.get("embedding", [1.0, 0.0])
    return {
        "props": props,
        "labels": stored.get("labels", labels),
        "degree": row[6] if len(row) > 6 else 0,
        "embedding": vector,
    }


class SimGraph:
    """Answers the three entity reads a run makes outside its records: rows
    by id (both projections) and the nearest-name search, by cosine over the
    nodes' vectors."""

    def __init__(self, nodes: dict[str, dict[str, Any]]) -> None:
        self.nodes = nodes

    def _entity_row(self, eid: str) -> list[Any]:
        n = self.nodes[eid]
        p = n["props"]
        labels = n["labels"]
        return [
            eid,
            p.get("name"),
            p.get("description"),
            labels[0] if labels else None,
            p.get("aliases"),
            p.get("is_stub"),
            n["degree"],
            labels,
            p.get("merged_labels"),
            bool(n["embedding"]),
        ]

    async def query_raw(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        params = params or {}
        ids = [i for i in params.get("ids", []) if i in self.nodes]
        if "db.idx.vector.queryNodes" in cypher:
            out = []
            for eid in ids:
                v = np.asarray(self.nodes[eid]["embedding"] or [], dtype=float)
                if not v.size:
                    continue
                scored = []
                for other, n in self.nodes.items():
                    e = np.asarray(n["embedding"] or [], dtype=float)
                    if other == eid or not e.size:
                        continue
                    cos = float(v @ e / (np.linalg.norm(v) * np.linalg.norm(e) or 1.0))
                    scored.append([other, 1.0 - cos, n["props"].get("name")])
                scored.sort(key=lambda hit: hit[1])
                out.append([eid, scored[: params["k"] - 1]])
            return SimpleNamespace(result_set=out)
        if "e[k] IS NOT NULL" in cypher:
            rows = []
            for eid in ids:
                props = self.nodes[eid]["props"]
                rows.append(
                    [eid, [[k, props[k]] for k in params["keys"] if props.get(k) is not None]]
                )
            return SimpleNamespace(result_set=rows)
        if "WITH e RETURN e.id AS id" in cypher:
            return SimpleNamespace(result_set=[self._entity_row(eid) for eid in ids])
        if "RETURN e.id, e.name, e.description" in cypher:
            rows = []
            for eid in ids:
                p = self.nodes[eid]["props"]
                rows.append(
                    [
                        eid,
                        p.get("name"),
                        p.get("description"),
                        self.nodes[eid]["labels"],
                        p.get("aliases"),
                        p.get("descriptions"),
                        p.get("is_stub"),
                    ]
                )
            return SimpleNamespace(result_set=rows)
        raise AssertionError(f"the simulation does not answer: {cypher[:80]}")


class AllSameJudge(DedupJudge):
    """Finds each set's members to be one entity: for tests of how a merge is
    written, whoever decided it. Rows on the simulation's default vector are
    all each other's nearest names, so a test using it holds one entity's
    rows."""

    async def judge(self, sets, *, retry=False):
        return [SetVerdict(groups=[JudgeGroup(members=list(range(len(s.entities))))]) for s in sets]


class AllDifferentJudge(DedupJudge):
    """Finds every member of a set a different entity: for tests of what the
    rules alone do, with nothing judged the same."""

    async def judge(self, sets, *, retry=False):
        return [SetVerdict() for _ in sets]


class SameSpellingJudge(DedupJudge):
    """Finds members one entity when their names are one spelling, ignoring
    case and spacing — what a name rule alone would merge."""

    async def judge(self, sets, *, retry=False):
        verdicts = []
        for s in sets:
            by_name: dict[str, list[int]] = {}
            for i, ent in enumerate(s.entities):
                by_name.setdefault(" ".join(ent.name.casefold().split()), []).append(i)
            groups = [JudgeGroup(members=m) for m in by_name.values() if len(m) > 1]
            verdicts.append(SetVerdict(groups=groups))
        return verdicts


class _Store:
    """The records a judged run writes before its plan: the claim only."""

    run_id = "sim"

    async def pending_pages(self) -> list:
        return []

    async def write_pending(self, pages: Any) -> None:
        return None


def _apply(nodes: dict[str, dict[str, Any]], op: PlanOp, sets, labels, params) -> None:
    """What the absorb query writes, applied to in-memory nodes."""
    s, d = nodes[op.survivor], nodes.pop(op.dup)
    sp, dp = s["props"], d["props"]
    if dp.get("source_chunk_ids"):
        old = list(sp.get("source_chunk_ids") or [])
        sp["source_chunk_ids"] = old + [c for c in dp["source_chunk_ids"] if c not in old]
    for item in sets:
        if item == "s.description = $desc":
            sp["description"] = params["desc"]
        elif item == "s.descriptions = $descs":
            sp["descriptions"] = list(params["descs"])
        elif item == "s.aliases = $aliases":
            sp["aliases"] = list(params["aliases"])
        elif item == "s += $carry":
            sp.update(params["carry"])
        elif item == "s += $override":
            sp.update(params["override"])
        elif item == "s.merged_labels = $merged_labels":
            sp["merged_labels"] = params["merged_labels"]
        else:
            raise AssertionError(f"unknown SET item {item!r}")
    s["labels"] = [*s["labels"], *(lab for lab in labels if lab not in s["labels"])]
    s["degree"] += d["degree"]


async def simulate(
    rows: list[list[Any]],
    *,
    stored: dict[str, dict[str, Any]] | None = None,
    judge: Any = None,
    declared_labels: set[str] | None = None,
    skip_pairs: set[frozenset[str]] | None = None,
    pending: list[str] | None = None,
    graph_class: type | None = None,
    ontology: Any = None,
) -> Sim:
    """Decide and apply one run over ``rows`` — ``[id, name, description,
    label or [labels], aliases, is_stub, degree, embedding]``, trailing
    fields optional — with every row a pending entity. ``stored`` adds, per
    id, ``props`` (``descriptions``, ``merged_labels`` ...), ``labels`` or an
    ``embedding`` the row does not carry. ``judge`` is a ``DedupJudge`` or an
    LLM double, judged through ``LLMJudge``; without it the judged phase is
    left out, so only the exact phase and the adoption decide. ``pending``
    names the rows that are new (the rest are the graph's): only they are
    query points, and the exact phase finds the others through the
    nearest-name search, as a real run does; by default every row is a query
    point, ``dedup_all``-style."""
    stored = stored or {}
    nodes = {row[0]: _node(row, stored.get(row[0], {})) for row in rows}
    graph = (graph_class or SimGraph)(nodes)
    dedup = EntityDeduplicator(graph, None)
    dedup._declared_labels = {label.lower() for label in declared_labels or set()}
    dedup._schema = AttributeSchema.from_ontology(ontology)
    dedup._dedup_all = pending is None
    run = _Run()
    points = list(nodes) if pending is None else list(pending)
    run.points = [_entity_from_row(graph._entity_row(eid)) for eid in points]
    run.listed = set(points)
    for ent in run.points:
        run.rows[ent["id"]] = ent
        if not ent["has_vector"]:
            run.unsearched[ent["id"]] = "no name vector"
    if skip_pairs:
        from graphrag_sdk.storage.identity import NearMiss

        dedup._ambiguous_mentions = [
            NearMiss("", a, b, a, b, "test", bridges_a_declared_source=False)
            for a, b in (sorted(pair) for pair in skip_pairs)
        ]
    await dedup._exact_phase(run, 500)
    if judge is not None:
        judging = judge if isinstance(judge, DedupJudge) else LLMJudge(judge)
        await dedup._judge_phase(_Store(), run, judging, 500)
    ops, decisions = dedup._plan(run)

    writes: list[Write] = []
    renames: list[PlanOp] = []
    for op in ops:
        if op.op == ABSORB:
            s, d = nodes[op.survivor], nodes[op.dup]
            read = MergeRead(
                keep=dict(s["props"]),
                dup=dict(d["props"]),
                keep_labels=list(s["labels"]),
                dup_labels=list(d["labels"]),
                kinds=[],
            )
            fold = dedup._fold(op, read)
            _apply(nodes, op, fold.sets, fold.labels, fold.params)
            writes.append(Write(op, fold.sets, fold.labels, fold.params, fold.collisions))
        else:
            p = nodes[op.survivor]["props"]
            names = [*(p.get("aliases") or []), p.get("name")]
            p["aliases"] = [
                n for i, n in enumerate(names) if n != op.rename_to and n not in names[:i]
            ]
            p["name"] = op.rename_to
            renames.append(op)
    return Sim(
        merged=len(writes),
        writes=writes,
        renames=renames,
        nodes=nodes,
        ops=ops,
        decisions=decisions,
        dedup=dedup,
        run=run,
        prompts=list(getattr(judge, "prompts", []) or []),
    )
