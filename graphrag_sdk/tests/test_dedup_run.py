"""Whole dedup runs on a real FalkorDB: records, crashes, takeovers, concurrent
deletes, failed and held checks.

Gated by ``RUN_INTEGRATION=1`` (``FALKOR_HOST`` / ``FALKOR_PORT``). Every test
uses its own graph, dropped afterwards. The decisions themselves (which merge,
which survivor, what the fold writes) are tested in memory by
``test_judge_dedup`` and ``test_deduplicator_queries``; the merge query by
``test_dedup_records``.
"""

from __future__ import annotations

import os
import re
from types import SimpleNamespace
from uuid import uuid4

import pytest

from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.exceptions import DatabaseError
from graphrag_sdk.core.models import Attribute, Entity, GraphNode, GraphRelationship, Ontology
from graphrag_sdk.core.providers.base import LLMBatchItem
from graphrag_sdk.storage.deduplicator import EntityDeduplicator
from graphrag_sdk.storage.graph_store import GraphStore
from graphrag_sdk.storage.system_store import (
    DedupRunActiveError,
    OwnershipLostError,
    SystemStore,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1", reason="Requires RUN_INTEGRATION=1"
)

VECTORS = {
    "Tim Cook": [1.0, 0.0, 0.0, 0.0],
    "Timothy Cook": [0.97, 0.2, 0.0, 0.0],
    "Tim Cooke": [0.95, 0.3, 0.0, 0.0],
    "Apple": [0.0, 1.0, 0.0, 0.0],
    "Apple Inc.": [0.0, 0.98, 0.15, 0.0],
    "Steve Jobs": [0.0, 0.0, 1.0, 0.0],
}


class Names:
    """Embeds a name as its row in ``VECTORS``."""

    async def aembed_documents(self, texts, **kw):
        return [VECTORS.get(t, [0.5, 0.5, 0.5, 0.5]) for t in texts]


class Judge:
    """Answers by name: ``same`` are the name groups that are one entity.
    ``fail`` names an entity whose sets get no answer, every time."""

    def __init__(self, same=(), fail=(), keep=None, name=None):
        self.same = [set(g) for g in same]
        self.fail = set(fail)
        # the name every merged group is given; by default its first member's
        self.name = name
        # attribute -> the name of the member whose value a merged group keeps
        self.keep = dict(keep or {})
        self.prompts: list[str] = []

    async def abatch_invoke(self, prompts, **kw):
        self.prompts.extend(prompts)
        out = []
        for i, p in enumerate(prompts):
            if any(name in p for name in self.fail):
                out.append(LLMBatchItem(index=i, error=RuntimeError("judge down")))
                continue
            lines = []
            for si, block in enumerate(re.split(r"--- Set \d+ ---\n", p)[1:], start=1):
                block = block.split("\n\nDECISION PROCEDURE")[0]
                names = re.findall(r"^(\d+)\. (.+?) \[", block, flags=re.M)
                used: set[str] = set()
                for num, name in names:
                    if num in used:
                        continue
                    grp = [n for n, nm in names if any({name, nm} <= g for g in self.same)]
                    if len(grp) < 2:
                        continue
                    used.update(grp)
                    keep = [
                        f"{attr}={n}"
                        for attr, owner in self.keep.items()
                        for n, nm in names
                        if n in grp and nm == owner
                    ]
                    lines.append(
                        f"SET {si} GROUP: {', '.join(grp)} | NAME: {self.name or name}"
                        + (f" | KEEP: {'; '.join(keep)}" if keep else "")
                    )
                if not any(line.startswith(f"SET {si} ") for line in lines):
                    lines.append(f"SET {si} NONE")
            lines.append("END")
            out.append(LLMBatchItem(index=i, response=SimpleNamespace(content="\n".join(lines))))
        return out


TIM = {"Tim Cook", "Timothy Cook"}
APPLE = {"Apple", "Apple Inc."}


@pytest.fixture
async def graph():
    conn = FalkorDBConnection(
        ConnectionConfig(
            host=os.getenv("FALKOR_HOST", "localhost"),
            port=int(os.getenv("FALKOR_PORT", "6379")),
            username=os.getenv("FALKOR_USERNAME") or None,
            password=os.getenv("FALKOR_PASSWORD") or None,
            graph_name=f"test_dedup_run_{uuid4().hex[:8]}",
        )
    )
    store = GraphStore(conn)
    yield store
    try:
        await store.delete_all()
    finally:
        await conn.close()


async def rows(graph, cypher, params=None):
    return (await graph.query_raw(cypher, params or {})).result_set or []


async def checked_graph(graph):
    """An old, checked Tim Cook and Apple, and new mentions written the way
    ingest writes them: no vector, nothing marked."""
    await graph.query_raw(
        "CREATE (:__Entity__:Person {id: 'tim', name: 'Tim Cook', description: 'Apple CEO', "
        "embedding: vecf32([1.0, 0.0, 0.0, 0.0])}), "
        "(:__Entity__:Organization {id: 'apple', name: 'Apple', description: 'iPhone maker', "
        "embedding: vecf32([0.0, 1.0, 0.0, 0.0])})"
    )
    await graph.upsert_nodes(
        [
            GraphNode(
                id="timothy",
                label="Person",
                properties={"name": "Timothy Cook", "description": "chief executive of Apple"},
            ),
            GraphNode(
                id="apple-inc",
                label="Organization",
                properties={"name": "Apple Inc.", "description": "Cupertino company"},
            ),
            GraphNode(
                id="jobs",
                label="Person",
                properties={"name": "Steve Jobs", "description": "co-founder of Apple"},
            ),
        ]
    )
    await graph.upsert_relationships(
        [
            GraphRelationship(
                start_node_id="timothy",
                end_node_id="apple-inc",
                type="RELATES",
                properties={"rel_type": "LEADS", "source_chunk_ids": ["c1"]},
            ),
            GraphRelationship(
                start_node_id="jobs",
                end_node_id="timothy",
                type="MANAGED",
                properties={"since": 2005},
            ),
        ]
    )


async def entity_ids(graph):
    return [r[0] for r in await rows(graph, "MATCH (e:__Entity__) RETURN e.id ORDER BY e.id")]


async def records(graph):
    return await rows(
        graph,
        "MATCH (n:__GraphRAGProcessing__) RETURN n.kind, n.ids, n.attempts, n.reasons "
        "ORDER BY n.kind",
    )


def dedup(graph):
    return EntityDeduplicator(graph, Names())


# ── a run, start to finish ───────────────────────────────────────


async def test_a_run_checks_only_what_is_new_and_leaves_no_records(graph):
    await checked_graph(graph)
    assert await rows(graph, "MATCH (e:__Entity__ {id: 'timothy'}) RETURN keys(e)") == [
        [["id", "name", "description"]]
    ], "ingest marks nothing on an entity"

    dd = dedup(graph)
    merged = await dd.deduplicate(judge_llm=Judge([TIM, APPLE]))
    assert merged == 2 and dd.last_report.checked == 3 and dd.last_report.pending == 0
    assert await records(graph) == [], "a completed run leaves no system record"
    assert await rows(graph, "MATCH (e:__Entity__) WHERE e.embedding IS NULL RETURN e") == []
    # Every relationship type moved, the custom one included.
    survivor = await rows(
        graph, "MATCH (e:__Entity__:Person) WHERE e.name STARTS WITH 'Tim' RETURN e.id"
    )
    assert len(survivor) == 1
    assert await rows(
        graph,
        "MATCH (:__Entity__ {id: 'jobs'})-[r:MANAGED]->(t) RETURN t.id",
    ) == [[survivor[0][0]]]

    judge = Judge([TIM, APPLE])
    assert await dedup(graph).deduplicate(judge_llm=judge) == 0
    assert judge.prompts == [], "nothing pending: no work"


async def test_an_entity_listed_before_a_crash_is_embedded_and_checked_by_the_next_run(graph):
    """A run lists an entity before it embeds it, so a crash between the two
    leaves a record without a vector: the next run takes the stale lock over,
    embeds the entity under its record, and checks it."""
    await checked_graph(graph)
    store = SystemStore(graph)
    await store.ensure_indexes()
    await store.acquire("crashed")
    await store.add_pending(["timothy"])  # listed; the crash came before its vector
    await graph.query_raw("MATCH (r:__GraphRAGProcessing__ {kind: 'run'}) SET r.heartbeat_at = 0")

    dd = dedup(graph)
    merged = await dd.deduplicate(judge_llm=Judge([TIM, APPLE]))
    assert merged == 2 and await records(graph) == []


# ── crashes and takeovers ────────────────────────────────────────


async def test_a_crash_after_sealing_is_finished_by_the_next_run(graph, monkeypatch):
    await checked_graph(graph)
    real = EntityDeduplicator._apply_absorb
    calls = {"n": 0}

    async def dies_on_the_second(self, *args, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("killed")
        return await real(self, *args, **kw)

    monkeypatch.setattr(EntityDeduplicator, "_apply_absorb", dies_on_the_second)
    with pytest.raises(RuntimeError, match="killed"):
        await dedup(graph).deduplicate(judge_llm=Judge([TIM, APPLE]))
    assert len(await entity_ids(graph)) == 4, "one of the two merges was applied"
    # The lock is left expired, not released: deletes still leave records.
    assert await rows(
        graph, "MATCH (r:__GraphRAGProcessing__ {kind: 'run'}) RETURN r.heartbeat_at"
    ) == [[0]]
    assert await rows(
        graph, "MATCH (h:__GraphRAGProcessing__ {kind: 'plan_header'}) RETURN h.sealed"
    ) == [[True]]

    monkeypatch.setattr(EntityDeduplicator, "_apply_absorb", real)
    dd = dedup(graph)
    merged = await dd.deduplicate(judge_llm=Judge([TIM, APPLE]))
    assert merged == 1 and dd.last_report.resumed == 1, "the second merge, once"
    assert len(await entity_ids(graph)) == 3 and await records(graph) == []


async def test_a_crash_before_sealing_merges_nothing_and_is_judged_again(graph, monkeypatch):
    await checked_graph(graph)

    async def dies(self, *args, **kw):
        raise RuntimeError("killed while judging")

    real = EntityDeduplicator._resolve
    monkeypatch.setattr(EntityDeduplicator, "_resolve", dies)
    with pytest.raises(RuntimeError):
        await dedup(graph).deduplicate(judge_llm=Judge([TIM, APPLE]))
    assert len(await entity_ids(graph)) == 5, "nothing merged"
    (pending,) = [r for r in await records(graph) if r[0] == "pending"]
    assert sorted(pending[1]) == ["apple-inc", "jobs", "timothy"]
    assert sorted(zip(pending[1], pending[2], pending[3])) == [
        ("apple-inc", 1, "interrupted"),
        ("jobs", 0, ""),  # no candidate set: not charged
        ("timothy", 1, "interrupted"),
    ]
    assert await rows(graph, "MATCH (r:__GraphRAGProcessing__ {kind: 'run'}) RETURN r") == []

    monkeypatch.setattr(EntityDeduplicator, "_resolve", real)
    assert await dedup(graph).deduplicate(judge_llm=Judge([TIM, APPLE])) == 2
    assert await records(graph) == []


async def test_a_run_that_lost_its_lock_stops_and_writes_nothing_more(graph, monkeypatch):
    await checked_graph(graph)
    taker = SystemStore(graph)

    async def taken_over(self, *args, **kw):
        await graph.query_raw(
            "MATCH (r:__GraphRAGProcessing__ {kind: 'run'}) SET r.heartbeat_at = 0"
        )
        await taker.acquire("another worker")

    monkeypatch.setattr(EntityDeduplicator, "_claim", taken_over)
    with pytest.raises(OwnershipLostError):
        await dedup(graph).deduplicate(judge_llm=Judge([TIM, APPLE]))
    assert len(await entity_ids(graph)) == 5
    assert await rows(graph, "MATCH (r:__GraphRAGProcessing__ {kind: 'run'}) RETURN r.owner") == [
        ["another worker"]
    ], "the frozen run released nothing it no longer owned"


async def test_a_second_run_at_once_is_refused(graph):
    store = SystemStore(graph)
    await store.ensure_indexes()
    await store.acquire("first")
    with pytest.raises(DedupRunActiveError):
        await dedup(graph).deduplicate()


# ── concurrent writes ────────────────────────────────────────────


async def test_an_entity_an_update_deletes_mid_run_is_not_merged_and_its_group_rechecked(
    graph, monkeypatch
):
    """``delete_orphan_entities`` runs while the plan waits to be applied: the
    delete leaves a deleted record in the same query, the merge that would
    have absorbed the entity is skipped as changed, and its group's pending
    members are checked again by the next run."""
    await checked_graph(graph)
    real = SystemStore.write_plan

    async def then_an_update_deletes(self, ops, decisions):
        plan_run = await real(self, ops, decisions)
        (dup,) = [op.dup for op in ops if op.dup in ("timothy", "tim")]
        await graph.delete_orphan_entities([dup])  # no MENTIONED_IN: an orphan
        return plan_run

    monkeypatch.setattr(SystemStore, "write_plan", then_an_update_deletes)
    dd = dedup(graph)
    merged = await dd.deduplicate(judge_llm=Judge([TIM, APPLE]))
    assert merged == 1 and dd.last_report.changed == 1
    assert "tim" not in await entity_ids(graph) and "timothy" in await entity_ids(graph)
    (pending,) = [r for r in await records(graph) if r[0] == "pending"]
    assert pending[1] == ["timothy"] and pending[3] == ["changed during the run"]


async def test_a_delete_outside_a_run_leaves_no_record(graph):
    await checked_graph(graph)
    await graph.delete_orphan_entities(["jobs"])
    assert await records(graph) == []


# ── failed checks ────────────────────────────────────────────────


async def test_a_failed_check_stays_pending_and_three_make_it_unresolved(graph):
    await checked_graph(graph)
    for attempt in (1, 2, 3):
        dd = dedup(graph)
        assert await dd.deduplicate(judge_llm=Judge([APPLE], fail={"Timothy Cook"})) == (
            1 if attempt == 1 else 0
        )
        (pending,) = [r for r in await records(graph) if r[0] == "pending"]
        assert dict(zip(pending[1], pending[2])) == {"timothy": attempt}
        assert dd.last_report.pending == 1

    dd = dedup(graph)
    judge = Judge([TIM], fail={"Timothy Cook"})
    await graph.upsert_nodes(
        [GraphNode(id="tim2", label="Person", properties={"name": "Tim Cooke", "description": "?"})]
    )
    await dd.deduplicate(judge_llm=judge)
    assert set(dd.last_report.unresolved) == {"timothy"}
    assert not any("Timothy Cook" in p for p in judge.prompts), "left out of every set"
    (unresolved,) = [r for r in await records(graph) if r[0] == "unresolved"]
    assert unresolved[1] == ["timothy"] and unresolved[3] == ["no usable verdict"]

    dd = dedup(graph)
    assert await dd.deduplicate(judge_llm=Judge([TIM]), dedup_all=True) == 1
    assert dd.last_report.unresolved == {}, "dedup_all retries it"


async def test_without_the_judge_the_check_stays_owed(graph):
    """``judge=False`` merges spellings and leaves the judged check owed: the
    pending entities stay pending, and one absorbed into an entity already
    checked hands its debt to that survivor. ``release_unjudged`` clears it."""
    await graph.query_raw(
        "CREATE (o:__Entity__:Organization {id: 'acme', name: 'Acme Corporation', "
        "description: 'anvils', embedding: vecf32([0.0, 0.0, 0.0, 1.0])})"
        "-[:MENTIONED_IN]->(:Chunk {id: 'c1'})"
    )
    await graph.upsert_nodes(
        [
            GraphNode(id="acme-2", label="Organization", properties={"name": "acme corporation"}),
            GraphNode(id="jobs", label="Person", properties={"name": "Steve Jobs"}),
        ]
    )
    dd = dedup(graph)
    assert await dd.deduplicate() == 1, "the exact phase still merges spellings"
    assert await entity_ids(graph) == ["acme", "jobs"]
    (pending,) = [r for r in await records(graph) if r[0] == "pending"]
    assert sorted(pending[1]) == ["acme", "jobs"]

    dd = dedup(graph)
    await dd.deduplicate(release_unjudged=True)
    assert await records(graph) == []


async def test_a_hand_off_needs_no_second_query_when_nothing_was_pending(graph, monkeypatch):
    """A dedup_all run checks entities no record lists; one whose check fails
    must become pending. With no record to append to, the new one is written
    in the query that ticks the decisions — a separate write afterwards could
    be lost to a crash, and the entity would look checked forever."""
    await graph.query_raw(
        "CREATE VECTOR INDEX FOR (n:__Entity__) ON (n.embedding) "
        "OPTIONS {dimension: 4, similarityFunction: 'cosine'}"
    )
    await graph.query_raw(
        "CREATE (:__Entity__:Person {id: 'tim', name: 'Tim Cook', description: 'Apple CEO', "
        "embedding: vecf32([1.0, 0.0, 0.0, 0.0])}), "
        "(:__Entity__:Person {id: 'timothy', name: 'Timothy Cook', "
        "description: 'chief executive of Apple', embedding: vecf32([0.97, 0.2, 0.0, 0.0])})"
    )

    async def no_second_write(self, ids):
        raise AssertionError(f"pending records written outside the tick: {ids}")

    monkeypatch.setattr(SystemStore, "add_pending", no_second_write)
    dd = dedup(graph)
    judge = Judge([TIM], fail={"Timothy Cook"})
    assert await dd.deduplicate(judge_llm=judge, dedup_all=True) == 0
    assert judge.prompts, "the pair was judged; the judge failed it"
    (pending,) = [r for r in await records(graph) if r[0] == "pending"]
    assert sorted(pending[1]) == ["tim", "timothy"] and dd.last_report.pending == 2


# ── attributes ───────────────────────────────────────────────────


async def test_the_judge_sees_the_ontologys_attributes_and_picks_the_value(graph):
    """The judge sees each entity's declared attributes and chooses which
    birth year the merged entity keeps; the other stays on it, in
    ``born__dropped``, and the run reports the collision."""
    ontology = Ontology(
        entities=[Entity(label="Person", properties=[Attribute(name="born", type="INTEGER")])]
    )
    await graph.query_raw(
        "CREATE (:__Entity__:Person {id: 'tim', name: 'Tim Cook', description: 'Apple CEO', "
        "born: 1961, embedding: vecf32([1.0, 0.0, 0.0, 0.0])})"
    )
    await graph.upsert_nodes(
        [
            GraphNode(
                id="timothy",
                label="Person",
                properties={
                    "name": "Timothy Cook",
                    "description": "chief executive of Apple",
                    "born": 1960,
                },
            )
        ]
    )
    judge = Judge([TIM], keep={"born": "Tim Cook"})
    dd = dedup(graph)
    assert await dd.deduplicate(judge_llm=judge, ontology=ontology) == 1
    assert "TYPES AND THEIR ATTRIBUTES\nPerson. Attributes: born (INTEGER)" in judge.prompts[0]
    assert "| born: 1961" in judge.prompts[0] and "| born: 1960" in judge.prompts[0]
    # Timothy Cook survives (the longer description); the judge kept Tim's year.
    assert await rows(graph, "MATCH (e:__Entity__) RETURN e.id, e.born, e.born__dropped") == [
        ["timothy", 1961, [1960]]
    ]
    assert dd.last_report.collisions == [
        "Timothy Cook: born kept 1961 (judge), dropped 1960 from Tim Cook"
    ]


async def test_a_merge_whose_reply_was_lost_still_counts_and_its_group_is_renamed(
    graph, monkeypatch
):
    """The server applies a merge and the reply is lost; the connection sends
    the query again, which finds it already ticked. That is a merge this run
    made: it counts, its collision is reported, and the group's rename — which
    runs only after a judged merge applied — still happens (Naseem's review)."""
    ontology = Ontology(
        entities=[Entity(label="Person", properties=[Attribute(name="born", type="INTEGER")])]
    )
    await graph.query_raw(
        "CREATE (:__Entity__:Person {id: 'tim', name: 'Tim Cook', description: 'Apple CEO', "
        "born: 1961, embedding: vecf32([1.0, 0.0, 0.0, 0.0])})"
    )
    await graph.upsert_nodes(
        [
            GraphNode(
                id="timothy",
                label="Person",
                properties={
                    "name": "Timothy Cook",
                    "description": "chief executive of Apple",
                    "born": 1960,
                },
            )
        ]
    )
    real = graph.query_raw
    lost: list[str] = []

    async def reply_lost_once(cypher, params=None):
        if "DETACH DELETE d" in cypher or "SET s.name = $name" in cypher:
            await real(cypher, params)  # applied on the server; the reply never arrives
            lost.append(cypher)
        return await real(cypher, params)  # the connection's retry

    monkeypatch.setattr(graph, "query_raw", reply_lost_once)
    dd = dedup(graph)
    judge = Judge([TIM], keep={"born": "Tim Cook"}, name="Tim Cook")
    assert await dd.deduplicate(judge_llm=judge, ontology=ontology) == 1
    assert len(lost) == 2, "the merge and the rename were each sent twice"
    assert await rows(
        graph, "MATCH (e:__Entity__) RETURN e.id, e.name, e.aliases, e.born, e.born__dropped"
    ) == [["timothy", "Tim Cook", ["Timothy Cook"], 1961, [1960]]]
    assert dd.last_report.collisions == [
        "Timothy Cook: born kept 1961 (judge), dropped 1960 from Tim Cook"
    ]
    assert dd.last_report.changed == 0 and await records(graph) == []


# ── the server ───────────────────────────────────────────────────


async def test_falkordb_before_4_20_is_refused(graph, monkeypatch):
    async def old():
        return (4, 18, 0)

    monkeypatch.setattr(graph, "server_version", old)
    with pytest.raises(DatabaseError, match="4.20 or later"):
        await dedup(graph).deduplicate()
    assert await records(graph) == [], "refused before the lock"


async def test_statistics_do_not_count_system_nodes(graph):
    await checked_graph(graph)
    store = SystemStore(graph)
    await store.ensure_indexes()
    await store.acquire("x")
    await store.add_pending(["timothy"])
    stats = await graph.get_statistics()
    assert stats["node_count"] == 5
    assert not any(label.startswith("__GraphRAG") for label in stats["entity_types"])
