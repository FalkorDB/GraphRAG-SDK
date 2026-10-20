"""The dedup run's system records and its single-query merge, on a real FalkorDB.

Gated by ``RUN_INTEGRATION=1`` (``FALKOR_HOST`` / ``FALKOR_PORT``). Every test
uses its own graph, dropped afterwards.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.storage.deduplicator import EntityDeduplicator
from graphrag_sdk.storage.graph_store import GraphStore
from graphrag_sdk.storage.merge_query import (
    APPLIED,
    CHANGED,
    DONE,
    SKIP_QUERY,
    STALE,
    PlanStateError,
    absorb_query,
    handled,
    outcome,
    read_for_merge,
    rename_query,
    snapshot_params,
)
from graphrag_sdk.storage.system_store import (
    ABSORB,
    JUDGE,
    RENAME,
    Decisions,
    DedupRunActiveError,
    OwnershipLostError,
    PendingPage,
    PlanOp,
    SystemStore,
    Unresolved,
    deleted_record_clause,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1", reason="Requires RUN_INTEGRATION=1"
)


@pytest.fixture
async def graph():
    conn = FalkorDBConnection(
        ConnectionConfig(
            host=os.getenv("FALKOR_HOST", "localhost"),
            port=int(os.getenv("FALKOR_PORT", "6379")),
            username=os.getenv("FALKOR_USERNAME") or None,
            password=os.getenv("FALKOR_PASSWORD") or None,
            graph_name=f"test_dedup_{uuid4().hex[:8]}",
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


async def held(graph) -> SystemStore:
    store = SystemStore(graph)
    await store.ensure_indexes()
    await store.acquire("test")
    return store


async def one_absorb_plan(
    store, survivor="s", survivor_name="Tim Cook", dup="d", dup_name="Timothy Cook"
):
    return await store.write_plan(
        [PlanOp(ABSORB, 0, survivor, survivor_name, dup=dup, dup_name=dup_name)], Decisions()
    )


async def absorb(
    store,
    graph,
    plan_run,
    *,
    op=0,
    survivor="s",
    dup="d",
    survivor_name="Tim Cook",
    dup_name="Timothy Cook",
    read=None,
    sets=(),
    labels=(),
):
    read = read or await read_for_merge(graph, survivor, dup)
    result = await store.fenced(
        absorb_query(read.kinds, list(sets), list(labels)),
        {
            "plan_run": plan_run,
            "page": 0,
            "op": op,
            "dup": dup,
            "dup_name": dup_name,
            "survivor": survivor,
            "survivor_name": survivor_name,
            "handled": handled(read.kinds),
            "aliases": ["Timothy Cook"],
            **snapshot_params(read, {}),
        },
    )
    return outcome(result[0], op)


async def tim_cook(graph):
    """A survivor and a duplicate with edges of four kinds, both directions."""
    await graph.query_raw(
        "CREATE (s:__Entity__:Person {id: 's', name: 'Tim Cook', source_chunk_ids: ['c1']}), "
        "(d:__Entity__:Person {id: 'd', name: 'Timothy Cook', description: 'CEO', "
        " source_chunk_ids: ['c2'], title: 'CEO'}), "
        "(apple:__Entity__:Organization {id: 'apple', name: 'Apple'}), "
        "(jobs:__Entity__:Person {id: 'jobs', name: 'Steve Jobs'}), "
        "(c1:Chunk {id: 'c1'}), (c2:Chunk {id: 'c2'}), "
        "(s)-[:RELATES {rel_type: 'WORKS_AT', source_chunk_ids: ['c1']}]->(apple), "
        "(d)-[:RELATES {rel_type: 'WORKS_AT', source_chunk_ids: ['c2']}]->(apple), "
        "(d)-[:RELATES {rel_type: 'LEADS', source_chunk_ids: ['c2']}]->(apple), "
        "(jobs)-[:RELATES {rel_type: 'HIRED', source_chunk_ids: ['c2']}]->(d), "
        "(d)-[:RELATES {rel_type: 'SAME', source_chunk_ids: ['c2']}]->(s), "
        "(d)-[:MENTIONED_IN]->(c2), (s)-[:MENTIONED_IN]->(c1), "
        "(jobs)-[:MANAGES {since: 1998}]->(d)"
    )


# ── the lock and the fence ───────────────────────────────────────


async def test_one_run_at_a_time_and_a_stale_lock_is_taken_over(graph):
    first = await held(graph)
    with pytest.raises(DedupRunActiveError):
        await SystemStore(graph).acquire("second")

    await graph.query_raw("MATCH (r:__GraphRAGProcessing__ {kind: 'run'}) SET r.heartbeat_at = 0")
    second = SystemStore(graph)
    assert await second.acquire("second") == first.run_id

    with pytest.raises(OwnershipLostError):
        await first.heartbeat()
    await second.heartbeat()
    await first.release()  # lost already: releases nothing
    assert await rows(graph, "MATCH (r:__GraphRAGProcessing__ {kind: 'run'}) RETURN r.run_id") == [
        [second.run_id]
    ]
    await second.release()
    assert await rows(graph, "MATCH (r:__GraphRAGProcessing__) RETURN count(r)") == [[0]]


async def test_pending_records_are_paged_and_claimed(graph):
    store = await held(graph)
    await store.add_pending([f"e{i}" for i in range(12_000)])
    pages = await store.pending_pages()
    assert [(p.page, len(p.ids)) for p in pages] == [(0, 10_000), (1, 2_000)]

    pages[1].attempts = [1] * 2_000
    pages[1].claimed_run = store.run_id
    await store.write_pending([pages[1]])
    claimed = (await store.pending_pages())[1]
    assert claimed.attempts[:2] == [1, 1] and claimed.claimed_run == store.run_id

    await store.write_pending([PendingPage(0, [], [], [])])
    assert [p.page for p in await store.pending_pages()] == [1]


async def test_a_delete_during_a_run_leaves_a_deleted_record_and_none_without_one(graph):
    await graph.query_raw("UNWIND range(0, 3) AS i CREATE (:__Entity__ {id: 'x' + toString(i)})")
    delete = (
        "MATCH (e:__Entity__) WHERE e.id IN $ids WITH collect(e) AS es, collect(e.id) AS gone "
        + deleted_record_clause("gone")
        + "FOREACH (e IN es | DETACH DELETE e)"
    )
    await graph.query_raw(delete, {"ids": ["x0"]})  # no run: no record
    store = await held(graph)
    await graph.query_raw(delete, {"ids": ["x1", "x2"]})
    assert await store.deleted_among(["x0", "x1", "x2", "x3"]) == {"x1", "x2"}
    await store.clear_deleted()
    assert await store.deleted_among(["x1", "x2"]) == set()


async def test_decisions_apply_once(graph):
    store = await held(graph)
    await store.add_pending(["a", "b", "c"])
    decided = Decisions(clear=["a", "b"], retry=["c"], reasons=["no verdict"])
    await store.write_plan([], decided)
    plan = await store.sealed_plan()
    assert plan.decisions == [decided]
    after = [PendingPage(0, ["c"], [0], [""])]
    assert await store.apply_decisions(plan.plan_run, 0, after) is True
    assert await store.apply_decisions(plan.plan_run, 0, [PendingPage(0, [], [], [])]) is False
    assert [p.ids for p in await store.pending_pages()] == [["c"]]
    assert (await store.sealed_plan()).decision_done == [True]


# ── the single-query merge ───────────────────────────────────────


async def test_an_absorb_moves_every_edge_once_and_deletes_the_duplicate(graph):
    await tim_cook(graph)
    store = await held(graph)
    plan_run = await one_absorb_plan(store)

    assert await absorb(store, graph, plan_run, sets=["s.aliases = $aliases"]) == APPLIED

    assert await rows(graph, "MATCH (e:__Entity__ {id: 'd'}) RETURN count(e)") == [[0]]
    edges = await rows(
        graph,
        "MATCH (a)-[r]->(b) RETURN a.id, type(r), r.rel_type, r.source_chunk_ids, b.id "
        "ORDER BY a.id, type(r), r.rel_type, b.id",
    )
    assert edges == [
        ["jobs", "MANAGES", None, None, "s"],
        ["jobs", "RELATES", "HIRED", ["c2"], "s"],
        ["s", "MENTIONED_IN", None, None, "c1"],
        ["s", "MENTIONED_IN", None, None, "c2"],
        ["s", "RELATES", "LEADS", ["c2"], "apple"],
        # Keyed on rel_type, provenance unioned.
        ["s", "RELATES", "WORKS_AT", ["c1", "c2"], "apple"],
    ]
    survivor = await rows(
        graph, "MATCH (s:__Entity__ {id: 's'}) RETURN s.source_chunk_ids, s.aliases"
    )
    assert survivor == [[["c1", "c2"], ["Timothy Cook"]]]
    assert (await store.sealed_plan()).nexts == [1]

    # A replay (a lost reply, a resumed run) writes nothing.
    read = await read_for_merge(graph, "s", "jobs")  # any read: the turn is checked first
    assert await absorb(store, graph, plan_run, read=read) == DONE


async def test_an_edge_type_the_query_was_not_built_for_stops_the_merge(graph):
    await tim_cook(graph)
    store = await held(graph)
    plan_run = await one_absorb_plan(store)
    read = await read_for_merge(graph, "s", "d")
    await graph.query_raw(
        "MATCH (d:__Entity__ {id: 'd'}), (a:__Entity__ {id: 'apple'}) CREATE (d)-[:OWNS]->(a)"
    )

    assert await absorb(store, graph, plan_run, read=read) == STALE
    assert await rows(graph, "MATCH (e:__Entity__ {id: 'd'})-[r]-() RETURN count(r)") == [[7]]
    assert (await store.sealed_plan()).nexts == [0]

    assert await absorb(store, graph, plan_run) == APPLIED  # read again: OWNS moves too
    assert await rows(graph, "MATCH (:__Entity__ {id: 's'})-[r:OWNS]->() RETURN count(r)") == [[1]]


async def test_a_property_written_after_the_read_stops_the_merge(graph):
    await tim_cook(graph)
    store = await held(graph)
    plan_run = await one_absorb_plan(store)
    read = await read_for_merge(graph, "s", "d")
    await graph.query_raw("MATCH (s:__Entity__ {id: 's'}) SET s.description = 'Apple CEO'")
    assert await absorb(store, graph, plan_run, read=read) == STALE

    read = await read_for_merge(graph, "s", "d")
    await graph.query_raw("MATCH (d:__Entity__ {id: 'd'}) SET d.title = 'Chief Executive'")
    result = await store.fenced(
        absorb_query(read.kinds, ["s += $carry"], []),
        {
            "plan_run": plan_run,
            "page": 0,
            "op": 0,
            "dup": "d",
            "dup_name": "Timothy Cook",
            "survivor": "s",
            "survivor_name": "Tim Cook",
            "handled": handled(read.kinds),
            **snapshot_params(read, {"title": read.dup["title"]}),
        },
    )
    assert outcome(result[0], 0) == STALE  # the carried title is not the one read


async def test_an_entity_deleted_or_renamed_since_judging_is_recorded_as_changed(graph):
    await tim_cook(graph)
    store = await held(graph)
    plan_run = await store.write_plan(
        [
            PlanOp(ABSORB, 0, "s", "Tim Cook", dup="d", dup_name="Timothy Cook"),
            PlanOp(ABSORB, 0, "s", "Tim Cook", dup="jobs", dup_name="Steven Jobs"),
        ],
        Decisions(),
    )
    await graph.query_raw(
        "MATCH (d:__Entity__ {id: 'd'}) WITH collect(d) AS es, ['d'] AS gone "
        + deleted_record_clause("gone")
        + "FOREACH (e IN es | DETACH DELETE e) "
        "CREATE (:__Entity__:Person {id: 'd', name: 'Timothy Cook'})"  # re-created under the id
    )

    assert await absorb(store, graph, plan_run) == CHANGED
    assert await absorb(store, graph, plan_run, op=1, dup="jobs", dup_name="Steven Jobs") == CHANGED
    assert await rows(
        graph, "MATCH (e:__Entity__) WHERE e.id IN ['d', 'jobs'] RETURN count(e)"
    ) == [[2]]
    plan = await store.sealed_plan()
    assert plan.changed == [[0, 1]] and plan.nexts == [2]


async def test_an_operation_out_of_turn_or_without_its_page_is_an_error(graph):
    await tim_cook(graph)
    store = await held(graph)
    plan_run = await one_absorb_plan(store)
    with pytest.raises(PlanStateError, match="before its turn"):
        await absorb(store, graph, plan_run, op=1)
    with pytest.raises(PlanStateError, match="missing"):
        await absorb(store, graph, "no-such-plan")


async def test_a_run_that_lost_the_lock_merges_nothing(graph):
    await tim_cook(graph)
    store = await held(graph)
    plan_run = await one_absorb_plan(store)
    await graph.query_raw("MATCH (r:__GraphRAGProcessing__ {kind: 'run'}) SET r.heartbeat_at = 0")
    await SystemStore(graph).acquire("taker")

    with pytest.raises(OwnershipLostError):
        await absorb(store, graph, plan_run)
    assert await rows(graph, "MATCH (e:__Entity__ {id: 'd'}) RETURN count(e)") == [[1]]


async def test_a_rename_and_a_skip(graph):
    await tim_cook(graph)
    store = await held(graph)
    plan_run = await store.write_plan(
        [
            PlanOp(ABSORB, 0, "s", "Tim Cook", dup="d", dup_name="Timothy Cook"),
            PlanOp(RENAME, 0, "s", "Tim Cook", rename_to="Timothy D. Cook"),
        ],
        Decisions(),
    )
    params = {"plan_run": plan_run, "page": 0, "survivor": "s", "survivor_name": "Tim Cook"}

    assert outcome((await store.fenced(SKIP_QUERY, {**params, "op": 0}))[0], 0) == CHANGED
    result = await store.fenced(
        rename_query(with_vector=True),
        {**params, "op": 1, "name": "Timothy D. Cook", "vector": [0.1, 0.2], "apply": True},
    )
    assert outcome(result[0], 1) == APPLIED
    assert await rows(
        graph, "MATCH (s:__Entity__ {id: 's'}) RETURN s.name, s.aliases, s.embedding IS NOT NULL"
    ) == [["Timothy D. Cook", ["Tim Cook"], True]]
    assert (await store.sealed_plan()).changed == [[0]]


async def test_unresolved_entities_are_recorded_reported_and_dropped(graph):
    store = await held(graph)
    await store.add_unresolved(
        {"p": Unresolved(3, "no usable verdict"), "q": Unresolved(3, "interrupted")}
    )
    await store.set_blocked_with({"p": "Tim Cook | Timothy Cook"})
    found = await store.unresolved()
    assert found["p"] == Unresolved(3, "no usable verdict", "Tim Cook | Timothy Cook")
    assert found["q"].reason == "interrupted"
    await store.drop_unresolved(["p", "q"])
    assert await store.unresolved() == {}
    assert await rows(graph, "MATCH (u:__GraphRAGProcessing__ {kind: 'unresolved'}) RETURN u") == []


async def test_a_plan_page_holds_every_column_and_decisions_page_together(graph):
    store = await held(graph)
    ops = [
        PlanOp(ABSORB, g, f"s{g}", f"S{g}", dup=f"d{g}", dup_name=f"D{g}", policy="judge")
        for g in range(3)
    ]
    retry = [f"r{i}" for i in range(10_001)]
    plan_run = await store.write_plan(
        ops, Decisions(retry=retry, reasons=["held"] * len(retry), handoff=["h"])
    )
    plan = await store.sealed_plan()
    assert plan.plan_run == plan_run and plan.ops == [ops] and plan.nexts == [0]
    assert [len(d.retry) for d in plan.decisions] == [10_000, 1]
    assert [len(d.reasons) for d in plan.decisions] == [10_000, 1]
    assert plan.decisions[0].handoff == ["h"] and plan.decision_done == [False, False]
    await store.drop_plans()
    assert await store.sealed_plan() is None


async def test_a_collision_keeps_the_chosen_value_and_the_other_in_dropped(graph):
    """The judge chose the duplicate's ``born``; the survivor's goes to
    ``born__dropped``, in the one merge query. A value rewritten after the
    read stops the merge: the fold would otherwise overwrite it."""
    await tim_cook(graph)
    await graph.query_raw(
        "MATCH (s:__Entity__ {id: 's'}), (d:__Entity__ {id: 'd'}) "
        "SET s.born = 1961, s.born__dropped = [1959], d.born = 1960"
    )
    store = await held(graph)
    op = PlanOp(
        ABSORB,
        0,
        "s",
        "Tim Cook",
        dup="d",
        dup_name="Timothy Cook",
        policy=JUDGE,
        keep='{"born": 1960}',
    )
    plan_run = await store.write_plan([op], Decisions())
    where = {
        "plan_run": plan_run,
        "page": 0,
        "op": 0,
        "dup": "d",
        "dup_name": "Timothy Cook",
        "survivor": "s",
        "survivor_name": "Tim Cook",
    }

    async def apply(read):
        fold = EntityDeduplicator(graph, None)._fold(op, read)
        result = await store.fenced(
            absorb_query(read.kinds, fold.sets, fold.labels),
            {**fold.params, **where, "handled": handled(read.kinds)},
        )
        return outcome(result[0], 0)

    read = await read_for_merge(graph, "s", "d")
    await graph.query_raw("MATCH (s:__Entity__ {id: 's'}) SET s.born = 1962")  # ingest, between
    assert await apply(read) == STALE
    await graph.query_raw("MATCH (s:__Entity__ {id: 's'}) SET s.born = 1961")

    assert await apply(await read_for_merge(graph, "s", "d")) == APPLIED
    assert await rows(graph, "MATCH (s:__Entity__ {id: 's'}) RETURN s.born, s.born__dropped") == [
        [1960, [1959, 1961]]
    ]
