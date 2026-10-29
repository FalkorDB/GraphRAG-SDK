"""The dedup run's system records and its single-query merge, on a real FalkorDB.

Gated by ``RUN_INTEGRATION=1`` (``FALKOR_HOST`` / ``FALKOR_PORT``). Every test
uses its own graph, dropped afterwards.
"""

from __future__ import annotations

import logging
import os

import pytest

from graphrag_sdk.storage.deduplicator import EntityDeduplicator
from graphrag_sdk.storage.graph_store import GraphStore
from graphrag_sdk.storage.merge_query import (
    ALREADY_APPLIED,
    ALREADY_CHANGED,
    APPLIED,
    CHANGED,
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
    PENDING,
    PLAN,
    PLAN_HEADER,
    PLAN_IDS,
    PROCESSING_LABEL,
    RENAME,
    RUN,
    UNRESOLVED,
    Decisions,
    DedupRunActiveError,
    OwnershipLostError,
    PendingPage,
    PlanOp,
    SystemStore,
    Unresolved,
    deleted_record_clause,
)

from .conftest import lose_each_reply_once, rows

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1", reason="Requires RUN_INTEGRATION=1"
)


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


async def fold_and_absorb(store, graph, plan_run, op: PlanOp, read):
    """Apply ``op``, the plan's first operation, with the write
    ``EntityDeduplicator._fold`` computes from ``read``."""
    fold = EntityDeduplicator(graph, None)._fold(op, read)
    result = await store.fenced(
        absorb_query(read.kinds, fold.sets, fold.labels),
        {
            **fold.params,
            "plan_run": plan_run,
            "page": 0,
            "op": 0,
            "dup": op.dup,
            "dup_name": op.dup_name,
            "survivor": op.survivor,
            "survivor_name": op.survivor_name,
            "handled": handled(read.kinds),
        },
    )
    return outcome(result[0], 0)


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


@pytest.mark.parametrize(
    "attempts, reasons, read",
    [
        # x's count lost: padding [3] gave x the 3 attempts of y, unjudged.
        ([3], ["", "no usable verdict"], ([0, 0], ["", "no usable verdict"])),
        ([0, 3], [], ([0, 3], ["", ""])),
        ([0, 3, 1], ["", "held"], ([0, 0], ["", "held"])),
    ],
)
async def test_a_pending_list_that_lost_or_gained_entries_is_reset_once(
    graph, caplog, attempts, reasons, read
):
    """FalkorDB 4.22.2 was seen to empty a stored list of a pending record.
    A list whose length no longer matches the ids cannot say which entries
    changed, so it is reset whole, never padded; the reset is written back,
    so it is reported once."""
    store = await held(graph)
    await store.add_pending(["x", "y"])
    await graph.query_raw(
        f"MATCH (p:{PROCESSING_LABEL} {{kind: '{PENDING}'}}) SET p.attempts = $a, p.reasons = $r",
        {"a": attempts, "r": reasons},
    )
    with caplog.at_level(logging.WARNING, logger="graphrag_sdk.storage.system_store"):
        (page,) = await store.pending_pages()
        assert (page.ids, page.attempts, page.reasons) == (["x", "y"], *read)
        assert await store.pending_pages() == [page], "written back as read"
    assert sum("Pending record" in r.getMessage() for r in caplog.records) == 1


async def test_a_pending_record_that_lost_its_ids_is_deleted_not_read(graph):
    """Written back without ids, the record is deleted: a hand-off put on it
    would list nothing."""
    store = await held(graph)
    await store.add_pending(["x", "y"])
    await store.add_pending(["z"])
    await graph.query_raw(
        f"MATCH (p:{PROCESSING_LABEL} {{kind: '{PENDING}', page: 1}}) SET p.ids = []"
    )
    assert [p.ids for p in await store.pending_pages()] == [["x", "y"]]
    assert await rows(
        graph, f"MATCH (p:{PROCESSING_LABEL} {{kind: '{PENDING}'}}) RETURN p.page"
    ) == [[0]]


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


async def test_a_label_wide_delete_records_every_entity_batch_by_batch(graph, monkeypatch):
    """Deleting a whole label collected every node of it into one query, to
    record the deleted ids beside the delete. Each batch now records its own."""
    monkeypatch.setattr(GraphStore, "_DELETE_BATCH", 2)
    await graph.query_raw(
        "UNWIND range(0, 4) AS i CREATE (:__Entity__:Gadget {id: 'g' + toString(i)})"
    )
    await graph.query_raw("CREATE (:Gadget {id: 'plain'})")
    store = await held(graph)
    real, batches = graph._conn.query, []

    async def query(cypher, params=None, **kw):
        result = await real(cypher, params, **kw)
        if "DETACH DELETE n" in cypher:
            batches.append(result.result_set[0][0] if result.result_set else None)
        return result

    monkeypatch.setattr(graph._conn, "query", query)
    assert await graph.delete_nodes_by_label("Gadget") == 6
    assert batches == [2, 2, 2, 0]
    assert await rows(graph, "MATCH (n:Gadget) RETURN count(n)") == [[0]]
    ids = [f"g{i}" for i in range(5)]
    assert await store.deleted_among([*ids, "plain"]) == set(ids)


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

    # A replay (a lost reply, a resumed run) writes nothing, and says the
    # earlier attempt applied it.
    read = await read_for_merge(graph, "s", "jobs")  # any read: the turn is checked first
    assert await absorb(store, graph, plan_run, read=read) == ALREADY_APPLIED


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


@pytest.mark.parametrize(
    "edit",
    [
        "MATCH (d:__Entity__ {id: 'd'}) SET d.title = 'CTO'",
        "MATCH (d:__Entity__ {id: 'd'}) SET d.phone = '555-0100'",
        "MATCH (s:__Entity__ {id: 's'}) SET s.title = 'Chairman'",
        "MATCH (d:__Entity__ {id: 'd'}) SET d:Executive REMOVE d:Person",
        "MATCH (s:__Entity__ {id: 's'}) SET s:Executive",
    ],
    ids=[
        "a shared value changed",
        "a property added",
        "the survivor's value changed",
        "the duplicate's label renamed",
        "a label added to the survivor",
    ],
)
async def test_any_change_to_what_the_merge_folds_stops_it(graph, edit):
    """Both hold title 'CEO', so the fold neither carries nor collides on it.
    A write to it after the read, or a property added to the duplicate, used
    to pass the check, and the duplicate was deleted with the new value. The
    labels the fold adds come from both nodes' labels: a label renamed since
    the read came back on the survivor."""
    await tim_cook(graph)
    await graph.query_raw("MATCH (s:__Entity__ {id: 's'}) SET s.title = 'CEO'")
    store = await held(graph)
    plan_run = await one_absorb_plan(store)
    read = await read_for_merge(graph, "s", "d")
    await graph.query_raw(edit)
    assert await absorb(store, graph, plan_run, read=read) == STALE
    assert await rows(graph, "MATCH (d:__Entity__ {id: 'd'}) RETURN count(d)") == [[1]]


async def test_a_name_vector_on_the_duplicate_does_not_block_the_merge(graph):
    """The name vector is not compared: it does not survive the client round
    trip, and only a name change — itself checked — replaces it."""
    await tim_cook(graph)
    await graph.query_raw(
        "MATCH (d:__Entity__ {id: 'd'}) SET d.embedding = vecf32([0.1, 0.2, 0.3, 0.4])"
    )
    store = await held(graph)
    plan_run = await one_absorb_plan(store)
    assert await absorb(store, graph, plan_run) == APPLIED


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
    # Sent again (a lost reply), a skipped operation says it was skipped.
    assert await absorb(store, graph, plan_run) == ALREADY_CHANGED


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

    async def apply(read):
        return await fold_and_absorb(store, graph, plan_run, op, read)

    read = await read_for_merge(graph, "s", "d")
    await graph.query_raw("MATCH (s:__Entity__ {id: 's'}) SET s.born = 1962")  # ingest, between
    assert await apply(read) == STALE
    await graph.query_raw("MATCH (s:__Entity__ {id: 's'}) SET s.born = 1961")

    assert await apply(await read_for_merge(graph, "s", "d")) == APPLIED
    assert await rows(graph, "MATCH (s:__Entity__ {id: 's'}) RETURN s.born, s.born__dropped") == [
        [1960, [1959, 1961]]
    ]


async def test_a_decimal_read_back_rounded_does_not_block_the_merge(graph):
    """``0.1 + 0.2`` is stored as 0.30000000000000004 and read back as 0.3.
    Compared exactly, the merge was stale on every read and never applied
    (Naseem's review of #355). Within the tolerance it applies; a real change,
    or a number turned into text, after the read still stops it."""
    await tim_cook(graph)
    await graph.query_raw(
        "MATCH (s:__Entity__ {id: 's'}), (d:__Entity__ {id: 'd'}) "
        "SET s.score = 0.1 + 0.2, d.score = 0.75, "
        "d.weight = 0.1 + 0.2, d.series = [0.1 + 0.2, 1]"
    )
    store = await held(graph)
    op = PlanOp(ABSORB, 0, "s", "Tim Cook", dup="d", dup_name="Timothy Cook")
    plan_run = await store.write_plan([op], Decisions())

    async def apply(read):
        return await fold_and_absorb(store, graph, plan_run, op, read)

    for change in (
        "SET s.score = 0.31",  # a real edit
        "SET s.score = 'high'",  # a number turned into text: no crash
        "SET d.weight = 0.4",
        "SET d.series = 5",  # a list turned into a number
        "SET d.series = [0.1 + 0.2, 2]",
    ):
        read = await read_for_merge(graph, "s", "d")
        assert read.keep["score"] == 0.3  # rounded on the way out
        await graph.query_raw(
            f"MATCH (s:__Entity__ {{id: 's'}}), (d:__Entity__ {{id: 'd'}}) {change}"
        )
        assert await apply(read) == STALE, change
        await graph.query_raw(
            "MATCH (s:__Entity__ {id: 's'}), (d:__Entity__ {id: 'd'}) "
            "SET s.score = 0.1 + 0.2, d.weight = 0.1 + 0.2, d.series = [0.1 + 0.2, 1]"
        )

    assert await apply(await read_for_merge(graph, "s", "d")) == APPLIED
    ((score, dropped, weight, series),) = await rows(
        graph,
        "MATCH (s:__Entity__ {id: 's'}) RETURN s.score, s.score__dropped, s.weight, s.series",
    )
    assert score == pytest.approx(0.3) and dropped == [0.75]
    assert weight == pytest.approx(0.3) and series == [pytest.approx(0.3), 1]


@pytest.mark.parametrize(
    "stored, edited",
    [("1700000000.25", "1700000000.75"), ("1e-12", "5e-10")],
    ids=["a timestamp half a second later", "a value near zero"],
)
async def test_a_small_real_edit_of_a_large_or_tiny_decimal_stops_the_merge(graph, stored, edited):
    """The tolerance was 1e-9 x (1 + |value|): a float timestamp could move by
    up to 1.7 s, and 1e-12 become 5e-10, and still read as unchanged."""
    await tim_cook(graph)
    await graph.query_raw(f"MATCH (d:__Entity__ {{id: 'd'}}) SET d.at = {stored}")
    store = await held(graph)
    plan_run = await one_absorb_plan(store)
    read = await read_for_merge(graph, "s", "d")
    await graph.query_raw(f"MATCH (d:__Entity__ {{id: 'd'}}) SET d.at = {edited}")
    assert await absorb(store, graph, plan_run, read=read) == STALE


async def test_a_hand_off_with_no_pending_record_is_written_with_the_tick(graph):
    """Entities that become pending when no record exists to take them get a
    new record in the query that ticks the decision page: a crash cannot leave
    the page done and them unlisted, looking checked. A replay adds nothing."""
    store = await held(graph)
    await store.write_plan([], Decisions(handoff=["h1", "h2"]))
    plan = await store.sealed_plan()
    created = PendingPage(0, ["h1", "h2"], [0, 0], ["", ""])
    assert await store.apply_decisions(plan.plan_run, 0, [], created) is True
    assert [(p.page, p.ids) for p in await store.pending_pages()] == [(0, ["h1", "h2"])]
    assert await store.apply_decisions(plan.plan_run, 0, [], created) is False
    assert len(await store.pending_pages()) == 1
    assert (await store.sealed_plan()).decision_done == [True]


async def test_a_record_write_whose_reply_was_lost_is_written_once(graph, monkeypatch):
    """A write that commits and loses its reply is sent again by the
    connection's retry. Plan pages, decision pages, the header, pending and
    unresolved records are written once all the same: with two copies of a
    page, recovery read pages by position and ticked the wrong one."""
    store = await held(graph)
    lose_each_reply_once(graph, monkeypatch, f"CREATE (:{PROCESSING_LABEL}")
    await store.add_pending(["a", "b"])
    await store.add_unresolved({"u": Unresolved(3, "no usable verdict", "")})
    op = PlanOp(ABSORB, 0, "s", "S", dup="d", dup_name="D", policy=JUDGE)
    await store.write_plan([op], Decisions(clear=["a"]))
    kinds = await rows(
        graph,
        f"MATCH (p:{PROCESSING_LABEL}) RETURN p.kind, count(p) ORDER BY p.kind",
    )
    assert kinds == [
        [k, 1] for k in sorted((PENDING, PLAN, PLAN_HEADER, PLAN_IDS, RUN, UNRESOLVED))
    ]
    plan = await store.sealed_plan()
    assert [len(ops) for ops in plan.ops] == [1] and len(plan.decisions) == 1


async def test_a_table_rekey_folds_into_the_taken_id_in_one_query(graph):
    """A re-synced row whose new id is taken folds into that node in one
    query. As several, a dedup merge applied between them found the old node
    half-moved and split the entity. Every edge type moves (the old remap
    moved only RELATES and MENTIONED_IN), its properties carry, the result is
    pending again, and during a dedup run both ids get a deleted record."""
    await graph.query_raw(
        "CREATE (o:__Entity__:Organization {id: 'old', name: 'Globex', role: 'supplier', "
        "source_chunk_ids: ['c1']}), "
        "(n:__Entity__:Organization {id: 'new', name: 'Initech', embedding: vecf32([1.0, 0.0])}), "
        "(a:__Entity__:Person {id: 'a', name: 'Ann'}), (c:Chunk {id: 'c1'}), "
        "(o)-[:RELATES {rel_type: 'SUPPLIES', source_chunk_ids: ['c1']}]->(a), "
        "(o)-[:MENTIONED_IN]->(c), (a)-[:OWNS]->(o)"
    )
    store = await held(graph)
    assert await graph._move_entity("old", "new") == (0, 1)
    assert await rows(graph, "MATCH (n:__Entity__ {id: 'old'}) RETURN count(n)") == [[0]]
    assert await rows(
        graph,
        "MATCH (n:__Entity__ {id: 'new'})-[r]-(x) "
        "RETURN type(r), startNode(r) = n, x.id ORDER BY type(r)",
    ) == [["MENTIONED_IN", True, "c1"], ["OWNS", False, "a"], ["RELATES", True, "a"]]
    assert await rows(
        graph,
        "MATCH (n:__Entity__ {id: 'new'}) RETURN n.role, n.embedding IS NULL, n.source_chunk_ids",
    ) == [["supplier", True, ["c1"]]]
    assert await store.deleted_among(["old", "new"]) == {"old", "new"}


async def _globex_and_initech(graph):
    """A re-synced row's old node (Globex) and the node at its new id."""
    await graph.query_raw(
        "CREATE (o:__Entity__:Organization {id: 'old', name: 'Globex', role: 'supplier', "
        "source_chunk_ids: ['c1']}), "
        "(n:__Entity__:Organization {id: 'new', name: 'Initech', source_chunk_ids: ['c5']}), "
        "(c:Chunk {id: 'c1'}), (o)-[:MENTIONED_IN]->(c)"
    )


@pytest.mark.parametrize(
    "edit, check, expected",
    [
        (
            "MATCH (o:__Entity__ {id: 'old'}) SET o.phone = '555-0100'",
            "MATCH (n:__Entity__ {id: 'new'}) RETURN n.phone",
            [["555-0100"]],
        ),
        (
            "MATCH (o:__Entity__ {id: 'old'}) SET o.role = 'partner'",
            "MATCH (n:__Entity__ {id: 'new'}) RETURN n.role",
            [["partner"]],
        ),
        (
            "MATCH (o:__Entity__ {id: 'old'}) SET o.source_chunk_ids = ['c1', 'c2']",
            "MATCH (n:__Entity__ {id: 'new'}) RETURN n.source_chunk_ids",
            [[["c5", "c1", "c2"]]],
        ),
        (
            "MATCH (n:__Entity__ {id: 'new'}) SET n.source_chunk_ids = ['c5', 'c9']",
            "MATCH (n:__Entity__ {id: 'new'}) RETURN n.source_chunk_ids",
            [[["c5", "c9", "c1"]]],
        ),
    ],
    ids=[
        "a property added",
        "a value changed",
        "provenance grew",
        "the new node's provenance grew",
    ],
)
async def test_a_rekey_fold_keeps_a_concurrent_edit(graph, monkeypatch, edit, check, expected):
    """A write between the fold's read and the fold — an ingest, a dedup merge
    into either node — was overwritten by what the read saw, or deleted with
    the old node. The fold checks what it was computed from, reads again when
    that changed, and unions provenance from the values at write time."""
    import graphrag_sdk.storage.graph_store as graph_store_module

    await _globex_and_initech(graph)
    real, reads = graph_store_module.read_for_merge, []

    async def read_then_edit(g, survivor, dup):
        read = await real(g, survivor, dup)
        reads.append(read)
        if len(reads) == 1:
            await graph.query_raw(edit)
        return read

    monkeypatch.setattr(graph_store_module, "read_for_merge", read_then_edit)
    assert await graph._move_entity("old", "new") == (0, 1)
    assert await rows(graph, check) == expected
    assert await rows(graph, "MATCH (n:__Entity__ {id: 'old'}) RETURN count(n)") == [[0]]


async def test_a_rekey_fold_carries_a_point_property(graph):
    """A point reads back as a map, which no property may hold: carried as
    read, FalkorDB refused the fold; it is copied on the server."""
    await _globex_and_initech(graph)
    await graph.query_raw(
        "MATCH (o:__Entity__ {id: 'old'}) SET o.site = point({latitude: 32.07, longitude: 34.78})"
    )
    assert await graph._move_entity("old", "new") == (0, 1)
    ((latitude,),) = await rows(graph, "MATCH (n:__Entity__ {id: 'new'}) RETURN n.site.latitude")
    assert latitude == pytest.approx(32.07)
