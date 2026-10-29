"""Tests for retrieval/graph_walk.py — DynamicGraphWalk."""

from __future__ import annotations

from typing import Any

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import ScoredPath
from graphrag_sdk.retrieval.graph_walk import (
    DynamicGraphWalk,
    GraphWalkRetrieval,
    score_path,
)


def make_neighbor_fn(adj: dict[str, list[tuple[str, float, str]]]):
    async def neighbor_fn(node_id: str) -> list[tuple[str, float, str]]:
        return adj.get(node_id, [])

    return neighbor_fn


class TestScorePath:
    def test_empty_path_scores_zero(self):
        assert score_path([]) == 0.0

    def test_length_penalty_prefers_short_paths(self):
        weights = {"a": 1.0, "b": 1.0, "c": 1.0}
        short = score_path(["a", "b"], weights, [1.0], length_penalty=0.5)
        long = score_path(["a", "b", "c"], weights, [1.0, 1.0], length_penalty=0.5)
        assert short > long

    def test_node_weights_increase_score(self):
        high = score_path(["a", "b"], {"a": 5.0, "b": 5.0}, [1.0])
        low = score_path(["a", "b"], {"a": 0.1, "b": 0.1}, [1.0])
        assert high > low


class TestBeamSearch:
    async def test_beam_width_caps_results(self, ctx: Context):
        adj = {"start": [(f"n{i}", 1.0, "REL") for i in range(20)]}
        walk = DynamicGraphWalk(make_neighbor_fn(adj), beam_width=3, max_depth=2)
        paths = await walk.beam_search("start", ctx=ctx)
        assert len(paths) <= 3
        assert all(isinstance(p, ScoredPath) for p in paths)

    async def test_low_scoring_goal_is_not_pruned_by_the_beam(self, ctx: Context):
        # The goal is a direct neighbour but scores below the other neighbour;
        # with beam_width=1 it used to be cut before the goal check.
        adj = {"a": [("x", 1.0, "R"), ("goal", 0.01, "R")], "x": [("y", 1.0, "R")]}
        walk = DynamicGraphWalk(make_neighbor_fn(adj), beam_width=1, max_depth=3)
        paths = await walk.beam_search("a", goal="goal", ctx=ctx)
        assert paths and paths[0].nodes == ["a", "goal"]

    async def test_reaches_goal(self, ctx: Context):
        adj = {
            "a": [("b", 1.0, "R")],
            "b": [("c", 1.0, "R")],
            "c": [("d", 1.0, "R")],
        }
        walk = DynamicGraphWalk(make_neighbor_fn(adj), beam_width=5, max_depth=5)
        paths = await walk.beam_search("a", goal="d", ctx=ctx)
        assert paths
        assert paths[0].nodes[0] == "a"
        assert paths[0].nodes[-1] == "d"

    async def test_no_cycles_within_path(self, ctx: Context):
        adj = {"a": [("b", 1.0, "R")], "b": [("a", 1.0, "R")]}
        walk = DynamicGraphWalk(make_neighbor_fn(adj), beam_width=5, max_depth=4)
        paths = await walk.beam_search("a", ctx=ctx)
        for p in paths:
            assert len(p.nodes) == len(set(p.nodes))

    async def test_higher_weighted_path_ranks_first(self, ctx: Context):
        adj = {"start": [("hi", 1.0, "R"), ("lo", 1.0, "R")]}
        weights = {"hi": 10.0, "lo": 0.0}
        walk = DynamicGraphWalk(
            make_neighbor_fn(adj), node_weights=weights, beam_width=2, max_depth=1
        )
        paths = await walk.beam_search("start", ctx=ctx)
        assert paths[0].nodes[-1] == "hi"

    def test_invalid_params_rejected(self):
        with pytest.raises(ValueError):
            DynamicGraphWalk(make_neighbor_fn({}), beam_width=0)
        with pytest.raises(ValueError):
            DynamicGraphWalk(make_neighbor_fn({}), max_depth=0)


class TestBidirectionalSearch:
    async def test_same_start_and_goal(self, ctx: Context):
        walk = DynamicGraphWalk(make_neighbor_fn({}), max_depth=3)
        path = await walk.bidirectional_search("a", "a", ctx=ctx)
        assert path is not None
        assert path.nodes == ["a"]

    async def test_finds_connecting_path(self, ctx: Context):
        adj = {
            "a": [("b", 1.0, "R")],
            "b": [("a", 1.0, "R"), ("c", 1.0, "R")],
            "c": [("b", 1.0, "R")],
        }
        walk = DynamicGraphWalk(make_neighbor_fn(adj), max_depth=4)
        path = await walk.bidirectional_search("a", "c", ctx=ctx)
        assert path is not None
        assert path.nodes[0] == "a"
        assert path.nodes[-1] == "c"
        assert "b" in path.nodes

    async def test_returns_none_when_disconnected(self, ctx: Context):
        adj = {"a": [("b", 1.0, "R")], "b": [("a", 1.0, "R")], "x": [("y", 1.0, "R")]}
        walk = DynamicGraphWalk(make_neighbor_fn(adj), max_depth=3)
        path = await walk.bidirectional_search("a", "x", ctx=ctx)
        assert path is None


class TestEdgeDirection:
    """Direction travels with every hop: True when the edge is stored from the
    node before it to the node after it on the path."""

    async def test_beam_search_carries_each_edges_direction(self):
        # Stored: alice-[WORKS_AT]->acme, acme-[IN]->berlin; walk from acme.
        adj = {
            "acme": [("alice", 1.0, "WORKS_AT", False), ("berlin", 1.0, "IN", True)],
        }
        paths = await DynamicGraphWalk(make_neighbor_fn(adj), max_depth=1).beam_search("acme")
        by_end = {p.nodes[-1]: p.edge_forward for p in paths}
        assert by_end == {"alice": [False], "berlin": [True]}

    async def test_three_tuples_leave_the_direction_unknown(self):
        adj = {"a": [("b", 1.0, "R")]}
        paths = await DynamicGraphWalk(make_neighbor_fn(adj), max_depth=1).beam_search("a")
        assert paths[0].edge_forward == [None]

    async def test_goal_side_edges_are_flipped_into_path_order(self):
        # Stored: alice-[WORKS_AT]->acme, acme-[IN]->berlin. Searched from
        # berlin to alice, both hops point against the path.
        adj = {
            "berlin": [("acme", 1.0, "IN", False)],
            "acme": [("berlin", 1.0, "IN", True), ("alice", 1.0, "WORKS_AT", False)],
            "alice": [("acme", 1.0, "WORKS_AT", True)],
        }
        walk = DynamicGraphWalk(make_neighbor_fn(adj), max_depth=2)
        path = await walk.bidirectional_search("berlin", "alice")
        assert path.nodes == ["berlin", "acme", "alice"]
        assert path.edges == ["IN", "WORKS_AT"] and path.edge_forward == [False, False]
        back = await walk.bidirectional_search("alice", "berlin")
        assert back.nodes == ["alice", "acme", "berlin"] and back.edge_forward == [True, True]


def _star(fanout: int, depth: int) -> dict[str, list[tuple[str, float, str]]]:
    """Every node below ``depth`` has ``fanout`` children (and its parent)."""
    adj: dict[str, list[tuple[str, float, str]]] = {}
    level = ["root"]
    for _ in range(depth):
        nxt = []
        for node in level:
            kids = [f"{node}.{i}" for i in range(fanout)]
            adj.setdefault(node, []).extend((k, 1.0, "R") for k in kids)
            for k in kids:
                adj.setdefault(k, []).append((node, 1.0, "R"))
            nxt.extend(kids)
        level = nxt
    return adj


class TestBoundedBidirectionalSearch:
    async def test_beam_bounds_lookups(self):
        adj = _star(50, 3)
        calls: list[str] = []

        async def neighbours(node: str):
            calls.append(node)
            return adj.get(node, [])

        walk = DynamicGraphWalk(neighbours, beam_width=5, max_depth=5)
        assert await walk.bidirectional_search("root", "nowhere") is None
        # ceil(5/2)=3 levels from start (1 + 5 + 5 nodes), 2 from the goal
        # (1 node, no neighbours): 12 lookups, not thousands.
        assert len(calls) == 12 and walk.stopped == "complete"

    async def test_expansion_cap_stops_the_search(self):
        adj = _star(50, 3)
        walk = DynamicGraphWalk(make_neighbor_fn(adj), beam_width=20, max_depth=5, max_expansions=7)
        assert await walk.bidirectional_search("root", "nowhere") is None
        assert walk.stopped == "expansion_limit"

    async def test_time_budget_stops_the_search(self):
        import asyncio

        async def slow(node: str):
            await asyncio.sleep(0.05)
            return [(node + "x", 1.0, "R")]

        walk = DynamicGraphWalk(slow, max_depth=5, time_budget_seconds=0.08)
        assert await walk.bidirectional_search("a", "z") is None
        assert walk.stopped == "time_budget"

    @pytest.mark.parametrize("max_depth, found", [(1, False), (2, True), (3, True)])
    async def test_max_depth_bounds_the_whole_path(self, max_depth, found):
        # a - b - c: the only path has 2 hops.
        adj = {
            "a": [("b", 1.0, "R")],
            "b": [("a", 1.0, "R"), ("c", 1.0, "R")],
            "c": [("b", 1.0, "R")],
        }
        walk = DynamicGraphWalk(make_neighbor_fn(adj), max_depth=max_depth)
        path = await walk.bidirectional_search("a", "c")
        assert (path is not None) == found
        if path is not None:
            assert path.length <= max_depth

    @pytest.mark.parametrize("max_depth", [1, 2, 3, 4, 5])
    async def test_no_path_exceeds_max_depth(self, max_depth):
        chain = [f"n{i}" for i in range(8)]
        adj: dict[str, list[tuple[str, float, str]]] = {}
        for x, y in zip(chain, chain[1:]):
            adj.setdefault(x, []).append((y, 1.0, "R"))
            adj.setdefault(y, []).append((x, 1.0, "R"))
        walk = DynamicGraphWalk(make_neighbor_fn(adj), max_depth=max_depth)
        for hops in range(1, 8):
            path = await walk.bidirectional_search("n0", chain[hops])
            assert (path is not None) == (hops <= max_depth)

    async def test_best_scoring_meeting_path_wins(self):
        # Two 2-hop routes a-?-c; the one through "strong" has heavier edges
        # but is listed second.
        adj = {
            "a": [("weak", 0.1, "R"), ("strong", 1.0, "R")],
            "weak": [("a", 0.1, "R"), ("c", 0.1, "R")],
            "strong": [("a", 1.0, "R"), ("c", 1.0, "R")],
            "c": [("weak", 0.1, "R"), ("strong", 1.0, "R")],
        }
        walk = DynamicGraphWalk(make_neighbor_fn(adj), max_depth=2)
        path = await walk.bidirectional_search("a", "c")
        assert path.nodes == ["a", "strong", "c"]

    async def test_lookup_errors_propagate(self):
        async def broken(node: str):
            raise RuntimeError("db down")

        walk = DynamicGraphWalk(broken, max_depth=2)
        with pytest.raises(RuntimeError, match="db down"):
            await walk.bidirectional_search("a", "b")

    async def test_lookup_timeout_is_not_reported_as_the_time_budget(self):
        import asyncio

        async def timed_out(node: str):
            raise asyncio.TimeoutError("db")

        walk = DynamicGraphWalk(timed_out, max_depth=2)
        with pytest.raises(asyncio.TimeoutError, match="db"):
            await walk.bidirectional_search("a", "b")
        assert walk.stopped == "complete"


class FakeWalkStore:
    def __init__(self, adj: dict[str, list[tuple[str, float, str]]], weights=None):
        self._adj = adj
        self._weights = weights or {}

    async def weighted_neighbors(
        self, node_id: str, *, limit: int = 50, with_direction: bool = False
    ):
        items = self._adj.get(node_id, [])
        return items if with_direction else [tuple(item[:3]) for item in items]

    async def pagerank(self, **kwargs: Any):
        return self._weights


class TestGraphWalkRetrieval:
    async def test_returns_paths_as_items(self, ctx: Context):
        adj = {"alice": [("acme", 1.0, "WORKS_AT")], "acme": [("bob", 1.0, "WORKS_AT")]}
        store = FakeWalkStore(adj)

        async def seed_fn(query: str, ctx: Context) -> list[str]:
            return ["alice"]

        strat = GraphWalkRetrieval(store, seed_fn, max_depth=2, use_pagerank=False)
        result = await strat.search("who works with alice", ctx)
        assert result.items
        assert "alice" in result.items[0].content

    async def test_no_seeds_returns_empty(self, ctx: Context):
        store = FakeWalkStore({})

        async def seed_fn(query: str, ctx: Context) -> list[str]:
            return []

        strat = GraphWalkRetrieval(store, seed_fn)
        result = await strat.search("q", ctx)
        assert result.items == []

    def test_negative_max_paths_rejected(self):
        async def seed_fn(query: str, ctx: Context) -> list[str]:
            return []

        with pytest.raises(ValueError, match="max_paths"):
            GraphWalkRetrieval(FakeWalkStore({}), seed_fn, max_paths=-1)

    @pytest.mark.parametrize(
        "outgoing, expected",
        [
            (True, "alice -[WORKS_AT]-> acme"),
            (False, "alice <-[WORKS_AT]- acme"),
            (None, "alice -[WORKS_AT]- acme"),
        ],
    )
    async def test_content_shows_relation_labels_and_direction(
        self, ctx: Context, outgoing, expected
    ):
        store = FakeWalkStore({"alice": [("acme", 1.0, "WORKS_AT", outgoing)]})

        async def seed_fn(query: str, ctx: Context) -> list[str]:
            return ["alice"]

        strat = GraphWalkRetrieval(store, seed_fn, max_depth=1, use_pagerank=False)
        result = await strat.search("q", ctx)
        assert [item.content for item in result.items] == [expected]

    async def test_pagerank_weights_rank_paths(self, ctx: Context):
        adj = {"s": [("a", 1.0, "R"), ("b", 1.0, "R")]}
        store = FakeWalkStore(adj, weights={"b": 1.0})

        async def seed_fn(query: str, ctx: Context) -> list[str]:
            return ["s"]

        strat = GraphWalkRetrieval(store, seed_fn, max_depth=1, use_pagerank=True)
        result = await strat.search("q", ctx)
        assert result.items[0].content.endswith(" b")
        # Without PageRank the equal-weight neighbours keep store order.
        plain = GraphWalkRetrieval(store, seed_fn, max_depth=1, use_pagerank=False)
        assert (await plain.search("q", ctx)).items[0].content.endswith(" a")

    async def test_paths_from_every_seed_are_truncated_to_max_paths(self, ctx: Context):
        adj = {
            "s1": [(f"a{i}", 1.0, "R") for i in range(3)],
            "s2": [(f"b{i}", 1.0, "R") for i in range(3)],
        }
        store = FakeWalkStore(adj, weights={"b2": 5.0, "a1": 4.0})

        async def seed_fn(query: str, ctx: Context) -> list[str]:
            return ["s1", "s2"]

        strat = GraphWalkRetrieval(store, seed_fn, max_depth=1, max_paths=2)
        result = await strat.search("q", ctx)
        assert [item.content for item in result.items] == ["s2 -[R]- b2", "s1 -[R]- a1"]
        assert result.metadata["num_paths"] == 2


class TestWeightedNeighborsQuery:
    async def test_query_is_deterministic_and_skips_bookkeeping_edges(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from graphrag_sdk.storage.graph_store import GraphStore

        conn = MagicMock()
        conn.query = AsyncMock(
            return_value=SimpleNamespace(result_set=[["acme", "WORKS_AT", 2.0], [None, "X", 1.0]])
        )
        out = await GraphStore(conn).weighted_neighbors("alice", limit=7)
        cypher = conn.query.await_args.args[0]
        assert conn.query.await_args.kwargs.get("read_only") is True
        assert "NOT type(r) IN ['SAME_AS', 'DISTINCT_FROM']" in cypher
        assert "ORDER BY weight DESC, id LIMIT 7" in cypher
        assert "max(w) AS weight" in cypher
        # Equal-weight parallel edges always yield the same label.
        assert "ORDER BY w DESC, coalesce(r.rel_type, type(r))" in cypher
        assert out == [("acme", 2.0, "WORKS_AT")]

    async def test_with_direction_reports_the_stored_direction(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from graphrag_sdk.storage.graph_store import GraphStore

        conn = MagicMock()
        conn.query = AsyncMock(
            return_value=SimpleNamespace(
                result_set=[["acme", "WORKS_AT", 2.0, True], ["bob", "KNOWS", 1.0, False]]
            )
        )
        out = await GraphStore(conn).weighted_neighbors("alice", with_direction=True)
        assert "ID(startNode(r)) = ID(n)" in conn.query.await_args.args[0]
        assert out == [("acme", 2.0, "WORKS_AT", True), ("bob", 1.0, "KNOWS", False)]

    async def test_database_errors_propagate(self):
        from unittest.mock import AsyncMock, MagicMock

        from graphrag_sdk.storage.graph_store import GraphStore

        conn = MagicMock()
        conn.query = AsyncMock(side_effect=RuntimeError("connection refused"))
        with pytest.raises(RuntimeError, match="connection refused"):
            await GraphStore(conn).weighted_neighbors("alice")


# ── Concurrent neighbour lookups ─────────────────────────────────


async def test_beam_round_fetches_neighbours_concurrently():
    import asyncio

    active = 0
    peak = 0

    async def neighbours(node: str):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        if node == "s":
            return [(f"n{i}", 1.0, "R") for i in range(4)]
        return [(f"{node}x", 1.0, "R")] if len(node) < 4 else []

    paths = await DynamicGraphWalk(neighbours, beam_width=4, max_depth=2).beam_search("s")
    assert peak == 4
    assert sorted(p.nodes[-1] for p in paths) == ["n0x", "n1x", "n2x", "n3x"]


async def test_walk_limits_concurrent_lookups():
    import asyncio

    active = 0
    peak = 0

    async def neighbours(node: str):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.005)
        active -= 1
        return [(f"{node}.{i}", 1.0, "R") for i in range(40)] if node == "s" else []

    walk = DynamicGraphWalk(neighbours, beam_width=40, max_depth=2, max_concurrency=5)
    await walk.beam_search("s")
    assert peak == 5


@pytest.mark.parametrize("value", [0, -3])
async def test_max_concurrency_is_at_least_one(value):
    seen: list[str] = []

    async def neighbours(node: str):
        seen.append(node)
        return [("t", 1.0, "R")] if node == "s" else []

    walk = DynamicGraphWalk(neighbours, max_depth=2, max_concurrency=value)
    assert walk._max_concurrency == 1
    await walk.beam_search("s")
    assert seen == ["s", "t"]
