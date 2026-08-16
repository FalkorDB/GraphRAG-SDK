"""Tests for retrieval/graph_walk.py — DynamicGraphWalk (Phase 3.3)."""
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


class FakeWalkStore:
    def __init__(self, adj: dict[str, list[tuple[str, float, str]]], weights=None):
        self._adj = adj
        self._weights = weights or {}

    async def weighted_neighbors(self, node_id: str, *, limit: int = 50):
        return self._adj.get(node_id, [])

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
