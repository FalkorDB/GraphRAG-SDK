"""Tests for retrieval/strategies/path_router.py and MultiPath path gating."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LatencyBudgetExceededError
from graphrag_sdk.retrieval.strategies import multi_path as mp_module
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval
from graphrag_sdk.retrieval.strategies.path_router import (
    RETRIEVAL_PATHS,
    HeuristicPathRouter,
    LLMPathRouter,
    all_paths,
    parse_paths,
)

from .conftest import MockEmbedder, MockLLM


# -- parse_paths / all_paths --


class TestParsePaths:
    def test_parses_comma_separated(self):
        assert parse_paths("relates, chunks") == {"relates", "chunks"}

    def test_parses_mixed_separators_and_case(self):
        assert parse_paths("RELATES\nexpansion  chunks") == {
            "relates",
            "expansion",
            "chunks",
        }

    def test_drops_unknown_tokens(self):
        assert parse_paths("relates, bogus, foo") == {"relates"}

    def test_empty_input_returns_empty_set(self):
        assert parse_paths("") == set()
        assert parse_paths("   ") == set()

    def test_all_paths_is_full_set(self):
        assert all_paths() == set(RETRIEVAL_PATHS)
        assert len(RETRIEVAL_PATHS) == 5


# -- LLMPathRouter --


class TestLLMPathRouter:
    async def test_returns_parsed_subset(self):
        router = LLMPathRouter(MockLLM(responses=["relates, expansion"]))
        plan = await router.plan("How are Alice and Bob connected?")
        assert plan == {"relates", "expansion"}

    async def test_empty_response_falls_back_to_all(self):
        router = LLMPathRouter(MockLLM(responses=[""]))
        plan = await router.plan("Who is Alice?")
        assert plan == all_paths()

    async def test_garbage_response_falls_back_to_all(self):
        router = LLMPathRouter(MockLLM(responses=["banana, foo, bar"]))
        plan = await router.plan("Who is Alice?")
        assert plan == all_paths()

    async def test_llm_exception_falls_back_to_all(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        router = LLMPathRouter(llm)
        plan = await router.plan("Who is Alice?")
        assert plan == all_paths()

    async def test_latency_budget_exceeded_propagates(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=LatencyBudgetExceededError("slow"))
        router = LLMPathRouter(llm)
        with pytest.raises(LatencyBudgetExceededError):
            await router.plan("Who is Alice?")


# -- HeuristicPathRouter --


class TestHeuristicPathRouter:
    async def test_always_includes_relates_and_chunks(self):
        plan = await HeuristicPathRouter().plan("xyz")
        assert {"relates", "chunks"} <= plan

    async def test_connection_query_adds_expansion(self):
        plan = await HeuristicPathRouter().plan(
            "how are Alice and Bob connected?"
        )
        assert "expansion" in plan

    async def test_proper_noun_adds_entity_paths(self):
        plan = await HeuristicPathRouter().plan("Tell me about Alice")
        assert {"entity_cypher", "entity_fulltext"} <= plan


# -- MultiPath gating --


def _make_strategy(router=None):
    graph = MagicMock()
    graph.query_raw = AsyncMock(return_value=MagicMock(result_set=[]))
    vector = MagicMock()
    vector.search_chunks = AsyncMock(return_value=[])
    vector.search_entities = AsyncMock(return_value=[])
    vector.search_relationships = AsyncMock(return_value=[])
    vector.fulltext_search_chunks = AsyncMock(return_value=[])
    vector.fulltext_search_entities = AsyncMock(return_value=[])
    return MultiPathRetrieval(
        graph_store=graph,
        vector_store=vector,
        embedder=MockEmbedder(dimension=8),
        llm=MockLLM(responses=["Alice"]),
        router=router,
    )


class _FixedRouter:
    def __init__(self, paths):
        self._paths = set(paths)

    async def plan(self, query, ctx=None):
        return self._paths


class TestMultiPathGating:
    async def test_router_prunes_paths(self, monkeypatch):
        calls = {}
        for name in (
            "search_relates_edges",
            "discover_entities",
            "expand_relationships",
            "retrieve_chunks",
        ):
            calls[name] = MagicMock()

        async def _relates(*a, **k):
            calls["search_relates_edges"]()
            return [], {}

        async def _discover(*a, **k):
            calls["discover_entities"]()
            return {}, {}

        async def _expand(*a, **k):
            calls["expand_relationships"]()
            return []

        async def _chunks(*a, **k):
            calls["retrieve_chunks"]()
            return {}, {}, {}

        monkeypatch.setattr(mp_module, "search_relates_edges", _relates)
        monkeypatch.setattr(mp_module, "discover_entities", _discover)
        monkeypatch.setattr(mp_module, "expand_relationships", _expand)
        monkeypatch.setattr(mp_module, "retrieve_chunks", _chunks)

        strategy = _make_strategy(router=_FixedRouter({"relates"}))
        await strategy.search("Who is Alice?")

        assert calls["search_relates_edges"].called
        assert not calls["discover_entities"].called
        assert not calls["expand_relationships"].called
        assert not calls["retrieve_chunks"].called

    async def test_no_router_runs_all_paths(self, monkeypatch):
        called = set()

        async def _relates(*a, **k):
            called.add("relates")
            return [], {}

        async def _discover(*a, **k):
            called.add("discover")
            return {}, {}

        async def _expand(*a, **k):
            called.add("expand")
            return []

        async def _chunks(*a, **k):
            called.add("chunks")
            return {}, {}, {}

        monkeypatch.setattr(mp_module, "search_relates_edges", _relates)
        monkeypatch.setattr(mp_module, "discover_entities", _discover)
        monkeypatch.setattr(mp_module, "expand_relationships", _expand)
        monkeypatch.setattr(mp_module, "retrieve_chunks", _chunks)

        strategy = _make_strategy(router=None)
        await strategy.search("Who is Alice?")

        assert called == {"relates", "discover", "expand", "chunks"}

    async def test_router_failure_falls_back_to_all_paths(self, monkeypatch):
        called = set()

        async def _relates(*a, **k):
            called.add("relates")
            return [], {}

        async def _discover(*a, **k):
            called.add("discover")
            return {}, {}

        async def _expand(*a, **k):
            called.add("expand")
            return []

        async def _chunks(*a, **k):
            called.add("chunks")
            return {}, {}, {}

        monkeypatch.setattr(mp_module, "search_relates_edges", _relates)
        monkeypatch.setattr(mp_module, "discover_entities", _discover)
        monkeypatch.setattr(mp_module, "expand_relationships", _expand)
        monkeypatch.setattr(mp_module, "retrieve_chunks", _chunks)

        class _BoomRouter:
            async def plan(self, query, ctx=None):
                raise RuntimeError("router down")

        strategy = _make_strategy(router=_BoomRouter())
        await strategy.search("Who is Alice?")

        assert called == {"relates", "discover", "expand", "chunks"}
