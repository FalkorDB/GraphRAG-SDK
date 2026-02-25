"""Tests for retrieval/router.py — SemanticRouter."""
from __future__ import annotations

from typing import Any

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import RawSearchResult, RetrieverResult
from graphrag_sdk.retrieval.router import SemanticRouter
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy


class NamedStrategy(RetrievalStrategy):
    """Strategy that returns its name in results."""

    def __init__(self, name: str):
        super().__init__()
        self._name = name

    async def _execute(self, query: str, ctx: Context, **kwargs: Any) -> RawSearchResult:
        return RawSearchResult(records=[f"from-{self._name}"], metadata={"name": self._name})


class TestSemanticRouter:
    async def test_default_strategy(self, ctx):
        default = NamedStrategy("default")
        router = SemanticRouter(default_strategy=default)
        result = await router.route("any query", ctx)
        assert isinstance(result, RetrieverResult)
        assert "from-default" in result.items[0].content

    async def test_condition_routing(self, ctx):
        default = NamedStrategy("default")
        local = NamedStrategy("local")
        router = SemanticRouter(default_strategy=default)
        router.register("local", local, condition=lambda q: "local" in q.lower())
        result = await router.route("use local search", ctx)
        assert "from-local" in result.items[0].content

    async def test_no_match_uses_default(self, ctx):
        default = NamedStrategy("default")
        specific = NamedStrategy("specific")
        router = SemanticRouter(default_strategy=default)
        router.register("specific", specific, condition=lambda q: "NEVER_MATCH" in q)
        result = await router.route("normal query", ctx)
        assert "from-default" in result.items[0].content

    async def test_first_match_wins(self, ctx):
        default = NamedStrategy("default")
        s1 = NamedStrategy("first")
        s2 = NamedStrategy("second")
        router = SemanticRouter(default_strategy=default)
        router.register("first", s1, condition=lambda q: True)
        router.register("second", s2, condition=lambda q: True)
        result = await router.route("anything", ctx)
        assert "from-first" in result.items[0].content

    async def test_condition_exception_skipped(self, ctx):
        """If a condition raises, it's skipped."""
        default = NamedStrategy("default")
        bad = NamedStrategy("bad")
        router = SemanticRouter(default_strategy=default)
        router.register("bad", bad, condition=lambda q: 1 / 0)  # raises ZeroDivisionError
        result = await router.route("test", ctx)
        assert "from-default" in result.items[0].content

    async def test_default_context(self):
        default = NamedStrategy("default")
        router = SemanticRouter(default_strategy=default)
        result = await router.route("test")  # no ctx
        assert isinstance(result, RetrieverResult)

    async def test_register_multiple(self, ctx):
        default = NamedStrategy("default")
        router = SemanticRouter(default_strategy=default)
        router.register("a", NamedStrategy("a"), condition=lambda q: "a" in q)
        router.register("b", NamedStrategy("b"), condition=lambda q: "b" in q)
        result_a = await router.route("query a", ctx)
        result_b = await router.route("query b", ctx)
        assert "from-a" in result_a.items[0].content
        assert "from-b" in result_b.items[0].content
