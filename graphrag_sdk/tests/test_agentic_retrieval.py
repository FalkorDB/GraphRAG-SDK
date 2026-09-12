"""Tests for retrieval/agentic — ReAct loop and tools (Phase 3.1)."""
from __future__ import annotations

from typing import Any

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import RetrieverResult, RetrieverResultItem
from graphrag_sdk.retrieval.agentic import (
    AgenticRetrieval,
    ToolRegistry,
    build_default_registry,
    is_read_only_cypher,
    parse_react_step,
)
from graphrag_sdk.retrieval.agentic.tools import Tool, make_search_tool


class ScriptedLLM:
    """LLM returning a fixed list of ReAct turns in order."""

    def __init__(self, turns: list[str]):
        self._turns = turns
        self._i = 0
        self.model_name = "scripted"

    async def ainvoke(self, prompt: str, **kwargs: Any):
        from graphrag_sdk.core.models import LLMResponse

        text = self._turns[min(self._i, len(self._turns) - 1)]
        self._i += 1
        return LLMResponse(content=text)


class FakeStrategy:
    async def search(self, query: str, ctx: Context) -> RetrieverResult:
        return RetrieverResult(
            items=[RetrieverResultItem(content="Alice works at Acme Corp.")]
        )


class TestParseReactStep:
    def test_parses_action_and_input(self):
        text = 'Thought: search now\nAction: search\nAction Input: {"query": "alice"}'
        parsed = parse_react_step(text)
        assert parsed["action"] == "search"
        assert parsed["action_input"] == {"query": "alice"}

    def test_parses_final_answer(self):
        parsed = parse_react_step("Thought: done\nFinal Answer: 42")
        assert parsed["final_answer"] == "42"

    def test_malformed_json_input_is_empty(self):
        parsed = parse_react_step("Action: search\nAction Input: {not json}")
        assert parsed["action"] == "search"
        assert parsed["action_input"] == {}


class TestReadOnlyCypher:
    def test_allows_match(self):
        assert is_read_only_cypher("MATCH (n) RETURN n")

    @pytest.mark.parametrize(
        "cypher",
        ["CREATE (n)", "MATCH (n) DELETE n", "MERGE (n)", "MATCH (n) SET n.x = 1"],
    )
    def test_rejects_writes(self, cypher: str):
        assert not is_read_only_cypher(cypher)


class TestToolRegistry:
    async def test_unknown_tool_returns_error(self, ctx: Context):
        reg = ToolRegistry()
        out = await reg.run("nope", {}, ctx)
        assert "unknown tool" in out.lower()

    async def test_handler_exception_is_caught(self, ctx: Context):
        async def boom(inp: dict, ctx: Context) -> str:
            raise RuntimeError("kaboom")

        reg = ToolRegistry()
        reg.register(Tool("boom", "desc", boom))
        out = await reg.run("boom", {}, ctx)
        assert "kaboom" in out

    async def test_search_tool_returns_snippets(self, ctx: Context):
        tool = make_search_tool(FakeStrategy())
        out = await tool.handler({"query": "alice"}, ctx)
        assert "Acme" in out

    def test_default_registry_only_search_without_store(self):
        reg = build_default_registry(strategy=FakeStrategy(), graph_store=None)
        assert reg.names() == ["search"]


class TestAgenticRetrieval:
    async def test_loop_runs_tool_then_answers(self, ctx: Context):
        llm = ScriptedLLM([
            'Thought: search\nAction: search\nAction Input: {"query": "alice"}',
            "Thought: I now have enough information.\nFinal Answer: Alice works at Acme.",
        ])
        agent = AgenticRetrieval(llm, strategy=FakeStrategy(), max_steps=4)
        result = await agent.search("where does alice work?", ctx)
        assert result.metadata["stop_reason"] == "final_answer"
        assert "Acme" in result.metadata["answer"]
        assert result.metadata["num_steps"] >= 1
        assert any("Acme" in item.content for item in result.items)

    async def test_stops_on_exhausted_budget(self):
        llm = ScriptedLLM(["Action: search\nAction Input: {}"])
        agent = AgenticRetrieval(llm, strategy=FakeStrategy(), max_steps=4)
        ctx = Context(latency_budget_ms=0.0)
        result = await agent.search("q", ctx)
        assert result.metadata["stop_reason"] == "budget_exceeded"
        assert result.metadata["num_steps"] == 0

    async def test_stops_when_no_action(self, ctx: Context):
        llm = ScriptedLLM(["Thought: hmm, nothing to do here."])
        agent = AgenticRetrieval(llm, strategy=FakeStrategy(), max_steps=3)
        result = await agent.search("q", ctx)
        assert result.metadata["stop_reason"] == "no_action"

    async def test_respects_max_steps(self, ctx: Context):
        # Always emits an action, never a final answer → should hit the cap.
        llm = ScriptedLLM(['Action: search\nAction Input: {"query": "x"}'])
        agent = AgenticRetrieval(llm, strategy=FakeStrategy(), max_steps=2)
        result = await agent.search("q", ctx)
        assert result.metadata["stop_reason"] == "max_steps"
        assert result.metadata["num_steps"] == 2

    def test_invalid_max_steps(self):
        with pytest.raises(ValueError):
            AgenticRetrieval(ScriptedLLM([]), max_steps=0)
