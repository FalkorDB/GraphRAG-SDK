"""Tests for AgentLimits: run bounds and the tool ceilings they set."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import LLMResponse, RetrieverResult, RetrieverResultItem, ToolCall
from graphrag_sdk.retrieval.agentic import (
    AgenticRetrieval,
    AgentLimits,
    Tool,
    ToolContext,
    ToolRegistry,
    build_default_registry,
)
from graphrag_sdk.retrieval.agentic.graph_tools import make_search_tool
from graphrag_sdk.retrieval.agentic.limits import truncate


class Strategy:
    def __init__(self) -> None:
        self.kwargs: list[dict[str, Any]] = []

    async def search(self, query: str, ctx: Context, **kwargs: Any) -> RetrieverResult:
        self.kwargs.append(kwargs)
        return RetrieverResult(items=[RetrieverResultItem(content=f"fact about {query}")])


class LLM:
    supports_tool_calling = True
    model_name = "m"

    def __init__(self, responses: list[LLMResponse], final: str = "Answer [1]."):
        self.responses = responses
        self.final = final
        self.requests: list[dict[str, Any]] = []

    async def ainvoke_with_tools(self, messages, tools, **kwargs):
        self.requests.append({"messages": list(messages), **kwargs})
        if kwargs.get("tool_choice") == "none":
            return LLMResponse(content=self.final)
        return self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]


def search_call(i: int) -> ToolCall:
    return ToolCall(id=f"c{i}", name="search", arguments={"query": f"q{i}"})


class TestAgentLimits:
    def test_defaults(self):
        limits = AgentLimits()
        assert limits.max_steps == 6 and limits.query_max_rows == 25
        assert limits.traverse_max_expansions == 200
        assert limits.to_dict()["max_tool_calls"] == 12

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
    def test_rejects_bad_values(self, bad):
        with pytest.raises(ValueError, match="max_tool_calls"):
            AgentLimits(max_tool_calls=bad)

    def test_history_may_be_zero(self):
        assert AgentLimits(max_history_turns=0).max_history_turns == 0

    def test_traverse_timeout_floor(self):
        with pytest.raises(ValueError):
            AgentLimits(traverse_max_timeout_ms=10)

    def test_shortcuts_override_limits(self):
        agent = AgenticRetrieval(
            LLM([]),
            strategy=Strategy(),
            limits=AgentLimits(max_steps=9, max_tool_calls=3),
            max_steps=2,
            history_turns=1,
        )
        assert agent.limits.max_steps == 2
        assert agent.limits.max_history_turns == 1
        assert agent.limits.max_tool_calls == 3

    def test_truncate(self):
        assert truncate("abc", 5) == "abc"
        assert truncate("abcdef", 3) == "abc…[truncated 3 characters]"


class TestRunLimits:
    async def test_total_tool_calls_are_capped(self, ctx):
        llm = LLM([LLMResponse(content="", tool_calls=[search_call(1)])])
        agent = AgenticRetrieval(
            llm, strategy=Strategy(), limits=AgentLimits(max_steps=10, max_tool_calls=2)
        )
        result = await agent.search("q", ctx)
        md = result.metadata
        assert md["stop_reason"] == "max_tool_calls"
        assert md["tool_calls"] == 2 and md["num_steps"] == 2
        assert llm.requests[-1]["tool_choice"] == "none"
        assert md["answer"] == "Answer [1]."
        assert md["limits"]["max_tool_calls"] == 2

    async def test_calls_beyond_the_turn_cap_are_refused(self, ctx):
        llm = LLM(
            [
                LLMResponse(
                    content="", tool_calls=[search_call(1), search_call(2), search_call(3)]
                ),
                LLMResponse(content="Done [1]."),
            ]
        )
        strategy = Strategy()
        agent = AgenticRetrieval(llm, strategy=strategy, limits=AgentLimits(max_calls_per_turn=2))
        result = await agent.search("q", ctx)
        steps = result.metadata["agent_trace"]["steps"]
        assert [s["status"] for s in steps] == ["ok", "ok", "refused"]
        assert "only 2 tool calls run per turn" in steps[2]["observation"]
        assert len(strategy.kwargs) == 2
        tool_messages = [m for m in llm.requests[1]["messages"] if m.role == "tool"]
        assert [m.tool_call_id for m in tool_messages] == ["c1", "c2", "c3"]

    async def test_react_tool_calls_are_capped(self, ctx):
        class Text:
            supports_tool_calling = False
            model_name = "t"

            def __init__(self):
                self.prompts: list[str] = []

            async def ainvoke(self, prompt, **kw):
                self.prompts.append(prompt)
                if prompt.rstrip().endswith("Final Answer:"):
                    return LLMResponse(content="Final Answer: enough [1]")
                return LLMResponse(content='Action: search\nAction Input: {"query": "x"}')

        llm = Text()
        agent = AgenticRetrieval(
            llm, strategy=Strategy(), limits=AgentLimits(max_steps=10, max_tool_calls=3)
        )
        result = await agent.search("q", ctx)
        assert result.metadata["stop_reason"] == "max_tool_calls"
        assert result.metadata["num_steps"] == 3
        assert result.metadata["answer"] == "enough [1]"

    async def test_observations_sent_to_the_model_are_truncated(self, ctx):
        async def big(args, tctx):
            return "x" * 500

        registry = ToolRegistry()
        registry.register(Tool("big", "Big output.", big))
        llm = LLM(
            [
                LLMResponse(content="", tool_calls=[ToolCall(id="b", name="big", arguments={})]),
                LLMResponse(content="ok [1]"),
            ]
        )
        agent = AgenticRetrieval(
            llm, registry=registry, limits=AgentLimits(max_observation_chars=50)
        )
        result = await agent.search("q", ctx)
        tool_msg = [m for m in llm.requests[1]["messages"] if m.role == "tool"][0]
        assert len(tool_msg.content) < 100 and "truncated" in tool_msg.content
        assert len(result.metadata["evidence"][0]["content"]) == 500  # evidence keeps it all

    async def test_trace_is_bounded(self, ctx):
        llm = LLM(
            [
                LLMResponse(
                    content="t" * 100,
                    tool_calls=[ToolCall(id="c", name="search", arguments={"query": "y" * 300})],
                ),
                LLMResponse(content="ok [1]"),
            ]
        )
        agent = AgenticRetrieval(llm, strategy=Strategy(), limits=AgentLimits(max_trace_chars=40))
        step = (await agent.search("q", ctx)).metadata["agent_trace"]["steps"][0]
        assert set(step["action_input"]) == {"_truncated"}
        assert len(step["action_input"]["_truncated"]) == 40
        assert "truncated" in step["observation"] and "truncated" in step["thought"]


class TestToolCeilings:
    async def test_search_limits_are_clamped(self):
        strategy = Strategy()
        tool = make_search_tool(strategy, max_limit=30)
        await tool.handler(
            {"query": "q", "chunk_top_k": 10_000, "max_passages_out": 0, "max_entities": -5},
            ToolContext(ctx=Context()),
        )
        assert strategy.kwargs[0] == {"chunk_top_k": 30, "max_passages_out": 0, "max_entities": 1}

    async def test_search_item_cap(self):
        class Many:
            async def search(self, q, ctx, **kw):
                return RetrieverResult(
                    items=[RetrieverResultItem(content=f"r{i}") for i in range(10)]
                )

        out = await make_search_tool(Many(), max_items=3).handler(
            {"query": "q"}, ToolContext(ctx=Context())
        )
        assert out.content.count("\n") == 2

    async def test_registry_tools_follow_the_limits(self):
        calls: list[dict[str, Any]] = []

        class Store:
            async def query_raw(self, cypher, params=None, *, read_only=False, timeout=None):
                calls.append({"cypher": cypher, "params": params or {}, "timeout": timeout})
                if "WHERE e.id = $k" in cypher:
                    return SimpleNamespace(result_set=[["a", "A"]], header=[])
                return SimpleNamespace(result_set=[], header=[])

        class Gen:
            async def ainvoke(self, prompt, **kw):
                return LLMResponse(content="```cypher\nMATCH (n) RETURN n.name LIMIT 500\n```")

        limits = AgentLimits(query_max_rows=5, query_timeout_ms=1234, traverse_max_expansions=7)
        registry = build_default_registry(
            graph_store=Store(), llm=Gen(), include_skills=False, limits=limits
        )
        state = ToolContext(ctx=Context(latency_budget_ms=10_000.0))
        await registry.run("query_graph", {"question": "names"}, state)
        query = next(c for c in calls if c["cypher"].startswith("MATCH (n)"))
        assert "LIMIT 5" in query["cypher"] and query["timeout"] == 1234
        await registry.run("traverse", {"start": "a", "max_expansions": 999}, state)
        expand = next(c for c in calls if "$frontier" in c["cypher"])
        assert expand["params"]["remaining"] == 7
