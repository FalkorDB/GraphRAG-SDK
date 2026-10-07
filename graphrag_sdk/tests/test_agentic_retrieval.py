"""Tests for retrieval/agentic — the agent loop (native + ReAct) and tool registry."""

from __future__ import annotations

from typing import Any

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LatencyBudgetExceededError
from graphrag_sdk.core.models import (
    ChatMessage,
    LLMResponse,
    RetrieverResult,
    RetrieverResultItem,
    ToolCall,
)
from graphrag_sdk.retrieval.agentic import (
    AgenticRetrieval,
    Tool,
    ToolContext,
    ToolRefusal,
    ToolRegistry,
    ToolResult,
    build_default_registry,
    history_as_messages,
    is_read_only_cypher,
    parse_react_step,
)
from graphrag_sdk.retrieval.agentic.graph_tools import make_search_tool
from graphrag_sdk.retrieval.agentic.tools import check_arguments

# ── Fakes ────────────────────────────────────────────────────────


class ScriptedLLM:
    """Text-only LLM returning a fixed list of ReAct turns in order."""

    supports_tool_calling = False

    def __init__(self, turns: list[str]):
        self._turns = turns
        self._i = 0
        self.model_name = "scripted"
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.prompts.append(prompt)
        text = self._turns[min(self._i, len(self._turns) - 1)]
        self._i += 1
        return LLMResponse(content=text)


def call(name: str, call_id: str = "c1", **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


class ToolCallingLLM:
    """Native tool-calling LLM scripted with LLMResponse objects."""

    supports_tool_calling = True
    model_name = "tool-calling"

    def __init__(self, responses: list[LLMResponse]):
        self._responses = responses
        self._i = 0
        self.requests: list[dict[str, Any]] = []

    async def ainvoke_with_tools(self, messages, tools, **kwargs: Any) -> LLMResponse:
        self.requests.append({"messages": list(messages), "tools": list(tools), **kwargs})
        resp = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return resp


class FakeStrategy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def search(self, query: str, ctx: Context, **kwargs: Any) -> RetrieverResult:
        self.calls.append((query, kwargs))
        return RetrieverResult(items=[RetrieverResultItem(content="Alice works at Acme Corp.")])


# ── Parsing / helpers ────────────────────────────────────────────


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

    def test_allows_write_words_inside_strings(self):
        assert is_read_only_cypher("MATCH (n) WHERE n.name = 'call center' RETURN n")

    @pytest.mark.parametrize(
        "cypher",
        ["CREATE (n)", "MATCH (n) DELETE n", "MERGE (n)", "MATCH (n) SET n.x = 1"],
    )
    def test_rejects_writes(self, cypher: str):
        assert not is_read_only_cypher(cypher)


class TestHistoryAsMessages:
    def test_keeps_text_turns_and_drops_the_rest(self):
        history = [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "hi"},
            ChatMessage(role="assistant", content="hello"),
            ChatMessage(role="assistant", content="", tool_calls=[call("search")]),
            ChatMessage(role="tool", content="result", tool_call_id="c1"),
            {"role": "user", "content": "   "},
        ]
        out = history_as_messages(history)
        assert [(m.role, m.content) for m in out] == [("user", "hi"), ("assistant", "hello")]

    def test_keeps_only_last_turns(self):
        history = [{"role": "user", "content": f"q{i}"} for i in range(12)]
        out = history_as_messages(history, max_turns=2)
        assert [m.content for m in out] == ["q8", "q9", "q10", "q11"]

    def test_empty(self):
        assert history_as_messages(None) == []
        assert history_as_messages([{"role": "user", "content": "x"}], max_turns=0) == []


class TestCheckArguments:
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
        "required": ["query"],
        "additionalProperties": False,
    }

    def test_valid(self):
        assert check_arguments(self.schema, {"query": "x", "k": 3}) == []

    def test_missing_unknown_and_wrong_type(self):
        problems = check_arguments(self.schema, {"k": "three", "extra": 1})
        assert any("'query' is required" in p for p in problems)
        assert any("unknown argument" in p for p in problems)
        assert any("'k' must be integer" in p for p in problems)

    def test_bool_is_not_an_integer(self):
        assert check_arguments(self.schema, {"query": "x", "k": True})


# ── Registry ─────────────────────────────────────────────────────


class TestToolRegistry:
    async def test_unknown_tool_is_refused(self, ctx: Context):
        out = await ToolRegistry().run("nope", {}, ctx)
        assert out.status == "refused"
        assert out.content.startswith("Refused: nope did not run")

    async def test_bad_arguments_are_refused_before_the_handler(self, ctx: Context):
        reg = build_default_registry(strategy=FakeStrategy())
        out = await reg.run("search", {}, ctx)
        assert out.status == "refused"
        assert "'query' is required" in out.content

    async def test_handler_exception_is_an_error_result(self, ctx: Context):
        async def boom(inp: dict, tctx: ToolContext) -> str:
            raise RuntimeError("kaboom")

        reg = ToolRegistry()
        reg.register(Tool("boom", "desc", boom))
        out = await reg.run("boom", {}, ctx)
        assert out.status == "error"
        assert "kaboom" in out.content

    async def test_tool_refusal_becomes_refused(self, ctx: Context):
        async def picky(inp: dict, tctx: ToolContext) -> str:
            raise ToolRefusal("no entity is called 'X'")

        reg = ToolRegistry()
        reg.register(Tool("picky", "desc", picky))
        out = await reg.run("picky", {}, ToolContext(ctx=ctx))
        assert out.status == "refused"
        assert out.content == "Refused: picky did not run: no entity is called 'X'"

    async def test_budget_error_propagates(self, ctx: Context):
        async def slow(inp: dict, tctx: ToolContext) -> str:
            raise LatencyBudgetExceededError("over budget")

        reg = ToolRegistry()
        reg.register(Tool("slow", "desc", slow))
        with pytest.raises(LatencyBudgetExceededError):
            await reg.run("slow", {}, ctx)

    def test_duplicate_names_rejected(self):
        reg = build_default_registry(strategy=FakeStrategy())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(make_search_tool(FakeStrategy()))

    async def test_search_tool_returns_snippets_and_forwards_limits(self, ctx: Context):
        strategy = FakeStrategy()
        tool = make_search_tool(strategy)
        out = await tool.handler({"query": "alice", "chunk_top_k": 30}, ToolContext(ctx=ctx))
        assert out.content == "[1] Alice works at Acme Corp."
        assert strategy.calls == [("alice", {"chunk_top_k": 30})]

    def test_default_registry_only_search_without_store(self):
        reg = build_default_registry(strategy=FakeStrategy(), graph_store=None)
        assert reg.names() == ["search"]
        assert reg.specs()[0].name == "search"
        assert reg.specs()[0].parameters["required"] == ["query"]


# ── Native loop ──────────────────────────────────────────────────


class TestNativeLoop:
    async def test_calls_tool_then_answers(self, ctx: Context):
        llm = ToolCallingLLM(
            [
                LLMResponse(content="", tool_calls=[call("search", query="alice")]),
                LLMResponse(content="Alice works at Acme [1]."),
            ]
        )
        agent = AgenticRetrieval(llm, strategy=FakeStrategy())
        result = await agent.search("where does alice work?", ctx)

        assert result.metadata["agent_mode"] == "native"
        assert result.metadata["stop_reason"] == "final_answer"
        assert result.metadata["answer"] == "Alice works at Acme [1]."
        assert result.metadata["grounded"] is True
        assert result.metadata["citations"][0]["n"] == 1
        assert result.items[0].content == "[1] Alice works at Acme Corp."
        assert result.items[0].metadata["citation"] == 1
        trace = result.metadata["agent_trace"]
        assert trace["steps"][0]["action"] == "search"
        assert trace["steps"][0]["status"] == "ok"
        assert trace["steps"][0]["call_id"] == "c1"
        # Second request carries the assistant tool call and the tool result.
        second = llm.requests[1]["messages"]
        assert second[-2].role == "assistant" and second[-2].tool_calls[0].name == "search"
        assert second[-1].role == "tool" and second[-1].tool_call_id == "c1"
        assert "Acme" in second[-1].content
        assert [t.name for t in llm.requests[0]["tools"]] == ["search"]

    async def test_answers_without_tools(self, ctx: Context):
        llm = ToolCallingLLM([LLMResponse(content="Hello!")])
        result = await AgenticRetrieval(llm, strategy=FakeStrategy()).search("hi", ctx)
        assert result.metadata["answer"] == "Hello!"
        assert result.metadata["num_steps"] == 0

    async def test_bad_json_arguments_are_refused_and_run_continues(self, ctx: Context):
        bad = ToolCall(id="c1", name="search", raw_arguments="{oops", parse_error="bad JSON")
        llm = ToolCallingLLM(
            [
                LLMResponse(content="", tool_calls=[bad]),
                LLMResponse(content="", tool_calls=[call("search", "c2", query="alice")]),
                LLMResponse(content="Acme [1]."),
            ]
        )
        result = await AgenticRetrieval(llm, strategy=FakeStrategy()).search("q", ctx)
        steps = result.metadata["agent_trace"]["steps"]
        assert [s["status"] for s in steps] == ["refused", "ok"]
        assert result.metadata["answer"] == "Acme [1]."

    async def test_multiple_calls_in_one_turn(self, ctx: Context):
        llm = ToolCallingLLM(
            [
                LLMResponse(
                    content="",
                    tool_calls=[call("search", "a", query="x"), call("search", "b", query="y")],
                ),
                LLMResponse(content="done"),
            ]
        )
        strategy = FakeStrategy()
        result = await AgenticRetrieval(llm, strategy=strategy).search("q", ctx)
        assert [q for q, _ in strategy.calls] == ["x", "y"]
        assert result.metadata["num_steps"] == 2
        tool_msgs = [m for m in llm.requests[1]["messages"] if m.role == "tool"]
        assert [m.tool_call_id for m in tool_msgs] == ["a", "b"]

    async def test_max_steps_forces_a_final_answer(self, ctx: Context):
        llm = ToolCallingLLM(
            [
                LLMResponse(content="", tool_calls=[call("search", query="x")]),
                LLMResponse(content="", tool_calls=[call("search", query="y")]),
                LLMResponse(content="Best answer from what I found [2]."),
            ]
        )
        agent = AgenticRetrieval(llm, strategy=FakeStrategy(), max_steps=2)
        result = await agent.search("q", ctx)
        assert result.metadata["stop_reason"] == "max_steps"
        assert result.metadata["answer"] == "Best answer from what I found [2]."
        assert llm.requests[-1]["tool_choice"] == "none"
        assert "limit of tool calls" in llm.requests[-1]["messages"][-1].content

    async def test_history_is_sent_before_the_question(self, ctx: Context):
        llm = ToolCallingLLM([LLMResponse(content="ok")])
        agent = AgenticRetrieval(llm, strategy=FakeStrategy())
        await agent.search(
            "and her manager?",
            ctx,
            history=[
                {"role": "user", "content": "where does alice work?"},
                {"role": "assistant", "content": "Acme."},
            ],
        )
        msgs = llm.requests[0]["messages"]
        assert [m.role for m in msgs] == ["system", "user", "assistant", "user"]
        assert msgs[-1].content == "and her manager?"

    async def test_budget_error_mid_run_keeps_the_trace(self, ctx: Context):
        class BudgetLLM(ToolCallingLLM):
            async def ainvoke_with_tools(self, messages, tools, **kwargs):
                if self._i == 1:
                    raise LatencyBudgetExceededError("out of time")
                return await super().ainvoke_with_tools(messages, tools, **kwargs)

        llm = BudgetLLM([LLMResponse(content="", tool_calls=[call("search", query="x")])])
        result = await AgenticRetrieval(llm, strategy=FakeStrategy()).search("q", ctx)
        assert result.metadata["stop_reason"] == "budget_exceeded"
        assert result.metadata["num_steps"] == 1
        assert any("Acme" in item.content for item in result.items)

    async def test_custom_system_prompt(self, ctx: Context):
        llm = ToolCallingLLM([LLMResponse(content="ok")])
        agent = AgenticRetrieval(llm, strategy=FakeStrategy(), system_prompt="Be brief.")
        await agent.search("q", ctx)
        assert llm.requests[0]["messages"][0].content == "Be brief."


# ── ReAct fallback ───────────────────────────────────────────────


class TestReactLoop:
    async def test_loop_runs_tool_then_answers(self, ctx: Context):
        llm = ScriptedLLM(
            [
                'Thought: search\nAction: search\nAction Input: {"query": "alice"}',
                "Thought: I now have enough information.\nFinal Answer: Alice works at Acme [1].",
            ]
        )
        agent = AgenticRetrieval(llm, strategy=FakeStrategy(), max_steps=4)
        result = await agent.search("where does alice work?", ctx)
        assert result.metadata["agent_mode"] == "react"
        assert result.metadata["stop_reason"] == "final_answer"
        assert "Acme" in result.metadata["answer"]
        assert result.metadata["num_steps"] == 1
        assert any("Acme" in item.content for item in result.items)

    async def test_stops_on_exhausted_budget(self):
        llm = ScriptedLLM(["Action: search\nAction Input: {}"])
        agent = AgenticRetrieval(llm, strategy=FakeStrategy(), max_steps=4)
        result = await agent.search("q", Context(latency_budget_ms=0.0))
        assert result.metadata["stop_reason"] == "budget_exceeded"
        assert result.metadata["num_steps"] == 0

    async def test_one_malformed_reply_gets_a_reminder(self, ctx: Context):
        llm = ScriptedLLM(["Thought: hmm.", "Final Answer: 42"])
        result = await AgenticRetrieval(llm, strategy=FakeStrategy()).search("q", ctx)
        assert result.metadata["raw_answer"] == "42"
        # An uncited figure given without any tool call is not conversation.
        assert result.metadata["grounded"] is False
        assert "did not follow the format" in llm.prompts[1]

    async def test_stops_after_two_malformed_replies(self, ctx: Context):
        llm = ScriptedLLM(["Thought: hmm, nothing to do here."])
        agent = AgenticRetrieval(llm, strategy=FakeStrategy(), max_steps=3)
        result = await agent.search("q", ctx)
        assert result.metadata["stop_reason"] == "no_action"

    async def test_max_steps_forces_a_final_answer(self, ctx: Context):
        llm = ScriptedLLM(
            [
                'Action: search\nAction Input: {"query": "x"}',
                'Action: search\nAction Input: {"query": "y"}',
                "Final Answer: summarised [1][2]",
            ]
        )
        agent = AgenticRetrieval(llm, strategy=FakeStrategy(), max_steps=2)
        result = await agent.search("q", ctx)
        assert result.metadata["stop_reason"] == "max_steps"
        assert result.metadata["num_steps"] == 2
        assert result.metadata["answer"] == "summarised [1][2]"

    async def test_refused_call_is_traced(self, ctx: Context):
        llm = ScriptedLLM(["Action: search\nAction Input: {}", "Final Answer: none"])
        result = await AgenticRetrieval(llm, strategy=FakeStrategy()).search("q", ctx)
        step = result.metadata["agent_trace"]["steps"][0]
        assert step["status"] == "refused"
        assert "Refused:" in llm.prompts[1]


# ── Construction / mode ──────────────────────────────────────────


class TestConstruction:
    def test_invalid_max_steps(self):
        with pytest.raises(ValueError):
            AgenticRetrieval(ScriptedLLM([]), max_steps=0)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            AgenticRetrieval(ScriptedLLM([]), mode="magic")  # type: ignore[arg-type]

    def test_auto_mode_follows_the_llm(self):
        assert AgenticRetrieval(ScriptedLLM([])).resolved_mode() == "react"
        assert AgenticRetrieval(ToolCallingLLM([])).resolved_mode() == "native"
        assert AgenticRetrieval(ToolCallingLLM([]), mode="react").resolved_mode() == "react"

    async def test_needs_a_tool(self, ctx: Context):
        from graphrag_sdk.core.exceptions import RetrieverError

        agent = AgenticRetrieval(ToolCallingLLM([]), registry=ToolRegistry())
        with pytest.raises(RetrieverError, match="at least one tool"):
            await agent.search("q", ctx)

    def test_tool_result_defaults(self):
        assert ToolResult("x").status == "ok"
