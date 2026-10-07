"""Tests for native tool calling: models, helpers and the built-in providers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag_sdk import GraphRAG
from graphrag_sdk.core.exceptions import LLMError, ToolCallingNotSupportedError
from graphrag_sdk.core.models import ChatMessage, LLMResponse, ToolCall, ToolSpec
from graphrag_sdk.core.providers import LiteLLM, LLMInterface, OpenRouterLLM
from graphrag_sdk.core.providers._tools import (
    parse_tool_calls,
    tool_choice_to_openai,
    validate_tools,
)

SEARCH = ToolSpec(
    name="search",
    description="Search the knowledge graph.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)
CYPHER = ToolSpec(name="cypher", description="Run a read-only query.")


def _tool_call_obj(call_id: str | None, name: str, arguments: Any) -> SimpleNamespace:
    """An OpenAI-SDK-shaped tool call (attribute access, like the real objects)."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(content: str | None, tool_calls: list | None, finish: str | None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish)
    return SimpleNamespace(choices=[choice])


class _TextOnlyLLM(LLMInterface):
    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return LLMResponse(content="text")


# ── Models ───────────────────────────────────────────────────────


class TestToolSpec:
    def test_to_openai(self):
        assert SEARCH.to_openai() == {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the knowledge graph.",
                "parameters": SEARCH.parameters,
            },
        }

    def test_default_parameters_is_empty_object(self):
        assert CYPHER.parameters == {"type": "object", "properties": {}}

    @pytest.mark.parametrize("name", ["", "has space", "dot.name", "x" * 65])
    def test_rejects_bad_names(self, name):
        with pytest.raises(ValueError, match="Invalid tool name"):
            ToolSpec(name=name)

    def test_rejects_non_object_parameters(self):
        with pytest.raises(ValueError, match="JSON Schema object"):
            ToolSpec(name="t", parameters={"type": "string"})


class TestChatMessageToolFields:
    def test_plain_message_dict_unchanged(self):
        assert ChatMessage(role="user", content="hi").to_dict() == {
            "role": "user",
            "content": "hi",
        }

    def test_assistant_tool_calls_dict(self):
        call = ToolCall(
            id="c1", name="search", arguments={"query": "x"}, raw_arguments='{"query": "x"}'
        )
        msg = ChatMessage(role="assistant", content="", tool_calls=[call])
        assert msg.to_dict() == {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query": "x"}'},
                }
            ],
        }

    def test_tool_call_without_raw_arguments_sends_empty_object(self):
        assert ToolCall(id="c1", name="cypher").to_openai()["function"]["arguments"] == "{}"

    def test_tool_result_dict(self):
        msg = ChatMessage(role="tool", content="3 rows", tool_call_id="c1")
        assert msg.to_dict() == {"role": "tool", "content": "3 rows", "tool_call_id": "c1"}

    def test_tool_message_requires_call_id(self):
        with pytest.raises(ValueError, match="tool_call_id"):
            ChatMessage(role="tool", content="x")

    def test_only_assistant_carries_tool_calls(self):
        with pytest.raises(ValueError, match="Only assistant"):
            ChatMessage(role="user", content="x", tool_calls=[ToolCall(id="c", name="search")])

    def test_only_tool_carries_call_id(self):
        with pytest.raises(ValueError, match="Only tool"):
            ChatMessage(role="user", content="x", tool_call_id="c1")

    def test_llm_response_coerces_tool_call_dicts(self):
        resp = LLMResponse(content="", tool_calls=[{"id": "c1", "name": "search"}])
        assert isinstance(resp.tool_calls[0], ToolCall)
        assert resp.finish_reason is None


# ── Helpers ──────────────────────────────────────────────────────


class TestParseToolCalls:
    def test_parses_valid_calls(self):
        msg = SimpleNamespace(
            tool_calls=[
                _tool_call_obj("c1", "search", '{"query": "acme"}'),
                _tool_call_obj("c2", "cypher", ""),
            ]
        )
        calls = parse_tool_calls(msg)
        assert [c.id for c in calls] == ["c1", "c2"]
        assert calls[0].arguments == {"query": "acme"}
        assert calls[0].raw_arguments == '{"query": "acme"}'
        assert calls[1].arguments == {} and calls[1].parse_error is None

    def test_invalid_json_keeps_raw_and_reports(self):
        calls = parse_tool_calls(
            SimpleNamespace(tool_calls=[_tool_call_obj("c1", "search", "{bad")])
        )
        assert calls[0].arguments == {}
        assert calls[0].raw_arguments == "{bad"
        assert "not valid JSON" in calls[0].parse_error

    def test_non_object_json_is_reported(self):
        calls = parse_tool_calls(
            SimpleNamespace(tool_calls=[_tool_call_obj("c1", "search", "[1]")])
        )
        assert calls[0].parse_error == "arguments must be a JSON object"

    def test_dict_shaped_calls_and_missing_id(self):
        msg = {"tool_calls": [{"function": {"name": "search", "arguments": {"query": "q"}}}]}
        calls = parse_tool_calls(msg)
        assert calls[0].id == "call_0"
        assert calls[0].arguments == {"query": "q"}
        assert calls[0].raw_arguments == '{"query": "q"}'

    def test_nameless_calls_are_skipped(self):
        msg = SimpleNamespace(tool_calls=[_tool_call_obj("c1", "", "{}")])
        assert parse_tool_calls(msg) is None

    @pytest.mark.parametrize("value", [None, [], MagicMock()])
    def test_no_tool_calls_returns_none(self, value):
        assert parse_tool_calls(SimpleNamespace(tool_calls=value)) is None

    def test_plain_magicmock_message_returns_none(self):
        # Existing provider tests build responses from MagicMock; those must
        # keep reporting "no tool calls".
        assert parse_tool_calls(MagicMock()) is None


class TestToolChoiceAndValidation:
    @pytest.mark.parametrize("choice", ["auto", "none", "required"])
    def test_string_choices_pass_through(self, choice):
        assert tool_choice_to_openai(choice, [SEARCH]) == choice

    def test_named_choice(self):
        assert tool_choice_to_openai({"name": "search"}, [SEARCH]) == {
            "type": "function",
            "function": {"name": "search"},
        }

    @pytest.mark.parametrize("choice", ["always", {"name": "missing"}, {"tool": "search"}])
    def test_bad_choices(self, choice):
        with pytest.raises(ValueError):
            tool_choice_to_openai(choice, [SEARCH])

    def test_validate_tools_rejects_empty_duplicates_and_wrong_type(self):
        with pytest.raises(ValueError, match="at least one"):
            validate_tools([])
        with pytest.raises(ValueError, match="Duplicate"):
            validate_tools([SEARCH, SEARCH])
        with pytest.raises(TypeError):
            validate_tools([{"name": "search"}])


# ── Base interface ───────────────────────────────────────────────


class TestBaseInterface:
    async def test_base_does_not_support_tools(self):
        llm = _TextOnlyLLM(model_name="m")
        assert llm.supports_tool_calling is False
        with pytest.raises(ToolCallingNotSupportedError, match="_TextOnlyLLM"):
            await llm.ainvoke_with_tools([ChatMessage(role="user", content="q")], [SEARCH])

    def test_error_is_an_llm_error_and_not_implemented(self):
        assert issubclass(ToolCallingNotSupportedError, LLMError)
        assert issubclass(ToolCallingNotSupportedError, NotImplementedError)


# ── Providers ────────────────────────────────────────────────────

CONVERSATION = [
    ChatMessage(role="system", content="Use tools."),
    ChatMessage(role="user", content="Who owns Acme?"),
    ChatMessage(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="c0", name="search", raw_arguments='{"query": "Acme"}')],
    ),
    ChatMessage(role="tool", content="[1] Acme is owned by Globex.", tool_call_id="c0"),
]


class TestLiteLLMTools:
    async def test_sends_tools_and_parses_calls(self):
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(
            return_value=_response(
                None,
                [_tool_call_obj("c1", "cypher", '{"cypher": "MATCH (n) RETURN n"}')],
                "tool_calls",
            )
        )
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            llm = LiteLLM(model="openai/gpt-4o", api_key="k")
            assert llm.supports_tool_calling is True
            resp = await llm.ainvoke_with_tools(
                CONVERSATION, [SEARCH, CYPHER], tool_choice={"name": "cypher"}
            )

        kw = mock_litellm.acompletion.call_args.kwargs
        assert kw["tools"] == [SEARCH.to_openai(), CYPHER.to_openai()]
        assert kw["tool_choice"] == {"type": "function", "function": {"name": "cypher"}}
        assert kw["messages"][2]["tool_calls"][0]["id"] == "c0"
        assert kw["messages"][3] == {
            "role": "tool",
            "content": "[1] Acme is owned by Globex.",
            "tool_call_id": "c0",
        }
        assert kw["api_key"] == "k"
        assert resp.content == ""
        assert resp.finish_reason == "tool_calls"
        assert resp.tool_calls[0].name == "cypher"
        assert resp.tool_calls[0].arguments == {"cypher": "MATCH (n) RETURN n"}

    async def test_final_answer_has_no_tool_calls(self):
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=_response("Globex [1]", None, "stop"))
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            resp = await LiteLLM(model="gpt-4o").ainvoke_with_tools(CONVERSATION, [SEARCH])
        assert resp.content == "Globex [1]"
        assert resp.tool_calls is None
        assert resp.finish_reason == "stop"
        assert mock_litellm.acompletion.call_args.kwargs["tool_choice"] == "auto"

    async def test_reasoning_model_translation_still_applies(self):
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=_response("ok", None, "stop"))
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            llm = LiteLLM(model="openai/gpt-5", max_tokens=256)
            await llm.ainvoke_with_tools(CONVERSATION, [SEARCH], temperature=0.3)
        kw = mock_litellm.acompletion.call_args.kwargs
        assert "temperature" not in kw
        assert kw["max_completion_tokens"] == 256

    async def test_retries_transient_errors(self):
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(
            side_effect=[RuntimeError("503"), _response("ok", None, "stop")]
        )
        with (
            patch.dict("sys.modules", {"litellm": mock_litellm}),
            patch("asyncio.sleep", new=AsyncMock()),
        ):
            resp = await LiteLLM(model="gpt-4o").ainvoke_with_tools(CONVERSATION, [SEARCH])
        assert resp.content == "ok"
        assert mock_litellm.acompletion.await_count == 2

    async def test_tools_in_kwargs_is_rejected(self):
        with pytest.raises(TypeError, match="tools"):
            await LiteLLM(model="gpt-4o").ainvoke_with_tools(
                CONVERSATION, [SEARCH], tools=[SEARCH.to_openai()]
            )

    async def test_plain_messages_call_unchanged(self):
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=_response("hello", None, "stop"))
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            resp = await LiteLLM(model="gpt-4o").ainvoke_messages(
                [ChatMessage(role="user", content="hi")]
            )
        kw = mock_litellm.acompletion.call_args.kwargs
        assert "tools" not in kw and "tool_choice" not in kw
        assert kw["messages"] == [{"role": "user", "content": "hi"}]
        assert resp.content == "hello" and resp.tool_calls is None


class TestOpenRouterTools:
    async def test_sends_tools_and_parses_calls(self):
        mock_openai = MagicMock()
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_response(
                "", [_tool_call_obj("c9", "search", '{"query": "Globex"}')], "tool_calls"
            )
        )
        mock_openai.AsyncOpenAI.return_value = client
        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenRouterLLM(model="openai/gpt-4o", api_key="or-key")
            assert llm.supports_tool_calling is True
            resp = await llm.ainvoke_with_tools(CONVERSATION, [SEARCH], tool_choice="required")

        kw = client.chat.completions.create.call_args.kwargs
        assert kw["tools"] == [SEARCH.to_openai()]
        assert kw["tool_choice"] == "required"
        assert kw["messages"][3]["tool_call_id"] == "c0"
        assert resp.tool_calls[0].id == "c9"
        assert resp.tool_calls[0].arguments == {"query": "Globex"}
        assert resp.finish_reason == "tool_calls"


# ── completion() history guard ───────────────────────────────────


class TestCompletionHistoryGuard:
    def test_rejects_tool_messages(self):
        with pytest.raises(ValueError, match="tool-calling messages"):
            GraphRAG._validate_history([CONVERSATION[3]])
        with pytest.raises(ValueError, match="tool-calling messages"):
            GraphRAG._validate_history([CONVERSATION[2]])

    def test_plain_history_still_accepted(self):
        out = GraphRAG._validate_history(
            [ChatMessage(role="user", content="a"), {"role": "assistant", "content": "b"}]
        )
        assert [m.role for m in out] == ["user", "assistant"]

    def test_dict_tool_role_still_rejected(self):
        with pytest.raises(ValueError, match="invalid role 'tool'"):
            GraphRAG._validate_history([{"role": "tool", "content": "x"}])
