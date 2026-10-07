# GraphRAG SDK — Core: native tool-calling helpers
# Shared by the OpenAI-compatible providers (LiteLLM, OpenRouter).

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from graphrag_sdk.core.models import ToolCall, ToolSpec

#: ``"auto"`` lets the model decide, ``"none"`` forbids tool calls,
#: ``"required"`` forces at least one call, and ``{"name": "<tool>"}``
#: forces that one tool.
ToolChoice = Literal["auto", "none", "required"] | Mapping[str, str]


def validate_tools(tools: Sequence[ToolSpec]) -> list[ToolSpec]:
    """Return ``tools`` as a list, refusing an empty list or duplicate names."""
    tool_list = list(tools)
    if not tool_list:
        raise ValueError("tools must contain at least one ToolSpec")
    seen: set[str] = set()
    for tool in tool_list:
        if not isinstance(tool, ToolSpec):
            raise TypeError(f"tools must be ToolSpec instances, got {type(tool).__name__}")
        if tool.name in seen:
            raise ValueError(f"Duplicate tool name {tool.name!r}")
        seen.add(tool.name)
    return tool_list


_CHOICE_HELP = "tool_choice must be 'auto', 'none', 'required' or {'name': <tool>}"


def tool_choice_to_openai(choice: ToolChoice, tools: Sequence[ToolSpec]) -> Any:
    """Translate a :data:`ToolChoice` into the OpenAI ``tool_choice`` value."""
    if isinstance(choice, str):
        if choice not in ("auto", "none", "required"):
            raise ValueError(f"{_CHOICE_HELP}, got {choice!r}")
        return choice
    if isinstance(choice, Mapping) and set(choice) == {"name"}:
        name = choice["name"]
        if name not in {t.name for t in tools}:
            raise ValueError(f"tool_choice names unknown tool {name!r}")
        return {"type": "function", "function": {"name": name}}
    raise ValueError(
        f"tool_choice must be 'auto', 'none', 'required' or {{'name': ...}}, got {choice!r}"
    )


def _field(obj: Any, key: str) -> Any:
    """Read ``key`` from an SDK object or a plain dict (providers return either)."""
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def parse_tool_calls(message: Any) -> list[ToolCall] | None:
    """Parse the ``tool_calls`` of an OpenAI-compatible response message.

    Returns ``None`` when the message has no tool calls. Arguments that are
    not valid JSON (or not a JSON object) never raise: the call keeps its
    raw text and a ``parse_error`` so the caller can tell the model.
    """
    raw_calls = _field(message, "tool_calls")
    if not isinstance(raw_calls, (list, tuple)) or not raw_calls:
        return None
    calls: list[ToolCall] = []
    for index, raw in enumerate(raw_calls):
        function = _field(raw, "function")
        name = _field(function, "name") if function is not None else None
        if not isinstance(name, str) or not name:
            continue
        call_id = _field(raw, "id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{index}"
        raw_args = _field(function, "arguments")
        arguments: dict[str, Any] = {}
        parse_error: str | None = None
        if isinstance(raw_args, Mapping):
            arguments = dict(raw_args)
            raw_text = json.dumps(arguments)
        else:
            raw_text = raw_args if isinstance(raw_args, str) else ""
            if raw_text.strip():
                try:
                    decoded = json.loads(raw_text)
                except json.JSONDecodeError as exc:
                    parse_error = f"arguments are not valid JSON: {exc.msg}"
                else:
                    if isinstance(decoded, dict):
                        arguments = decoded
                    else:
                        parse_error = "arguments must be a JSON object"
        calls.append(
            ToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
                raw_arguments=raw_text,
                parse_error=parse_error,
            )
        )
    return calls or None


def finish_reason_of(choice: Any) -> str | None:
    """Return the choice's ``finish_reason`` when the provider reports a string."""
    reason = _field(choice, "finish_reason")
    return reason if isinstance(reason, str) else None
