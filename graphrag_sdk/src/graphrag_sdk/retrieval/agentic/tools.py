# GraphRAG SDK — Agentic Retrieval: Tool registry (Phase 3.1)
# Core tool types. The default graph tools live in graph_tools.py.
# Each tool publishes a JSON Schema for its arguments so the
# loop can offer it through native tool calling (ToolSpec) or describe it
# in a ReAct prompt, and every call returns a ToolResult whose status says
# whether it ran, was refused, or failed.

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LatencyBudgetExceededError
from graphrag_sdk.core.models import ToolSpec

logger = logging.getLogger(__name__)

ToolStatus = Literal["ok", "refused", "error"]

#: What a refused call's text starts with. The trace and the model both read
#: it: the call did not run, and the rest of the line says why.
REFUSED_PREFIX = "Refused:"

# Cypher write/DDL keywords rejected by the read-only cypher tool. Matched on
# word boundaries so substrings of legitimate identifiers (e.g. "recall",
# "asset") are not falsely flagged. ``call`` blocks every stored-procedure path
# (algo.*, db.*, dbms.*, apoc.*) since procedures can mutate the graph.
_WRITE_KEYWORD_RE = re.compile(
    r"\b(create|merge|delete|set|remove|drop|detach|call|load\s+csv)\b",
    re.IGNORECASE,
)

# String literals ('..'/".." with backslash escapes) and backtick-quoted
# identifiers. Stripped before the keyword scan so data values like
# 'call center' or names such as `delete_log` can't trip the write gate.
_CYPHER_QUOTED_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|`[^`]*`")


class ToolRefusal(Exception):
    """Raised by a tool handler that declines to run the call.

    The message is shown to the model after ``"Refused: <tool> did not run:"``
    so it can correct the call (a missing argument, a name that matches
    nothing) instead of the run failing.
    """


@dataclass
class Evidence:
    """One numbered piece of evidence a tool returned during a run.

    ``n`` is the number the model cites as ``[n]``. Numbers run across all
    tools of one run, so a citation is unambiguous whichever tool produced it.
    """

    n: int
    tool: str
    kind: str
    content: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "tool": self.tool,
            "kind": self.kind,
            "content": self.content,
            "source": self.source,
            **({"metadata": self.metadata} if self.metadata else {}),
        }


class EvidenceLedger:
    """The run's numbered evidence, shared by every tool call."""

    def __init__(self) -> None:
        self._items: list[Evidence] = []

    def add(
        self,
        *,
        tool: str,
        kind: str,
        content: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Evidence:
        ev = Evidence(
            n=len(self._items) + 1,
            tool=tool,
            kind=kind,
            content=content,
            source=source,
            metadata=dict(metadata or {}),
        )
        self._items.append(ev)
        return ev

    def get(self, n: int) -> Evidence | None:
        return self._items[n - 1] if 1 <= n <= len(self._items) else None

    @property
    def items(self) -> list[Evidence]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class ToolContext:
    """Per-run state handed to every tool call.

    Attributes:
        ctx: The execution context of the retrieval run (budgets, logging).
        evidence: The run's numbered evidence; tools add what the model may cite.
        state: Scratch space shared by the run's tool calls (e.g. a schema
            cache, the Cypher queries tried), discarded when the run ends.
    """

    ctx: Context
    evidence: EvidenceLedger = field(default_factory=EvidenceLedger)
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Outcome of one tool call.

    ``content`` is what the model reads. ``status`` is ``"ok"`` when the tool
    ran, ``"refused"`` when it declined (bad arguments, nothing to act on) and
    ``"error"`` when it failed unexpectedly.
    """

    content: str
    status: ToolStatus = "ok"
    data: dict[str, Any] = field(default_factory=dict)


ToolHandler = Callable[[dict[str, Any], ToolContext], Awaitable["str | ToolResult"]]


def object_schema(
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
    *,
    additional: bool = False,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties or {},
        "additionalProperties": additional,
    }
    if required:
        schema["required"] = required
    return schema


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def check_arguments(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """Check ``args`` against the top level of a tool's JSON Schema.

    Covers what a model gets wrong in practice: missing required arguments,
    unknown arguments (when ``additionalProperties`` is false) and the JSON
    type of each declared argument. Nested schemas are not walked.
    """
    problems: list[str] = []
    properties: dict[str, Any] = schema.get("properties", {}) or {}
    for name in schema.get("required", []) or []:
        if args.get(name) in (None, ""):
            problems.append(f"'{name}' is required")
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(args) - set(properties))
        if unknown:
            problems.append(
                f"unknown argument(s) {', '.join(repr(u) for u in unknown)}; "
                f"accepted: {', '.join(sorted(properties)) or 'none'}"
            )
    for name, value in args.items():
        expected = (properties.get(name) or {}).get("type")
        if value is None or not isinstance(expected, str) or expected not in _JSON_TYPES:
            continue
        python_types = _JSON_TYPES[expected]
        is_bool = isinstance(value, bool)
        if not isinstance(value, python_types) or (is_bool and expected != "boolean"):
            problems.append(f"'{name}' must be {expected}, got {type(value).__name__}")
    return problems


@dataclass
class Tool:
    """A single agent tool: name, description, argument schema and handler.

    ``parameters`` is a JSON Schema object. Arguments are checked against it
    before the handler runs, so handlers can rely on required arguments
    being present and of the declared JSON type.

    ``citable`` tools produce evidence. One that does not number its own
    output gets a single evidence number for the whole result; set it to
    ``False`` for navigation tools whose output is not a source (e.g. an
    entity lookup).
    """

    name: str
    description: str
    handler: ToolHandler
    parameters: dict[str, Any] = field(default_factory=lambda: object_schema(additional=True))
    citable: bool = True

    def spec(self) -> ToolSpec:
        """The tool as a :class:`ToolSpec` for native tool calling."""
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def refused(tool: str, reason: str) -> ToolResult:
    """A refused :class:`ToolResult` with the standard ``Refused:`` wording."""
    return ToolResult(content=f"{REFUSED_PREFIX} {tool} did not run: {reason}", status="refused")


class ToolRegistry:
    """Ordered registry of agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"A tool named {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        """All tools as :class:`ToolSpec` entries, in registration order."""
        return [t.spec() for t in self._tools.values()]

    def describe(self) -> str:
        return "\n".join(
            f"- {t.name}: {t.description} Arguments (JSON Schema): {t.parameters}"
            for t in self._tools.values()
        )

    async def run(
        self,
        name: str,
        tool_input: dict[str, Any],
        tctx: ToolContext | Context,
    ) -> ToolResult:
        """Run one tool call; never raises except for an exhausted latency budget.

        Unknown tools and invalid arguments come back refused, and a handler
        that raises comes back as an error result, so one bad call never ends
        the agent's run.
        """
        if isinstance(tctx, Context):
            tctx = ToolContext(ctx=tctx)
        tool = self._tools.get(name)
        if tool is None:
            return refused(name or "(no tool)", f"unknown tool; valid tools: {self.names()}")
        if not isinstance(tool_input, dict):
            return refused(name, "arguments must be a JSON object")
        problems = check_arguments(tool.parameters, tool_input)
        if problems:
            return refused(name, "; ".join(problems))
        try:
            out = await tool.handler(tool_input, tctx)
        except LatencyBudgetExceededError:
            raise
        except ToolRefusal as exc:
            return refused(name, str(exc) or "the tool declined the call")
        except Exception as exc:
            logger.warning("Tool %s failed: %s", name, exc)
            logger.debug("Tool failure details", exc_info=True)
            message = str(exc).strip()[:300]
            detail = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
            return ToolResult(content=f"Error running tool '{name}': {detail}", status="error")
        if isinstance(out, ToolResult):
            return out
        return ToolResult(content=str(out))

    def __len__(self) -> int:
        return len(self._tools)


def is_read_only_cypher(cypher: str) -> bool:
    """Reject Cypher that mutates the graph (agent tools are read-only).

    Quoted strings and backtick identifiers are stripped first, so write
    keywords appearing only inside data values (``n.name = 'call center'``,
    ``CONTAINS "delete"``) don't cause false rejections. An unterminated
    quote leaves its remainder scanned conservatively (fail-safe).
    """
    scannable = _CYPHER_QUOTED_RE.sub(" ", cypher)
    return _WRITE_KEYWORD_RE.search(scannable) is None


# ── Skill tools ──────────────────────────────────────────────────


def make_skill_tool(skill: Any, *, max_data_chars: int = 4000) -> Tool:
    """Wrap a :class:`Skill` instance as an agent tool.

    The tool's arguments are the skill's ``parameters`` schema. The model
    reads the skill's summary (when it has an LLM) plus its structured
    findings as JSON; a ``ValueError`` from the skill (a missing argument,
    an entity name that matches nothing) comes back as a refusal.
    """

    async def handler(tool_input: dict[str, Any], tctx: ToolContext) -> str:
        try:
            result = await skill.run(tctx.ctx, **tool_input)
        except ValueError as exc:
            raise ToolRefusal(str(exc)) from exc
        data = json.dumps(result.data, default=str, ensure_ascii=False)
        if len(data) > max_data_chars:
            data = data[:max_data_chars] + "…[truncated]"
        parts = [result.summary.strip()] if result.summary else []
        parts.append(f"Findings ({skill.name}): {data}")
        return "\n".join(parts)

    parameters = getattr(skill, "parameters", None) or object_schema(additional=True)
    return Tool(
        name=skill.name,
        description=skill.description,
        handler=handler,
        parameters=parameters,
    )
