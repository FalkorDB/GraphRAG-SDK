# GraphRAG SDK — Agentic Retrieval: Tool registry (Phase 3.1)
# Tools are thin async wrappers around existing retrieval + storage
# primitives. Each tool publishes a JSON Schema for its arguments so the
# loop can offer it through native tool calling (ToolSpec) or describe it
# in a ReAct prompt, and every call returns a ToolResult whose status says
# whether it ran, was refused, or failed.

from __future__ import annotations

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
class ToolContext:
    """Per-run state handed to every tool call.

    Attributes:
        ctx: The execution context of the retrieval run (budgets, logging).
        state: Scratch space shared by the run's tool calls (e.g. a schema
            cache), discarded when the run ends.
    """

    ctx: Context
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


def _object_schema(
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
    """

    name: str
    description: str
    handler: ToolHandler
    parameters: dict[str, Any] = field(default_factory=lambda: _object_schema(additional=True))

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


# ── Default tool builders ────────────────────────────────────────

_SEARCH_OVERRIDES = (
    "chunk_top_k",
    "max_entities",
    "max_relationships",
    "rel_top_k",
    "max_cypher_out",
    "max_entities_out",
    "max_relationships_out",
    "max_facts_out",
    "max_passages_out",
)


def make_search_tool(strategy: Any, *, max_chars: int | None = None) -> Tool:
    """Vector/multi-path search over the graph, returning context snippets.

    The underlying strategy already bounds its own output (e.g. MultiPath caps
    passages via ``chunk_top_k`` plus entity/relationship limits), so by default
    no character truncation is applied and the agent sees the full result.
    Pass ``max_chars`` only to force an additional hard cap.
    """

    async def handler(tool_input: dict[str, Any], tctx: ToolContext) -> str:
        query = str(tool_input["query"]).strip()
        overrides = {k: tool_input[k] for k in _SEARCH_OVERRIDES if k in tool_input}
        result = await strategy.search(query, tctx.ctx, **overrides)
        snippets = [item.content for item in result.items]
        joined = "\n---\n".join(snippets) if snippets else "No results."
        return joined if max_chars is None else joined[:max_chars]

    properties: dict[str, Any] = {
        "query": {
            "type": "string",
            "description": "Self-contained search query with concrete names, not pronouns.",
        }
    }
    for key in _SEARCH_OVERRIDES:
        properties[key] = {"type": "integer", "description": "Optional retrieval limit."}
    return Tool(
        name="search",
        description=(
            "Semantic search of the knowledge graph: returns the most relevant entities, "
            "relationships, facts and source passages for a query."
        ),
        handler=handler,
        parameters=_object_schema(properties, ["query"]),
    )


def _format_cypher_value(value: Any) -> Any:
    """Render a Cypher result value readably for the LLM.

    FalkorDB returns Node/Edge objects whose ``str()`` is an opaque
    ``<... object at 0x...>``. Surface their properties instead so the
    agent can actually read entity names, types, and facts.
    """
    props = getattr(value, "properties", None)
    if isinstance(props, dict):
        labels = getattr(value, "labels", None) or getattr(value, "relation", None)
        readable = {k: v for k, v in props.items() if k != "embedding"}
        if labels:
            return {"labels": labels, **readable}
        return readable
    if isinstance(value, list | tuple):
        return [_format_cypher_value(v) for v in value]
    return value


def make_cypher_tool(graph_store: Any, *, max_rows: int = 25) -> Tool:
    """Run a read-only Cypher query against the graph."""

    async def handler(tool_input: dict[str, Any], tctx: ToolContext) -> str:
        cypher = str(tool_input["cypher"]).strip()
        if not is_read_only_cypher(cypher):
            raise ToolRefusal("only read-only Cypher (MATCH ... RETURN ...) is permitted")
        try:
            rows_cap = int(tool_input.get("max_rows", max_rows))
        except (TypeError, ValueError):
            rows_cap = max_rows
        if rows_cap < 1:
            rows_cap = max_rows
        result = await graph_store.query_raw(cypher)
        rows = list(getattr(result, "result_set", []) or [])[:rows_cap]
        if not rows:
            return "No rows."
        return "\n".join(str(_format_cypher_value(r)) for r in rows)

    return Tool(
        name="cypher",
        description="Run a read-only Cypher query (MATCH ... RETURN ...) against the graph.",
        handler=handler,
        parameters=_object_schema(
            {
                "cypher": {"type": "string", "description": "A read-only Cypher query."},
                "max_rows": {"type": "integer", "description": "Rows to return."},
            },
            ["cypher"],
        ),
    )


def make_traverse_tool(
    graph_store: Any,
    *,
    beam_width: int = 5,
    max_depth: int = 3,
) -> Tool:
    """Weighted graph walk from a start entity (optionally toward a goal)."""

    async def handler(tool_input: dict[str, Any], tctx: ToolContext) -> str:
        from graphrag_sdk.retrieval.graph_walk import DynamicGraphWalk

        start = str(tool_input["start"]).strip()
        goal = tool_input.get("goal")

        def _pos_int(key: str, default: int) -> int:
            try:
                val = int(tool_input.get(key, default))
            except (TypeError, ValueError):
                return default
            return val if val > 0 else default

        bw = _pos_int("beam_width", beam_width)
        depth = _pos_int("max_depth", max_depth)

        try:
            weights = await graph_store.pagerank()
        except Exception:
            weights = {}

        async def neighbor_fn(node_id: str) -> list[tuple[str, float, str]]:
            return await graph_store.weighted_neighbors(node_id)

        walk = DynamicGraphWalk(
            neighbor_fn,
            node_weights=weights,
            beam_width=bw,
            max_depth=depth,
        )
        if goal:
            path = await walk.bidirectional_search(start, str(goal), ctx=tctx.ctx)
            if path is None:
                return f"No path found between '{start}' and '{goal}'."
            return " -> ".join(path.nodes)
        paths = await walk.beam_search(start, ctx=tctx.ctx)
        if not paths:
            return f"No neighbors found for '{start}'."
        return "\n".join(f"{' -> '.join(p.nodes)} (score={p.score:.3f})" for p in paths)

    return Tool(
        name="traverse",
        description="Walk the graph from a start entity id, optionally toward a goal entity id.",
        handler=handler,
        parameters=_object_schema(
            {
                "start": {"type": "string", "description": "Start entity id."},
                "goal": {"type": "string", "description": "Optional goal entity id."},
                "beam_width": {"type": "integer"},
                "max_depth": {"type": "integer"},
            },
            ["start"],
        ),
    )


def make_skill_tool(skill: Any) -> Tool:
    """Wrap a :class:`Skill` instance as an agent tool."""

    async def handler(tool_input: dict[str, Any], tctx: ToolContext) -> str:
        result = await skill.run(tctx.ctx, **tool_input)
        if result.summary:
            return result.summary
        return str(result.data)

    parameters = getattr(skill, "parameters", None) or _object_schema(additional=True)
    return Tool(
        name=skill.name,
        description=skill.description,
        handler=handler,
        parameters=parameters,
    )


def build_default_registry(
    *,
    strategy: Any | None = None,
    graph_store: Any | None = None,
    llm: Any | None = None,
    include_skills: bool = True,
) -> ToolRegistry:
    """Assemble the standard agent toolset from available primitives."""
    registry = ToolRegistry()
    if strategy is not None:
        registry.register(make_search_tool(strategy))
    if graph_store is not None:
        registry.register(make_cypher_tool(graph_store))
        registry.register(make_traverse_tool(graph_store))
        if include_skills:
            from graphrag_sdk.skills import SKILL_REGISTRY

            for skill_cls in SKILL_REGISTRY.values():
                registry.register(make_skill_tool(skill_cls(graph_store, llm)))
    return registry
