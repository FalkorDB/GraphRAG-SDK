# GraphRAG SDK — Agentic Retrieval: Tool registry (Phase 3.1)
# Tools are thin async wrappers around existing retrieval + storage
# primitives, exposed to the ReAct loop. They also back the skill and
# graph-walk actions so the agent can search, traverse, run Cypher, and
# invoke high-level skills.

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from graphrag_sdk.core.context import Context

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any], Context], Awaitable[str]]

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


@dataclass
class Tool:
    """A single agent tool: name, description, and async handler."""

    name: str
    description: str
    handler: ToolHandler

    def schema(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}


class ToolRegistry:
    """Ordered registry of agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    async def run(self, name: str, tool_input: dict[str, Any], ctx: Context) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'. Valid tools: {self.names()}"
        try:
            return await tool.handler(tool_input, ctx)
        except Exception as exc:
            logger.warning("Tool %s failed: %s", name, exc)
            return f"Error running tool '{name}': {exc}"

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


def make_search_tool(strategy: Any, *, max_chars: int | None = None) -> Tool:
    """Vector/multi-path search over the graph, returning context snippets.

    The underlying strategy already bounds its own output (e.g. MultiPath caps
    passages via ``chunk_top_k`` plus entity/relationship limits), so by default
    no character truncation is applied and the agent sees the full result.
    Pass ``max_chars`` only to force an additional hard cap.
    """

    async def handler(tool_input: dict[str, Any], ctx: Context) -> str:
        query = str(tool_input.get("query", "")).strip()
        if not query:
            return "Error: 'query' is required."
        overrides = {
            k: tool_input[k]
            for k in (
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
            if k in tool_input
        }
        result = await strategy.search(query, ctx, **overrides)
        snippets = [item.content for item in result.items]
        joined = "\n---\n".join(snippets) if snippets else "No results."
        return joined if max_chars is None else joined[:max_chars]

    return Tool(
        name="search",
        description=(
            'Semantic search of the knowledge graph. Required: {"query": str}. '
            "Optional ints to widen/narrow retrieval: chunk_top_k, max_entities, "
            "max_relationships, rel_top_k, and output caps max_entities_out, "
            "max_relationships_out, max_facts_out, max_passages_out."
        ),
        handler=handler,
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

    async def handler(tool_input: dict[str, Any], ctx: Context) -> str:
        cypher = str(tool_input.get("cypher", "")).strip()
        if not cypher:
            return "Error: 'cypher' is required."
        if not is_read_only_cypher(cypher):
            return "Error: only read-only Cypher (MATCH/RETURN) is permitted."
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
        description=(
            "Run a read-only Cypher query (MATCH ... RETURN ...). "
            'Input: {"cypher": str, "max_rows"?: int}.'
        ),
        handler=handler,
    )


def make_traverse_tool(
    graph_store: Any,
    *,
    beam_width: int = 5,
    max_depth: int = 3,
) -> Tool:
    """Weighted graph walk from a start entity (optionally toward a goal)."""

    async def handler(tool_input: dict[str, Any], ctx: Context) -> str:
        from graphrag_sdk.retrieval.graph_walk import DynamicGraphWalk

        start = str(tool_input.get("start", "")).strip()
        if not start:
            return "Error: 'start' entity id is required."
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
            path = await walk.bidirectional_search(start, str(goal), ctx=ctx)
            if path is None:
                return f"No path found between '{start}' and '{goal}'."
            return " -> ".join(path.nodes)
        paths = await walk.beam_search(start, ctx=ctx)
        if not paths:
            return f"No neighbors found for '{start}'."
        return "\n".join(f"{' -> '.join(p.nodes)} (score={p.score:.3f})" for p in paths)

    return Tool(
        name="traverse",
        description=(
            "Walk the graph from a start entity, optionally toward a goal. "
            'Input: {"start": str, "goal"?: str, "beam_width"?: int, "max_depth"?: int}.'
        ),
        handler=handler,
    )


def make_skill_tool(skill: Any) -> Tool:
    """Wrap a :class:`Skill` instance as an agent tool."""

    async def handler(tool_input: dict[str, Any], ctx: Context) -> str:
        result = await skill.run(ctx, **tool_input)
        if result.summary:
            return result.summary
        return str(result.data)

    return Tool(
        name=skill.name,
        description=skill.description + " Input: skill-specific JSON arguments.",
        handler=handler,
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
