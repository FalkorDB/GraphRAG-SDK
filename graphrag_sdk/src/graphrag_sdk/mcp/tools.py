# GraphRAG SDK — MCP: Tool definitions (Phase 3.2)
# Transport-agnostic tool specs that wrap a GraphRAG facade. Kept free of
# any `mcp` package import so they can be unit-tested without the optional
# dependency installed; server.py adapts these into a live MCP server.

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MCPHandler = Callable[[dict[str, Any]], Awaitable[str]]

#: Bounds on what an MCP client may ask for.
CYPHER_DEFAULT_ROWS, CYPHER_MAX_ROWS = 25, 100
CYPHER_TIMEOUT_MS = 5000
WALK_MAX_BEAM_WIDTH, WALK_MAX_DEPTH = 20, 5
AGENT_MAX_STEPS = 12
_CITATION_CHARS = 500


@dataclass
class MCPTool:
    """A single MCP tool: name, description, JSON input schema, and handler."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: MCPHandler

    def to_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required or [],
    }


def _clamp(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _graph_store(rag: Any) -> Any:
    """The facade's graph store.

    The facade keeps storage off its public surface, so this reads the
    private attribute; a ``graph_store`` attribute (e.g. on a test double or
    a wrapper) takes precedence.
    """
    store = getattr(rag, "graph_store", None)
    return store if store is not None else rag._graph_store


@dataclass
class GraphRAGToolset:
    """Builds the standard MCP tool surface over a GraphRAG instance."""

    rag: Any
    tools: list[MCPTool] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tools = self._build()

    def by_name(self, name: str) -> MCPTool | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def specs(self) -> list[dict[str, Any]]:
        return [t.to_spec() for t in self.tools]

    def _build(self) -> list[MCPTool]:
        rag = self.rag

        async def ingest(args: dict[str, Any]) -> str:
            result = await rag.ingest(text=args["text"], document_id=args.get("document_id"))
            return _dump(
                {
                    "nodes_created": getattr(result, "nodes_created", None),
                    "relationships_created": getattr(result, "relationships_created", None),
                    "chunks_indexed": getattr(result, "chunks_indexed", None),
                }
            )

        async def retrieve(args: dict[str, Any]) -> str:
            result = await rag.retrieve(args["question"])
            return _dump({"items": [i.content for i in result.items]})

        async def answer(args: dict[str, Any]) -> str:
            result = await rag.completion(args["question"])
            return _dump({"answer": result.answer, "metadata": result.metadata})

        async def ask_agent(args: dict[str, Any]) -> str:
            options: dict[str, Any] = {}
            if "max_steps" in args:
                options["max_steps"] = _clamp(args["max_steps"], 6, 1, AGENT_MAX_STEPS)
            agent = rag.agentic_retrieval(**options)
            result = await rag.completion(
                args["question"], strategy=agent, history=args.get("history") or None
            )
            md = result.metadata or {}
            citations = [
                {
                    "n": c.get("n"),
                    "tool": c.get("tool"),
                    "source": c.get("source"),
                    "content": str(c.get("content", ""))[:_CITATION_CHARS],
                }
                for c in md.get("citations", []) or []
            ]
            agent_md = md.get("agent", {}) or {}
            return _dump(
                {
                    "answer": result.answer,
                    "grounded": md.get("grounded"),
                    "citations": citations,
                    "stop_reason": agent_md.get("stop_reason"),
                    "num_steps": agent_md.get("num_steps"),
                    "generated_cypher": agent_md.get("generated_cypher", []),
                }
            )

        async def cypher_query(args: dict[str, Any]) -> str:
            from graphrag_sdk.retrieval.agentic.cypher_guard import (
                enforce_row_cap,
                validate_read_query,
            )
            from graphrag_sdk.retrieval.agentic.graph_tools import format_value

            cypher = str(args["cypher"])
            problems = validate_read_query(cypher)
            if problems:
                return "Error: only read-only Cypher is permitted via MCP: " + "; ".join(problems)
            max_rows = _clamp(
                args.get("max_rows", CYPHER_DEFAULT_ROWS), CYPHER_DEFAULT_ROWS, 1, CYPHER_MAX_ROWS
            )
            bounded = enforce_row_cap(cypher, max_rows)
            result = await _graph_store(rag).query_raw(
                bounded, read_only=True, timeout=CYPHER_TIMEOUT_MS
            )
            rows = list(getattr(result, "result_set", []) or [])[:max_rows]
            return _dump(
                {
                    "rows": [[_jsonable(format_value(c)) for c in row] for row in rows],
                    "row_cap": max_rows,
                }
            )

        async def graph_walk(args: dict[str, Any]) -> str:
            from graphrag_sdk.retrieval.graph_walk import DynamicGraphWalk

            store = _graph_store(rag)
            try:
                weights = await store.pagerank()
            except Exception:
                weights = {}

            async def neighbor_fn(node_id: str) -> list[tuple[str, float, str]]:
                return await store.weighted_neighbors(node_id)

            walk = DynamicGraphWalk(
                neighbor_fn,
                node_weights=weights,
                beam_width=_clamp(args.get("beam_width", 5), 5, 1, WALK_MAX_BEAM_WIDTH),
                max_depth=_clamp(args.get("max_depth", 3), 3, 1, WALK_MAX_DEPTH),
            )
            goal = args.get("goal")
            if goal:
                path = await walk.bidirectional_search(args["start"], str(goal))
                return _dump({"path": path.model_dump() if path else None})
            paths = await walk.beam_search(args["start"])
            return _dump({"paths": [p.model_dump() for p in paths]})

        async def run_skill(args: dict[str, Any]) -> str:
            from graphrag_sdk.retrieval.agentic.tools import check_arguments
            from graphrag_sdk.skills import build_skill

            try:
                skill = build_skill(args["skill"], _graph_store(rag), rag.llm)
            except KeyError as exc:
                return f"Error: {exc.args[0]}"
            params = args.get("params", {}) or {}
            problems = check_arguments(skill.parameters, params)
            if problems:
                return f"Error: invalid arguments for skill '{skill.name}': {'; '.join(problems)}"
            try:
                result = await skill.run(None, **params)
            except ValueError as exc:
                return f"Error: {exc}"
            return _dump(result.model_dump())

        async def get_statistics(_args: dict[str, Any]) -> str:
            return _dump(await rag.get_statistics())

        async def get_ontology(_args: dict[str, Any]) -> str:
            ontology = await rag.get_ontology()
            return _dump(ontology.model_dump() if hasattr(ontology, "model_dump") else {})

        from graphrag_sdk.skills import SKILL_REGISTRY

        skill_lines = "; ".join(
            f"{name}: {cls.description}" for name, cls in sorted(SKILL_REGISTRY.items())
        )

        return [
            MCPTool(
                "ingest",
                "Ingest raw text into the knowledge graph.",
                _obj(
                    {
                        "text": {"type": "string"},
                        "document_id": {"type": "string"},
                    },
                    ["text"],
                ),
                ingest,
            ),
            MCPTool(
                "retrieve",
                "Retrieve graph context for a question (no generation).",
                _obj({"question": {"type": "string"}}, ["question"]),
                retrieve,
            ),
            MCPTool(
                "answer",
                "Full RAG pipeline: retrieve context and generate an answer.",
                _obj({"question": {"type": "string"}}, ["question"]),
                answer,
            ),
            MCPTool(
                "ask_agent",
                "Answer a question with the agentic retriever: it searches, queries and "
                "walks the graph with tools, then answers with [N] citations to the "
                "evidence it found. Returns the answer, whether it is grounded, and the "
                "cited evidence.",
                _obj(
                    {
                        "question": {"type": "string"},
                        "history": {
                            "type": "array",
                            "description": "Earlier turns as {role, content} objects.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string", "enum": ["user", "assistant"]},
                                    "content": {"type": "string"},
                                },
                                "required": ["role", "content"],
                            },
                        },
                        "max_steps": {
                            "type": "integer",
                            "description": f"Tool-calling turns, 1-{AGENT_MAX_STEPS}.",
                        },
                    },
                    ["question"],
                ),
                ask_agent,
            ),
            MCPTool(
                "cypher_query",
                "Run a read-only Cypher query against the graph (no writes, no procedure "
                f"calls; at most {CYPHER_MAX_ROWS} rows, {CYPHER_TIMEOUT_MS} ms).",
                _obj(
                    {
                        "cypher": {"type": "string"},
                        "max_rows": {
                            "type": "integer",
                            "description": f"Rows to return, 1-{CYPHER_MAX_ROWS}.",
                        },
                    },
                    ["cypher"],
                ),
                cypher_query,
            ),
            MCPTool(
                "graph_walk",
                "PageRank-weighted graph walk from a start entity id.",
                _obj(
                    {
                        "start": {"type": "string"},
                        "goal": {"type": "string"},
                        "beam_width": {
                            "type": "integer",
                            "description": f"1-{WALK_MAX_BEAM_WIDTH}",
                        },
                        "max_depth": {"type": "integer", "description": f"1-{WALK_MAX_DEPTH}"},
                    },
                    ["start"],
                ),
                graph_walk,
            ),
            MCPTool(
                "run_skill",
                f"Run a high-level reasoning skill. Available: {skill_lines}.",
                _obj(
                    {
                        "skill": {"type": "string", "enum": sorted(SKILL_REGISTRY)},
                        "params": {
                            "type": "object",
                            "description": "Skill arguments (see each skill's parameters).",
                        },
                    },
                    ["skill"],
                ),
                run_skill,
            ),
            MCPTool(
                "get_statistics",
                "Return node/edge counts and graph statistics.",
                _obj({}),
                get_statistics,
            ),
            MCPTool(
                "get_ontology",
                "Return the active ontology (entities, relations, attributes).",
                _obj({}),
                get_ontology,
            ),
        ]


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _dump(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)
