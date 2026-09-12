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


@dataclass
class GraphRAGToolset:
    """Builds the standard 8-tool MCP surface over a GraphRAG instance."""

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

        async def cypher_query(args: dict[str, Any]) -> str:
            from graphrag_sdk.retrieval.agentic.tools import is_read_only_cypher

            cypher = args["cypher"]
            if not is_read_only_cypher(cypher):
                return "Error: only read-only Cypher is permitted via MCP."
            result = await rag._graph_store.query_raw(cypher)
            rows = list(getattr(result, "result_set", []) or [])
            return _dump({"rows": [[_jsonable(c) for c in row] for row in rows]})

        async def graph_walk(args: dict[str, Any]) -> str:
            from graphrag_sdk.retrieval.graph_walk import DynamicGraphWalk

            store = rag._graph_store
            try:
                weights = await store.pagerank()
            except Exception:
                weights = {}

            async def neighbor_fn(node_id: str) -> list[tuple[str, float, str]]:
                return await store.weighted_neighbors(node_id)

            walk = DynamicGraphWalk(
                neighbor_fn,
                node_weights=weights,
                beam_width=int(args.get("beam_width", 5)),
                max_depth=int(args.get("max_depth", 3)),
            )
            goal = args.get("goal")
            if goal:
                path = await walk.bidirectional_search(args["start"], str(goal))
                return _dump({"path": path.model_dump() if path else None})
            paths = await walk.beam_search(args["start"])
            return _dump({"paths": [p.model_dump() for p in paths]})

        async def run_skill(args: dict[str, Any]) -> str:
            from graphrag_sdk.skills import build_skill

            skill = build_skill(args["skill"], rag._graph_store, rag.llm)
            params = args.get("params", {}) or {}
            result = await skill.run(None, **params)
            return _dump(result.model_dump())

        async def get_statistics(_args: dict[str, Any]) -> str:
            return _dump(await rag.get_statistics())

        async def get_ontology(_args: dict[str, Any]) -> str:
            ontology = await rag.get_ontology()
            return _dump(ontology.model_dump() if hasattr(ontology, "model_dump") else {})

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
                "cypher_query",
                "Run a read-only Cypher query against the graph.",
                _obj({"cypher": {"type": "string"}}, ["cypher"]),
                cypher_query,
            ),
            MCPTool(
                "graph_walk",
                "PageRank-weighted graph walk from a start entity.",
                _obj(
                    {
                        "start": {"type": "string"},
                        "goal": {"type": "string"},
                        "beam_width": {"type": "integer"},
                        "max_depth": {"type": "integer"},
                    },
                    ["start"],
                ),
                graph_walk,
            ),
            MCPTool(
                "run_skill",
                "Run a high-level skill (entity_comparison, impact_analysis, "
                "contradiction_detection, gap_analysis, timeline_reconstruction).",
                _obj(
                    {
                        "skill": {"type": "string"},
                        "params": {"type": "object"},
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
