# GraphRAG SDK — Tools: tool specifications (single source of truth)
# Adapters (pydantic-ai, LangGraph, MCP) generate their tool definitions
# from tool_specs() — descriptions and schemas are never duplicated downstream.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SearchInput(BaseModel):
    """Arguments for graph_search."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(
        min_length=1, description="Natural-language query about entities/relationships."
    )
    top_k: int = Field(
        default=8, ge=1, le=25, description="How many entities and chunks to return."
    )
    expand_hops: int = Field(
        default=1, ge=1, le=3, description="Relationship expansion depth from found entities."
    )
    include_chunks: bool = Field(
        default=True, description="Include source text passages (heavier output)."
    )


class AnswerInput(BaseModel):
    """Arguments for graph_answer."""

    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, description="The question to answer from the graph.")
    top_k: int = Field(
        default=8, ge=1, le=25, description="Retrieval breadth used to build the answer context."
    )


class SchemaInput(BaseModel):
    """Arguments for graph_schema (none)."""

    model_config = ConfigDict(extra="forbid")


class CypherReadInput(BaseModel):
    """Arguments for cypher_read."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, description="Read-only Cypher. Write clauses are rejected.")
    params: dict[str, Any] | None = Field(
        default=None, description="Query parameters — always prefer over inlining values."
    )
    limit: int = Field(
        default=100, ge=1, le=1000, description="Row cap injected as LIMIT when the query has none."
    )
    timeout_ms: int = Field(
        default=5000, ge=100, le=60000, description="Server-side query timeout in milliseconds."
    )


class EntityInput(BaseModel):
    """Arguments for graph_entity."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, description="Entity name (exact or partial).")
    hops: int = Field(
        default=1, ge=1, le=3, description="Neighborhood depth to include around the entity."
    )


class RememberInput(BaseModel):
    """Arguments for graph_remember."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(
        min_length=1,
        max_length=200_000,
        description="Text to extract into the knowledge graph.",
    )
    document_id: str | None = Field(
        default=None, description="Stable document id; auto-generated when omitted."
    )


class FlushInput(BaseModel):
    """Arguments for graph_flush (none)."""

    model_config = ConfigDict(extra="forbid")


class ToolSpec(BaseModel):
    """Machine-readable tool definition consumed by agent-framework adapters."""

    model_config = ConfigDict(frozen=True)
    name: str
    description: str
    input_schema: dict[str, Any]
    output_hint: str


@dataclass(frozen=True)
class ToolDef:
    """Internal registry row binding a tool name to a toolkit method."""

    name: str
    method: str
    input_model: type[BaseModel]
    description: str
    output_hint: str
    is_write: bool = False
    manual_only: bool = False


_TOOL_REGISTRY: tuple[ToolDef, ...] = (
    ToolDef(
        "graph_search",
        "search",
        SearchInput,
        "Search the knowledge graph for entities, relationships, facts, and source "
        "passages relevant to a query. Use this when you will compose the reply "
        "yourself (prefer it over graph_answer for multi-step reasoning), and cite "
        "sources with the returned document_id/chunk_id values.",
        "SearchResult{query, entities[], relations[], facts[], chunks[], documents[]}; "
        "call .to_llm_text() for prompt-ready text.",
    ),
    ToolDef(
        "graph_answer",
        "answer",
        AnswerInput,
        "Ask the knowledge graph a natural-language question and get a fully generated "
        "answer with citations. Use for one-shot Q&A when you do not need to inspect "
        "the raw context yourself.",
        "AnswerResult{answer, citations[], entities_touched[], cypher_used}; "
        "call .to_llm_text() for prompt-ready text.",
    ),
    ToolDef(
        "graph_schema",
        "schema",
        SchemaInput,
        "List the graph's entity labels, relationship types, directional patterns, and "
        "live counts. Call once before other graph tools to learn what the graph "
        "contains and plan queries.",
        "SchemaResult{entities[], relations[], node_count, edge_count}; "
        "call .to_llm_text() for prompt-ready text.",
    ),
    ToolDef(
        "graph_entity",
        "entity",
        EntityInput,
        "Look up one entity by name and get its properties, relationships up to `hops` "
        "away, and source documents. Use when the user asks about a specific named "
        "person, organization, or thing.",
        "EntityResult{query, found, entity, neighbors[], nearby[], documents[]}; "
        "call .to_llm_text() for prompt-ready text.",
    ),
    ToolDef(
        "cypher_read",
        "cypher_read",
        CypherReadInput,
        "Run a read-only Cypher query against the knowledge graph. Use ONLY for "
        "aggregations or precise filters graph_search cannot express (counts, sorting, "
        "property predicates). Write clauses are rejected; a LIMIT is added if missing.",
        "CypherResult{columns[], rows[], row_count, truncated}; "
        "call .to_llm_text() for prompt-ready text.",
    ),
    ToolDef(
        "graph_remember",
        "remember",
        RememberInput,
        "Store new text (a fact, note, or document) into the knowledge graph so future "
        "searches can find it. Use when the user tells you something worth remembering.",
        "RememberResult{document_id, chunks_indexed, nodes_created, "
        "relationships_created, finalized}; call .to_llm_text() for prompt-ready text.",
        is_write=True,
    ),
    ToolDef(
        "graph_flush",
        "flush",
        FlushInput,
        "Run graph finalization (entity dedup, embeddings, indexes) after one or more "
        "graph_remember calls. Expensive — O(graph size); call once at the end of a "
        "write session, never after every write.",
        "Returns null.",
        is_write=True,
        manual_only=True,
    ),
)

TOOL_NAMES: tuple[str, ...] = tuple(td.name for td in _TOOL_REGISTRY)


def build_tool_specs(
    *,
    read_only: bool,
    finalize_policy: str,
    include: frozenset[str] | None,
) -> list[ToolSpec]:
    """Build the advertised ToolSpec list for a toolkit configuration."""
    specs: list[ToolSpec] = []
    for td in _TOOL_REGISTRY:
        if read_only and td.is_write:
            continue
        if td.manual_only and finalize_policy != "manual":
            continue
        if include is not None and td.name not in include:
            continue
        specs.append(
            ToolSpec(
                name=td.name,
                description=td.description,
                input_schema=td.input_model.model_json_schema(),
                output_hint=td.output_hint,
            )
        )
    return specs
