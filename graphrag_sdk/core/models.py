# GraphRAG SDK 2.0 — Core: Data Models
# Pydantic v2 schemas used across the entire SDK.
# Origin: Shared types + Neo4j DataModel pattern + schema types.

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Base ─────────────────────────────────────────────────────────


class DataModel(BaseModel):
    """Base for all SDK data transfer objects.

    Provides Pydantic validation, serialisation, and a consistent
    interface for every piece of data flowing through strategies.
    Taken from Neo4j's DataModel pattern.
    """

    class Config:
        frozen = False
        extra = "allow"


# ── Graph Data Types ─────────────────────────────────────────────


class GraphNode(DataModel):
    """A node in the knowledge graph."""

    id: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)
    embedding_properties: Optional[dict[str, list[float]]] = None

    def __hash__(self) -> int:
        return hash(self.id)


class GraphRelationship(DataModel):
    """A relationship between two nodes in the knowledge graph."""

    start_node_id: str
    end_node_id: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    embedding_properties: Optional[dict[str, list[float]]] = None


# ── Text Processing Types ────────────────────────────────────────


class TextChunk(DataModel):
    """A chunk of text extracted from a document."""

    text: str
    index: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    uid: str = Field(default_factory=lambda: str(uuid4()))


class TextChunks(DataModel):
    """Collection of text chunks from a single document."""

    chunks: list[TextChunk] = Field(default_factory=list)


class DocumentInfo(DataModel):
    """Metadata about the source document."""

    path: Optional[str] = None
    uid: str = Field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentOutput(DataModel):
    """Output from a data loader — document text + metadata."""

    text: str
    document_info: DocumentInfo = Field(default_factory=DocumentInfo)


# ── Schema Types ─────────────────────────────────────────────────


class PropertyType(DataModel):
    """A property definition for a node or relationship type."""

    name: str
    type: str = "STRING"  # STRING, INTEGER, FLOAT, BOOLEAN, DATE, LIST
    description: Optional[str] = None
    required: bool = False


class EntityType(DataModel):
    """Definition of a node/entity type in the graph schema."""

    label: str
    description: Optional[str] = None
    properties: list[PropertyType] = Field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.label)


class RelationType(DataModel):
    """Definition of a relationship type in the graph schema."""

    label: str
    description: Optional[str] = None
    properties: list[PropertyType] = Field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.label)


class SchemaPattern(DataModel):
    """A valid source→relationship→target pattern in the schema."""

    source: str
    relationship: str
    target: str


class GraphSchema(DataModel):
    """Complete schema definition for the knowledge graph.

    Used by extraction strategies to constrain LLM output and
    by pruning to filter non-conforming data.
    """

    entities: list[EntityType] = Field(default_factory=list)
    relations: list[RelationType] = Field(default_factory=list)
    patterns: list[SchemaPattern] = Field(default_factory=list)


# ── Extraction / Resolution Output Types ─────────────────────────


class GraphData(DataModel):
    """Entities and relationships extracted from text."""

    nodes: list[GraphNode] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)


class ExtractedEntity(DataModel):
    """An entity extracted from a text chunk."""

    name: str
    type: str
    description: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)


class ExtractedRelation(DataModel):
    """A relationship extracted from a text chunk."""

    source: str
    target: str
    type: str
    keywords: str = ""
    description: str = ""
    weight: float = 1.0
    source_chunk_ids: list[str] = Field(default_factory=list)

    def to_fact_triple(self, chunk_id: str) -> "FactTriple":
        return FactTriple(
            subject=self.source,
            predicate=self.type,
            object=self.target,
            chunk_id=chunk_id,
        )


class FactTriple(DataModel):
    """A HippoRAG-style fact triple tied to a source chunk."""

    subject: str
    predicate: str
    object: str
    chunk_id: str = ""


class EntityMention(DataModel):
    """A mention of an entity in a specific chunk."""

    chunk_id: str
    entity_id: str


class ExtractionOutput(DataModel):
    """Output from an extraction strategy."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    facts: list[FactTriple] = Field(default_factory=list)
    mentions: list[EntityMention] = Field(default_factory=list)


class ResolutionResult(DataModel):
    """Result of entity resolution (deduplication)."""

    nodes: list[GraphNode] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)
    merged_count: int = 0


# ── Retrieval Types ──────────────────────────────────────────────


class RetrieverResultItem(DataModel):
    """A single item returned by a retrieval strategy."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None


class RetrieverResult(DataModel):
    """Aggregated result from a retrieval operation."""

    items: list[RetrieverResultItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RawSearchResult(DataModel):
    """Raw result from the database before formatting."""

    records: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── LLM Types ────────────────────────────────────────────────────


class LLMMessage(DataModel):
    """A message for the LLM conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


class LLMResponse(DataModel):
    """Response from an LLM provider."""

    content: str
    tool_calls: Optional[list[dict[str, Any]]] = None


# ── RAG Types ────────────────────────────────────────────────────


class RagResult(DataModel):
    """Result from a GraphRAG query operation."""

    answer: str
    retriever_result: Optional[RetrieverResult] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Ingestion Types ──────────────────────────────────────────────


class IngestionResult(DataModel):
    """Result from an ingestion pipeline run."""

    document_info: DocumentInfo = Field(default_factory=DocumentInfo)
    nodes_created: int = 0
    relationships_created: int = 0
    chunks_indexed: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Enums ────────────────────────────────────────────────────────


class SearchType(str, Enum):
    """Type of search operation."""

    VECTOR = "vector"
    FULLTEXT = "fulltext"
    HYBRID = "hybrid"
