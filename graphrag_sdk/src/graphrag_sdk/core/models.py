# GraphRAG SDK — Core: Data Models
# Pydantic v2 schemas used across the entire SDK.
# Origin: Shared types + Neo4j DataModel pattern + schema types.

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Generic, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

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
    embedding_properties: dict[str, list[float]] | None = None

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphNode):
            return NotImplemented
        return self.id == other.id


class GraphRelationship(DataModel):
    """A relationship between two nodes in the knowledge graph."""

    start_node_id: str
    end_node_id: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    embedding_properties: dict[str, list[float]] | None = None

    def to_fact_text(self) -> str:
        """Return a human-readable fact string for embedding on this edge.

        For ``RELATES`` edges, reads the original relationship type from
        ``properties["rel_type"]``; otherwise falls back to ``self.type``.
        Prefers the pre-built ``fact`` property if available.
        """
        if self.properties.get("fact"):
            return self.properties["fact"]
        src = self.properties.get("src_name", self.start_node_id)
        tgt = self.properties.get("tgt_name", self.end_node_id)
        rel_type = self.properties.get("rel_type", self.type)
        desc = self.properties.get("description", "")
        base = f"({src}, {rel_type}, {tgt})"
        return f"{base}: {desc}" if desc else base


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

    path: str | None = None
    uid: str = Field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentElement(DataModel):
    """A structural element parsed from a document (e.g., section, paragraph, table)."""

    type: str  # e.g., "header", "paragraph", "list", "table", "code"
    content: str | None = None
    level: int | None = None  # e.g., 1 for H1
    breadcrumbs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    children: list[DocumentElement] = Field(default_factory=list)


class DocumentOutput(DataModel):
    """Output from a data loader — document text + metadata + structural elements."""

    text: str
    document_info: DocumentInfo = Field(default_factory=DocumentInfo)
    elements: list[DocumentElement] | None = None


class DocumentRecord(DataModel):
    """Persisted state of a Document node, as read back from the graph.

    Returned by ``GraphStore.get_document_record()``. Pre-1.1.0 graphs
    lack ``content_hash`` (defaults to ``None``); callers should treat
    a ``None`` hash as "always run the full update path" since there
    is no stored value to compare against.
    """

    path: str | None = None
    content_hash: str | None = None


# ── Schema Types ─────────────────────────────────────────────────


class PropertyType(DataModel):
    """A property definition for a node or relationship type."""

    name: str
    type: str = "STRING"  # STRING, INTEGER, FLOAT, BOOLEAN, DATE, LIST
    description: str | None = None
    required: bool = False


class EntityType(DataModel):
    """Definition of a node/entity type in the graph schema."""

    label: str
    description: str | None = None
    properties: list[PropertyType] = Field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EntityType):
            return NotImplemented
        return self.label == other.label


class RelationType(DataModel):
    """Definition of a relationship type in the graph schema.

    ``patterns`` lists allowed directional ``(source_label, target_label)``
    pairs. Direction matters: ``("Person", "Company")`` means
    ``(Person)-[REL]->(Company)`` — *not* the inverse.

    Example::

        # "Person works at Company"
        RelationType(
            label="WORKS_AT",
            patterns=[("Person", "Company")],   # source -> target
        )

    An empty list means the relation is allowed between any entity types
    (open mode). Pattern mismatches at extraction time are silently pruned;
    a structured warning naming the offending ``(src, tgt)`` pairs is logged
    so a swapped direction is easy to spot.
    """

    label: str
    description: str | None = None
    patterns: list[tuple[str, str]] = Field(default_factory=list)

    # Identity is by label only — two RelationType instances with the same
    # label but different patterns compare/hash equal. Schemas are expected
    # to declare each relation label once.
    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RelationType):
            return NotImplemented
        return self.label == other.label


class GraphSchema(DataModel):
    """Complete schema definition for the knowledge graph.

    Used by extraction strategies to constrain LLM output and
    by pruning to filter non-conforming data.
    """

    entities: list[EntityType] = Field(default_factory=list)
    relations: list[RelationType] = Field(default_factory=list)

    @model_validator(mode="after")
    def _warn_on_undeclared_pattern_labels(self) -> GraphSchema:
        """Warn when a ``RelationType.patterns`` references undeclared entity labels.

        Catches typos like ``("Persn", "Company")`` at config time, before any
        extraction has run. We warn rather than raise: open-schema setups may
        legitimately reference labels not (yet) listed in ``entities``.
        """
        if not self.entities:
            return self
        declared = {e.label for e in self.entities}
        for rel in self.relations:
            for src, tgt in rel.patterns:
                missing = [lbl for lbl in (src, tgt) if lbl not in declared]
                if missing:
                    logger.warning(
                        "RelationType '%s' pattern (%s, %s) references "
                        "entity label(s) not declared in schema.entities: %s",
                        rel.label,
                        src,
                        tgt,
                        ", ".join(missing),
                    )
        return self


# ── Extraction / Resolution Output Types ─────────────────────────


class GraphData(DataModel):
    """Entities and relationships extracted from text."""

    nodes: list[GraphNode] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)
    mentions: list[EntityMention] = Field(default_factory=list)
    extracted_entities: list[ExtractedEntity] = Field(default_factory=list)
    extracted_relations: list[ExtractedRelation] = Field(default_factory=list)


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


class EntityMention(DataModel):
    """A mention of an entity in a specific chunk."""

    chunk_id: str
    entity_id: str


class ExtractionOutput(DataModel):
    """Output from an extraction strategy."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    mentions: list[EntityMention] = Field(default_factory=list)


class ResolutionResult(DataModel):
    """Result of entity resolution (deduplication).

    ``remap`` records every merge decision the resolver made: keys are
    pre-resolution entity ids that were merged away, values are the
    survivor's id. The ingestion pipeline rewrites ``graph_data.mentions``
    through this mapping so MENTIONED_IN edges are written against
    surviving entity ids instead of silently failing on a MATCH-not-found
    against a merged-away node.

    ``ExactMatchResolution`` groups by ``(label, resolve_property)`` and
    populates ``remap`` whenever a group has more than one node. With the
    default ``resolve_property='id'``, the duplicate and survivor already
    share the same id, so any remap entry is identity (``a → a``) —
    non-empty but a no-op when applied. With a non-id ``resolve_property``
    (e.g. ``'name'``), genuinely distinct ids can collapse into a survivor
    and ``remap`` captures each merge. Fuzzy resolvers (semantic,
    LLM-verified) similarly populate ``remap`` with every
    ``(loser_id → survivor_id)`` pair their merge produces.
    """

    nodes: list[GraphNode] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)
    merged_count: int = 0
    remap: dict[str, str] = Field(default_factory=dict)


# ── Retrieval Types ──────────────────────────────────────────────


class RetrieverResultItem(DataModel):
    """A single item returned by a retrieval strategy."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None


class RetrieverResult(DataModel):
    """Aggregated result from a retrieval operation."""

    items: list[RetrieverResultItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RawSearchResult(DataModel):
    """Raw result from the database before formatting."""

    records: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── LLM Types ────────────────────────────────────────────────────


class ChatMessage(DataModel):
    """A validated message for multi-turn LLM conversations.

    Used by ``completion(history=...)`` and ``LLMInterface.ainvoke_messages()``.

    Example::

        ChatMessage(role="system", content="You are a helpful assistant.")
        ChatMessage(role="user", content="What is GraphRAG?")
        ChatMessage(role="assistant", content="GraphRAG is...")
    """

    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to the ``{"role": ..., "content": ...}`` dict format used by LLM APIs."""
        return {"role": self.role, "content": self.content}


# Backward-compatible alias
LLMMessage = ChatMessage


class LLMResponse(DataModel):
    """Response from an LLM provider."""

    content: str
    tool_calls: list[dict[str, Any]] | None = None


# ── RAG Types ────────────────────────────────────────────────────


class RagResult(DataModel):
    """Result from a GraphRAG query operation."""

    answer: str
    retriever_result: RetrieverResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Ingestion Types ──────────────────────────────────────────────


class IngestionResult(DataModel):
    """Result from an ingestion pipeline run."""

    document_info: DocumentInfo = Field(default_factory=DocumentInfo)
    nodes_created: int = 0
    relationships_created: int = 0
    chunks_indexed: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class FinalizeResult(DataModel):
    """Result from ``GraphRAG.finalize()``.

    Aggregates counts from each post-ingestion step. Replaces the previous
    untyped ``dict[str, Any]`` so callers get IDE autocomplete and mypy
    enforcement on field access.
    """

    null_stubs_removed: int = 0
    entities_deduplicated: int = 0
    entities_embedded: int = 0
    relationships_embedded: int = 0
    indexes: dict[str, bool] = Field(default_factory=dict)


class UpdateResult(IngestionResult):
    """Result from ``GraphRAG.update()``.

    Extends ``IngestionResult`` so callers can treat ingest and update
    results polymorphically.

    - ``no_op=True`` indicates the new content matched the stored
      ``Document.content_hash`` and no work was done.
    - ``replaced_existing=True`` indicates an existing document was
      found and replaced (in-place update or no-op). ``False`` means
      ``if_missing="ingest"`` fell through to a fresh ingest because
      the id was unknown.

    The document id is always available at ``document_info.uid``.
    """

    chunks_deleted: int = 0
    entities_deleted: int = 0
    no_op: bool = False
    replaced_existing: bool = False


class DeleteDocumentResult(DataModel):
    """Result from ``GraphRAG.delete_document()``.

    Reports which document was removed (``document_uid``) and the
    counts of chunks and orphan entities cleaned up. Field name
    matches ``DocumentInfo.uid`` for cross-result consistency.
    """

    document_uid: str
    chunks_deleted: int = 0
    entities_deleted: int = 0


T_BatchResult = TypeVar("T_BatchResult", bound=DataModel)


class BatchEntry(DataModel, Generic[T_BatchResult]):
    """One entry in a batch result — either a success or a failure.

    Wraps the per-file outcome of ``apply_changes()``. On success,
    ``result`` carries the typed payload. On failure, ``error`` carries
    the formatted message and ``error_type`` carries the exception
    class name (e.g. ``"DocumentNotFoundError"``) for programmatic
    branching without re-raising.

    Storing the error as a string instead of an ``Exception`` keeps the
    model JSON-serialisable and avoids ``arbitrary_types_allowed``,
    which would otherwise disable Pydantic validation for the entry.
    """

    result: T_BatchResult | None = None
    error: str | None = None
    error_type: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None

    @classmethod
    def ok(cls, result: T_BatchResult) -> BatchEntry[T_BatchResult]:
        return cls(result=result)

    @classmethod
    def fail(cls, exc: BaseException) -> BatchEntry[T_BatchResult]:
        return cls(error=str(exc), error_type=type(exc).__name__)


class ApplyChangesResult(DataModel):
    """Aggregate result from ``GraphRAG.apply_changes()``.

    Each list aligns with the corresponding input list by index. Per-file
    failures are wrapped as ``BatchEntry`` with ``error`` set; the batch
    never raises. Callers branch on ``entry.is_success`` (or check
    ``entry.error_type`` for specific failures).
    """

    added: list[BatchEntry[IngestionResult]] = Field(default_factory=list)
    modified: list[BatchEntry[UpdateResult]] = Field(default_factory=list)
    deleted: list[BatchEntry[DeleteDocumentResult]] = Field(default_factory=list)


# ── Enums ────────────────────────────────────────────────────────


class SearchType(str, Enum):
    """Type of search operation."""

    VECTOR = "vector"
    FULLTEXT = "fulltext"
    HYBRID = "hybrid"
