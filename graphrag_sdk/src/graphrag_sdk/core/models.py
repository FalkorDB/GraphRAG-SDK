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


_PROPERTY_TYPES: frozenset[str] = frozenset(
    {"STRING", "INTEGER", "FLOAT", "BOOLEAN", "DATE", "LIST"}
)

# Keys the SDK writes on every node/edge during ingestion. Declaring an
# Attribute with any of these names will shadow the system-written value
# on extracted entities. The Ontology validator warns against this; the
# discovery validator rejects it outright unless it's in
# ``_SDK_MANAGED_ATTRIBUTE_NAMES`` below.
_RESERVED_ATTRIBUTE_NAMES: frozenset[str] = frozenset(
    {
        "name",
        "description",
        "source_chunk_ids",
        "spans",
        "rel_type",
        "fact",
        "src_name",
        "tgt_name",
        "id",
        "label",
    }
)

# Attributes the schema may declare for documentation, even though the SDK
# fills them automatically during extraction (top-level entity fields from
# GLiNER + step-2 LLM, not from the per-entity ``attributes`` dict).
#
# The discovery layer requires these on every entity type so the schema
# accurately reflects the data graph. The extraction layer filters them
# out of the prompt's per-entity attribute block so the LLM is not asked
# to extract them twice (which would cause two writes to the same field
# and silent overwrite races).
#
# Concretely: ``Person.name`` appears in the saved ``ontology.json``, the
# user sees it, but the extraction LLM is never asked "what is this
# person's name?" — that value is the entity's identifier from step 1.
_SDK_MANAGED_ATTRIBUTE_NAMES: frozenset[str] = frozenset({"name"})


class Attribute(DataModel):
    """A property definition for a node or relationship type."""

    name: str
    type: str = "STRING"  # STRING, INTEGER, FLOAT, BOOLEAN, DATE, LIST
    description: str | None = None

    @model_validator(mode="after")
    def _normalize_type(self) -> Attribute:
        normalized = (self.type or "STRING").strip().upper()
        if normalized not in _PROPERTY_TYPES:
            raise ValueError(
                f"Attribute '{self.name}' has unsupported type "
                f"{self.type!r}. Allowed: {sorted(_PROPERTY_TYPES)}"
            )
        self.type = normalized
        return self


class Entity(DataModel):
    """Definition of a node/entity type in the ontology."""

    label: str
    description: str | None = None
    properties: list[Attribute] = Field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.label == other.label


class Relation(DataModel):
    """Definition of a relationship type in the ontology.

    ``patterns`` lists allowed directional ``(source_label, target_label)``
    pairs. Direction matters: ``("Person", "Company")`` means
    ``(Person)-[REL]->(Company)`` — *not* the inverse.

    Example::

        # "Person works at Company"
        Relation(
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
    properties: list[Attribute] = Field(default_factory=list)

    # Identity is by label only — two Relation instances with the same
    # label but different patterns compare/hash equal. Schemas are expected
    # to declare each relation label once.
    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relation):
            return NotImplemented
        return self.label == other.label


class Ontology(DataModel):
    """The user-facing schema of a knowledge graph.

    Lists declared entity types, relation types, and their typed attributes.
    Used by extraction to constrain LLM output and by Cypher generation to
    surface available labels / properties / patterns to the LLM.
    """

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _warn_on_undeclared_pattern_labels(self) -> Ontology:
        """Warn when a ``Relation.patterns`` references undeclared entity labels.

        Catches typos like ``("Persn", "Company")`` at config time, before any
        extraction has run. Open-schema setups may legitimately reference
        labels not yet listed in ``entities``, so this is a warning, not an
        error.

        Note: the SDK writes certain keys on every node/edge (``name``,
        ``description``, ``source_chunk_ids``, ``spans``, ``rel_type``,
        ``fact``, ``src_name``, ``tgt_name``, ``id``, ``label``). Declaring
        a ``Attribute`` with one of these names will shadow the system
        value on extracted nodes. Don't do that.
        """
        if not self.entities:
            return self
        declared = {e.label for e in self.entities}
        for rel in self.relations:
            for src, tgt in rel.patterns:
                missing = [lbl for lbl in (src, tgt) if lbl not in declared]
                if missing:
                    logger.warning(
                        "Relation '%s' pattern (%s, %s) references "
                        "entity label(s) not declared in ontology.entities: %s",
                        rel.label,
                        src,
                        tgt,
                        ", ".join(missing),
                    )
        return self

    @classmethod
    def from_file(cls, path: str) -> Ontology:
        """Load a ``Ontology`` from a JSON file.

        The schema-as-config workflow: keep the canonical schema in a JSON
        file under version control, load it into the SDK with one call. See
        :py:meth:`save_to_file` for the reverse direction.
        """
        from pathlib import Path

        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_schema_org(cls) -> Ontology:
        """Return a curated subset of `Schema.org <https://schema.org>`_ as an ``Ontology``.

        Schema.org is the standard web vocabulary Google / Microsoft / Yahoo
        use for structured-data markup on web pages. It defines ~800 types;
        this method ships a small, opinionated subset covering the most
        common: ``Person``, ``Organization``, ``Place``, ``Event``, and
        ``CreativeWork``, with their typical attributes and the usual
        relations between them.

        Use it when:

        - You are indexing general web content / news / Wikipedia-style
          text and the standard vocabulary fits.
        - You want JSON-LD interop with other systems that already speak
          Schema.org.
        - You don't have domain-specific entities and want a starter
          schema in one line.

        Don't use it when your domain has specific types (biotech, legal,
        finance) that don't map cleanly onto Person / Organization / Place
        — :py:meth:`from_sources` will draft a better fit from a corpus.

        Extend or trim freely after construction:

        >>> ontology = Ontology.from_schema_org()
        >>> ontology = ontology.merge(my_extra_types)

        Attribute names are snake-cased to match the SDK's convention; the
        original camelCase Schema.org name is recorded in each
        attribute's ``description`` so the mapping stays discoverable.
        """
        return cls(
            entities=[
                Entity(
                    label="Person",
                    description="A human individual (Schema.org: Person)",
                    properties=[
                        Attribute(name="name", type="STRING"),
                        Attribute(
                            name="birth_date",
                            type="DATE",
                            description="Schema.org birthDate",
                        ),
                        Attribute(
                            name="death_date",
                            type="DATE",
                            description="Schema.org deathDate",
                        ),
                        Attribute(
                            name="job_title",
                            type="STRING",
                            description="Schema.org jobTitle",
                        ),
                        Attribute(name="email", type="STRING"),
                    ],
                ),
                Entity(
                    label="Organization",
                    description=(
                        "A business, NGO, government body, school, etc. (Schema.org: Organization)"
                    ),
                    properties=[
                        Attribute(name="name", type="STRING"),
                        Attribute(
                            name="founding_date",
                            type="DATE",
                            description="Schema.org foundingDate",
                        ),
                        Attribute(name="email", type="STRING"),
                        Attribute(name="url", type="STRING"),
                    ],
                ),
                Entity(
                    label="Place",
                    description=(
                        "A geographic place — city, country, building, etc. (Schema.org: Place)"
                    ),
                    properties=[
                        Attribute(name="name", type="STRING"),
                        Attribute(name="address", type="STRING"),
                        Attribute(name="latitude", type="FLOAT"),
                        Attribute(name="longitude", type="FLOAT"),
                    ],
                ),
                Entity(
                    label="Event",
                    description=(
                        "A scheduled happening — conference, concert, sale, "
                        "etc. (Schema.org: Event)"
                    ),
                    properties=[
                        Attribute(name="name", type="STRING"),
                        Attribute(
                            name="start_date",
                            type="DATE",
                            description="Schema.org startDate",
                        ),
                        Attribute(
                            name="end_date",
                            type="DATE",
                            description="Schema.org endDate",
                        ),
                    ],
                ),
                Entity(
                    label="CreativeWork",
                    description=(
                        "A piece of authored content — article, book, film, "
                        "dataset, software (Schema.org: CreativeWork)"
                    ),
                    properties=[
                        Attribute(name="name", type="STRING"),
                        Attribute(
                            name="date_published",
                            type="DATE",
                            description="Schema.org datePublished",
                        ),
                        Attribute(name="url", type="STRING"),
                    ],
                ),
            ],
            relations=[
                Relation(
                    label="WORKS_AT",
                    description="Person works for Organization (Schema.org: worksFor)",
                    patterns=[("Person", "Organization")],
                ),
                Relation(
                    label="ALUMNI_OF",
                    description=("Person studied at Organization (Schema.org: alumniOf)"),
                    patterns=[("Person", "Organization")],
                ),
                Relation(
                    label="FOUNDED",
                    description=(
                        "Person founded Organization (Schema.org: founder, inverse direction)"
                    ),
                    patterns=[("Person", "Organization")],
                ),
                Relation(
                    label="BORN_IN",
                    description="Person was born in Place (Schema.org: birthPlace)",
                    patterns=[("Person", "Place")],
                ),
                Relation(
                    label="LOCATED_IN",
                    description=(
                        "Subject is located in Place (Schema.org: location, containedInPlace)"
                    ),
                    patterns=[
                        ("Organization", "Place"),
                        ("Event", "Place"),
                        ("Place", "Place"),
                    ],
                ),
                Relation(
                    label="ORGANIZED_BY",
                    description=(
                        "Event is organized by an Organization or Person (Schema.org: organizer)"
                    ),
                    patterns=[
                        ("Event", "Organization"),
                        ("Event", "Person"),
                    ],
                ),
                Relation(
                    label="AUTHORED_BY",
                    description=("CreativeWork was created by Person (Schema.org: author)"),
                    patterns=[("CreativeWork", "Person")],
                ),
                Relation(
                    label="PUBLISHED_BY",
                    description=(
                        "CreativeWork was published by Organization (Schema.org: publisher)"
                    ),
                    patterns=[("CreativeWork", "Organization")],
                ),
            ],
        )

    @classmethod
    async def from_sources(
        cls,
        sources: str | list[str],
        llm: Any,
        *,
        boundaries: str | None = None,
        existing: Ontology | None = None,
        sample_chunks_per_doc: int = 3,
        max_retries: int = 3,
        concurrency: int = 4,
        chunker: Any | None = None,
        loader: Any | None = None,
        ctx: Any | None = None,
        seed: int | None = None,
    ) -> Ontology:
        """Auto-discover an ontology from a corpus of documents.

        Samples chunks from each source, asks the LLM to propose entity
        types and relation types, merges across the corpus, and runs a
        normalization pass to collapse synonyms before returning. The
        returned ontology is intended as a *draft* — inspect, edit, save
        with :py:meth:`save_to_file`, then pass into ``GraphRAG``.

        When ``existing`` is supplied, its labels are used as a soft
        controlled vocabulary in the prompts and the discovery output is
        merged with it on return — re-running with the same ``existing``
        prior across new documents lets the ontology grow coherently
        instead of drifting.

        See :py:class:`~graphrag_sdk.discovery.OntologyDiscoveryError`
        for the exception raised by hard failures inside individual LLM
        calls. The pipeline itself is soft-fail: a single bad chunk does
        not abort the run.

        Example::

            draft = await Ontology.from_sources(
                ["docs/paper1.md", "docs/paper2.pdf"],
                llm=LiteLLM(model="openai/gpt-4o-mini"),
                boundaries="biotech research papers about CRISPR",
                sample_chunks_per_doc=3,
            )
            draft.save_to_file("ontology.json")  # user edits / curates
        """
        # Lazy import so models.py stays Pydantic-only at module-load time
        # (no LLM provider / loader imports pulled into the data model).
        from graphrag_sdk.discovery.pipeline import discover_ontology

        return await discover_ontology(
            sources,
            llm,
            boundaries=boundaries,
            existing=existing,
            sample_chunks_per_doc=sample_chunks_per_doc,
            max_retries=max_retries,
            concurrency=concurrency,
            chunker=chunker,
            loader=loader,
            ctx=ctx,
            seed=seed,
        )

    def save_to_file(self, path: str, *, indent: int = 2) -> None:
        """Write this schema to ``path`` as JSON (overwrites existing files)."""
        from pathlib import Path

        Path(path).write_text(self.model_dump_json(indent=indent), encoding="utf-8")

    def merge(self, other: Ontology) -> Ontology:
        """Return a new ``Ontology`` that is the union of ``self`` and ``other``.

        - Entity / relation types are unioned by ``label``.
        - For each type, ``properties`` are unioned by ``name``. When the same
          property name appears in both, the incoming type/description overrides
          (last-write-wins).
        - For relations, ``patterns`` are unioned (order-preserving, deduped).
        """

        def _merge_props(existing: list[Attribute], incoming: list[Attribute]) -> list[Attribute]:
            by_name: dict[str, Attribute] = {p.name: p for p in existing}
            for p in incoming:
                by_name[p.name] = p
            return list(by_name.values())

        ent_by_label: dict[str, Entity] = {e.label: e for e in self.entities}
        for e in other.entities:
            if e.label in ent_by_label:
                cur = ent_by_label[e.label]
                ent_by_label[e.label] = Entity(
                    label=cur.label,
                    description=e.description or cur.description,
                    properties=_merge_props(cur.properties, e.properties),
                )
            else:
                ent_by_label[e.label] = e

        rel_by_label: dict[str, Relation] = {r.label: r for r in self.relations}
        for r in other.relations:
            if r.label in rel_by_label:
                cur = rel_by_label[r.label]
                seen: set[tuple[str, str]] = set()
                merged_patterns: list[tuple[str, str]] = []
                for pat in list(cur.patterns) + list(r.patterns):
                    if pat not in seen:
                        seen.add(pat)
                        merged_patterns.append(pat)
                rel_by_label[r.label] = Relation(
                    label=cur.label,
                    description=r.description or cur.description,
                    patterns=merged_patterns,
                    properties=_merge_props(cur.properties, r.properties),
                )
            else:
                rel_by_label[r.label] = r

        return Ontology(
            entities=list(ent_by_label.values()),
            relations=list(rel_by_label.values()),
        )


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
    attributes: dict[str, Any] = Field(default_factory=dict)


class ExtractedRelation(DataModel):
    """A relationship extracted from a text chunk."""

    source: str
    target: str
    type: str
    keywords: str = ""
    description: str = ""
    weight: float = 1.0
    source_chunk_ids: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


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


# ── Deprecation aliases ──────────────────────────────────────────
#
# These older names were used prior to the v1.2.x ontology vocabulary rename
# (commit 363a53d). We keep them importable so existing code keeps working,
# but every access emits a ``DeprecationWarning`` pointing at the new name.
# Once downstream callers have migrated we can drop this shim.

_LEGACY_MODEL_ALIASES: dict[str, str] = {
    "GraphSchema": "Ontology",
    "EntityType": "Entity",
    "RelationType": "Relation",
    "PropertyType": "Attribute",
}


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _LEGACY_MODEL_ALIASES:
        import warnings

        new_name = _LEGACY_MODEL_ALIASES[name]
        warnings.warn(
            f"`{name}` has been renamed to `{new_name}` (graphrag_sdk v1.2+). "
            f"Update your imports — the alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
