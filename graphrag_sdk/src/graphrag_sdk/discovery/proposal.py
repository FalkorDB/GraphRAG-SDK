# GraphRAG SDK — Discovery: Proposal types
#
# Pydantic models that the LLM is asked to emit at each step of
# ``Ontology.from_sources``. Kept narrow so that ``model_json_schema()``
# gives the LLM a precise target and ``model_validate_json()`` rejects
# anything else.

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from graphrag_sdk.core.models import Attribute, Entity, Relation

# Pydantic v2 silently *ignores* unknown keys on a BaseModel by default. For
# discovery's LLM-emitted shapes we want the opposite — an extra field means
# the model misunderstood the schema and the response should be rejected so
# the validation-retry loop in ``extract_with_retry`` can feed back specific
# correction.
_STRICT = ConfigDict(extra="forbid")


class _ProposedAttribute(BaseModel):
    """An attribute proposed by the LLM for an entity or relation type."""

    model_config = _STRICT

    name: str
    type: str = "STRING"
    description: str | None = None


class _ProposedEntity(BaseModel):
    """An entity type proposed by the LLM for a chunk."""

    model_config = _STRICT

    label: str
    description: str | None = None
    properties: list[_ProposedAttribute] = Field(default_factory=list)


class _ProposedRelation(BaseModel):
    """A relation type proposed by the LLM for a chunk.

    ``patterns`` lists allowed ``(source_label, target_label)`` pairs in
    extraction direction. Labels here must reference entities proposed in
    the same chunk; the validator enforces that.
    """

    model_config = _STRICT

    label: str
    description: str | None = None
    patterns: list[tuple[str, str]] = Field(default_factory=list)
    properties: list[_ProposedAttribute] = Field(default_factory=list)


class DocSummary(BaseModel):
    """Per-document LLM output that anchors per-chunk extraction.

    Returned by the doc-summary pre-step. ``main_entities`` lists the
    document's central concrete entities (a short list of proper nouns —
    not types); ``aboutness`` is a one-sentence "what this doc is about"
    prefix that every per-chunk prompt for that doc carries.
    """

    model_config = _STRICT

    main_entities: list[str] = Field(default_factory=list)
    aboutness: str = ""


class ChunkProposal(BaseModel):
    """Per-chunk LLM output: entity types and relation types observed in one chunk."""

    model_config = _STRICT

    entities: list[_ProposedEntity] = Field(default_factory=list)
    relations: list[_ProposedRelation] = Field(default_factory=list)


class NormalizedDraft(BaseModel):
    """Post-merge LLM output: a normalized draft after synonym collapse.

    The normalization pass takes the merged corpus-level draft and asks
    the LLM to canonicalize labels (``Org`` + ``Organization`` → one
    label) and fix obviously-reversed relation directions. Same shape as
    ``ChunkProposal`` but the semantics are "final, normalized" — the
    pipeline converts this directly into an ``Ontology``.
    """

    model_config = _STRICT

    entities: list[_ProposedEntity] = Field(default_factory=list)
    relations: list[_ProposedRelation] = Field(default_factory=list)


class SchemaExtensionProposal(BaseModel):
    """Structured proposal of additions to an existing ontology.

    Returned by :py:meth:`graphrag_sdk.GraphRAG.suggest_schema_extensions`.
    Carries only *additions* — never modifications or deletions of
    existing schema. Each field is a list the user can review and apply
    selectively via the v1.2.x mutation API
    (``add_entity`` / ``add_relation_pattern`` / ``add_attribute``).

    Nothing in this object is applied to the graph until the user
    explicitly calls a mutation method.

    Attributes:
        new_entities: Entity types not present in the committed
            ontology. Apply with ``rag.add_entity(entity)``.
        new_relations: Relation types not present in the committed
            ontology. Apply with ``rag.add_relation_pattern(...)`` once
            per pattern.
        new_patterns: Additional ``(rel_label, src, tgt)`` patterns for
            relation types that already exist. Apply with
            ``rag.add_relation_pattern(rel_label, src, tgt)``.
        new_attributes: Additional ``(owner_label, attribute)`` pairs
            for entity types that already exist. Apply with
            ``await rag.add_attribute(owner_label, attribute)``.
        sources_scanned: Source identifiers the proposal was derived
            from. Coarse-grained evidence — see the documentation for
            the rationale and the planned upgrade path.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    new_entities: list[Entity] = Field(default_factory=list)
    new_relations: list[Relation] = Field(default_factory=list)
    new_patterns: list[tuple[str, str, str]] = Field(default_factory=list)
    new_attributes: list[tuple[str, Attribute]] = Field(default_factory=list)
    sources_scanned: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when the proposal has no additions to apply."""
        return not (
            self.new_entities or self.new_relations or self.new_patterns or self.new_attributes
        )

    def summary(self) -> str:
        """One-line summary suitable for logs and CLI output."""
        return (
            f"SchemaExtensionProposal("
            f"entities=+{len(self.new_entities)}, "
            f"relations=+{len(self.new_relations)}, "
            f"patterns=+{len(self.new_patterns)}, "
            f"attributes=+{len(self.new_attributes)}, "
            f"sources_scanned={len(self.sources_scanned)})"
        )


class OntologyDiscoveryError(RuntimeError):
    """Raised when ``Ontology.from_sources`` cannot produce a valid draft.

    ``extract_with_retry`` raises this when an LLM call exhausts its
    retry budget without returning JSON that parses and passes
    validation.

    Attributes:
        chunk_id: Identifier of the unit being processed (a chunk uid, a
            doc id for the summary step, or ``"normalize"`` for the
            normalization pass). ``None`` when not applicable.
        attempts: How many LLM calls were made before giving up.
        last_error: The last validation/parse error encountered.
    """

    def __init__(
        self,
        message: str,
        *,
        chunk_id: str | None = None,
        attempts: int = 0,
        last_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.chunk_id = chunk_id
        self.attempts = attempts
        self.last_error = last_error
