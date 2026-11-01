# GraphRAG SDK — Tools: tool specifications (single source of truth)
# Every adapter (MCP, pydantic-ai, LangGraph, a hand-written loop) builds its
# tool definitions from GraphRAGToolkit.tool_specs(); names, descriptions and
# input schemas are never duplicated downstream. A tool's name is also the
# name of the toolkit method that runs it.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from graphrag_sdk.tools.files import MAX_FILES_PER_CALL, SUPPORTED_SUFFIXES


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StatusInput(_Input):
    """Arguments for status (none)."""


class OntologyInput(_Input):
    """Arguments for get_ontology (none)."""


#: The search paths retrieve can run, as the agent names them.
RetrievePath = Literal["facts", "exact_names", "fuzzy_names", "links", "passages"]
RETRIEVE_PATHS: tuple[str, ...] = ("facts", "exact_names", "fuzzy_names", "links", "passages")


class RetrieveInput(_Input):
    """Arguments for retrieve."""

    question: str = Field(
        min_length=1, max_length=2000, description="The user's question, word for word."
    )
    top_k: int = Field(default=8, ge=1, le=25, description="Entities and passages to return.")
    expand_hops: int = Field(
        default=1,
        ge=1,
        le=5,
        description=(
            "Relation hops the links path follows from the entities found. Keep 1 "
            "first; on a second call use 2 or 3 when the question chains several "
            "relations, up to 5 for a longer chain. Relations stop at 4 x top_k "
            "(40 at most); each hop gets a share, so deeper hops still run."
        ),
    )
    include_chunks: bool = Field(default=True, description="Include source text passages.")
    paths: list[RetrievePath] | None = Field(
        default=None,
        min_length=1,
        description=(
            "The searches to run. Read the question, pick every path it needs "
            "(usually two or three), and leave paths out only for broad questions or "
            "when you cannot tell.\n"
            "- passages. Finds: the source text. Use when: the answer is a detail, a "
            "number, a reason, a description or a summary. Example: 'Why was the "
            "project cancelled?' Skip when: the question is only about who is "
            "connected to whom.\n"
            "- exact_names. Finds: entities whose name appears in the question as "
            "written. Use when: the question names a person, place, organisation, "
            "product or event. Example: 'What does Acme make?' Skip when: nothing is "
            "named.\n"
            "- fuzzy_names. Finds: entities by part of a name, a misspelling or words "
            "in their descriptions. Use when: a name is partial or uncertain, or the "
            "question describes something without naming it. Example: 'Which "
            "company did the old founder start?' Skip when: every name is given "
            "in full.\n"
            "- links. Finds: the relations around the entities the other paths found "
            "(adds exact_names if you pick no entity path). Use when: the question "
            "asks who or what is connected to something, or goes from one entity to "
            "another. Example: 'Who works at Acme?' Skip when: the answer is in one "
            "passage.\n"
            "- facts. Finds: stored statements about relations, matched by meaning. "
            "Use when: the question asks how things relate, what caused or affected "
            "what, or what depends on what. Example: 'How did the merger affect the "
            "staff?' Skip when: the question asks for a single detail.\n"
            "Examples: 'What is Alice's role at Acme?' -> exact_names, passages. "
            "'Who reports to Alice?' -> exact_names, links. 'Summarise why sales "
            "fell' -> passages, facts."
        ),
    )


class CheckCitationsInput(_Input):
    """Arguments for check_citations."""

    answer: str = Field(max_length=50_000, description="Your draft answer with [n] citations.")


class PreviewFilesInput(_Input):
    """Arguments for preview_files."""

    pattern: str = Field(
        default="**/*",
        min_length=1,
        max_length=500,
        description="Glob relative to the files root, e.g. 'reports/*.pdf'; '*' crosses folders.",
    )
    offset: int = Field(default=0, ge=0, description="Files to skip (from next_offset).")
    limit: int = Field(default=20, ge=1, le=100, description="Files to sample in this call.")
    max_chars: int = Field(
        default=1000, ge=100, le=4000, description="Characters sampled from each file."
    )


Label = Annotated[str, Field(min_length=1, max_length=100)]
Description = Annotated[str, Field(max_length=1000)]


class OntologyEntityInput(_Input):
    """An entity type to declare."""

    label: Label = Field(description="Type name, e.g. 'Drug'.")
    description: Description | None = Field(
        default=None, description="What counts as this type; guides extraction."
    )


class OntologyRelationInput(_Input):
    """A relation type to declare."""

    label: Label = Field(description="Relation name, e.g. 'TREATS'.")
    description: Description | None = Field(default=None, description="What the relation means.")
    patterns: list[Annotated[list[Label], Field(min_length=2, max_length=2)]] = Field(
        default_factory=list,
        max_length=50,
        description="Allowed [source, target] entity types, e.g. [['Drug', 'Disease']].",
    )


class SetOntologyInput(_Input):
    """Arguments for set_ontology."""

    entities: list[OntologyEntityInput] = Field(
        default_factory=list,
        max_length=200,
        description="Entity types to add, or to give a new description.",
    )
    relations: list[OntologyRelationInput] = Field(
        default_factory=list,
        max_length=200,
        description="Relation types to add, or new [source, target] patterns for one.",
    )


#: Option names for plan rules; they match ingestion_planner.CHUNKERS etc.
ChunkerName = Literal["sentence", "fixed", "structural", "contextual"]
ExtractorName = Literal["gliner", "llm"]
ResolverName = Literal["exact", "llm_verified"]


class PlanRuleInput(_Input):
    """One plan rule: files matching ``match`` use these strategies."""

    match: str = Field(
        default="*", min_length=1, max_length=500, description="Glob, e.g. '*.md' or 'notes/*'."
    )
    chunker: ChunkerName | None = None
    extractor: ExtractorName | None = None
    resolver: ResolverName | None = None
    chunker_params: dict[str, float] | None = Field(
        default=None, description="e.g. {'chunk_size': 1500} for fixed."
    )
    extractor_params: dict[str, float] | None = None
    resolver_params: dict[str, float] | None = None
    reason: str | None = Field(default=None, max_length=300)


class IngestInput(_Input):
    """Arguments for ingest."""

    text: str | None = Field(
        default=None, min_length=1, max_length=200_000, description="Text to add to the graph."
    )
    document_id: str | None = Field(
        default=None,
        max_length=500,
        description="Stable id for text; generated when omitted.",
    )
    files: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "Paths or globs relative to the files root, instead of text; at most "
            f"{MAX_FILES_PER_CALL} files per call."
        ),
    )
    plan: list[PlanRuleInput] | None = Field(
        default=None,
        max_length=50,
        description="Rules for files, first match wins; files no rule matches use the defaults.",
    )
    reextract: bool = Field(
        default=False,
        description=(
            "Re-extract content that is unchanged (after extending the ontology); "
            "for text, pass the document_id it was stored under. Without a plan, a "
            "file keeps the plan of its last ingest."
        ),
    )

    @model_validator(mode="after")
    def _one_source(self) -> IngestInput:
        if (self.text is None) == (self.files is None):
            raise ValueError("pass exactly one of text or files")
        if self.text is not None and self.plan is not None:
            raise ValueError("plan applies to files only")
        if self.text is not None and self.reextract and self.document_id is None:
            raise ValueError("reextract on text needs the document_id it was stored under")
        if self.files is not None and self.document_id is not None:
            raise ValueError("document_id applies to text only; a file's id is its path")
        return self


class FinalizeInput(_Input):
    """Arguments for finalize (none)."""


class ToolSpec(BaseModel):
    """A machine-readable tool definition for an agent framework or MCP."""

    model_config = ConfigDict(frozen=True)
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool = True
    destructive: bool = False


@dataclass(frozen=True)
class ToolDef:
    """Registry row: a tool's name (= toolkit method), input model and contract."""

    name: str
    input_model: type[BaseModel]
    description: str
    read_only: bool = True
    destructive: bool = False
    manual_only: bool = False
    #: Part of ingestion: hidden and refused on a read-only toolkit even if
    #: it writes nothing (preview_files reads local files).
    ingest_only: bool = False
    #: Needs a files root (GraphRAGToolkit(files_root=...)).
    needs_files: bool = False


_CITE = " Results carry evidence numbers [n]; cite them in your answer."

TOOL_REGISTRY: tuple[ToolDef, ...] = (
    ToolDef(
        "status",
        StatusInput,
        "What the graph holds: entity, relation, document and chunk counts, and each "
        "entity label and relation type with how many there are. Call this first.",
    ),
    ToolDef(
        "get_ontology",
        OntologyInput,
        "The graph's declared ontology: entity types with their properties and relation "
        "types with their (source, target) patterns.",
    ),
    ToolDef(
        "retrieve",
        RetrieveInput,
        "Hybrid search (vector + full-text + graph) for a question: the most relevant "
        "entities, relations, facts and source passages. Use it for open questions or "
        "when you do not know which entity to start from. Pass the user's question word "
        "for word; keywords find less. Choose paths for the question (see paths); if "
        "the paths you chose find nothing, run it again without paths. Needs the "
        "embedding model the graph was built with." + _CITE,
    ),
    ToolDef(
        "check_citations",
        CheckCitationsInput,
        "Check your draft answer before you reply: every [n] must be evidence a tool "
        "returned in this session. Returns the answer with invalid markers removed, "
        "whether it is grounded, the cited evidence, claims with no [n] of their own, "
        "and cited claims whose figures or names the cited evidence does not contain.",
    ),
    ToolDef(
        "preview_files",
        PreviewFilesInput,
        "Before ingesting files: list the documents under the files root (paged with "
        "offset / next_offset) with one sample from the middle of each, to see what the "
        "corpus is about and design an ontology from it. Reads files; writes nothing.",
        ingest_only=True,
        needs_files=True,
    ),
    ToolDef(
        "set_ontology",
        SetOntologyInput,
        "Declare entity and relation types before ingesting: extraction keeps only "
        "declared types. Adds to the current ontology (existing types keep theirs; a "
        "description given here replaces the old one; new [source, target] patterns are "
        "added to an existing relation). Labels: letters, digits, spaces, '_' or '-'. "
        "Returns the full ontology.",
        read_only=False,
        # It rewrites the stored ontology that every later ingest prunes by,
        # so MCP clients ask the user before running it.
        destructive=True,
        ingest_only=True,
    ),
    ToolDef(
        "ingest",
        IngestInput,
        "Add documents to the graph: entities and relations are extracted and stored. "
        "Pass text (a note or document), or files: paths or globs under the files root "
        f"({', '.join(SUPPORTED_SUFFIXES)}), at most {MAX_FILES_PER_CALL} per call, with "
        "an optional plan, one rule per kind of file, e.g. [{match: '*.md', chunker: "
        "'structural'}, {match: '*.pdf', chunker: 'fixed', chunker_params: {chunk_size: "
        "1500}}]. Unchanged files are skipped; reextract=true runs them again (with "
        "their last plan unless you pass one). The "
        "result gives each file's plan and the types dropped because the ontology lacks "
        "them.",
        read_only=False,
        destructive=True,
        ingest_only=True,
    ),
    ToolDef(
        "finalize",
        FinalizeInput,
        "Finish a batch of ingests: deduplicate entities, embed them and build indexes. "
        "Its cost grows with the graph; call it once after a batch, not after every "
        "ingest.",
        read_only=False,
        destructive=True,
        manual_only=True,
        ingest_only=True,
    ),
)

TOOL_NAMES: tuple[str, ...] = tuple(td.name for td in TOOL_REGISTRY)


def _plan_options(plan_limits: Mapping[str, Any] | None) -> str:
    from graphrag_sdk.ingestion.ingestion_planner import (
        CHEAP_PLAN_LIMITS,
        CHUNKERS,
        EXTRACTORS,
        RESOLVERS,
    )

    limits = CHEAP_PLAN_LIMITS if plan_limits is None else plan_limits
    parts = [
        f"{slot} in ({', '.join(limits.get(key) or default)})"
        for slot, key, default in (
            ("chunker", "chunkers", CHUNKERS),
            ("extractor", "extractors", EXTRACTORS),
            ("resolver", "resolvers", RESOLVERS),
        )
    ]
    return f" Allowed plan options: {'; '.join(parts)}; others fall back to these."


def _description(
    td: ToolDef,
    finalize_policy: str,
    plan_limits: Mapping[str, Any] | None = None,
) -> str:
    if td.name == "ingest":
        description = td.description + _plan_options(plan_limits)
        if finalize_policy == "manual":
            # Only an agent that has the finalize tool is told to call it.
            description += " Call finalize once after the whole batch."
        return description
    return td.description


def tool_enabled(
    td: ToolDef,
    *,
    read_only: bool,
    finalize_policy: str,
    include: frozenset[str] | None,
    files: bool = False,
) -> bool:
    """Whether a toolkit configuration serves tool ``td``."""
    if read_only and (not td.read_only or td.ingest_only):
        return False
    if td.manual_only and finalize_policy != "manual":
        return False
    if td.needs_files and not files:
        return False
    return include is None or td.name in include


def build_tool_specs(
    *,
    read_only: bool,
    finalize_policy: str,
    include: frozenset[str] | None,
    files: bool = False,
    plan_limits: Mapping[str, Any] | None = None,
) -> list[ToolSpec]:
    """The tool definitions a toolkit configuration advertises.

    ``files`` says whether the toolkit has a files root (``preview_files``
    needs one); ``plan_limits`` names the plan options ``ingest`` allows.
    """
    return [
        ToolSpec(
            name=td.name,
            description=_description(td, finalize_policy, plan_limits),
            input_schema=td.input_model.model_json_schema(),
            read_only=td.read_only,
            destructive=td.destructive,
        )
        for td in TOOL_REGISTRY
        if tool_enabled(
            td,
            read_only=read_only,
            finalize_policy=finalize_policy,
            include=include,
            files=files,
        )
    ]
