# GraphRAG SDK — Tools: tool specifications (single source of truth)
# Every adapter (MCP, pydantic-ai, LangGraph, a hand-written loop) builds its
# tool definitions from GraphRAGToolkit.tool_specs(); names, descriptions and
# input schemas are never duplicated downstream. A tool's name is also the
# name of the toolkit method that runs it.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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
        le=3,
        description=(
            "Relation hops the links path follows from the entities found. Keep 1 "
            "first; on a second call use 2 or 3 when the question chains several "
            "relations."
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


class IngestInput(_Input):
    """Arguments for ingest."""

    text: str = Field(min_length=1, max_length=200_000, description="Text to add to the graph.")
    document_id: str | None = Field(
        default=None, max_length=500, description="Stable document id; generated when omitted."
    )


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
        "ingest",
        IngestInput,
        "Add text (a note, fact or document) to the graph: entities and relations are "
        "extracted and stored.",
        read_only=False,
        destructive=True,
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
    ),
)

TOOL_NAMES: tuple[str, ...] = tuple(td.name for td in TOOL_REGISTRY)


def _description(td: ToolDef, finalize_policy: str) -> str:
    if td.name == "ingest" and finalize_policy == "manual":
        # Only an agent that has the finalize tool is told to call it.
        return f"{td.description} Call finalize after a batch of ingests."
    return td.description


def build_tool_specs(
    *,
    read_only: bool,
    finalize_policy: str,
    include: frozenset[str] | None,
) -> list[ToolSpec]:
    """The tool definitions a toolkit configuration advertises."""
    specs: list[ToolSpec] = []
    for td in TOOL_REGISTRY:
        if read_only and not td.read_only:
            continue
        if td.manual_only and finalize_policy != "manual":
            continue
        if include is not None and td.name not in include:
            continue
        specs.append(
            ToolSpec(
                name=td.name,
                description=_description(td, finalize_policy),
                input_schema=td.input_model.model_json_schema(),
                read_only=td.read_only,
                destructive=td.destructive,
            )
        )
    return specs
