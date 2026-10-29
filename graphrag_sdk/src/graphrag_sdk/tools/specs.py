# GraphRAG SDK — Tools: tool specifications (single source of truth)
# Every adapter (MCP, pydantic-ai, LangGraph, a hand-written loop) builds its
# tool definitions from GraphRAGToolkit.tool_specs(); names, descriptions and
# input schemas are never duplicated downstream. A tool's name is also the
# name of the toolkit method that runs it.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from graphrag_sdk.tools import graph_ops as ops


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StatusInput(_Input):
    """Arguments for status (none)."""


class OntologyInput(_Input):
    """Arguments for get_ontology (none)."""


class LookupEntityInput(_Input):
    """Arguments for lookup_entity."""

    name: str = Field(min_length=1, max_length=500, description="A name or part of one.")
    limit: int = Field(
        default=ops.LOOKUP_DEFAULT_LIMIT,
        ge=1,
        le=ops.LOOKUP_MAX_LIMIT,
        description="Most matches to return.",
    )


class TraverseInput(_Input):
    """Arguments for traverse."""

    start: str = Field(
        min_length=1, max_length=500, description="Exact entity name or id to start from."
    )
    via: list[str] | None = Field(
        default=None,
        max_length=ops.TRAVERSE_MAX_VIA,
        description=(
            "Optional relation path, e.g. ['WORKS_AT', 'LOCATED_IN']: follow exactly these "
            "relations in order and return where the path ends. Omit to explore all "
            "neighbours breadth-first."
        ),
    )
    max_depth: int = Field(
        default=ops.TRAVERSE_DEFAULT_DEPTH,
        ge=1,
        le=ops.TRAVERSE_MAX_DEPTH,
        description="Most hops to walk (ignored with via).",
    )
    max_expansions: int = Field(
        default=ops.TRAVERSE_DEFAULT_EXPANSIONS,
        ge=1,
        le=ops.TRAVERSE_MAX_EXPANSIONS,
        description="Most edges to read before stopping.",
    )
    timeout_ms: int = Field(
        default=ops.TRAVERSE_DEFAULT_TIMEOUT_MS,
        ge=ops.TRAVERSE_MIN_TIMEOUT_MS,
        le=ops.TRAVERSE_MAX_TIMEOUT_MS,
        description="Time budget for the whole walk.",
    )


class GraphWalkInput(_Input):
    """Arguments for graph_walk."""

    start: str = Field(min_length=1, max_length=500, description="Exact entity name or id.")
    goal: str | None = Field(
        default=None,
        max_length=500,
        description="Optional second entity: find the strongest path from start to it.",
    )
    beam_width: int = Field(default=5, ge=1, le=20, description="Paths kept per round.")
    max_depth: int = Field(default=3, ge=1, le=5, description="Most hops per path.")


class CypherQueryInput(_Input):
    """Arguments for cypher_query."""

    cypher: str = Field(min_length=1, max_length=5000, description="One read-only Cypher query.")
    params: dict[str, Any] | None = Field(
        default=None, description="Query parameters ($name); prefer them to inlined values."
    )
    max_rows: int = Field(
        default=25, ge=1, le=100, description="Most rows to return; every LIMIT is capped to it."
    )
    timeout_ms: int = Field(
        default=5000, ge=100, le=10000, description="Database timeout in milliseconds."
    )


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
        default=1, ge=1, le=3, description="Relation hops to expand around found entities."
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


class RunSkillInput(_Input):
    """Arguments for run_skill."""

    skill: str = Field(min_length=1, max_length=64, description="Skill name.")
    params: dict[str, Any] = Field(
        default_factory=dict, description="The skill's arguments (see the tool description)."
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
        "types with their (source, target) patterns. Use it to write traverse paths and "
        "Cypher with the right names.",
    ),
    ToolDef(
        "lookup_entity",
        LookupEntityInput,
        "Find entities by (part of) their name: exact names, ids, labels and short "
        "descriptions, best match first. Use it to turn a name from the question into "
        "the exact name or id the other tools need. Not a source to cite.",
    ),
    ToolDef(
        "traverse",
        TraverseInput,
        "Walk the graph from one entity. Without via: every neighbour breadth-first up "
        "to max_depth, with each edge and its fact. With via: follow that exact relation "
        "path and return where it ends (e.g. via=['WORKS_AT','LOCATED_IN'] for 'where is "
        "the company X works at'). Bounded by depth, edge and time budgets; 'stopped' "
        "says which one ended the walk." + _CITE,
    ),
    ToolDef(
        "graph_walk",
        GraphWalkInput,
        "Find the strongest paths out of an entity, ranked by relation weight and "
        "PageRank importance, or (with goal) the strongest path connecting two entities. "
        "Use it for 'how is A connected to B' and 'what matters most around A'." + _CITE,
    ),
    ToolDef(
        "cypher_query",
        CypherQueryInput,
        "Run one read-only Cypher query for what the other tools cannot express: "
        "counts, sums, sorting, property filters. Entities are (:__Entity__:<Label>) "
        "nodes with name/description; semantic relations are stored as "
        "[:RELATES {rel_type: '<TYPE>', fact: '...'}]. Writes are rejected, every LIMIT "
        "is capped to max_rows; if 'truncated' is true, aggregate with count() instead "
        "of reading rows. Use names from status/get_ontology." + _CITE,
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
        "run_skill",
        RunSkillInput,
        "Run a built-in multi-step analysis over the graph.",
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
        "extracted and stored. Call finalize after a batch of ingests.",
        read_only=False,
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


def skill_catalog() -> str:
    """The registered skills, one line each, for the run_skill description."""
    from graphrag_sdk.skills import SKILL_REGISTRY

    lines = []
    for name in sorted(SKILL_REGISTRY):
        cls = SKILL_REGISTRY[name]
        props = (cls.parameters or {}).get("properties", {})
        required = set((cls.parameters or {}).get("required", []))
        args = ", ".join(f"{p}{'' if p in required else '?'}" for p in props)
        lines.append(f"- {name}({args}): {cls.description}")
    return "\n".join(lines)


def _description(td: ToolDef) -> str:
    if td.name == "run_skill":
        return (
            f"{td.description} Pass the skill's arguments in params. Skills:\n"
            f"{skill_catalog()}\nThe result is one piece of evidence [n] to cite."
        )
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
                description=_description(td),
                input_schema=td.input_model.model_json_schema(),
                read_only=td.read_only,
                destructive=td.destructive,
            )
        )
    return specs
