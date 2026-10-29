"""Tests for graphrag_sdk.tools.specs: tool definitions and input models."""

from __future__ import annotations

import json

import jsonschema
import pytest
from pydantic import BaseModel, ValidationError

from graphrag_sdk.skills import SKILL_REGISTRY
from graphrag_sdk.tools import graph_ops as ops
from graphrag_sdk.tools.specs import (
    TOOL_NAMES,
    TOOL_REGISTRY,
    CheckCitationsInput,
    CypherQueryInput,
    FinalizeInput,
    GraphWalkInput,
    IngestInput,
    LookupEntityInput,
    OntologyInput,
    RetrieveInput,
    RunSkillInput,
    StatusInput,
    ToolDef,
    ToolSpec,
    TraverseInput,
    build_tool_specs,
    skill_catalog,
)

ALL_NAMES = (
    "status",
    "get_ontology",
    "lookup_entity",
    "traverse",
    "graph_walk",
    "cypher_query",
    "retrieve",
    "run_skill",
    "check_citations",
    "ingest",
    "finalize",
)

VALID_INPUTS: dict[type[BaseModel], dict] = {
    StatusInput: {},
    OntologyInput: {},
    LookupEntityInput: {"name": "Acme"},
    TraverseInput: {"start": "Alice"},
    GraphWalkInput: {"start": "Alice"},
    CypherQueryInput: {"cypher": "MATCH (n) RETURN n"},
    RetrieveInput: {"question": "Who?"},
    RunSkillInput: {"skill": "impact_analysis"},
    CheckCitationsInput: {"answer": "x [1]"},
    IngestInput: {"text": "Alice works at Acme."},
    FinalizeInput: {},
}


def specs(**kwargs) -> list[ToolSpec]:
    return build_tool_specs(
        read_only=kwargs.get("read_only", False),
        finalize_policy=kwargs.get("finalize_policy", "manual"),
        include=kwargs.get("include"),
    )


# ── Registry ─────────────────────────────────────────────────────


def test_registry_has_eleven_tools_in_order():
    assert TOOL_NAMES == ALL_NAMES
    assert [td.name for td in TOOL_REGISTRY] == list(ALL_NAMES)
    assert all(isinstance(td, ToolDef) for td in TOOL_REGISTRY)


def test_every_tool_has_its_own_input_model():
    models = [td.input_model for td in TOOL_REGISTRY]
    assert len(set(models)) == len(models)
    assert set(models) == set(VALID_INPUTS)


def test_write_flags():
    flags = {td.name: (td.read_only, td.destructive, td.manual_only) for td in TOOL_REGISTRY}
    # Both write tools are marked destructive so MCP clients ask the user first.
    assert flags["ingest"] == (False, True, False)
    assert flags["finalize"] == (False, True, True)
    for name in ALL_NAMES[:-2]:
        assert flags[name] == (True, False, False), name


# ── build_tool_specs ─────────────────────────────────────────────


def test_default_build_lists_every_tool():
    built = specs()
    assert [s.name for s in built] == list(ALL_NAMES)
    final = built[-1]
    assert final.name == "finalize" and final.destructive and not final.read_only
    assert all(isinstance(s, ToolSpec) for s in built)


def test_read_only_drops_writing_tools():
    names = [s.name for s in specs(read_only=True)]
    assert names == list(ALL_NAMES[:-2])
    assert all(s.read_only and not s.destructive for s in specs(read_only=True))


@pytest.mark.parametrize("policy", ["on_write", "never"])
def test_finalize_is_only_offered_with_the_manual_policy(policy):
    names = [s.name for s in specs(finalize_policy=policy)]
    assert "finalize" not in names and "ingest" in names


@pytest.mark.parametrize(
    ("policy", "mentioned"), [("manual", True), ("on_write", False), ("never", False)]
)
def test_ingest_mentions_finalize_only_when_the_agent_calls_it(policy, mentioned):
    ingest = next(s for s in specs(finalize_policy=policy) if s.name == "ingest")
    assert ("Call finalize" in ingest.description) is mentioned


def test_include_filters_and_keeps_registry_order():
    built = specs(include=frozenset({"retrieve", "status", "nonexistent"}))
    assert [s.name for s in built] == ["status", "retrieve"]
    assert specs(include=frozenset()) == []
    assert specs(read_only=True, include=frozenset({"ingest", "status"}))[0].name == "status"
    assert len(specs(read_only=True, include=frozenset({"ingest", "status"}))) == 1


def test_specs_are_frozen():
    spec = specs()[0]
    with pytest.raises(ValidationError):
        spec.name = "other"


def test_specs_are_json_serialisable():
    payload = json.dumps([s.model_dump() for s in specs()])
    again = [ToolSpec.model_validate(s) for s in json.loads(payload)]
    assert again == specs()


@pytest.mark.parametrize("spec", specs(), ids=lambda s: s.name)
def test_input_schemas_are_valid_and_closed(spec):
    jsonschema.Draft202012Validator.check_schema(spec.input_schema)
    assert spec.input_schema["type"] == "object"
    assert spec.input_schema["additionalProperties"] is False
    for name, prop in spec.input_schema.get("properties", {}).items():
        assert prop.get("description"), f"{spec.name}.{name} has no description"


@pytest.mark.parametrize("spec", specs(), ids=lambda s: s.name)
def test_descriptions_are_substantial(spec):
    assert len(spec.description) >= 40


def test_schema_accepts_valid_and_rejects_extra_arguments():
    by_name = {s.name: s for s in specs()}
    schema = by_name["cypher_query"].input_schema
    jsonschema.validate({"cypher": "RETURN 1", "max_rows": 5}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"cypher": "RETURN 1", "write": True}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"cypher": "RETURN 1", "max_rows": 101}, schema)


def test_tools_that_return_evidence_say_so():
    by_name = {s.name: s for s in specs()}
    for name in ("traverse", "graph_walk", "cypher_query", "retrieve"):
        assert "evidence numbers [n]" in by_name[name].description
    assert "Not a source to cite" in by_name["lookup_entity"].description


# ── Skill catalogue ──────────────────────────────────────────────


def test_skill_catalog_lists_every_skill_with_its_arguments():
    catalog = skill_catalog()
    lines = catalog.splitlines()
    assert len(lines) == len(SKILL_REGISTRY)
    assert [line.split("(")[0] for line in lines] == [f"- {n}" for n in sorted(SKILL_REGISTRY)]
    for name, cls in SKILL_REGISTRY.items():
        line = next(ln for ln in lines if ln.startswith(f"- {name}("))
        assert line.endswith(f"): {cls.description}")
        props = (cls.parameters or {}).get("properties", {})
        required = set((cls.parameters or {}).get("required", []))
        for prop in props:
            assert (prop if prop in required else f"{prop}?") in line


def test_run_skill_description_embeds_the_catalog():
    run_skill = next(s for s in specs() if s.name == "run_skill")
    assert f"Skills:\n{skill_catalog()}\n" in run_skill.description
    assert "params" in run_skill.description
    for name in SKILL_REGISTRY:
        assert f"- {name}(" in run_skill.description


# ── Input models ─────────────────────────────────────────────────


@pytest.mark.parametrize("model", list(VALID_INPUTS), ids=lambda m: m.__name__)
def test_inputs_accept_minimal_arguments(model):
    model.model_validate(VALID_INPUTS[model])


@pytest.mark.parametrize("model", list(VALID_INPUTS), ids=lambda m: m.__name__)
def test_inputs_reject_unknown_arguments(model):
    with pytest.raises(ValidationError, match="extra"):
        model.model_validate({**VALID_INPUTS[model], "unexpected": 1})


def test_cypher_query_defaults_and_bounds():
    q = CypherQueryInput(cypher="RETURN 1")
    assert (q.max_rows, q.timeout_ms, q.params) == (25, 5000, None)
    CypherQueryInput(cypher="RETURN 1", max_rows=100, timeout_ms=100)
    CypherQueryInput(cypher="RETURN 1", max_rows=1, timeout_ms=10_000)
    CypherQueryInput(cypher="x" * 5000)
    bad = [
        {"max_rows": 0},
        {"max_rows": 101},
        {"timeout_ms": 99},
        {"timeout_ms": 10_001},
    ]
    for extra in bad:
        with pytest.raises(ValidationError):
            CypherQueryInput(cypher="RETURN 1", **extra)
    for cypher in ("", "x" * 5001):
        with pytest.raises(ValidationError):
            CypherQueryInput(cypher=cypher)


def test_traverse_defaults_and_bounds():
    t = TraverseInput(start="Alice")
    assert t.via is None
    assert t.max_depth == ops.TRAVERSE_DEFAULT_DEPTH
    assert t.max_expansions == ops.TRAVERSE_DEFAULT_EXPANSIONS
    assert t.timeout_ms == ops.TRAVERSE_DEFAULT_TIMEOUT_MS
    TraverseInput(start="a", via=["R"] * 5, max_depth=5, max_expansions=200, timeout_ms=5000)
    TraverseInput(start="a", max_depth=1, max_expansions=1, timeout_ms=50)
    bad = [
        {"via": ["R"] * 6},
        {"max_depth": 0},
        {"max_depth": 6},
        {"max_expansions": 0},
        {"max_expansions": 201},
        {"timeout_ms": 49},
        {"timeout_ms": 5001},
    ]
    for extra in bad:
        with pytest.raises(ValidationError):
            TraverseInput(start="a", **extra)
    with pytest.raises(ValidationError):
        TraverseInput(start="")


def test_traverse_bounds_match_graph_ops():
    assert (ops.TRAVERSE_MAX_VIA, ops.TRAVERSE_MAX_DEPTH) == (5, 5)
    assert ops.TRAVERSE_MAX_EXPANSIONS == 200
    assert (ops.TRAVERSE_MIN_TIMEOUT_MS, ops.TRAVERSE_MAX_TIMEOUT_MS) == (50, 5000)


@pytest.mark.parametrize(
    ("model", "base", "field", "ok", "bad"),
    [
        (LookupEntityInput, {"name": "a"}, "limit", 25, 26),
        (RetrieveInput, {"question": "q"}, "top_k", 25, 26),
        (RetrieveInput, {"question": "q"}, "expand_hops", 3, 4),
        (GraphWalkInput, {"start": "a"}, "beam_width", 20, 21),
        (GraphWalkInput, {"start": "a"}, "max_depth", 5, 6),
    ],
)
def test_other_upper_bounds(model, base, field, ok, bad):
    model(**base, **{field: ok})
    with pytest.raises(ValidationError):
        model(**base, **{field: bad})
    with pytest.raises(ValidationError):
        model(**base, **{field: 0})


@pytest.mark.parametrize(
    ("model", "field", "limit"),
    [
        (LookupEntityInput, "name", 500),
        (RetrieveInput, "question", 2000),
        (RunSkillInput, "skill", 64),
        (CheckCitationsInput, "answer", 50_000),
        (IngestInput, "text", 200_000),
    ],
)
def test_string_length_limits(model, field, limit):
    model(**{field: "x" * limit})
    with pytest.raises(ValidationError):
        model(**{field: "x" * (limit + 1)})


def test_empty_answer_may_be_checked_but_empty_text_not_ingested():
    CheckCitationsInput(answer="")
    with pytest.raises(ValidationError):
        IngestInput(text="")
    with pytest.raises(ValidationError):
        RetrieveInput(question="")
