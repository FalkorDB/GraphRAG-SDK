"""Tests for graphrag_sdk.tools.specs: tool definitions and input models."""

from __future__ import annotations

import json

import jsonschema
import pytest
from pydantic import BaseModel, ValidationError

from graphrag_sdk.tools.specs import (
    TOOL_NAMES,
    TOOL_REGISTRY,
    CheckCitationsInput,
    CypherQueryInput,
    FinalizeInput,
    IngestInput,
    OntologyInput,
    RetrieveInput,
    StatusInput,
    ToolDef,
    ToolSpec,
    build_tool_specs,
)

ALL_NAMES = (
    "status",
    "get_ontology",
    "cypher_query",
    "retrieve",
    "check_citations",
    "ingest",
    "finalize",
)

VALID_INPUTS: dict[type[BaseModel], dict] = {
    StatusInput: {},
    OntologyInput: {},
    CypherQueryInput: {"cypher": "MATCH (n) RETURN n"},
    RetrieveInput: {"question": "Who?"},
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


def test_registry_has_eight_tools_in_order():
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
    for name in ("cypher_query", "retrieve"):
        assert "evidence numbers [n]" in by_name[name].description


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


@pytest.mark.parametrize(
    ("model", "base", "field", "ok", "bad"),
    [
        (RetrieveInput, {"question": "q"}, "top_k", 25, 26),
        (RetrieveInput, {"question": "q"}, "expand_hops", 3, 4),
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
        (RetrieveInput, "question", 2000),
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
