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
    FinalizeInput,
    IngestInput,
    OntologyInput,
    PlanRuleInput,
    PreviewFilesInput,
    RetrieveInput,
    SetOntologyInput,
    StatusInput,
    ToolDef,
    ToolSpec,
    build_tool_specs,
    tool_enabled,
)

ALL_NAMES = (
    "status",
    "get_ontology",
    "retrieve",
    "check_citations",
    "preview_files",
    "set_ontology",
    "ingest",
    "finalize",
)
READ_NAMES = ALL_NAMES[:4]

VALID_INPUTS: dict[type[BaseModel], dict] = {
    StatusInput: {},
    OntologyInput: {},
    RetrieveInput: {"question": "Who?"},
    CheckCitationsInput: {"answer": "x [1]"},
    PreviewFilesInput: {},
    SetOntologyInput: {"entities": [{"label": "Drug"}]},
    IngestInput: {"text": "Alice works at Acme."},
    FinalizeInput: {},
}


def specs(**kwargs) -> list[ToolSpec]:
    return build_tool_specs(
        read_only=kwargs.get("read_only", False),
        finalize_policy=kwargs.get("finalize_policy", "manual"),
        include=kwargs.get("include"),
        files=kwargs.get("files", True),
        plan_limits=kwargs.get("plan_limits"),
    )


# ── Registry ─────────────────────────────────────────────────────


def test_registry_has_ten_tools_in_order():
    assert TOOL_NAMES == ALL_NAMES
    assert [td.name for td in TOOL_REGISTRY] == list(ALL_NAMES)
    assert all(isinstance(td, ToolDef) for td in TOOL_REGISTRY)


def test_every_tool_has_its_own_input_model():
    models = [td.input_model for td in TOOL_REGISTRY]
    assert len(set(models)) == len(models)
    assert set(models) == set(VALID_INPUTS)


def test_write_flags():
    flags = {
        td.name: (td.read_only, td.destructive, td.manual_only, td.ingest_only)
        for td in TOOL_REGISTRY
    }
    # The write tools are marked destructive so MCP clients ask the user first.
    assert flags["preview_files"] == (True, False, False, True)
    assert flags["set_ontology"] == (False, True, False, True)
    assert flags["ingest"] == (False, True, False, True)
    assert flags["finalize"] == (False, True, True, True)
    for name in READ_NAMES:
        assert flags[name] == (True, False, False, False), name
    assert [td.name for td in TOOL_REGISTRY if td.needs_files] == ["preview_files"]


# ── build_tool_specs ─────────────────────────────────────────────


def test_default_build_lists_every_tool():
    built = specs()
    assert [s.name for s in built] == list(ALL_NAMES)
    final = built[-1]
    assert final.name == "finalize" and final.destructive and not final.read_only
    assert all(isinstance(s, ToolSpec) for s in built)


def test_read_only_drops_the_ingestion_tools():
    names = [s.name for s in specs(read_only=True)]
    assert names == list(READ_NAMES)
    assert all(s.read_only and not s.destructive for s in specs(read_only=True))


def test_preview_files_needs_a_files_root():
    names = [s.name for s in specs(files=False)]
    assert "preview_files" not in names
    assert {"set_ontology", "ingest", "finalize"} <= set(names)
    sample = next(td for td in TOOL_REGISTRY if td.name == "preview_files")
    assert tool_enabled(sample, read_only=False, finalize_policy="manual", include=None, files=True)
    assert not tool_enabled(
        sample, read_only=True, finalize_policy="manual", include=None, files=True
    )


def test_ingest_description_lists_the_allowed_plan_options():
    ingest = next(s for s in specs() if s.name == "ingest")
    assert "Allowed plan options" in ingest.description
    assert (
        "gliner" in ingest.description
        and "llm_verified" not in ingest.description.split("Allowed plan options")[1]
    )
    wide = next(
        s
        for s in specs(plan_limits={"extractors": ("gliner", "llm"), "resolvers": ("exact",)})
        if s.name == "ingest"
    )
    assert "llm" in wide.description.split("Allowed plan options")[1]


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
    schema = by_name["retrieve"].input_schema
    jsonschema.validate({"question": "Who?", "top_k": 5}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"question": "Who?", "write": True}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"question": "Who?", "top_k": 26}, schema)


def test_tools_that_return_evidence_say_so():
    by_name = {s.name: s for s in specs()}
    assert "evidence numbers [n]" in by_name["retrieve"].description


# ── Input models ─────────────────────────────────────────────────


@pytest.mark.parametrize("model", list(VALID_INPUTS), ids=lambda m: m.__name__)
def test_inputs_accept_minimal_arguments(model):
    model.model_validate(VALID_INPUTS[model])


@pytest.mark.parametrize("model", list(VALID_INPUTS), ids=lambda m: m.__name__)
def test_inputs_reject_unknown_arguments(model):
    with pytest.raises(ValidationError, match="extra"):
        model.model_validate({**VALID_INPUTS[model], "unexpected": 1})


@pytest.mark.parametrize(
    ("model", "base", "field", "ok", "bad"),
    [
        (RetrieveInput, {"question": "q"}, "top_k", 25, 26),
        (RetrieveInput, {"question": "q"}, "expand_hops", 5, 6),
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


def test_ingest_input_modes():
    IngestInput(files=["*.md"])
    IngestInput(files=["a.md"], plan=[{"match": "*.md", "chunker": "structural"}], reextract=True)
    IngestInput(text="x", document_id="doc")
    bad = [
        {},
        {"text": "x", "files": ["a.md"]},
        {"text": "x", "plan": [{"match": "*"}]},
        {"text": "x", "reextract": True},
        {"files": ["a.md"], "document_id": "d"},
        {"files": []},
        {"files": ["a.md"], "plan": [{"chunker": "semantic"}]},
        {"files": ["a.md"], "plan": [{"match": "*", "nope": 1}]},
    ]
    for args in bad:
        with pytest.raises(ValidationError):
            IngestInput(**args)


def test_plan_rule_names_match_the_planner_options():
    from graphrag_sdk.ingestion.ingestion_planner import CHUNKERS, EXTRACTORS, RESOLVERS

    schema = PlanRuleInput.model_json_schema()["properties"]

    def options(field: str) -> set[str]:
        return {v for alt in schema[field]["anyOf"] for v in alt.get("enum", [])}

    assert options("chunker") == set(CHUNKERS)
    assert options("extractor") == set(EXTRACTORS)
    assert options("resolver") == set(RESOLVERS)


def test_preview_files_and_set_ontology_bounds():
    PreviewFilesInput(limit=100, max_chars=4000)
    for bad in ({"limit": 0}, {"limit": 101}, {"max_chars": 99}, {"offset": -1}):
        with pytest.raises(ValidationError):
            PreviewFilesInput(**bad)
    SetOntologyInput(relations=[{"label": "TREATS", "patterns": [["Drug", "Disease"]]}])
    for bad_rel in ({"label": "R", "patterns": [["A"]]}, {"label": "", "patterns": []}):
        with pytest.raises(ValidationError):
            SetOntologyInput(relations=[bad_rel])


def test_empty_answer_may_be_checked_but_empty_text_not_ingested():
    CheckCitationsInput(answer="")
    with pytest.raises(ValidationError):
        IngestInput(text="")
    with pytest.raises(ValidationError):
        RetrieveInput(question="")
