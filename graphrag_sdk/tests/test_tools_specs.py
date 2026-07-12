"""tool_specs contract: JSON-Schema validity, filtering, stable order."""

from __future__ import annotations

import json

import jsonschema
import pytest

from graphrag_sdk.tools import ToolSpec
from graphrag_sdk.tools.specs import _TOOL_REGISTRY, TOOL_NAMES, build_tool_specs

ALL_NAMES = [
    "graph_search",
    "graph_answer",
    "graph_schema",
    "graph_entity",
    "cypher_read",
    "graph_remember",
    "graph_flush",
]


def test_registry_names_and_order():
    assert list(TOOL_NAMES) == ALL_NAMES


def test_default_build_has_all_tools():
    specs = build_tool_specs(read_only=False, finalize_policy="manual", include=None)
    assert [s.name for s in specs] == ALL_NAMES


def test_read_only_removes_write_tools():
    specs = build_tool_specs(read_only=True, finalize_policy="manual", include=None)
    names = [s.name for s in specs]
    assert "graph_remember" not in names and "graph_flush" not in names
    assert "graph_search" in names


@pytest.mark.parametrize("policy", ["on_write", "never"])
def test_non_manual_policy_hides_flush(policy):
    specs = build_tool_specs(read_only=False, finalize_policy=policy, include=None)
    names = [s.name for s in specs]
    assert "graph_flush" not in names and "graph_remember" in names


def test_include_filters_and_preserves_order():
    specs = build_tool_specs(
        read_only=False,
        finalize_policy="manual",
        include=frozenset({"graph_schema", "graph_search"}),
    )
    assert [s.name for s in specs] == ["graph_search", "graph_schema"]


def test_schemas_are_valid_draft202012_and_strict():
    for spec in build_tool_specs(read_only=False, finalize_policy="manual", include=None):
        jsonschema.Draft202012Validator.check_schema(spec.input_schema)
        assert spec.input_schema.get("additionalProperties") is False, spec.name


def test_specs_round_trip_through_json():
    specs = build_tool_specs(read_only=False, finalize_policy="manual", include=None)
    dumped = json.dumps([s.model_dump() for s in specs])
    assert [ToolSpec.model_validate(d) for d in json.loads(dumped)] == specs


def test_descriptions_are_llm_ready():
    for td in _TOOL_REGISTRY:
        assert len(td.description) >= 40, td.name  # says when to use it
        assert td.output_hint and len(td.output_hint) <= 200, td.name


def test_field_descriptions_present():
    for td in _TOOL_REGISTRY:
        for fname, field in td.input_model.model_fields.items():
            assert field.description, f"{td.name}.{fname} missing description"
