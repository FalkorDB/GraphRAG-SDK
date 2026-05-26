"""Tests for the attribute-aware prompt and coercion helpers in
``graph_extraction.py``."""
from __future__ import annotations

import pytest

from graphrag_sdk.core.models import (
    Entity,
    Ontology,
    Attribute,
    Relation,
)
from graphrag_sdk.ingestion.extraction_strategies.graph_extraction import (
    VERIFY_EXTRACT_RELS_PROMPT,
    _DEFAULT_JSON_EXAMPLE,
    _JSON_EXAMPLE_WITH_ATTRS,
    _coerce_attribute_value,
    _coerce_attributes,
    _render_attribute_block,
    _ontology_has_attributes,
)


# ── _render_attribute_block ──────────────────────────────


class TestRenderAttributeSchemaBlock:
    def test_empty_schema_renders_empty(self):
        assert _render_attribute_block(Ontology()) == ""

    def test_schema_with_no_attributes_renders_empty(self):
        s = Ontology(entities=[Entity(label="Person")])
        assert _render_attribute_block(s) == ""

    def test_includes_only_types_with_declared_properties(self):
        s = Ontology(
            entities=[
                Entity(
                    label="Person",
                    properties=[Attribute(name="age", type="INTEGER")],
                ),
                Entity(label="Company"),  # no properties; should not appear
            ],
        )
        block = _render_attribute_block(s)
        assert "Person" in block
        assert "age (INTEGER)" in block
        # Bare Company without a colon shouldn't appear; the only "Company"
        # in the rendered output would be as part of an entity bullet.
        assert "- Company:" not in block

    def test_renders_relation_attributes(self):
        s = Ontology(
            entities=[Entity(label="Person"), Entity(label="Company")],
            relations=[
                Relation(
                    label="WORKS_AT",
                    properties=[Attribute(name="since", type="DATE")],
                ),
            ],
        )
        block = _render_attribute_block(s)
        assert "Relation attributes" in block
        assert "WORKS_AT" in block
        assert "since (DATE)" in block


class TestSchemaHasAttributes:
    def test_empty_schema(self):
        assert _ontology_has_attributes(Ontology()) is False

    def test_only_entity_attrs(self):
        s = Ontology(
            entities=[
                Entity(
                    label="Person",
                    properties=[Attribute(name="age", type="INTEGER")],
                )
            ]
        )
        assert _ontology_has_attributes(s) is True

    def test_only_relation_attrs(self):
        s = Ontology(
            relations=[
                Relation(
                    label="WORKS_AT",
                    properties=[Attribute(name="since", type="DATE")],
                )
            ]
        )
        assert _ontology_has_attributes(s) is True


# ── _coerce_attribute_value ──────────────────────────────────────


class TestCoerceAttributeValue:
    @pytest.mark.parametrize(
        "value,prop_type,expected",
        [
            ("56", "INTEGER", (True, 56)),
            (56.7, "INTEGER", (True, 56)),
            ("abc", "INTEGER", (False, None)),
            (None, "INTEGER", (False, None)),
            ("1867-11-07", "DATE", (True, "1867-11-07")),
            ("1867-11-07T12:00:00", "DATE", (True, "1867-11-07")),
            ("not a date", "DATE", (False, None)),
            ("yes", "BOOLEAN", (True, True)),
            ("False", "BOOLEAN", (True, False)),
            ("maybe", "BOOLEAN", (False, None)),
            (1, "BOOLEAN", (True, True)),
            ("hello", "STRING", (True, "hello")),
            ("  ", "STRING", (False, None)),
            (42, "STRING", (True, "42")),
            (["a", "b"], "LIST", (True, ["a", "b"])),
            ("solo", "LIST", (True, ["solo"])),
            ({"bad": True}, "LIST", (False, None)),
        ],
    )
    def test_matrix(self, value, prop_type, expected):
        assert _coerce_attribute_value(value, prop_type) == expected


class TestCoerceAttributes:
    def test_every_declared_property_appears_in_result(self):
        """Result shape is uniform: every declared key is present, with the
        coerced value or None. Never drop the enclosing record."""
        declared = {
            "age": Attribute(name="age", type="INTEGER"),
            "birth_date": Attribute(name="birth_date", type="DATE"),
            "nickname": Attribute(name="nickname", type="STRING"),
        }
        result = _coerce_attributes(
            {"age": "56", "birth_date": "1867-11-07"}, declared
        )
        assert result == {"age": 56, "birth_date": "1867-11-07", "nickname": None}

    def test_uncoercible_value_becomes_none(self):
        declared = {"age": Attribute(name="age", type="INTEGER")}
        assert _coerce_attributes({"age": "abc"}, declared) == {"age": None}

    def test_undeclared_keys_are_ignored(self):
        declared = {"age": Attribute(name="age", type="INTEGER")}
        result = _coerce_attributes({"age": 56, "unknown": "x"}, declared)
        assert result == {"age": 56}


# ── prompt template integration ──────────────────────────────────


class TestPromptTemplate:
    def test_property_less_schema_keeps_block_empty(self):
        s = Ontology(entities=[Entity(label="Person")])
        prompt = VERIFY_EXTRACT_RELS_PROMPT.format(
            entity_types="Person",
            relation_patterns="",
            attribute_block=_render_attribute_block(s),
            relationship_type_instruction="- type: ...\n",
            entities_json="[]",
            text="...",
            json_example=_DEFAULT_JSON_EXAMPLE,
        )
        # No attribute-extraction section should appear.
        assert "## Attribute extraction" not in prompt
        # Original entity-shape example must be intact.
        assert '"description": "..."' in prompt

    def test_schema_with_attributes_injects_section_and_example(self):
        s = Ontology(
            entities=[
                Entity(
                    label="Person",
                    properties=[Attribute(name="age", type="INTEGER")],
                ),
            ]
        )
        prompt = VERIFY_EXTRACT_RELS_PROMPT.format(
            entity_types="Person",
            relation_patterns="",
            attribute_block=_render_attribute_block(s),
            relationship_type_instruction="- type: ...\n",
            entities_json="[]",
            text="...",
            json_example=_JSON_EXAMPLE_WITH_ATTRS,
        )
        assert "## Attribute extraction" in prompt
        assert "age (INTEGER)" in prompt
        # The example output ontology must mention attributes so the LLM
        # knows to include the field.
        assert '"attributes"' in prompt
