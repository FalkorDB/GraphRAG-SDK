"""Tests for entity utility functions in entity_extractors.py."""

from __future__ import annotations


from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
    DEFAULT_ENTITY_TYPES,
    UNKNOWN_LABEL,
    compute_entity_id,
    is_valid_entity_name,
    label_for_type,
)


class TestComputeEntityId:
    def test_lowercase_strip(self):
        assert compute_entity_id("  Alice  ") == "alice"

    def test_spaces_to_underscores(self):
        assert compute_entity_id("Acme Corp") == "acme_corp"

    def test_already_normalised(self):
        assert compute_entity_id("bob") == "bob"

    def test_type_qualified_id(self):
        assert compute_entity_id("Paris", "Location") == "paris__location"
        assert compute_entity_id("Paris", "Person") == "paris__person"

    def test_cross_type_collision_prevented(self):
        id_person = compute_entity_id("Paris", "Person")
        id_location = compute_entity_id("Paris", "Location")
        assert id_person != id_location

    def test_no_type_backwards_compatible(self):
        assert compute_entity_id("Alice") == "alice"
        assert compute_entity_id("Alice", "") == "alice"

    def test_empty_name(self):
        assert compute_entity_id("") == ""

    def test_type_with_spaces(self):
        assert compute_entity_id("Paris", " Work of Art ") == "paris__work of art"


class TestIsValidEntityName:
    def test_valid_names(self):
        assert is_valid_entity_name("Alice")
        assert is_valid_entity_name("Acme Corp")
        assert is_valid_entity_name("The Great Gatsby")

    def test_empty_rejected(self):
        assert not is_valid_entity_name("")
        assert not is_valid_entity_name("   ")

    def test_too_short_rejected(self):
        assert not is_valid_entity_name("A")

    def test_too_long_rejected(self):
        assert not is_valid_entity_name("x" * 81)

    def test_pronouns_rejected(self):
        assert not is_valid_entity_name("he")
        assert not is_valid_entity_name("She")
        assert not is_valid_entity_name("THEY")

    def test_generic_references_rejected(self):
        assert not is_valid_entity_name("narrator")
        assert not is_valid_entity_name("the narrator")
        assert not is_valid_entity_name("author")

    def test_metatextual_rejected(self):
        assert not is_valid_entity_name("story")
        assert not is_valid_entity_name("chapter")
        assert not is_valid_entity_name("book")


class TestLabelForType:
    def test_exact_match(self):
        assert label_for_type("Person", DEFAULT_ENTITY_TYPES) == "Person"

    def test_case_insensitive(self):
        assert label_for_type("person", DEFAULT_ENTITY_TYPES) == "Person"

    def test_normalized_match(self):
        assert label_for_type("data_set", DEFAULT_ENTITY_TYPES) == "Dataset"

    def test_unknown_fallback(self):
        assert label_for_type("FooBarBaz", DEFAULT_ENTITY_TYPES) == UNKNOWN_LABEL

    def test_empty_string(self):
        assert label_for_type("", DEFAULT_ENTITY_TYPES) == UNKNOWN_LABEL

    def test_custom_types(self):
        custom = ["Vehicle", "Animal"]
        assert label_for_type("vehicle", custom) == "Vehicle"
        assert label_for_type("Person", custom) == UNKNOWN_LABEL


class TestDefaultEntityTypes:
    def test_has_common_types(self):
        assert "Person" in DEFAULT_ENTITY_TYPES
        assert "Organization" in DEFAULT_ENTITY_TYPES
        assert "Location" in DEFAULT_ENTITY_TYPES
        assert "Event" in DEFAULT_ENTITY_TYPES
        assert "Technology" in DEFAULT_ENTITY_TYPES
        assert "Product" in DEFAULT_ENTITY_TYPES
        assert "Date" in DEFAULT_ENTITY_TYPES
        assert "Concept" in DEFAULT_ENTITY_TYPES
        assert "Law" in DEFAULT_ENTITY_TYPES
        assert "Dataset" in DEFAULT_ENTITY_TYPES
        assert "Method" in DEFAULT_ENTITY_TYPES

    def test_no_duplicates(self):
        assert len(DEFAULT_ENTITY_TYPES) == len(set(DEFAULT_ENTITY_TYPES))
