"""Mapping declaration: what a structured source promises about itself.

A mapping is the whole contract between a table and the graph. Everything it
gets wrong is wrong silently later, in a graph that looks populated, so these
tests are mostly about rejection: the malformed declaration must fail at
construction, where the traceback still points at the user's own code.
"""

from __future__ import annotations

import pytest

from graphrag_sdk.ingestion.mapping import (
    RESERVED_PROPERTY_NAMES,
    Column,
    EdgeMapping,
    MappingError,
    NodeMapping,
    RecordMapping,
    Table,
)


class TestColumn:
    """A column declares a type, and casting is where the type is enforced."""

    @pytest.mark.parametrize(
        ("declared", "raw", "expected"),
        [
            ("STRING", "Acme", "Acme"),
            ("STRING", 42, "42"),
            ("INTEGER", "1200", 1200),
            ("INTEGER", " 1200 ", 1200),
            ("FLOAT", "1.5", 1.5),
            ("BOOLEAN", "yes", True),
            ("BOOLEAN", "0", False),
            ("DATE", "2019-04-01", "2019-04-01"),
            ("LIST", "a,b", ["a", "b"]),
        ],
    )
    def test_cast_converts_declared_types(self, declared, raw, expected):
        assert Column("c", declared).cast(raw) == expected

    def test_unknown_type_is_rejected_at_declaration(self):
        with pytest.raises(MappingError, match="unknown type"):
            Column("c", "TIMESTAMP")

    def test_cast_failure_names_the_column(self):
        """The error has to say which column, or a 40-column CSV is a hunt."""
        with pytest.raises(MappingError, match="'age'"):
            Column("age", "INTEGER").cast("thirty")

    def test_empty_value_is_absent_not_zero(self):
        """A blank cell is missing data. Coercing it to 0 invents a fact."""
        assert Column("age", "INTEGER").cast("") is None
        assert Column("age", "INTEGER").cast(None) is None


class TestNodeMapping:
    def test_reserved_property_names_are_rejected(self):
        """These are SDK-written keys. Mapping one shadows a system value."""
        for reserved in sorted(RESERVED_PROPERTY_NAMES):
            with pytest.raises(MappingError, match="written by the SDK"):
                NodeMapping(label="Person", key="id_col", properties={reserved: "c"})

    def test_name_is_reserved_because_it_has_its_own_slot(self):
        """``name`` is the display name, declared as ``name=``, not a property."""
        assert "name" in RESERVED_PROPERTY_NAMES
        with pytest.raises(MappingError):
            NodeMapping(label="Person", key="e", properties={"name": "full_name"})

    def test_alias_defaults_to_the_label(self):
        assert NodeMapping(label="Person", key="e").alias == "Person"

    def test_columns_reports_every_column_the_node_reads(self):
        node = NodeMapping(
            label="Person",
            key="employee_id",
            name="full_name",
            properties={"age": Column("age", "INTEGER")},
        )
        assert node.columns == {"employee_id", "full_name", "age"}

    def test_reference_node_carries_no_properties(self):
        """A foreign key points at an entity; it does not describe it."""
        with pytest.raises(MappingError, match="cannot declare"):
            NodeMapping(
                label="Organization",
                key="org_id",
                reference=True,
                properties={"hq_country": "hq"},
            )

    def test_reference_node_may_still_carry_a_name_column(self):
        """A denormalised name is identity, not description, so it is allowed:
        it lets the stub be called "Acme Corp" instead of "ORG-42"."""
        node = NodeMapping(label="Organization", key="org_id", reference=True, name="org_name")
        assert node.name == "org_name"


class TestRecordMapping:
    def test_edges_must_address_declared_nodes(self):
        with pytest.raises(MappingError, match="not a declared node alias"):
            RecordMapping(
                nodes=[NodeMapping(label="Person", key="e")],
                edges=[EdgeMapping(type="WORKS_AT", source="Person", target="Organization")],
            )

    def test_duplicate_aliases_are_rejected(self):
        """Two nodes under one alias makes every edge ambiguous."""
        with pytest.raises(MappingError):
            RecordMapping(
                nodes=[
                    NodeMapping(label="Person", key="a"),
                    NodeMapping(label="Person", key="b"),
                ]
            )

    def test_at_least_one_node_is_required(self):
        with pytest.raises(MappingError):
            RecordMapping(nodes=[])

    def test_anchor_is_the_first_non_reference_node(self):
        """The anchor identifies the record, so it can't be a foreign key."""
        mapping = RecordMapping(
            nodes=[
                NodeMapping(label="Organization", key="org_id", reference=True),
                NodeMapping(label="Person", key="employee_id", name="full_name"),
            ],
            edges=[EdgeMapping(type="WORKS_AT", source="Person", target="Organization")],
        )
        assert mapping.anchor.label == "Person"

    def test_a_mapping_of_only_references_is_rejected(self):
        with pytest.raises(MappingError):
            RecordMapping(nodes=[NodeMapping(label="Organization", key="org_id", reference=True)])


class TestValidateAgainstHeader:
    """Checked against the real header before anything is written."""

    def _mapping(self):
        return RecordMapping(
            nodes=[
                NodeMapping(
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    properties={"age": Column("age", "INTEGER")},
                )
            ]
        )

    def test_a_fitting_mapping_reports_nothing(self):
        problems = self._mapping().validate_against(["employee_id", "full_name", "age"])
        assert problems == []

    def test_missing_key_column_is_reported(self):
        problems = self._mapping().validate_against(["full_name", "age"])
        assert any("employee_id" in p for p in problems)

    def test_missing_property_column_is_reported(self):
        problems = self._mapping().validate_against(["employee_id", "full_name"])
        assert any("age" in p for p in problems)

    def test_unmapped_column_is_reported_only_in_strict_mode(self):
        header = ["employee_id", "full_name", "age", "salary"]
        assert self._mapping().validate_against(header) == []
        strict = self._mapping().validate_against(header, strict=True)
        assert any("salary" in p for p in strict)


class TestToOntology:
    """The projection that makes typed columns visible to text-to-Cypher."""

    def test_declared_types_reach_the_ontology(self):
        mapping = RecordMapping(
            nodes=[
                NodeMapping(
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    properties={
                        "age": Column("age", "INTEGER"),
                        "title": Column("job_title"),
                    },
                )
            ]
        )
        entity = mapping.to_ontology().entities[0]
        types = {p.name: p.type for p in entity.properties}
        assert types == {"employee_id": "STRING", "age": "INTEGER", "title": "STRING"}

    def test_name_is_never_declared_as_an_attribute(self):
        """Regression. ``GraphExtraction._entities_to_nodes`` merges
        ontology-declared attributes *over* the system properties it just built,
        so a declared ``name`` invites the extractor to answer it with a null and
        blank out the display name of every entity extracted from prose. The
        symptom is remote from the cause: CSVs ingest fine, then documents
        produce nameless nodes that later resolve into nothing.
        """
        mapping = Table("Organization", key="org_id", name="org_name")
        declared = {p.name for p in mapping.to_ontology().entities[0].properties}
        assert "name" not in declared
        assert "org_id" in declared

    def test_a_key_column_called_name_cannot_smuggle_it_back(self):
        mapping = Table("Organization", key="name")
        declared = {p.name for p in mapping.to_ontology().entities[0].properties}
        assert declared == set()

    def test_edge_patterns_are_declared_with_endpoint_labels(self):
        mapping = RecordMapping(
            nodes=[
                NodeMapping(label="Person", key="employee_id", name="full_name"),
                NodeMapping(label="Organization", key="org_id", reference=True),
            ],
            edges=[EdgeMapping(type="WORKS_AT", source="Person", target="Organization")],
        )
        relation = mapping.to_ontology().relations[0]
        assert relation.label == "WORKS_AT"
        assert list(relation.patterns) == [("Person", "Organization")]

    def test_reference_only_labels_still_get_an_entry(self):
        """An edge pointing at an undeclared label trips the validator."""
        mapping = RecordMapping(
            nodes=[
                NodeMapping(label="Person", key="employee_id", name="full_name"),
                NodeMapping(label="Organization", key="org_id", reference=True),
            ],
            edges=[EdgeMapping(type="WORKS_AT", source="Person", target="Organization")],
        )
        labels = {e.label for e in mapping.to_ontology().entities}
        assert labels == {"Person", "Organization"}


class TestTableShorthand:
    def test_table_is_one_node_keyed_and_named(self):
        mapping = Table(
            "Organization",
            key="org_id",
            name="org_name",
            hq_country="hq_country",
            employee_count=Column("employee_count", "INTEGER"),
        )
        node = mapping.anchor
        assert (node.label, node.key, node.name) == ("Organization", "org_id", "org_name")
        assert node.properties["employee_count"].type == "INTEGER"
        assert node.properties["hq_country"].type == "STRING"
