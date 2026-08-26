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
    Link,
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


class TestNamesMustBeUsable:
    """A declaration that cannot be written to a graph is rejected at the point
    it is written, not deep inside the driver.

    Measured before this: a property named with a backtick and a comment marker
    surfaced as ``DatabaseError: Invalid input at end of input`` from a query the
    caller never wrote. No injection occurred, but nothing pointed at the cause.
    """

    HOSTILE = "x`) DETACH DELETE (n) //"

    def test_a_property_name_must_be_an_identifier(self):
        with pytest.raises(MappingError, match="not a usable name"):
            NodeMapping(label="Org", key="k", properties={self.HOSTILE: "c"})

    def test_a_property_name_with_a_space_is_rejected(self):
        """Generated Cypher writes these as ``n.name``, so a name needing quotes
        is not usable even though a graph would store it."""
        with pytest.raises(MappingError, match="not a usable name"):
            NodeMapping(label="Org", key="k", properties={"hq country": "c"})

    def test_an_awkward_column_gets_a_usable_property_name(self):
        """The escape: name the property, point it at the column."""
        node = NodeMapping(label="Org", key="k", properties={"hq_country": Column("HQ Country")})
        assert node.typed_properties["hq_country"].name == "HQ Country"

    def test_a_relationship_type_must_be_an_identifier(self):
        with pytest.raises(MappingError, match="not a usable name"):
            EdgeMapping(type="WORKS AT", source="a", target="b")

    def test_a_label_that_would_be_silently_rewritten_is_rejected(self):
        """Labels are quoted on write, so spaces and unicode are fine. What is
        not fine is a label the sanitiser has to change, because the graph would
        then hold a different name than the one declared."""
        with pytest.raises(MappingError, match="cannot be written as a label"):
            NodeMapping(label="Org`) DETACH DELETE (n) //", key="k")

    @pytest.mark.parametrize("label", ["Person", "Legal Entity", "Org-Unit", "Ünïcode"])
    def test_a_label_a_graph_can_hold_is_accepted(self, label):
        assert NodeMapping(label=label, key="k").label == label


class TestCastingRefusesValuesThatPoisonQueries:
    @pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "NaN", "Infinity"])
    def test_non_finite_floats_are_rejected(self, raw):
        """One NaN turns an ``avg()`` over the whole column into NaN, with
        nothing in the result to point at the row that caused it."""
        with pytest.raises(MappingError):
            Column("score", "FLOAT").cast(raw)

    def test_a_quoted_list_element_may_contain_a_comma(self):
        """Splitting on commas turns one value into several, silently."""
        assert Column("tags", "LIST").cast('"a,b",c') == ["a,b", "c"]

    def test_an_ordinary_list_still_splits(self):
        assert Column("tags", "LIST").cast("a,b, c") == ["a", "b", "c"]


class TestTableIsTheOnlyFormYouWrite:
    """One declaration that grows, instead of two that you switch between.

    A table with a foreign key used to require rewriting the whole declaration
    into a different shape. Adding a link is now an argument, not a rewrite.
    """

    def test_a_link_adds_a_reference_and_an_edge(self):
        table = Table(
            "Person",
            key="employee_id",
            name="full_name",
            age=Column("age", "INTEGER"),
            links=[Link("WORKS_AT", to="Organization", by="org_id")],
        )
        assert [n.label for n in table.nodes] == ["Person", "Organization"]
        assert table.anchor.label == "Person", "the subject stays the record's own entity"
        target = table.nodes[1]
        assert (target.key, target.reference) == ("org_id", True), (
            "a link points at something without claiming to describe it"
        )
        assert [(e.type, e.source, e.target) for e in table.edges] == [
            ("WORKS_AT", "Person", "Organization")
        ]

    def test_a_table_without_links_is_unchanged(self):
        table = Table("Organization", key="org_id", name="org_name", hq="hq_country")
        assert len(table.nodes) == 1
        assert table.edges == []

    def test_a_link_may_name_its_target(self):
        """A denormalised name makes the placeholder "Acme Corp" not "ORG-42"."""
        table = Table(
            "Person",
            key="e",
            links=[Link("WORKS_AT", to="Organization", by="org_id", name="org_name")],
        )
        assert table.nodes[1].name == "org_name"

    def test_link_properties_land_on_the_edge(self):
        table = Table(
            "Person",
            key="e",
            links=[
                Link(
                    "WORKS_AT",
                    to="Organization",
                    by="org_id",
                    properties={"since": Column("start_date", "DATE")},
                )
            ],
        )
        assert table.edges[0].typed_properties["since"].type == "DATE"

    def test_the_ontology_declares_the_relationship_pattern(self):
        table = Table("Person", key="e", links=[Link("WORKS_AT", to="Organization", by="org_id")])
        relation = table.to_ontology().relations[0]
        assert relation.label == "WORKS_AT"
        assert list(relation.patterns) == [("Person", "Organization")]

    def test_two_links_to_the_same_label_get_distinct_handles(self):
        table = Table(
            "Contract",
            key="contract_id",
            links=[
                Link("BUYER", to="Organization", by="buyer_id"),
                Link("SELLER", to="Organization", by="seller_id"),
            ],
        )
        assert len({n.handle for n in table.nodes}) == 3, "handles must stay unique"
        assert {e.target for e in table.edges} == {n.handle for n in table.nodes[1:]}

    def test_a_link_on_the_records_own_key_is_rejected(self):
        """It would link the record to itself, which says nothing."""
        with pytest.raises(MappingError, match="own key"):
            Table("Person", key="employee_id", links=[Link("KNOWS", to="Person", by="employee_id")])

    def test_links_must_be_link_objects(self):
        with pytest.raises(MappingError, match="must contain Link objects"):
            Table("Person", key="e", links=[("WORKS_AT", "Organization", "org_id")])  # type: ignore[list-item]

    def test_a_link_validates_its_names(self):
        with pytest.raises(MappingError, match="not a usable name"):
            Link("WORKS AT", to="Organization", by="org_id")
        with pytest.raises(MappingError, match="cannot be written as a label"):
            Link("WORKS_AT", to="Org`) DELETE (n) //", by="org_id")

    def test_columns_include_every_column_a_link_reads(self):
        table = Table(
            "Person",
            key="employee_id",
            name="full_name",
            links=[
                Link(
                    "WORKS_AT",
                    to="Organization",
                    by="org_id",
                    name="org_name",
                    properties={"since": Column("start_date", "DATE")},
                )
            ],
        )
        assert table.columns == {"employee_id", "full_name", "org_id", "org_name", "start_date"}

    def test_validation_against_a_header_covers_link_columns(self):
        table = Table(
            "Person", key="employee_id", links=[Link("WORKS_AT", to="Organization", by="org_id")]
        )
        assert table.validate_against(["employee_id", "org_id"]) == []
        problems = table.validate_against(["employee_id"])
        assert any("org_id" in p for p in problems)

    def test_three_links_to_one_label_are_all_kept(self):
        """Handles are derived, so a third link must not collide with a second.

        Disambiguating by relationship type collided as soon as two links shared
        one, and the failure surfaced as a complaint about "duplicate aliases" —
        a word the caller never wrote.
        """
        table = Table(
            "Contract",
            key="contract_id",
            links=[
                Link("PARTY_TO", to="Organization", by="a_id"),
                Link("PARTY_TO", to="Organization", by="b_id"),
                Link("PARTY_TO", to="Organization", by="c_id"),
            ],
        )
        assert [n.key for n in table.nodes] == ["contract_id", "a_id", "b_id", "c_id"]
        assert len({n.handle for n in table.nodes}) == 4
        assert len(table.edges) == 3

    def test_two_links_naming_the_same_target_twice_are_refused(self):
        with pytest.raises(MappingError, match="same target twice"):
            Table(
                "Contract",
                key="contract_id",
                links=[
                    Link("A", to="Organization", by="a_id"),
                    Link("B", to="Organization", by="org_id"),
                    Link("C", to="Organization", by="org_id"),
                ],
            )

    def test_a_link_may_point_at_the_records_own_label(self):
        """A manager is a Person too. Only the record's own *key* is refused."""
        table = Table(
            "Person", key="employee_id", links=[Link("MANAGES", to="Person", by="manager_id")]
        )
        assert [n.key for n in table.nodes] == ["employee_id", "manager_id"]

    def test_a_property_cannot_be_called_name(self):
        """Structurally impossible: ``name`` binds to the display-name parameter,
        so it can never reach the property map."""
        table = Table("Organization", key="org_id", **{"name": "org_name"})
        assert table.anchor.name == "org_name"
        assert table.anchor.properties == {}

    def test_link_order_does_not_change_the_fingerprint(self):
        """Reordering a declaration is not a change, so it must not force a
        re-sync of an unchanged source."""
        a = Table("P", key="k", links=[Link("A", to="X", by="x"), Link("B", to="Y", by="y")])
        b = Table("P", key="k", links=[Link("B", to="Y", by="y"), Link("A", to="X", by="x")])
        assert a.fingerprint == b.fingerprint
