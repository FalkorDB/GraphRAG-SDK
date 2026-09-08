"""Mapping declaration: what a structured source promises about itself.

A mapping is the whole contract between a table and the graph. Everything it
gets wrong is wrong silently later, in a graph that looks populated, so these
tests are mostly about rejection: the malformed declaration must fail while it
is being declared or normalised, where the traceback still points at the user's
own code.

A user writes a :class:`TableMapping` and hands it to the ontology;
``record_mapping_for()`` is the translation into the nodes-and-edges form the
write path consumes, and it is where a link's own faults surface.
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
    TableMapping,
    record_mapping_for,
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
        mapping = TableMapping(
            source="orgs.csv", label="Organization", key="org_id", name="org_name"
        )
        entity = record_mapping_for(mapping).to_ontology().entities[0]
        declared = {p.name for p in entity.properties}
        assert "name" not in declared
        # The key is still published, under the name the write path gives it:
        # a declared mapping signs what it writes with its source.
        assert mapping.signed_name("org_id") in declared

    def test_a_key_column_called_name_cannot_smuggle_it_back(self):
        """`name` must not become a declared attribute, whatever the header is.

        Declaring it lets the extractor answer it with a null for prose mentions
        and blank out the display name of everything it extracts. It is still
        never declared — but the key is no longer dropped to achieve that: a
        header the SDK owns is published under its signed ``col_`` name, so the
        column the user declared as identity stays queryable instead of
        vanishing from the ontology without a word.
        """
        mapping = TableMapping(source="orgs.csv", label="Organization", key="name")
        record = record_mapping_for(mapping)
        declared = {p.name for p in record.to_ontology().entities[0].properties}
        assert "name" not in declared
        assert declared == {mapping.signed_name("col_name")}
        assert record.nodes[0].key_property == "col_name"

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


class TestTableMappingShorthand:
    def test_table_is_one_node_keyed_and_named(self):
        mapping = TableMapping(
            source="orgs.csv",
            label="Organization",
            key="org_id",
            name="org_name",
            properties={
                "hq_country": "hq_country",
                "employee_count": Column("employee_count", "INTEGER"),
            },
        )
        node = record_mapping_for(mapping).anchor
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

    @pytest.mark.parametrize(
        "raw", ["2024-01-05", "2024-01-05T10:00:00", "2024-01-05T10:00:00Z", "2024-01-05 10:00Z"]
    )
    def test_a_date_accepts_the_shapes_exports_write(self, raw):
        """Python 3.10's parser rejects the ``Z`` suffix most API exports use."""
        assert Column("signed_on", "DATE").cast(raw) == "2024-01-05"


class TestTableMappingIsTheOnlyFormYouWrite:
    """One declaration that grows, instead of two that you switch between.

    A table with a foreign key used to require rewriting the whole declaration
    into a different shape. Adding a link is now an argument, not a rewrite.

    The declaration is what the user writes; ``record_mapping_for()`` is the
    normalised nodes-and-edges form the write path reads, so that is what these
    tests inspect.
    """

    def test_a_link_adds_a_reference_and_an_edge(self):
        table = record_mapping_for(
            TableMapping(
                source="hr.csv",
                label="Person",
                key="employee_id",
                name="full_name",
                properties={"age": Column("age", "INTEGER")},
                links=[Link("WORKS_AT", to="Organization", by="org_id")],
            )
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
        table = record_mapping_for(
            TableMapping(
                source="orgs.csv",
                label="Organization",
                key="org_id",
                name="org_name",
                properties={"hq": "hq_country"},
            )
        )
        assert len(table.nodes) == 1
        assert table.edges == []

    def test_a_link_may_name_its_target(self):
        """A denormalised name makes the placeholder "Acme Corp" not "ORG-42"."""
        table = record_mapping_for(
            TableMapping(
                source="hr.csv",
                label="Person",
                key="e",
                links=[Link("WORKS_AT", to="Organization", by="org_id", name="org_name")],
            )
        )
        assert table.nodes[1].name == "org_name"

    def test_link_properties_land_on_the_edge(self):
        table = record_mapping_for(
            TableMapping(
                source="hr.csv",
                label="Person",
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
        )
        assert table.edges[0].typed_properties["since"].type == "DATE"

    def test_the_ontology_declares_the_relationship_pattern(self):
        table = record_mapping_for(
            TableMapping(
                source="hr.csv",
                label="Person",
                key="e",
                links=[Link("WORKS_AT", to="Organization", by="org_id")],
            )
        )
        relation = table.to_ontology().relations[0]
        assert relation.label == "WORKS_AT"
        assert list(relation.patterns) == [("Person", "Organization")]

    def test_two_links_to_the_same_label_get_distinct_handles(self):
        table = record_mapping_for(
            TableMapping(
                source="contracts.csv",
                label="Contract",
                key="contract_id",
                links=[
                    Link("BUYER", to="Organization", by="buyer_id"),
                    Link("SELLER", to="Organization", by="seller_id"),
                ],
            )
        )
        assert len({n.handle for n in table.nodes}) == 3, "handles must stay unique"
        assert {e.target for e in table.edges} == {n.handle for n in table.nodes[1:]}

    def test_a_link_on_the_records_own_key_is_rejected(self):
        """It would link the record to itself, which says nothing."""
        with pytest.raises(MappingError, match="own key"):
            record_mapping_for(
                TableMapping(
                    source="hr.csv",
                    label="Person",
                    key="employee_id",
                    links=[Link("KNOWS", to="Person", by="employee_id")],
                )
            )

    def test_links_must_be_link_objects(self):
        with pytest.raises(MappingError, match="must contain Link objects"):
            record_mapping_for(
                TableMapping(
                    source="hr.csv",
                    label="Person",
                    key="e",
                    links=[("WORKS_AT", "Organization", "org_id")],  # type: ignore[list-item]
                )
            )

    def test_a_link_validates_its_names(self):
        with pytest.raises(MappingError, match="not a usable name"):
            Link("WORKS AT", to="Organization", by="org_id")
        with pytest.raises(MappingError, match="cannot be written as a label"):
            Link("WORKS_AT", to="Org`) DELETE (n) //", by="org_id")

    def test_columns_include_every_column_a_link_reads(self):
        table = TableMapping(
            source="hr.csv",
            label="Person",
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
        expected = {"employee_id", "full_name", "org_id", "org_name", "start_date"}
        assert table.columns == expected
        # The declaration and the form the write path reads must agree: the
        # header check runs against one and the rows are read through the other.
        assert record_mapping_for(table).columns == expected

    def test_validation_against_a_header_covers_link_columns(self):
        table = record_mapping_for(
            TableMapping(
                source="hr.csv",
                label="Person",
                key="employee_id",
                links=[Link("WORKS_AT", to="Organization", by="org_id")],
            )
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
        table = record_mapping_for(
            TableMapping(
                source="contracts.csv",
                label="Contract",
                key="contract_id",
                links=[
                    Link("PARTY_TO", to="Organization", by="a_id"),
                    Link("PARTY_TO", to="Organization", by="b_id"),
                    Link("PARTY_TO", to="Organization", by="c_id"),
                ],
            )
        )
        assert [n.key for n in table.nodes] == ["contract_id", "a_id", "b_id", "c_id"]
        assert len({n.handle for n in table.nodes}) == 4
        assert len(table.edges) == 3

    def test_two_links_naming_the_same_target_twice_are_refused(self):
        """Refused where the declaration is written, not at the first ingest,
        so it cannot be saved into an ontology first."""
        with pytest.raises(MappingError, match="same target twice"):
            TableMapping(
                source="contracts.csv",
                label="Contract",
                key="contract_id",
                links=[
                    Link("A", to="Organization", by="a_id"),
                    Link("B", to="Organization", by="org_id"),
                    Link("C", to="Organization", by="org_id"),
                ],
            )

    def test_a_link_by_the_records_own_key_is_refused_at_declaration(self):
        with pytest.raises(MappingError, match="link the record to itself"):
            TableMapping(
                source="p.csv",
                label="Person",
                key="id_",
                links=[Link("KNOWS", to="Person", by="id_")],
            )

    def test_a_links_edge_properties_are_checked_at_declaration(self):
        """``TableMapping.properties`` were validated eagerly and a link's were
        not, so a reserved or unusable edge property survived until ingest."""
        with pytest.raises(MappingError, match="written by the SDK"):
            Link("WORKS_AT", to="Organization", by="org_id", properties={"id": "x"})
        with pytest.raises(MappingError, match="not a usable name"):
            Link("WORKS_AT", to="Organization", by="org_id", properties={"bad name": "y"})
        link = Link("WORKS_AT", to="Organization", by="org_id", properties={"since": "since"})
        assert link.properties == {"since": Column("since")}

    def test_an_edge_property_changes_the_declaration_fingerprint(self):
        """``finalize().mapping_changed`` reads this digest; retyping an edge
        property is a re-declaration the same as retyping a node property."""
        plain = Link("WORKS_AT", to="Organization", by="org_id")
        dated = Link(
            "WORKS_AT",
            to="Organization",
            by="org_id",
            properties={"since": Column("since", "DATE")},
        )
        a = TableMapping(source="p.csv", label="P", key="k", links=[plain])
        b = TableMapping(source="p.csv", label="P", key="k", links=[dated])
        assert a.fingerprint_of_declaration != b.fingerprint_of_declaration

    def test_a_link_may_point_at_the_records_own_label(self):
        """A manager is a Person too. Only the record's own *key* is refused."""
        table = record_mapping_for(
            TableMapping(
                source="hr.csv",
                label="Person",
                key="employee_id",
                links=[Link("MANAGES", to="Person", by="manager_id")],
            )
        )
        assert [n.key for n in table.nodes] == ["employee_id", "manager_id"]

    def test_a_property_cannot_be_called_name(self):
        """``name`` can never reach the property map: it binds to the
        display-name parameter, and the property map refuses the word outright.
        """
        table = TableMapping(source="orgs.csv", label="Organization", key="org_id", name="org_name")
        assert record_mapping_for(table).anchor.name == "org_name"
        assert record_mapping_for(table).anchor.properties == {}
        with pytest.raises(MappingError, match="written by the SDK"):
            TableMapping(
                source="orgs.csv",
                label="Organization",
                key="org_id",
                properties={"name": "org_name"},
            )

    def test_link_order_does_not_change_the_fingerprint(self):
        """Reordering a declaration is not a change, so it must not force a
        re-sync of an unchanged source."""
        a = TableMapping(
            source="p.csv",
            label="P",
            key="k",
            links=[Link("A", to="X", by="x"), Link("B", to="Y", by="y")],
        )
        b = TableMapping(
            source="p.csv",
            label="P",
            key="k",
            links=[Link("B", to="Y", by="y"), Link("A", to="X", by="x")],
        )
        # The digest ``update()``'s no-op short circuit folds in, ...
        assert record_mapping_for(a).fingerprint == record_mapping_for(b).fingerprint
        # ... and the one that tells a re-declaration from the same declaration.
        assert a.fingerprint_of_declaration == b.fingerprint_of_declaration
