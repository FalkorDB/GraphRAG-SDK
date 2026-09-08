"""``Ontology.tables`` — the mapping is part of the schema, so it must survive.

A field added to ``Ontology`` is dropped by anything that rebuilds the model by
listing its fields. That already happened in this codebase once: ``GraphData``
rebuilt field-by-field lost ``mentions``, and every extracted entity lost its
provenance with it. These tests walk a mapping through every path that returns an
``Ontology`` so the same class of bug cannot come back quietly.

The store tests require ``RUN_INTEGRATION=1``.
"""

from __future__ import annotations

import pytest

from graphrag_sdk import Column, Entity, Link, Ontology
from graphrag_sdk.core.tables import TableMapping, signature_for
from graphrag_sdk.discovery.pipeline import _ensure_sdk_managed_attributes

HR = TableMapping(
    source="hr.csv",
    label="Person",
    key="employee_id",
    name="full_name",
    properties={"age": Column("age", "INTEGER"), "title": Column("job_title")},
    links=[Link("WORKS_AT", to="Organization", by="org_id", name="org_name")],
)


def _ontology() -> Ontology:
    return Ontology(
        entities=[Entity(label="Person"), Entity(label="Organization")],
        tables=[HR],
    )


class TestTheMappingSurvivesEveryRebuild:
    def test_merge(self):
        assert _ontology().merge(Ontology()).tables == [HR]

    def test_merge_from_the_other_side(self):
        assert Ontology().merge(_ontology()).tables == [HR]

    def test_merge_does_not_duplicate_one_source(self):
        merged = _ontology().merge(_ontology())
        assert [mapping.source for mapping in merged.tables] == ["hr.csv"]

    def test_sdk_managed_attributes(self):
        """Takes an ontology and returns a modified one — the dangerous shape."""
        assert _ensure_sdk_managed_attributes(_ontology()).tables == [HR]

    def test_json_round_trip(self):
        back = Ontology.model_validate_json(_ontology().model_dump_json())
        assert len(back.tables) == 1
        # the declaration form, not the normalised one: record_mapping_for()
        # flattens `links` away into nodes and edges, so a mapping held in the
        # normalised form could never round-trip them.
        assert [link.type for link in back.tables[0].links] == ["WORKS_AT"]
        assert back.tables[0].typed_properties["age"].type == "INTEGER"

    def test_model_copy(self):
        assert _ontology().model_copy().tables == [HR]


class TestEveryFieldIsPersisted:
    """A field added to ``TableMapping`` and not to the store round-trips as its
    default, silently. That is how ``derived`` shipped broken for one commit: the
    field existed, ``finalize()`` read it, and the store never wrote it.
    """

    def test_the_store_writes_and_reads_every_declared_field(self):
        import dataclasses

        from graphrag_sdk.storage import ontology_store

        source = ontology_store.__file__
        text = open(source).read()
        skip = {"properties", "links"}  # their own nodes, not scalar properties
        missing = [
            field.name
            for field in dataclasses.fields(TableMapping)
            if field.name not in skip and f"t.{field.name}" not in text
        ]
        assert not missing, (
            f"TableMapping field(s) {missing} are never written to or read from the "
            f"ontology graph, so they will round-trip as their default"
        )


class TestSignatureCollisionsAreRefused:
    """Every property a source writes is ``<signature>__<property>``.

    Two sources reducing to one signature share a property namespace and would
    overwrite each other. ``signature_for`` is not injective, so the collision is
    caught where every mapping is visible at once.
    """

    def test_two_names_reducing_alike(self):
        with pytest.raises(ValueError, match="one property signature"):
            Ontology(
                entities=[Entity(label="Person")],
                tables=[
                    TableMapping(source="hr.csv", label="Person", key="k"),
                    TableMapping(source="HR.CSV", label="Person", key="k"),
                ],
            )

    def test_one_source_declared_twice(self):
        with pytest.raises(ValueError, match="declared twice"):
            Ontology(
                entities=[Entity(label="Person")],
                tables=[
                    TableMapping(source="hr.csv", label="Person", key="a"),
                    TableMapping(source="hr.csv", label="Person", key="b"),
                ],
            )

    def test_distinct_signatures_are_fine(self):
        ontology = Ontology(
            entities=[Entity(label="Person")],
            tables=[
                TableMapping(source="hr.csv", label="Person", key="k"),
                TableMapping(source="finance.csv", label="Person", key="k"),
            ],
        )
        assert [mapping.signature for mapping in ontology.tables] == ["hr", "finance"]

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("hr.csv", "hr"),
            ("/data/2026-08/hr.csv", "hr"),
            ("./hr.csv", "hr"),
            ("hr-final(1).csv", "hr_final_1"),
            ("2024_hr.csv", "s_2024_hr"),
        ],
    )
    def test_the_signature_ignores_the_directory_and_one_extension(self, source, expected):
        assert signature_for(source) == expected


class TestANewLabelMustConnect:
    def test_a_label_a_sibling_link_points_at_is_connected(self):
        """The table owning ``Organization`` is reached through every
        ``WORKS_AT`` that keys it; that is how a placeholder gets filled."""
        ontology = Ontology(
            tables=[
                TableMapping(
                    source="hr.csv",
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    links=[Link("WORKS_AT", to="Organization", by="org_id")],
                ),
                TableMapping(source="orgs.csv", label="Organization", name="org_name"),
            ]
        )
        assert {m.label for m in ontology.tables} == {"Person", "Organization"}

    def test_a_label_nothing_reaches_is_refused(self):
        with pytest.raises(ValueError, match="does not say how it connects"):
            Ontology(tables=[TableMapping(source="orgs.csv", label="Organization", name="n")])


class TestStandalone:
    def test_standalone_with_links_is_refused(self):
        from graphrag_sdk.ingestion.mapping import MappingError

        with pytest.raises(MappingError, match="standalone"):
            TableMapping(
                source="x.csv",
                label="Thing",
                key="k",
                standalone=True,
                links=[Link("R", to="Other", by="b")],
            )


class TestTheStoreRoundTrip:
    """Requires RUN_INTEGRATION=1."""

    @pytest.fixture
    async def store(self, real_falkordb_rag_factory, llm):
        from graphrag_sdk import ExactMatchResolution

        rag = real_falkordb_rag_factory(
            llm=llm, resolver=ExactMatchResolution(resolve_property="name")
        )
        yield rag._ontology_store
        await rag.close()

    async def test_a_mapping_round_trips_through_the_graph(self, store):
        await store.register(_ontology())
        back = await store.load()

        assert [mapping.source for mapping in back.tables] == ["hr.csv"]
        stored = back.tables[0]
        assert stored.label == "Person"
        assert stored.key == "employee_id"
        assert stored.name == "full_name"
        assert stored.typed_properties["age"].type == "INTEGER"
        assert stored.typed_properties["title"].name == "job_title"
        assert [(link.type, link.to, link.by, link.name) for link in stored.links] == [
            ("WORKS_AT", "Organization", "org_id", "org_name")
        ]

    async def test_a_links_declared_columns_survive_the_round_trip(self, store):
        """The store wrote a link's type/to/by/name and nothing else.

        So ``Link(properties=...)`` came back empty on the next session: every
        new row silently stopped carrying the edge column, while the reloaded
        ontology still advertised it to text-to-Cypher.
        """
        mapping = TableMapping(
            source="hr.csv",
            label="Person",
            key="employee_id",
            name="full_name",
            links=[
                Link(
                    "WORKS_AT",
                    to="Organization",
                    by="org_id",
                    properties={"since": Column("start_date", "DATE")},
                )
            ],
        )
        await store.register(
            Ontology(
                entities=[Entity(label="Person"), Entity(label="Organization")], tables=[mapping]
            )
        )
        back = (await store.load()).tables[0]

        assert back.links[0].typed_properties["since"].name == "start_date"
        assert back.links[0].typed_properties["since"].type == "DATE"
        # The declaration is what round-trips, so the drift check must agree too.
        assert back.fingerprint_of_declaration == mapping.fingerprint_of_declaration

    async def test_a_link_column_dropped_from_a_redeclaration_does_not_linger(self, store):
        with_col = TableMapping(
            source="hr.csv",
            label="Person",
            key="employee_id",
            links=[
                Link(
                    "WORKS_AT",
                    to="Organization",
                    by="org_id",
                    properties={"since": Column("start_date", "DATE")},
                )
            ],
        )
        without = TableMapping(
            source="hr.csv",
            label="Person",
            key="employee_id",
            links=[Link("WORKS_AT", to="Organization", by="org_id")],
        )
        onto = lambda m: Ontology(  # noqa: E731
            entities=[Entity(label="Person"), Entity(label="Organization")], tables=[m]
        )
        await store.register(onto(with_col))
        await store.register(onto(without))
        assert (await store.load()).tables[0].links[0].properties == {}

    async def test_a_signature_already_claimed_by_a_stored_source_is_refused(self, store):
        """Two monthly exports, both named hr.csv, registered in separate calls.

        The pydantic validator only sees one Ontology at a time, so these never
        met and the second was written. Every later ``load()`` then built an
        Ontology holding both and raised — a graph no new GraphRAG could open.
        Refused before the first write instead, so the graph stays readable.
        """
        from graphrag_sdk.storage.ontology_store import OntologyContradictionError

        def monthly(month):
            return TableMapping(
                source=f"/exports/2026-{month}/hr.csv",
                label="Person",
                key="employee_id",
                name="full_name",
                properties={"age": Column("age", "INTEGER")},
            )

        await store.register(Ontology(entities=[Entity(label="Person")], tables=[monthly("01")]))
        with pytest.raises(OntologyContradictionError, match="2026-02"):
            await store.register(
                Ontology(entities=[Entity(label="Person")], tables=[monthly("02")])
            )

        # The graph must still open — that is the part that made this worse than
        # the overwrite it was guarding against.
        assert [m.source for m in (await store.load()).tables] == ["/exports/2026-01/hr.csv"]

    async def test_a_tables_only_ontology_is_not_treated_as_empty(self, store):
        """register() used to return early when entities and relations were
        empty, discarding every mapping it had just been handed."""
        result = await store.register(Ontology(tables=[HR]))
        assert [mapping.source for mapping in result.tables] == ["hr.csv"]

    async def test_re_declaring_replaces_the_children(self, store):
        """A mapping is authoritative about its own shape.

        A re-declaration that drops a column must drop it from the store too, or
        the stored mapping and the user's code drift and the diff that reports a
        change can never come out clean.
        """
        await store.register(_ontology())
        await store.register(
            Ontology(
                entities=[Entity(label="Person")],
                tables=[
                    TableMapping(
                        source="hr.csv",
                        label="Person",
                        key="employee_id",
                        name="full_name",
                        properties={"age": Column("age", "INTEGER")},
                    )
                ],
            )
        )
        stored = next(m for m in (await store.load()).tables if m.source == "hr.csv")
        assert sorted(stored.typed_properties) == ["age"]
        assert stored.links == []

    async def test_other_mappings_are_untouched_by_a_re_declaration(self, store):
        finance = TableMapping(
            source="finance.csv",
            label="Person",
            key="employee_id",
            properties={"grade": Column("grade")},
        )
        await store.register(
            Ontology(
                entities=[Entity(label="Person"), Entity(label="Organization")],
                tables=[HR, finance],
            )
        )
        await store.register(Ontology(tables=[HR]))
        sources = sorted(m.source for m in (await store.load()).tables)
        assert sources == ["finance.csv", "hr.csv"]


class TestATableQueryFailureCannotTakeTheOntologyDown:
    """The table queries have their own try/except for a stated reason.

    The five entity/relation queries share one handler that returns a bare
    ``Ontology()`` and logs at DEBUG, so a table query placed up there would
    silently discard every entity and relation. The separate handler logs at
    WARNING and keeps them — but only if every row list it hands on is bound.
    ``MAPS_LINK_COLUMN`` was assigned on the success path only, so a failure raised
    ``UnboundLocalError`` out of ``load()`` and took down exactly what the second
    handler exists to protect.

    Requires RUN_INTEGRATION=1.
    """

    @pytest.fixture
    async def store(self, real_falkordb_rag_factory, llm):
        from graphrag_sdk import ExactMatchResolution

        rag = real_falkordb_rag_factory(
            llm=llm, resolver=ExactMatchResolution(resolve_property="name")
        )
        yield rag._ontology_store
        await rag.close()

    async def test_the_entities_survive_and_the_failure_is_reported(self, store, caplog):
        import logging

        await store.register(_ontology())
        real_query = store._query

        async def failing(cypher, *args, **kwargs):
            if "MappedLinkColumn" in cypher:
                raise RuntimeError("simulated table-query failure")
            return await real_query(cypher, *args, **kwargs)

        store._query = failing  # type: ignore[method-assign]
        try:
            with caplog.at_level(logging.WARNING):
                loaded = await store.load()
        finally:
            store._query = real_query  # type: ignore[method-assign]

        assert [e.label for e in loaded.entities], "entities were discarded by a table failure"
        assert loaded.tables == [], "a failed table load must yield no tables, not raise"
        assert any("Table mappings could not be loaded" in m for m in caplog.messages)


class TestTheKeyDefaultsToTheName:
    """Most tables have one column that is both identifier and display name.

    Requiring ``key=`` as well was pure friction for them. Left out, the key is
    the name column, and the node id is then exactly what prose extraction would
    compute for that name — so a row and a document mention share an id outright
    and nothing has to be merged.

    Requires RUN_INTEGRATION=1 for the graph test.
    """

    def test_key_falls_back_to_name(self):
        m = TableMapping(source="orgs.csv", label="Organization", name="org_name")
        assert m.key == "org_name"
        assert m.key_column == "org_name"

    def test_neither_is_refused_with_the_fix_named(self):
        from graphrag_sdk.ingestion.mapping import MappingError

        with pytest.raises(MappingError, match="name= column .* or an explicit key="):
            TableMapping(source="orgs.csv", label="Organization")

    def test_an_explicit_key_still_wins(self):
        m = TableMapping(source="hr.csv", label="Person", key="employee_id", name="full_name")
        assert m.key == "employee_id"

    async def test_a_row_lands_on_the_id_prose_would_compute(
        self, real_falkordb_rag_factory, llm, tmp_path
    ):
        """No alias, no merge step needed: the ids are identical from the start."""
        from graphrag_sdk import ExactMatchResolution
        from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
            compute_entity_id,
        )

        path = tmp_path / "orgs.csv"
        path.write_text("org_name,country\nNorthwind Energy,Norway\n", encoding="utf-8")
        rag = real_falkordb_rag_factory(
            llm=llm,
            resolver=ExactMatchResolution(resolve_property="name"),
            ontology=Ontology(
                entities=[Entity(label="Organization")],
                tables=[
                    TableMapping(
                        source="orgs.csv",
                        label="Organization",
                        name="org_name",
                        properties={"country": Column("country")},
                    )
                ],
            ),
        )
        await rag.ingest(str(path))
        rows = await rag.query("MATCH (o:Organization) RETURN o.id, o.name, o.orgs__country")
        assert rows == [
            [compute_entity_id("Northwind Energy", "Organization"), "Northwind Energy", "Norway"]
        ]
        await rag.close()
