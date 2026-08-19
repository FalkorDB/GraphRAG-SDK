"""Structured ingestion against a real FalkorDB, end to end.

The unit tests pin what the pipeline *intends* to write. These pin what actually
lands, and specifically the part no unit test can reach: a table and a document
describing the same company must end up on one node, with the table's typed
columns still there afterwards.

Skipped unless ``RUN_INTEGRATION=1``.
"""

from __future__ import annotations

import os

import pytest

from graphrag_sdk.ingestion.mapping import (
    Column,
    EdgeMapping,
    NodeMapping,
    RecordMapping,
    Table,
)
from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution

ORGS = Table(
    "Organization",
    key="org_id",
    name="org_name",
    hq_country="hq_country",
    employee_count=Column("employee_count", "INTEGER"),
)

EMPLOYEES = RecordMapping(
    nodes=[
        NodeMapping(
            label="Person",
            key="employee_id",
            name="full_name",
            properties={"age": Column("age", "INTEGER"), "title": Column("job_title")},
        ),
        NodeMapping(label="Organization", key="org_id", reference=True),
    ],
    edges=[EdgeMapping(type="WORKS_AT", source="Person", target="Organization")],
)


@pytest.fixture
def orgs_csv(tmp_path):
    path = tmp_path / "orgs.csv"
    path.write_text(
        "org_id,org_name,hq_country,employee_count\n"
        "ORG-42,Acme Corp,US,1200\n"
        "ORG-7,Globex,GB,340\n",
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def employees_csv(tmp_path):
    path = tmp_path / "employees.csv"
    path.write_text(
        "employee_id,full_name,age,job_title,org_id\n"
        "E-1,Alice Smith,34,Engineer,ORG-42\n"
        "E-2,Bob Jones,45,CFO,ORG-42\n"
        "E-3,Carol White,29,Engineer,ORG-7\n",
        encoding="utf-8",
    )
    return str(path)


async def _rows(rag, cypher, params=None):
    result = await rag._graph_store.query_raw(cypher, params)
    return result.result_set or []


@pytest.fixture
def resolver(embedder):
    return ExactMatchResolution(resolve_property="name")


class TestStructuredIngestIntoARealGraph:
    async def test_a_table_becomes_typed_nodes_and_edges(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv, employees_csv
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(orgs_csv, mapping=ORGS)
        result = await rag.ingest(employees_csv, mapping=EMPLOYEES)

        assert result.records == 3
        assert result.edges == 3

        # Typed, so it can be aggregated. A string "34" cannot be averaged.
        rows = await _rows(
            rag,
            "MATCH (p:Person)-[r:RELATES]->(o:Organization) "
            "WHERE r.rel_type = 'WORKS_AT' AND o.name = 'Acme Corp' "
            "RETURN avg(p.age) AS mean_age, count(p) AS people",
        )
        assert rows == [[39.5, 2]]

    async def test_every_record_is_traceable_to_its_row(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)

        rows = await _rows(
            rag,
            "MATCH (p:Person {name:'Alice Smith'})-[:MENTIONED_IN]->(c:Chunk)"
            "<-[:PART_OF]-(d:Document) RETURN d.id, c.kind, c.record_key",
        )
        assert rows == [[os.path.normpath(employees_csv), "record", "E-1"]]

    async def test_the_row_is_recoverable_from_its_chunk(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        """The typed projection lives on the entity; the chunk keeps the cells
        verbatim, so the original row survives in the graph."""
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)

        rows = await _rows(
            rag,
            "MATCH (c:Chunk {record_key:'E-1'}) RETURN c.full_name, c.job_title, c.text",
        )
        assert rows[0][0] == "Alice Smith"
        assert rows[0][1] == "Engineer"
        assert "Alice Smith" in rows[0][2]

    async def test_records_are_not_chained_into_a_false_sequence(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)
        rows = await _rows(rag, "MATCH ()-[r:NEXT_CHUNK]->() RETURN count(r)")
        assert rows == [[0]]

    async def test_a_foreign_key_cannot_overwrite_a_real_name(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv, employees_csv
    ):
        """orgs.csv names ORG-42 "Acme Corp". employees.csv only points at it.
        The pointer must not rename it to "ORG-42"."""
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(orgs_csv, mapping=ORGS)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)
        rows = await _rows(
            rag, "MATCH (o:Organization {org_id:'ORG-42'}) RETURN o.name, o.employee_count"
        )
        assert rows == [["Acme Corp", 1200]]

    async def test_a_foreign_key_seen_first_is_named_when_its_own_source_arrives(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv, employees_csv
    ):
        """Reverse order. The stub exists first, keyed but not yet named."""
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)
        stub = await _rows(rag, "MATCH (o:Organization {org_id:'ORG-42'}) RETURN o.name, o.is_stub")
        assert stub == [["ORG-42", True]]

        await rag.ingest(orgs_csv, mapping=ORGS)
        named = await _rows(
            rag,
            "MATCH (o:Organization {org_id:'ORG-42'}) RETURN o.name, o.is_stub, o.employee_count",
        )
        assert named == [["Acme Corp", False, 1200]]

    async def test_re_ingesting_the_same_table_does_not_duplicate(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)
        rows = await _rows(rag, "MATCH (p:Person) RETURN count(p)")
        assert rows == [[3]]

    async def test_the_mapping_declares_typed_columns_in_the_ontology(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        """Without this, generated Cypher cannot see that age is a number."""
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)
        ontology = await rag.get_ontology()
        person = next(e for e in ontology.entities if e.label == "Person")
        types = {p.name: p.type for p in person.properties}
        assert types["age"] == "INTEGER"
        assert types["employee_id"] == "STRING"
        assert "name" not in types, (
            "declaring `name` lets the extractor answer it with a null and blank "
            "out the display name of everything extracted from prose"
        )

    async def test_a_second_table_extends_an_existing_label(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv, employees_csv
    ):
        """employees.csv declares Organization by key only; orgs.csv then adds
        employee_count to the same label. The ontology store refuses to *modify*
        an existing label, so this has to go through ontology evolution."""
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)
        await rag.ingest(orgs_csv, mapping=ORGS)
        ontology = await rag.get_ontology()
        org = next(e for e in ontology.entities if e.label == "Organization")
        types = {p.name: p.type for p in org.properties}
        assert types["employee_count"] == "INTEGER"
        assert types["org_id"] == "STRING"

    async def test_a_mapping_that_does_not_fit_writes_nothing(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv
    ):
        from graphrag_sdk.ingestion.mapping import MappingError

        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        with pytest.raises(MappingError):
            await rag.ingest(orgs_csv, mapping=EMPLOYEES)
        rows = await _rows(rag, "MATCH (n) RETURN count(n)")
        assert rows == [[0]], "a rejected mapping must not leave a half-written graph"


class TestTheTwoHalvesBecomeOneGraph:
    async def test_a_row_and_a_sentence_about_one_company_resolve_to_one_node(
        self, real_falkordb_rag_factory, scripted_llm, resolver, orgs_csv, employees_csv
    ):
        """The point of the whole feature. The CSV knows Acme Corp has 1200
        employees; the note knows it missed its revenue target. One node has to
        hold both, or neither question can be answered from the other's facts.
        """
        llm = scripted_llm(
            [
                ("Acme Corp", "Organization", "Reported a revenue miss"),
                ("Alice Smith", "Person", "An engineer at Acme Corp"),
            ]
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)

        await rag.ingest(orgs_csv, mapping=ORGS)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)
        await rag.ingest(
            text="Acme Corp reported a Q3 revenue miss. Alice Smith, an engineer "
            "at Acme Corp, presented the remediation plan.",
            document_id="board_note.txt",
            resolver=resolver,
        )
        await rag.finalize()

        rows = await _rows(rag, "MATCH (o:Organization {name:'Acme Corp'}) RETURN count(o)")
        assert rows == [[1]], "the table's Acme and the note's Acme must be one node"

        # And the typed column survived the merge, which is the part that breaks
        # when a duplicate is deleted before its properties are carried over.
        rows = await _rows(
            rag,
            "MATCH (o:Organization {name:'Acme Corp'}) "
            "RETURN o.employee_count, o.hq_country, o.org_id",
        )
        assert rows == [[1200, "US", "ORG-42"]]

    async def test_the_merged_node_is_reachable_from_both_sources(
        self, real_falkordb_rag_factory, scripted_llm, resolver, orgs_csv, employees_csv
    ):
        llm = scripted_llm(
            [
                ("Acme Corp", "Organization", "Reported a revenue miss"),
            ]
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(orgs_csv, mapping=ORGS)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)
        await rag.ingest(
            text="Acme Corp reported a Q3 revenue miss.",
            document_id="board_note.txt",
            resolver=resolver,
        )
        await rag.finalize()

        rows = await _rows(
            rag,
            "MATCH (o:Organization {name:'Acme Corp'})-[:MENTIONED_IN]->(:Chunk)"
            "<-[:PART_OF]-(d:Document) RETURN DISTINCT d.id ORDER BY d.id",
        )
        sources = [r[0] for r in rows]
        assert os.path.normpath(orgs_csv) in sources
        assert "board_note.txt" in sources

    async def test_the_cross_source_question_is_answerable_in_one_query(
        self, real_falkordb_rag_factory, scripted_llm, resolver, orgs_csv, employees_csv
    ):
        """Neither half can answer this alone: "who works at the company that
        reported the revenue miss, and how old are they"."""
        llm = scripted_llm(
            [
                ("Acme Corp", "Organization", "Reported a revenue miss"),
            ]
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(orgs_csv, mapping=ORGS)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)
        await rag.ingest(
            text="Acme Corp reported a Q3 revenue miss.",
            document_id="board_note.txt",
            resolver=resolver,
        )
        await rag.finalize()

        rows = await _rows(
            rag,
            "MATCH (c:Chunk)<-[:MENTIONED_IN]-(o:Organization)<-[r:RELATES]-(p:Person) "
            "WHERE c.text CONTAINS 'revenue miss' AND r.rel_type = 'WORKS_AT' "
            "RETURN p.name, p.age ORDER BY p.name",
        )
        assert rows == [["Alice Smith", 34], ["Bob Jones", 45]]

    async def test_re_ingesting_a_table_after_a_merge_does_not_duplicate(
        self, real_falkordb_rag_factory, scripted_llm, resolver, employees_csv, tmp_path
    ):
        """Regression, and the reason the survivor rule prefers a keyed id.

        finalize() folds the note's "Alice Smith" and the CSV's E-1 together. If
        the prose node survives, the keyed id no longer exists, so re-ingesting a
        corrected export recreates it: two E-1 people, one titled "Engineer" and
        one "engineer", each holding half the facts.
        """
        llm = scripted_llm(
            [("Alice Smith", "Person", "An engineer at Acme Corp")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)

        await rag.ingest(employees_csv, mapping=EMPLOYEES)
        await rag.ingest(
            text="Alice Smith presented the remediation plan.",
            document_id="board_note.txt",
            resolver=resolver,
        )
        await rag.finalize()

        assert await _rows(rag, "MATCH (p:Person {name:'Alice Smith'}) RETURN count(p)") == [[1]]

        corrected = tmp_path / "employees_v2.csv"
        corrected.write_text(
            "employee_id,full_name,age,job_title,org_id\n"
            "E-1,Alice Smith,34,Principal Engineer,ORG-42\n",
            encoding="utf-8",
        )
        await rag.ingest(str(corrected), mapping=EMPLOYEES, document_id="employees.csv")

        rows = await _rows(rag, "MATCH (p:Person {employee_id:'E-1'}) RETURN p.title, count(p)")
        assert rows == [["Principal Engineer", 1]], (
            "the corrected export must update the same node, not create a second one"
        )

    async def test_the_surviving_node_keeps_what_the_document_added(
        self, real_falkordb_rag_factory, scripted_llm, resolver, employees_csv
    ):
        """Preferring the keyed id must not cost the prose node's description."""
        llm = scripted_llm(
            [("Alice Smith", "Person", "Led the remediation plan for the board")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)
        await rag.ingest(
            text="Alice Smith presented the remediation plan.",
            document_id="board_note.txt",
            resolver=resolver,
        )
        await rag.finalize()

        rows = await _rows(rag, "MATCH (p:Person {employee_id:'E-1'}) RETURN p.description, p.age")
        assert rows[0][0], "the description the document supplied must survive the merge"
        assert rows[0][1] == 34, "and the table's typed column must survive it too"


class TestReSyncingAStructuredSource:
    """A table is a snapshot, so the graph has to follow it downwards too.

    Rows that change and rows that appear were always handled. A row *deleted*
    from the source was not: nothing rewrote it, so it stayed in the graph
    forever with an orphaned record chunk still attached to its document. These
    pin the behaviour from both entry points, because the one users reach for
    first is plain ``ingest``.
    """

    @staticmethod
    def _write(path, rows: list[str]) -> str:
        path.write_text(
            "org_id,org_name,hq_country,employee_count\n" + "".join(rows),
            encoding="utf-8",
        )
        return str(path)

    ACME = "ORG-42,Acme Corp,US,1200\n"
    GLOBEX = "ORG-7,Globex,GB,340\n"

    async def test_update_removes_a_row_that_left_the_source(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        source = tmp_path / "orgs.csv"
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(self._write(source, [self.ACME, self.GLOBEX]), mapping=ORGS)

        result = await rag.update(self._write(source, [self.ACME]), mapping=ORGS)

        assert result.entities_deleted == 1
        assert await _rows(rag, "MATCH (o:Organization) RETURN o.org_id") == [["ORG-42"]]
        assert await _rows(rag, "MATCH (:Document)-[:PART_OF]->(c:Chunk) RETURN count(c)") == [
            [1]
        ], "the departed row's chunk must go with it"

    async def test_re_ingest_removes_a_row_that_left_the_source(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """Ingesting a source already in the graph means "this is its current
        state", so it re-syncs rather than writing over the top."""
        source = tmp_path / "orgs.csv"
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(self._write(source, [self.ACME, self.GLOBEX]), mapping=ORGS)

        result = await rag.ingest(self._write(source, [self.ACME]), mapping=ORGS)

        assert result.replaced_existing is True
        assert result.entities_deleted == 1
        assert await _rows(rag, "MATCH (o:Organization) RETURN o.org_id") == [["ORG-42"]]

    async def test_re_ingest_does_not_double_a_source_chunks(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """Record chunk ids derive from the effective document id, which the
        update path deliberately makes a pending one. Writing over the top
        afterwards would key the same row to a second chunk."""
        source = tmp_path / "orgs.csv"
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        path = self._write(source, [self.ACME, self.GLOBEX])
        await rag.ingest(path, mapping=ORGS)
        await rag.update(path, mapping=ORGS, document_id=os.path.normpath(path))
        await rag.ingest(path, mapping=ORGS)

        assert await _rows(rag, "MATCH (:Document)-[:PART_OF]->(c:Chunk) RETURN count(c)") == [[2]]

    async def test_an_unchanged_source_is_a_no_op(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        source = tmp_path / "orgs.csv"
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        path = self._write(source, [self.ACME, self.GLOBEX])
        await rag.ingest(path, mapping=ORGS)

        result = await rag.update(path, mapping=ORGS)
        assert result.no_op is True

    async def test_a_changed_mapping_is_not_an_unchanged_source(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """Identical rows under a different declaration produce a different
        graph, so the content hash covers the mapping. Without that, adding a
        column to a mapping would be silently ignored."""
        source = tmp_path / "orgs.csv"
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        path = self._write(source, [self.ACME, self.GLOBEX])
        await rag.ingest(path, mapping=ORGS)

        wider = Table(
            "Organization",
            key="org_id",
            name="org_name",
            hq_country="hq_country",
            employee_count=Column("employee_count", "INTEGER"),
            staff=Column("employee_count", "INTEGER"),
        )
        result = await rag.update(path, mapping=wider)

        assert result.no_op is False
        assert await _rows(rag, "MATCH (o:Organization {org_id:'ORG-42'}) RETURN o.staff") == [
            [1200]
        ]

    async def test_an_entity_another_source_still_mentions_survives(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path, employees_csv
    ):
        """Scoped cleanup: Globex leaves orgs.csv but employees.csv still points
        at it through Carol, so it must not be deleted."""
        source = tmp_path / "orgs.csv"
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(self._write(source, [self.ACME, self.GLOBEX]), mapping=ORGS)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)

        await rag.update(self._write(source, [self.ACME]), mapping=ORGS)

        assert await _rows(rag, "MATCH (o:Organization {org_id:'ORG-7'}) RETURN o.name") == [
            ["Globex"]
        ], "still referenced by employees.csv, so it stays"

    async def test_a_mapping_that_stops_fitting_leaves_the_graph_alone(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        from graphrag_sdk.ingestion.mapping import MappingError

        source = tmp_path / "orgs.csv"
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        path = self._write(source, [self.ACME, self.GLOBEX])
        await rag.ingest(path, mapping=ORGS)

        source.write_text("something,else\n1,2\n", encoding="utf-8")
        with pytest.raises(MappingError):
            await rag.update(path, mapping=ORGS)

        assert await _rows(rag, "MATCH (o:Organization) RETURN count(o)") == [[2]]
        pendings = await _rows(
            rag, "MATCH (d:Document) WHERE d.id CONTAINS 'pending' RETURN count(d)"
        )
        assert pendings == [[0]], "a rejected mapping must not leave a pending Document behind"

    async def test_update_rejects_arguments_that_cannot_apply(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        source = self._write(tmp_path / "orgs.csv", [self.ACME])
        with pytest.raises(ValueError, match="does not apply"):
            await rag.update(source, mapping=ORGS, chunker=object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="cache_unchanged_chunks"):
            await rag.update(source, mapping=ORGS, cache_unchanged_chunks=True)
        with pytest.raises(ValueError, match="only apply with 'mapping'"):
            await rag.update(source, strict_mapping=True)

    async def test_update_can_create_a_source_it_has_never_seen(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        source = self._write(tmp_path / "orgs.csv", [self.ACME])
        result = await rag.update(source, mapping=ORGS, if_missing="ingest")
        assert result.replaced_existing is False
        assert await _rows(rag, "MATCH (o:Organization) RETURN o.org_id") == [["ORG-42"]]


class TestTheTableOwnsItsColumns:
    """Who wins when a document and a table disagree about a declared column.

    A mapping declares its columns so generated Cypher can see their types, but
    that also puts them in front of the extractor, which answers them from prose.
    Measured before this: Alice's title arrived as ``"engineer"`` from a memo and
    overwrote the ``"Engineer"`` the HR export spelled. The export owns that
    column, so the memo does not get to write it.
    """

    async def test_prose_cannot_overwrite_a_column_the_table_declared(
        self, real_falkordb_rag_factory, scripted_llm, resolver, employees_csv
    ):
        llm = scripted_llm(
            [("Alice Smith", "Person", "An engineer at Acme Corp")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)
        await rag.ingest(
            text="Alice Smith is an engineer who presented the plan.",
            document_id="board_note.txt",
            resolver=resolver,
        )
        await rag.finalize()

        rows = await _rows(rag, "MATCH (p:Person {employee_id:'E-1'}) RETURN p.title, p.age")
        assert rows == [["Engineer", 34]], "the export spelled it, so the export keeps it"

    async def test_a_document_still_contributes_what_it_legitimately_knows(
        self, real_falkordb_rag_factory, scripted_llm, resolver, employees_csv
    ):
        """Only the owned column's value is discarded. The entity, its name and
        its description are exactly as extracted."""
        llm = scripted_llm(
            [("Alice Smith", "Person", "Led the remediation plan for the board")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)
        await rag.ingest(
            text="Alice Smith led the remediation plan.",
            document_id="board_note.txt",
            resolver=resolver,
        )
        await rag.finalize()

        rows = await _rows(rag, "MATCH (p:Person {employee_id:'E-1'}) RETURN p.description, p.name")
        assert "remediation" in (rows[0][0] or "")
        assert rows[0][1] == "Alice Smith"

    async def test_ownership_is_scoped_to_the_declared_columns(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        """The guard is per property, not per label. ``name`` in particular is
        never owned: it is not declarable as a mapped property at all, so a
        document remains free to name an entity."""
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)
        ontology = await rag.get_ontology()
        person = next(e for e in ontology.entities if e.label == "Person")
        owned = {p.name for p in person.properties if p.structured}
        assert owned == {"employee_id", "age", "title"}
        assert "name" not in owned

    async def test_ownership_is_recorded_in_the_persisted_ontology(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)
        reloaded = await rag.get_ontology()
        person = next(e for e in reloaded.entities if e.label == "Person")
        assert all(p.structured for p in person.properties), (
            "ownership has to survive a reload, or the guard only works in the "
            "process that happened to run the ingest"
        )
