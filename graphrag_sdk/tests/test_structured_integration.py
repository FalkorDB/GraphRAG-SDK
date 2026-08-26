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

from graphrag_sdk.ingestion.mapping import Column, Link, Table
from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution

ORGS = Table(
    "Organization",
    key="org_id",
    name="org_name",
    hq_country="hq_country",
    employee_count=Column("employee_count", "INTEGER"),
)

EMPLOYEES = Table(
    "Person",
    key="employee_id",
    name="full_name",
    age=Column("age", "INTEGER"),
    title=Column("job_title"),
    links=[Link("WORKS_AT", to="Organization", by="org_id")],
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


class TestAggregationThroughThePublicApi:
    """The payoff, reached the documented way.

    Declaring that ``age`` is an INTEGER exists so a question can be answered by
    querying rather than by finding a passage that states the answer. That needs
    the text-to-Cypher path, and it needs it holding the ontology the mapping
    declared during an ingest that happened after the client was built.
    """

    async def test_declared_columns_reach_the_retrieval_strategy(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, enable_cypher=True)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)

        strategy_ontology = rag._retrieval_strategy._ontology
        person = next(e for e in strategy_ontology.entities if e.label == "Person")
        assert {p.name for p in person.properties} >= {"age", "title", "employee_id"}, (
            "the mapping declared these during an ingest that ran after the "
            "strategy was constructed; a stale copy cannot aggregate over them"
        )

    async def test_the_public_query_reads_the_graph(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv, employees_csv
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(orgs_csv, mapping=ORGS)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)

        rows = await rag.query(
            "MATCH (p:Person)-[r:RELATES]->(o:Organization) "
            "WHERE r.rel_type = 'WORKS_AT' AND o.name = $company "
            "RETURN avg(p.age) AS mean_age, count(p) AS people",
            {"company": "Acme Corp"},
        )
        assert rows == [[39.5, 2]]

    async def test_query_returns_an_empty_list_when_nothing_matches(
        self, real_falkordb_rag_factory, llm, resolver
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        assert await rag.query("MATCH (p:Person {name:'Nobody'}) RETURN p.name") == []


class TestADocumentRemembersHowItWasWritten:
    """An update may not change its mind about what a document is.

    Measured before this: ``update(path)`` on a document written from a CSV
    re-read it as prose, replaced its record chunks with one text chunk, and took
    every entity with them. Two organizations before the call, none after, and
    nothing raised. The same call arrives from ``apply_changes(modified=[...])``,
    which is how a scheduled sync would have quietly emptied a table.
    """

    async def test_updating_a_table_as_prose_is_refused(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(orgs_csv, mapping=ORGS)

        with pytest.raises(ValueError, match="written from a structured source"):
            await rag.update(orgs_csv)

        assert await _rows(rag, "MATCH (o:Organization) RETURN count(o)") == [[2]]

    async def test_apply_changes_reports_it_instead_of_destroying_the_table(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(orgs_csv, mapping=ORGS)

        result = await rag.apply_changes(modified=[orgs_csv])

        assert result.modified[0].error_type == "ValueError"
        assert await _rows(rag, "MATCH (o:Organization) RETURN count(o)") == [[2]]

    async def test_updating_prose_with_a_mapping_is_refused(
        self, real_falkordb_rag_factory, scripted_llm, resolver, orgs_csv
    ):
        llm = scripted_llm([("Acme Corp", "Organization", "A company")])
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(text="Acme Corp exists.", document_id="note.txt", resolver=resolver)

        with pytest.raises(ValueError, match="written from text"):
            await rag.update(orgs_csv, mapping=ORGS, document_id="note.txt")

    async def test_the_kind_is_persisted_on_the_document(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv
    ):
        """The guard reads this back, so it has to survive a reconnect rather than
        living only in the process that ran the ingest."""
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(orgs_csv, mapping=ORGS)
        record = await rag._graph_store.get_document_record(os.path.normpath(orgs_csv))
        assert record is not None and record.kind == "structured"


class TestWritesAreIndexedForScale:
    async def test_every_written_label_gets_an_id_index(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv, employees_csv
    ):
        """Every write is ``MERGE (n:Label {id: ...})``, and a MERGE can only use
        an index on the label in its own pattern. Indexing ``__Entity__.id`` does
        not help, because that label is added by a later SET. Without a per-label
        index, writing n nodes costs O(n^2): 25k rows took 95s, and 14s with.
        """
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(orgs_csv, mapping=ORGS)
        await rag.ingest(employees_csv, mapping=EMPLOYEES)

        rows = await rag.query(
            "CALL db.indexes() YIELD label, properties, types RETURN label, properties, types"
        )
        indexed = {row[0] for row in rows if "id" in (row[1] or []) and "RANGE" in str(row[2])}
        assert {"Person", "Organization", "Chunk", "Document"} <= indexed


class TestIngestOrderMustNotDecideTheGraph:
    """Declaring mappings up front, and why it exists.

    The extractor can only label an entity with a label the ontology already
    has. Read a document before a mapping is declared and "Carbon Farming" is
    filed under a built-in guess like ``Concept``; the table then declares it a
    ``MitigationPractice``, and resolution — which matches on name *and* label so
    "Apple" the company never joins "Apple" the fruit — correctly refuses to
    merge them. Same files, only the order changed: prose first merged 0,
    tables first merged 5. Nothing was raised either way.
    """

    NOTE = (
        "Carbon Farming is the umbrella set of agricultural practices for "
        "sequestration. Alternate Wetting and Drying (AWD) is a rice water "
        "management practice that reduces methane."
    )

    @pytest.fixture
    def practices_csv(self, tmp_path):
        path = tmp_path / "practices.csv"
        path.write_text(
            "practice_id,practice_name,crop_system\n"
            "PR-CF,Carbon Farming,multi-crop\n"
            "PR-AWD,Alternate Wetting and Drying (AWD),rice\n",
            encoding="utf-8",
        )
        return str(path)

    @staticmethod
    def _mapping():
        return Table(
            "MitigationPractice", key="practice_id", name="practice_name", crop_system="crop_system"
        )

    async def test_declaring_a_mapping_writes_no_data(
        self, real_falkordb_rag_factory, llm, resolver, practices_csv
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.declare_mapping(self._mapping())

        assert await _rows(rag, "MATCH (n) RETURN count(n)") == [[0]]
        ontology = await rag.get_ontology()
        assert any(e.label == "MitigationPractice" for e in ontology.entities)

    async def test_declaring_is_idempotent(self, real_falkordb_rag_factory, llm, resolver):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.declare_mapping(self._mapping())
        await rag.declare_mapping(self._mapping())
        ontology = await rag.get_ontology()
        practice = next(e for e in ontology.entities if e.label == "MitigationPractice")
        assert {p.name for p in practice.properties} == {"practice_id", "crop_system"}

    async def test_constructor_mappings_are_declared_before_the_first_ingest(
        self, real_falkordb_rag_factory, scripted_llm, resolver, practices_csv
    ):
        """The whole point: prose arrives first and still lands on the declared
        label, so the halves join."""
        llm = scripted_llm(
            [
                ("Carbon Farming", "MitigationPractice", "Umbrella set of practices"),
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, mappings=[self._mapping()])
        await rag.ingest(text=self.NOTE, document_id="note.txt", resolver=resolver)
        await rag.ingest(practices_csv, mapping=self._mapping())
        result = await rag.finalize()

        assert result.entities_deduplicated >= 1
        bridged = await _rows(
            rag,
            "MATCH (e:__Entity__)-[:MENTIONED_IN]->(:Chunk)<-[:PART_OF]-(d:Document) "
            "WITH e, count(DISTINCT d) AS n WHERE n > 1 RETURN count(e)",
        )
        assert bridged == [[1]]

    async def test_a_guessed_label_is_adopted_into_the_declared_one(
        self, real_falkordb_rag_factory, scripted_llm, resolver, practices_csv
    ):
        """The fix, without the caller doing anything.

        The extractor guessed ``Concept`` from a built-in list; the mapping
        *declares* ``MitigationPractice``. Same thing, described by two sources,
        one of which knows its type. The declared label survives and absorbs the
        other, so a table and a document join regardless of arrival order.
        """
        llm = scripted_llm(
            [("Carbon Farming", "Concept", "An agricultural approach")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(text=self.NOTE, document_id="note.txt", resolver=resolver)
        await rag.ingest(practices_csv, mapping=self._mapping())
        result = await rag.finalize()

        assert result.entities_deduplicated >= 1
        rows = await _rows(
            rag,
            "MATCH (e:__Entity__) WHERE e.name = 'Carbon Farming' "
            "RETURN count(e), head([l IN labels(e) WHERE l <> '__Entity__'])",
        )
        assert rows == [[1, "MitigationPractice"]], "one node, under the declared label"
        assert result.unmerged_name_collisions == {}

    async def test_the_documents_description_survives_the_adoption(
        self, real_falkordb_rag_factory, scripted_llm, resolver, practices_csv
    ):
        """Adopting must not cost what the document knew."""
        llm = scripted_llm(
            [("Carbon Farming", "Concept", "Umbrella practices for sequestration")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(text=self.NOTE, document_id="note.txt", resolver=resolver)
        await rag.ingest(practices_csv, mapping=self._mapping())
        await rag.finalize()

        rows = await _rows(
            rag,
            "MATCH (e:MitigationPractice {name:'Carbon Farming'}) "
            "RETURN e.description, e.practice_id",
        )
        assert "sequestration" in (rows[0][0] or "")
        assert rows[0][1] == "PR-CF", "and the table's own key is still there"

    async def test_two_guessed_labels_are_still_kept_apart(
        self, real_falkordb_rag_factory, scripted_llm, resolver
    ):
        """The guard that matters stays. Neither label is declared by a mapping,
        so there is nothing authoritative to prefer, and "Apple" the company must
        not become "Apple" the fruit."""
        llm = scripted_llm(
            [
                ("Apple", "Organization", "A technology company"),
                ("Apple", "Product", "A fruit"),
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(text="Apple builds computers.", document_id="a.txt", resolver=resolver)
        await rag.ingest(text="Apple grows on trees.", document_id="b.txt", resolver=resolver)
        result = await rag.finalize()

        assert await _rows(rag, "MATCH (e:__Entity__) WHERE e.name = 'Apple' RETURN count(e)") == [
            [2]
        ], "no mapping declared either label, so neither wins"
        assert "apple" in result.unmerged_name_collisions

    async def test_nothing_is_reported_when_labels_agree(
        self, real_falkordb_rag_factory, llm, resolver, practices_csv
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(practices_csv, mapping=self._mapping())
        result = await rag.finalize()
        assert result.unmerged_name_collisions == {}


class TestAdoptionMovesEverythingItShould:
    """The adoption pass deletes a node, so what was on it must survive.

    Merging across labels is the one place the resolver crosses a line it
    otherwise holds, and it does so by deleting the guessed node. Anything the
    document contributed has to arrive on the survivor first.
    """

    PRACTICES = Table("Practice", key="pid", name="pname", crop=Column("crop"))

    @pytest.fixture
    def practices_csv(self, tmp_path):
        path = tmp_path / "p.csv"
        path.write_text("pid,pname,crop\nPR-CF,Carbon Farming,multi\n", encoding="utf-8")
        return str(path)

    async def test_edges_in_both_directions_move_to_the_survivor(
        self, real_falkordb_rag_factory, llm, resolver, practices_csv
    ):
        """Adoption deletes the guessed node, so its edges must move first.

        The edges are written directly here rather than extracted: the scripted
        extraction fixture emits ``"relationships": []``, so a prose-driven test
        would pass while proving nothing.
        """
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(practices_csv, mapping=self.PRACTICES)

        # A guessed twin of the declared node, with an edge each way.
        await rag.query(
            "MERGE (g:Concept {id:'carbon_farming__concept'}) "
            "SET g.name = 'Carbon Farming', g.description = 'from prose' "
            "SET g:__Entity__ "
            "MERGE (o:Location {id:'kenya__location'}) SET o.name = 'Kenya' SET o:__Entity__ "
            "MERGE (g)-[:RELATES {rel_type:'PRACTISED_IN'}]->(o) "
            "MERGE (o)-[:RELATES {rel_type:'HOSTS'}]->(g)"
        )

        await rag.finalize()

        assert await _rows(
            rag, "MATCH (e:__Entity__ {id:'carbon_farming__concept'}) RETURN count(e)"
        ) == [[0]], "the guessed node is gone"
        outgoing = await _rows(
            rag,
            "MATCH (p:Practice {name:'Carbon Farming'})-[r:RELATES]->(o) RETURN r.rel_type, o.name",
        )
        incoming = await _rows(
            rag,
            "MATCH (p:Practice {name:'Carbon Farming'})<-[r:RELATES]-(o) RETURN r.rel_type, o.name",
        )
        assert outgoing == [["PRACTISED_IN", "Kenya"]], "direction must be preserved"
        assert incoming == [["HOSTS", "Kenya"]]

    async def test_the_documents_provenance_moves_too(
        self, real_falkordb_rag_factory, scripted_llm, resolver, practices_csv
    ):
        """If the MENTIONED_IN edges did not move, the surviving node would no
        longer remember the document it came from."""
        llm = scripted_llm([("Carbon Farming", "Concept", "An approach")])
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(text="Carbon Farming matters.", document_id="note.txt", resolver=resolver)
        await rag.ingest(practices_csv, mapping=self.PRACTICES)
        await rag.finalize()

        rows = await _rows(
            rag,
            "MATCH (p:Practice {name:'Carbon Farming'})-[:MENTIONED_IN]->(:Chunk)"
            "<-[:PART_OF]-(d:Document) RETURN DISTINCT d.id ORDER BY d.id",
        )
        sources = [r[0] for r in rows]
        assert "note.txt" in sources, "the document must still be reachable"
        assert len(sources) == 2, "and so must the table"

    async def test_running_finalize_twice_changes_nothing(
        self, real_falkordb_rag_factory, scripted_llm, resolver, practices_csv
    ):
        llm = scripted_llm([("Carbon Farming", "Concept", "An approach")])
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(text="Carbon Farming matters.", document_id="note.txt", resolver=resolver)
        await rag.ingest(practices_csv, mapping=self.PRACTICES)

        first = await rag.finalize()
        snapshot = await _rows(rag, "MATCH (e:__Entity__) RETURN e.id ORDER BY e.id")
        second = await rag.finalize()

        assert first.entities_deduplicated >= 1
        assert second.entities_deduplicated == 0, "nothing left to merge"
        assert await _rows(rag, "MATCH (e:__Entity__) RETURN e.id ORDER BY e.id") == snapshot
