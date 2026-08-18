"""Structured ingestion against a real FalkorDB, end to end.

The unit tests pin what the pipeline *intends* to write. These pin what actually
lands, and specifically the part no unit test can reach: a table and a document
describing the same company must end up on one node, with the table's typed
columns still there afterwards.

Skipped unless ``RUN_INTEGRATION=1``.
"""

from __future__ import annotations

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
        assert rows == [["employees.csv", "record", "E-1"]]

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
        assert "orgs.csv" in sources
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
