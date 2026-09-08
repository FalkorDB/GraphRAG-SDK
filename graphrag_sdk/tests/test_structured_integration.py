"""Structured ingestion against a real FalkorDB, end to end.

The unit tests pin what the pipeline *intends* to write. These pin what actually
lands, and specifically the part no unit test can reach: a table and a document
describing the same company must end up on one node, with the table's typed
columns still there afterwards.

Two things about the surface these tests drive:

- A mapping is part of the ontology. It is declared once, at construction —
  ``GraphRAG(ontology=Ontology(tables=[...]))`` — and ``ingest`` finds it by the
  **basename** of the file it is handed. Declaring it later is not equivalent: a
  mapping's labels have to be registered before anything is extracted.
- Every property a mapping writes is stored as ``<signature>__<property>``,
  where the signature comes from the source's file name. So the assertions below
  read ``p.employees__age`` where they once read ``p.age`` — same column, same
  claim. ``name`` and the other keys the SDK owns stay unsigned, which is what
  lets a keyed row and a prose mention meet on one node.

Skipped unless ``RUN_INTEGRATION=1``.
"""

from __future__ import annotations

import pytest

from graphrag_sdk import Column, Entity, Link, Ontology, TableMapping
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import DEFAULT_ENTITY_TYPES
from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution

ORGS = TableMapping(
    source="orgs.csv",
    label="Organization",
    key="org_id",
    name="org_name",
    properties={
        "hq_country": "hq_country",
        "employee_count": Column("employee_count", "INTEGER"),
    },
)

EMPLOYEES = TableMapping(
    source="employees.csv",
    label="Person",
    key="employee_id",
    name="full_name",
    properties={"age": Column("age", "INTEGER"), "title": Column("job_title")},
    links=[Link("WORKS_AT", to="Organization", by="org_id")],
)


def _ontology(*tables: TableMapping) -> Ontology:
    """The ontology these tests declare up front.

    The mappings have to be here rather than passed per call: ``ingest`` looks
    one up by the source's basename, and the labels it declares must exist before
    a document is extracted. The entity list is the SDK's own default set — what
    an ontology-less run used to seed on first connection — kept so the extractor
    still recognises the labels the scripted documents use.
    """
    return Ontology(
        entities=[Entity(label=label) for label in DEFAULT_ENTITY_TYPES],
        tables=list(tables),
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
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(ORGS, EMPLOYEES)
        )
        await rag.ingest(orgs_csv)
        result = await rag.ingest(employees_csv)

        assert result.records == 3
        assert result.edges == 3

        # Typed, so it can be aggregated. A string "34" cannot be averaged.
        rows = await _rows(
            rag,
            "MATCH (p:Person)-[r:RELATES]->(o:Organization) "
            "WHERE r.rel_type = 'WORKS_AT' AND o.name = 'Acme Corp' "
            "RETURN avg(p.employees__age) AS mean_age, count(p) AS people",
        )
        assert rows == [[39.5, 2]]

    async def test_every_record_is_traceable_to_its_row(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(EMPLOYEES))
        await rag.ingest(employees_csv)

        rows = await _rows(
            rag,
            "MATCH (p:Person {name:'Alice Smith'})-[:MENTIONED_IN]->(c:Chunk)"
            "<-[:PART_OF]-(d:Document) RETURN d.id, c.kind, c.record_key",
        )
        # The Document id is the table's name, not the path it was read from.
        assert rows == [["employees.csv", "record", "E-1"]]

    async def test_the_row_is_recoverable_from_its_chunk(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        """The typed projection lives on the entity; the chunk keeps the cells
        verbatim, so the original row survives in the graph."""
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(EMPLOYEES))
        await rag.ingest(employees_csv)

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
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(EMPLOYEES))
        await rag.ingest(employees_csv)
        rows = await _rows(rag, "MATCH ()-[r:NEXT_CHUNK]->() RETURN count(r)")
        assert rows == [[0]]

    async def test_a_foreign_key_cannot_overwrite_a_real_name(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv, employees_csv
    ):
        """orgs.csv names ORG-42 "Acme Corp". employees.csv only points at it.
        The pointer must not rename it to "ORG-42"."""
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(ORGS, EMPLOYEES)
        )
        await rag.ingest(orgs_csv)
        await rag.ingest(employees_csv)
        rows = await _rows(
            rag,
            "MATCH (o:Organization {orgs__org_id:'ORG-42'}) RETURN o.name, o.orgs__employee_count",
        )
        assert rows == [["Acme Corp", 1200]]
        # Counted too: keyed on orgs' signature alone, a duplicate stub written
        # by the pointer would sit there unseen by the query above.
        one = await _rows(
            rag,
            "MATCH (o:Organization) "
            "WHERE o.employees__org_id = 'ORG-42' OR o.orgs__org_id = 'ORG-42' "
            "RETURN count(o)",
        )
        assert one == [[1]]

    async def test_a_foreign_key_seen_first_is_named_when_its_own_source_arrives(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv, employees_csv
    ):
        """Reverse order. The stub exists first, keyed but not yet named.

        The stub's key is signed by the source that pointed at it —
        ``employees__org_id`` — and orgs.csv adds its own ``orgs__org_id`` when it
        arrives. One node throughout: the id comes from the key's value.
        """
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(ORGS, EMPLOYEES)
        )
        await rag.ingest(employees_csv)
        stub = await _rows(
            rag, "MATCH (o:Organization {employees__org_id:'ORG-42'}) RETURN o.name, o.is_stub"
        )
        assert stub == [["ORG-42", True]]

        await rag.ingest(orgs_csv)
        # Keyed on BOTH signatures at once, deliberately. Each source writes only
        # its own signed key, so if orgs.csv had created a second node instead of
        # filling in the stub, no node would carry both and this matches nothing.
        # Probing either signature alone passes on a graph that quietly holds two
        # organizations -- one named 'ORG-42' and orphaned, one named 'Acme Corp'
        # -- which is the regression this test exists for.
        named = await _rows(
            rag,
            "MATCH (o:Organization {employees__org_id:'ORG-42', orgs__org_id:'ORG-42'}) "
            "RETURN o.name, o.is_stub, o.orgs__employee_count",
        )
        assert named == [["Acme Corp", False, 1200]]
        one = await _rows(
            rag,
            "MATCH (o:Organization) "
            "WHERE o.employees__org_id = 'ORG-42' OR o.orgs__org_id = 'ORG-42' "
            "RETURN count(o)",
        )
        assert one == [[1]], "the pointer and the source must land on one node"

    async def test_re_ingesting_the_same_table_does_not_duplicate(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(EMPLOYEES))
        await rag.ingest(employees_csv)
        await rag.ingest(employees_csv)
        rows = await _rows(rag, "MATCH (p:Person) RETURN count(p)")
        assert rows == [[3]]

    async def test_the_mapping_declares_typed_columns_in_the_ontology(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        """Without this, generated Cypher cannot see that age is a number."""
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(EMPLOYEES))
        await rag.ingest(employees_csv)
        ontology = await rag.get_ontology()
        person = next(e for e in ontology.entities if e.label == "Person")
        types = {p.name: p.type for p in person.properties}
        assert types["employees__age"] == "INTEGER"
        assert types["employees__employee_id"] == "STRING"
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
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(ORGS, EMPLOYEES)
        )
        await rag.ingest(employees_csv)
        await rag.ingest(orgs_csv)
        ontology = await rag.get_ontology()
        org = next(e for e in ontology.entities if e.label == "Organization")
        types = {p.name: p.type for p in org.properties}
        assert types["employees__org_id"] == "STRING"
        assert types["orgs__employee_count"] == "INTEGER"
        assert types["orgs__org_id"] == "STRING"

    async def test_a_mapping_that_does_not_fit_writes_nothing(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv
    ):
        from graphrag_sdk.ingestion.mapping import MappingError

        # Declared for orgs.csv but describing employees.csv's columns. The
        # lookup is by basename, so this is the mapping the ingest below finds,
        # and it cannot fit the file it is applied to.
        misdeclared = TableMapping(
            source="orgs.csv",
            label="Person",
            key="employee_id",
            name="full_name",
            properties={"age": Column("age", "INTEGER")},
            links=[Link("WORKS_AT", to="Organization", by="org_id")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(misdeclared))
        with pytest.raises(MappingError):
            await rag.ingest(orgs_csv)
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
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(ORGS, EMPLOYEES)
        )

        await rag.ingest(orgs_csv)
        await rag.ingest(employees_csv)
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
            "RETURN o.orgs__employee_count, o.orgs__hq_country, o.orgs__org_id",
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
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(ORGS, EMPLOYEES)
        )
        await rag.ingest(orgs_csv)
        await rag.ingest(employees_csv)
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
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(ORGS, EMPLOYEES)
        )
        await rag.ingest(orgs_csv)
        await rag.ingest(employees_csv)
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
            "RETURN p.name, p.employees__age ORDER BY p.name",
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

        The corrected export keeps the file *name* — the mapping is found by
        basename and its signature comes from the same name, so a re-export filed
        under a new name would be a different table writing different properties.
        """
        llm = scripted_llm(
            [("Alice Smith", "Person", "An engineer at Acme Corp")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(EMPLOYEES))

        await rag.ingest(employees_csv)
        await rag.ingest(
            text="Alice Smith presented the remediation plan.",
            document_id="board_note.txt",
            resolver=resolver,
        )
        await rag.finalize()

        assert await _rows(rag, "MATCH (p:Person {name:'Alice Smith'}) RETURN count(p)") == [[1]]

        corrected = tmp_path / "corrected" / "employees.csv"
        corrected.parent.mkdir()
        corrected.write_text(
            "employee_id,full_name,age,job_title,org_id\n"
            "E-1,Alice Smith,34,Principal Engineer,ORG-42\n",
            encoding="utf-8",
        )
        await rag.ingest(str(corrected), document_id="employees.csv")

        rows = await _rows(
            rag,
            "MATCH (p:Person {employees__employee_id:'E-1'}) RETURN p.employees__title, count(p)",
        )
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
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(EMPLOYEES))
        await rag.ingest(employees_csv)
        await rag.ingest(
            text="Alice Smith presented the remediation plan.",
            document_id="board_note.txt",
            resolver=resolver,
        )
        await rag.finalize()

        rows = await _rows(
            rag,
            "MATCH (p:Person {employees__employee_id:'E-1'}) "
            "RETURN p.description, p.employees__age",
        )
        assert rows[0][0], "the description the document supplied must survive the merge"
        assert rows[0][1] == 34, "and the table's typed column must survive it too"


class TestASharedNodeKeepsWhatEachSourceGave:
    """One node, two writers. What the second one adds must not cost the first.

    The three faults these pin were found by a loss audit of a mixed corpus: a
    pointer that never keyed a node a document had created, a provenance list
    that remembered only the last document, and a dropped table that left its
    identity behind. Each is a store-level rule, so each is pinned against the
    real store.
    """

    # The link denormalises the target's name, so the pointer's id is the
    # name's — the same id a document's mention of "Acme Corp" derives.
    EMPLOYEES_NAMING_THEIR_ORG = TableMapping(
        source="employees.csv",
        label="Person",
        key="employee_id",
        name="full_name",
        properties={"age": Column("age", "INTEGER")},
        links=[Link("WORKS_AT", to="Organization", by="org_id", name="org_name")],
    )

    @pytest.fixture
    def employees_naming_their_org_csv(self, tmp_path):
        path = tmp_path / "employees.csv"
        path.write_text(
            "employee_id,full_name,age,org_id,org_name\n"
            "E-1,Alice Smith,34,ORG-42,Acme Corp\n"
            "E-2,Bob Jones,45,ORG-42,Acme Corp\n",
            encoding="utf-8",
        )
        return str(path)

    async def test_a_pointer_keys_the_entity_a_document_mentioned_first(
        self, real_falkordb_rag_factory, scripted_llm, resolver, employees_naming_their_org_csv
    ):
        """The note creates Acme Corp; employees.csv then points at it as ORG-42.

        Written ON CREATE only, the pointer found the node and left it exactly as
        the note had it: no ``entity_key``, no signed key, so nothing joinable and
        the next table keyed on ORG-42 would raise a placeholder beside it. The
        node must come out keyed, flagged as not yet owned, and otherwise as the
        note wrote it.
        """
        llm = scripted_llm([("Acme Corp", "Organization", "Reported a revenue miss")])
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(self.EMPLOYEES_NAMING_THEIR_ORG)
        )
        await rag.ingest(
            text="Acme Corp reported a Q3 revenue miss.",
            document_id="board_note.txt",
            resolver=resolver,
        )
        await rag.ingest(employees_naming_their_org_csv)

        rows = await _rows(
            rag,
            "MATCH (o:Organization) RETURN o.name, o.description, o.entity_key, o.is_stub, "
            "o.employees__org_id ORDER BY o.name",
        )
        assert rows == [["Acme Corp", "Reported a revenue miss", "ORG-42", True, "ORG-42"]], (
            "one node: the note's name and description, the pointer's key on top"
        )
        # Keyed means the row's edge landed on the same node the note created.
        via = await _rows(
            rag,
            "MATCH (p:Person)-[r:RELATES {rel_type:'WORKS_AT'}]->"
            "(o:Organization {entity_key:'ORG-42'}) RETURN p.name ORDER BY p.name",
        )
        assert [r[0] for r in via] == ["Alice Smith", "Bob Jones"]
        await rag.close()

    async def test_the_second_document_adds_its_chunks_instead_of_replacing_the_first(
        self, real_falkordb_rag_factory, scripted_llm, resolver, employees_csv
    ):
        """Two notes mention the row's Alice. ``source_chunk_ids`` must name both;
        deleting one note must take only that note's chunk back out."""
        llm = scripted_llm(
            [("Alice Smith", "Person", "Presented the plan")],
            [("Alice Smith", "Person", "Was promoted")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(EMPLOYEES))
        await rag.ingest(employees_csv)
        await rag.ingest(
            text="Alice Smith presented the plan.", document_id="note1.txt", resolver=resolver
        )
        await rag.ingest(
            text="Alice Smith was promoted.", document_id="note2.txt", resolver=resolver
        )

        async def provenance():
            rows = await _rows(
                rag,
                "MATCH (p:Person {name:'Alice Smith'}) "
                "OPTIONAL MATCH (p)-[:MENTIONED_IN]->(c:Chunk)<-[:PART_OF]-(d:Document) "
                "WHERE NOT d.id ENDS WITH '.csv' "
                "RETURN p.employees__age, p.source_chunk_ids, collect(c.id), collect(d.id)",
            )
            assert len(rows) == 1, "the row and both notes are one node"
            age, listed, mentioned, docs = rows[0]
            return age, sorted(listed or []), sorted(mentioned), sorted(docs)

        age, listed, mentioned, docs = await provenance()
        assert age == 34
        assert docs == ["note1.txt", "note2.txt"]
        assert listed == mentioned, "the property must agree with the MENTIONED_IN edges"
        assert len(listed) == 2

        await rag.delete_document("note2.txt")

        age, listed, mentioned, docs = await provenance()
        assert age == 34
        assert docs == ["note1.txt"]
        assert listed == mentioned, "note2's chunk is gone from the list, note1's is not"
        assert len(listed) == 1
        await rag.close()


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
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(ORGS))
        await rag.ingest(self._write(source, [self.ACME, self.GLOBEX]))

        result = await rag.update(self._write(source, [self.ACME]))

        assert result.entities_deleted == 1
        assert await _rows(rag, "MATCH (o:Organization) RETURN o.orgs__org_id") == [["ORG-42"]]
        assert await _rows(rag, "MATCH (:Document)-[:PART_OF]->(c:Chunk) RETURN count(c)") == [
            [1]
        ], "the departed row's chunk must go with it"

    async def test_re_ingest_removes_a_row_that_left_the_source(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """Ingesting a source already in the graph means "this is its current
        state", so it re-syncs rather than writing over the top."""
        source = tmp_path / "orgs.csv"
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(ORGS))
        await rag.ingest(self._write(source, [self.ACME, self.GLOBEX]))

        result = await rag.ingest(self._write(source, [self.ACME]))

        assert result.replaced_existing is True
        assert result.entities_deleted == 1
        assert await _rows(rag, "MATCH (o:Organization) RETURN o.orgs__org_id") == [["ORG-42"]]

    async def test_re_ingest_does_not_double_a_source_chunks(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """Record chunk ids derive from the effective document id, which the
        update path deliberately makes a pending one. Writing over the top
        afterwards would key the same row to a second chunk."""
        source = tmp_path / "orgs.csv"
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(ORGS))
        path = self._write(source, [self.ACME, self.GLOBEX])
        await rag.ingest(path)
        await rag.update(path)
        await rag.ingest(path)

        assert await _rows(rag, "MATCH (:Document)-[:PART_OF]->(c:Chunk) RETURN count(c)") == [[2]]

    async def test_an_unchanged_source_is_a_no_op(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        source = tmp_path / "orgs.csv"
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(ORGS))
        path = self._write(source, [self.ACME, self.GLOBEX])
        await rag.ingest(path)

        result = await rag.update(path)
        assert result.no_op is True

    async def test_a_changed_mapping_is_not_an_unchanged_source(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """Identical rows under a different declaration produce a different
        graph, so the content hash covers the mapping. Without that, adding a
        column to a mapping would be silently ignored."""
        source = tmp_path / "orgs.csv"
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(ORGS))
        path = self._write(source, [self.ACME, self.GLOBEX])
        await rag.ingest(path)

        # Same source, so the same signature: this is a re-declaration of one
        # table, not a second table describing the same rows.
        wider = TableMapping(
            source="orgs.csv",
            label="Organization",
            key="org_id",
            name="org_name",
            properties={
                "hq_country": "hq_country",
                "employee_count": Column("employee_count", "INTEGER"),
                "staff": Column("employee_count", "INTEGER"),
            },
        )
        # The mapping lives in the ontology, so changing it means changing the
        # ontology and re-ingesting -- there is nowhere else to hand a new one in.
        await rag.set_ontology(_ontology(wider))
        result = await rag.ingest(path)

        assert result.no_op is False
        assert await _rows(
            rag, "MATCH (o:Organization {orgs__org_id:'ORG-42'}) RETURN o.orgs__staff"
        ) == [[1200]]

    async def test_an_entity_another_source_still_mentions_survives(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path, employees_csv
    ):
        """Scoped cleanup: Globex leaves orgs.csv but employees.csv still points
        at it through Carol, so it must not be deleted."""
        source = tmp_path / "orgs.csv"
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(ORGS, EMPLOYEES)
        )
        await rag.ingest(self._write(source, [self.ACME, self.GLOBEX]))
        await rag.ingest(employees_csv)

        await rag.update(self._write(source, [self.ACME]))

        assert await _rows(rag, "MATCH (o:Organization {orgs__org_id:'ORG-7'}) RETURN o.name") == [
            ["Globex"]
        ], "still referenced by employees.csv, so it stays"

    async def test_a_mapping_that_stops_fitting_leaves_the_graph_alone(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        from graphrag_sdk.ingestion.mapping import MappingError

        source = tmp_path / "orgs.csv"
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(ORGS))
        path = self._write(source, [self.ACME, self.GLOBEX])
        await rag.ingest(path)

        source.write_text("something,else\n1,2\n", encoding="utf-8")
        with pytest.raises(MappingError):
            await rag.update(path)

        assert await _rows(rag, "MATCH (o:Organization) RETURN count(o)") == [[2]]
        pendings = await _rows(
            rag, "MATCH (d:Document) WHERE d.id CONTAINS 'pending' RETURN count(d)"
        )
        assert pendings == [[0]], "a rejected mapping must not leave a pending Document behind"

    async def test_update_rejects_arguments_that_cannot_apply(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(ORGS))
        source = self._write(tmp_path / "orgs.csv", [self.ACME])
        with pytest.raises(ValueError, match="does not apply"):
            await rag.update(source, chunker=object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="cache_unchanged_chunks"):
            await rag.update(source, cache_unchanged_chunks=True)
        # strict_mapping is about columns, so it is meaningless for prose -- and
        # the suffix is what decides which path a source takes.
        with pytest.raises(ValueError, match="only apply to a structured source"):
            await rag.update(str(tmp_path / "note.txt"), strict_mapping=True)

    async def test_update_can_create_a_source_it_has_never_seen(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(ORGS))
        source = self._write(tmp_path / "orgs.csv", [self.ACME])
        result = await rag.update(source, if_missing="ingest")
        assert result.replaced_existing is False
        assert await _rows(rag, "MATCH (o:Organization) RETURN o.orgs__org_id") == [["ORG-42"]]


class TestTheTableOwnsItsColumns:
    """Who wins when a document and a table disagree about a declared column.

    A mapping declares its columns so generated Cypher can see their types, but
    that also puts them in front of the extractor, which answers them from prose.
    Measured before this: Alice's title arrived as ``"engineer"`` from a memo and
    overwrote the ``"Engineer"`` the HR export spelled. The export owns that
    column — it is stored under the export's signature, and an extracted
    property is unsigned by construction — so the memo does not get to write it.
    """

    async def test_prose_cannot_overwrite_a_column_the_table_declared(
        self, real_falkordb_rag_factory, scripted_llm, resolver, employees_csv
    ):
        llm = scripted_llm(
            [("Alice Smith", "Person", "An engineer at Acme Corp")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(EMPLOYEES))
        await rag.ingest(employees_csv)
        await rag.ingest(
            text="Alice Smith is an engineer who presented the plan.",
            document_id="board_note.txt",
            resolver=resolver,
        )
        await rag.finalize()

        rows = await _rows(
            rag,
            "MATCH (p:Person {employees__employee_id:'E-1'}) "
            "RETURN p.employees__title, p.employees__age",
        )
        assert rows == [["Engineer", 34]], "the export spelled it, so the export keeps it"

    async def test_a_document_still_contributes_what_it_legitimately_knows(
        self, real_falkordb_rag_factory, scripted_llm, resolver, employees_csv
    ):
        """Only the owned column's value is discarded. The entity, its name and
        its description are exactly as extracted."""
        llm = scripted_llm(
            [("Alice Smith", "Person", "Led the remediation plan for the board")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(EMPLOYEES))
        await rag.ingest(employees_csv)
        await rag.ingest(
            text="Alice Smith led the remediation plan.",
            document_id="board_note.txt",
            resolver=resolver,
        )
        await rag.finalize()

        rows = await _rows(
            rag,
            "MATCH (p:Person {employees__employee_id:'E-1'}) RETURN p.description, p.name",
        )
        assert "remediation" in (rows[0][0] or "")
        assert rows[0][1] == "Alice Smith"

    async def test_ownership_is_scoped_to_the_declared_columns(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        """The guard is per property, not per label. ``name`` in particular is
        never owned: it is not declarable as a mapped property at all, so a
        document remains free to name an entity."""
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(EMPLOYEES))
        await rag.ingest(employees_csv)
        ontology = await rag.get_ontology()
        person = next(e for e in ontology.entities if e.label == "Person")
        owned = {p.name for p in person.properties if p.structured}
        assert owned == {"employees__employee_id", "employees__age", "employees__title"}
        assert "name" not in owned

    async def test_ownership_is_recorded_in_the_persisted_ontology(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(EMPLOYEES))
        await rag.ingest(employees_csv)
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
    declared, which is registered during the first ingest — after the client, and
    its strategies, were built.
    """

    async def test_declared_columns_reach_the_retrieval_strategy(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, enable_cypher=True, ontology=_ontology(EMPLOYEES)
        )
        await rag.ingest(employees_csv)

        strategy_ontology = rag._retrieval_strategy._ontology
        person = next(e for e in strategy_ontology.entities if e.label == "Person")
        assert {p.name for p in person.properties} >= {
            "employees__age",
            "employees__title",
            "employees__employee_id",
        }, (
            "the mapping was registered during an ingest that ran after the "
            "strategy was constructed; a stale copy cannot aggregate over them"
        )

    async def test_the_public_query_reads_the_graph(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv, employees_csv
    ):
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(ORGS, EMPLOYEES)
        )
        await rag.ingest(orgs_csv)
        await rag.ingest(employees_csv)

        rows = await rag.query(
            "MATCH (p:Person)-[r:RELATES]->(o:Organization) "
            "WHERE r.rel_type = 'WORKS_AT' AND o.name = $company "
            "RETURN avg(p.employees__age) AS mean_age, count(p) AS people",
            {"company": "Acme Corp"},
        )
        assert rows == [[39.5, 2]]

    async def test_query_returns_an_empty_list_when_nothing_matches(
        self, real_falkordb_rag_factory, llm, resolver
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology())
        assert await rag.query("MATCH (p:Person {name:'Nobody'}) RETURN p.name") == []


class TestADocumentRemembersHowItWasWritten:
    """An update may not change its mind about what a document is.

    Measured before this: ``update(path)`` on a document written from a CSV
    re-read it as prose, replaced its record chunks with one text chunk, and took
    every entity with them. Two organizations before the call, none after, and
    nothing raised. The same call arrives from ``apply_changes(modified=[...])``,
    which is how a scheduled sync would have quietly emptied a table.
    """

    async def test_a_table_is_re_synced_as_a_table_not_read_as_prose(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv
    ):
        """``update`` on a table used to be refused, because there was nowhere to
        hand it the mapping and reading a CSV as prose destroys it.

        The mapping is in the ontology now, so the suffix is enough: the re-sync
        goes down the record path and the rows keep their types. The refusal it
        replaces was a limitation, not a guarantee.
        """
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(ORGS))
        await rag.ingest(orgs_csv)

        result = await rag.update(orgs_csv)

        assert result.no_op is True, "unchanged content is a hash comparison, not a rewrite"
        assert await _rows(rag, "MATCH (o:Organization) RETURN count(o)") == [[2]]
        # Still typed, i.e. it did not come back through the text path.
        assert await _rows(
            rag, "MATCH (o:Organization {orgs__org_id:'ORG-42'}) RETURN o.orgs__employee_count"
        ) == [[1200]]

    async def test_apply_changes_re_syncs_a_table_instead_of_failing_on_it(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv
    ):
        """``apply_changes`` could never supply a mapping, so a CSV in its
        ``modified`` list came back as a ValueError and was skipped. Nothing has
        to supply one now, so a mixed batch of documents and tables just works.
        """
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(ORGS))
        await rag.ingest(orgs_csv)

        result = await rag.apply_changes(modified=[orgs_csv])

        assert result.modified[0].error_type is None, result.modified[0].error
        assert await _rows(rag, "MATCH (o:Organization) RETURN count(o)") == [[2]]

    async def test_a_table_cannot_take_over_a_document_written_from_text(
        self, real_falkordb_rag_factory, scripted_llm, resolver, orgs_csv
    ):
        """The document remembers how it was written, and a re-sync may not
        change that: the chunks and entities behind it were produced a different
        way, so writing records over them would leave the old ones orphaned.

        There is no mapping argument to refuse any more -- the source's suffix is
        what routes it -- so what is refused is the document_id.
        """
        llm = scripted_llm([("Acme Corp", "Organization", "A company")])
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(ORGS))
        await rag.ingest(text="Acme Corp exists.", document_id="note.txt", resolver=resolver)

        with pytest.raises(ValueError, match="written from text"):
            await rag.update(orgs_csv, document_id="note.txt")

    async def test_the_kind_is_persisted_on_the_document(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv
    ):
        """The guard reads this back, so it has to survive a reconnect rather than
        living only in the process that ran the ingest."""
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(ORGS))
        await rag.ingest(orgs_csv)
        record = await rag._graph_store.get_document_record("orgs.csv")
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
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(ORGS, EMPLOYEES)
        )
        await rag.ingest(orgs_csv)
        await rag.ingest(employees_csv)

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

    Which is why a mapping is now part of the ontology and is passed at
    construction: the declaration lands before the first document is read, so
    arrival order cannot decide the graph.
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
        return TableMapping(
            source="practices.csv",
            label="MitigationPractice",
            key="practice_id",
            name="practice_name",
            properties={"crop_system": "crop_system"},
            standalone=True,
        )

    async def test_declaring_a_mapping_writes_no_data(
        self, real_falkordb_rag_factory, llm, resolver
    ):
        """Declaring is a schema act. The label has to exist before any source is
        read, and nothing about the data may exist yet."""
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(self._mapping())
        )
        ontology = await rag.get_ontology()

        assert await _rows(rag, "MATCH (n) RETURN count(n)") == [[0]]
        assert any(e.label == "MitigationPractice" for e in ontology.entities)

    async def test_declaring_is_idempotent(self, real_falkordb_rag_factory, llm, resolver):
        ontology = _ontology(self._mapping())
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        await rag.get_ontology()
        await rag.set_ontology(ontology)
        reloaded = await rag.get_ontology()
        practice = next(e for e in reloaded.entities if e.label == "MitigationPractice")
        assert {p.name for p in practice.properties} == {
            "practices__practice_id",
            "practices__crop_system",
        }

    async def test_declared_mappings_are_registered_before_the_first_ingest(
        self, real_falkordb_rag_factory, scripted_llm, resolver, practices_csv
    ):
        """The whole point: prose arrives first and still lands on the declared
        label, so the halves join.

        They join with **nothing to merge**. Identity is the name for both halves,
        so the prose mention and the CSV row compute the same id and are one node
        from the first write; finalize() finds no duplicate because there never
        was one. The bridge below -- one entity reachable from both documents --
        is the proof.
        """
        llm = scripted_llm(
            [
                ("Carbon Farming", "MitigationPractice", "Umbrella set of practices"),
            ],
        )
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(self._mapping())
        )
        await rag.ingest(text=self.NOTE, document_id="note.txt", resolver=resolver)
        await rag.ingest(practices_csv)
        result = await rag.finalize()

        assert result.entities_deduplicated == 0, "the halves should never have been two nodes"
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
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(self._mapping())
        )
        await rag.ingest(text=self.NOTE, document_id="note.txt", resolver=resolver)
        await rag.ingest(practices_csv)
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
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(self._mapping())
        )
        await rag.ingest(text=self.NOTE, document_id="note.txt", resolver=resolver)
        await rag.ingest(practices_csv)
        await rag.finalize()

        rows = await _rows(
            rag,
            "MATCH (e:MitigationPractice {name:'Carbon Farming'}) "
            "RETURN e.description, e.practices__practice_id",
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
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology())
        await rag.ingest(text="Apple builds computers.", document_id="a.txt", resolver=resolver)
        await rag.ingest(text="Apple grows on trees.", document_id="b.txt", resolver=resolver)
        result = await rag.finalize()

        assert await _rows(rag, "MATCH (e:__Entity__) WHERE e.name = 'Apple' RETURN count(e)") == [
            [2]
        ], "no mapping declared either label, so neither wins"
        # Reported under the spelling the user's data actually uses, not the
        # lower-cased grouping key — the point of the report is that they can go
        # find the name.
        assert "Apple" in result.unmerged_name_collisions

    async def test_nothing_is_reported_when_labels_agree(
        self, real_falkordb_rag_factory, llm, resolver, practices_csv
    ):
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(self._mapping())
        )
        await rag.ingest(practices_csv)
        result = await rag.finalize()
        assert result.unmerged_name_collisions == {}


class TestAdoptionMovesEverythingItShould:
    """The adoption pass deletes a node, so what was on it must survive.

    Merging across labels is the one place the resolver crosses a line it
    otherwise holds, and it does so by deleting the guessed node. Anything the
    document contributed has to arrive on the survivor first.
    """

    PRACTICES = TableMapping(
        source="p.csv",
        label="Practice",
        key="pid",
        name="pname",
        properties={"crop": Column("crop")},
        standalone=True,
    )

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
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(self.PRACTICES)
        )
        await rag.ingest(practices_csv)

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
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(self.PRACTICES)
        )
        await rag.ingest(text="Carbon Farming matters.", document_id="note.txt", resolver=resolver)
        await rag.ingest(practices_csv)
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
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(self.PRACTICES)
        )
        await rag.ingest(text="Carbon Farming matters.", document_id="note.txt", resolver=resolver)
        await rag.ingest(practices_csv)

        first = await rag.finalize()
        snapshot = await _rows(rag, "MATCH (e:__Entity__) RETURN e.id ORDER BY e.id")
        second = await rag.finalize()

        assert first.entities_deduplicated >= 1
        assert second.entities_deduplicated == 0, "nothing left to merge"
        assert await _rows(rag, "MATCH (e:__Entity__) RETURN e.id ORDER BY e.id") == snapshot


class TestAdoptionOnlyCorrectsAGuessTheExtractorCouldNotAvoid:
    """Cross-label adoption exists for one case and must not fire for another.

    The case it exists for: a mapping declares a label the extractor never had
    (``MitigationPractice``), so prose read earlier filed the same thing under a
    built-in guess (``Concept``). Same thing, one source knows the type.

    The case it must not touch: the mapping declares a **built-in** label. The
    extractor always has Organization on its list, so a name it filed under
    Product instead is not a guess — it is the fruit beside the company. Adopting
    it DETACH DELETEd the fruit and wrote its description onto the supplier, on
    the default finalize() path, once mappings made declared_labels non-empty.
    """

    async def test_a_homonym_under_a_built_in_label_is_kept_and_reported(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        suppliers = tmp_path / "suppliers.csv"
        suppliers.write_text("supplier_id,supplier_name,spend_usd\nS-14,Apple,1200000\n")
        rag = real_falkordb_rag_factory(
            llm=llm,
            resolver=resolver,
            ontology=Ontology(
                entities=[Entity(label="Organization"), Entity(label="Product")],
                tables=[
                    TableMapping(
                        source="suppliers.csv",
                        label="Organization",
                        key="supplier_id",
                        name="supplier_name",
                        properties={"spend": Column("spend_usd", "INTEGER")},
                    )
                ],
            ),
        )
        await rag.query(
            "CREATE (:__Entity__:Product {id:'apple__product', name:'Apple', "
            "description:'a fruit grown in orchards'})"
        )
        await rag.ingest(str(suppliers))
        summary = await rag.finalize()

        rows = await _rows(
            rag, "MATCH (e:__Entity__ {name:'Apple'}) RETURN labels(e), e.description ORDER BY e.id"
        )
        assert len(rows) == 2, f"the fruit was merged into the company: {rows}"
        assert any("a fruit" in (desc or "") for _, desc in rows), "the fruit lost its description"
        assert "Apple" in summary.unmerged_name_collisions
        await rag.close()

    async def test_a_guess_under_a_missing_custom_label_is_still_adopted(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        practices = tmp_path / "practices.csv"
        practices.write_text("practice_id,practice\nMP-1,Carbon Farming\n")
        rag = real_falkordb_rag_factory(
            llm=llm,
            resolver=resolver,
            ontology=Ontology(
                entities=[Entity(label="Concept")],
                tables=[
                    TableMapping(
                        source="practices.csv",
                        label="MitigationPractice",
                        key="practice_id",
                        name="practice",
                        standalone=True,
                    )
                ],
            ),
        )
        await rag.query(
            "CREATE (:__Entity__:Concept {id:'carbon_farming__concept', name:'Carbon Farming', "
            "description:'a soil practice'})"
        )
        await rag.ingest(str(practices))
        await rag.finalize()

        rows = await _rows(
            rag, "MATCH (e:__Entity__ {name:'Carbon Farming'}) RETURN labels(e), e.description"
        )
        assert len(rows) == 1, f"the ordering mistake was not corrected: {rows}"
        assert "MitigationPractice" in rows[0][0], "the declared label must survive"
        assert rows[0][1] == "a soil practice", "the prose description must be carried over"
        await rag.close()


class TestOneIdentitySchemeForBothHalves:
    """An entity's id is derived from its name whether it came from a row or a
    sentence, so the two halves land on one node with no merge step. The key is
    an attribute -- ``entity_key`` -- that links and re-sync resolve through.
    """

    async def test_a_prose_mention_and_a_row_are_one_node_before_finalize(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        """No merge, no alias: the same string is the id on both sides."""
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(EMPLOYEES))
        await rag.query(
            "CREATE (:__Entity__:Person {id:'alice_smith__person', name:'Alice Smith', "
            "description:'led the platform team'})"
        )
        await rag.ingest(employees_csv)
        # deliberately NOT calling finalize()
        rows = await _rows(
            rag,
            "MATCH (p:Person {name:'Alice Smith'}) "
            "RETURN p.id, p.employees__age, p.description, p.entity_key",
        )
        assert rows == [["alice_smith__person", 34, "led the platform team", "E-1"]]
        await rag.close()

    async def test_a_pointer_arriving_after_its_target_attaches_to_it(
        self, real_falkordb_rag_factory, llm, resolver, orgs_csv, employees_csv
    ):
        """The owner first, the foreign key later: no placeholder may be raised
        beside the real node. The pointer looks the target up by key."""
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(ORGS, EMPLOYEES)
        )
        await rag.ingest(orgs_csv)
        await rag.ingest(employees_csv)
        rows = await _rows(
            rag,
            "MATCH (o:Organization {entity_key:'ORG-42'}) "
            "OPTIONAL MATCH (p:Person)-[x:RELATES]->(o) "
            "RETURN count(DISTINCT o), o.is_stub, count(p)",
        )
        assert rows == [[1, False, 2]], f"a stub was raised beside the real node: {rows}"
        await rag.close()

    async def test_a_renamed_row_keeps_its_node_and_its_prose(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """E-1 is still E-1 next quarter, spelled differently. Identity is the
        name, so the id changes -- but the node written last time carries the
        key, and it is moved to the new id rather than orphaned with the PDF's
        description still on it."""
        path = tmp_path / "employees.csv"
        path.write_text(
            "employee_id,full_name,age,job_title,org_id\nE-1,Alice Smith,34,Engineer,ORG-42\n"
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(EMPLOYEES))
        await rag.query(
            "CREATE (:__Entity__:Person {id:'alice_smith__person', name:'Alice Smith', "
            "description:'led the platform team'})"
        )
        await rag.ingest(str(path))
        path.write_text(
            "employee_id,full_name,age,job_title,org_id\nE-1,Alice Smith-Jones,35,Engineer,ORG-42\n"
        )
        await rag.ingest(str(path))

        rows = await _rows(
            rag, "MATCH (p:Person) RETURN p.id, p.name, p.employees__age, p.description"
        )
        assert rows == [
            ["alice_smith-jones__person", "Alice Smith-Jones", 35, "led the platform team"]
        ]
        await rag.close()
