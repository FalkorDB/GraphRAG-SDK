"""Structured ingestion: the cases where a load used to lose data quietly.

Every test here is a bug that reproduced against a real FalkorDB, produced a
graph that looked fine, and reported success. They are grouped together because
they share a shape rather than a code path: the SDK knew something was wrong and
did not say so. A regression in any of them is not a cosmetic one.

Requires ``RUN_INTEGRATION=1`` and FalkorDB on ``$FALKOR_HOST:$FALKOR_PORT``.
"""

from __future__ import annotations

import logging

import pytest

from graphrag_sdk import (
    Column,
    Entity,
    ExactMatchResolution,
    Link,
    Ontology,
    TableMapping,
)
from graphrag_sdk.ingestion.mapping import MappingError

from .conftest import MockLLM


@pytest.fixture
def resolver():
    return ExactMatchResolution(resolve_property="name")


class TestNothingIsDeletedWithoutSaying:
    async def test_a_table_with_no_name_column_survives_finalize(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """A fact export has no display name, and is entitled not to.

        ``finalize()`` step 1 removes NULL-name stub entities left by a legacy
        path-MERGE bug. Unscoped, it took every row of such a table with it: two
        Reading nodes before the call and none after, reported only as a
        legacy-stub count.
        """
        ontology = Ontology(
            entities=[Entity(label="Reading")],
            tables=[
                TableMapping(
                    source="readings.csv",
                    label="Reading",
                    key="reading_id",
                    properties={
                        "sensor_code": "sensor_code",
                        "value": Column("value", "FLOAT"),
                    },
                    standalone=True,
                )
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        path = tmp_path / "readings.csv"
        path.write_text("reading_id,sensor_code,value\nR-1,S-100,42.5\nR-2,S-101,17.25\n")
        await rag.ingest(str(path))
        summary = await rag.finalize()

        rows = await rag.query(
            "MATCH (n:Reading) RETURN n.readings__reading_id ORDER BY n.readings__reading_id"
        )
        assert [r[0] for r in rows] == ["R-1", "R-2"]
        assert summary.null_stubs_removed == 0
        await rag.close()

    async def test_two_rows_sharing_a_name_stay_two_entities(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """Two people called John Smith are two people.

        Dedup groups by display name, so it merged them and deleted one — a
        five-row export arriving as four people, reported as a successful dedup.
        A declared key is an assertion of identity and outranks a shared name.
        """
        ontology = Ontology(
            entities=[Entity(label="Person")],
            tables=[
                TableMapping(
                    source="people.csv",
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    properties={"age": Column("age", "INTEGER")},
                    standalone=True,
                )
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        path = tmp_path / "people.csv"
        path.write_text("employee_id,full_name,age\nE-1,John Smith,34\nE-7,John Smith,52\n")
        await rag.ingest(str(path))
        await rag.finalize()

        rows = await rag.query(
            "MATCH (p:Person) RETURN p.people__employee_id, p.people__age "
            "ORDER BY p.people__employee_id"
        )
        assert rows == [["E-1", 34], ["E-7", 52]]
        await rag.close()

    async def test_names_that_differ_only_where_the_id_flattens_them_stay_apart(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """``John Smith`` and ``John_Smith`` are two rows, and became one node.

        Collisions were looked for on the lowercased name, but the node id also
        turns spaces into underscores, so these two spelled the same id and the
        second row overwrote the first in place -- E1 gone without a word.
        Collisions are looked for on the id the rows would actually get.
        """
        ontology = Ontology(
            entities=[Entity(label="Person")],
            tables=[
                TableMapping(
                    source="people.csv",
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    standalone=True,
                )
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        path = tmp_path / "people.csv"
        path.write_text("employee_id,full_name\nE1,John Smith\nE2,John_Smith\n")
        await rag.ingest(str(path))
        await rag.finalize()

        rows = await rag.query(
            "MATCH (p:Person) RETURN p.id, p.people__employee_id, p.name "
            "ORDER BY p.people__employee_id"
        )
        assert rows == [["e1__person", "E1", "John Smith"], ["e2__person", "E2", "John_Smith"]]
        await rag.close()

    async def test_a_mention_of_a_name_two_rows_share_is_not_given_to_either(
        self, real_falkordb_rag_factory, scripted_llm, resolver, tmp_path
    ):
        """A note about "John Smith", and two rows called that.

        The two rows were kept apart, but the note's John was then folded into
        whichever of them ranked first -- by degree, description length or id,
        none of which say which John the note meant. The wrong employee got the
        note's facts and the note got his key. The mention stays its own node and
        the pair is reported for a resolver or the reader to decide.
        """
        ontology = Ontology(
            entities=[Entity(label="Person")],
            tables=[
                TableMapping(
                    source="people.csv",
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    properties={"age": Column("age", "INTEGER")},
                    standalone=True,
                )
            ],
        )
        llm = scripted_llm([("John Smith", "Person", "Presented the roadmap")])
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        path = tmp_path / "people.csv"
        path.write_text("employee_id,full_name,age\nE-1,John Smith,34\nE-7,John Smith,52\n")
        await rag.ingest(str(path))
        await rag.ingest(text="John Smith presented the roadmap.", document_id="note.txt")

        # No resolver: this is about what the name rule alone may do.
        summary = await rag.finalize(resolve=False)

        rows = await rag.query(
            "MATCH (p:Person {name:'John Smith'}) RETURN p.people__employee_id, p.description"
        )
        assert sorted(rows, key=str) == [
            ["E-1", None],
            ["E-7", None],
            [None, "Presented the roadmap"],
        ], "neither row took the note, and the note kept its own node"
        assert summary.entities_deduplicated == 0
        reported = [line for line in summary.probable_duplicates if "2 keyed rows" in line]
        assert len(reported) == 2, summary.probable_duplicates
        await rag.close()

    async def test_the_default_finalize_does_not_give_the_mention_to_either_row_either(
        self, real_falkordb_rag_factory, resolver, tmp_path
    ):
        """The same note and rows, under ``finalize()`` as shipped.

        ``resolve=False`` kept the three nodes; the default did not. The resolver
        it builds merges identical names before it asks anything, and the merge
        loop accepted the mention-to-row assignment the exact phase had just
        refused -- so the note's facts and provenance landed on one Alice Smith,
        chosen by rank. The refusal has to hold in every phase: the resolver is
        told the pairs are settled, and a merge it returns anyway is dropped.
        """
        ontology = Ontology(
            entities=[Entity(label="Person")],
            tables=[
                TableMapping(
                    source="people.csv",
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    properties={"title": "title"},
                    standalone=True,
                )
            ],
        )
        extraction = (
            '{"entities": [{"name": "Alice Smith", "type": "Person", '
            '"description": "Presented the plan"}], "relationships": []}'
        )
        # Would say YES to anything, so a merge here can only mean it was asked.
        llm = MockLLM([extraction, "YES", "YES", "YES"], strict=True)
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        path = tmp_path / "people.csv"
        path.write_text(
            "employee_id,full_name,title\nE-1,Alice Smith,Engineer\nE-2,Alice Smith,Designer\n"
        )
        await rag.ingest(str(path))
        await rag.ingest(text="Alice Smith presented the plan.", document_id="board.txt")

        summary = await rag.finalize()

        rows = await rag.query(
            "MATCH (p:Person) RETURN p.id, p.people__employee_id, p.description ORDER BY p.id"
        )
        assert rows == [
            ["alice_smith__person", None, "Presented the plan"],
            ["e-1__person", "E-1", None],
            ["e-2__person", "E-2", None],
        ], "three nodes in, three nodes out; the note is nobody's until something says whose"
        assert summary.resolved_duplicates == [] and summary.entities_deduplicated == 0
        assert llm._call_index == 1, "extraction only: the pairs were never put to the model"
        reported = [line for line in summary.probable_duplicates if "2 keyed rows" in line]
        assert len(reported) == 2, summary.probable_duplicates
        await rag.close()

    @staticmethod
    def _reporting_ontology() -> Ontology:
        return Ontology(
            entities=[Entity(label="Person")],
            tables=[
                TableMapping(
                    source="staff.csv",
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    links=[Link("REPORTS_TO", to="Person", by="manager_id", name="manager_name")],
                )
            ],
        )

    async def test_two_managers_sharing_a_name_are_told_apart_by_key(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """Alice reports to ``E1``; ``E1`` and ``E2`` are both called John.

        Two references naming the same person derived one id from the name, so
        the remap that follows a same-batch row held one entry for both, and the
        last one written won: Alice's edge went to ``E2``. Two keys under one
        name is ambiguous for a reference exactly as it is for a row, and both
        fall back to the key.
        """
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=self._reporting_ontology()
        )
        path = tmp_path / "staff.csv"
        path.write_text(
            "employee_id,full_name,manager_id,manager_name\n"
            "E1,John Park,,\n"
            "E2,John Vance,,\n"
            "E3,Alice Smith,E1,John\n"
            "E4,Bob Jones,E2,John\n"
        )
        await rag.ingest(str(path))

        rows = await rag.query(
            "MATCH (p:Person)-[r:RELATES {rel_type:'REPORTS_TO'}]->(m:Person) "
            "RETURN p.name, m.staff__employee_id ORDER BY p.name"
        )
        assert rows == [["Alice Smith", "E1"], ["Bob Jones", "E2"]]
        assert await rag.query("MATCH (p:Person) RETURN count(p)") == [[4]], (
            "no placeholder named John beside the two rows"
        )
        await rag.close()

    async def test_renaming_a_row_someone_reports_to_leaves_no_placeholder_behind(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """``E1`` Alice becomes Alice New, and Bob's ``manager_name`` says so too.

        Bob's reference was looked up in the graph by key before Alice's row was
        moved to its new id, so it -- and Bob's edge -- pointed at her *old* id;
        the write then recreated that id as a placeholder beside the renamed
        row. A reference to a row this batch writes is not looked up: the row
        is where it belongs.
        """
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=self._reporting_ontology()
        )
        path = tmp_path / "staff.csv"
        path.write_text(
            "employee_id,full_name,manager_id,manager_name\n"
            "E1,Alice Smith,,\n"
            "E2,Bob Jones,E1,Alice Smith\n"
        )
        await rag.ingest(str(path))
        path.write_text(
            "employee_id,full_name,manager_id,manager_name\n"
            "E1,Alice New,,\n"
            "E2,Bob Jones,E1,Alice New\n"
        )
        await rag.ingest(str(path))

        people = await rag.query(
            "MATCH (p:Person) RETURN p.name, p.staff__employee_id, p.is_stub ORDER BY p.name"
        )
        assert people == [["Alice New", "E1", False], ["Bob Jones", "E2", False]]
        assert await rag.query(
            "MATCH (:Person {name:'Bob Jones'})-[:RELATES {rel_type:'REPORTS_TO'}]->(m) "
            "RETURN m.name"
        ) == [["Alice New"]]
        await rag.close()

    async def test_an_edge_between_two_renamed_rows_goes_when_the_export_drops_it(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """Alice and Bob become Alicia and Robert, and Bob stops reporting to her.

        ``update()`` reads the live document's entities before the write and
        scopes its cleanup to them; the write then moved both rows to the ids
        their new names give them, so the cleanup looked for the stale edge
        between two ids that no longer existed and left it. ``Alicia REPORTS_TO
        Robert`` survived an export that said no such thing.
        """
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=self._reporting_ontology()
        )
        path = tmp_path / "staff.csv"
        path.write_text(
            "employee_id,full_name,manager_id,manager_name\n"
            "E1,Alice Smith,,\n"
            "E2,Bob Jones,E1,Alice Smith\n"
        )
        await rag.ingest(str(path))
        path.write_text(
            "employee_id,full_name,manager_id,manager_name\nE1,Alicia Smith,,\nE2,Robert Jones,,\n"
        )
        result = await rag.ingest(str(path))

        assert result.as_dict()["entities_moved"] == 2
        assert await rag.query("MATCH ()-[r:RELATES]->() RETURN count(r)") == [[0]]
        assert await rag.query("MATCH (p:Person) RETURN p.name ORDER BY p.name") == [
            ["Alicia Smith"],
            ["Robert Jones"],
        ]
        await rag.close()

    async def test_rows_without_a_key_are_counted_and_reported(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path, caplog
    ):
        """A blank key cell drops the row, which is defensible; silence is not.

        The result reported ``records: 3`` for a four-row file — true about what
        was written, false about what the file said, with nothing to tell the two
        apart. On a re-sync a newly-blank key also deletes the row's entity.
        """
        ontology = Ontology(
            entities=[Entity(label="Person")],
            tables=[
                TableMapping(
                    source="gap.csv",
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    properties={"age": Column("age", "INTEGER")},
                    standalone=True,
                )
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        path = tmp_path / "gap.csv"
        path.write_text(
            "employee_id,full_name,age\nE-1,Maya Ellison,34\n,Tomas Reyes,47\nE-3,Priya Raman,39\n"
        )
        with caplog.at_level(logging.WARNING):
            result = await rag.ingest(str(path))

        assert result.rows_skipped == 1
        assert result.rows_in_source == 3
        assert result.records == 2
        assert "rows_skipped" in result.as_dict()
        assert any("no value in the declared key" in m for m in caplog.messages)
        await rag.close()


class TestTwoTablesOwningOneLabelStayApart:
    """Two exports number the same label from 1, and neither may rename the other's rows."""

    @staticmethod
    def _ontology() -> Ontology:
        return Ontology(
            entities=[Entity(label="Person")],
            tables=[
                TableMapping(
                    source="hr.csv",
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    standalone=True,
                ),
                TableMapping(
                    source="crm.csv",
                    label="Person",
                    key="contact_id",
                    name="contact_name",
                    standalone=True,
                ),
            ],
        )

    async def test_a_shared_surrogate_key_is_not_a_shared_identity(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """``hr.csv`` row 1 is Alice; ``crm.csv`` row 1 is Bob.

        Both write ``entity_key = "1"`` under ``Person``. The re-sync that moves a
        renamed row to its new id matched on that key alone, so loading the CRM
        export renamed Alice to Bob, and re-loading HR renamed Bob back — one
        person left of two, and every load reported success.
        """
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=self._ontology())
        hr = tmp_path / "hr.csv"
        hr.write_text("employee_id,full_name\n1,Alice Smith\n")
        crm = tmp_path / "crm.csv"
        crm.write_text("contact_id,contact_name\n1,Bob Jones\n")

        await rag.ingest(str(hr))
        await rag.ingest(str(crm))
        await rag.ingest(str(hr))
        await rag.finalize(resolve=False)

        rows = await rag.query(
            "MATCH (p:Person) RETURN p.name, p.hr__employee_id, p.crm__contact_id ORDER BY p.name"
        )
        assert rows == [["Alice Smith", "1", None], ["Bob Jones", None, "1"]]
        await rag.close()

    async def test_one_person_two_tables_two_keys_is_not_a_reason_to_merge_a_third(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """HR knows Alice as 1 and Bob as 2; CRM knows Alice as 2.

        Alice's node then holds ``entity_key = "2"`` -- the slot every table
        writes, last writer wins -- next to ``hr__employee_id = "1"``. The
        re-sync used to claim any node with this table's key *present* and the
        shared key *equal*, so HR's row 2 found Alice and folded her into Bob:
        her CRM email and her record chunks moved onto him. A row is claimed by
        the value of the key as this table signs it, nothing less.
        """
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=self._ontology())
        hr = tmp_path / "hr.csv"
        hr.write_text("employee_id,full_name\n1,Alice Smith\n2,Bob Jones\n")
        crm = tmp_path / "crm.csv"
        crm.write_text("contact_id,contact_name\n2,Alice Smith\n")

        await rag.ingest(str(hr))
        await rag.ingest(str(crm))
        # A changed export, so the re-sync runs rather than short-circuiting on
        # an unchanged content hash.
        hr.write_text("employee_id,full_name\n1,Alice Smith\n2,Bob Jones\n3,Carol White\n")
        await rag.ingest(str(hr))

        rows = await rag.query(
            "MATCH (p:Person) RETURN p.name, p.hr__employee_id, p.crm__contact_id, "
            "size((p)-[:MENTIONED_IN]->()) ORDER BY p.name"
        )
        assert rows == [
            ["Alice Smith", "1", "2", 2],
            ["Bob Jones", "2", None, 1],
            ["Carol White", "3", None, 1],
        ]
        await rag.close()

    async def test_fuzzy_dedup_leaves_two_keyed_rows_alone(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path, embedder
    ):
        """Names that embed alike are still two rows when both carry a key.

        The exact phase refuses this merge; the optional embedding phase judged
        the same pair by cosine score alone and deleted one of them.
        """
        from graphrag_sdk.storage.deduplicator import EntityDeduplicator

        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=self._ontology())
        hr = tmp_path / "hr.csv"
        hr.write_text("employee_id,full_name\n1,Jon Smith\n2,John Smith\n")
        await rag.ingest(str(hr))

        # A threshold of -1 declares every pair similar, so nothing about the
        # embedder decides this: only the keyed-row guard can keep them apart.
        dedup = EntityDeduplicator(rag._graph_store, embedder)
        merged = await dedup.deduplicate(fuzzy=True, similarity_threshold=-1.0)

        assert merged == 0
        rows = await rag.query("MATCH (p:Person) RETURN p.hr__employee_id ORDER BY p.name")
        assert rows == [["2"], ["1"]]
        await rag.close()

    async def test_a_foreign_key_two_rows_answer_to_is_given_to_neither(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """HR's 1 is Alice, CRM's 1 is Bob; ``tickets.csv`` says the assignee is 1.

        The lookup that attaches a foreign key to an existing node returned one
        id per key -- whichever the graph listed last -- so the ticket landed on
        Alice or on Bob depending on nothing the caller could see, and the load
        reported success. Two matches is not a match: the edge stays on a
        placeholder that says only "Person keyed 1", and the result says so.
        """
        ontology = self._ontology()
        ontology.tables.append(
            TableMapping(
                source="tickets.csv",
                label="Ticket",
                key="ticket_id",
                name="ticket_id",
                links=[Link("ASSIGNED_TO", to="Person", by="assignee_id")],
            )
        )
        ontology.entities.append(Entity(label="Ticket"))
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        (tmp_path / "hr.csv").write_text("employee_id,full_name\n1,Alice Smith\n")
        (tmp_path / "crm.csv").write_text("contact_id,contact_name\n1,Bob Jones\n")
        (tmp_path / "tickets.csv").write_text("ticket_id,assignee_id\nT-1,1\n")
        await rag.ingest(str(tmp_path / "hr.csv"))
        await rag.ingest(str(tmp_path / "crm.csv"))

        result = await rag.ingest(str(tmp_path / "tickets.csv"))

        assert result.references_ambiguous == [("Person", "1")]
        assert result.as_dict()["references_ambiguous"] == ["Person:1"]
        rows = await rag.query(
            "MATCH (:Ticket)-[:RELATES {rel_type:'ASSIGNED_TO'}]->(p:Person) "
            "RETURN p.name, p.is_stub, p.entity_key"
        )
        assert rows == [["1", True, "1"]], f"the ticket was given to one of them: {rows}"

        # The next export of either table must not make the choice the link
        # refused to: the placeholder is not a row of hr.csv to claim.
        (tmp_path / "hr.csv").write_text("employee_id,full_name\n1,Alice Smith\n2,Carol White\n")
        resynced = await rag.ingest(str(tmp_path / "hr.csv"))
        assert resynced.entities_moved == 0
        assert await rag.query(
            "MATCH (:Ticket)-[:RELATES {rel_type:'ASSIGNED_TO'}]->(p:Person) "
            "RETURN p.name, p.is_stub"
        ) == [["1", True]], "hr.csv's re-sync took the ticket for Alice"

        # And the report survives the re-sync path, which ingest() routes
        # through update() and rebuilds from its metadata.
        (tmp_path / "tickets.csv").write_text("ticket_id,assignee_id\nT-1,1\nT-2,1\n")
        again = await rag.ingest(str(tmp_path / "tickets.csv"))
        assert again.replaced_existing and again.references_ambiguous == [("Person", "1")]
        await rag.close()

    async def test_two_rows_that_swap_names_stay_two_rows(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """A corrected export: row 1 was really Bob and row 2 really Alice.

        Each row's new id is the other's current id. Moved one at a time, the
        first found a node at its destination and was folded into it -- Alice's
        facts onto Bob, Bob's description gone -- and the second then renamed
        the merged node. One person left of two, ``entities_moved: 2``.
        """
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=self._ontology())
        hr = tmp_path / "hr.csv"
        hr.write_text("employee_id,full_name\n1,Alice Smith\n2,Bob Jones\n")
        await rag.ingest(str(hr))
        await rag.query(
            "MATCH (p:Person {id:'alice_smith__person'}) SET p.description = 'led platform' "
            "WITH p MATCH (q:Person {id:'bob_jones__person'}) SET q.description = 'led sales'"
        )

        hr.write_text("employee_id,full_name\n1,Bob Jones\n2,Alice Smith\n")
        result = await rag.ingest(str(hr))

        assert result.entities_moved == 2
        rows = await rag.query(
            "MATCH (p:Person) RETURN p.id, p.name, p.hr__employee_id, p.description ORDER BY p.id"
        )
        assert rows == [
            ["alice_smith__person", "Alice Smith", "2", "led sales"],
            ["bob_jones__person", "Bob Jones", "1", "led platform"],
        ], "the node keyed 1 is still the one keyed 1, whatever it is now called"
        await rag.close()


class TestAFailedLoadWritesNothing:
    async def test_one_bad_cell_leaves_the_graph_untouched(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """The docs promise this; the write order did not deliver it.

        The chunk pass committed the Document, a chunk per row and the content
        hash before the mapping pass ran, so a cast failure left a document made
        entirely of orphan chunks — and because the document record then existed,
        the retry routed through ``update()`` and compared hashes instead of
        writing. The graph stayed broken with nothing raised.
        """
        ontology = Ontology(
            entities=[Entity(label="Person")],
            tables=[
                TableMapping(
                    source="people.csv",
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    properties={"age": Column("age", "INTEGER")},
                    standalone=True,
                )
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        path = tmp_path / "people.csv"
        path.write_text(
            "employee_id,full_name,age\n"
            "E-1,Maya Ellison,34\n"
            "E-2,Tomas Reyes,N/A\n"
            "E-3,Priya Raman,39\n"
        )
        with pytest.raises(MappingError, match="declares INTEGER but holds 'N/A'"):
            await rag.ingest(str(path))

        for label in ("Document", "Chunk", "Person"):
            count = await rag.query(f"MATCH (n:{label}) RETURN count(n)")
            assert count[0][0] == 0, f"{label} was written before the failure"

        # The retry is a first write, not a hash comparison against a ghost.
        path.write_text(
            "employee_id,full_name,age\n"
            "E-1,Maya Ellison,34\nE-2,Tomas Reyes,47\nE-3,Priya Raman,39\n"
        )
        result = await rag.ingest(str(path))
        assert result.records == 3
        assert not result.replaced_existing
        await rag.close()


class TestEveryDeclaredColumnIsQueryable:
    async def test_a_key_named_id_keeps_its_provenance(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """``id`` is the commonest CSV header there is.

        Written verbatim it overwrote the node's graph id, which cost every row
        its MENTIONED_IN edges — no provenance and no way to join to prose — on a
        load that reported success.
        """
        ontology = Ontology(
            entities=[Entity(label="Person")],
            tables=[
                TableMapping(
                    source="people.csv",
                    label="Person",
                    key="id",
                    name="full_name",
                    properties={"age": Column("age", "INTEGER")},
                    standalone=True,
                )
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        path = tmp_path / "people.csv"
        path.write_text("id,full_name,age\n1,Maya Ellison,34\n2,Tomas Reyes,47\n")
        await rag.ingest(str(path))

        mentions = await rag.query("MATCH (:Person)-[:MENTIONED_IN]->(c:Chunk) RETURN count(c)")
        assert mentions[0][0] == 2

        rows = await rag.query(
            "MATCH (p:Person) RETURN p.id, p.people__col_id ORDER BY p.people__col_id"
        )
        assert [r[1] for r in rows] == ["1", "2"]
        assert all(r[0] not in ("1", "2") for r in rows), "graph id was overwritten"
        await rag.close()

    @pytest.mark.parametrize("header", ["HQ Country", "Revenue (M USD)", "hq-country"])
    async def test_a_header_that_is_not_an_identifier_still_loads(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path, header
    ):
        """Exports are written by other systems and their headers show it.

        A header with a space reached the driver inside a parameter map and
        surfaced as ``DatabaseError: Invalid input at end of input``, from a query
        the caller never wrote.
        """
        ontology = Ontology(
            entities=[Entity(label="Organization")],
            tables=[
                TableMapping(
                    source="orgs.csv",
                    label="Organization",
                    key="org_id",
                    name="org_name",
                    properties={"detail": Column(header)},
                    standalone=True,
                )
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        path = tmp_path / "orgs.csv"
        path.write_text(f"org_id,org_name,{header}\nORG-1,Northwind Energy,Norway\n")
        await rag.ingest(str(path))

        rows = await rag.query("MATCH (o:Organization) RETURN o.name, o.orgs__detail")
        assert rows == [["Northwind Energy", "Norway"]]
        await rag.close()

    async def test_the_key_reaches_the_ontology_under_a_queryable_name(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """The key is published to the text-to-Cypher schema block.

        Published verbatim it rendered as ``- Org ID (STRING)``, inviting
        ``WHERE o.Org ID = ...`` — not valid Cypher, on the one property every
        mapping is guaranteed to declare.
        """
        ontology = Ontology(
            entities=[Entity(label="Organization")],
            tables=[
                TableMapping(
                    source="orgs.csv",
                    label="Organization",
                    key="Org ID",
                    name="org_name",
                    standalone=True,
                )
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        path = tmp_path / "orgs.csv"
        path.write_text("Org ID,org_name\nORG-42,Northwind Energy\n")
        await rag.ingest(str(path))

        published_ontology = await rag.get_ontology()
        published = [
            attribute.name
            for entity in published_ontology.entities
            if entity.label == "Organization"
            for attribute in entity.properties
        ]
        assert published and all(" " not in name for name in published)
        rows = await rag.query("MATCH (o:Organization) RETURN o.orgs__col_org_id")
        assert rows == [["ORG-42"]]
        await rag.close()


class TestNumbersMeanWhatTheSourceSaid:
    @pytest.mark.parametrize(
        ("cell", "expected"),
        [
            ("880,5", 880.5),  # a decimal comma
            ("1.234,56", 1234.56),  # German
            ("1,234.56", 1234.56),  # US
            ("1,234,567", 1234567),  # grouped, unambiguously
            ("12 345,6", 12345.6),  # grouped with spaces
        ],
    )
    async def test_a_comma_is_not_always_a_thousands_separator(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path, cell, expected
    ):
        """Stripping every comma made a German export's 880,5 into 8805.0.

        Every figure in the column out by a factor of ten, with nothing to point
        at, on a load that raised nothing.
        """
        ontology = Ontology(
            entities=[Entity(label="Organization")],
            tables=[
                TableMapping(
                    source="eu.csv",
                    label="Organization",
                    key="org_id",
                    name="org_name",
                    properties={"revenue": Column("revenue", "FLOAT")},
                    standalone=True,
                )
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        path = tmp_path / "eu.csv"
        path.write_text(f'org_id,org_name,revenue\nORG-1,Northwind Energy,"{cell}"\n')
        await rag.ingest(str(path))
        rows = await rag.query("MATCH (o:Organization) RETURN o.eu__revenue")
        assert rows[0][0] == pytest.approx(expected)
        await rag.close()

    @pytest.mark.parametrize(
        ("cell", "expected"), [("1,000", 1000), ("1,234", 1234), ("1,234,567", 1234567)]
    )
    async def test_an_integer_column_reads_a_lone_comma_as_grouping(self, cell, expected):
        """``1,000`` is ambiguous for FLOAT and not for INTEGER.

        An integer has no fractional part on offer, so grouping is the only
        reading left and refusing it would be pedantry.
        """
        assert Column("headcount", "INTEGER").cast(cell) == expected

    async def test_a_genuinely_ambiguous_number_is_refused(self):
        """``1,234`` is one thousand or one-point-two-three-four.

        Nothing in the cell says which, so it is refused rather than guessed, and
        the message names both readings.
        """
        with pytest.raises(MappingError) as exc:
            Column("revenue", "FLOAT").cast("1,234")
        message = str(exc.value)
        assert "ambiguous" in message
        assert "'1234'" in message and "'1.234'" in message


class TestForwardReferencesStillWork:
    async def test_a_link_to_a_table_loaded_later(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """The employee export arrives before the org export it points at."""
        ontology = Ontology(
            entities=[Entity(label="Person"), Entity(label="Organization")],
            tables=[
                TableMapping(
                    source="employees.csv",
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    links=[Link("WORKS_AT", to="Organization", by="org_id")],
                ),
                TableMapping(
                    source="orgs.csv",
                    label="Organization",
                    key="org_id",
                    name="org_name",
                    properties={"employee_count": Column("employee_count", "INTEGER")},
                ),
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        employees = tmp_path / "employees.csv"
        employees.write_text("employee_id,full_name,org_id\nE-1,Maya Ellison,ORG-NW\n")
        orgs = tmp_path / "orgs.csv"
        orgs.write_text("org_id,org_name,employee_count\nORG-NW,Northwind Energy,1240\n")

        await rag.ingest(str(employees))
        await rag.ingest(str(orgs))
        await rag.finalize()

        rows = await rag.query(
            "MATCH (p:Person)-[r:RELATES]->(o:Organization) "
            "WHERE r.rel_type = 'WORKS_AT' "
            "RETURN p.name, o.name, o.orgs__employee_count"
        )
        assert rows == [["Maya Ellison", "Northwind Energy", 1240]]
        await rag.close()


class TestAdviceAboutAColumnTypeIsMeasuredNotSampled:
    """The type-narrowing warning must not recommend a type that breaks the load.

    ``properties={"age": "age"}`` is the documented shorthand and it means STRING,
    so the column this whole feature exists for arrives unaggregatable from a
    declaration that looks right. The SDK says so at ingest — but sampled, it said
    so about a column that is clean for 500 rows and holds "N/A" on row 501, and
    taking that advice makes the very next load raise MappingError. Advice that
    breaks the thing it advises about is worse than silence.
    """

    @pytest.fixture
    def resolver(self):
        return ExactMatchResolution(resolve_property="name")

    @staticmethod
    def _ontology():
        return Ontology(
            entities=[Entity(label="Person")],
            tables=[
                TableMapping(
                    source="hr.csv",
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    properties={"age": "age"},  # the shorthand: STRING
                )
            ],
        )

    async def test_a_uniformly_numeric_column_is_flagged(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path, caplog
    ):
        path = tmp_path / "hr.csv"
        path.write_text(
            "employee_id,full_name,age\n"
            + "".join(f"E-{i},Person {i},{20 + i % 40}\n" for i in range(1, 501)),
            encoding="utf-8",
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=self._ontology())
        with caplog.at_level(logging.WARNING):
            await rag.ingest(str(path))
        assert any("declared STRING hold only one narrower type" in m for m in caplog.messages)
        assert any("Column('age', 'INTEGER')" in m for m in caplog.messages), (
            "the warning has to name the fix, not just the problem"
        )
        await rag.close()

    async def test_one_unparseable_value_past_the_sample_silences_it(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path, caplog
    ):
        """Row 501 is what makes STRING the right declaration.

        Profiled over a 500-row sample this warned anyway, and INTEGER would then
        fail on the value the sample never reached.
        """
        path = tmp_path / "hr.csv"
        path.write_text(
            "employee_id,full_name,age\n"
            + "".join(f"E-{i},Person {i},{20 + i % 40}\n" for i in range(1, 501))
            + "E-501,Person 501,N/A\n",
            encoding="utf-8",
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=self._ontology())
        with caplog.at_level(logging.WARNING):
            await rag.ingest(str(path))
        assert not any(
            "declared STRING hold only one narrower type" in m for m in caplog.messages
        ), "advised a type that the whole file contradicts"
        await rag.close()
