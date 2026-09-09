"""Guards the structured path was missing that the prose path already had.

The two paths diverged: `ingest()` checks a reserved document-id substring and
refuses to rebind an id to a different file, but the structured branch returns
before either check and `_ingest_structured` never repeated them. Both gaps were
reproduced against a live FalkorDB, and the first one destroyed data.

Every mapping here is declared in the ontology and passed at construction, which
is where a mapping now lives: ``ingest`` finds it by the ingested file's
basename, so each test's mapping ``source`` is the name of the file it writes.

Requires ``RUN_INTEGRATION=1``.
"""

from __future__ import annotations

import json

import pytest

from graphrag_sdk import Column, Entity, ExactMatchResolution, Link, Ontology, TableMapping
from graphrag_sdk.storage.ontology_store import OntologyContradictionError

from .conftest import MockLLM

ONE_ROW = "employee_id,full_name,age\nE-1,Maya Ellison,34\n"


def people(source: str) -> TableMapping:
    """The people export's mapping, bound to the file ``source`` names.

    ``ingest`` looks a mapping up by ``os.path.basename``, so the name here has
    to be the name of the file the test actually writes: a mismatch is not an
    error, it silently falls back to the natural reading of the file under a
    label derived from the filename, and the guard under test is then reached
    from the wrong place.
    """
    return TableMapping(
        source=source,
        label="Person",
        key="employee_id",
        name="full_name",
        properties={"age": Column("age", "INTEGER")},
    )


def ontology(*mappings: TableMapping) -> Ontology:
    """``Person`` plus the given table mappings.

    Passed at construction, not later: a mapping's labels and column types have
    to be registered before anything is extracted.
    """
    return Ontology(entities=[Entity(label="Person")], tables=list(mappings))


@pytest.fixture
def resolver():
    return ExactMatchResolution(resolve_property="name")


class TestAReservedIdCannotDestroyADocument:
    """``__pending__`` is the separator ``update()``'s cutover uses.

    A document id containing it is prefix-matched by ``find_pending()``, so the
    next re-sync of the real document classifies it as a leftover pending write
    and deletes it. Measured: two documents in, one document out, and the
    reported counts showed nothing.
    """

    async def test_an_explicit_id_carrying_the_marker_is_refused(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=ontology(people("hr.csv"))
        )
        path = tmp_path / "hr.csv"
        path.write_text(ONE_ROW)
        with pytest.raises(ValueError, match="__pending__"):
            await rag.ingest(str(path), document_id="hr.csv__pending__deadbeef")
        await rag.close()

    async def test_a_filename_carrying_the_marker_is_refused(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """The id defaults to the path, so the *filename* can carry it too.

        ``ingest()``'s own check only sees an explicit ``document_id``, so this
        case needs the check on the resolved id.
        """
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=ontology(people("hr__pending__x.csv"))
        )
        path = tmp_path / "hr__pending__x.csv"
        path.write_text(ONE_ROW)
        with pytest.raises(ValueError, match="__pending__"):
            await rag.ingest(str(path))
        await rag.close()

    async def test_the_real_document_survives_a_re_sync(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=ontology(people("hr.csv"))
        )
        path = tmp_path / "hr.csv"
        path.write_text(ONE_ROW)
        with pytest.raises(ValueError):
            await rag.ingest(str(path), document_id="hr.csv__pending__deadbeef")
        await rag.ingest(str(path), document_id="hr.csv")
        path.write_text("employee_id,full_name,age\nE-1,Maya Ellison,35\n")
        await rag.ingest(str(path), document_id="hr.csv")

        documents = await rag.query("MATCH (d:Document) RETURN d.id ORDER BY d.id")
        assert [row[0] for row in documents] == ["hr.csv"]
        await rag.close()


class TestAnIdStaysBoundToItsFile:
    """A stable ``document_id`` must not silently follow a different file.

    Measured: a renamed source no-opped on its unchanged content hash while
    ``Document.path`` kept pointing at the old file, and only a later unrelated
    data change happened to correct it.
    """

    async def test_a_renamed_source_cannot_rebind(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(
            llm=llm,
            resolver=resolver,
            # Both spellings are declared, so the refusal below is the id guard
            # and not a missing mapping for the renamed file.
            ontology=ontology(people("hr.csv"), people("hr_2026.csv")),
        )
        first, renamed = tmp_path / "hr.csv", tmp_path / "hr_2026.csv"
        first.write_text(ONE_ROW)
        renamed.write_text(ONE_ROW)
        await rag.ingest(str(first), document_id="hr")
        with pytest.raises(ValueError, match="already bound to path"):
            await rag.ingest(str(renamed), document_id="hr")

        rows = await rag.query("MATCH (d:Document {id:'hr'}) RETURN d.path")
        assert rows[0][0].endswith("hr.csv")
        await rag.close()

    async def test_the_ordinary_re_sync_still_works(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=ontology(people("hr.csv"))
        )
        path = tmp_path / "hr.csv"
        path.write_text(ONE_ROW)
        await rag.ingest(str(path), document_id="hr")
        path.write_text("employee_id,full_name,age\nE-1,Maya Ellison,35\n")
        result = await rag.ingest(str(path), document_id="hr")

        assert result.replaced_existing
        # ``hr__age``, not ``age``: every property a table writes is signed with
        # its source, which is what keeps two tables off one name.
        rows = await rag.query("MATCH (p:Person) RETURN p.hr__age")
        assert rows[0][0] == 35
        await rag.close()

    async def test_a_respelt_path_is_not_a_false_trip(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """``./x/hr.csv`` and ``x/hr.csv`` are one file; both sides normalise."""
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=ontology(people("hr.csv"))
        )
        folder = tmp_path / "x"
        folder.mkdir()
        path = folder / "hr.csv"
        path.write_text(ONE_ROW)
        await rag.ingest(str(path), document_id="hr")
        respelt = str(tmp_path / "." / "x" / "hr.csv")
        assert (await rag.ingest(respelt, document_id="hr")).no_op
        await rag.close()


def graded(source: str, column_type: str) -> TableMapping:
    """A people export whose ``grade`` column is declared ``column_type``."""
    return TableMapping(
        source=source,
        label="Person",
        key="employee_id",
        name="full_name",
        properties={"grade": Column("grade", column_type)},
    )


class TestOnePropertyCannotBeRetyped:
    """Pass 2 of the structured ontology registration skipped known names.

    The store's own retype check lives inside ``add_entity_property``, which that
    branch never calls, so a second declaration could redeclare one property as
    another type. Measured: ``grade`` declared INTEGER and then STRING both
    succeeded, the node held a string, and the ontology still said INTEGER —
    after which text-to-Cypher writes numeric predicates against a string column.

    Two *different* sources can no longer reach one property name at all, since
    each signs what it writes with its own source. What is left to guard is one
    source whose declaration changed type between sessions, which lands on the
    same ``hr__grade`` and is the same skipped check.
    """

    async def test_a_contradiction_is_refused(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=ontology(graded("hr.csv", "INTEGER"))
        )
        hr = tmp_path / "hr.csv"
        hr.write_text("employee_id,full_name,grade\nE-1,Maya Ellison,5\n")
        await rag.ingest(str(hr))

        # A second session over the same graph, declaring the same source's
        # grade as a STRING. The ontology it would contradict is the persisted
        # one, so this is the whole re-declaration.
        retyped = real_falkordb_rag_factory(
            llm=llm,
            resolver=resolver,
            ontology=ontology(graded("hr.csv", "STRING")),
            connection=rag._conn.config,
        )
        hr.write_text("employee_id,full_name,grade\nE-1,Maya Ellison,G5\n")
        with pytest.raises(OntologyContradictionError, match="already registered as INTEGER"):
            await retyped.ingest(str(hr))
        await retyped.close()

        persisted = await rag.get_ontology()
        types = {
            attribute.name: attribute.type
            for entity in persisted.entities
            if entity.label == "Person"
            for attribute in entity.properties
        }
        assert types["hr__grade"] == "INTEGER"
        # Refused before the write, so the node still holds the integer the
        # ontology promises — not the "G5" the retyped declaration carried.
        rows = await rag.query("MATCH (p:Person) RETURN p.hr__grade")
        assert rows[0][0] == 5
        await rag.close()

    async def test_a_refused_declaration_changes_nothing(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """The retyped declaration also drops ``title``. Neither half may land.

        The checks ran after the mapping was stored and after the columns it no
        longer named were retracted, so the rejection left the rejected mapping
        as the table's declaration and every ``title`` gone from the nodes.
        """
        with_title = TableMapping(
            source="hr.csv",
            label="Person",
            key="employee_id",
            name="full_name",
            properties={"grade": Column("grade", "INTEGER"), "title": "job_title"},
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology(with_title))
        hr = tmp_path / "hr.csv"
        hr.write_text("employee_id,full_name,grade,job_title\nE-1,Maya Ellison,5,Director\n")
        await rag.ingest(str(hr))

        retyped = real_falkordb_rag_factory(
            llm=llm,
            resolver=resolver,
            ontology=ontology(graded("hr.csv", "STRING")),
            connection=rag._conn.config,
        )
        with pytest.raises(OntologyContradictionError):
            await retyped.ingest(str(hr))
        await retyped.close()

        stored = next(t for t in (await rag.get_ontology()).tables if t.source == "hr.csv")
        assert stored.fingerprint_of_declaration == with_title.fingerprint_of_declaration, (
            "the declaration on record is the one that was accepted"
        )
        assert await rag.query("MATCH (p:Person) RETURN p.hr__title, p.hr__grade") == [
            ["Director", 5]
        ]
        await rag.close()

    async def test_two_sources_agreeing_on_a_type_is_fine(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """The common case: same property name, same type, two tables.

        Each keeps its own value under its own signed name, so agreeing on the
        type is not a contradiction and neither source's answer is lost.
        """
        rag = real_falkordb_rag_factory(
            llm=llm,
            resolver=resolver,
            ontology=ontology(graded("hr.csv", "INTEGER"), graded("finance.csv", "INTEGER")),
        )
        for name, value in (("hr.csv", "5"), ("finance.csv", "9")):
            path = tmp_path / name
            path.write_text(f"employee_id,full_name,grade\nE-1,Maya Ellison,{value}\n")
            await rag.ingest(str(path))

        rows = await rag.query("MATCH (p:Person) RETURN p.hr__grade, p.finance__grade")
        assert rows[0] == [5, 9]
        persisted = await rag.get_ontology()
        types = {
            attribute.name: attribute.type
            for entity in persisted.entities
            if entity.label == "Person"
            for attribute in entity.properties
        }
        assert types["hr__grade"] == "INTEGER"
        assert types["finance__grade"] == "INTEGER"
        await rag.close()


class TestDeleteAllTakesTheOntologyWithIt:
    """Not a bug — asserted so it stays true now that mappings live in the
    ontology, and the companion graph therefore holds them too."""

    async def test_the_ontology_graph_does_not_outlive_the_data_graph(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        from redis.asyncio import Redis

        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=ontology(people("hr.csv"))
        )
        path = tmp_path / "hr.csv"
        path.write_text(ONE_ROW)
        await rag.ingest(str(path))

        companion = rag._conn.config.graph_name + "__ontology"
        redis = Redis(host="localhost", port=6379)
        try:
            listed = {key.decode() for key in await redis.execute_command("GRAPH.LIST")}
            assert companion in listed, "nothing to assert if it was never created"
            await rag.delete_all()
            listed = {key.decode() for key in await redis.execute_command("GRAPH.LIST")}
            assert companion not in listed
        finally:
            await redis.aclose()
        await rag.close()


class TestACorrectedExportArrivesUnderItsOwnName:
    """``ingest("employees_v2.csv", document_id="employees.csv")`` is a re-sync.

    Exports arrive as ``employees_2026Q3.csv``, not by overwriting last quarter's
    file. The mapping is looked up by the explicit ``document_id`` before the
    filename, and an id that names a declared table is allowed to move to the
    new path — it is the caller saying which table this file is. Before this,
    the only way to reload a table was to copy the new file over the old name,
    which is what the notebook did.
    """

    async def test_the_table_is_re_synced_not_duplicated(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=ontology(people("employees.csv"))
        )
        first = tmp_path / "employees.csv"
        first.write_text(ONE_ROW)
        # No document_id: a declared table's Document is named after the table,
        # so the later export finds it. When the id still defaulted to the path,
        # this first call had to name the table too — measured without it, the
        # second file was read under the label `employees_v2`, a second Document
        # appeared, and Maya stayed 34.
        await rag.ingest(str(first))

        second = tmp_path / "employees_v2.csv"
        second.write_text("employee_id,full_name,age\nE-1,Maya Ellison,35\nE-2,Lene Holm,29\n")
        result = await rag.ingest(str(second), document_id="employees.csv")

        assert result.replaced_existing is True
        assert result.records == 2
        documents = await rag.query("MATCH (d:Document) RETURN d.id, d.path ORDER BY d.id")
        assert [row[0] for row in documents] == ["employees.csv"]
        assert documents[0][1] == str(second)
        # Signed with the declaration's source, not the file's name: the mapping
        # found by document_id is the one that names the properties.
        ages = await rag.query("MATCH (p:Person) RETURN p.name, p.employees__age ORDER BY p.name")
        assert ages == [["Lene Holm", 29], ["Maya Ellison", 35]]
        await rag.close()

    async def test_an_unchanged_export_still_moves_the_path(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=ontology(people("employees.csv"))
        )
        first = tmp_path / "employees.csv"
        first.write_text(ONE_ROW)
        await rag.ingest(str(first))
        second = tmp_path / "employees_v2.csv"
        second.write_text(ONE_ROW)

        result = await rag.ingest(str(second), document_id="employees.csv")

        assert result.no_op is True
        (path,) = (await rag.query("MATCH (d:Document {id: 'employees.csv'}) RETURN d.path"))[0]
        assert path == str(second)
        await rag.close()

    async def test_an_id_with_no_declaration_behind_it_still_refuses_to_rebind(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """The prose path's guard is kept for a table nobody declared."""
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology())
        first = tmp_path / "readings.csv"
        first.write_text("reading_id,value\nR-1,10\n")
        await rag.ingest(str(first), document_id="readings")
        second = tmp_path / "readings_v2.csv"
        second.write_text("reading_id,value\nR-1,11\n")
        with pytest.raises(ValueError, match="refusing to rebind"):
            await rag.ingest(str(second), document_id="readings")
        await rag.close()


class TestATableDoesNotRewriteWhatALabelMeans:
    """Registering a table keeps the description the user gave its label.

    The fragment a mapping contributes describes an existing label only by what
    the table did ("Declared by a structured source, keyed on employee_id"), and
    the store's coalesce took that over the declared text. Measured on a corpus
    with twelve tables: every declared description was gone after the first
    load, and a model asked to place an undeclared grants table chose Experiment
    because Experiment now read "keyed on exp_id" rather than "a field experiment
    measuring methane flux". The extractor and text-to-Cypher read the same
    descriptions.
    """

    async def test_the_declared_description_survives_the_table(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        declared = Ontology(
            entities=[
                Entity(label="Person", description="A researcher or lab member"),
                Entity(label="Organization", description="A university or funder"),
            ],
            tables=[
                TableMapping(
                    source="hr.csv",
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    properties={"age": Column("age", "INTEGER")},
                    links=[Link("WORKS_AT", to="Organization", by="org")],
                )
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=declared)
        path = tmp_path / "hr.csv"
        path.write_text("employee_id,full_name,age,org\nE-1,Maya Ellison,34,Acme\n")
        await rag.ingest(str(path))

        stored = {e.label: e.description for e in (await rag._ontology_store.load()).entities}
        assert stored["Person"] == "A researcher or lab member", "the row's own label"
        assert stored["Organization"] == "A university or funder", "the link's target"
        await rag.close()

    async def test_a_label_nobody_described_takes_the_tables_note(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=ontology(people("hr.csv"))
        )
        path = tmp_path / "hr.csv"
        path.write_text(ONE_ROW)
        await rag.ingest(str(path))

        stored = {e.label: e.description for e in (await rag._ontology_store.load()).entities}
        assert stored["Person"] == "Declared by a structured source, keyed on employee_id"
        await rag.close()


class TestATableIsAddressedByItsName:
    """A table's Document id is the basename of its declaration's ``source``.

    A document is addressed by its path because the path is all there is to
    know about it. A table has a declaration, and exports move: keyed on the
    path, ``/exports/2026-02/hr.csv`` was a second Document under the same label
    with January's left behind reading as current. Keyed on the table's name it
    is a re-sync of the table, which is what a new export of a known table is.
    """

    async def test_the_document_is_named_after_the_table(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=ontology(people("hr.csv"))
        )
        path = tmp_path / "exports" / "2026-01" / "hr.csv"
        path.parent.mkdir(parents=True)
        path.write_text(ONE_ROW)
        result = await rag.ingest(str(path))

        assert result.document_id == "hr.csv"
        rows = await rag.query("MATCH (d:Document) RETURN d.id, d.path")
        assert rows == [["hr.csv", str(path)]]
        await rag.close()

    async def test_a_new_export_from_another_folder_is_a_re_sync(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=ontology(people("hr.csv"))
        )
        january = tmp_path / "2026-01" / "hr.csv"
        february = tmp_path / "2026-02" / "hr.csv"
        for path in (january, february):
            path.parent.mkdir()
        january.write_text(ONE_ROW)
        february.write_text("employee_id,full_name,age\nE-1,Maya Ellison,35\n")
        await rag.ingest(str(january))
        result = await rag.ingest(str(february))

        assert result.replaced_existing is True
        documents = await rag.query("MATCH (d:Document) RETURN d.id, d.path")
        assert documents == [["hr.csv", str(february)]]
        assert (await rag.query("MATCH (p:Person) RETURN p.hr__age")) == [[35]]
        await rag.close()

    async def test_update_and_delete_use_the_same_name(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=ontology(people("hr.csv"))
        )
        path = tmp_path / "hr.csv"
        path.write_text(ONE_ROW)
        await rag.ingest(str(path))
        path.write_text("employee_id,full_name,age\nE-1,Maya Ellison,36\n")
        updated = await rag.update(str(path))
        assert updated.no_op is False
        assert (await rag.query("MATCH (p:Person) RETURN p.hr__age")) == [[36]]

        deleted = await rag.delete_document("hr.csv")
        assert deleted.chunks_deleted == 1
        assert (await rag.query("MATCH (d:Document) RETURN count(d)")) == [[0]]
        await rag.close()

    async def test_an_explicit_id_is_still_honoured(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=ontology(people("hr.csv"))
        )
        path = tmp_path / "hr.csv"
        path.write_text(ONE_ROW)
        result = await rag.ingest(str(path), document_id="people-2026")
        assert result.document_id == "people-2026"
        await rag.close()


PROPOSAL = json.dumps(
    {
        "label": "Person",
        "name": "full_name",
        "key": "employee_id",
        "properties": [{"column": "age", "type": "INTEGER"}],
        "links": [],
        "reasoning": "Each row is an employee, a person.",
    }
)


class TestATableNobodyDeclaredGetsAProposedMapping:
    """``ingest("employees.csv")`` with no declaration asks the model — once.

    The proposal is checked against the file, stored in the ontology as
    ``derived``, used by every later load, reported by ``finalize()``, and
    replaced the moment a ``TableMapping`` is declared for the source. Refusing
    it is ``drop_table()``.
    """

    async def test_the_rows_land_on_the_label_the_model_chose(
        self, real_falkordb_rag_factory, resolver, tmp_path
    ):
        llm = MockLLM([PROPOSAL], strict=True)
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology())
        path = tmp_path / "employees.csv"
        path.write_text(ONE_ROW)
        result = await rag.ingest(str(path))

        assert result.document_id == "employees.csv"
        assert llm._call_index == 1, "one call for the whole table, none per row"
        # Joined by name, exactly as a declared mapping would have: the id is the
        # one a prose mention of "Maya Ellison" computes.
        rows = await rag.query("MATCH (p:Person) RETURN p.id, p.name, p.employees__age")
        assert rows == [["maya_ellison__person", "Maya Ellison", 34]]

        stored = [m for m in (await rag._ontology_store.load()).tables]
        assert [(m.source, m.label, m.derived) for m in stored] == [
            ("employees.csv", "Person", True)
        ]
        summary = await rag.finalize()
        assert summary.proposed_mappings == ["employees.csv"]
        await rag.close()

    async def test_a_later_load_reuses_the_proposal_without_asking_again(
        self, real_falkordb_rag_factory, resolver, tmp_path
    ):
        llm = MockLLM([PROPOSAL], strict=True)
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology())
        path = tmp_path / "employees.csv"
        path.write_text(ONE_ROW)
        await rag.ingest(str(path))
        path.write_text("employee_id,full_name,age\nE-1,Maya Ellison,35\n")

        # A fresh GraphRAG on the same graph: the proposal is in the ontology,
        # not in this process.
        again = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=ontology(), connection=rag._conn
        )
        result = await again.ingest(str(path))

        assert result.replaced_existing is True
        assert llm._call_index == 1
        assert (await again.query("MATCH (p:Person) RETURN p.employees__age")) == [[35]]
        await rag.close()
        await again.close()

    async def test_a_model_that_cannot_answer_falls_back_to_reading_the_file_as_is(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """The default MockLLM answers an extraction shape, never a mapping."""
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology())
        path = tmp_path / "readings.csv"
        path.write_text("reading_id,value\nR-1,10\n")
        await rag.ingest(str(path))

        (mapping,) = (await rag._ontology_store.load()).tables
        assert mapping.label == "readings" and mapping.name is None and mapping.derived
        assert (await rag.query("MATCH (r:readings) RETURN r.readings__value")) == [[10]]
        await rag.close()

    async def test_a_declaration_replaces_the_proposal(
        self, real_falkordb_rag_factory, resolver, tmp_path
    ):
        """The user's own mapping wins, however the path is spelled, and the
        stored proposal does not linger beside it as a second table."""
        llm = MockLLM([PROPOSAL], strict=True)
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology())
        path = tmp_path / "employees.csv"
        path.write_text(ONE_ROW)
        await rag.ingest(str(path))

        declared = TableMapping(
            source=str(path),  # the full path, not the basename the proposal used
            label="Person",
            key="employee_id",
            name="full_name",
            properties={"age_years": Column("age", "INTEGER")},
        )
        again = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=ontology(declared), connection=rag._conn
        )
        result = await again.ingest(str(path))

        assert result.replaced_existing is True, "same table, so a re-sync"
        stored = (await again._ontology_store.load()).tables
        assert [(m.source, m.derived) for m in stored] == [(str(path), False)]
        rows = await again.query("MATCH (p:Person) RETURN p.employees__age_years, p.employees__age")
        assert rows == [[34, None]], "the proposal's property left with it"
        assert (await again.finalize()).proposed_mappings == []
        await rag.close()
        await again.close()

    async def test_drop_table_takes_the_rows_the_values_and_the_mapping(
        self, real_falkordb_rag_factory, resolver, tmp_path
    ):
        # Call 1 is the note's extraction, call 2 the table's proposal.
        llm = MockLLM(['{"nodes": [], "relationships": []}', PROPOSAL], strict=True)
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology())
        # A person the documents also know, so she survives the drop.
        await rag.ingest(text="Maya Ellison joined the board.", document_id="note.txt")
        await rag.query(
            "MERGE (p:Person:__Entity__ {id: 'maya_ellison__person'}) "
            "SET p.name = 'Maya Ellison' "
            "WITH p MATCH (c:Chunk) MERGE (p)-[:MENTIONED_IN]->(c)"
        )
        path = tmp_path / "employees.csv"
        path.write_text("employee_id,full_name,age\nE-1,Maya Ellison,34\nE-2,Lene Holm,29\n")
        await rag.ingest(str(path))
        assert (await rag.query("MATCH (p:Person) RETURN count(p)")) == [[2]]

        ontology_after = await rag.drop_table("./anywhere/employees.csv")

        assert ontology_after.tables == []
        assert (await rag.query("MATCH (d:Document) RETURN d.id")) == [["note.txt"]]
        people_left = await rag.query(
            "MATCH (p:Person) RETURN p.name, p.employees__age, p.employees__employee_id, "
            "p.entity_key, p.is_stub"
        )
        assert people_left == [["Maya Ellison", None, None, None, None]], (
            "kept by the note, minus the table's value, its key and the identity it gave her"
        )
        person = next(e for e in ontology_after.entities if e.label == "Person")
        declared = {prop.name for prop in person.properties}
        assert "employees__age" not in declared
        assert "employees__employee_id" not in declared, "the key column leaves with the table"
        with pytest.raises(ValueError, match="No table named 'employees.csv'"):
            await rag.drop_table("employees.csv")
        await rag.close()
