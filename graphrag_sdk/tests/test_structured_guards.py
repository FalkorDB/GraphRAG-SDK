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

import pytest

from graphrag_sdk import Column, Entity, ExactMatchResolution, Ontology, TableMapping
from graphrag_sdk.storage.ontology_store import OntologyContradictionError

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
