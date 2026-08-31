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

from graphrag_sdk import Column, ExactMatchResolution, Link, Table
from graphrag_sdk.ingestion.mapping import MappingError


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
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        path = tmp_path / "readings.csv"
        path.write_text("reading_id,sensor_code,value\nR-1,S-100,42.5\nR-2,S-101,17.25\n")
        mapping = Table(
            "Reading",
            key="reading_id",
            sensor_code="sensor_code",
            value=Column("value", "FLOAT"),
        )
        await rag.ingest(str(path), mapping=mapping)
        summary = await rag.finalize()

        rows = await rag.query("MATCH (n:Reading) RETURN n.reading_id ORDER BY n.reading_id")
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
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        path = tmp_path / "people.csv"
        path.write_text(
            "employee_id,full_name,age\nE-1,John Smith,34\nE-7,John Smith,52\n"
        )
        mapping = Table(
            "Person", key="employee_id", name="full_name", age=Column("age", "INTEGER")
        )
        await rag.ingest(str(path), mapping=mapping)
        await rag.finalize()

        rows = await rag.query(
            "MATCH (p:Person) RETURN p.employee_id, p.age ORDER BY p.employee_id"
        )
        assert rows == [["E-1", 34], ["E-7", 52]]
        await rag.close()

    async def test_rows_without_a_key_are_counted_and_reported(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path, caplog
    ):
        """A blank key cell drops the row, which is defensible; silence is not.

        The result reported ``records: 3`` for a four-row file — true about what
        was written, false about what the file said, with nothing to tell the two
        apart. On a re-sync a newly-blank key also deletes the row's entity.
        """
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        path = tmp_path / "gap.csv"
        path.write_text(
            "employee_id,full_name,age\n"
            "E-1,Maya Ellison,34\n"
            ",Tomas Reyes,47\n"
            "E-3,Priya Raman,39\n"
        )
        mapping = Table(
            "Person", key="employee_id", name="full_name", age=Column("age", "INTEGER")
        )
        with caplog.at_level(logging.WARNING):
            result = await rag.ingest(str(path), mapping=mapping)

        assert result.rows_skipped == 1
        assert result.rows_in_source == 3
        assert result.records == 2
        assert "rows_skipped" in result.as_dict()
        assert any("no value in the declared key" in m for m in caplog.messages)
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
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        path = tmp_path / "people.csv"
        path.write_text(
            "employee_id,full_name,age\n"
            "E-1,Maya Ellison,34\n"
            "E-2,Tomas Reyes,N/A\n"
            "E-3,Priya Raman,39\n"
        )
        mapping = Table(
            "Person", key="employee_id", name="full_name", age=Column("age", "INTEGER")
        )
        with pytest.raises(MappingError, match="declares INTEGER but holds 'N/A'"):
            await rag.ingest(str(path), mapping=mapping)

        for label in ("Document", "Chunk", "Person"):
            count = await rag.query(f"MATCH (n:{label}) RETURN count(n)")
            assert count[0][0] == 0, f"{label} was written before the failure"

        # The retry is a first write, not a hash comparison against a ghost.
        path.write_text(
            "employee_id,full_name,age\n"
            "E-1,Maya Ellison,34\nE-2,Tomas Reyes,47\nE-3,Priya Raman,39\n"
        )
        result = await rag.ingest(str(path), mapping=mapping)
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
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        path = tmp_path / "people.csv"
        path.write_text("id,full_name,age\n1,Maya Ellison,34\n2,Tomas Reyes,47\n")
        mapping = Table("Person", key="id", name="full_name", age=Column("age", "INTEGER"))
        await rag.ingest(str(path), mapping=mapping)

        mentions = await rag.query(
            "MATCH (:Person)-[:MENTIONED_IN]->(c:Chunk) RETURN count(c)"
        )
        assert mentions[0][0] == 2

        rows = await rag.query("MATCH (p:Person) RETURN p.id, p.col_id ORDER BY p.col_id")
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
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        path = tmp_path / "orgs.csv"
        path.write_text(f"org_id,org_name,{header}\nORG-1,Northwind Energy,Norway\n")
        mapping = Table(
            "Organization", key="org_id", name="org_name", detail=Column(header)
        )
        await rag.ingest(str(path), mapping=mapping)

        rows = await rag.query("MATCH (o:Organization) RETURN o.name, o.detail")
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
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        path = tmp_path / "orgs.csv"
        path.write_text("Org ID,org_name\nORG-42,Northwind Energy\n")
        await rag.ingest(str(path), mapping=Table("Organization", key="Org ID", name="org_name"))

        ontology = await rag.get_ontology()
        published = [
            attribute.name
            for entity in ontology.entities
            if entity.label == "Organization"
            for attribute in entity.properties
        ]
        assert published and all(" " not in name for name in published)
        rows = await rag.query("MATCH (o:Organization) RETURN o.col_org_id")
        assert rows == [["ORG-42"]]
        await rag.close()


class TestNumbersMeanWhatTheSourceSaid:
    @pytest.mark.parametrize(
        ("cell", "expected"),
        [
            ("880,5", 880.5),        # a decimal comma
            ("1.234,56", 1234.56),   # German
            ("1,234.56", 1234.56),   # US
            ("1,234,567", 1234567),  # grouped, unambiguously
            ("12 345,6", 12345.6),   # grouped with spaces
        ],
    )
    async def test_a_comma_is_not_always_a_thousands_separator(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path, cell, expected
    ):
        """Stripping every comma made a German export's 880,5 into 8805.0.

        Every figure in the column out by a factor of ten, with nothing to point
        at, on a load that raised nothing.
        """
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        path = tmp_path / "eu.csv"
        path.write_text(f'org_id,org_name,revenue\nORG-1,Northwind Energy,"{cell}"\n')
        mapping = Table(
            "Organization",
            key="org_id",
            name="org_name",
            revenue=Column("revenue", "FLOAT"),
        )
        await rag.ingest(str(path), mapping=mapping)
        rows = await rag.query("MATCH (o:Organization) RETURN o.revenue")
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
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        employees = tmp_path / "employees.csv"
        employees.write_text(
            "employee_id,full_name,org_id\nE-1,Maya Ellison,ORG-NW\n"
        )
        orgs = tmp_path / "orgs.csv"
        orgs.write_text("org_id,org_name,employee_count\nORG-NW,Northwind Energy,1240\n")

        await rag.ingest(
            str(employees),
            mapping=Table(
                "Person",
                key="employee_id",
                name="full_name",
                links=[Link("WORKS_AT", to="Organization", by="org_id")],
            ),
        )
        await rag.ingest(
            str(orgs),
            mapping=Table(
                "Organization",
                key="org_id",
                name="org_name",
                employee_count=Column("employee_count", "INTEGER"),
            ),
        )
        await rag.finalize()

        rows = await rag.query(
            "MATCH (p:Person)-[r:RELATES]->(o:Organization) "
            "WHERE r.rel_type = 'WORKS_AT' "
            "RETURN p.name, o.name, o.employee_count"
        )
        assert rows == [["Maya Ellison", "Northwind Energy", 1240]]
        await rag.close()
