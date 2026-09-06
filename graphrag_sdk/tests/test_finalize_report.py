"""What ``finalize()`` says about what did *not* happen.

Every silent failure in this feature hid in the gap between the counts a load
reported and the graph it actually produced. These fields close that gap: each
one names a condition that used to be invisible.

Requires ``RUN_INTEGRATION=1``.
"""

from __future__ import annotations

import pytest

from graphrag_sdk import (
    Column,
    Entity,
    ExactMatchResolution,
    Link,
    Ontology,
    TableMapping,
)


@pytest.fixture
def resolver():
    return ExactMatchResolution(resolve_property="name")


async def _load(rag, path, text):
    """Write ``text`` to ``path`` and ingest it.

    No mapping argument: a mapping is part of the ontology the factory was given,
    and ``ingest`` finds it by the file's basename. So every ``source=`` below
    has to be the basename of the file the test actually writes.
    """
    path.write_text(text)
    return await rag.ingest(str(path))


class TestACleanGraphReportsNothing:
    async def test_every_field_is_empty(self, real_falkordb_rag_factory, llm, resolver, tmp_path):
        """The fields must be quiet when there is nothing wrong, or they are noise
        a reader learns to skip."""
        ontology = Ontology(
            entities=[Entity(label="Person")],
            tables=[
                TableMapping(
                    source="hr.csv",
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    properties={"age": Column("age", "INTEGER")},
                ),
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        await _load(
            rag,
            tmp_path / "hr.csv",
            "employee_id,full_name,age\nE-1,Maya Ellison,34\n",
        )
        summary = await rag.finalize()

        assert summary.property_conflicts == []
        assert summary.unresolved_references == {}
        assert summary.entities_without_a_name == {}
        assert summary.tables_without_a_mapping == []
        assert summary.mapping_changed == []
        assert summary.stale_signed_properties == []
        assert summary.probable_duplicates == []
        await rag.close()


class TestPropertyConflicts:
    async def test_one_property_from_two_tables_is_reported(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """Nothing was lost — both values are signed and kept. What is reported is
        that a question asking for "the grade" now has to choose."""
        ontology = Ontology(
            entities=[Entity(label="Person")],
            tables=[
                TableMapping(
                    source=source,
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    properties={"grade": Column("grade")},
                )
                for source in ("hr.csv", "finance.csv")
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        for source, grade in (("hr.csv", "P4"), ("finance.csv", "L5")):
            await _load(
                rag,
                tmp_path / source,
                f"employee_id,full_name,grade\nE-1,Maya Ellison,{grade}\n",
            )
        summary = await rag.finalize()

        assert len(summary.property_conflicts) == 1
        reported = summary.property_conflicts[0]
        assert "Person.grade" in reported
        assert "hr.csv" in reported and "finance.csv" in reported

        # and both values really are still there
        props = (await rag.query("MATCH (p:Person) RETURN properties(p)"))[0][0]
        assert props["hr__grade"] == "P4"
        assert props["finance__grade"] == "L5"
        await rag.close()


class TestUnresolvedReferences:
    async def test_a_link_target_nobody_described_is_reported(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """The employee export points at organizations. Nothing ever loads them."""
        ontology = Ontology(
            entities=[Entity(label="Person"), Entity(label="Organization")],
            tables=[
                TableMapping(
                    source="hr.csv",
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    links=[Link("WORKS_AT", to="Organization", by="org_id")],
                ),
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        await _load(
            rag,
            tmp_path / "hr.csv",
            "employee_id,full_name,org_id\nE-1,Maya Ellison,ORG-NW\nE-2,Tomas Reyes,ORG-KG\n",
        )
        summary = await rag.finalize()
        assert summary.unresolved_references == {"Organization": 2}
        await rag.close()

    async def test_it_clears_once_the_owning_table_arrives(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        ontology = Ontology(
            entities=[Entity(label="Person"), Entity(label="Organization")],
            tables=[
                TableMapping(
                    source="hr.csv",
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
                ),
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        await _load(
            rag,
            tmp_path / "hr.csv",
            "employee_id,full_name,org_id\nE-1,Maya Ellison,ORG-NW\n",
        )
        await _load(
            rag,
            tmp_path / "orgs.csv",
            "org_id,org_name\nORG-NW,Northwind\n",
        )
        assert (await rag.finalize()).unresolved_references == {}
        await rag.close()


class TestEntitiesWithoutAName:
    async def test_a_fact_export_is_reported_not_refused(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """Legitimate, and identical in shape to a wrong ``name=`` column.

        Nothing without a name can join to prose, so it is worth seeing either
        way — but the rows survive, which is the bug this used to be.
        """
        ontology = Ontology(
            tables=[
                TableMapping(
                    source="readings.csv",
                    label="Reading",
                    key="reading_id",
                    standalone=True,
                    properties={
                        "sensor": Column("sensor"),
                        "value": Column("value", "FLOAT"),
                    },
                ),
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        await _load(
            rag,
            tmp_path / "readings.csv",
            "reading_id,sensor,value\nR-1,S-100,42.5\nR-2,S-101,17.25\n",
        )
        summary = await rag.finalize()
        assert summary.entities_without_a_name == {"Reading": 2}
        rows = await rag.query("MATCH (r:Reading) RETURN count(r)")
        assert rows[0][0] == 2, "the rows must survive finalize"
        await rag.close()


class TestTablesWithoutAMapping:
    async def test_a_derived_mapping_is_reported(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """Read by the natural reading of the file, so unjoined by design.

        Nothing declares ``employees.csv`` in the ontology, so ``ingest`` falls
        back to the natural reading of the file — which is exactly the condition
        this field exists to name.
        """
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        path = tmp_path / "employees.csv"
        await _load(rag, path, "employee_id,full_name,age\nE-1,Maya Ellison,34\n")
        summary = await rag.finalize()

        assert summary.tables_without_a_mapping == [str(path)]
        await rag.close()


class TestMappingChanged:
    async def test_a_replaced_declaration_is_reported(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """A mapping edited in code rather than deliberately should not pass
        unnoticed — its dropped properties leave the nodes with it."""
        declared = TableMapping(
            source="hr.csv",
            label="Person",
            key="employee_id",
            name="full_name",
            properties={"age": Column("age", "INTEGER"), "grade": Column("grade")},
        )
        replacement = TableMapping(
            source="hr.csv",
            label="Person",
            key="employee_id",
            name="full_name",
            properties={"age": Column("age", "INTEGER")},
        )
        rag = real_falkordb_rag_factory(
            llm=llm,
            resolver=resolver,
            ontology=Ontology(entities=[Entity(label="Person")], tables=[declared]),
        )
        path = tmp_path / "hr.csv"
        await _load(rag, path, "employee_id,full_name,age,grade\nE-1,Maya Ellison,34,P4\n")
        path.write_text("employee_id,full_name,age\nE-1,Maya Ellison,35\n")
        await rag.set_ontology(Ontology(entities=[Entity(label="Person")], tables=[replacement]))
        await rag.ingest(str(path))
        summary = await rag.finalize()

        assert summary.mapping_changed == ["hr.csv"]
        props = (await rag.query("MATCH (p:Person) RETURN properties(p)"))[0][0]
        assert "hr__grade" not in props, (
            "a property the replaced mapping no longer declares must leave the nodes"
        )
        await rag.close()

    async def test_the_same_declaration_twice_is_not_a_change(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        mapping = TableMapping(
            source="hr.csv",
            label="Person",
            key="employee_id",
            name="full_name",
            properties={"age": Column("age", "INTEGER")},
        )
        rag = real_falkordb_rag_factory(
            llm=llm,
            resolver=resolver,
            ontology=Ontology(entities=[Entity(label="Person")], tables=[mapping]),
        )
        path = tmp_path / "hr.csv"
        await _load(rag, path, "employee_id,full_name,age\nE-1,Maya Ellison,34\n")
        path.write_text("employee_id,full_name,age\nE-1,Maya Ellison,35\n")
        await rag.update(str(path))
        assert (await rag.finalize()).mapping_changed == []
        await rag.close()
