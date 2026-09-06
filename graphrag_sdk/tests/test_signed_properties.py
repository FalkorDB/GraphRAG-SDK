"""Signed properties: one entity, several tables, nobody overwritten.

Every structured property is written as ``<source>__<property>``, so two tables
describing the same thing cannot occupy one name. That is what makes a
disagreement between them survivable, and what makes a collision between a table
and prose impossible rather than guarded — an extracted property is unsigned by
construction.

Each test here is a measured loss from before signing existed: ``grade`` going
``"G5"`` to ``"F9"`` unrecoverably, last week's value left on a node looking
current, an edge property two tables silently fought over.

Requires ``RUN_INTEGRATION=1``.
"""

from __future__ import annotations

import pytest

from graphrag_sdk import Column, Entity, ExactMatchResolution, Link, Ontology, TableMapping
from graphrag_sdk.ingestion.mapping import record_mapping_for

HR = TableMapping(
    source="hr.csv",
    label="Person",
    key="employee_id",
    name="full_name",
    properties={"age": Column("age", "INTEGER")},
)
FINANCE = TableMapping(
    source="finance.csv",
    label="Person",
    key="employee_id",
    name="full_name",
    properties={"grade": Column("grade")},
)


def _ontology(*tables: TableMapping) -> Ontology:
    """The schema these mappings are part of.

    A mapping is a member of the ontology, not an argument to ``ingest``, and it
    has to be registered before anything is written — so every test hands the
    whole schema to the constructor. ``Person`` and ``Organization`` are named
    alongside the mappings so no mapping introduces a label the ontology has
    never heard of.
    """
    return Ontology(
        entities=[Entity(label="Person"), Entity(label="Organization")],
        tables=list(tables),
    )


@pytest.fixture
def resolver():
    return ExactMatchResolution(resolve_property="name")


async def _load(rag, path, text):
    """Write the file, then load it.

    One call: the mapping is in the ontology the rag was constructed with, and
    ``ingest`` resolves it from the file's basename — which is also what the
    property signature is derived from, so the two cannot drift apart.
    """
    path.write_text(text)
    return await rag.ingest(str(path))


class TestTwoTablesShareOneNode:
    async def test_each_signs_its_own_properties(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """Before signing, ``grade`` went "P4" -> "L5" and P4 was unrecoverable."""
        hr_mapping = TableMapping(
            source="hr.csv",
            label="Person",
            key="employee_id",
            name="full_name",
            properties={"age": Column("age", "INTEGER"), "grade": Column("grade")},
        )
        finance_mapping = TableMapping(
            source="finance.csv",
            label="Person",
            key="employee_id",
            name="full_name",
            properties={"cost_centre": Column("cost_centre"), "grade": Column("grade")},
        )
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(hr_mapping, finance_mapping)
        )
        await _load(
            rag,
            tmp_path / "hr.csv",
            "employee_id,full_name,age,grade\nE-1,Maya Ellison,34,P4\n",
        )
        await _load(
            rag,
            tmp_path / "finance.csv",
            "employee_id,full_name,cost_centre,grade\nE-1,Maya Ellison,CC-204,L5\n",
        )

        rows = await rag.query("MATCH (p:Person) RETURN properties(p)")
        assert len(rows) == 1, "both tables must write to one node"
        props = rows[0][0]
        assert props["hr__grade"] == "P4"
        assert props["finance__grade"] == "L5"
        assert props["hr__age"] == 34
        assert props["finance__cost_centre"] == "CC-204"
        assert "grade" not in props, "an unsigned name must not hold a table's value"
        assert props["name"] == "Maya Ellison", "name is identity, so unsigned"
        await rag.close()

    async def test_the_ontology_names_the_source(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """The description is load bearing, not decoration.

        The property is ``hr__age``; this description is the only thing telling
        text-to-Cypher that the word "age" means that column and that hr.csv is
        where it came from.
        """
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(HR))
        await _load(
            rag,
            tmp_path / "hr.csv",
            "employee_id,full_name,age\nE-1,Maya Ellison,34\n",
        )
        ontology = await rag.get_ontology()
        published = {
            attribute.name: attribute
            for entity in ontology.entities
            if entity.label == "Person"
            for attribute in entity.properties
        }
        assert published["hr__age"].type == "INTEGER"
        assert "hr" in (published["hr__age"].description or "")
        # the key too: unsigned here while the node carried hr__employee_id meant
        # the schema advertised a property that does not exist
        assert "hr__employee_id" in published
        await rag.close()

    async def test_a_link_still_resolves(self, real_falkordb_rag_factory, llm, resolver, tmp_path):
        """A reference stub and the table that owns the entity merge on the keyed
        id, each contributing its own signed properties."""
        rag = real_falkordb_rag_factory(
            llm=llm,
            resolver=resolver,
            ontology=_ontology(
                TableMapping(
                    source="orgs.csv",
                    label="Organization",
                    key="org_id",
                    name="org_name",
                    properties={"employee_count": Column("employee_count", "INTEGER")},
                ),
                TableMapping(
                    source="hr.csv",
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    links=[Link("WORKS_AT", to="Organization", by="org_id")],
                ),
            ),
        )
        await _load(
            rag,
            tmp_path / "orgs.csv",
            "org_id,org_name,employee_count\nORG-NW,Northwind,1240\n",
        )
        await _load(
            rag,
            tmp_path / "hr.csv",
            "employee_id,full_name,org_id\nE-1,Maya Ellison,ORG-NW\n",
        )
        rows = await rag.query(
            "MATCH (p:Person)-[r:RELATES]->(o:Organization) "
            "WHERE r.rel_type = 'WORKS_AT' "
            "RETURN p.name, o.name, o.orgs__employee_count"
        )
        assert rows == [["Maya Ellison", "Northwind", 1240]]
        await rag.close()


class TestADroppedColumnLeavesTheNodes:
    """A mapping is authoritative about its own shape.

    Measured before this: a mapping that stopped declaring ``age`` left
    ``age: 34`` on a node whose source said 35, silently.
    """

    async def test_the_property_is_removed_without_collateral_damage(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        with_grade = TableMapping(
            source="hr.csv",
            label="Person",
            key="employee_id",
            name="full_name",
            properties={"age": Column("age", "INTEGER"), "grade": Column("grade")},
        )
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=resolver, ontology=_ontology(with_grade, FINANCE)
        )
        hr_path = tmp_path / "hr.csv"
        await _load(
            rag,
            hr_path,
            "employee_id,full_name,age,grade\nE-1,Maya Ellison,34,P4\n",
        )
        await _load(
            rag,
            tmp_path / "finance.csv",
            "employee_id,full_name,grade\nE-1,Maya Ellison,L5\n",
        )
        # hr.csv stops declaring grade. The declaration lives in the ontology, so
        # dropping a column is an ontology change followed by a re-sync of that
        # one source: nothing else can hand a narrower mapping in, which is the
        # point — the stored declaration and the one being applied cannot drift.
        without_grade = TableMapping(
            source="hr.csv",
            label="Person",
            key="employee_id",
            name="full_name",
            properties={"age": Column("age", "INTEGER")},
        )
        hr_path.write_text("employee_id,full_name,age\nE-1,Maya Ellison,35\n")
        await rag.set_ontology(_ontology(without_grade, FINANCE))
        await rag.ingest(str(hr_path))

        props = (await rag.query("MATCH (p:Person) RETURN properties(p)"))[0][0]
        assert "hr__grade" not in props, "the dropped property is still on the node"
        assert props["hr__age"] == 35
        assert props["finance__grade"] == "L5", "another source's value was collateral"
        await rag.close()

    async def test_an_unchanged_mapping_drops_nothing(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        mapping = TableMapping(
            source="hr.csv",
            label="Person",
            key="employee_id",
            name="full_name",
            properties={"age": Column("age", "INTEGER"), "grade": Column("grade")},
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(mapping))
        path = tmp_path / "hr.csv"
        await _load(rag, path, "employee_id,full_name,age,grade\nE-1,Maya Ellison,34,P4\n")
        await _load(rag, path, "employee_id,full_name,age,grade\nE-1,Maya Ellison,35,P5\n")
        props = (await rag.query("MATCH (p:Person) RETURN properties(p)"))[0][0]
        assert props["hr__age"] == 35
        assert props["hr__grade"] == "P5"
        await rag.close()


class TestAStalePropertyIsReported:
    """One source stops mentioning a row another source still describes.

    The entity survives — correctly, the orphan predicate is global — but the
    first source's values stay on it belonging to nobody. Not removed: the graph
    cannot tell "that source dropped the row" from "that source has not been
    reloaded yet", and guessing would delete live data.
    """

    async def test_it_is_reported_and_the_entity_survives(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(HR, FINANCE))
        hr_path = tmp_path / "hr.csv"
        await _load(
            rag,
            hr_path,
            "employee_id,full_name,age\nE-1,Maya Ellison,34\nE-2,Tomas Reyes,47\n",
        )
        await _load(
            rag,
            tmp_path / "finance.csv",
            "employee_id,full_name,grade\nE-1,Maya Ellison,P4\n",
        )
        assert (await rag.finalize()).stale_signed_properties == []

        await _load(rag, hr_path, "employee_id,full_name,age\nE-2,Tomas Reyes,47\n")
        summary = await rag.finalize()

        assert len(summary.stale_signed_properties) == 1
        reported = summary.stale_signed_properties[0]
        assert "hr__age" in reported
        assert "hr.csv" in reported
        ids = [row[0] for row in await rag.query("MATCH (p:Person) RETURN p.id")]
        assert "maya_ellison__person" in ids, "finance still mentions it, so it must survive"
        await rag.close()

    async def test_a_single_source_graph_reports_nothing(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """A row dropped by its only source takes the entity with it, so there is
        nothing left to be stale about."""
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=_ontology(HR))
        path = tmp_path / "hr.csv"
        await _load(
            rag,
            path,
            "employee_id,full_name,age\nE-1,Maya Ellison,34\nE-2,Tomas Reyes,47\n",
        )
        await _load(rag, path, "employee_id,full_name,age\nE-2,Tomas Reyes,47\n")
        assert (await rag.finalize()).stale_signed_properties == []
        await rag.close()


class TestEdgePropertiesAreSignedToo:
    """The edge half of a mapping was writing bare property names.

    Nodes were signed; edges, forty-four lines away in the same function, were
    not. So two tables declaring one edge property overwrote each other and the
    loser's value left the graph — the exact loss signing exists to prevent,
    surviving in the one place nobody looked.
    """

    HR_EDGE = TableMapping(
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
    FIN_EDGE = TableMapping(
        source="finance.csv",
        label="Person",
        key="employee_id",
        links=[
            Link(
                "WORKS_AT",
                to="Organization",
                by="org_id",
                properties={"since": Column("start", "DATE")},
            )
        ],
    )

    def test_two_tables_declaring_one_edge_property_cannot_collide(self):
        hr = [e.signed("since") for e in record_mapping_for(self.HR_EDGE).edges]
        fin = [e.signed("since") for e in record_mapping_for(self.FIN_EDGE).edges]
        assert hr == ["hr__since"]
        assert fin == ["finance__since"]

    def test_the_ontology_advertises_the_name_the_edge_actually_carries(self):
        """A schema naming a property no edge holds sends text-to-Cypher at a
        column that returns null on every row."""
        mapping = record_mapping_for(self.HR_EDGE)
        advertised = sorted(a.name for r in mapping.to_ontology().relations for a in r.properties)
        written = sorted(e.signed("since") for e in mapping.edges)
        assert advertised == written

    def test_an_sdk_owned_edge_key_cannot_be_claimed_by_a_declaration(self):
        """``rel_type`` is how every retrieval path finds an edge's type.

        Nothing is exempt from signing on an edge, so a column declared with that
        name lands as ``hr__rel_type`` and the real one is untouched.
        """
        claim = TableMapping(
            source="hr.csv",
            label="Person",
            key="employee_id",
            links=[Link("WORKS_AT", to="Organization", by="org_id", properties={"rel_type": "x"})],
        )
        edge = record_mapping_for(claim).edges[0]
        assert edge.signed("rel_type") == "hr__rel_type"
