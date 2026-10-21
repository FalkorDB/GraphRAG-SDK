"""Real-FalkorDB tests for storage/ontology_store.py.

Skipped unless ``RUN_INTEGRATION=1``. FalkorDB 4.x and 6.0.x return unordered
rows in different orders; these pin that the ontology read back does not
depend on it.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.models import Attribute, Entity, Ontology, Relation
from graphrag_sdk.storage.ontology_store import OntologyStore


@pytest.fixture
async def onto():
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("Set RUN_INTEGRATION=1 to run real-FalkorDB integration tests")
    name = f"test_os_{uuid4().hex[:8]}"
    connection = FalkorDBConnection(
        ConnectionConfig(
            host=os.getenv("FALKOR_HOST", "localhost"),
            port=int(os.getenv("FALKOR_PORT", "6379")),
            username=os.getenv("FALKOR_USERNAME") or None,
            password=os.getenv("FALKOR_PASSWORD") or None,
            graph_name=name,
        )
    )
    store = OntologyStore(connection, name)
    yield store
    try:
        await store.clear()
    finally:
        await connection.close()


async def _register(onto, relation: Relation) -> None:
    await onto.register(
        Ontology(entities=[Entity(label="A"), Entity(label="B")], relations=[relation])
    )


class TestRelationDescription:
    async def test_a_later_pattern_does_not_replace_the_declared_description(self, onto):
        await _register(onto, Relation(label="R", description="orig", patterns=[("A", "B")]))

        await onto.add_relation_pattern_node("R", "B", "A", description="from a table")

        (relation,) = (await onto.load()).relations
        assert relation.description == "orig"
        assert relation.patterns == [("A", "B"), ("B", "A")]

    async def test_a_reused_node_id_does_not_make_a_new_pattern_the_oldest(self, onto):
        """FalkorDB reuses the ids of deleted nodes, so a pattern added after a
        drop can get a lower id than the declared one; id order would then
        let its description win."""
        await onto.register(
            Ontology(
                entities=[Entity(label="A"), Entity(label="B"), Entity(label="C")],
                relations=[Relation(label="TMP", patterns=[("A", "C")])],
            )
        )
        await onto.register(
            Ontology(relations=[Relation(label="R", description="orig", patterns=[("A", "B")])])
        )
        await onto.drop_relation_label("TMP")  # frees an id lower than R's

        await onto.add_relation_pattern_node("R", "B", "A", description="from a table")

        ids = (
            await onto._query(
                "MATCH (r:Relation {label: 'R'})-[:SOURCE]->(s:Entity) RETURN s.label, id(r)"
            )
        ).result_set
        by_src = dict(ids)
        assert by_src["B"] < by_src["A"], "precondition: the new pattern reused a lower id"
        (relation,) = (await onto.load()).relations
        assert relation.description == "orig"

    async def test_a_new_pattern_copies_properties_from_the_oldest_sibling(self, onto):
        await _register(
            onto,
            Relation(
                label="R",
                patterns=[("A", "A"), ("A", "B")],
                properties=[Attribute(name="p", description="shared")],
            ),
        )
        # Siblings that disagree — only reachable by hand or from legacy data.
        await onto._query(
            "MATCH (r:Relation {label: 'R'})-[:TARGET]->(:Entity {label: 'B'}) "
            "MATCH (r)-[:HAS_PROPERTY]->(p:Property {label: 'p'}) "
            "SET p.description = 'from the newer sibling', p.structured = true"
        )

        await onto.add_relation_pattern_node("R", "B", "B")

        rows = (
            await onto._query(
                "MATCH (r:Relation {label: 'R'})-[:SOURCE]->(:Entity {label: 'B'}) "
                "MATCH (r)-[:HAS_PROPERTY]->(p:Property) RETURN p.description, p.structured"
            )
        ).result_set
        assert rows == [["shared", True]]
