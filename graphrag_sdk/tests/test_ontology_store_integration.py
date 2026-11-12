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

from .conftest import falkordb_test_server


@pytest.fixture
async def onto():
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("Set RUN_INTEGRATION=1 to run real-FalkorDB integration tests")
    name = f"test_os_{uuid4().hex[:8]}"
    connection = FalkorDBConnection(
        ConnectionConfig(
            **falkordb_test_server(),
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


@pytest.fixture
async def onto_and_data():
    """A store and its data graph, both dropped afterwards."""
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("Set RUN_INTEGRATION=1 to run real-FalkorDB integration tests")
    name = f"test_os_{uuid4().hex[:8]}"
    connection = FalkorDBConnection(ConnectionConfig(**falkordb_test_server(), graph_name=name))
    store = OntologyStore(connection, name)
    yield store, connection
    try:
        await store.clear()
    finally:
        try:
            await connection.delete_graph()
        finally:
            await connection.close()


class TestAddingToALabelWithNoData:
    """The data check runs against a real data graph, missing graph included."""

    async def test_a_bare_label_takes_attributes_while_no_node_has_it(self, onto_and_data):
        onto, _ = onto_and_data
        await onto.register(Ontology(entities=[Entity(label="Event")]))  # as the seed writes it

        await onto.register(
            Ontology(
                entities=[
                    Entity(
                        label="Event",
                        properties=[
                            Attribute(name="event_type", type="STRING"),
                            Attribute(name="timestamp", type="DATE"),
                        ],
                    )
                ]
            )
        )

        (event,) = (await onto.load()).entities
        assert sorted(p.name for p in event.properties) == ["event_type", "timestamp"]

    async def test_a_label_with_a_node_is_refused(self, onto_and_data):
        from graphrag_sdk.storage.ontology_store import OntologyModificationNotAllowedError

        onto, connection = onto_and_data
        await onto.register(Ontology(entities=[Entity(label="Event")]))
        await connection.query("CREATE (:Event {name: 'Launch'})")

        with pytest.raises(OntologyModificationNotAllowedError):
            await onto.register(
                Ontology(
                    entities=[
                        Entity(label="Event", properties=[Attribute(name="when", type="DATE")])
                    ]
                )
            )

    async def test_a_relation_with_a_relates_edge_is_refused(self, onto_and_data):
        from graphrag_sdk.storage.ontology_store import OntologyModificationNotAllowedError

        onto, connection = onto_and_data
        await _register(onto, Relation(label="R", patterns=[("A", "B")]))
        await connection.query(
            "CREATE (:__Entity__:A {name: 'a'})-[:RELATES {rel_type: 'R'}]->(:__Entity__:B {name: 'b'})"
        )

        with pytest.raises(OntologyModificationNotAllowedError):
            await onto.register(
                Ontology(relations=[Relation(label="R", patterns=[("A", "B"), ("B", "A")])])
            )

    async def test_a_relation_with_a_typed_edge_is_refused(self, onto_and_data):
        """backfill_relation_pattern writes the relation type itself, not RELATES."""
        from graphrag_sdk.storage.ontology_store import OntologyModificationNotAllowedError

        onto, connection = onto_and_data
        await _register(onto, Relation(label="R", patterns=[("A", "B")]))
        await connection.query("CREATE (:__Entity__:A {name: 'a'})-[:R]->(:__Entity__:B {name: 'b'})")

        with pytest.raises(OntologyModificationNotAllowedError):
            await onto.register(
                Ontology(relations=[Relation(label="R", patterns=[("A", "B"), ("B", "A")])])
            )

    async def test_a_relation_with_no_edges_takes_a_pattern(self, onto_and_data):
        onto, connection = onto_and_data
        await _register(onto, Relation(label="R", patterns=[("A", "B")]))
        await connection.query(
            "CREATE (:__Entity__:A {name: 'a'})-[:RELATES {rel_type: 'OTHER'}]->(:__Entity__:B {name: 'b'})"
        )

        await onto.register(Ontology(relations=[Relation(label="R", patterns=[("A", "B"), ("B", "A")])]))

        (relation,) = (await onto.load()).relations
        assert sorted(relation.patterns) == [("A", "B"), ("B", "A")]


class TestStructureEdgesAreNotRelationData:
    """Every ingested document writes Document-[:PART_OF]->Chunk; a declared
    PART_OF relation with no edges of its own still holds no data."""

    async def test_a_part_of_relation_with_only_structure_edges_takes_a_pattern(
        self, onto_and_data
    ):
        onto, connection = onto_and_data
        await _register(onto, Relation(label="PART_OF", patterns=[("Organ", "BodySystem")]))
        await connection.query(
            "CREATE (:Document {id: 'd'})-[:PART_OF]->(:Chunk {id: 'c'}), "
            "(:Chunk {id: 'c2'})-[:NEXT_CHUNK]->(:Chunk {id: 'c3'})"
        )

        await onto.register(
            Ontology(
                relations=[
                    Relation(
                        label="PART_OF",
                        patterns=[("Organ", "BodySystem"), ("Cell", "Organ")],
                    )
                ]
            )
        )

        (relation,) = (await onto.load()).relations
        assert sorted(relation.patterns) == [("Cell", "Organ"), ("Organ", "BodySystem")]


class TestAFirstOntologyOverTheDefaultSeed:
    """GraphRAG-Server, in production: a new graph, read once (which seeds the
    eleven default labels), then a first ontology whose ``Event`` has
    attributes. It was refused: "Refusing to add new attribute(s)
    ['event_type', 'timestamp'] to existing label 'Event'"."""

    async def test_set_ontology_gives_the_seeded_event_its_attributes(
        self, real_falkordb_rag_factory, llm
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=None)
        try:
            seeded = await rag.get_ontology()
            assert "Event" in {e.label for e in seeded.entities}, "precondition: the seed ran"

            ontology = await rag.set_ontology(
                Ontology(
                    entities=[
                        Entity(
                            label="Event",
                            properties=[
                                Attribute(name="event_type", type="STRING"),
                                Attribute(name="timestamp", type="DATE"),
                            ],
                        )
                    ]
                )
            )

            (event,) = [e for e in ontology.entities if e.label == "Event"]
            assert {"event_type", "timestamp"} <= {p.name for p in event.properties}
        finally:
            await rag._ontology_store.clear()


class TestAnAttributeOnlyDeclarationOfAPatternedRelation:
    """``Relation(label, properties=[...])`` with no patterns, on a stored
    patterned relation with no edges: the attribute must reach that pattern,
    not an open-mode node the load never reads for it."""

    async def test_the_attribute_lands_on_the_stored_pattern(self, onto_and_data):
        onto, _ = onto_and_data
        await _register(onto, Relation(label="R", patterns=[("A", "B")]))

        await onto.register(
            Ontology(relations=[Relation(label="R", properties=[Attribute(name="role", type="STRING")])])
        )

        (relation,) = (await onto.load()).relations
        assert relation.patterns == [("A", "B")]
        assert [p.name for p in relation.properties] == ["role"]
