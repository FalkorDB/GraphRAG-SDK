"""Tests for storage/ontology_store.py — persistent ontology graph.

The store talks to FalkorDB directly; these unit tests mock the graph handle
through ``FalkorDBConnection._driver.select_graph()``. Real-FalkorDB exercise
lives in the integration suite.

Storage shape: three node types — ``:Entity``, ``:Relation``, ``:Property`` —
connected like a schema diagram::

    (:Entity   {label, description})
    (:Relation {label, description})
    (:Property {label, type, description})

    (:Entity)-[:HAS_PROPERTY]->(:Property)
    (:Relation)-[:SOURCE]->(:Entity)
    (:Relation)-[:TARGET]->(:Entity)
    (:Relation)-[:HAS_PROPERTY]->(:Property)

One ``:Relation`` node per declared ``(label, src, tgt)`` triple. Open-mode
relations (no patterns) become a single ``:Relation`` node with no SOURCE /
TARGET edges.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from graphrag_sdk.core.models import (
    Attribute,
    Entity,
    Ontology,
    Relation,
)
from graphrag_sdk.storage.ontology_store import (
    OntologyContradictionError,
    OntologyModificationNotAllowedError,
    OntologyStore,
)


class _FakeQueryResult:
    def __init__(self, rows):
        self.result_set = rows


class _FakeGraph:
    """Records every query call so tests can assert on the on-graph shape.

    The five canned-response slots correspond to the five MATCH queries the
    store fires in ``load()``: entities, entity-properties, patterned-relations,
    open-relations, and relation-properties.
    """

    def __init__(self):
        self.calls: list[tuple[str, dict | None]] = []
        self._entity_rows: list[list] = []
        self._entity_prop_rows: list[list] = []
        self._patterned_rel_rows: list[list] = []
        self._open_rel_rows: list[list] = []
        self._rel_prop_rows: list[list] = []

    def set_load_response(
        self,
        *,
        entities: list[list] | None = None,
        entity_properties: list[list] | None = None,
        patterned_relations: list[list] | None = None,
        open_relations: list[list] | None = None,
        relation_properties: list[list] | None = None,
    ):
        self._entity_rows = entities or []
        self._entity_prop_rows = entity_properties or []
        self._patterned_rel_rows = patterned_relations or []
        self._open_rel_rows = open_relations or []
        self._rel_prop_rows = relation_properties or []

    async def query(self, cypher: str, params: dict | None = None):
        self.calls.append((cypher, params))
        # MATCH (e:Entity) RETURN ...
        if cypher.startswith("MATCH (e:Entity) RETURN"):
            return _FakeQueryResult(self._entity_rows)
        # MATCH (e:Entity)-[:HAS_PROPERTY]->(p:Property) ...
        if "(e:Entity)-[:HAS_PROPERTY]->(p:Property)" in cypher and "RETURN" in cypher:
            return _FakeQueryResult(self._entity_prop_rows)
        # Patterned relations
        if "(r:Relation)-[:SOURCE]->(s:Entity), (r)-[:TARGET]->(t:Entity)" in cypher:
            return _FakeQueryResult(self._patterned_rel_rows)
        # Open-mode relations
        if cypher.startswith("MATCH (r:Relation) WHERE NOT (r)-[:SOURCE]->()"):
            return _FakeQueryResult(self._open_rel_rows)
        # Relation properties
        if "(r:Relation)-[:HAS_PROPERTY]->(p:Property)" in cypher and "RETURN" in cypher:
            return _FakeQueryResult(self._rel_prop_rows)
        return _FakeQueryResult([])


@pytest.fixture
def fake_graph():
    return _FakeGraph()


@pytest.fixture
def store_factory(fake_graph):
    """Returns a callable producing an ``OntologyStore`` wired to ``fake_graph``."""

    def _make(data_graph_name: str = "kg") -> OntologyStore:
        conn = MagicMock()
        conn._ensure_client = MagicMock()
        conn._driver = SimpleNamespace(select_graph=MagicMock(return_value=fake_graph))
        return OntologyStore(conn, data_graph_name)

    return _make


# ── store identity ───────────────────────────────────────────────


class TestOntologyStoreGraphName:
    def test_suffix(self, store_factory):
        store = store_factory("my_kg")
        assert store.graph_name == "my_kg__ontology"


# ── register — entity shape ──────────────────────────────────────


class TestRegisterEntityShape:
    @pytest.mark.asyncio
    async def test_entity_node_carries_label_and_description(self, store_factory, fake_graph):
        store = store_factory()
        await store.register(Ontology(entities=[Entity(label="Person", description="A human")]))
        ent_merges = [c for c in fake_graph.calls if "MERGE (e:Entity {label: $label})" in c[0]]
        assert len(ent_merges) == 1
        assert (ent_merges[0][1] or {})["label"] == "Person"
        assert (ent_merges[0][1] or {})["description"] == "A human"

    @pytest.mark.asyncio
    async def test_each_attribute_becomes_a_property_node(self, store_factory, fake_graph):
        store = store_factory()
        await store.register(
            Ontology(
                entities=[
                    Entity(
                        label="Person",
                        properties=[
                            Attribute(name="age", type="INTEGER", description="Years"),
                            Attribute(name="birth_place", type="STRING"),
                        ],
                    ),
                ]
            )
        )
        prop_merges = [
            c
            for c in fake_graph.calls
            if "MATCH (e:Entity {label: $owner})" in c[0]
            and "MERGE (e)-[:HAS_PROPERTY]->(p:Property {label: $name})" in c[0]
        ]
        assert len(prop_merges) == 2
        observed = {
            ((c[1] or {})["owner"], (c[1] or {})["name"], (c[1] or {})["type"]) for c in prop_merges
        }
        assert observed == {
            ("Person", "age", "INTEGER"),
            ("Person", "birth_place", "STRING"),
        }


# ── register — relation shape ────────────────────────────────────


class TestRegisterRelationShape:
    @pytest.mark.asyncio
    async def test_patterned_relation_creates_one_relation_node_per_pattern(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        await store.register(
            Ontology(
                entities=[
                    Entity(label="Person"),
                    Entity(label="Company"),
                    Entity(label="Organization"),
                ],
                relations=[
                    Relation(
                        label="WORKS_AT",
                        patterns=[("Person", "Company"), ("Person", "Organization")],
                        properties=[Attribute(name="since", type="DATE")],
                    ),
                ],
            )
        )
        # One Relation node MERGE per (src, tgt) pattern.
        rel_merges = [
            c
            for c in fake_graph.calls
            if "MERGE (s)<-[:SOURCE]-(r:Relation {label: $rel_label})-[:TARGET]->(t)" in c[0]
        ]
        assert len(rel_merges) == 2
        endpoints = {((c[1] or {})["src"], (c[1] or {})["tgt"]) for c in rel_merges}
        assert endpoints == {("Person", "Company"), ("Person", "Organization")}
        # One HAS_PROPERTY edge per pattern (properties attached to each
        # Relation node).
        prop_merges = [
            c
            for c in fake_graph.calls
            if "MATCH (s:Entity {label: $src})<-[:SOURCE]-(r:Relation {label: $rel_label})" in c[0]
            and "MERGE (r)-[:HAS_PROPERTY]->(p:Property {label: $name})" in c[0]
        ]
        assert len(prop_merges) == 2

    @pytest.mark.asyncio
    async def test_open_relation_creates_relation_node_with_no_endpoints(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        await store.register(
            Ontology(
                relations=[
                    Relation(
                        label="OPEN_REL",
                        description="Open relation",
                        properties=[Attribute(name="extra", type="STRING")],
                    ),
                ],
            )
        )
        open_merges = [
            c
            for c in fake_graph.calls
            if c[0].startswith("MERGE (r:Relation {label: $rel_label}) SET")
        ]
        assert len(open_merges) == 1
        # Open-mode property attaches to a Relation that lacks SOURCE.
        open_prop_merges = [
            c
            for c in fake_graph.calls
            if c[0].startswith("MATCH (r:Relation {label: $rel_label}) WHERE NOT (r)-[:SOURCE]->()")
        ]
        assert len(open_prop_merges) == 1


# ── load ─────────────────────────────────────────────────────────


class TestLoad:
    @pytest.mark.asyncio
    async def test_empty_graph_yields_empty_ontology(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_load_response()
        result = await store.load()
        assert result.entities == []
        assert result.relations == []

    @pytest.mark.asyncio
    async def test_reconstructs_entity_with_property_nodes(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_load_response(
            entities=[
                ["Person", "A human"],
                ["Company", None],
            ],
            entity_properties=[
                ["Person", "age", "INTEGER", "Years"],
                ["Person", "birth_place", "STRING", None],
            ],
        )
        ontology = await store.load()
        labels = {e.label for e in ontology.entities}
        assert labels == {"Person", "Company"}
        person = next(e for e in ontology.entities if e.label == "Person")
        assert sorted((a.name, a.type, a.description) for a in person.properties) == [
            ("age", "INTEGER", "Years"),
            ("birth_place", "STRING", None),
        ]

    @pytest.mark.asyncio
    async def test_groups_relation_nodes_by_label_into_patterns(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_load_response(
            patterned_relations=[
                ["WORKS_AT", "Employment", "Person", "Company"],
                ["WORKS_AT", "Employment", "Person", "Organization"],
                ["KNOWS", None, "Person", "Person"],
            ],
            relation_properties=[
                # Same property repeated across pattern nodes — deduped on load.
                ["WORKS_AT", "since", "DATE", None],
                ["WORKS_AT", "since", "DATE", None],
            ],
        )
        ontology = await store.load()
        by_label = {r.label: r for r in ontology.relations}
        assert set(by_label) == {"WORKS_AT", "KNOWS"}
        assert set(by_label["WORKS_AT"].patterns) == {
            ("Person", "Company"),
            ("Person", "Organization"),
        }
        assert [(a.name, a.type) for a in by_label["WORKS_AT"].properties] == [("since", "DATE")]
        assert by_label["KNOWS"].patterns == [("Person", "Person")]

    @pytest.mark.asyncio
    async def test_open_relation_loaded_with_empty_patterns(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_load_response(
            open_relations=[
                ["OPEN_REL", "Open relation"],
            ],
            relation_properties=[
                ["OPEN_REL", "extra", "STRING", None],
            ],
        )
        ontology = await store.load()
        rel = next(r for r in ontology.relations if r.label == "OPEN_REL")
        assert rel.patterns == []
        assert rel.description == "Open relation"
        assert [(a.name, a.type) for a in rel.properties] == [("extra", "STRING")]

    @pytest.mark.asyncio
    async def test_introspection_failure_returns_empty(self, store_factory, fake_graph):
        async def boom(cypher, params=None):
            raise RuntimeError("connection blew up")

        fake_graph.query = boom
        store = store_factory()
        ontology = await store.load()
        assert ontology.entities == []
        assert ontology.relations == []


# ── contradiction validation ─────────────────────────────────────


class TestContradictionDetection:
    """Type re-declarations are rejected before any partial state persists."""

    @pytest.mark.asyncio
    async def test_exact_subset_redeclaration_is_accepted(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_load_response(
            entities=[["Person", None]],
            entity_properties=[["Person", "age", "INTEGER", None]],
        )
        await store.register(
            Ontology(
                entities=[
                    Entity(
                        label="Person",
                        properties=[Attribute(name="age", type="INTEGER")],
                    ),
                ],
            )
        )
        await store.register(Ontology(entities=[Entity(label="Person")]))

    @pytest.mark.asyncio
    async def test_redefining_entity_property_type_is_rejected(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_load_response(
            entities=[["Person", None]],
            entity_properties=[["Person", "age", "INTEGER", None]],
        )
        with pytest.raises(OntologyContradictionError) as exc:
            await store.register(
                Ontology(
                    entities=[
                        Entity(
                            label="Person",
                            properties=[Attribute(name="age", type="STRING")],
                        ),
                    ],
                )
            )
        assert "Person.age" in str(exc.value)
        # Reject must fire before any upsert.
        ent_merges = [c for c in fake_graph.calls if "MERGE (e:Entity {label: $label})" in c[0]]
        assert ent_merges == []

    @pytest.mark.asyncio
    async def test_redefining_relation_property_type_is_rejected(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_load_response(
            patterned_relations=[["WORKS_AT", None, "Person", "Company"]],
            relation_properties=[["WORKS_AT", "since", "DATE", None]],
        )
        with pytest.raises(OntologyContradictionError) as exc:
            await store.register(
                Ontology(
                    relations=[
                        Relation(
                            label="WORKS_AT",
                            properties=[Attribute(name="since", type="STRING")],
                        ),
                    ],
                )
            )
        assert "WORKS_AT.since" in str(exc.value)


# ── strict modification rejection ────────────────────────────────


class TestStrictModification:
    @pytest.mark.asyncio
    async def test_adding_new_label_is_accepted(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_load_response(
            entities=[["Person", None]],
            entity_properties=[["Person", "age", "INTEGER", None]],
        )
        await store.register(
            Ontology(
                entities=[
                    Entity(
                        label="Company",
                        properties=[Attribute(name="founded", type="INTEGER")],
                    ),
                ],
            )
        )
        company_merges = [
            c
            for c in fake_graph.calls
            if "MERGE (e:Entity {label: $label})" in c[0] and (c[1] or {}).get("label") == "Company"
        ]
        assert len(company_merges) == 1

    @pytest.mark.asyncio
    async def test_adding_property_to_existing_label_is_rejected(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_load_response(
            entities=[["Person", None]],
            entity_properties=[["Person", "age", "INTEGER", None]],
        )
        with pytest.raises(OntologyModificationNotAllowedError) as exc:
            await store.register(
                Ontology(
                    entities=[
                        Entity(
                            label="Person",
                            properties=[
                                Attribute(name="age", type="INTEGER"),
                                Attribute(name="birth_date", type="DATE"),
                            ],
                        ),
                    ],
                )
            )
        assert "Person" in str(exc.value)
        assert "birth_date" in str(exc.value)
        person_merges = [
            c
            for c in fake_graph.calls
            if "MERGE (e:Entity {label: $label})" in c[0] and (c[1] or {}).get("label") == "Person"
        ]
        assert person_merges == []

    @pytest.mark.asyncio
    async def test_adding_pattern_to_existing_relation_is_rejected(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_load_response(
            patterned_relations=[["WORKS_AT", None, "Person", "Company"]],
        )
        with pytest.raises(OntologyModificationNotAllowedError) as exc:
            await store.register(
                Ontology(
                    relations=[
                        Relation(
                            label="WORKS_AT",
                            patterns=[
                                ("Person", "Company"),
                                ("Person", "Organization"),
                            ],
                        ),
                    ],
                )
            )
        assert "WORKS_AT" in str(exc.value)

    @pytest.mark.asyncio
    async def test_adding_property_to_existing_relation_is_rejected(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        fake_graph.set_load_response(
            patterned_relations=[["WORKS_AT", None, "Person", "Company"]],
            relation_properties=[["WORKS_AT", "since", "DATE", None]],
        )
        with pytest.raises(OntologyModificationNotAllowedError):
            await store.register(
                Ontology(
                    relations=[
                        Relation(
                            label="WORKS_AT",
                            properties=[
                                Attribute(name="since", type="DATE"),
                                Attribute(name="role", type="STRING"),
                            ],
                        ),
                    ],
                )
            )
