"""Tests for storage/ontology_store.py — persistent ontology graph.

The store talks to FalkorDB directly; these unit tests mock the graph handle
through ``FalkorDBConnection._driver.select_graph()``. Real-FalkorDB exercise
lives in the integration suite.

Storage shape (Option B):
- Entity types are nodes carrying both the user label and ``:__Ontology``,
  with a JSON-encoded ``attributes`` property.
- Relation types are real edges between entity-type nodes, one edge per
  declared pattern, carrying their own JSON-encoded ``attributes``.
- Open-mode relations (no patterns) become self-loops on a
  ``:__OpenRelation:__Ontology`` placeholder.
"""

from __future__ import annotations

import json
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
    _decode_attributes,
    _encode_attributes,
)


class _FakeQueryResult:
    def __init__(self, rows):
        self.result_set = rows


class _FakeGraph:
    """Records every query call so tests can assert on the on-graph shape.

    Serves canned responses for the three load queries (entities, patterned
    relations, open relations); everything else returns an empty result set.
    """

    def __init__(self):
        self.calls: list[tuple[str, dict | None]] = []
        self._ent_rows: list[list] = []
        self._rel_rows: list[list] = []
        self._open_rel_rows: list[list] = []

    def set_load_response(
        self,
        entity_rows: list[list],
        relation_rows: list[list],
        open_relation_rows: list[list] | None = None,
    ):
        self._ent_rows = entity_rows
        self._rel_rows = relation_rows
        self._open_rel_rows = open_relation_rows or []

    async def query(self, cypher: str, params: dict | None = None):
        self.calls.append((cypher, params))
        # Entity-type load
        if "MATCH (e:`__Ontology`)" in cypher and "e.attributes AS attributes" in cypher:
            return _FakeQueryResult(self._ent_rows)
        # Patterned-relation load
        if (
            "MATCH (s:`__Ontology`)-[r]->(t:`__Ontology`)" in cypher
            and "rel_label" in cypher
        ):
            return _FakeQueryResult(self._rel_rows)
        # Open-relation load
        if "MATCH (o:`__OpenRelation`)-[r]->(o)" in cypher:
            return _FakeQueryResult(self._open_rel_rows)
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


# ── encoding helpers ─────────────────────────────────────────────


class TestEncodeAttributes:
    def test_empty_props_encodes_empty_object(self):
        assert _encode_attributes([]) == "{}"

    def test_props_encode_and_decode_roundtrip(self):
        props = [
            Attribute(name="age", type="INTEGER", description="Years"),
            Attribute(name="birth_date", type="DATE"),
        ]
        encoded = _encode_attributes(props)
        data = json.loads(encoded)
        assert data == {
            "age": {"type": "INTEGER", "description": "Years"},
            "birth_date": {"type": "DATE"},
        }
        decoded = _decode_attributes(encoded)
        assert [(a.name, a.type, a.description) for a in decoded] == [
            ("age", "INTEGER", "Years"),
            ("birth_date", "DATE", None),
        ]


class TestDecodeAttributes:
    def test_returns_empty_on_garbage(self):
        assert _decode_attributes(None) == []
        assert _decode_attributes("") == []
        assert _decode_attributes("not json") == []
        assert _decode_attributes("[]") == []  # JSON but not an object

    def test_tolerates_bare_type_string(self):
        """Older shapes might have stored ``{"age": "INTEGER"}`` directly."""
        decoded = _decode_attributes(json.dumps({"age": "INTEGER"}))
        assert [(a.name, a.type) for a in decoded] == [("age", "INTEGER")]


# ── store identity ───────────────────────────────────────────────


class TestOntologyStoreGraphName:
    def test_suffix(self, store_factory):
        store = store_factory("my_kg")
        assert store.graph_name == "my_kg__ontology"


# ── register / load ──────────────────────────────────────────────


class TestRegisterEntityShape:
    @pytest.mark.asyncio
    async def test_entity_node_uses_dual_label_and_attributes_json(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        await store.register(
            Ontology(
                entities=[
                    Entity(
                        label="Person",
                        description="A human",
                        properties=[
                            Attribute(name="age", type="INTEGER"),
                            Attribute(name="birth_date", type="DATE"),
                        ],
                    ),
                ]
            )
        )
        merges = [c for c in fake_graph.calls if "MERGE (e:`Person`:`__Ontology`" in c[0]]
        assert len(merges) == 1
        params = merges[0][1] or {}
        attrs = json.loads(params["attributes"])
        assert attrs == {
            "age": {"type": "INTEGER"},
            "birth_date": {"type": "DATE"},
        }


class TestRegisterRelationShape:
    @pytest.mark.asyncio
    async def test_patterned_relation_materialises_one_edge_per_pattern(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        await store.register(
            Ontology(
                entities=[Entity(label="Person"), Entity(label="Company")],
                relations=[
                    Relation(
                        label="WORKS_AT",
                        patterns=[("Person", "Company"), ("Person", "Organization")],
                        properties=[Attribute(name="since", type="DATE")],
                    ),
                ],
            )
        )
        edge_merges = [
            c
            for c in fake_graph.calls
            if "MERGE (s)-[r:`WORKS_AT`]->(t)" in c[0]
        ]
        # One MERGE per pattern.
        assert len(edge_merges) == 2
        endpoints = {((c[1] or {}).get("src"), (c[1] or {}).get("tgt")) for c in edge_merges}
        assert endpoints == {("Person", "Company"), ("Person", "Organization")}
        # Each edge carries the attributes JSON.
        for call in edge_merges:
            attrs = json.loads((call[1] or {}).get("attributes", "{}"))
            assert attrs == {"since": {"type": "DATE"}}

    @pytest.mark.asyncio
    async def test_open_relation_lands_on_placeholder_self_loop(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        await store.register(
            Ontology(
                relations=[
                    Relation(label="KNOWS_SOMETHING", description="Open relation"),
                ],
            )
        )
        open_merges = [
            c
            for c in fake_graph.calls
            if "MERGE (o:`__OpenRelation`:`__Ontology`)" in c[0]
            and "MERGE (o)-[r:`KNOWS_SOMETHING`]->(o)" in c[0]
        ]
        assert len(open_merges) == 1


class TestLoad:
    @pytest.mark.asyncio
    async def test_empty_graph_yields_empty_ontology(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_load_response([], [], [])
        result = await store.load()
        assert result.entities == []
        assert result.relations == []

    @pytest.mark.asyncio
    async def test_reconstructs_entity_with_attributes(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_load_response(
            entity_rows=[
                [
                    "Person",
                    "A human",
                    json.dumps(
                        {
                            "age": {"type": "INTEGER", "description": "Years"},
                            "birth_date": {"type": "DATE"},
                        }
                    ),
                ],
                ["Company", None, "{}"],
            ],
            relation_rows=[],
        )
        ontology = await store.load()
        assert {e.label for e in ontology.entities} == {"Person", "Company"}
        person = next(e for e in ontology.entities if e.label == "Person")
        assert sorted((a.name, a.type, a.description) for a in person.properties) == [
            ("age", "INTEGER", "Years"),
            ("birth_date", "DATE", None),
        ]

    @pytest.mark.asyncio
    async def test_groups_relation_edges_by_label_into_patterns(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        fake_graph.set_load_response(
            entity_rows=[],
            relation_rows=[
                ["Person", "Company", "WORKS_AT", "Employment", json.dumps({"since": {"type": "DATE"}})],
                ["Person", "Organization", "WORKS_AT", "Employment", json.dumps({"since": {"type": "DATE"}})],
                ["Person", "Person", "KNOWS", None, "{}"],
            ],
        )
        ontology = await store.load()
        rel_by_label = {r.label: r for r in ontology.relations}
        assert set(rel_by_label) == {"WORKS_AT", "KNOWS"}
        assert set(rel_by_label["WORKS_AT"].patterns) == {
            ("Person", "Company"),
            ("Person", "Organization"),
        }
        assert [
            (a.name, a.type) for a in rel_by_label["WORKS_AT"].properties
        ] == [("since", "DATE")]
        assert rel_by_label["KNOWS"].patterns == [("Person", "Person")]

    @pytest.mark.asyncio
    async def test_open_relation_loaded_with_empty_patterns(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        fake_graph.set_load_response(
            entity_rows=[],
            relation_rows=[],
            open_relation_rows=[
                ["KNOWS_SOMETHING", "Open relation", "{}"],
            ],
        )
        ontology = await store.load()
        rel = next(r for r in ontology.relations if r.label == "KNOWS_SOMETHING")
        assert rel.patterns == []
        assert rel.description == "Open relation"

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
    async def test_exact_subset_redeclaration_is_accepted(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        fake_graph.set_load_response(
            entity_rows=[
                ["Person", None, json.dumps({"age": {"type": "INTEGER"}})],
            ],
            relation_rows=[],
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
    async def test_redefining_entity_property_type_is_rejected(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        fake_graph.set_load_response(
            entity_rows=[
                ["Person", None, json.dumps({"age": {"type": "INTEGER"}})],
            ],
            relation_rows=[],
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
        upserts = [c for c in fake_graph.calls if "MERGE (e:`Person`:`__Ontology`" in c[0]]
        assert upserts == []

    @pytest.mark.asyncio
    async def test_redefining_relation_property_type_is_rejected(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        fake_graph.set_load_response(
            entity_rows=[],
            relation_rows=[
                ["Person", "Company", "WORKS_AT", None, json.dumps({"since": {"type": "DATE"}})],
            ],
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
            entity_rows=[
                ["Person", None, json.dumps({"age": {"type": "INTEGER"}})],
            ],
            relation_rows=[],
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
        ent_merges = [c for c in fake_graph.calls if "MERGE (e:`Company`:`__Ontology`" in c[0]]
        assert len(ent_merges) == 1

    @pytest.mark.asyncio
    async def test_adding_property_to_existing_label_is_rejected(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        fake_graph.set_load_response(
            entity_rows=[
                ["Person", None, json.dumps({"age": {"type": "INTEGER"}})],
            ],
            relation_rows=[],
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
        upserts = [c for c in fake_graph.calls if "MERGE (e:`Person`:`__Ontology`" in c[0]]
        assert upserts == []

    @pytest.mark.asyncio
    async def test_adding_pattern_to_existing_relation_is_rejected(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        fake_graph.set_load_response(
            entity_rows=[],
            relation_rows=[
                ["Person", "Company", "WORKS_AT", None, "{}"],
            ],
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
            entity_rows=[],
            relation_rows=[
                ["Person", "Company", "WORKS_AT", None, json.dumps({"since": {"type": "DATE"}})],
            ],
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
