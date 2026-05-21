"""Tests for storage/ontology_store.py — persistent ontology graph.

The store talks to FalkorDB directly; unit tests here mock the graph handle
through ``FalkorDBConnection``'s ``_driver.select_graph()`` seam. Real-FalkorDB
exercise is left for the integration suite.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from graphrag_sdk.core.models import (
    EntityType,
    GraphSchema,
    PropertyType,
    RelationType,
)
from graphrag_sdk.storage.ontology_store import (
    OntologyContradictionError,
    OntologyStore,
    _decode_patterns,
    _encode_patterns,
    _props_from_rows,
)


class _FakeQueryResult:
    """Stand-in for FalkorDB's QueryResult."""

    def __init__(self, rows):
        self.result_set = rows


class _FakeGraph:
    """In-memory async ``query()`` substitute. Records calls and serves canned
    responses for the load + read-existing-patterns queries."""

    def __init__(self):
        self.calls: list[tuple[str, dict | None]] = []
        self._ent_rows: list[list] = []
        self._rel_rows: list[list] = []
        self._patterns_for_label: dict[str, list[str]] = {}

    def set_load_response(self, entity_rows, relation_rows):
        self._ent_rows = entity_rows
        self._rel_rows = relation_rows

    def set_existing_patterns(self, label: str, patterns: list[str]):
        self._patterns_for_label[label] = patterns

    async def query(self, cypher: str, params: dict | None = None):
        self.calls.append((cypher, params))
        if "MATCH (e:OntologyEntityType)" in cypher and "collect" in cypher:
            return _FakeQueryResult(self._ent_rows)
        if "MATCH (r:OntologyRelationType)" in cypher and "collect" in cypher:
            return _FakeQueryResult(self._rel_rows)
        if "MATCH (r:OntologyRelationType {label: $label})" in cypher and "patterns" in cypher:
            label = (params or {}).get("label", "")
            return _FakeQueryResult([[self._patterns_for_label.get(label, [])]])
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


# ── small helpers ────────────────────────────────────────────────


class TestEncoders:
    def test_roundtrip(self):
        patterns = [("Person", "Company"), ("Person", "Organization")]
        encoded = _encode_patterns(patterns)
        assert encoded == ["Person|Company", "Person|Organization"]
        assert _decode_patterns(encoded) == patterns

    def test_decode_tolerates_garbage(self):
        assert _decode_patterns(None) == []
        assert _decode_patterns(["no-pipe", "a|b"]) == [("a", "b")]


class TestPropsFromRows:
    def test_filters_null_keyed_rows(self):
        rows = [
            {"name": "age", "type": "INTEGER", "description": None},
            {"name": None, "type": None, "description": None},  # optional-match empty
            None,
            "not a dict",
        ]
        result = _props_from_rows(rows)
        assert [(p.name, p.type) for p in result] == [("age", "INTEGER")]


# ── store identity ───────────────────────────────────────────────


class TestOntologyStoreGraphName:
    def test_suffix(self, store_factory):
        store = store_factory("my_kg")
        assert store.graph_name == "my_kg__ontology"


# ── register / load ──────────────────────────────────────────────


class TestRegister:
    @pytest.mark.asyncio
    async def test_empty_schema_short_circuits_to_load(self, store_factory, fake_graph):
        store = store_factory()
        result = await store.register(GraphSchema())
        # No upsert queries — only the two load queries.
        upserts = [c for c in fake_graph.calls if "MERGE" in c[0]]
        assert upserts == []
        assert isinstance(result, GraphSchema)

    @pytest.mark.asyncio
    async def test_persists_entity_type_with_properties(self, store_factory, fake_graph):
        store = store_factory()
        schema = GraphSchema(
            entities=[
                EntityType(
                    label="Person",
                    description="A human",
                    properties=[
                        PropertyType(name="age", type="INTEGER"),
                        PropertyType(name="birth_date", type="DATE"),
                    ],
                ),
            ],
        )
        await store.register(schema)
        ent_merges = [c for c in fake_graph.calls if "MERGE (e:OntologyEntityType" in c[0]]
        prop_merges = [c for c in fake_graph.calls if "MERGE (ent)-[:HAS_PROPERTY]->" in c[0]]
        assert len(ent_merges) == 1
        assert len(prop_merges) == 2
        prop_names = {(c[1] or {}).get("name") for c in prop_merges}
        assert prop_names == {"age", "birth_date"}

    @pytest.mark.asyncio
    async def test_unions_relation_patterns(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_existing_patterns("WORKS_AT", ["Person|Company"])
        schema = GraphSchema(
            entities=[EntityType(label="Person"), EntityType(label="Org")],
            relations=[RelationType(label="WORKS_AT", patterns=[("Person", "Org")])],
        )
        await store.register(schema)
        rel_set_calls = [
            c
            for c in fake_graph.calls
            if "MERGE (r:OntologyRelationType {label: $label})" in c[0] and "SET r." in c[0]
        ]
        assert rel_set_calls, "expected SET on RelationType"
        patterns = (rel_set_calls[-1][1] or {})["patterns"]
        assert "Person|Company" in patterns
        assert "Person|Org" in patterns


class TestLoad:
    @pytest.mark.asyncio
    async def test_empty_graph_yields_empty_schema(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_load_response([], [])
        schema = await store.load()
        assert schema.entities == []
        assert schema.relations == []

    @pytest.mark.asyncio
    async def test_introspection_failure_returns_empty(self, store_factory, fake_graph):
        async def boom(cypher, params=None):
            raise RuntimeError("connection blew up")

        fake_graph.query = boom
        store = store_factory()
        schema = await store.load()
        assert schema.entities == []
        assert schema.relations == []

    @pytest.mark.asyncio
    async def test_reconstructs_schema(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_load_response(
            entity_rows=[
                [
                    "Person",
                    "A human",
                    [
                        {"name": "age", "type": "INTEGER", "description": None},
                        # OPTIONAL MATCH empty row — must be filtered.
                        {"name": None, "type": None, "description": None},
                    ],
                ],
                ["Company", None, []],
            ],
            relation_rows=[
                [
                    "WORKS_AT",
                    "Employment",
                    ["Person|Company"],
                    [{"name": "since", "type": "DATE", "description": None}],
                ],
            ],
        )
        schema = await store.load()
        assert {e.label for e in schema.entities} == {"Person", "Company"}
        person = next(e for e in schema.entities if e.label == "Person")
        assert [(p.name, p.type) for p in person.properties] == [("age", "INTEGER")]
        works = next(r for r in schema.relations if r.label == "WORKS_AT")
        assert works.patterns == [("Person", "Company")]
        assert [(p.name, p.type) for p in works.properties] == [("since", "DATE")]


# ── contradiction validation ─────────────────────────────────────


class TestContradictionDetection:
    """Additive-only schema: re-typing an existing property is rejected."""

    @pytest.mark.asyncio
    async def test_compatible_addition_is_accepted(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_load_response(
            entity_rows=[
                [
                    "Person",
                    None,
                    [{"name": "age", "type": "INTEGER", "description": None}],
                ],
            ],
            relation_rows=[],
        )
        incoming = GraphSchema(
            entities=[
                EntityType(
                    label="Person",
                    properties=[
                        PropertyType(name="age", type="INTEGER"),       # same as before
                        PropertyType(name="birth_date", type="DATE"),   # new property
                    ],
                ),
            ],
        )
        await store.register(incoming)  # must not raise

    @pytest.mark.asyncio
    async def test_redefining_entity_property_type_is_rejected(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        fake_graph.set_load_response(
            entity_rows=[
                [
                    "Person",
                    None,
                    [{"name": "age", "type": "INTEGER", "description": None}],
                ],
            ],
            relation_rows=[],
        )
        incoming = GraphSchema(
            entities=[
                EntityType(
                    label="Person",
                    properties=[PropertyType(name="age", type="STRING")],
                ),
            ],
        )
        with pytest.raises(OntologyContradictionError) as exc:
            await store.register(incoming)
        assert "Person.age" in str(exc.value)
        # And no MERGE-on-EntityType happened (validation runs before persistence).
        upserts = [c for c in fake_graph.calls if "MERGE (e:OntologyEntityType" in c[0]]
        assert upserts == []

    @pytest.mark.asyncio
    async def test_redefining_relation_property_type_is_rejected(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        fake_graph.set_load_response(
            entity_rows=[],
            relation_rows=[
                [
                    "WORKS_AT",
                    None,
                    [],
                    [{"name": "since", "type": "DATE", "description": None}],
                ],
            ],
        )
        incoming = GraphSchema(
            relations=[
                RelationType(
                    label="WORKS_AT",
                    properties=[PropertyType(name="since", type="STRING")],
                ),
            ],
        )
        with pytest.raises(OntologyContradictionError) as exc:
            await store.register(incoming)
        assert "WORKS_AT.since" in str(exc.value)


# ── delete lifecycle ─────────────────────────────────────────────


class TestDeleteLifecycle:
    """Forward-only deletion: removes the declaration from the ontology
    graph; data-graph nodes/edges are untouched (caller's responsibility)."""

    @pytest.mark.asyncio
    async def test_delete_property_emits_detach_delete_on_property_node(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        await store.delete_property("Person", "age")
        deletes = [
            c
            for c in fake_graph.calls
            if "MATCH (o:OntologyEntityType" in c[0] and "DETACH DELETE p" in c[0]
        ]
        assert len(deletes) == 1
        assert (deletes[0][1] or {}).get("label") == "Person"
        assert (deletes[0][1] or {}).get("name") == "age"

    @pytest.mark.asyncio
    async def test_delete_property_on_relation_type(self, store_factory, fake_graph):
        store = store_factory()
        await store.delete_property("WORKS_AT", "since", on_relation=True)
        deletes = [
            c
            for c in fake_graph.calls
            if "MATCH (o:OntologyRelationType" in c[0] and "DETACH DELETE p" in c[0]
        ]
        assert len(deletes) == 1

    @pytest.mark.asyncio
    async def test_delete_entity_type_removes_type_and_properties(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        await store.delete_entity_type("Person")
        deletes = [c for c in fake_graph.calls if "DETACH DELETE e, p" in c[0]]
        assert len(deletes) == 1
        assert (deletes[0][1] or {}).get("label") == "Person"

    @pytest.mark.asyncio
    async def test_delete_relation_type_removes_type_and_properties(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        await store.delete_relation_type("WORKS_AT")
        deletes = [c for c in fake_graph.calls if "DETACH DELETE r, p" in c[0]]
        assert len(deletes) == 1
        assert (deletes[0][1] or {}).get("label") == "WORKS_AT"
