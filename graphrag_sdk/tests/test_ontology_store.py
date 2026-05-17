"""Tests for storage/ontology_store.py — persistent ontology graph layer.

The store talks to FalkorDB directly; unit tests here mock the graph handle
through ``FalkorDBConnection``'s private ``_driver.select_graph()`` seam.
Real-FalkorDB exercise is left for the integration suite.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.models import (
    EntityType,
    GraphSchema,
    PropertyType,
    RelationType,
)
from graphrag_sdk.storage.ontology_store import (
    OntologyStore,
    _decode_patterns,
    _encode_patterns,
)


class _FakeQueryResult:
    """Stand-in for falkordb's QueryResult."""

    def __init__(self, rows: list[list]):
        self.result_set = rows


class _FakeGraph:
    """In-memory ``AsyncGraph`` substitute.

    Captures every ``query()`` call (cypher + params) so tests can assert on
    them, and serves canned responses for the load() queries.
    """

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
    """Returns a callable producing an OntologyStore whose `_query` is wired
    to ``fake_graph``. Skips the real driver/connection plumbing entirely."""

    def _make(data_graph_name: str = "kg") -> OntologyStore:
        conn = MagicMock()
        conn._ensure_client = MagicMock()
        conn._driver = SimpleNamespace(
            select_graph=MagicMock(return_value=fake_graph),
        )
        return OntologyStore(conn, data_graph_name)

    return _make


class TestEncoders:
    def test_encode_then_decode_roundtrip(self):
        patterns = [("Person", "Company"), ("Person", "Organization")]
        encoded = _encode_patterns(patterns)
        assert encoded == ["Person|Company", "Person|Organization"]
        assert _decode_patterns(encoded) == patterns

    def test_decode_handles_none_and_bad_strings(self):
        assert _decode_patterns(None) == []
        assert _decode_patterns(["no-pipe", "a|b"]) == [("a", "b")]


class TestOntologyStoreGraphName:
    def test_suffix(self, store_factory):
        store = store_factory("my_kg")
        assert store.graph_name == "my_kg__ontology"


class TestOntologyStoreRegister:
    @pytest.mark.asyncio
    async def test_empty_schema_short_circuits_to_load(self, store_factory, fake_graph):
        store = store_factory()
        result = await store.register(GraphSchema())
        # Only the two load queries should have been issued.
        ents_q = [c for c in fake_graph.calls if "MATCH (e:OntologyEntityType)" in c[0]]
        assert len(ents_q) == 1
        assert isinstance(result, GraphSchema)

    @pytest.mark.asyncio
    async def test_registers_entity_type_and_its_properties(self, store_factory, fake_graph):
        store = store_factory()
        schema = GraphSchema(
            entities=[
                EntityType(
                    label="Person",
                    description="A human",
                    properties=[
                        PropertyType(name="age", type="INTEGER"),
                        PropertyType(name="birth_date", type="DATE", required=True),
                    ],
                ),
            ],
        )
        await store.register(schema)
        # One MERGE for the entity-type node, plus one MERGE per property edge.
        entity_merges = [
            c for c in fake_graph.calls if "MERGE (e:OntologyEntityType" in c[0]
        ]
        property_merges = [
            c for c in fake_graph.calls if "MERGE (o)-[:HAS_PROPERTY]->" in c[0]
        ]
        assert len(entity_merges) == 1
        assert len(property_merges) == 2
        # Property params carry the declared type and required flag.
        prop_names = {(c[1] or {}).get("name") for c in property_merges}
        assert prop_names == {"age", "birth_date"}

    @pytest.mark.asyncio
    async def test_unions_relation_patterns_with_existing(
        self, store_factory, fake_graph
    ):
        store = store_factory()
        fake_graph.set_existing_patterns("WORKS_AT", ["Person|Company"])
        schema = GraphSchema(
            entities=[EntityType(label="Person"), EntityType(label="Org")],
            relations=[
                RelationType(label="WORKS_AT", patterns=[("Person", "Org")]),
            ],
        )
        await store.register(schema)
        rel_set = [
            c
            for c in fake_graph.calls
            if "MERGE (r:OntologyRelationType {label: $label})" in c[0] and "SET r." in c[0]
        ]
        assert rel_set, "expected a SET on the RelationType node"
        params = rel_set[-1][1] or {}
        assert "Person|Company" in params["patterns"]
        assert "Person|Org" in params["patterns"]


class TestOntologyStoreLoad:
    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty_schema(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_load_response([], [])
        result = await store.load()
        assert result.entities == []
        assert result.relations == []

    @pytest.mark.asyncio
    async def test_reconstructs_schema_from_query_rows(self, store_factory, fake_graph):
        store = store_factory()
        fake_graph.set_load_response(
            entity_rows=[
                [
                    "Person",
                    "A human",
                    [
                        {
                            "name": "age",
                            "type": "INTEGER",
                            "description": None,
                            "required": False,
                        },
                        # collect() of an OPTIONAL MATCH that found nothing
                        # may produce an all-None dict — must be filtered.
                        {
                            "name": None,
                            "type": None,
                            "description": None,
                            "required": None,
                        },
                    ],
                ],
                ["Company", None, []],
            ],
            relation_rows=[
                [
                    "WORKS_AT",
                    "Employment",
                    ["Person|Company"],
                    [
                        {
                            "name": "since",
                            "type": "DATE",
                            "description": None,
                            "required": False,
                        }
                    ],
                ],
            ],
        )
        schema = await store.load()
        assert {e.label for e in schema.entities} == {"Person", "Company"}
        person = next(e for e in schema.entities if e.label == "Person")
        assert {p.name for p in person.properties} == {"age"}
        works = next(r for r in schema.relations if r.label == "WORKS_AT")
        assert works.patterns == [("Person", "Company")]
        assert {p.name for p in works.properties} == {"since"}
