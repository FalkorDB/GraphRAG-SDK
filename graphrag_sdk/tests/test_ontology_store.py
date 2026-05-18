"""Tests for storage/ontology_store.py — data-graph ontology inference."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from graphrag_sdk.storage.ontology_store import (
    OntologyStore,
    _normalize_type,
    _props_from_rows,
)


class _FakeResult:
    def __init__(self, rows):
        self.result_set = rows


def _make_connection(handler):
    """Wrap an async function (cypher, params) -> _FakeResult into a fake
    ``FalkorDBConnection`` that just routes ``query`` to ``handler``."""
    conn = MagicMock()
    conn.query = handler
    return conn


# ── small helpers ────────────────────────────────────────────────


class TestNormalizeType:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("string", "STRING"),
            ("STRING", "STRING"),
            ("integer", "INTEGER"),
            ("double", "FLOAT"),
            ("float", "FLOAT"),
            ("boolean", "BOOLEAN"),
            ("array", "LIST"),
            ("list", "LIST"),
            ("point", None),
            ("null", None),
            (None, None),
            ("", None),
        ],
    )
    def test_matrix(self, raw, expected):
        assert _normalize_type(raw) == expected


class TestPropsFromRows:
    def test_skips_reserved_and_unknown_types(self):
        rows = [
            ["age", "integer", 5],
            ["name", "string", 10],          # reserved
            ["source_chunk_ids", "array", 5],  # reserved
            ["weird", "point", 1],            # unmapped type
            ["email", "string", 8],
            ["age", "string", 1],             # duplicate -> first wins
        ]
        out = _props_from_rows(rows)
        assert [(p.name, p.type) for p in out] == [
            ("age", "INTEGER"),
            ("email", "STRING"),
        ]

    def test_empty_input(self):
        assert _props_from_rows(None) == []
        assert _props_from_rows([]) == []


# ── infer() end-to-end (mocked driver) ───────────────────────────


class TestOntologyStoreInfer:
    @pytest.mark.asyncio
    async def test_filters_structural_labels_and_edge_types(self):
        calls = []

        async def handler(cypher, params=None):
            calls.append(cypher)
            if "db.labels()" in cypher:
                return _FakeResult(
                    [["Person"], ["Chunk"], ["Document"], ["__Entity__"]]
                )
            if "db.relationshipTypes()" in cypher:
                return _FakeResult(
                    [["PART_OF"], ["NEXT_CHUNK"], ["MENTIONED_IN"]]
                )
            return _FakeResult([])

        store = OntologyStore(_make_connection(handler))
        schema = await store.infer()
        assert [e.label for e in schema.entities] == ["Person"]
        assert schema.relations == []

    @pytest.mark.asyncio
    async def test_relates_subtypes_are_surfaced_with_patterns_and_props(self):
        async def handler(cypher, params=None):
            if "db.labels()" in cypher:
                return _FakeResult([["Person"], ["Company"], ["Location"]])
            if "db.relationshipTypes()" in cypher:
                return _FakeResult([["RELATES"], ["MENTIONED_IN"]])
            if "MATCH (n:`Person`)" in cypher:
                return _FakeResult([["age", "integer", 3], ["name", "string", 3]])
            if "MATCH (n:`Company`)" in cypher or "MATCH (n:`Location`)" in cypher:
                return _FakeResult([])
            if "DISTINCT r.rel_type" in cypher:
                return _FakeResult([["WORKS_AT"], ["LOCATED_IN"]])
            if "labels(a)" in cypher:
                sub = (params or {}).get("sub", "")
                if sub == "WORKS_AT":
                    return _FakeResult(
                        [[["Person"], ["Company", "__Entity__"]]]
                    )
                if sub == "LOCATED_IN":
                    return _FakeResult(
                        [[["Person"], ["Location", "__Entity__"]]]
                    )
            if "UNWIND keys(r)" in cypher:
                sub = (params or {}).get("sub", "")
                if sub == "WORKS_AT":
                    # rel_type is reserved and must be skipped from
                    # discovered properties.
                    return _FakeResult(
                        [["since", "string", 2], ["rel_type", "string", 2]]
                    )
                return _FakeResult([])
            return _FakeResult([])

        store = OntologyStore(_make_connection(handler))
        schema = await store.infer()

        assert {e.label for e in schema.entities} == {"Person", "Company", "Location"}
        person = next(e for e in schema.entities if e.label == "Person")
        assert [(p.name, p.type) for p in person.properties] == [("age", "INTEGER")]

        rel_by_label = {r.label: r for r in schema.relations}
        assert set(rel_by_label) == {"WORKS_AT", "LOCATED_IN"}
        # Patterns strip __Entity__ and pick the user-visible label.
        assert rel_by_label["WORKS_AT"].patterns == [("Person", "Company")]
        # Reserved rel property is suppressed from the inferred schema.
        assert [
            (p.name, p.type) for p in rel_by_label["WORKS_AT"].properties
        ] == [("since", "STRING")]
        assert rel_by_label["LOCATED_IN"].properties == []

    @pytest.mark.asyncio
    async def test_endpoint_pattern_drops_structural_targets(self):
        async def handler(cypher, params=None):
            if "db.labels()" in cypher:
                return _FakeResult([["Person"]])
            if "db.relationshipTypes()" in cypher:
                return _FakeResult([["RELATES"]])
            if "MATCH (n:`Person`)" in cypher:
                return _FakeResult([])
            if "DISTINCT r.rel_type" in cypher:
                return _FakeResult([["KNOWS"]])
            if "labels(a)" in cypher:
                # Stale data has both a real and a structural target.
                return _FakeResult(
                    [
                        [["Person"], ["Chunk"]],          # structural target -> dropped
                        [["Person"], ["Person"]],          # real -> kept
                    ]
                )
            if "UNWIND keys(r)" in cypher:
                return _FakeResult([])
            return _FakeResult([])

        store = OntologyStore(_make_connection(handler))
        schema = await store.infer()
        knows = next(r for r in schema.relations if r.label == "KNOWS")
        assert knows.patterns == [("Person", "Person")]

    @pytest.mark.asyncio
    async def test_introspection_failure_returns_empty_schema(self):
        async def handler(cypher, params=None):
            raise RuntimeError("connection blew up")

        store = OntologyStore(_make_connection(handler))
        schema = await store.infer()
        assert schema.entities == []
        assert schema.relations == []
