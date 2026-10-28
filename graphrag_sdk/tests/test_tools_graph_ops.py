"""Tests for graphrag_sdk.tools.graph_ops (stubbed graph stores)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from falkordb import Edge, Node

from graphrag_sdk.core.exceptions import GraphRAGError, LatencyBudgetExceededError
from graphrag_sdk.tools import graph_ops as ops
from graphrag_sdk.tools.errors import ToolError


def result(rows: list[list[Any]], header: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(result_set=rows, header=[[1, h] for h in (header or [])])


class FakeGraphStore:
    """Answers query_raw by the first matching (substring, responder) rule."""

    def __init__(self, rules: list[tuple[str, Callable[[str, dict], Any]]] | None = None):
        self.rules = rules or []
        self.calls: list[dict[str, Any]] = []

    async def query_raw(self, cypher, params=None, *, read_only=False, timeout=None):
        self.calls.append(
            {"cypher": cypher, "params": params or {}, "read_only": read_only, "timeout": timeout}
        )
        for needle, responder in self.rules:
            if needle in cypher:
                out = responder(cypher, params or {})
                if isinstance(out, BaseException):
                    raise out
                return out
        return result([])

    def calls_with(self, needle: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if needle in c["cypher"]]


def remaining(ms: int = 5000) -> Callable[[], int]:
    return lambda: ms


# ── ToolError ────────────────────────────────────────────────────


def test_tool_error_carries_a_hint():
    err = ToolError("No entity is called 'X'.", hint="Call lookup_entity first.")
    assert isinstance(err, GraphRAGError)
    assert err.hint == "Call lookup_entity first."
    assert str(err) == "No entity is called 'X'. Hint: Call lookup_entity first."
    plain = ToolError("boom")
    assert plain.hint == "" and str(plain) == "boom"


# ── clamp_int / bounded_query ────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [(5, 5), ("7", 7), (0, 1), (-3, 1), (10_000, 200), (None, 50), ("x", 50), (3.9, 3)],
)
def test_clamp_int(value, expected):
    assert ops.clamp_int(value, 50, 1, 200) == expected


async def test_bounded_query_runs_read_only_with_a_database_timeout():
    store = FakeGraphStore([("RETURN 1", lambda c, p: result([[1]]))])
    out = await ops.bounded_query(store, "RETURN 1", None, 1234)
    assert out.result_set == [[1]]
    assert store.calls == [{"cypher": "RETURN 1", "params": {}, "read_only": True, "timeout": 1234}]


async def test_bounded_query_has_a_client_side_timeout():
    class Slow:
        async def query_raw(self, cypher, params=None, *, read_only=False, timeout=None):
            await asyncio.sleep(5)

    with pytest.raises(ops.TIMEOUTS):
        await ops.bounded_query(Slow(), "RETURN 1", {}, 1)


async def test_a_server_side_timeout_is_a_timeout():
    """FalkorDB's own TIMEOUT reaches us wrapped in a DatabaseError."""
    from graphrag_sdk.core.exceptions import DatabaseError

    class TimedOut:
        async def query_raw(self, cypher, params=None, *, read_only=False, timeout=None):
            raise DatabaseError("Query failed: Query timed out")

    with pytest.raises(TimeoutError):
        await ops.bounded_query(TimedOut(), "RETURN 1", {}, 1000)


async def test_other_database_errors_pass_through():
    from graphrag_sdk.core.exceptions import DatabaseError

    class Broken:
        async def query_raw(self, cypher, params=None, *, read_only=False, timeout=None):
            raise DatabaseError("Type mismatch")

    with pytest.raises(DatabaseError, match="Type mismatch"):
        await ops.bounded_query(Broken(), "RETURN 1", {}, 1000)


def test_timeouts_cover_both_timeout_errors():
    assert TimeoutError in ops.TIMEOUTS and asyncio.TimeoutError in ops.TIMEOUTS


# ── to_jsonable / column_names ───────────────────────────────────


class TestToJsonable:
    def test_scalars(self):
        for value in (None, True, 3, 2.5, "text"):
            assert ops.to_jsonable(value) == value

    def test_falkordb_node_and_edge(self):
        node = Node(
            node_id=1,
            labels=["Person", "__Entity__"],
            properties={"name": "Alice", "embedding": [0.1] * 4, "source_chunk_ids": ["c1"]},
        )
        edge = Edge(
            src_node=1,
            relation="RELATES",
            dest_node=2,
            edge_id=9,
            properties={"fact": "x", "rel_type": "WORKS_AT", "embedding": [0.2] * 4},
        )
        assert ops.to_jsonable(node) == {"labels": ["Person"], "name": "Alice"}
        assert ops.to_jsonable(edge) == {"type": "RELATES", "fact": "x", "rel_type": "WORKS_AT"}

    def test_node_like_objects(self):
        node = SimpleNamespace(labels=["Person"], properties={"name": "A", "embedding": [0.1]})
        assert ops.to_jsonable(node) == {"labels": ["Person"], "name": "A"}
        bare = SimpleNamespace(properties={"name": "A", "vec": [0.5] * 64})
        assert ops.to_jsonable(bare) == {"name": "A"}

    def test_embedding_keys_are_dropped_from_maps(self):
        assert ops.to_jsonable({"name": "A", "name_embedding": [1]}) == {"name": "A"}
        assert ops.to_jsonable({"name": "A", "source_chunk_ids": ["c"]}) == {"name": "A"}

    def test_internal_config_node_is_never_rendered(self):
        node = SimpleNamespace(labels=["__GraphRAGConfig__"], properties={"sdk_version": "1"})
        assert ops.to_jsonable(node) == "[internal node omitted]"
        props = {"id": "default", "sdk_version": "1", "embedding_dimension": 8, "updated_at": "x"}
        assert ops.to_jsonable(props) == "[internal node omitted]"
        assert ops.to_jsonable([props]) == ["[internal node omitted]"]
        assert ops.to_jsonable({"name": "Acme"}) == {"name": "Acme"}

    def test_vectors_are_never_rendered(self):
        assert ops.to_jsonable([0.1] * 64) == "[vector of 64 numbers omitted]"
        assert ops.to_jsonable([1, 2, 3]) == [1, 2, 3]
        assert ops.to_jsonable([True] * 64) == [True] * 64  # booleans are not a vector
        assert ops.to_jsonable({"v": [0.5] * 100})["v"] == "[vector of 100 numbers omitted]"
        assert ops.to_jsonable((1, 2)) == [1, 2]

    def test_long_strings_are_cut_and_control_chars_stripped(self):
        out = ops.to_jsonable("x" * 50, max_chars=10)
        assert out == "x" * 10 + "…[truncated]"
        assert ops.to_jsonable("plain\x00text") == "plaintext"
        assert ops.to_jsonable({"k": "y" * 50}, max_chars=5) == {"k": "yyyyy…[truncated]"}

    def test_other_values_become_strings(self):
        class Point:
            def __str__(self):
                return "point(1, 2)"

        assert ops.to_jsonable(Point()) == "point(1, 2)"


def test_column_names():
    assert ops.column_names(result([], ["name", "n"])) == ["name", "n"]
    assert ops.column_names(SimpleNamespace(header=["a", "b"])) == ["a", "b"]
    assert ops.column_names(SimpleNamespace()) == []


# ── Status / ontology counts ─────────────────────────────────────


async def test_schema_counts():
    store = FakeGraphStore(
        [
            ("UNWIND labels(n)", lambda c, p: result([["Company", 1], ["Person", 2]])),
            ("r.rel_type AS t", lambda c, p: result([["WORKS_AT", 3]])),
        ]
    )
    labels, rels = await ops.schema_counts(store)
    assert labels == {"Company": 1, "Person": 2}
    assert rels == {"WORKS_AT": 3}
    assert all(c["read_only"] for c in store.calls)


async def test_graph_counts():
    store = FakeGraphStore(
        [
            ("RETURN count(n)", lambda c, p: result([[4]])),
            ("RETURN count(r)", lambda c, p: result([[None]])),
            ("RETURN count(d)", lambda c, p: result([[2]])),
        ]
    )
    assert await ops.graph_counts(store) == {
        "entities": 4,
        "relations": 0,
        "documents": 2,
        "chunks": 0,
    }


# ── Retrieval enrichment ─────────────────────────────────────────


async def test_enrich_entities_maps_rows_and_drops_bulky_properties():
    row = [
        "e1",
        "Alice",
        "Engineer",
        ["__Entity__", "Person"],
        {
            "id": "e1",
            "name": "Alice",
            "description": "Engineer",
            "seniority": "senior",
            "bio": "b" * 400,
            "embedding": [0.1],
            "source_chunk_ids": ["c1"],
        },
    ]
    store = FakeGraphStore([("UNWIND $ids", lambda c, p: result([row, ["e2", None, None]]))])
    cards = await ops.enrich_entities(store, ["e1", "e2"])
    card = cards["e1"]
    assert card["name"] == "Alice" and card["label"] == "Person"
    assert card["description"] == "Engineer"
    assert set(card["properties"]) == {"seniority", "bio"}
    assert card["properties"]["bio"] == "b" * 300 + "…[truncated]"
    assert cards["e2"] == {
        "id": "e2",
        "name": "",
        "label": "",
        "description": None,
        "properties": {},
    }
    assert store.calls[0]["params"] == {"ids": ["e1", "e2"]}


async def test_enrich_entities_short_circuits():
    store = FakeGraphStore()
    assert await ops.enrich_entities(store, []) == {}
    assert store.calls == []


async def test_expand_triples_dedupes_and_walks_hop_by_hop():
    rows = [
        ["A", "REL", "B", "", "e2"],
        ["a", "REL", "b", "", "e2"],  # same triple, other case
        ["B", "REL2", "C", "evidence", "e3"],
        ["B", None, "C", "", "e1"],  # incomplete edge, seed id
    ]
    store = FakeGraphStore([("r.src_name", lambda c, p: result(rows))])
    triples = await ops.expand_triples(store, ["e1"], hops=1, cap=10)
    assert triples == [
        {"source": "A", "type": "REL", "target": "B", "fact": ""},
        {"source": "B", "type": "REL2", "target": "C", "fact": "evidence"},
    ]
    assert store.calls[0]["params"] == {"ids": ["e1"], "limit": 25}

    store.calls.clear()
    await ops.expand_triples(store, ["e1"], hops=2, cap=10, per_hop_limit=7)
    assert len(store.calls) == 2  # one query per hop
    second = store.calls[1]["params"]
    assert second["ids"] == ["e2", "e3"] and second["limit"] == 7  # visited ids excluded

    assert len(await ops.expand_triples(store, ["e1"], hops=1, cap=1)) == 1


async def test_expand_triples_stops_when_the_frontier_is_empty():
    store = FakeGraphStore([("r.src_name", lambda c, p: result([["A", "R", "B", "", "e1"]]))])
    await ops.expand_triples(store, ["e1"], hops=3)
    assert len(store.calls) == 1
    store.calls.clear()
    await ops.expand_triples(store, ["e1"], hops=0)
    assert len(store.calls) == 1  # at least one hop


async def test_chunk_documents():
    store = FakeGraphStore(
        [("PART_OF", lambda c, p: result([["c1", "doc-a", "docs/a.md"], ["c2", "doc-b", None]]))]
    )
    assert await ops.chunk_documents(store, ["c1", "c2"]) == {
        "c1": ("doc-a", "docs/a.md"),
        "c2": ("doc-b", ""),
    }
    store.calls.clear()
    assert await ops.chunk_documents(store, []) == {}
    assert store.calls == []


async def test_entity_names():
    store = FakeGraphStore(
        [("RETURN e.id, e.name", lambda c, p: result([["a", "A"], ["b", None]]))]
    )
    assert await ops.entity_names(store, ["a", "b", "a"]) == {"a": "A", "b": "b"}
    assert store.calls[0]["params"] == {"ids": ["a", "b"]}
    assert await ops.entity_names(store, []) == {}


# ── Entity lookup ────────────────────────────────────────────────


def lookup_rows(c, p):
    return result([["control_board_cx10", "Control Board CX-10", "Part", "A board"]])


class TestFindEntities:
    async def test_lists_candidates_with_normalised_parameters(self):
        store = FakeGraphStore([("CONTAINS $needle", lookup_rows)])
        found = await ops.find_entities(store, "  CX-10   Board ")
        assert found == [
            {
                "id": "control_board_cx10",
                "name": "Control Board CX-10",
                "label": "Part",
                "description": "A board",
            }
        ]
        call = store.calls[0]
        assert call["params"]["needle"] == "cx-10 board"
        assert call["params"]["tokens"] == ["cx-10", "board"]
        assert call["params"]["limit"] == ops.LOOKUP_DEFAULT_LIMIT
        assert call["read_only"] and call["timeout"] == ops.LOOKUP_TIMEOUT_MS

    async def test_short_and_stop_words_are_not_tokens(self):
        store = FakeGraphStore()
        await ops.find_entities(store, "what is the entity of ab Acme", limit=3, timeout_ms=200)
        params = store.calls[0]["params"]
        assert params["tokens"] == ["acme"] and params["limit"] == 3
        assert store.calls[0]["timeout"] == 200

    async def test_at_most_six_tokens(self):
        store = FakeGraphStore()
        await ops.find_entities(store, " ".join(f"word{i}" for i in range(10)))
        assert len(store.calls[0]["params"]["tokens"]) == 6

    async def test_no_match_and_empty_text(self):
        store = FakeGraphStore()
        assert await ops.find_entities(store, "zzz") == []
        assert await ops.find_entities(store, "   ") == []
        assert len(store.calls) == 1  # empty text runs no query

    async def test_rows_without_a_name_are_skipped(self):
        store = FakeGraphStore([("CONTAINS", lambda c, p: result([["x", None], ["y", "Y"]]))])
        found = await ops.find_entities(store, "y name")
        assert found == [{"id": "y", "name": "Y", "label": "", "description": ""}]

    async def test_failure_returns_none(self):
        store = FakeGraphStore([("CONTAINS", lambda c, p: RuntimeError("down"))])
        assert await ops.find_entities(store, "acme") is None

    async def test_budget_errors_propagate(self):
        store = FakeGraphStore([("CONTAINS", lambda c, p: LatencyBudgetExceededError("late"))])
        with pytest.raises(LatencyBudgetExceededError):
            await ops.find_entities(store, "acme")


def test_render_candidates():
    out = ops.render_candidates(
        [
            {"id": "a1", "name": "Apex", "label": "Company"},
            {"id": "a2", "name": "N" * 130, "label": ""},
        ]
    )
    assert out == f'"Apex" (Company, id a1); "{"N" * 120}…" (id a2)'


# ── resolve_start ────────────────────────────────────────────────

EXACT = "WHERE e.id = $k OR e.name = $k"
CASELESS = "toLower(toString(e.name)) = toLower($k)"


def deadline(seconds: float = 5.0) -> float:
    return time.monotonic() + seconds


class TestResolveStart:
    async def test_exact_id_or_name(self):
        store = FakeGraphStore([(EXACT, lambda c, p: result([["alice", "Alice"]]))])
        assert await ops.resolve_start(store, "alice", deadline()) == ("alice", "Alice")
        assert store.calls[0]["params"] == {"k": "alice"} and store.calls[0]["read_only"]

    async def test_case_insensitive_fallback(self):
        store = FakeGraphStore([(CASELESS, lambda c, p: result([["alice", "Alice"]]))])
        assert await ops.resolve_start(store, "ALICE", deadline()) == ("alice", "Alice")
        assert len(store.calls) == 2

    async def test_ambiguous_start_lists_the_ids(self):
        store = FakeGraphStore([(EXACT, lambda c, p: result([["a1", "Apex"], ["a2", "Apex"]]))])
        with pytest.raises(ToolError, match="matches 2 entities") as exc:
            await ops.resolve_start(store, "Apex", deadline())
        assert '"Apex" (id a1)' in str(exc.value) and "one of these ids" in exc.value.hint

    async def test_unknown_start_suggests_candidates(self):
        store = FakeGraphStore([("CONTAINS $needle", lookup_rows)])
        with pytest.raises(ToolError, match="No entity is called 'CX-10'") as exc:
            await ops.resolve_start(store, "CX-10", deadline())
        assert '"Control Board CX-10" (Part, id control_board_cx10)' in str(exc.value)
        assert "exact names or ids" in exc.value.hint

    async def test_unknown_start_without_candidates(self):
        with pytest.raises(ToolError, match="No entity is called 'zzz'") as exc:
            await ops.resolve_start(FakeGraphStore(), "zzz", deadline())
        assert "lookup_entity" in exc.value.hint

    async def test_empty_start(self):
        with pytest.raises(ToolError, match="empty"):
            await ops.resolve_start(FakeGraphStore(), "", deadline())

    async def test_resolution_uses_the_remaining_budget(self):
        store = FakeGraphStore([(EXACT, lambda c, p: result([["a", "A"]]))])
        await ops.resolve_start(store, "a", deadline(0.2))
        assert 0 < store.calls[0]["timeout"] <= 200
        with pytest.raises(ToolError, match="time budget"):
            await ops.resolve_start(store, "a", time.monotonic() - 1)

    async def test_timeout_is_a_tool_error(self):
        store = FakeGraphStore([(EXACT, lambda c, p: asyncio.TimeoutError())])
        with pytest.raises(ToolError, match="timed out"):
            await ops.resolve_start(store, "a", deadline())


# ── expand (breadth-first traverse) ──────────────────────────────

EDGES = [
    ("alice", "WORKS_AT", "acme"),
    ("alice", "KNOWS", "bob"),
    ("acme", "LOCATED_IN", "berlin"),
]


def walk_store(edges=EDGES) -> FakeGraphStore:
    """Answers the hop query like FalkorDB: each edge from both of its ends."""

    def hop(c, p):
        rows = []
        for s in p["frontier"]:
            for src, rel, tgt in edges:
                if s == src:
                    rows.append([s, s.title(), tgt, tgt.title(), rel, True, f"{src} {rel} {tgt}"])
                elif s == tgt:
                    rows.append([s, s.title(), src, src.title(), rel, False, f"{src} {rel} {tgt}"])
        rows.sort(key=lambda r: (r[0], r[2]))
        return result(rows[: p["remaining"]])

    return FakeGraphStore([("s.id IN $frontier", hop)])


class TestExpand:
    async def test_breadth_first_walk(self):
        store = walk_store()
        nodes, edges, stopped = await ops.expand(store, "alice", "Alice", 3, 50, remaining())
        assert stopped == "complete"  # berlin has no further neighbours
        assert [(n["id"], n["depth"]) for n in nodes] == [("acme", 1), ("bob", 1), ("berlin", 2)]
        assert nodes[-1]["from_id"] == "acme" and nodes[-1]["relation"] == "LOCATED_IN"
        # Each edge once, in its stored direction, with its fact.
        assert [(e["source_id"], e["relation"], e["target_id"]) for e in edges] == [
            ("alice", "WORKS_AT", "acme"),
            ("alice", "KNOWS", "bob"),
            ("acme", "LOCATED_IN", "berlin"),
        ]
        assert edges[0]["source"] == "Alice" and edges[0]["fact"] == "alice WORKS_AT acme"
        call = store.calls[0]
        assert call["read_only"] and call["timeout"] == 5000
        assert "SAME_AS" in call["params"]["structural"]
        assert "MENTIONED_IN" in call["params"]["structural"]

    async def test_incoming_edges_keep_their_direction(self):
        nodes, edges, _ = await ops.expand(walk_store(), "berlin", "Berlin", 1, 50, remaining())
        assert nodes == [
            {
                "id": "acme",
                "name": "Acme",
                "depth": 1,
                "from_id": "berlin",
                "relation": "LOCATED_IN",
                "outgoing": False,
            }
        ]
        assert (edges[0]["source_id"], edges[0]["target_id"]) == ("acme", "berlin")

    async def test_depth_budget(self):
        nodes, _, stopped = await ops.expand(walk_store(), "alice", "Alice", 1, 50, remaining())
        assert stopped == "depth_budget" and len(nodes) == 2

    async def test_expansion_budget(self):
        store = walk_store()
        nodes, _, stopped = await ops.expand(store, "alice", "Alice", 3, 1, remaining())
        assert stopped == "expansion_budget" and len(nodes) == 1
        assert store.calls[0]["params"]["remaining"] == 1

    async def test_remaining_allowance_goes_into_each_hop(self):
        store = walk_store()
        await ops.expand(store, "alice", "Alice", 3, 3, remaining())
        assert [c["params"]["remaining"] for c in store.calls] == [3, 1]

    async def test_time_budget_before_a_hop(self):
        store = walk_store()
        nodes, edges, stopped = await ops.expand(store, "alice", "Alice", 3, 50, remaining(0))
        assert (nodes, edges, stopped) == ([], [], "time_budget") and store.calls == []

    @pytest.mark.parametrize("error", [asyncio.TimeoutError, TimeoutError])
    async def test_timeout_is_a_time_budget_stop(self, error):
        store = FakeGraphStore([("s.id IN $frontier", lambda c, p: error())])
        nodes, _, stopped = await ops.expand(store, "a", "A", 2, 50, remaining())
        assert stopped == "time_budget" and nodes == []

    async def test_isolated_start_is_complete(self):
        nodes, edges, stopped = await ops.expand(FakeGraphStore(), "x", "X", 2, 50, remaining())
        assert (nodes, edges, stopped) == ([], [], "complete")


# ── typed_path ───────────────────────────────────────────────────


class TestTypedPath:
    async def test_relation_names_are_parameters(self):
        store = FakeGraphStore(
            [("RETURN DISTINCT n1.id", lambda c, p: result([["berlin", "Berlin"], [None, "x"]]))]
        )
        nodes, stopped = await ops.typed_path(store, "alice", ["WORKS_AT", "LOCATED_IN"], 900)
        assert stopped == "complete"
        assert nodes == [{"id": "berlin", "name": "Berlin", "depth": 2, "relation": "LOCATED_IN"}]
        call = store.calls[0]
        assert call["params"] == {"start": "alice", "t0": "WORKS_AT", "t1": "LOCATED_IN"}
        assert "WORKS_AT" not in call["cypher"] and "LOCATED_IN" not in call["cypher"]
        assert call["read_only"] and call["timeout"] == 900

    async def test_return_limit(self):
        rows = [[f"n{i}", f"N{i}"] for i in range(ops.TRAVERSE_PATH_LIMIT + 1)]
        store = FakeGraphStore([("RETURN DISTINCT", lambda c, p: result(rows))])
        nodes, stopped = await ops.typed_path(store, "a", ["R"], 500)
        assert stopped == "return_limit" and len(nodes) == ops.TRAVERSE_PATH_LIMIT
        assert f"LIMIT {ops.TRAVERSE_PATH_LIMIT + 1}" in store.calls[0]["cypher"]

    async def test_timeout(self):
        store = FakeGraphStore([("RETURN DISTINCT", lambda c, p: asyncio.TimeoutError())])
        assert await ops.typed_path(store, "a", ["R"], 500) == ([], "time_budget")
