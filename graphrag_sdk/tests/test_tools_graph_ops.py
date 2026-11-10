"""Tests for graphrag_sdk.tools.graph_ops (stubbed graph stores)."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from falkordb import Edge, Node, Path

from graphrag_sdk.core.exceptions import GraphRAGError
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
    err = ToolError("No entity is called 'X'.", hint="Call retrieve first.")
    assert isinstance(err, GraphRAGError)
    assert err.hint == "Call retrieve first."
    assert str(err) == "No entity is called 'X'. Hint: Call retrieve first."
    plain = ToolError("boom")
    assert plain.hint == "" and str(plain) == "boom"


# ── to_jsonable ──────────────────────────────────────────────────


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
        assert ops.to_jsonable({"name": "A", "spans": '{"c1": []}'}) == {"name": "A"}

    def test_fact_body_drops_the_repeated_triple(self):
        assert ops.fact_body("(A, WORKS_AT, B): A works at B.", "WORKS_AT") == "A works at B."
        assert ops.fact_body("(A, WORKS_AT, B)", "WORKS_AT") == ""
        assert ops.fact_body("A works at B.", "WORKS_AT") == "A works at B."
        assert ops.fact_body("(see note) A works at B.", "WORKS_AT") == "(see note) A works at B."
        assert ops.fact_body("(A, WORKS_AT, B) (draft)", "WORKS_AT") == "(A, WORKS_AT, B) (draft)"

    def test_internal_config_node_is_never_rendered(self):
        node = SimpleNamespace(labels=["__GraphRAGConfig__"], properties={"sdk_version": "1"})
        assert ops.to_jsonable(node) == "[internal node omitted]"
        props = {"id": "default", "sdk_version": "1", "embedding_dimension": 8, "updated_at": "x"}
        assert ops.to_jsonable(props) == "[internal node omitted]"
        assert ops.to_jsonable([props]) == ["[internal node omitted]"]
        assert ops.to_jsonable({"name": "Acme"}) == {"name": "Acme"}

    def test_system_nodes_are_never_rendered_however_reached(self):
        # A label built at run time ('__GraphRAG' + 'Processing__') passes the
        # guard's text check; the result is still redacted.
        processing = Node(
            node_id=7,
            labels=["__GraphRAGProcessing__"],
            properties={"kind": "pending", "ids": ["e1"], "attempts": [1]},
        )
        assert ops.to_jsonable(processing) == "[internal node omitted]"
        assert ops.to_jsonable([{"kind": "plan", "next": 0}]) == ["[internal node omitted]"]
        # One config key is enough, as in a partial projection.
        assert ops.to_jsonable({"id": "default", "sdk_version": "1"}) == "[internal node omitted]"
        # Entity maps (they have a name) and ordinary "kind" values are kept.
        assert ops.to_jsonable({"name": "Acme", "kind": "plan"}) == {"name": "Acme", "kind": "plan"}
        assert ops.to_jsonable({"kind": "invoice"}) == {"kind": "invoice"}
        entity = Node(node_id=1, labels=["__Entity__", "Person"], properties={"name": "A"})
        assert ops.to_jsonable(entity) == {"labels": ["Person"], "name": "A"}

    def test_paths_read_as_names_and_relation_types(self):
        alice = Node(node_id=1, labels=["__Entity__", "Person"], properties={"name": "Alice"})
        acme = Node(
            node_id=2,
            labels=["__Entity__", "Company"],
            properties={"name": "Acme", "embedding": [0.1] * 64},
        )
        paris = Node(node_id=3, labels=["City"], properties={"id": "paris"})
        works = Edge(1, "RELATES", 2, edge_id=10, properties={"rel_type": "WORKS_AT", "fact": "x"})
        # Stored Paris -> Acme, so walking Acme to Paris goes against it.
        hq = Edge(3, "LOCATED_IN", 2, edge_id=11, properties={})
        path = Path([alice, acme, paris], [works, hq])
        rendered = ops.to_jsonable(path)
        assert rendered == "(Alice)-[WORKS_AT]->(Acme)<-[LOCATED_IN]-(paris)"
        assert ops.to_jsonable(Path([alice], [])) == "(Alice)"
        # A system node on a path is never named.
        config = Node(node_id=4, labels=["__GraphRAGConfig__"], properties={"id": "default"})
        assert "default" not in ops.to_jsonable(
            Path([alice, config], [Edge(1, "R", 4, edge_id=12, properties={})])
        )

    def test_property_maps_never_carry_vectors(self):
        props = {"name": "A", "embedding": [0.1] * 4, "custom_vec": [0.2] * 128}
        assert ops.to_jsonable(props) == {
            "name": "A",
            "custom_vec": "[vector of 128 numbers omitted]",
        }

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


async def test_enrich_entities_hides_bookkeeping_and_caps_the_description():
    long_desc = " | ".join(f"description number {i}" for i in range(40))
    row = [
        "e1",
        "Alice",
        long_desc,
        ["__Entity__", "Person"],
        {
            "name": "Alice",
            "description": long_desc,
            "descriptions": ["a", "b"],
            "description_summarised": True,
            "merged_labels": "Person",
            "is_stub": False,
            "entity_key": "E-1",
            "type": "Person",
            "born__dropped": ["1961"],
            "aliases": ["A. Smith"],
            "born": 1960,
        },
    ]
    store = FakeGraphStore([("UNWIND $ids", lambda c, p: result([row]))])
    card = (await ops.enrich_entities(store, ["e1"]))["e1"]
    assert set(card["properties"]) == {"aliases", "born"}
    assert card["description"].endswith(" …") and len(card["description"]) <= 402
    assert card["description"].startswith("description number 0 | ")


async def test_enrich_entities_short_circuits():
    store = FakeGraphStore()
    assert await ops.enrich_entities(store, []) == {}
    assert store.calls == []


async def test_expand_triples_dedupes_and_walks_hop_by_hop():
    rows = [
        ["A", "REL", "B", "", "e2", 10],
        ["a", "REL", "b", "", "e2", 11],  # same triple, other case
        ["B", "REL2", "C", "evidence", "e3", 12],
        ["B", None, "C", "", "e1"],  # incomplete edge, seed id
    ]
    store = FakeGraphStore([("r.src_name", lambda c, p: result(rows))])
    triples = await ops.expand_triples(store, ["e1"], hops=1, cap=10)
    assert triples == [
        {"source": "A", "type": "REL", "target": "B", "fact": ""},
        {"source": "B", "type": "REL2", "target": "C", "fact": "evidence"},
    ]
    assert store.calls[0]["params"] == {"ids": ["e1"], "seen_edges": [], "limit": 10}

    store.calls.clear()
    await ops.expand_triples(store, ["e1"], hops=2, cap=10, per_hop_limit=7)
    assert len(store.calls) == 2  # one query per hop
    assert store.calls[0]["params"]["limit"] == 5  # half the cap for the first of two hops
    second = store.calls[1]["params"]
    assert second["ids"] == ["e2", "e3"] and second["limit"] == 7  # visited ids excluded
    assert second["seen_edges"] == [10, 11, 12]  # back-edges excluded before LIMIT

    assert len(await ops.expand_triples(store, ["e1"], hops=1, cap=1)) == 1


async def test_expand_triples_gives_every_hop_a_share_of_the_cap():
    limits: list[int] = []

    def respond(cypher: str, params: dict) -> Any:
        limits.append(params["limit"])
        hop = len(limits)
        return result(
            [[f"N{hop}", "R", f"M{hop}-{i}", "", f"h{hop}-{i}", hop * 100 + i] for i in range(40)][
                : params["limit"]
            ]
        )

    store = FakeGraphStore([("r.src_name", respond)])
    triples = await ops.expand_triples(store, ["e1"], hops=3, cap=40)
    assert limits == [13, 13, 14]  # the first hop cannot fill the cap
    assert len(triples) == 40 and triples[-1]["source"] == "N3"

    limits.clear()
    await ops.expand_triples(store, ["e1"], hops=5, cap=8)
    assert limits == [1, 1, 2, 2, 2]  # every hop runs; the rest rolls forward


async def test_expand_triples_runs_every_hop_on_a_small_cap():
    limits: list[int] = []

    def respond(cypher: str, params: dict) -> Any:
        limits.append(params["limit"])
        hop = len(limits)
        return result(
            [[f"N{hop}", "R", f"M{hop}-{i}", "", f"h{hop}-{i}", hop * 100 + i] for i in range(40)][
                : params["limit"]
            ]
        )

    store = FakeGraphStore([("r.src_name", respond)])
    # retrieve's cap for top_k=2 is 8: all three hops must get rows.
    triples = await ops.expand_triples(store, ["e1"], hops=3, cap=8)
    assert len(limits) == 3 and all(limit > 0 for limit in limits)
    assert sum(limits) == 8 and len(triples) == 8 and triples[-1]["source"] == "N3"

    limits.clear()
    await ops.expand_triples(store, ["e1"], hops=4, cap=2)
    assert limits == [1, 1]  # the cap runs out before the last hops


async def test_expand_triples_reads_table_edges_and_matches_seed_pairs_once():
    store = FakeGraphStore([("r.src_name", lambda c, p: result([]))])
    await ops.expand_triples(store, ["e1", "e2"], hops=1)
    cypher = store.calls[0]["cypher"]
    # The endpoints' current names come first: a dedup merge leaves the
    # extraction-time src_name/tgt_name on a moved edge. Edges written from
    # tables carry neither; an endpoint with no name shows its id.
    assert "coalesce(startNode(r).name, r.src_name, startNode(r).id) AS src" in cypher
    assert "coalesce(endNode(r).name, r.tgt_name, endNode(r).id) AS tgt" in cypher
    assert "ORDER BY dist, src, r.rel_type, tgt, b.id" in cypher
    assert "0.0 AS dist" in cypher  # no question vector: name order
    # An edge between two seeds is matched from one side only.
    assert "(NOT b.id IN $ids OR a.id <= b.id)" in cypher
    assert store.calls[0]["read_only"] is True


async def test_expand_triples_prefers_the_edge_description_to_its_fact():
    store = FakeGraphStore([("r.src_name", lambda c, p: result([]))])
    await ops.expand_triples(store, ["e1"], hops=1)
    cypher = store.calls[0]["cypher"]
    assert "WHEN trim(coalesce(r.description, '')) <> '' THEN r.description" in cypher
    assert "ELSE coalesce(r.fact, '') END" in cypher


async def test_expand_triples_ranks_by_the_question_when_given_its_vector():
    store = FakeGraphStore([("r.src_name", lambda c, p: result([]))])
    await ops.expand_triples(store, ["e1"], hops=1, query_vector=[0.1, 0.2])
    call = store.calls[0]
    assert "vec.cosineDistance(r.embedding, vecf32($qv))" in call["cypher"]
    assert "CASE WHEN r.embedding IS NULL THEN 3.0" in call["cypher"]
    assert call["params"]["qv"] == [0.1, 0.2]


async def test_expand_triples_shows_a_relationship_tables_values():
    rows = [
        [
            "Unit 7",
            "CONSUMED",
            "Batch 3",
            "(Unit, CONSUMED, Batch)",
            "b3",
            1,
            [["consumed_batch__consumed_on", "2024-01-01"], ["consumed_batch__qty", 5]],
        ],
    ]
    store = FakeGraphStore([("r.src_name", lambda c, p: result(rows))])
    triples = await ops.expand_triples(store, ["u7"], hops=1)
    assert triples == [
        {
            "source": "Unit 7",
            "type": "CONSUMED",
            "target": "Batch 3",
            "fact": "consumed_batch__consumed_on: 2024-01-01; consumed_batch__qty: 5",
        }
    ]


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
