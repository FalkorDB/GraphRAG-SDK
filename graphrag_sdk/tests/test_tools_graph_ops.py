"""graph_ops behavior + toolkit schema/cypher_read/entity (stubbed stores)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from falkordb import Edge, Node

from graphrag_sdk.core.exceptions import ReadOnlyViolation
from graphrag_sdk.core.models import Attribute, Entity, Ontology, Relation
from graphrag_sdk.tools import GraphRAGToolkit
from graphrag_sdk.tools.graph_ops import (
    chunk_documents,
    convert_query_result,
    enrich_entities,
    expand_triples,
    find_entity_matches,
)


def _res(rows, header=None):
    return SimpleNamespace(result_set=rows, header=header or [])


# ── convert_query_result ─────────────────────────────────────────


def test_convert_query_result_nodes_edges_and_columns():
    node = Node(
        node_id=1,
        alias="n",
        labels=["Person", "__Entity__"],
        properties={"name": "Alice", "embedding": [0.1] * 4, "source_chunk_ids": ["c1"]},
    )
    edge = Edge(
        src_node=1,
        relation="RELATES",
        dest_node=2,
        edge_id=9,
        properties={"fact": "x", "embedding": [0.2] * 4},
    )
    res = SimpleNamespace(
        header=[[1, "n"], [1, "r"], [1, "k"]], result_set=[[node, edge, "plain\x00text"]]
    )
    cr = convert_query_result(res, limit=10, limit_injected=True)
    assert cr.columns == ["n", "r", "k"]
    n_dict, e_dict, s = cr.rows[0]
    assert n_dict["labels"] == ["Person", "__Entity__"]
    assert "embedding" not in n_dict["properties"]
    assert "source_chunk_ids" not in n_dict["properties"]
    assert n_dict["properties"]["name"] == "Alice"
    assert e_dict["type"] == "RELATES" and e_dict["src"] == 1 and e_dict["dst"] == 2
    assert "embedding" not in e_dict["properties"]
    assert s == "plaintext"  # control char stripped
    assert cr.truncated is False  # 1 row != limit
    assert cr.model_dump_json()  # JSON-serializable end to end


def test_convert_truncated_heuristic():
    res = SimpleNamespace(header=[[1, "x"]], result_set=[[1], [2]])
    assert convert_query_result(res, limit=2, limit_injected=True).truncated is True
    assert convert_query_result(res, limit=2, limit_injected=False).truncated is False
    assert convert_query_result(res, limit=5, limit_injected=True).truncated is False


# ── graph_ops helpers ────────────────────────────────────────────


async def test_enrich_entities_maps_rows_and_short_circuits():
    store = MagicMock()
    store.query_raw = AsyncMock(
        return_value=_res(
            [
                [
                    "e1",
                    "Alice",
                    "Engineer",
                    ["Person", "__Entity__"],
                    {
                        "id": "e1",
                        "name": "Alice",
                        "description": "Engineer",
                        "seniority": "senior",
                        "embedding": [0.1],
                        "source_chunk_ids": ["c1"],
                    },
                ]
            ]
        )
    )
    cards = await enrich_entities(store, ["e1"])
    card = cards["e1"]
    assert card.name == "Alice" and card.label == "Person"
    assert card.properties == {"seniority": "senior"}  # bulky/system props dropped

    store.query_raw.reset_mock()
    assert await enrich_entities(store, []) == {}
    store.query_raw.assert_not_awaited()


async def test_expand_triples_dedupes_frontier_and_caps():
    store = MagicMock()
    store.query_raw = AsyncMock(
        return_value=_res(
            [
                ["A", "REL", "B", "", "e2"],
                ["A", "REL", "B", "", "e2"],  # duplicate triple
                ["B", "REL2", "C", "evidence", "e3"],
            ]
        )
    )
    triples = await expand_triples(store, ["e1"], hops=1, cap=10)
    assert len(triples) == 2
    assert triples[0].fact is None and triples[1].fact == "evidence"
    assert store.query_raw.await_count == 1

    store.query_raw.reset_mock()
    triples = await expand_triples(store, ["e1"], hops=2, cap=10)
    assert store.query_raw.await_count == 2  # one query per hop
    # second hop frontier excludes already-visited ids
    second_ids = store.query_raw.await_args_list[1].args[1]["ids"]
    assert "e1" not in second_ids and set(second_ids) == {"e2", "e3"}

    capped = await expand_triples(store, ["e1"], hops=1, cap=1)
    assert len(capped) == 1


async def test_chunk_documents_maps_ids():
    store = MagicMock()
    store.query_raw = AsyncMock(return_value=_res([["c1", "doc-a", "docs/a.md"]]))
    mapping = await chunk_documents(store, ["c1"])
    assert mapping == {"c1": ("doc-a", "docs/a.md")}
    store.query_raw.reset_mock()
    assert await chunk_documents(store, []) == {}
    store.query_raw.assert_not_awaited()


async def test_find_entity_matches_passes_rank_ordering_through():
    store = MagicMock()
    store.query_raw = AsyncMock(
        return_value=_res(
            [
                ["e1", "Alice", "Engineer", ["Person", "__Entity__"], {}, 0],
                ["e9", "Alice Smith", None, ["Person", "__Entity__"], {}, 2],
            ]
        )
    )
    matches = await find_entity_matches(store, "Alice")
    assert [eid for eid, _ in matches] == ["e1", "e9"]
    assert matches[0][1].name == "Alice" and matches[1][1].name == "Alice Smith"


# ── toolkit read methods ─────────────────────────────────────────


def _stub_rag() -> MagicMock:
    rag = MagicMock()
    rag._conn.query = AsyncMock(return_value=_res([]))
    rag._graph_store.query_raw = AsyncMock(return_value=_res([]))
    return rag


async def test_cypher_read_guard_fires_before_connection():
    rag = _stub_rag()
    tk = GraphRAGToolkit(rag)
    with pytest.raises(ReadOnlyViolation):
        await tk.cypher_read("CREATE (n) RETURN n")
    rag._conn.query.assert_not_awaited()


async def test_cypher_read_injects_limit_and_forwards_kwargs():
    rag = _stub_rag()
    rag._conn.query = AsyncMock(
        return_value=SimpleNamespace(header=[[1, "name"]], result_set=[["Alice"], ["Bob"]])
    )
    tk = GraphRAGToolkit(rag)
    result = await tk.cypher_read("MATCH (n) RETURN n.name", {"x": 1}, limit=2, timeout_ms=1234)
    query_arg = rag._conn.query.await_args.args[0]
    assert query_arg.endswith("LIMIT 2")
    assert rag._conn.query.await_args.kwargs == {"params": {"x": 1}, "timeout": 1234}
    assert result.columns == ["name"] and result.row_count == 2
    assert result.truncated is True


async def test_cypher_read_respects_existing_limit():
    rag = _stub_rag()
    tk = GraphRAGToolkit(rag)
    await tk.cypher_read("MATCH (n) RETURN n LIMIT 3")
    assert rag._conn.query.await_args.args[0] == "MATCH (n) RETURN n LIMIT 3"


async def test_schema_merges_ontology_and_live_counts():
    rag = _stub_rag()
    rag.get_ontology = AsyncMock(
        return_value=Ontology(
            entities=[
                Entity(
                    label="Person",
                    description="A human",
                    properties=[Attribute(name="seniority")],
                ),
                Entity(label="Location"),  # declared but absent from the live graph
            ],
            relations=[Relation(label="WORKS_AT", patterns=[("Person", "Organization")])],
        )
    )
    rag.get_statistics = AsyncMock(return_value={"node_count": 7, "edge_count": 9})

    async def route(cypher, params=None):
        if "UNWIND labels(n)" in cypher:
            return _res([["Person", 2], ["Organization", 1]])
        if "r.rel_type AS t" in cypher:
            return _res([["WORKS_AT", 1]])
        return _res([])

    rag._graph_store.query_raw = AsyncMock(side_effect=route)
    tk = GraphRAGToolkit(rag)
    schema = await tk.schema()
    by_label = {e.label: e for e in schema.entities}
    assert by_label["Person"].count == 2
    assert by_label["Person"].description == "A human"
    assert by_label["Person"].properties == ["seniority"]
    assert by_label["Organization"].count == 1 and by_label["Organization"].description is None
    assert by_label["Location"].count == 0  # declared labels always listed
    assert schema.relations[0].label == "WORKS_AT"
    assert schema.relations[0].patterns == [("Person", "Organization")]
    assert schema.node_count == 7 and schema.edge_count == 9


async def test_entity_found_with_nearby_hops_and_documents():
    rag = _stub_rag()
    calls: list[str] = []

    async def route(cypher, params=None):
        calls.append(cypher)
        if "CONTAINS toLower($name)" in cypher:
            return _res(
                [
                    [
                        "e1",
                        "Alice",
                        "Engineer",
                        ["Person", "__Entity__"],
                        {"seniority": "senior", "embedding": [0.1]},
                        0,
                    ],
                    ["e9", "Alice Smith", None, ["Person", "__Entity__"], {}, 2],
                ]
            )
        if "r.src_name" in cypher:
            return _res([["Alice", "WORKS_AT", "Acme Corp", "employment", "e2"]])
        if "MENTIONED_IN" in cypher:
            return _res([["doc-a", "docs/a.md"]])
        return _res([])

    rag._graph_store.query_raw = AsyncMock(side_effect=route)
    tk = GraphRAGToolkit(rag)
    er = await tk.entity("Alice", hops=2)
    assert er.found and er.entity is not None
    assert er.entity.name == "Alice" and er.entity.label == "Person"
    assert er.entity.properties == {"seniority": "senior"}
    assert er.nearby == ["Alice Smith"]
    assert any(t.type == "WORKS_AT" and t.target == "Acme Corp" for t in er.neighbors)
    assert er.documents[0].document_id == "doc-a"
    assert len([c for c in calls if "r.src_name" in c]) == 2  # hops=2 -> two frontier queries


async def test_entity_not_found():
    rag = _stub_rag()
    tk = GraphRAGToolkit(rag)
    er = await tk.entity("Zorp")
    assert er.found is False and er.entity is None
    assert er.to_llm_text().startswith("No entity")
    assert rag._graph_store.query_raw.await_count == 1  # only the match query ran
