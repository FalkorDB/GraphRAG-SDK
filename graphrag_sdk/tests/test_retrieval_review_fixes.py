"""Review fixes for retrieval: bounded values, same-name rows, walk dedup, Cypher rendering."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from graphrag_sdk.retrieval.strategies.local import _entity_text
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval
from graphrag_sdk.retrieval.strategies.relationship_expansion import expand_relationships
from graphrag_sdk.retrieval.strategies.result_assembly import (
    assemble_raw_result,
    format_relation_line,
)
from graphrag_sdk.tools import graph_ops as ops

from .conftest import MockEmbedder, MockLLM
from .test_tools_graph_ops import FakeGraphStore, result


def test_an_entity_with_no_name_is_skipped_not_raised():
    raw = assemble_raw_result(
        [
            ("x", {"name": None, "description": "orphan"}),
            ("a", {"name": "Acme", "description": ""}),
        ],
        [],
        [],
        [],
    )
    assert "- Acme" in str(raw.records)


def test_local_entity_attributes_are_bounded():
    entity = {
        "id": "acme__org",
        "labels": ["Org"],
        "properties": {"name": "Acme", **{f"col{i}": "x" * 50 for i in range(20)}},
    }
    text = _entity_text(entity)
    assert len(text) < 300 and text.endswith("…]")


class Chunk:  # the shape of falkordb.Node
    labels = ["Chunk"]
    properties = {"text": "Alice joined Acme in 2019.", "index": 0, "embedding": [0.123]}


async def test_a_returned_chunk_node_shows_its_text():
    from graphrag_sdk.core.models import LLMResponse
    from graphrag_sdk.retrieval.strategies.cypher_generation import execute_cypher_retrieval

    rows = MagicMock(header=[[1, "c"]], result_set=[[Chunk()]])
    graph = MagicMock()
    graph.query_raw = AsyncMock(side_effect=[rows, MagicMock(result_set=[])])
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=LLMResponse(content="```cypher\nMATCH (c:Chunk) RETURN c LIMIT 25\n```")
    )
    facts, _ = await execute_cypher_retrieval(graph, llm, "When did Alice join?")
    assert facts == ["c: Alice joined Acme in 2019."]


async def test_lines_already_shown_are_left_out_before_the_cap():
    def rows(n, rel, src, tgt):
        return [[f"{src}{i}", rel, f"{tgt}{i}", "", "", 0.0, f"{tgt.lower()}{i}"] for i in range(n)]

    store = MagicMock()
    store.query_raw = AsyncMock(
        side_effect=[
            MagicMock(result_set=rows(30, "KNOWS", "A", "B")),
            MagicMock(result_set=rows(10, "WORKS_AT", "B", "C")),
            MagicMock(result_set=[]),
        ]
    )
    shown = {f"A{i} —[KNOWS]→ B{i}" for i in range(5)}
    lines = await expand_relationships(store, [("a__x", {"name": "A"})], 20, exclude=shown)
    assert not shown & set(lines)
    deeper = sum(1 for line in lines if "WORKS_AT" in line)
    assert len(lines) == 20 and deeper == 6


async def test_enrich_entities_card_shows_every_merged_label():
    row = ["e1", "Ada", "", ["__Entity__", "Person", "Engineer"], {"merged_labels": "Engineer"}]
    store = FakeGraphStore([("UNWIND $ids", lambda c, p: result([row]))])
    card = (await ops.enrich_entities(store, ["e1"]))["e1"]
    assert card["label"] == "Person, Engineer"


async def test_vector_fact_shows_the_edges_table_values():
    from graphrag_sdk.retrieval.strategies.entity_discovery import search_relates_edges

    vs = MagicMock()
    vs.search_relationships = AsyncMock(
        return_value=[
            {
                "src_name": "Alice",
                "type": "CONSUMED",
                "tgt_name": "Batch 7",
                "fact": "(Alice, CONSUMED, Batch 7)",
                "score": 0.9,
                "edge_values": [["consumed__consumed_on", "2024-03-01"]],
            }
        ]
    )
    facts, _ = await search_relates_edges(vs, [0.1] * 8)
    assert facts[0][0] == "Alice —[CONSUMED]→ Batch 7 [consumed__consumed_on: 2024-03-01]"


async def test_search_relationships_returns_edge_values():
    from graphrag_sdk.storage.vector_store import VectorStore

    conn = MagicMock()
    row = ["A", "R", "B", "", "", "a", "b", "", "", 0.9, [["t__col", 5]]]
    conn.query = AsyncMock(return_value=MagicMock(result_set=[row]))
    vs = VectorStore(conn, embedder=None)
    hits = await vs.search_relationships([0.1] * 8, top_k=1)
    assert hits[0]["edge_values"] == [["t__col", 5]]
    assert "AS edge_values" in conn.query.call_args[0][0]


async def test_entity_attribute_query_leaves_dropped_values_in_the_database():
    from graphrag_sdk.retrieval.strategies.entity_discovery import fetch_entity_attributes

    store = MagicMock()
    store.query_raw = AsyncMock(return_value=MagicMock(result_set=[]))
    await fetch_entity_attributes(store, ["e1"])
    assert "NOT k ENDS WITH '__dropped'" in store.query_raw.call_args[0][0]


async def test_later_rounds_exclude_edges_already_returned():
    from graphrag_sdk.retrieval.strategies.relationship_expansion import expand_relationships

    store = MagicMock()
    store.query_raw = AsyncMock(
        side_effect=[
            MagicMock(result_set=[["A", "KNOWS", "B", "", "", 0.1, "b", [], 7]]),
            MagicMock(result_set=[["B", "WORKS_AT", "C", "", "", 0.2, "c", [], 8]]),
            MagicMock(result_set=[]),
        ]
    )
    await expand_relationships(store, [("a", {"name": "A"})], 20, query_vector=[0.1])
    calls = store.query_raw.call_args_list
    assert all("NOT ID(r) IN $seen_edges" in c[0][0] for c in calls)
    assert [c[0][1]["seen_edges"] for c in calls] == [[], [7], [7, 8]]


async def test_local_retrieval_leaves_out_inactive_chunks():
    from graphrag_sdk.core.context import Context
    from graphrag_sdk.retrieval.strategies.local import LocalRetrieval

    vs = MagicMock()
    vs.search_chunks = AsyncMock(
        return_value=[{"id": "live", "text": "old"}, {"id": "pend", "text": "new"}]
    )
    graph = MagicMock()
    graph.query_raw = AsyncMock(return_value=MagicMock(result_set=[["pend"]]))
    graph.get_connected_entities = AsyncMock(return_value=[])
    strategy = LocalRetrieval(graph, vs, MockEmbedder(dimension=8))
    out = await strategy.search("q", Context())
    assert [i.metadata["chunk_id"] for i in out.items] == ["live"]


async def test_hidden_facts_do_not_remove_expanded_relationships(monkeypatch):
    from graphrag_sdk.retrieval.strategies import multi_path

    from .conftest import MockEmbedder

    vs = MagicMock()
    for m in (
        "search_chunks",
        "search_entities",
        "fulltext_search_chunks",
        "fulltext_search_entities",
    ):
        setattr(vs, m, AsyncMock(return_value=[]))
    vs.search_relationships = AsyncMock(
        return_value=[{"src_name": "S0", "tgt_name": "T0", "type": "R", "score": 0.9}]
    )
    graph = MagicMock()
    graph.query_raw = AsyncMock(return_value=MagicMock(result_set=[]))

    async def expand(*args, **kwargs):
        assert kwargs["exclude"] == set()  # no fact shown, so nothing excluded
        return ["S0 —[R]→ T0"]

    monkeypatch.setattr(multi_path, "expand_relationships", expand)
    s = MultiPathRetrieval(
        graph_store=graph,
        vector_store=vs,
        embedder=MockEmbedder(dimension=8),
        llm=MockLLM(responses=["S0"]),
    )
    result = await s.search("q", max_facts_out=0)
    sections = {i.metadata["section"]: i.content for i in result.items}
    assert "facts" not in sections
    assert "S0 —[R]→ T0" in sections.get("relationships", "")


async def test_names_are_lowered_by_falkordb_not_python():
    from graphrag_sdk.retrieval.strategies.cypher_generation import _resolve_entities

    graph = MagicMock()
    graph.query_raw = AsyncMock(return_value=MagicMock(result_set=[]))
    await _resolve_entities(graph, ["İstanbul", "istanbul", "Ankara"], None)
    cypher, params = graph.query_raw.call_args[0]
    assert "[n IN $names | toLower(n)]" in cypher
    # Python lowers the two spellings differently, so both are sent.
    assert params["names"] == ["İstanbul", "istanbul", "Ankara"]


async def test_expand_triples_caps_relationship_table_values():
    class Store:
        async def query_raw(self, cypher, params=None, *, read_only=False, timeout=None):
            return SimpleNamespace(
                result_set=[["A", "REL", "B", "", "e2", 10, [["rel__note", "x" * 5000]]]]
            )

    triples = await ops.expand_triples(Store(), ["e1"], hops=1, cap=10)
    assert len(triples[0]["fact"]) < 400


async def test_same_name_edges_with_different_values_are_both_kept():
    rows = [
        # src, rel, tgt, description, fact, distance, far end, edge values
        ["John Smith", "WORKS_AT", "Acme", "", "", 0.1, "acme", [["works__since", 2019]]],
        ["John Smith", "WORKS_AT", "Acme", "", "", 0.2, "acme", [["works__since", 2023]]],
        ["John Smith", "WORKS_AT", "Acme", "", "", 0.3, "acme", [["works__since", 2023]]],
    ]

    async def query_raw(cypher, params=None):
        return MagicMock(result_set=rows if params["eids"] != ["acme"] else [])

    store = MagicMock()
    store.query_raw = AsyncMock(side_effect=query_raw)
    lines = await expand_relationships(
        store,
        [("js1", {"name": "John Smith"}), ("js2", {"name": "John Smith"})],
        query_vector=[0.1],
    )
    assert lines == [
        "John Smith —[WORKS_AT]→ Acme [works__since: 2019]",
        "John Smith —[WORKS_AT]→ Acme [works__since: 2023]",
    ]


async def test_resolver_ranks_direct_names_before_aliases():
    from graphrag_sdk.retrieval.strategies.cypher_generation import _resolve_entities

    graph = MagicMock()
    graph.query_raw = AsyncMock(return_value=MagicMock(result_set=[]))
    await _resolve_entities(graph, ["Acme"], None)
    cypher = graph.query_raw.call_args[0][0]
    assert "ORDER BY CASE WHEN toLower(e.name) IN names THEN 0 ELSE 1 END, e.id" in cypher


def test_same_name_descriptions_that_differ_past_the_cut_are_one_line():
    shared = "word " * 100
    raw = assemble_raw_result(
        [
            ("acme1", {"name": "Acme", "description": shared + "first tail"}),
            ("acme2", {"name": "Acme", "description": shared + "second tail"}),
        ],
        [],
        [],
        [],
    )
    assert str(raw.records).count("- Acme") == 1


def test_bare_name_gives_way_to_a_later_described_entity():
    result = assemble_raw_result(
        [
            ("x1", {"name": "Acme", "description": ""}),
            ("x2", {"name": "Acme", "description": "", "attributes": [("employees", 1200)]}),
            ("x3", {"name": "Beta", "description": "A rival."}),
        ],
        [],
        [],
        [],
        max_entities=2,
    )
    entities = next(r["content"] for r in result.records if r["section"] == "entities")
    assert entities == "## Key Entities\n- Acme (employees: 1200)\n- Beta: A rival."


def test_fact_keeps_a_leading_qualifier_that_is_not_the_triple():
    assert format_relation_line("A", "HAS", "B", fact="(2024 estimate) 5M people") == (
        "A —[HAS]→ B: (2024 estimate) 5M people"
    )
    assert format_relation_line("A", "HAS", "B", fact="(A, HAS, B): 5M people") == (
        "A —[HAS]→ B: 5M people"
    )


def test_nodes_inside_lists_and_maps_render_as_name_and_description():
    from graphrag_sdk.retrieval.strategies.cypher_generation import _render_value

    class Node:  # the shape of falkordb.Node
        labels = ["Person"]
        properties = {"name": "Alice", "description": "An engineer.", "embedding": [0.123]}

    assert _render_value([Node()]) == "[Alice: An engineer.]"
    assert _render_value({"p": Node()}) == "{p: Alice: An engineer.}"


async def test_acronym_is_kept_out_of_contains_whatever_case_comes_first():
    from graphrag_sdk.retrieval.strategies.entity_discovery import discover_entities

    calls = []

    async def capture(cypher, params=None):
        calls.append((cypher, params or {}))
        return MagicMock(result_set=[])

    graph = MagicMock()
    graph.query_raw = AsyncMock(side_effect=capture)
    vector = MagicMock()
    vector.fulltext_search_entities = AsyncMock(return_value=[])
    await discover_entities(graph, vector, ["us", "US", "Acme"], [])
    contains = [p for c, p in calls if "CONTAINS toLower(kw)" in c]
    assert contains and contains[0]["keywords"] == ["Acme"]


async def test_relationship_search_breaks_ties_on_the_edge_text():
    from graphrag_sdk.storage.vector_store import VectorStore

    conn = MagicMock()
    conn.query = AsyncMock(return_value=MagicMock(result_set=[]))
    store = VectorStore(conn)
    await store.search_relationships(query_vector=[0.1] * 8, top_k=2)
    searches = [c.args[0] for c in conn.query.call_args_list if "ORDER BY" in c.args[0]]
    assert len(searches) == 2  # the index query, then the fallback scan
    assert all("type, description, fact" in q for q in searches)


# ── Review round: part 1 ────────────────────────────────────────


async def test_local_retrieval_refills_slots_of_inactive_chunks():
    from graphrag_sdk.core.context import Context
    from graphrag_sdk.retrieval.strategies.local import LocalRetrieval

    hits = [{"id": "pend", "text": "new"}, {"id": "a", "text": "a"}, {"id": "b", "text": "b"}]

    async def search_chunks(query_vector, top_k):
        return hits[:top_k]

    vs = MagicMock()
    vs.search_chunks = AsyncMock(side_effect=search_chunks)

    async def query_raw(cypher, params=None):
        return MagicMock(result_set=[["pend"]] if "pend" in params["ids"] else [])

    graph = MagicMock()
    graph.query_raw = AsyncMock(side_effect=query_raw)
    graph.get_connected_entities = AsyncMock(return_value=[])
    strategy = LocalRetrieval(graph, vs, MockEmbedder(dimension=8), top_k=2)
    out = await strategy.search("q", Context())
    assert [i.metadata["chunk_id"] for i in out.items] == ["a", "b"]


async def test_vector_slots_of_inactive_chunks_are_refilled():
    from graphrag_sdk.retrieval.strategies.chunk_retrieval import retrieve_chunks

    hits = [{"id": f"c{i}", "text": f"t{i}"} for i in range(6)]

    async def search_chunks(query_vector, top_k):
        return hits[:top_k]

    vs = MagicMock()
    vs.fulltext_search_chunks = AsyncMock(return_value=[])
    vs.search_chunks = AsyncMock(side_effect=search_chunks)

    async def query_raw(cypher, params=None):
        if "pending_delete" in cypher and "RETURN DISTINCT c.id" in cypher:
            return MagicMock(result_set=[[c] for c in ("c0", "c1") if c in params["ids"]])
        return MagicMock(result_set=[])

    graph = MagicMock()
    graph.query_raw = AsyncMock(side_effect=query_raw)
    chunks, _, _ = await retrieve_chunks(vs, graph, "q", [0.1], [], [], [], vector_top_k=3)
    assert list(chunks) == ["c2", "c3", "c4"]


def test_a_contextual_chunk_node_shows_the_documents_own_text():
    from graphrag_sdk.retrieval.strategies.cypher_generation import _render_value

    node = SimpleNamespace(
        labels=["Chunk"],
        properties={
            "text": "Context: about Acme.\n\nAlice joined.",
            "original_chunk": "Alice joined.",
        },
    )
    assert _render_value(node) == "Alice joined."


async def test_nodes_inside_collected_lists_seed_entities():
    from graphrag_sdk.core.models import LLMResponse
    from graphrag_sdk.retrieval.strategies.cypher_generation import execute_cypher_retrieval

    alice = SimpleNamespace(labels=["Person"], properties={"name": "Alice"})
    rows = MagicMock(header=[[1, "people"]], result_set=[[[alice], {"p": alice}]])
    graph = MagicMock()
    graph.query_raw = AsyncMock(
        side_effect=[rows, MagicMock(result_set=[["alice__person", "Alice", ""]])]
    )
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=LLMResponse(
            content="```cypher\nMATCH (p:Person) RETURN collect(p) AS people LIMIT 5\n```"
        )
    )
    _, entities = await execute_cypher_retrieval(graph, llm, "Who?")
    assert entities == {"alice__person": {"name": "Alice", "description": ""}}
    assert graph.query_raw.call_args_list[1][0][1]["names"] == ["Alice"]


async def test_walk_breaks_ties_on_the_edge_before_the_limit():
    store = MagicMock()
    store.query_raw = AsyncMock(return_value=MagicMock(result_set=[]))
    await expand_relationships(store, [("a", {"name": "A"})], 20, query_vector=[0.1])
    cypher = store.query_raw.call_args_list[0][0][0]
    assert "r.rel_type ASC, ID(r) ASC" in cypher
    assert cypher.index("ID(r) ASC") < cypher.index("LIMIT")


async def test_expand_triples_keeps_same_name_edges_with_different_values():
    rows = [
        ["John Smith", "WORKS_AT", "Acme", "", "acme", 1, [["works__since", 2019]]],
        ["John Smith", "WORKS_AT", "Acme", "", "acme", 2, [["works__since", 2023]]],
        ["John Smith", "WORKS_AT", "Acme", "", "acme", 3, [["works__since", 2023]]],
    ]
    store = FakeGraphStore([("r.src_name", lambda c, p: result(rows))])
    triples = await ops.expand_triples(store, ["js1", "js2"], hops=1, cap=10)
    assert [t["fact"] for t in triples] == ["works__since: 2019", "works__since: 2023"]
