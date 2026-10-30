"""Real-FalkorDB integration tests for graphrag_sdk.tools and the MCP server.

Every tool runs against a live graph built by scripted extraction (no LLM
key needed). Gated on RUN_INTEGRATION=1; run against both FalkorDB engines.
"""

from __future__ import annotations

import json
import os
from uuid import uuid4

import pytest

from graphrag_sdk.core.exceptions import DatabaseError, ReadOnlyViolation
from graphrag_sdk.tools import GraphRAGToolkit, ToolError
from graphrag_sdk.tools import graph_ops as ops

# Both markers are load-bearing: `-m integration` selects the file in the CI
# integration job; the skipif keeps it out of the plain unit run.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION") != "1",
        reason="Set RUN_INTEGRATION=1 to run real-FalkorDB tests",
    ),
]


def _step2(entities, relationships=()):
    """One scripted GraphExtraction step-2 response (entities + relationships)."""
    return json.dumps(
        {
            "entities": [{"name": n, "type": t, "description": d} for n, t, d in entities],
            "relationships": [
                {
                    "source": s,
                    "target": o,
                    "type": r,
                    "description": d,
                    "keywords": "",
                    "weight": 0.9,
                }
                for s, r, o, d in relationships
            ],
        }
    )


DOCS = [
    ("doc-alice", "Alice is a software engineer at Acme Corp."),
    ("doc-bob", "Bob leads the data team at Acme Corp."),
    ("doc-hq", "Acme Corp is headquartered in Berlin."),
]


def _scripted_llm():
    from .conftest import MockLLM

    return MockLLM(
        responses=[
            _step2(
                [
                    ("Alice", "Person", "Engineer at Acme"),
                    ("Acme Corp", "Organization", "A tech company"),
                ],
                [("Alice", "WORKS_AT", "Acme Corp", "Alice is employed at Acme")],
            ),
            _step2(
                [
                    ("Bob", "Person", "Data lead at Acme"),
                    ("Acme Corp", "Organization", "A tech company"),
                ],
                [("Bob", "WORKS_AT", "Acme Corp", "Bob works at Acme")],
            ),
            _step2(
                [
                    ("Acme Corp", "Organization", "A tech company"),
                    ("Berlin", "Location", "Capital of Germany"),
                ],
                [("Acme Corp", "HEADQUARTERED_IN", "Berlin", "Acme HQ is in Berlin")],
            ),
            _step2(
                [("Carol", "Person", "CFO of Acme"), ("Acme Corp", "Organization", "A company")],
                [("Carol", "WORKS_AT", "Acme Corp", "Carol is the CFO of Acme")],
            ),
            "Alice, Acme Corp",  # keyword extraction during retrieve (clamped)
        ]
    )


@pytest.fixture
async def toolkit(real_falkordb_rag_factory):
    """A toolkit over a live graph: Alice and Bob work at Acme Corp, which is
    headquartered in Berlin; Carol is added through the ingest tool."""
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution

    resolver = ExactMatchResolution()
    rag = real_falkordb_rag_factory(llm=_scripted_llm(), resolver=resolver)
    for doc_id, text in DOCS:
        await rag.ingest(text=text, document_id=doc_id, resolver=resolver)
    tk = GraphRAGToolkit(rag)
    ingested = await tk.call(
        "ingest", {"text": "Carol is the CFO of Acme Corp.", "document_id": "doc-carol"}
    )
    assert ingested.document_id == "doc-carol" and not ingested.finalized
    report = await tk.call("finalize", {})
    assert report.finalized
    return tk


async def test_status_and_ontology(toolkit: GraphRAGToolkit):
    status = await toolkit.call("status", {})
    assert status.entities >= 5 and status.relations >= 4
    assert status.documents == 4 and status.chunks >= 4
    labels = {t.label: t.count for t in status.entity_types}
    assert labels.get("Person") == 3 and labels.get("Organization") == 1
    relations = {t.label: t.count for t in status.relation_types}
    assert relations.get("WORKS_AT") == 3 and relations.get("HEADQUARTERED_IN") == 1
    assert status.hint is None

    onto = await toolkit.call("get_ontology", {})
    assert {"Person", "Organization", "Location"} <= {t.label for t in onto.entity_types}
    assert "WORKS_AT" in {t.label for t in onto.relation_types}


async def test_lookup_entity(toolkit: GraphRAGToolkit):
    exact = await toolkit.call("lookup_entity", {"name": "alice"})
    assert exact.matches[0].name == "Alice" and exact.matches[0].label == "Person"
    partial = await toolkit.call("lookup_entity", {"name": "Acme"})
    assert partial.matches and partial.matches[0].name == "Acme Corp"
    none = await toolkit.call("lookup_entity", {"name": "Zebulon"})
    assert none.matches == [] and none.hint
    assert toolkit.ledger.items == []


async def test_traverse(toolkit: GraphRAGToolkit):
    near = await toolkit.call("traverse", {"start": "Alice", "max_depth": 1})
    assert near.start == "Alice"
    assert any(e.type == "WORKS_AT" and e.target == "Acme Corp" for e in near.edges)
    assert all(e.n for e in near.edges)

    far = await toolkit.call("traverse", {"start": "Alice", "max_depth": 2})
    assert "Berlin" in {n.name for n in far.nodes}
    assert "Bob" in {n.name for n in far.nodes}  # edges are followed both ways

    typed = await toolkit.call(
        "traverse", {"start": "Alice", "via": ["works_at", "headquartered_in"]}
    )
    assert [n.name for n in typed.nodes] == ["Berlin"] and typed.nodes[0].n
    assert typed.via == ["WORKS_AT", "HEADQUARTERED_IN"]

    with pytest.raises(ToolError, match="Unknown relation") as bad_via:
        await toolkit.call("traverse", {"start": "Alice", "via": ["EMPLOYS"]})
    assert "WORKS_AT" in (bad_via.value.hint or "")
    with pytest.raises(ToolError, match="Zebulon"):
        await toolkit.call("traverse", {"start": "Zebulon"})

    budget = await toolkit.call("traverse", {"start": "Acme Corp", "max_expansions": 1})
    assert budget.truncated and budget.stopped == "expansion_budget"


async def test_graph_walk(toolkit: GraphRAGToolkit):
    walk = await toolkit.call("graph_walk", {"start": "Alice", "goal": "Berlin"})
    assert walk.paths and walk.paths[0].nodes[0] == "Alice"
    assert walk.paths[0].nodes[-1] == "Berlin" and walk.paths[0].n
    out = await toolkit.call("graph_walk", {"start": "Alice"})
    assert out.paths and all(len(p.nodes) >= 2 for p in out.paths)


async def test_graph_walk_states_edges_in_their_stored_direction(toolkit: GraphRAGToolkit):
    """The stored edges are Alice-[WORKS_AT]->Acme Corp-[HEADQUARTERED_IN]->Berlin;
    walking from the far end must not reverse them."""
    out = await toolkit.call("graph_walk", {"start": "Acme Corp", "max_depth": 1})
    texts = {toolkit.ledger.get(p.n).content for p in out.paths}
    assert "Acme Corp <-[WORKS_AT]- Alice" in texts
    assert "Acme Corp -[HEADQUARTERED_IN]-> Berlin" in texts
    assert not any("Acme Corp -[WORKS_AT]->" in t for t in texts)

    back = await toolkit.call("graph_walk", {"start": "Berlin", "goal": "Alice"})
    assert toolkit.ledger.get(back.paths[0].n).content == (
        "Berlin <-[HEADQUARTERED_IN]- Acme Corp <-[WORKS_AT]- Alice"
    )
    near = await toolkit.call("graph_walk", {"start": "Alice", "goal": "Berlin", "max_depth": 1})
    assert near.paths == [] and "at most 1 hop " in (near.hint or "")


async def test_cypher_query_renders_paths_as_names(toolkit: GraphRAGToolkit):
    out = await toolkit.call(
        "cypher_query",
        {
            "cypher": "MATCH p = (b:Location {name: 'Berlin'})<-[:RELATES]-(:Organization)"
            "<-[:RELATES]-(a:Person {name: 'Alice'}) RETURN p"
        },
    )
    assert out.rows[0].values == ["(Berlin)<-[HEADQUARTERED_IN]-(Acme Corp)<-[WORKS_AT]-(Alice)"]


async def test_traverse_via_never_returns_the_start(toolkit: GraphRAGToolkit):
    # A second WORKS_AT edge lets Alice -WORKS_AT- Acme -WORKS_AT- Alice match.
    await toolkit.rag._graph_store.query_raw(
        "MATCH (a:__Entity__ {name: 'Alice'}), (c:__Entity__ {name: 'Acme Corp'}) "
        "CREATE (a)-[:RELATES {rel_type: 'WORKS_AT'}]->(c)"
    )
    out = await toolkit.call("traverse", {"start": "Alice", "via": ["WORKS_AT", "WORKS_AT"]})
    assert sorted(n.name for n in out.nodes) == ["Bob", "Carol"]


async def test_cypher_query(toolkit: GraphRAGToolkit):
    counted = await toolkit.call(
        "cypher_query", {"cypher": "MATCH (p:Person) RETURN count(p) AS people"}
    )
    assert counted.columns == ["people"] and counted.rows[0].values == [3]

    typed = await toolkit.call(
        "cypher_query",
        {
            "cypher": "MATCH (p:Person)-[r:RELATES]->(c:Organization) "
            "WHERE r.rel_type = 'WORKS_AT' AND c.name = $org RETURN p.name ORDER BY p.name",
            "params": {"org": "Acme Corp"},
        },
    )
    assert [r.values[0] for r in typed.rows] == ["Alice", "Bob", "Carol"]

    in_map = await toolkit.call(
        "cypher_query",
        {"cypher": "MATCH (a)-[:RELATES {rel_type: 'HEADQUARTERED_IN'}]->(b) RETURN b.name"},
    )
    assert in_map.rows[0].values == ["Berlin"]

    untyped = await toolkit.call(
        "cypher_query",
        {"cypher": "MATCH ()-[r:RELATES]->() RETURN r.rel_type AS t, count(*) AS n ORDER BY t"},
    )
    assert dict(r.values for r in untyped.rows) == {"HEADQUARTERED_IN": 1, "WORKS_AT": 3}

    capped = await toolkit.call(
        "cypher_query", {"cypher": "MATCH (p:Person) RETURN p.name LIMIT 100", "max_rows": 2}
    )
    assert capped.row_count == 2 and capped.truncated and "count" in capped.hint

    passages = await toolkit.call(
        "cypher_query", {"cypher": "MATCH (c:Chunk) RETURN c.text LIMIT 1"}
    )
    assert passages.rows

    vectors = await toolkit.call("cypher_query", {"cypher": "MATCH (p:Person) RETURN p LIMIT 1"})
    assert "embedding" not in json.dumps(vectors.model_dump())


async def test_cypher_query_never_returns_system_nodes(toolkit: GraphRAGToolkit):
    store = toolkit._store
    await store.query_raw(
        "MERGE (c:__GraphRAGConfig__ {id: 'default'}) "
        "SET c.embedding_model = 'secret-model', c.embedding_dimension = 8, "
        "c.sdk_version = '9.9.9-probe'"
    )
    await store.query_raw(
        "CREATE (:__GraphRAGProcessing__ {kind: 'pending', ids: ['probe-id'], attempts: [1]})"
    )
    for label in ("Config", "Processing"):
        # The label name is built at run time, past the guard's text check.
        result = await toolkit.call(
            "cypher_query",
            {
                "cypher": f"MATCH (n) WHERE ('__GraphRAG' + '{label}__') IN labels(n) "
                "RETURN n, properties(n) AS p, [n] AS wrapped"
            },
        )
        dumped = json.dumps(result.model_dump())
        assert result.rows, label
        for leaked in ("secret-model", "9.9.9-probe", "probe-id"):
            assert leaked not in dumped, (label, leaked)
    props = await toolkit.call(
        "cypher_query", {"cypher": "MATCH (p:Person) RETURN properties(p) AS p LIMIT 1"}
    )
    assert props.rows and "embedding" not in json.dumps(props.model_dump())


async def test_cypher_query_aggregates_after_an_inner_limit(real_falkordb_rag_factory):
    """An inner LIMIT larger than max_rows must not be lowered: count() and
    collect() after it see every row it lets through."""
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution

    from .conftest import MockLLM

    rag = real_falkordb_rag_factory(llm=MockLLM(responses=[]), resolver=ExactMatchResolution())
    await rag._graph_store.query_raw(
        "UNWIND range(1, 300) AS i "
        "CREATE (:__Entity__:Person {id: 'p' + toString(i), name: 'P' + toString(1000 + i)})"
    )
    tk = GraphRAGToolkit(rag, read_only=True)
    counted = await tk.call(
        "cypher_query",
        {"cypher": "MATCH (n:Person) WITH n ORDER BY n.name LIMIT 100 RETURN count(n) AS c"},
    )
    assert counted.rows[0].values == [100] and not counted.truncated
    collected = await tk.call(
        "cypher_query",
        {"cypher": "MATCH (n:Person) WITH n LIMIT 100 RETURN collect(n.name) AS names"},
    )
    assert len(collected.rows[0].values[0]) == 100 and not collected.truncated
    rows = await tk.call(
        "cypher_query",
        {"cypher": "MATCH (n:Person) RETURN n.name LIMIT $n", "params": {"n": 40}},
    )
    assert rows.row_count == 25 and rows.truncated
    small = await tk.call(
        "cypher_query",
        {"cypher": "MATCH (n:Person) RETURN n.name ORDER BY n.name LIMIT $n", "params": {"n": 3}},
    )
    assert [r.values[0] for r in small.rows] == ["P1001", "P1002", "P1003"]
    assert not small.truncated
    over = await tk.call("cypher_query", {"cypher": "MATCH (n:Person) RETURN n.name LIMIT 100"})
    assert over.row_count == 25 and over.truncated


@pytest.mark.parametrize(
    "cypher, message",
    [
        ("CREATE (n:Hack) RETURN n", "CREATE"),
        ("MATCH (n) DETACH DELETE n", "DETACH"),
        ("MATCH (n:Person) SET n.name = 'x' RETURN n", "SET"),
        ("CALL db.labels()", "CALL"),
        ("MATCH (n:Employee) RETURN n", "Employee"),
        ("MATCH (n:Person) RETURN n.salary", "salary"),
        ("MATCH (a)-[r:RELATES {rel_type: 'EMPLOYS'}]->(b) RETURN b", "EMPLOYS"),
        ("MATCH (a)-[:WORKS_AT]->(b) RETURN b", "stored as RELATES"),
        ("MATCH (n:__GraphRAGConfig__) RETURN n", "internal"),
        ("MATCH (n:Person) RETURN n.embedding", "embedding"),
    ],
)
async def test_cypher_query_rejections(toolkit: GraphRAGToolkit, cypher: str, message: str):
    before = (await toolkit.call("status", {})).entities
    with pytest.raises(ToolError, match=message) as exc:
        await toolkit.call("cypher_query", {"cypher": cypher})
    assert exc.value.hint
    assert (await toolkit.call("status", {})).entities == before


async def test_cypher_query_runtime_errors(toolkit: GraphRAGToolkit):
    # FalkorDB's own TIMEOUT answer (not only the client-side wait) is a timeout.
    with pytest.raises(ToolError, match="The query timed out after"):
        await toolkit.call(
            "cypher_query",
            {
                "cypher": "UNWIND range(1, 100000000) AS x WITH x WHERE x % 7 = 3 RETURN count(x)",
                "timeout_ms": 100,
            },
        )
    with pytest.raises(ToolError, match="failed"):
        await toolkit.call(
            "cypher_query", {"cypher": "MATCH (p:Person) RETURN toInteger({a: p.name})"}
        )


async def test_the_database_refuses_writes_too(toolkit: GraphRAGToolkit):
    """Below the guard, every tool query runs as GRAPH.RO_QUERY."""
    with pytest.raises(DatabaseError):
        await ops.bounded_query(toolkit.rag._graph_store, "CREATE (n:Hack) RETURN n", None, 2000)
    hacked = await toolkit.call("cypher_query", {"cypher": "MATCH (n) RETURN count(n) AS c"})
    assert hacked.rows


async def test_retrieve(toolkit: GraphRAGToolkit):
    result = await toolkit.call("retrieve", {"question": "Who works at Acme Corp?", "top_k": 8})
    assert result.entities or result.chunks
    numbers = [x.n for x in [*result.entities, *result.relations, *result.facts, *result.chunks]]
    assert numbers == sorted(numbers) and len(set(numbers)) == len(numbers)
    assert all(c.document_id for c in result.chunks)
    assert "embedding" not in json.dumps(result.model_dump())


async def test_retrieve_on_chosen_paths(toolkit: GraphRAGToolkit):
    passages = await toolkit.call(
        "retrieve", {"question": "Who works at Acme Corp?", "paths": ["passages"]}
    )
    assert passages.chunks and not passages.relations
    names = await toolkit.call(
        "retrieve", {"question": "Who works at Acme Corp?", "paths": ["exact_names", "links"]}
    )
    assert names.entities and names.relations and not names.chunks
    assert any(r.type == "WORKS_AT" for r in names.relations)


async def test_run_skill(toolkit: GraphRAGToolkit):
    compared = await toolkit.call(
        "run_skill",
        {"skill": "entity_comparison", "params": {"entity_a": "Alice", "entity_b": "Bob"}},
    )
    assert compared.n and compared.data
    impact = await toolkit.call(
        "run_skill", {"skill": "impact_analysis", "params": {"entity": "Acme Corp"}}
    )
    assert impact.n
    with pytest.raises(ToolError, match="Zebulon"):
        await toolkit.call(
            "run_skill", {"skill": "impact_analysis", "params": {"entity": "Zebulon"}}
        )


async def test_citations_across_tools(toolkit: GraphRAGToolkit):
    session = toolkit.new_session()
    walk = await session.call("traverse", {"start": "Alice", "max_depth": 1})
    rows = await session.call(
        "cypher_query", {"cypher": "MATCH (p:Person) RETURN count(p) AS people"}
    )
    edge, row = walk.edges[0].n, rows.rows[0].n
    checked = await session.call(
        "check_citations",
        {"answer": f"Alice works at Acme Corp [{edge}]. There are 3 people [{row}][999]."},
    )
    assert checked.grounded and checked.cited == [edge, row] and checked.invalid == [999]
    assert {s.tool for s in checked.sources} == {"traverse", "cypher_query"}
    assert "[999]" not in checked.answer

    fresh = await toolkit.new_session().call("check_citations", {"answer": f"x [{edge}]"})
    assert not fresh.grounded


async def test_read_only_toolkit(toolkit: GraphRAGToolkit):
    ro = GraphRAGToolkit(toolkit.rag, read_only=True)
    assert "ingest" not in [s.name for s in ro.tool_specs()]
    with pytest.raises(ReadOnlyViolation):
        await ro.call("ingest", {"text": "x"})
    assert (await ro.call("lookup_entity", {"name": "Bob"})).matches


async def test_mcp_round_trip(toolkit: GraphRAGToolkit):
    pytest.importorskip("mcp")
    from graphrag_sdk.mcp import GraphRAGMCPServer

    from .test_mcp_server import connect

    server = GraphRAGMCPServer(toolkit.rag, read_only=True)
    async with connect(server) as (client, _):
        listed = await client.call_tool("lookup_entity", {"name": "Bob"})
        assert json.loads(listed.content[0].text)["matches"][0]["name"] == "Bob"
        walked = json.loads(
            (await client.call_tool("traverse", {"start": "Bob", "max_depth": 1})).content[0].text
        )
        n = walked["edges"][0]["n"]
        checked = json.loads(
            (
                await client.call_tool(
                    "check_citations", {"answer": f"Bob works at Acme Corp [{n}]."}
                )
            )
            .content[0]
            .text
        )
        assert checked["grounded"] is True
        denied = await client.call_tool("ingest", {"text": "x"})
        assert "Unknown tool" in denied.content[0].text or "read-only" in denied.content[0].text


async def test_for_tenant_isolated_graphs(embedder):
    from graphrag_sdk.core.connection import ConnectionConfig

    from .conftest import MockLLM

    base = ConnectionConfig(
        host=os.getenv("FALKOR_HOST", "localhost"),
        port=int(os.getenv("FALKOR_PORT", "6379")),
        username=os.getenv("FALKOR_USERNAME") or None,
        password=os.getenv("FALKOR_PASSWORD") or None,
        graph_name=f"test_{uuid4().hex[:8]}",
    )
    llm_a = MockLLM(
        responses=[_step2([("Alice", "Person", "Engineer")]), "Alice"],
    )
    tk_a = GraphRAGToolkit.for_tenant(
        base, "tenant-a", llm=llm_a, embedder=embedder, embedding_dimension=embedder.dimension
    )
    tk_b = GraphRAGToolkit.for_tenant(
        base,
        "tenant-b",
        llm=MockLLM(responses=["Alice"]),
        embedder=embedder,
        embedding_dimension=embedder.dimension,
    )
    try:
        assert tk_a.rag._conn.config.graph_name == f"{base.graph_name}__tenant-a"
        assert tk_b.rag._conn.config.graph_name == f"{base.graph_name}__tenant-b"
        await tk_a.ingest("Alice is an engineer.", document_id="doc-a")
        await tk_a.finalize()
        assert (await tk_a.lookup_entity("Alice")).matches
        # Tenant b's graph was never written, so it cannot see a's entities.
        with pytest.raises(ToolError, match="__tenant-b' does not exist"):
            await tk_b.lookup_entity("Alice")
    finally:
        for tk in (tk_a, tk_b):
            try:
                await tk.rag._graph_store.delete_all()
            finally:
                await tk.aclose()


async def test_get_ontology_on_a_fresh_graph_writes_nothing(real_falkordb_rag_factory):
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution

    from .conftest import MockLLM

    rag = real_falkordb_rag_factory(llm=MockLLM(responses=[]), resolver=ExactMatchResolution())
    tk = GraphRAGToolkit(rag, read_only=True)
    onto = await tk.call("get_ontology", {})
    assert onto.entity_types == [] and onto.relation_types == []
    # cypher_query on the missing graph refuses without writing anything.
    with pytest.raises(ToolError, match="does not exist"):
        await tk.call("cypher_query", {"cypher": "MATCH (n) RETURN count(n) AS c"})
    stored = await rag._ontology_store.load()
    assert stored.entities == [] and stored.relations == []


async def test_read_tools_on_a_fresh_graph_create_no_keys(real_falkordb_rag_factory):
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution

    from .conftest import MockLLM

    rag = real_falkordb_rag_factory(llm=MockLLM(responses=[]), resolver=ExactMatchResolution())
    tk = GraphRAGToolkit(rag, read_only=True)
    data_graph = rag._conn.config.graph_name
    await tk.call("get_ontology", {})
    for tool, args in [
        ("retrieve", {"question": "Who founded Acme?"}),
        ("cypher_query", {"cypher": "MATCH (n) RETURN count(n) AS c"}),
        ("lookup_entity", {"name": "Acme"}),
    ]:
        with pytest.raises(ToolError, match=f"Graph '{data_graph}' does not exist"):
            await tk.call(tool, args)
    status = await tk.call("status", {})
    assert "does not exist" in (status.hint or "")
    store = rag._ontology_store
    store._ensure_graph()
    graphs = await store._conn._driver.list_graphs()
    assert store.graph_name not in graphs and data_graph not in graphs
    # Retrieval ran without registering the default types.
    assert rag._ontology_initialized is False


async def test_a_server_timeout_is_a_timeout(toolkit: GraphRAGToolkit):
    # A generous client-side wait, so only FalkorDB's own timeout can fire.
    slow = "UNWIND range(1, 100000000) AS x WITH x WHERE x % 7 = 3 RETURN count(x)"
    with pytest.raises(TimeoutError):
        await ops.bounded_query(toolkit._store, slow, None, 50)
