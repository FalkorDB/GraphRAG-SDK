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
        assert (await tk_b.lookup_entity("Alice")).matches == []
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
    # cypher_query builds the schema, which reads the ontology too.
    await tk.call("cypher_query", {"cypher": "MATCH (n) RETURN count(n) AS c"})
    stored = await rag._ontology_store.load()
    assert stored.entities == [] and stored.relations == []


# ── Agent ingestion over a folder ─────────────────────────────────────

FOLDER = {
    "notes/aspirin.md": "# Aspirin\n\nAspirin treats headache. Aspirin inhibits the COX1 gene.",
    "reports/ibuprofen.txt": "Ibuprofen treats fever.",
}
EXTRACTIONS = {
    "Aspirin treats": _step2(
        [
            ("Aspirin", "Drug", "A painkiller"),
            ("Headache", "Disease", "Pain in the head"),
            ("COX1", "Gene", "Cyclooxygenase 1"),
        ],
        [
            ("Aspirin", "TREATS", "Headache", "Aspirin treats headache"),
            ("Aspirin", "INHIBITS", "COX1", "Aspirin inhibits COX1"),
        ],
    ),
    "Ibuprofen treats": _step2(
        [("Ibuprofen", "Drug", "A painkiller"), ("Fever", "Disease", "High temperature")],
        [("Ibuprofen", "TREATS", "Fever", "Ibuprofen treats fever")],
    ),
}


def _routing_llm():
    """Answers by document text, so files ingested concurrently get their own
    extraction for both steps (the NER step reads ``entities`` too)."""
    from graphrag_sdk.core.models import LLMResponse

    from .conftest import MockLLM

    class RoutingLLM(MockLLM):
        def invoke(self, prompt: str, **kwargs):
            for marker, response in EXTRACTIONS.items():
                if marker in prompt:
                    return LLMResponse(content=response)
            return LLMResponse(content="[]")

        async def ainvoke_messages(self, messages, *, max_retries=3, **kwargs):
            text = "\n".join(str(getattr(m, "content", m)) for m in messages)
            return self.invoke(text)

    return RoutingLLM()


# The default limits allow only GLiNER, which needs a model download; the
# scripted LLM extractor keeps this test offline.
OFFLINE_LIMITS = {
    "chunkers": ("sentence", "fixed", "structural"),
    "extractors": ("llm",),
    "resolvers": ("exact",),
    "max_llm_pairs": 0,
}


async def test_agent_ingests_a_folder(real_falkordb_rag_factory, tmp_path):
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution

    for rel, text in FOLDER.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    rag = real_falkordb_rag_factory(llm=_routing_llm(), resolver=ExactMatchResolution())
    tk = GraphRAGToolkit(rag, files_root=tmp_path, plan_limits=OFFLINE_LIMITS)

    # 1. Look at the folder.
    sampled = await tk.call("preview_files", {})
    assert sampled.total == 2 and sampled.types == {"md": 1, "txt": 1}
    assert "Aspirin treats headache" in sampled.files[0].sample

    # 2. Declare the ontology before anything is stored: no default types.
    await tk.call(
        "set_ontology",
        {
            "entities": [{"label": "Drug"}, {"label": "Disease"}],
            "relations": [{"label": "TREATS", "patterns": [["Drug", "Disease"]]}],
        },
    )
    stored = await rag._ontology_store.load()
    assert {e.label for e in stored.entities} == {"Drug", "Disease"}

    # 3. Ingest the folder with one plan rule per file type.
    plan = [
        {"match": "*.md", "chunker": "structural", "extractor": "llm"},
        {"match": "*", "chunker": "fixed", "extractor": "llm"},
    ]
    first = await tk.call("ingest", {"files": ["**/*"], "plan": plan})
    docs = {d.path: d for d in first.documents}
    assert [d.status for d in first.documents] == ["ingested", "ingested"]
    assert docs["notes/aspirin.md"].plan["chunker"] == "structural"
    assert docs["reports/ibuprofen.txt"].plan["chunker"] == "fixed"
    # Extraction is told the declared types, so COX1 is kept as Unknown and
    # its relation type is dropped: both point at what the ontology misses.
    assert first.pruned is not None and first.pruned.unknown == 1
    assert first.pruned.relations == {"INHIBITS": 1}
    assert "MATCH (n:Unknown)" in first.hint
    unknown = await tk.call("cypher_query", {"cypher": "MATCH (n:Unknown) RETURN n.name AS name"})
    assert [r.values for r in unknown.rows] == [["COX1"]]
    assert "ingest(files=['notes/aspirin.md'], reextract=true)" in first.hint
    assert not first.finalized

    # 4. Same files again: unchanged, nothing re-extracted.
    again = await tk.call("ingest", {"files": ["**/*"], "plan": plan})
    assert {d.status for d in again.documents} == {"unchanged"}
    assert "reextract=true" in again.hint

    # 5. Extend the ontology, then re-extract only the affected file.
    await tk.call(
        "set_ontology",
        {
            "entities": [{"label": "Gene", "description": "A gene"}],
            "relations": [{"label": "INHIBITS", "patterns": [["Drug", "Gene"]]}],
        },
    )
    # Without a plan, the file keeps the plan of its last ingest.
    redo = await tk.call("ingest", {"files": ["notes/aspirin.md"], "reextract": True})
    assert [d.status for d in redo.documents] == ["reextracted"]
    assert redo.documents[0].plan["chunker"] == "structural"
    assert redo.documents[0].plan["extractor"] == "llm"
    assert redo.pruned is None

    # 6. Finalize once, then the graph holds every declared type.
    assert (await tk.call("finalize", {})).finalized
    status = await tk.call("status", {})
    labels = {t.label: t.count for t in status.entity_types}
    assert labels == {"Drug": 2, "Disease": 2, "Gene": 1}
    relations = {t.label: t.count for t in status.relation_types}
    assert relations == {"TREATS": 2, "INHIBITS": 1}
    assert status.documents == 2
    found = await tk.call("lookup_entity", {"name": "COX1"})
    assert found.matches and found.matches[0].label == "Gene"


async def test_agent_ingestion_respects_the_root(real_falkordb_rag_factory, tmp_path):
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution

    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret.md").write_text("Aspirin treats headache.")
    rag = real_falkordb_rag_factory(llm=_routing_llm(), resolver=ExactMatchResolution())
    tk = GraphRAGToolkit(rag, files_root=root, plan_limits=OFFLINE_LIMITS)
    with pytest.raises(ToolError, match="outside the files root"):
        await tk.call("ingest", {"files": ["../secret.md"]})
    with pytest.raises(ToolError, match="outside the files root"):
        await tk.call("preview_files", {"pattern": "../*"})
    assert (await tk.call("status", {})).documents == 0
    with pytest.raises(ReadOnlyViolation):
        await GraphRAGToolkit(rag, files_root=root, read_only=True).call(
            "ingest", {"files": ["*.md"]}
        )


async def test_read_tools_on_a_fresh_graph_create_no_keys(real_falkordb_rag_factory):
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution

    from .conftest import MockLLM

    rag = real_falkordb_rag_factory(llm=MockLLM(responses=[]), resolver=ExactMatchResolution())
    tk = GraphRAGToolkit(rag, read_only=True)
    await tk.call("get_ontology", {})
    await tk.call("retrieve", {"question": "Who founded Acme?"})
    store = rag._ontology_store
    store._ensure_graph()
    graphs = await store._conn._driver.list_graphs()
    assert store.graph_name not in graphs
    # Retrieval ran without registering the default types.
    assert rag._ontology_initialized is False


async def test_a_server_timeout_is_a_timeout(toolkit: GraphRAGToolkit):
    # A generous client-side wait, so only FalkorDB's own timeout can fire.
    slow = "UNWIND range(1, 100000000) AS x WITH x WHERE x % 7 = 3 RETURN count(x)"
    with pytest.raises(TimeoutError):
        await ops.bounded_query(toolkit._store, slow, None, 50)
