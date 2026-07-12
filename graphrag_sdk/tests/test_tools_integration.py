"""Real-FalkorDB integration tests for graphrag_sdk.tools (RUN_INTEGRATION-gated)."""

from __future__ import annotations

import json
import os
from uuid import uuid4

import pytest

from graphrag_sdk.core.exceptions import ReadOnlyViolation
from graphrag_sdk.tools import GraphRAGToolkit

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


async def test_toolkit_round_trip(real_falkordb_rag_factory):
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution

    from .conftest import MockLLM

    resolver = ExactMatchResolution()
    llm = MockLLM(
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
            _step2([("Carol", "Person", "CFO of Acme")]),  # consumed by remember()
            "Alice, Acme Corp",  # keyword-extraction / completion clamp
        ]
    )
    rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
    for doc_id, text in [
        ("doc-alice", "Alice is a software engineer at Acme Corp."),
        ("doc-bob", "Bob leads the data team at Acme Corp."),
        ("doc-hq", "Acme Corp is headquartered in Berlin."),
    ]:
        await rag.ingest(text=text, document_id=doc_id, resolver=resolver)

    tk = GraphRAGToolkit(rag, finalize_policy="manual")

    remembered = await tk.remember("Carol is the CFO of Acme Corp.", document_id="doc-carol")
    assert remembered.document_id == "doc-carol" and not remembered.finalized
    await tk.flush()  # dedup + embeddings + fulltext/vector indexes

    sr = await tk.search("Who works at Acme Corp?", top_k=8)
    names = {e.name for e in sr.entities}
    assert "Alice" in names or "Acme Corp" in names
    assert any(r.type == "WORKS_AT" for r in sr.relations)
    assert sr.chunks and all(c.document_id for c in sr.chunks)
    assert sr.to_llm_text()
    assert len(sr.to_llm_text(max_chars=500)) <= 500

    er = await tk.entity("Alice", hops=2)
    assert er.found and er.entity is not None and er.entity.label == "Person"
    assert any(t.type == "WORKS_AT" and t.target == "Acme Corp" for t in er.neighbors)
    assert any(d.document_id == "doc-alice" for d in er.documents)

    sch = await tk.schema()
    labels = {e.label: e.count for e in sch.entities}
    assert labels.get("Person", 0) >= 2 and labels.get("Organization", 0) >= 1
    assert any(r.label == "WORKS_AT" and r.count >= 1 for r in sch.relations)
    assert sch.node_count > 0

    cr = await tk.cypher_read(
        "MATCH (e:__Entity__) WHERE e.name = $name RETURN e.name", {"name": "Alice"}
    )
    assert cr.rows == [["Alice"]] and cr.columns
    with pytest.raises(ReadOnlyViolation):
        await tk.cypher_read("CREATE (n:Hack) RETURN n")

    ar = await tk.answer("Who works at Acme Corp?", top_k=5)
    assert ar.answer.strip() and len(ar.citations) >= 1
    assert all(c.chunk_id and c.document_id for c in ar.citations)


async def test_cypher_read_limit_injection_live(real_falkordb_rag_factory):
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution

    from .conftest import MockLLM

    resolver = ExactMatchResolution()
    llm = MockLLM(
        responses=[
            _step2(
                [
                    ("D1", "Concept", "one"),
                    ("D2", "Concept", "two"),
                    ("D3", "Concept", "three"),
                ]
            )
        ]
    )
    rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
    await rag.ingest(text="D1 and D2 and D3 are concepts.", document_id="doc-d", resolver=resolver)

    tk = GraphRAGToolkit(rag)
    cr = await tk.cypher_read("MATCH (e:__Entity__) RETURN e.name", limit=2)
    assert cr.row_count <= 2
    if cr.row_count == 2:
        assert cr.truncated is True


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
    llm_a = MockLLM(responses=[_step2([("Alice", "Person", "Engineer")])])
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

        await tk_a.remember("Alice is an engineer.", document_id="doc-a")
        await tk_a.flush()
        found_a = await tk_a.entity("Alice")
        assert found_a.found

        # tenant-b's graph never saw Alice
        found_b = await tk_b.entity("Alice")
        assert found_b.found is False
    finally:
        for tk in (tk_a, tk_b):
            try:
                await tk.rag._graph_store.delete_all()
            finally:
                await tk.aclose()


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
async def test_real_llm_smoke(real_falkordb_rag_factory):
    """One end-to-end pass with a real LLM (kept tiny: one 2-sentence doc)."""
    pytest.importorskip("litellm")
    from graphrag_sdk import LiteLLM, LiteLLMEmbedder
    from graphrag_sdk.api.main import GraphRAG
    from graphrag_sdk.core.connection import ConnectionConfig

    config = ConnectionConfig(
        host=os.getenv("FALKOR_HOST", "localhost"),
        port=int(os.getenv("FALKOR_PORT", "6379")),
        graph_name=f"test_{uuid4().hex[:8]}",
    )
    llm = LiteLLM(model="openai/gpt-4o-mini")
    embedder = LiteLLMEmbedder(model="openai/text-embedding-3-small", dimensions=256)
    rag = GraphRAG(connection=config, llm=llm, embedder=embedder, embedding_dimension=256)
    try:
        tk = GraphRAGToolkit(rag)
        await tk.remember(
            "Ada Lovelace wrote the first computer program. "
            "She collaborated with Charles Babbage on the Analytical Engine.",
            document_id="doc-ada",
        )
        await tk.flush()
        ar = await tk.answer("Who wrote the first computer program?")
        assert ar.answer.strip()
        assert len(ar.citations) >= 1
    finally:
        try:
            await rag._graph_store.delete_all()
        finally:
            await rag.close()
