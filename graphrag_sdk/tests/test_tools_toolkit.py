"""GraphRAGToolkit unit tests (stubbed GraphRAG — no server, no LLM)."""

from __future__ import annotations

import inspect
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.connection import ConnectionConfig
from graphrag_sdk.core.exceptions import ConfigError, ReadOnlyViolation
from graphrag_sdk.core.models import (
    DocumentInfo,
    FinalizeResult,
    IngestionResult,
    RagResult,
    RetrieverResult,
)
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval
from graphrag_sdk.tools import GraphRAGToolkit, RememberResult
from graphrag_sdk.tools.specs import _TOOL_REGISTRY


def make_stub_rag() -> MagicMock:
    rag = MagicMock()
    rag.ingest = AsyncMock(
        return_value=IngestionResult(
            document_info=DocumentInfo(uid="doc-1"),
            nodes_created=2,
            relationships_created=1,
            chunks_indexed=1,
        )
    )
    rag.finalize = AsyncMock(return_value=FinalizeResult())
    rag.close = AsyncMock()
    return rag


def test_ctor_validates_policy_and_include():
    rag = make_stub_rag()
    with pytest.raises(ValueError, match="finalize_policy"):
        GraphRAGToolkit(rag, finalize_policy="sometimes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="graph_serch"):
        GraphRAGToolkit(rag, include=["graph_serch"])


async def test_remember_manual_policy_defers_finalize():
    rag = make_stub_rag()
    tk = GraphRAGToolkit(rag, finalize_policy="manual")
    result = await tk.remember("Alice works at Acme.", document_id="doc-1")
    assert isinstance(result, RememberResult)
    assert result.document_id == "doc-1" and result.finalized is False
    rag.finalize.assert_not_awaited()
    rag.ingest.assert_awaited_once()
    assert rag.ingest.await_args.kwargs["text"] == "Alice works at Acme."
    await tk.flush()
    rag.finalize.assert_awaited_once()


async def test_remember_on_write_policy_finalizes():
    rag = make_stub_rag()
    tk = GraphRAGToolkit(rag, finalize_policy="on_write")
    result = await tk.remember("x")
    assert result.finalized is True
    rag.finalize.assert_awaited_once()
    await tk.flush()  # documented no-op
    rag.finalize.assert_awaited_once()


async def test_never_policy_flush_raises():
    tk = GraphRAGToolkit(make_stub_rag(), finalize_policy="never")
    with pytest.raises(ConfigError):
        await tk.flush()


async def test_read_only_blocks_writes():
    rag = make_stub_rag()
    tk = GraphRAGToolkit(rag, read_only=True)
    with pytest.raises(ReadOnlyViolation):
        await tk.remember("x")
    with pytest.raises(ReadOnlyViolation):
        await tk.flush()
    with pytest.raises(ReadOnlyViolation):
        await tk.call("graph_remember", {"text": "x"})
    rag.ingest.assert_not_awaited()
    assert "graph_remember" not in [s.name for s in tk.tool_specs()]


async def test_call_dispatch_and_validation():
    rag = make_stub_rag()
    tk = GraphRAGToolkit(rag)
    result = await tk.call("graph_remember", {"text": "hi", "document_id": "d"})
    assert isinstance(result, RememberResult)
    with pytest.raises(ValueError, match="Unknown tool"):
        await tk.call("nope", {})
    with pytest.raises(Exception):  # pydantic ValidationError on extra arg
        await tk.call("graph_remember", {"text": "hi", "bogus": 1})


async def test_call_flush_unavailable_under_non_manual_policy():
    tk = GraphRAGToolkit(make_stub_rag(), finalize_policy="on_write")
    with pytest.raises(ValueError, match="unavailable"):
        await tk.call("graph_flush", {})


async def test_include_gates_call_and_specs():
    tk = GraphRAGToolkit(make_stub_rag(), include=["graph_schema"])
    assert [s.name for s in tk.tool_specs()] == ["graph_schema"]
    with pytest.raises(ValueError, match="not enabled"):
        await tk.call("graph_remember", {"text": "x"})


def test_signature_matches_input_model():
    for td in _TOOL_REGISTRY:
        method = getattr(GraphRAGToolkit, td.method)
        params = {
            n: p
            for n, p in inspect.signature(method).parameters.items()
            if n not in {"self", "ctx"}
        }
        fields = td.input_model.model_fields
        assert set(params) == set(fields), td.name
        for pname, param in params.items():
            field = fields[pname]
            expected = inspect.Parameter.empty if field.is_required() else field.default
            assert param.default == expected, f"{td.name}.{pname} default drift"


def test_for_tenant_derives_graph_name_lazily():
    base = ConnectionConfig(graph_name="app")
    tk = GraphRAGToolkit.for_tenant(base, "acme", llm=MagicMock(), embedder=MagicMock())
    assert tk.rag._conn.config.graph_name == "app__acme"
    assert base.graph_name == "app"  # base untouched
    with pytest.raises(ValueError, match="tenant_id"):
        GraphRAGToolkit.for_tenant(base, "bad tenant!", llm=MagicMock(), embedder=MagicMock())


async def test_aclose_closes_owned_rag_only():
    rag = make_stub_rag()
    tk = GraphRAGToolkit(rag)
    await tk.aclose()
    rag.close.assert_not_awaited()
    owned = GraphRAGToolkit(rag, owns_rag=True)
    async with owned:
        pass
    rag.close.assert_awaited_once()


def test_module_imports_cleanly():
    proc = subprocess.run(
        [sys.executable, "-c", "import graphrag_sdk.tools"], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


# ── search() / answer() over provenance metadata ─────────────────


_PROVENANCE = {
    "entities": [
        {"id": "e1", "name": "Alice", "description": "Engineer"},
        {"id": "e2", "name": "Ghost", "description": ""},
        {"id": "e3", "name": "Extra", "description": ""},
    ],
    "chunks": [
        {"id": "c1", "text": "Alice works at Acme.", "document_path": "docs/a.md"},
        {"id": "c2", "text": "Bob leads data.", "document_path": "docs/b.md"},
        {"id": "c3", "text": "HQ is in Berlin.", "document_path": "docs/c.md"},
    ],
    "facts": [f"fact {i}" for i in range(6)],
    "relationships": ["Alice —[WORKS_AT]→ Acme Corp"],
}


def _rows(rows):
    result = MagicMock()
    result.result_set = rows
    return result


def make_search_rag() -> MagicMock:
    rag = make_stub_rag()
    rag.retrieve = AsyncMock(
        return_value=RetrieverResult(items=[], metadata={"provenance": _PROVENANCE})
    )
    rag.completion = AsyncMock(
        return_value=RagResult(
            answer="Alice and Bob work at Acme.",
            retriever_result=RetrieverResult(items=[], metadata={"provenance": _PROVENANCE}),
            metadata={},
        )
    )

    async def route(cypher, params=None):
        if "UNWIND $ids AS eid MATCH (e:__Entity__" in cypher:  # enrichment
            return _rows(
                [["e1", "Alice", "Engineer", ["Person", "__Entity__"], {"seniority": "senior"}]]
            )
        if "r.src_name" in cypher:  # triples expansion
            return _rows([["Alice", "WORKS_AT", "Acme Corp", "employment", "e9"]])
        if "PART_OF" in cypher:  # chunk -> document ids
            return _rows([["c1", "doc-a", "docs/a.md"]])
        return _rows([])

    rag._graph_store.query_raw = AsyncMock(side_effect=route)
    return rag


async def test_search_tunes_strategy_and_maps_provenance():
    rag = make_search_rag()
    tk = GraphRAGToolkit(rag)
    sr = await tk.search("Who works at Acme?", top_k=2)

    strategy = rag.retrieve.await_args.kwargs["strategy"]
    assert isinstance(strategy, MultiPathRetrieval)
    assert strategy._chunk_top_k == 2 and strategy._rel_top_k == 2
    assert strategy._max_entities == 10  # max(2*top_k, 10)

    # entities: provenance order, capped to top_k, enrichment fallback for e2
    assert [e.name for e in sr.entities] == ["Alice", "Ghost"]
    assert sr.entities[0].label == "Person"
    assert sr.entities[0].properties == {"seniority": "senior"}
    assert sr.entities[1].label == "" and sr.entities[1].description is None

    assert [c.chunk_id for c in sr.chunks] == ["c1", "c2"]
    assert sr.chunks[0].document_id == "doc-a"
    assert sr.chunks[1].document_id == "" and sr.chunks[1].document_path == "docs/b.md"
    assert [d.document_id for d in sr.documents] == ["doc-a"]

    assert len(sr.facts) == 4  # 2 * top_k
    assert any(r.type == "WORKS_AT" for r in sr.relations)


async def test_search_include_chunks_false_skips_chunk_queries():
    rag = make_search_rag()
    tk = GraphRAGToolkit(rag)
    sr = await tk.search("q", top_k=2, include_chunks=False)
    assert sr.chunks == [] and sr.documents == []
    part_of_calls = [
        c.args[0] for c in rag._graph_store.query_raw.await_args_list if "PART_OF" in c.args[0]
    ]
    assert part_of_calls == []


async def test_search_expand_hops_controls_frontier_queries():
    rag = make_search_rag()
    tk = GraphRAGToolkit(rag)
    await tk.search("q", top_k=2, expand_hops=2)
    rel_calls = [
        c.args[0] for c in rag._graph_store.query_raw.await_args_list if "r.src_name" in c.args[0]
    ]
    assert len(rel_calls) == 2


async def test_answer_builds_citations_and_entities_touched():
    rag = make_search_rag()
    long_text = "X" * 250
    provenance = {
        "entities": [
            {"id": "e1", "name": "Alice"},
            {"id": "e2", "name": "alice"},  # case-insensitive duplicate
            {"id": "e3", "name": "Bob"},
        ],
        "chunks": [{"id": "c1", "text": long_text, "document_path": "docs/a.md"}],
        "facts": [],
        "relationships": [],
    }
    rag.completion = AsyncMock(
        return_value=RagResult(
            answer="Alice works at Acme.",
            retriever_result=RetrieverResult(items=[], metadata={"provenance": provenance}),
            metadata={},
        )
    )
    tk = GraphRAGToolkit(rag)
    ar = await tk.answer("Who is Alice?", top_k=3)

    assert rag.completion.await_args.kwargs["return_context"] is True
    strategy = rag.completion.await_args.kwargs["strategy"]
    assert isinstance(strategy, MultiPathRetrieval) and strategy._chunk_top_k == 3

    assert ar.answer == "Alice works at Acme."
    assert len(ar.citations) == 1
    citation = ar.citations[0]
    assert citation.chunk_id == "c1" and citation.document_id == "doc-a"
    assert len(citation.snippet) == 200 and citation.snippet.endswith("…")
    assert ar.entities_touched == ["Alice", "Bob"]
    assert ar.cypher_used is None


async def test_answer_degrades_without_provenance():
    rag = make_search_rag()
    rag.completion = AsyncMock(
        return_value=RagResult(answer="ok", retriever_result=None, metadata={})
    )
    tk = GraphRAGToolkit(rag)
    ar = await tk.answer("q")
    assert ar.answer == "ok" and ar.citations == [] and ar.entities_touched == []
