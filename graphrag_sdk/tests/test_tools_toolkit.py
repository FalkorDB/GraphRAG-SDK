"""GraphRAGToolkit unit tests (stubbed GraphRAG — no server, no LLM)."""

from __future__ import annotations

import inspect
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.connection import ConnectionConfig
from graphrag_sdk.core.exceptions import ConfigError, ReadOnlyViolation
from graphrag_sdk.core.models import DocumentInfo, FinalizeResult, IngestionResult
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
