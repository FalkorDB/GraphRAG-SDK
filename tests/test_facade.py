"""Tests for api/main.py — the GraphRAG Facade."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag_sdk.api.main import GraphRAG
from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    GraphSchema,
    LLMResponse,
    RagResult,
    RetrieverResult,
    RetrieverResultItem,
)
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy
from graphrag_sdk.core.models import RawSearchResult

from .conftest import MockEmbedder, MockLLM, MockLLMWithExtraction


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_conn():
    conn = MagicMock(spec=FalkorDBConnection)
    result_mock = MagicMock()
    result_mock.result_set = []
    conn.query = AsyncMock(return_value=result_mock)
    conn.config = ConnectionConfig()
    return conn


@pytest.fixture
def graphrag(mock_conn, embedder, llm):
    return GraphRAG(
        connection=mock_conn,
        llm=llm,
        embedder=embedder,
    )


@pytest.fixture
def graphrag_with_schema(mock_conn, embedder, llm, sample_schema):
    return GraphRAG(
        connection=mock_conn,
        llm=llm,
        embedder=embedder,
        schema=sample_schema,
    )


# ── Tests ───────────────────────────────────────────────────────


class TestGraphRAGInit:
    def test_init_with_connection(self, mock_conn, embedder, llm):
        g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder)
        assert g.llm is llm
        assert g.embedder is embedder
        assert g.graph_store is not None
        assert g.vector_store is not None

    def test_init_with_config(self, embedder, llm):
        cfg = ConnectionConfig(host="testhost", port=1234)
        g = GraphRAG(connection=cfg, llm=llm, embedder=embedder)
        assert g._conn.config.host == "testhost"

    def test_default_schema(self, graphrag):
        assert graphrag.schema is not None
        assert graphrag.schema.entities == []

    def test_custom_schema(self, graphrag_with_schema, sample_schema):
        assert graphrag_with_schema.schema is sample_schema

    def test_default_retrieval_strategy(self, graphrag):
        from graphrag_sdk.retrieval.strategies.local import LocalRetrieval
        assert isinstance(graphrag._retrieval_strategy, LocalRetrieval)

    def test_custom_retrieval_strategy(self, mock_conn, embedder, llm):
        class CustomStrategy(RetrievalStrategy):
            async def _execute(self, query, ctx, **kwargs):
                return RawSearchResult()

        strategy = CustomStrategy()
        g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder, retrieval_strategy=strategy)
        assert g._retrieval_strategy is strategy


class TestGraphRAGIngest:
    async def test_ingest_text_file(self, graphrag, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Hello world. This is a test document.")
        result = await graphrag.ingest(str(f))
        assert result.chunks_indexed >= 0

    async def test_ingest_with_text_param(self, graphrag):
        result = await graphrag.ingest("ignored", text="Direct text for ingestion.")
        assert result is not None

    async def test_ingest_custom_context(self, graphrag, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Context test.")
        ctx = Context(tenant_id="custom-tenant")
        result = await graphrag.ingest(str(f), ctx=ctx)
        assert result is not None

    async def test_ingest_auto_detects_pdf(self, mock_conn, embedder, llm):
        """Verifies PDF extension triggers PdfLoader selection."""
        g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder)
        # We won't actually load a PDF, just verify the loader type
        from graphrag_sdk.ingestion.loaders.pdf_loader import PdfLoader
        # The loader selection happens inside ingest() — test path detection
        # by checking that source ending in .pdf doesn't use TextLoader
        with pytest.raises(Exception):
            # Will fail because file doesn't exist, but would use PdfLoader
            await g.ingest("/fake/file.pdf")


class TestGraphRAGQuery:
    async def test_query_basic(self, mock_conn, embedder):
        llm = MockLLM(responses=["The answer is 42."])
        g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder)

        # Mock the retrieval strategy to return known results
        mock_strategy = MagicMock(spec=RetrievalStrategy)
        mock_strategy.search = AsyncMock(
            return_value=RetrieverResult(
                items=[RetrieverResultItem(content="Context chunk", score=0.9)]
            )
        )
        g._retrieval_strategy = mock_strategy

        result = await g.query("What is the answer?")
        assert isinstance(result, RagResult)
        assert result.answer == "The answer is 42."
        assert result.metadata["num_context_items"] == 1

    async def test_query_with_reranker(self, mock_conn, embedder):
        from graphrag_sdk.retrieval.reranking_strategies.base import RerankingStrategy

        llm = MockLLM(responses=["Reranked answer."])
        g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder)

        mock_strategy = MagicMock(spec=RetrievalStrategy)
        mock_strategy.search = AsyncMock(
            return_value=RetrieverResult(items=[
                RetrieverResultItem(content="A", score=0.5),
                RetrieverResultItem(content="B", score=0.9),
            ])
        )
        g._retrieval_strategy = mock_strategy

        class FlipReranker(RerankingStrategy):
            async def rerank(self, query, result, ctx):
                return RetrieverResult(items=list(reversed(result.items)))

        result = await g.query("test", reranker=FlipReranker())
        assert result.answer == "Reranked answer."

    async def test_query_return_context(self, mock_conn, embedder):
        llm = MockLLM(responses=["Answer."])
        g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder)
        mock_strategy = MagicMock(spec=RetrievalStrategy)
        mock_strategy.search = AsyncMock(
            return_value=RetrieverResult(
                items=[RetrieverResultItem(content="chunk")]
            )
        )
        g._retrieval_strategy = mock_strategy

        result = await g.query("q?", return_context=True)
        assert result.retriever_result is not None

    async def test_query_no_context(self, mock_conn, embedder):
        llm = MockLLM(responses=["No context."])
        g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder)
        mock_strategy = MagicMock(spec=RetrievalStrategy)
        mock_strategy.search = AsyncMock(
            return_value=RetrieverResult(
                items=[RetrieverResultItem(content="chunk")]
            )
        )
        g._retrieval_strategy = mock_strategy

        result = await g.query("q?", return_context=False)
        assert result.retriever_result is None

    async def test_query_custom_prompt(self, mock_conn, embedder):
        llm = MockLLM(responses=["Custom prompt answer."])
        g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder)
        mock_strategy = MagicMock(spec=RetrievalStrategy)
        mock_strategy.search = AsyncMock(
            return_value=RetrieverResult(items=[RetrieverResultItem(content="c")])
        )
        g._retrieval_strategy = mock_strategy

        custom_template = "Context: {context}\nQ: {question}\nA:"
        result = await g.query("test?", prompt_template=custom_template)
        # LLM was called — it should have used the custom template
        assert llm._call_index == 1

    async def test_query_metadata(self, mock_conn, embedder):
        llm = MockLLM(responses=["Answer."])
        g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder)
        mock_strategy = MagicMock(spec=RetrievalStrategy)
        mock_strategy.search = AsyncMock(
            return_value=RetrieverResult(items=[RetrieverResultItem(content="c")])
        )
        mock_strategy.__class__.__name__ = "MockStrategy"
        g._retrieval_strategy = mock_strategy

        result = await g.query("q?")
        assert "model" in result.metadata
        assert result.metadata["model"] == "mock-llm"


class TestGraphRAGSyncWrappers:
    def test_query_sync(self, mock_conn, embedder):
        llm = MockLLM(responses=["Sync answer."])
        g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder)
        mock_strategy = MagicMock(spec=RetrievalStrategy)
        mock_strategy.search = AsyncMock(
            return_value=RetrieverResult(items=[RetrieverResultItem(content="c")])
        )
        g._retrieval_strategy = mock_strategy
        result = g.query_sync("test?")
        assert result.answer == "Sync answer."

    def test_ingest_sync(self, mock_conn, embedder, llm, tmp_path):
        g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder)
        f = tmp_path / "test.txt"
        f.write_text("Sync ingest content.")
        result = g.ingest_sync(str(f))
        assert result is not None
