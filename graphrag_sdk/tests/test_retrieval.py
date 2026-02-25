"""Tests for retrieval/strategies/ — base (Template Method) and local retrieval."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import RetrieverError
from graphrag_sdk.core.models import RawSearchResult, RetrieverResult, RetrieverResultItem
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy
from graphrag_sdk.retrieval.strategies.local import LocalRetrieval


# ── Concrete test strategy ──────────────────────────────────────


class StubRetrieval(RetrievalStrategy):
    """Minimal concrete strategy for testing Template Method base."""

    def __init__(self, records=None, should_fail=False):
        super().__init__()
        self._records = records or [{"text": "result 1"}, {"text": "result 2"}]
        self._should_fail = should_fail

    async def _execute(self, query: str, ctx: Context, **kwargs: Any) -> RawSearchResult:
        if self._should_fail:
            raise RuntimeError("deliberate failure")
        return RawSearchResult(
            records=self._records,
            metadata={"strategy": "stub"},
        )


# ── Tests for base class (Template Method) ──────────────────────


class TestRetrievalStrategyBase:
    async def test_search_returns_result(self, ctx):
        strategy = StubRetrieval()
        result = await strategy.search("test query", ctx)
        assert isinstance(result, RetrieverResult)
        assert len(result.items) == 2

    async def test_search_validates_empty_query(self, ctx):
        strategy = StubRetrieval()
        with pytest.raises(RetrieverError, match="Empty query"):
            await strategy.search("", ctx)

    async def test_search_validates_whitespace_query(self, ctx):
        strategy = StubRetrieval()
        with pytest.raises(RetrieverError, match="Empty query"):
            await strategy.search("   ", ctx)

    async def test_search_wraps_exception(self, ctx):
        strategy = StubRetrieval(should_fail=True)
        with pytest.raises(RetrieverError, match="failed"):
            await strategy.search("valid query", ctx)

    async def test_search_default_context(self):
        """search() creates default Context if none provided."""
        strategy = StubRetrieval()
        result = await strategy.search("test")
        assert len(result.items) == 2

    async def test_default_format(self, ctx):
        strategy = StubRetrieval(records=["record1", "record2"])
        result = await strategy.search("test", ctx)
        # Default format converts records to string
        assert result.items[0].content == "record1"
        assert result.items[1].content == "record2"

    async def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            RetrievalStrategy()  # type: ignore[abstract]

    async def test_metadata_propagated(self, ctx):
        strategy = StubRetrieval()
        result = await strategy.search("test", ctx)
        assert result.metadata.get("strategy") == "stub"


# ── Tests for LocalRetrieval ────────────────────────────────────


class TestLocalRetrieval:
    @pytest.fixture
    def local_strategy(self, mock_graph_store, mock_vector_store, embedder):
        mock_vector_store.search = AsyncMock(
            return_value=[
                {"id": "chunk-1", "text": "Alice works at Acme.", "score": 0.95},
                {"id": "chunk-2", "text": "Bob works at Acme.", "score": 0.85},
            ]
        )
        mock_graph_store.get_connected_entities = AsyncMock(
            return_value=[
                {"id": "alice", "labels": ["Person"], "properties": {"name": "Alice"}},
            ]
        )
        return LocalRetrieval(
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            embedder=embedder,
            top_k=5,
        )

    async def test_local_search(self, ctx, local_strategy):
        result = await local_strategy.search("Who works at Acme?", ctx)
        assert isinstance(result, RetrieverResult)
        assert len(result.items) == 2

    async def test_local_embeds_query(self, ctx, local_strategy, embedder):
        await local_strategy.search("test query", ctx)
        assert embedder.call_count >= 1

    async def test_local_includes_entities(self, ctx, local_strategy):
        result = await local_strategy.search("test", ctx)
        # Should include entity info in content
        content = result.items[0].content
        assert "Alice" in content or "Related entities" in content

    async def test_local_no_entities(self, ctx, mock_graph_store, mock_vector_store, embedder):
        mock_vector_store.search = AsyncMock(
            return_value=[{"id": "c1", "text": "Chunk text", "score": 0.9}]
        )
        strategy = LocalRetrieval(
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            embedder=embedder,
            include_entities=False,
        )
        result = await strategy.search("test", ctx)
        assert len(result.items) == 1

    async def test_local_empty_results(self, ctx, mock_graph_store, mock_vector_store, embedder):
        mock_vector_store.search = AsyncMock(return_value=[])
        strategy = LocalRetrieval(
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            embedder=embedder,
        )
        result = await strategy.search("no results", ctx)
        assert len(result.items) == 0

    async def test_local_custom_top_k(self, ctx, local_strategy, mock_vector_store):
        await local_strategy.search("test", ctx, top_k=3)
        call_kwargs = mock_vector_store.search.call_args
        assert call_kwargs[1]["top_k"] == 3

    async def test_local_scores_in_items(self, ctx, local_strategy):
        result = await local_strategy.search("test", ctx)
        assert result.items[0].score == 0.95

    async def test_local_chunk_id_in_metadata(self, ctx, local_strategy):
        result = await local_strategy.search("test", ctx)
        assert result.items[0].metadata["chunk_id"] == "chunk-1"
