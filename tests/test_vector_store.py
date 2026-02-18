"""Tests for storage/vector_store.py — Vector index management and search."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.models import TextChunk, TextChunks
from graphrag_sdk.storage.vector_store import VectorStore
from .conftest import MockEmbedder


@pytest.fixture
def vector_store(mock_connection, embedder):
    return VectorStore(
        mock_connection,
        embedder=embedder,
        index_name="test_idx",
        embedding_dimension=8,
    )


@pytest.fixture
def vector_store_no_embedder(mock_connection):
    return VectorStore(mock_connection, embedder=None)


class TestVectorStoreIndex:
    async def test_create_vector_index(self, vector_store, mock_connection):
        await vector_store.create_vector_index()
        cypher = mock_connection.query.call_args[0][0]
        assert "CREATE VECTOR INDEX" in cypher
        assert "Chunk" in cypher
        assert "8" in cypher  # dimension

    async def test_create_fulltext_index(self, vector_store, mock_connection):
        await vector_store.create_fulltext_index()
        cypher = mock_connection.query.call_args[0][0]
        assert "fulltext.createNodeIndex" in cypher
        assert "Chunk" in cypher

    async def test_create_fulltext_index_custom_props(self, vector_store, mock_connection):
        await vector_store.create_fulltext_index("Chunk", "text", "title")
        cypher = mock_connection.query.call_args[0][0]
        assert "'text'" in cypher
        assert "'title'" in cypher

    async def test_drop_vector_index(self, vector_store, mock_connection):
        await vector_store.drop_vector_index()
        cypher = mock_connection.query.call_args[0][0]
        assert "vector.drop" in cypher

    async def test_create_index_error_handled(self, vector_store, mock_connection):
        mock_connection.query = AsyncMock(side_effect=Exception("index exists"))
        # Should not raise, just warn
        await vector_store.create_vector_index()


class TestVectorStoreIndexChunks:
    async def test_index_chunks_uses_unwind(self, vector_store, mock_connection, embedder):
        """index_chunks should use UNWIND batch query."""
        chunks = TextChunks(
            chunks=[
                TextChunk(text="Hello world", index=0, uid="c-0"),
                TextChunk(text="Goodbye world", index=1, uid="c-1"),
            ]
        )
        result = await vector_store.index_chunks(chunks)
        assert result == 2
        assert embedder.call_count == 2
        # 1 UNWIND batch query (not 2 individual queries)
        assert mock_connection.query.call_count == 1
        cypher = mock_connection.query.call_args[0][0]
        assert "UNWIND" in cypher
        params = mock_connection.query.call_args[0][1]
        assert len(params["batch"]) == 2

    async def test_index_chunks_no_embedder(self, vector_store_no_embedder):
        chunks = TextChunks(chunks=[TextChunk(text="Hi", index=0)])
        result = await vector_store_no_embedder.index_chunks(chunks)
        assert result == 0

    async def test_index_chunks_batch_fallback(self, vector_store, mock_connection, embedder):
        """When UNWIND batch fails, should fall back to individual queries."""
        mock_connection.query = AsyncMock(
            side_effect=[Exception("batch fail"), Exception("c-0 fail"), MagicMock()]
        )
        chunks = TextChunks(
            chunks=[
                TextChunk(text="A", index=0, uid="c-0"),
                TextChunk(text="B", index=1, uid="c-1"),
            ]
        )
        result = await vector_store.index_chunks(chunks)
        # UNWIND fails → c-0 individual fails → c-1 individual succeeds
        assert result == 1

    async def test_index_empty_chunks(self, vector_store, mock_connection):
        result = await vector_store.index_chunks(TextChunks())
        assert result == 0


class TestVectorStoreSearch:
    async def test_search(self, vector_store, mock_connection):
        result_mock = MagicMock()
        result_mock.result_set = [
            ["chunk-1", "Hello world", 0.95],
            ["chunk-2", "Goodbye world", 0.80],
        ]
        mock_connection.query = AsyncMock(return_value=result_mock)
        results = await vector_store.search(query_vector=[0.1] * 8, top_k=5)
        assert len(results) == 2
        assert results[0]["id"] == "chunk-1"
        assert results[0]["score"] == 0.95

    async def test_search_empty(self, vector_store, mock_connection):
        result_mock = MagicMock()
        result_mock.result_set = []
        mock_connection.query = AsyncMock(return_value=result_mock)
        results = await vector_store.search(query_vector=[0.1] * 8)
        assert results == []

    async def test_search_error_returns_empty(self, vector_store, mock_connection):
        mock_connection.query = AsyncMock(side_effect=Exception("search failed"))
        results = await vector_store.search(query_vector=[0.1] * 8)
        assert results == []

    async def test_search_custom_top_k(self, vector_store, mock_connection):
        result_mock = MagicMock()
        result_mock.result_set = []
        mock_connection.query = AsyncMock(return_value=result_mock)
        await vector_store.search(query_vector=[0.1] * 8, top_k=10)
        params = mock_connection.query.call_args[0][1]
        assert params["top_k"] == 10


class TestVectorStoreFulltextSearch:
    async def test_fulltext_search(self, vector_store, mock_connection):
        result_mock = MagicMock()
        result_mock.result_set = [
            ["chunk-1", "Match text", 1.5],
        ]
        mock_connection.query = AsyncMock(return_value=result_mock)
        results = await vector_store.fulltext_search("Match")
        assert len(results) == 1
        assert results[0]["text"] == "Match text"

    async def test_fulltext_search_error_returns_empty(self, vector_store, mock_connection):
        mock_connection.query = AsyncMock(side_effect=Exception("search failed"))
        results = await vector_store.fulltext_search("test")
        assert results == []


class TestVectorStoreBackfillEntityEmbeddings:
    async def test_backfill_uses_unwind(self, vector_store, mock_connection, embedder):
        """backfill_entity_embeddings should use UNWIND batch write."""
        # First call: query entities → return 2 entities
        entity_result = MagicMock()
        entity_result.result_set = [
            ["e1", "Alice", "A person"],
            ["e2", "Bob", "Another person"],
        ]
        # Second call: UNWIND batch write → success
        # Third call: query entities → return empty (no more)
        empty_result = MagicMock()
        empty_result.result_set = []
        mock_connection.query = AsyncMock(
            side_effect=[entity_result, MagicMock(), empty_result]
        )
        result = await vector_store.backfill_entity_embeddings()
        assert result == 2
        # Second call should be the UNWIND batch write
        second_call_cypher = mock_connection.query.call_args_list[1][0][0]
        assert "UNWIND" in second_call_cypher

    async def test_backfill_no_embedder(self, vector_store_no_embedder):
        result = await vector_store_no_embedder.backfill_entity_embeddings()
        assert result == 0

    async def test_backfill_batch_fallback(self, vector_store, mock_connection, embedder):
        """When UNWIND batch write fails, should fall back to individual writes."""
        entity_result = MagicMock()
        entity_result.result_set = [["e1", "Alice", "A person"]]
        empty_result = MagicMock()
        empty_result.result_set = []
        mock_connection.query = AsyncMock(
            side_effect=[
                entity_result,      # fetch entities
                Exception("batch"), # UNWIND fails
                MagicMock(),         # individual e1 succeeds
                empty_result,        # next fetch → empty
            ]
        )
        result = await vector_store.backfill_entity_embeddings()
        assert result == 1


class TestEnsureIndicesSkip:
    async def test_ensure_indices_skips_on_repeat(self, vector_store, mock_connection):
        """ensure_indices should skip on subsequent calls."""
        await vector_store.ensure_indices()
        first_call_count = mock_connection.query.call_count

        # Reset mock to track new calls
        mock_connection.query.reset_mock()
        result = await vector_store.ensure_indices()
        assert result == {}
        mock_connection.query.assert_not_called()

    async def test_ensure_indices_flag_starts_false(self, mock_connection, embedder):
        store = VectorStore(mock_connection, embedder=embedder)
        assert store._indices_ensured is False
