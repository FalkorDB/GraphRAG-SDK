"""Tests for VectorStore.search_entity_anchored_chunks()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.storage.vector_store import VectorStore


# -- Fixtures --


@pytest.fixture
def mock_conn():
    conn = MagicMock()
    conn.query = AsyncMock(return_value=MagicMock(result_set=[]))
    return conn


@pytest.fixture
def store(mock_conn):
    return VectorStore(connection=mock_conn, embedding_dimension=8)


# -- Tests --


class TestSearchEntityAnchoredChunks:
    async def test_basic_returns_scored_chunks(self, store, mock_conn):
        """Mock connection returns scored chunks — correct output."""
        mock_conn.query = AsyncMock(
            return_value=MagicMock(
                result_set=[
                    ["chunk-1", "Alice is an engineer.", 0.92],
                    ["chunk-2", "Bob works at Acme.", 0.85],
                ]
            )
        )

        results = await store.search_entity_anchored_chunks(
            entity_ids=["eid-1", "eid-2"],
            query_vector=[0.1] * 8,
            top_k=5,
            similarity_threshold=0.70,
            max_hops=1,
        )

        assert len(results) == 2
        assert results[0]["id"] == "chunk-1"
        assert results[0]["text"] == "Alice is an engineer."
        assert results[0]["score"] == 0.92
        assert results[1]["id"] == "chunk-2"

    async def test_empty_entity_list_returns_empty(self, store, mock_conn):
        """Empty entity list should return empty, no query executed."""
        results = await store.search_entity_anchored_chunks(
            entity_ids=[],
            query_vector=[0.1] * 8,
        )

        assert results == []
        mock_conn.query.assert_not_called()

    async def test_connection_failure_propagates(self, store, mock_conn):
        """Connection raises → exception propagates to caller for fallback handling."""
        mock_conn.query = AsyncMock(side_effect=Exception("connection error"))

        with pytest.raises(Exception, match="connection error"):
            await store.search_entity_anchored_chunks(
                entity_ids=["eid-1"],
                query_vector=[0.1] * 8,
            )

    async def test_threshold_filtering_in_cypher(self, store, mock_conn):
        """Threshold should be passed to the Cypher query params."""
        mock_conn.query = AsyncMock(
            return_value=MagicMock(result_set=[])
        )

        await store.search_entity_anchored_chunks(
            entity_ids=["eid-1"],
            query_vector=[0.1] * 8,
            similarity_threshold=0.80,
            max_hops=1,
        )

        # Verify threshold was passed in params
        call_args = mock_conn.query.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params", {})
        assert params["threshold"] == 0.80

    async def test_2hop_uses_relates_pattern(self, store, mock_conn):
        """max_hops=2 should use RELATES hop in Cypher."""
        mock_conn.query = AsyncMock(
            return_value=MagicMock(result_set=[])
        )

        await store.search_entity_anchored_chunks(
            entity_ids=["eid-1"],
            query_vector=[0.1] * 8,
            max_hops=2,
        )

        cypher = mock_conn.query.call_args[0][0]
        assert "RELATES" in cypher
        assert "MENTIONED_IN" in cypher

    async def test_1hop_does_not_use_relates(self, store, mock_conn):
        """max_hops=1 should NOT use RELATES, only MENTIONED_IN."""
        mock_conn.query = AsyncMock(
            return_value=MagicMock(result_set=[])
        )

        await store.search_entity_anchored_chunks(
            entity_ids=["eid-1"],
            query_vector=[0.1] * 8,
            max_hops=1,
        )

        cypher = mock_conn.query.call_args[0][0]
        assert "MENTIONED_IN" in cypher
        assert "RELATES" not in cypher

    async def test_deduplication_via_distinct(self, store, mock_conn):
        """Same chunk from multiple entities should appear only once (DISTINCT in Cypher)."""
        mock_conn.query = AsyncMock(
            return_value=MagicMock(
                result_set=[
                    ["chunk-1", "shared chunk text", 0.90],
                ]
            )
        )

        results = await store.search_entity_anchored_chunks(
            entity_ids=["eid-1", "eid-2", "eid-3"],
            query_vector=[0.1] * 8,
            max_hops=1,
        )

        # Only one result even though 3 entities
        assert len(results) == 1
        assert results[0]["id"] == "chunk-1"

        # Verify DISTINCT is in the Cypher
        cypher = mock_conn.query.call_args[0][0]
        assert "DISTINCT" in cypher

    async def test_top_k_passed_to_query(self, store, mock_conn):
        """top_k should be passed to Cypher LIMIT."""
        mock_conn.query = AsyncMock(
            return_value=MagicMock(result_set=[])
        )

        await store.search_entity_anchored_chunks(
            entity_ids=["eid-1"],
            query_vector=[0.1] * 8,
            top_k=25,
        )

        params = mock_conn.query.call_args[0][1]
        assert params["top_k"] == 25

    async def test_entity_ids_passed_to_query(self, store, mock_conn):
        """Entity IDs should be passed as $eids parameter."""
        mock_conn.query = AsyncMock(
            return_value=MagicMock(result_set=[])
        )

        eids = ["eid-a", "eid-b", "eid-c"]
        await store.search_entity_anchored_chunks(
            entity_ids=eids,
            query_vector=[0.1] * 8,
        )

        params = mock_conn.query.call_args[0][1]
        assert params["eids"] == eids
