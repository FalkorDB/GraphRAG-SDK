"""Tests for the S8 pagination safety net.

Verifies that the four ``while True`` pagination loops in storage/* terminate
even when underlying data never depletes — protecting against pathological
server behavior or driver bugs.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

from graphrag_sdk.storage import deduplicator as dedup_mod
from graphrag_sdk.storage import vector_store as vs_mod
from graphrag_sdk.storage.deduplicator import EntityDeduplicator
from graphrag_sdk.storage.vector_store import VectorStore


class TestDeduplicatorPaginationCap:
    async def test_fetch_all_entities_caps_iterations(self, caplog):
        graph = MagicMock()
        # Result always non-empty → would iterate forever without the cap.
        non_empty = MagicMock()
        non_empty.result_set = [["e1", "name", "desc", "Person"]]
        graph.query_raw = AsyncMock(return_value=non_empty)

        embedder = MagicMock()
        d = EntityDeduplicator(graph, embedder)

        with patch.object(dedup_mod, "_MAX_PAGINATION_ITERATIONS", 3), \
             caplog.at_level(logging.ERROR, logger="graphrag_sdk.storage.deduplicator"):
            entities = await d._fetch_all_entities(batch_size=1)

        # Capped at the patched limit → returns partial data, no infinite loop.
        assert len(entities) == 3
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any("exceeded" in r.getMessage() for r in errors)

    async def test_fetch_all_entities_normal_exit_no_error(self, caplog):
        graph = MagicMock()
        non_empty = MagicMock()
        non_empty.result_set = [["e1", "name", "desc", "Person"]]
        empty = MagicMock()
        empty.result_set = []
        # Returns one batch, then empty → natural exit, cap never hit.
        graph.query_raw = AsyncMock(side_effect=[non_empty, empty])

        embedder = MagicMock()
        d = EntityDeduplicator(graph, embedder)

        with caplog.at_level(logging.ERROR, logger="graphrag_sdk.storage.deduplicator"):
            entities = await d._fetch_all_entities(batch_size=1)

        assert len(entities) == 1
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert not any("exceeded" in r.getMessage() for r in errors)


class TestVectorStorePaginationCap:
    async def test_backfill_entity_embeddings_caps_iterations(self, caplog):
        conn = MagicMock()
        non_empty = MagicMock()
        non_empty.result_set = [["e1", "Alice", "desc"]]
        # Every query returns one un-embedded entity → would loop forever.
        conn.query = AsyncMock(return_value=non_empty)

        embedder = MagicMock()
        embedder.aembed_documents = AsyncMock(return_value=[[0.1, 0.2]])

        store = VectorStore(conn, embedder=embedder, embedding_dimension=2)

        with patch.object(vs_mod, "_MAX_PAGINATION_ITERATIONS", 3), \
             caplog.at_level(logging.ERROR, logger="graphrag_sdk.storage.vector_store"):
            await store.backfill_entity_embeddings(batch_size=1)

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any("exceeded" in r.getMessage() for r in errors)
