"""Tests for retrieval/reranking_strategies/cosine.py — CosineReranker."""
from __future__ import annotations

from typing import Any

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import RetrieverResult, RetrieverResultItem
from graphrag_sdk.retrieval.reranking_strategies.cosine import CosineReranker

from .conftest import MockEmbedder


# ── Tests ───────────────────────────────────────────────────────


class TestCosineReranker:
    async def test_rerank_sorts_by_similarity(self):
        embedder = MockEmbedder(dimension=8)
        reranker = CosineReranker(embedder=embedder, top_k=10)

        items = [
            RetrieverResultItem(content="Irrelevant text about cooking", score=0.5),
            RetrieverResultItem(content="Alice works at Acme Corp", score=0.3),
            RetrieverResultItem(content="Who is Alice?", score=0.1),
        ]
        result = RetrieverResult(items=items)
        ctx = Context(tenant_id="test")

        reranked = await reranker.rerank("Who is Alice?", result, ctx)
        assert isinstance(reranked, RetrieverResult)
        # The item most similar to "Who is Alice?" should be first
        # (exact match of query text)
        assert reranked.items[0].content == "Who is Alice?"

    async def test_rerank_top_k_limits(self):
        embedder = MockEmbedder(dimension=8)
        reranker = CosineReranker(embedder=embedder, top_k=2)

        items = [
            RetrieverResultItem(content=f"Item {i}") for i in range(5)
        ]
        result = RetrieverResult(items=items)
        ctx = Context(tenant_id="test")

        reranked = await reranker.rerank("test query", result, ctx)
        assert len(reranked.items) <= 2

    async def test_rerank_empty(self):
        embedder = MockEmbedder(dimension=8)
        reranker = CosineReranker(embedder=embedder, top_k=5)

        result = RetrieverResult(items=[])
        ctx = Context(tenant_id="test")

        reranked = await reranker.rerank("test", result, ctx)
        assert len(reranked.items) == 0

    async def test_scores_set(self):
        embedder = MockEmbedder(dimension=8)
        reranker = CosineReranker(embedder=embedder, top_k=5)

        items = [
            RetrieverResultItem(content="Alice works at Acme"),
            RetrieverResultItem(content="Bob is a manager"),
        ]
        result = RetrieverResult(items=items)
        ctx = Context(tenant_id="test")

        reranked = await reranker.rerank("Who is Alice?", result, ctx)
        for item in reranked.items:
            assert item.score is not None
            assert isinstance(item.score, float)
