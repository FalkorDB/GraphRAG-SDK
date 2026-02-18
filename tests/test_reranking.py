"""Tests for retrieval/reranking_strategies/base.py."""
from __future__ import annotations

from typing import Any

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import RetrieverResult, RetrieverResultItem
from graphrag_sdk.retrieval.reranking_strategies.base import RerankingStrategy


class ScoreBoostReranker(RerankingStrategy):
    """Test reranker that multiplies scores."""

    def __init__(self, boost: float = 2.0):
        self._boost = boost

    async def rerank(self, query: str, result: RetrieverResult, ctx: Context) -> RetrieverResult:
        new_items = []
        for item in result.items:
            new_score = (item.score or 0.0) * self._boost
            new_items.append(
                RetrieverResultItem(content=item.content, score=new_score, metadata=item.metadata)
            )
        new_items.sort(key=lambda x: x.score or 0.0, reverse=True)
        return RetrieverResult(items=new_items, metadata={"reranked": True})


class TestRerankingStrategy:
    async def test_rerank(self, ctx):
        reranker = ScoreBoostReranker(boost=3.0)
        result = RetrieverResult(
            items=[
                RetrieverResultItem(content="low", score=0.3),
                RetrieverResultItem(content="high", score=0.9),
            ]
        )
        reranked = await reranker.rerank("query", result, ctx)
        assert reranked.items[0].content == "high"
        assert reranked.items[0].score == pytest.approx(2.7)
        assert reranked.metadata["reranked"] is True

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            RerankingStrategy()  # type: ignore[abstract]
