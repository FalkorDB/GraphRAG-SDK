# GraphRAG SDK 2.0 — Retrieval: Reranking Strategy ABC
# Pattern: Strategy — composable result quality layer.
# Origin: User design — absent in Neo4j.

from __future__ import annotations

from abc import ABC, abstractmethod

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import RetrieverResult


class RerankingStrategy(ABC):
    """Abstract base class for result reranking strategies.

    A reranking strategy takes an existing ``RetrieverResult`` and
    reorders/filters items to improve quality. This is a composable
    layer applied after retrieval.

    Example::

        class CrossEncoderReranker(RerankingStrategy):
            async def rerank(self, query, result, ctx):
                # Score each item with a cross-encoder model
                ...
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        result: RetrieverResult,
        ctx: Context,
    ) -> RetrieverResult:
        """Rerank retrieval results.

        Args:
            query: The original user query.
            result: Retrieval results to rerank.
            ctx: Execution context.

        Returns:
            Reranked RetrieverResult (same or fewer items, reordered).
        """
        ...
