# GraphRAG SDK 2.0 — Retrieval: Cosine Reranker
# Embedding-based reranking using cosine similarity.

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import RetrieverResult, RetrieverResultItem
from graphrag_sdk.core.providers import Embedder
from graphrag_sdk.retrieval.reranking_strategies.base import RerankingStrategy

logger = logging.getLogger(__name__)


class CosineReranker(RerankingStrategy):
    """Rerank retrieval results by cosine similarity to the query.

    Batch-embeds the query and all item contents, computes cosine
    similarity, and returns the top_k items sorted by similarity.

    Args:
        embedder: Embedding provider for vectorising texts.
        top_k: Maximum number of items to return.
    """

    def __init__(self, embedder: Embedder, top_k: int = 15) -> None:
        self._embedder = embedder
        self._top_k = top_k

    async def rerank(
        self,
        query: str,
        result: RetrieverResult,
        ctx: Context,
    ) -> RetrieverResult:
        if not result.items:
            return result

        texts = [query] + [item.content for item in result.items]
        vectors = await self._embedder.aembed_documents(texts)

        query_vec = np.array(vectors[0])
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return RetrieverResult(
                items=result.items[: self._top_k], metadata=result.metadata
            )

        scored: list[tuple[int, float]] = []
        for i, vec in enumerate(vectors[1:]):
            if vec is None:
                scored.append((i, 0.0))
                continue
            v = np.array(vec)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                scored.append((i, 0.0))
            else:
                sim = float(np.dot(query_vec, v) / (q_norm * v_norm))
                scored.append((i, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        items = []
        for idx, sim in scored[: self._top_k]:
            original = result.items[idx]
            items.append(
                RetrieverResultItem(
                    content=original.content,
                    metadata=original.metadata,
                    score=sim,
                )
            )

        return RetrieverResult(items=items, metadata=result.metadata)
