# GraphRAG SDK 2.0 — Retrieval: Local Strategy
# Vector similarity search + 1-hop graph traversal.
# The default retrieval strategy for v1.

from __future__ import annotations

import logging
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import RawSearchResult
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy

logger = logging.getLogger(__name__)


class LocalRetrieval(RetrievalStrategy):
    """Vector similarity + 1-hop graph traversal retrieval.

    1. Embed the query
    2. Vector search for matching chunk nodes
    3. For each chunk, traverse 1-hop to find connected entities
    4. Return chunks + entity context

    Args:
        graph_store: Graph data access object.
        vector_store: Vector data access object.
        embedder: Embedder for query embedding.
        top_k: Number of vector search results.
        include_entities: Whether to include 1-hop entities in results.
    """

    def __init__(
        self,
        graph_store: Any,
        vector_store: Any,
        embedder: Any,
        top_k: int = 5,
        include_entities: bool = True,
    ) -> None:
        super().__init__(graph_store=graph_store, vector_store=vector_store)
        self._embedder = embedder
        self._top_k = top_k
        self._include_entities = include_entities

    async def _execute(
        self,
        query: str,
        ctx: Context,
        **kwargs: Any,
    ) -> RawSearchResult:
        top_k = kwargs.get("top_k", self._top_k)

        # Step 1: Embed the query
        query_vector = await self._embedder.aembed_query(query)

        # Step 2: Vector search for matching chunks
        chunk_results = await self._vector.search(
            query_vector=query_vector,
            top_k=top_k,
        )

        if not chunk_results:
            return RawSearchResult()

        # Step 3: Optionally fetch 1-hop entity context
        records: list[dict[str, Any]] = []
        for chunk in chunk_results:
            record: dict[str, Any] = {
                "chunk_text": chunk.get("text", ""),
                "chunk_id": chunk.get("id", ""),
                "score": chunk.get("score", 0.0),
            }

            if self._include_entities and self._graph:
                # 1-hop: find entities extracted from this chunk
                entities = await self._graph.get_connected_entities(
                    chunk_id=chunk.get("id", ""),
                )
                record["entities"] = entities

            records.append(record)

        return RawSearchResult(
            records=records,
            metadata={"strategy": "local", "top_k": top_k},
        )

    def _format(self, raw: RawSearchResult) -> Any:
        """Format local retrieval results with chunk text + entity context."""
        from graphrag_sdk.core.models import RetrieverResult, RetrieverResultItem

        items: list[RetrieverResultItem] = []
        for record in raw.records:
            parts = [record.get("chunk_text", "")]

            entities = record.get("entities", [])
            if entities:
                entity_strs = [str(e) for e in entities]
                parts.append("Related entities: " + ", ".join(entity_strs))

            items.append(
                RetrieverResultItem(
                    content="\n".join(parts),
                    score=record.get("score"),
                    metadata={"chunk_id": record.get("chunk_id", "")},
                )
            )

        return RetrieverResult(items=items, metadata=raw.metadata)
