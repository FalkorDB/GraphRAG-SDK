# GraphRAG SDK 2.0 — Ingestion: Chunking Strategy ABC
# Pattern: Strategy — every text splitting approach implements this interface.

from __future__ import annotations

from abc import ABC, abstractmethod

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import TextChunks


class ChunkingStrategy(ABC):
    """Abstract base class for text chunking strategies.

    A chunking strategy takes raw text and produces a ``TextChunks``
    collection. Each chunk carries a unique ID for provenance tracking.

    Example::

        class SemanticChunker(ChunkingStrategy):
            async def chunk(self, text: str, ctx: Context) -> TextChunks:
                # Use embeddings to find semantic boundaries
                ...
    """

    @abstractmethod
    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        """Split text into chunks.

        Args:
            text: Raw text to split.
            ctx: Execution context.

        Returns:
            TextChunks collection.
        """
        ...
