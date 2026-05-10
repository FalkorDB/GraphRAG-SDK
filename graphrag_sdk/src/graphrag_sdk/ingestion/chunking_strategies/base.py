# GraphRAG SDK — Ingestion: Chunking Strategy ABC
# Pattern: Strategy — every text splitting approach implements this interface.

from __future__ import annotations

from abc import ABC, abstractmethod

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import DocumentOutput, TextChunks


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

    async def chunk_document(self, document: DocumentOutput, ctx: Context) -> TextChunks:
        """Split a document into chunks.

        By default, delegates to ``chunk`` using the raw text. Strategies that need
        structural elements can override this method.

        Args:
            document: DocumentOutput with text and optional structural elements.
            ctx: Execution context.

        Returns:
            TextChunks collection.
        """
        return await self.chunk(document.text, ctx)
