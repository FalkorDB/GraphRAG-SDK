# GraphRAG SDK 2.0 — Ingestion: Fixed-Size Chunking

from __future__ import annotations

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import TextChunk, TextChunks
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy


class FixedSizeChunking(ChunkingStrategy):
    """Split text into fixed-size character windows with optional overlap.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Example::

        chunker = FixedSizeChunking(chunk_size=500, chunk_overlap=50)
        result = await chunker.chunk(long_text, ctx)
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        ctx.log(
            f"Chunking text ({len(text)} chars) with "
            f"size={self.chunk_size}, overlap={self.chunk_overlap}"
        )

        chunks: list[TextChunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            if chunk_text.strip():  # Skip purely whitespace chunks
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        index=index,
                        metadata={
                            "start_char": start,
                            "end_char": end,
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap,
                        },
                    )
                )
                index += 1

            start += step

        ctx.log(f"Created {len(chunks)} chunks")
        return TextChunks(chunks=chunks)
