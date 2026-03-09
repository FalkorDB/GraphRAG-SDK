# GraphRAG SDK 2.0 — Ingestion: LlamaIndex SentenceSplitter Chunking
#
# Wraps llama_index.core.node_parser.SentenceSplitter inside the SDK's
# ChunkingStrategy interface. Splits at sentence boundaries and groups
# sentences into token-capped chunks with overlap.
#
# Requires: pip install graphrag-sdk[llama]

from __future__ import annotations

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import TextChunk, TextChunks
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy


class LlamaSentenceChunking(ChunkingStrategy):
    """Sentence-boundary chunking via LlamaIndex SentenceSplitter.

    Splits text at sentence boundaries and groups sentences into chunks
    up to ``chunk_size`` tokens with ``chunk_overlap`` token overlap.

    Args:
        chunk_size: Max tokens per chunk. Default 512.
        chunk_overlap: Token overlap between consecutive chunks. Default 50.

    Example::

        chunker = LlamaSentenceChunking(chunk_size=512, chunk_overlap=50)
        result = await chunker.chunk(text, ctx)
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        ctx.log(
            f"Chunking text ({len(text)} chars) with "
            f"LlamaSentenceChunking(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )

        try:
            from llama_index.core.node_parser import SentenceSplitter
            from llama_index.core import Document
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "LlamaIndex is required for LlamaSentenceChunking. "
                "Install with: pip install graphrag-sdk[llama]"
            )

        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        nodes = splitter.get_nodes_from_documents([Document(text=text)])

        chunks = [
            TextChunk(
                text=node.text,
                index=i,
                metadata={
                    "strategy": "llama_sentence",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "char_count": len(node.text),
                },
            )
            for i, node in enumerate(
                node for node in nodes if node.text.strip()
            )
        ]

        ctx.log(f"LlamaSentenceChunking produced {len(chunks)} chunks")
        return TextChunks(chunks=chunks)
