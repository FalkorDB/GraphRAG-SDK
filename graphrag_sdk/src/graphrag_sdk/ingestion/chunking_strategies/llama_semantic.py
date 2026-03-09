# GraphRAG SDK 2.0 — Ingestion: LlamaIndex SemanticSplitterNodeParser Chunking
#
# Wraps llama_index.core.node_parser.SemanticSplitterNodeParser inside the SDK's
# ChunkingStrategy interface. Uses embeddings to detect topic shifts and splits
# where meaning changes significantly.
#
# Requires: pip install graphrag-sdk[llama]

from __future__ import annotations

import os

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import TextChunk, TextChunks
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy


class LlamaSemanticChunking(ChunkingStrategy):
    """Embedding-based semantic chunking via LlamaIndex SemanticSplitterNodeParser.

    Analyzes semantic similarity between sentences and splits where meaning
    changes significantly (topic shifts). Uses OpenAI embeddings to measure
    sentence similarity.

    Args:
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
        embed_model_name: OpenAI embedding model to use. Default ``text-embedding-3-small``.
        buffer_size: Sentences to look ahead when detecting breakpoints. Default 1.
        breakpoint_percentile_threshold: Sensitivity — higher means fewer splits. Default 95.

    Example::

        chunker = LlamaSemanticChunking(api_key="sk-...")
        result = await chunker.chunk(text, ctx)
    """

    def __init__(
        self,
        api_key: str | None = None,
        embed_model_name: str = "text-embedding-3-small",
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
    ) -> None:
        super().__init__()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.embed_model_name = embed_model_name
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold

    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        ctx.log(
            f"Chunking text ({len(text)} chars) with "
            f"LlamaSemanticChunking(buffer={self.buffer_size}, "
            f"threshold={self.breakpoint_percentile_threshold})"
        )

        try:
            from llama_index.core.node_parser import SemanticSplitterNodeParser
            from llama_index.embeddings.openai import OpenAIEmbedding
            from llama_index.core import Document
        except ImportError:
            raise ImportError(
                "LlamaIndex is required for LlamaSemanticChunking. "
                "Install with: pip install graphrag-sdk[llama]"
            )

        embed_model = OpenAIEmbedding(
            model=self.embed_model_name,
            api_key=self.api_key,
        )
        splitter = SemanticSplitterNodeParser(
            buffer_size=self.buffer_size,
            breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
            embed_model=embed_model,
        )
        nodes = await splitter.aget_nodes_from_documents([Document(text=text)])

        chunks = [
            TextChunk(
                text=node.text,
                index=i,
                metadata={
                    "strategy": "llama_semantic",
                    "buffer_size": self.buffer_size,
                    "breakpoint_percentile_threshold": self.breakpoint_percentile_threshold,
                    "char_count": len(node.text),
                },
            )
            for i, node in enumerate(
                node for node in nodes if node.text.strip()
            )
        ]

        ctx.log(f"LlamaSemanticChunking produced {len(chunks)} chunks")
        return TextChunks(chunks=chunks)
