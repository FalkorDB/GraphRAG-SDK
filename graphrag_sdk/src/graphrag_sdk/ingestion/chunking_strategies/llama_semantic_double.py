# GraphRAG SDK 2.0 — Ingestion: LlamaIndex SemanticDoubleMergingSplitterNodeParser Chunking
#
# Wraps llama_index.core.node_parser.SemanticDoubleMergingSplitterNodeParser inside the
# SDK's ChunkingStrategy interface. Two-pass approach: first splits into tiny pieces,
# then re-merges semantically similar adjacent chunks.
#
# Requires: pip install graphrag-sdk[llama]

from __future__ import annotations

import os

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import TextChunk, TextChunks
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy


class LlamaSemanticDoubleChunking(ChunkingStrategy):
    """Two-pass semantic chunking via LlamaIndex SemanticDoubleMergingSplitterNodeParser.

    Pass 1 — splits text into small initial pieces.
    Pass 2 — re-merges semantically similar adjacent pieces into coherent chunks,
    then appends any orphaned small pieces to their nearest neighbor.

    Uses OpenAI text-embedding-3-small for similarity measurement.

    Args:
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
        initial_threshold: Similarity bar for first-pass merging. Lower = more merging. Default 0.4.
        appending_threshold: Similarity bar for attaching orphan pieces. Default 0.5.
        merging_threshold: Final-pass merge strictness. Default 0.5.
        max_chunk_size: Safety cap — no merged chunk exceeds this size. Default 512.

    Example::

        chunker = LlamaSemanticDoubleChunking(api_key="sk-...")
        result = await chunker.chunk(text, ctx)
    """

    def __init__(
        self,
        api_key: str | None = None,
        initial_threshold: float = 0.4,
        appending_threshold: float = 0.5,
        merging_threshold: float = 0.5,
        max_chunk_size: int = 512,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.initial_threshold = initial_threshold
        self.appending_threshold = appending_threshold
        self.merging_threshold = merging_threshold
        self.max_chunk_size = max_chunk_size

    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        ctx.log(
            f"Chunking text ({len(text)} chars) with "
            f"LlamaSemanticDoubleChunking(initial={self.initial_threshold}, "
            f"appending={self.appending_threshold}, max={self.max_chunk_size})"
        )

        try:
            from llama_index.core.node_parser import SemanticDoubleMergingSplitterNodeParser
            from llama_index.core.node_parser import LanguageConfig
            from llama_index.embeddings.openai import OpenAIEmbedding
            from llama_index.core import Document
        except ImportError:
            raise ImportError(
                "LlamaIndex is required for LlamaSemanticDoubleChunking. "
                "Install with: pip install graphrag-sdk[llama]"
            )

        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=self.api_key,
        )
        language_config = LanguageConfig(language="english")
        splitter = SemanticDoubleMergingSplitterNodeParser(
            language_config=language_config,
            initial_threshold=self.initial_threshold,
            appending_threshold=self.appending_threshold,
            merging_threshold=self.merging_threshold,
            max_chunk_size=self.max_chunk_size,
            embed_model=embed_model,
        )
        nodes = splitter.get_nodes_from_documents([Document(text=text)])

        chunks = [
            TextChunk(
                text=node.text,
                index=i,
                metadata={
                    "strategy": "llama_semantic_double",
                    "initial_threshold": self.initial_threshold,
                    "appending_threshold": self.appending_threshold,
                    "merging_threshold": self.merging_threshold,
                    "max_chunk_size": self.max_chunk_size,
                    "char_count": len(node.text),
                },
            )
            for i, node in enumerate(nodes)
            if node.text.strip()
        ]

        ctx.log(f"LlamaSemanticDoubleChunking produced {len(chunks)} chunks")
        return TextChunks(chunks=chunks)
