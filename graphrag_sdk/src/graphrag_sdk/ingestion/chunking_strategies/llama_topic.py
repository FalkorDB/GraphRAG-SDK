# GraphRAG SDK 2.0 — Ingestion: LlamaIndex TopicNodeParser Chunking
#
# Wraps llama_index.node_parser.topic.TopicNodeParser inside the SDK's
# ChunkingStrategy interface. Groups text by detected topics/subtopics,
# creating hierarchical topic-aligned chunks. Requires an LLM for topic detection.
#
# Requires: pip install graphrag-sdk[llama]

from __future__ import annotations

import os

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import TextChunk, TextChunks
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy


class LlamaTopicChunking(ChunkingStrategy):
    """Topic-based chunking via LlamaIndex TopicNodeParser.

    Groups text by topic boundaries. Uses an OpenAI LLM to detect
    when the topic shifts, creating chunks aligned to topical sections.

    Args:
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
        llm_model_name: OpenAI model to use for topic detection. Default ``gpt-4o-mini``.
        similarity_threshold: How sensitive to topic changes (higher = more splits). Default 0.8.
        window_size: Sentences to consider when analyzing topic shifts. Default 3.
        max_chunk_size: Safety cap on chunk size in tokens. Default 512.

    Example::

        chunker = LlamaTopicChunking(api_key="sk-...")
        result = await chunker.chunk(text, ctx)
    """

    def __init__(
        self,
        api_key: str | None = None,
        llm_model_name: str = "gpt-4o-mini",
        similarity_threshold: float = 0.8,
        window_size: int = 3,
        max_chunk_size: int = 512,
    ) -> None:
        super().__init__()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.llm_model_name = llm_model_name
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.max_chunk_size = max_chunk_size

    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        ctx.log(
            f"Chunking text ({len(text)} chars) with "
            f"LlamaTopicChunking(threshold={self.similarity_threshold}, "
            f"window={self.window_size}, max={self.max_chunk_size})"
        )

        try:
            from llama_index.node_parser.topic import TopicNodeParser
            from llama_index.llms.openai import OpenAI as LlamaOpenAI
            from llama_index.core import Document
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "LlamaIndex is required for LlamaTopicChunking. "
                "Install with: pip install graphrag-sdk[llama]"
            )

        llm = LlamaOpenAI(model=self.llm_model_name, api_key=self.api_key)
        splitter = TopicNodeParser.from_defaults(
            llm=llm,
            max_chunk_size=self.max_chunk_size,
            similarity_threshold=self.similarity_threshold,
            window_size=self.window_size,
        )
        nodes = await splitter.aget_nodes_from_documents([Document(text=text)])

        chunks = [
            TextChunk(
                text=node.text,
                index=i,
                metadata={
                    "strategy": "llama_topic",
                    "similarity_threshold": self.similarity_threshold,
                    "window_size": self.window_size,
                    "max_chunk_size": self.max_chunk_size,
                    "char_count": len(node.text),
                },
            )
            for i, node in enumerate(
                node for node in nodes if node.text.strip()
            )
        ]

        ctx.log(f"LlamaTopicChunking produced {len(chunks)} chunks")
        return TextChunks(chunks=chunks)
