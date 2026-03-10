# GraphRAG SDK 2.0 — Ingestion: Callable Chunking Adapter
#
# Adapts any user-supplied function into the SDK's ChunkingStrategy interface.
# This lets users plug in any chunking framework (LlamaIndex, LangChain,
# Unstructured, spaCy, custom logic, etc.) without the SDK carrying those deps.

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Union

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import TextChunk, TextChunks
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy

# Accepted signatures: sync or async, returning list of strings.
ChunkFn = Union[Callable[[str], list[str]], Callable[[str], Awaitable[list[str]]]]


class CallableChunking(ChunkingStrategy):
    """Adapt any ``text → list[str]`` function into a ChunkingStrategy.

    Works with both sync and async callables.  The SDK never needs to
    know which chunking library you use — just wrap it in a function.

    Args:
        fn: A callable that takes raw text and returns a list of chunk strings.
            Can be sync or async.
        strategy_name: Optional label stored in chunk metadata. Default ``"custom"``.

    Examples::

        # LlamaIndex
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core import Document

        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        chunker = CallableChunking(
            lambda text: [n.text for n in splitter.get_nodes_from_documents([Document(text=text)])]
        )

        # LangChain
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        lc = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunker = CallableChunking(lc.split_text)

        # Plain function
        chunker = CallableChunking(lambda text: text.split("\\n\\n"))
    """

    def __init__(self, fn: ChunkFn, *, strategy_name: str = "custom") -> None:
        super().__init__()
        self.fn = fn
        self.strategy_name = strategy_name

    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        ctx.log(
            f"Chunking text ({len(text)} chars) with "
            f"CallableChunking(strategy={self.strategy_name!r})"
        )

        if asyncio.iscoroutinefunction(self.fn):
            raw_chunks = await self.fn(text)
        else:
            raw_chunks = self.fn(text)  # type: ignore[arg-type]

        chunks = [
            TextChunk(
                text=c,
                index=i,
                metadata={
                    "strategy": self.strategy_name,
                    "char_count": len(c),
                },
            )
            for i, c in enumerate(c for c in raw_chunks if c.strip())
        ]

        ctx.log(f"CallableChunking produced {len(chunks)} chunks")
        return TextChunks(chunks=chunks)
