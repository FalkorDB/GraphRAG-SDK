# GraphRAG SDK 2.0 — Ingestion: Sentence + Token Cap Chunking
#
# Best of both worlds: never splits mid-sentence (sentence boundaries via
# regex) AND enforces a token cap per chunk (via tiktoken).
# No LLM or embedder needed — tiktoken is a core dependency.

from __future__ import annotations

import re

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import TextChunk, TextChunks
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


class SentenceTokenCapChunking(ChunkingStrategy):
    """Sentence-boundary chunking with a hard token cap per chunk.

    Splits text at sentence endings then greedily merges sentences into
    chunks as long as the total token count stays within ``max_tokens``.
    Once adding the next sentence would exceed the cap, the current chunk
    is emitted and a new one begins, overlapping by ``overlap_sentences``
    sentences for context continuity.

    Args:
        max_tokens: Token cap per chunk. Default 512.
        overlap_sentences: Sentences shared between consecutive chunks. Default 2.
        encoding_name: tiktoken encoding to use. Default ``cl100k_base`` (GPT-4/3.5).

    Example::

        chunker = SentenceTokenCapChunking(max_tokens=512, overlap_sentences=2)
        result = await chunker.chunk(text, ctx)
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_sentences: int = 2,
        encoding_name: str = "cl100k_base",
    ) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.encoding_name = encoding_name

    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        ctx.log(
            f"Chunking text ({len(text)} chars) with "
            f"SentenceTokenCapChunking(max_tokens={self.max_tokens}, overlap={self.overlap_sentences})"
        )

        import tiktoken
        enc = tiktoken.get_encoding(self.encoding_name)

        sentences = [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]
        token_counts = [len(enc.encode(s)) for s in sentences]

        chunks: list[TextChunk] = []
        index = 0
        start = 0

        while start < len(sentences):
            buf: list[str] = []
            buf_tokens = 0
            j = start

            while j < len(sentences):
                needed = token_counts[j] + (1 if buf else 0)  # +1 for space token
                if buf_tokens + needed <= self.max_tokens:
                    buf.append(sentences[j])
                    buf_tokens += needed
                    j += 1
                else:
                    break

            if not buf:
                # Single sentence exceeds cap — emit as-is
                buf = [sentences[start]]
                buf_tokens = token_counts[start]
                j = start + 1

            chunk_text = " ".join(buf)
            chunks.append(TextChunk(
                text=chunk_text,
                index=index,
                metadata={
                    "strategy": "sentence_token_cap",
                    "max_tokens": self.max_tokens,
                    "overlap_sentences": self.overlap_sentences,
                    "token_count": buf_tokens,
                    "sentence_count": len(buf),
                    "char_count": len(chunk_text),
                },
            ))
            index += 1

            if j >= len(sentences):
                break

            # Roll back by overlap_sentences for next window
            start = max(j - self.overlap_sentences, start + 1)

        ctx.log(f"SentenceTokenCapChunking produced {len(chunks)} chunks")
        return TextChunks(chunks=chunks)
