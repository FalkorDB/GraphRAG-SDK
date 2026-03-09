# GraphRAG SDK 2.0 — Ingestion: Contextual Chunking
#
# Anthropic's approach: for each chunk, ask the LLM to generate a
# short context summary ("This chunk is from Chapter 3, describing
# Elias's confrontation with the harbor master...") and prepend it
# to the chunk text before embedding.
#
# Better context per chunk → better retrieval for co-reference and
# cross-document questions.
#
# Cost: 1 extra LLM call per chunk during ingestion.
# Reference: https://www.anthropic.com/news/contextual-retrieval

from __future__ import annotations

import re

import tiktoken

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import TextChunk, TextChunks
from graphrag_sdk.core.providers import LLMInterface
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

_CONTEXT_PROMPT = (
    "Here is a document:\n"
    "<document>\n"
    "{full_document}\n"
    "</document>\n\n"
    "Here is a chunk from that document:\n"
    "<chunk>\n"
    "{chunk_text}\n"
    "</chunk>\n\n"
    "Write a short (1-2 sentence) context that situates this chunk within "
    "the overall document. Focus on who/what/where so that a reader of only "
    "this chunk can understand its place in the document. "
    "Reply with ONLY the context sentences, nothing else."
)


class ContextualChunking(ChunkingStrategy):
    """Contextual chunking: sentence-boundary split + LLM-generated context prefix.

    Each chunk gets a short LLM-generated context summary prepended to
    its text before storage and embedding.  This dramatically improves
    retrieval for questions that require cross-chunk co-reference
    resolution (e.g. pronouns, "the town", "the battle").

    Chunks are first produced by sentence-boundary splitting with a token
    cap, then enriched with LLM context in a single batched call.

    Args:
        llm: LLM provider used to generate context summaries.
        max_tokens: Token cap per chunk. Default 512.
        overlap_sentences: Sentences shared between consecutive chunks. Default 2.
        encoding_name: tiktoken encoding for token counting. Default ``cl100k_base``.
        max_document_tokens: Maximum tokens of the source document included in each
            context prompt. Documents exceeding this are truncated before being sent
            to the LLM, preventing context-window overflows on large inputs. Default 16 000.

    Example::

        chunker = ContextualChunking(llm=my_llm, max_tokens=512)
        result = await chunker.chunk(text, ctx)
    """

    def __init__(
        self,
        llm: LLMInterface,
        max_tokens: int = 512,
        overlap_sentences: int = 2,
        encoding_name: str = "cl100k_base",
        max_document_tokens: int = 16_000,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.encoding_name = encoding_name
        self.max_document_tokens = max_document_tokens

    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        ctx.log(
            f"Chunking text ({len(text)} chars) with "
            f"ContextualChunking(max_tokens={self.max_tokens})"
        )

        enc = tiktoken.get_encoding(self.encoding_name)
        # ── 1. Sentence split ────────────────────────────────────────
        sentences = [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]
        if not sentences:
            return TextChunks(chunks=[])

        token_counts = [len(enc.encode(s)) for s in sentences]

        # ── 2. Greedily merge into token-capped chunks ───────────────
        raw_chunks: list[tuple[str, int]] = []  # (text, token_count)
        start = 0

        while start < len(sentences):
            buf: list[str] = []
            buf_tokens = 0
            j = start

            while j < len(sentences):
                needed = token_counts[j] + (1 if buf else 0)
                if buf_tokens + needed <= self.max_tokens:
                    buf.append(sentences[j])
                    buf_tokens += needed
                    j += 1
                else:
                    break

            if not buf:
                buf = [sentences[start]]
                buf_tokens = token_counts[start]
                j = start + 1

            raw_chunks.append((" ".join(buf), buf_tokens))

            if j >= len(sentences):
                break
            start = max(j - self.overlap_sentences, start + 1)

        # ── 3. Batch LLM call for context generation ─────────────────
        # Truncate the document reference to avoid exceeding the LLM context window.
        doc_tokens = enc.encode(text)
        if len(doc_tokens) > self.max_document_tokens:
            ctx.log(
                f"Document ({len(doc_tokens)} tokens) exceeds max_document_tokens "
                f"({self.max_document_tokens}); truncating for context prompts."
            )
            document_ref = enc.decode(doc_tokens[: self.max_document_tokens])
        else:
            document_ref = text

        prompts = [
            _CONTEXT_PROMPT.replace("{full_document}", document_ref).replace("{chunk_text}", chunk_text)
            for chunk_text, _ in raw_chunks
        ]

        ctx.log(f"Generating context for {len(prompts)} chunks via LLM...")
        batch_results = await self.llm.abatch_invoke(prompts)

        # Build result map: index → context string
        context_map: dict[int, str] = {}
        for item in batch_results:
            if item.ok:
                context_map[item.index] = item.response.content.strip()
            else:
                ctx.log(
                    f"Context generation failed for chunk {item.index}: {item.error}",
                )
                context_map[item.index] = ""

        # ── 4. Build final TextChunk list ────────────────────────────
        chunks: list[TextChunk] = []
        for i, (chunk_text, tok_count) in enumerate(raw_chunks):
            context = context_map.get(i, "")
            # Prepend context to the chunk text for embedding/storage
            enriched_text = f"{context}\n\n{chunk_text}" if context else chunk_text
            chunks.append(
                TextChunk(
                    text=enriched_text,
                    index=i,
                    metadata={
                        "strategy": "contextual_chunking",
                        "max_tokens": self.max_tokens,
                        "overlap_sentences": self.overlap_sentences,
                        "token_count": len(enc.encode(enriched_text)),
                        "raw_token_count": tok_count,
                        "char_count": len(enriched_text),
                        "context_prefix": context,
                        "original_chunk": chunk_text,
                    },
                )
            )

        ctx.log(f"ContextualChunking produced {len(chunks)} enriched chunks")
        return TextChunks(chunks=chunks)
