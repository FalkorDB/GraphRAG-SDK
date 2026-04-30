# GraphRAG SDK — Ingestion: Contextual Chunking
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
from graphrag_sdk.core.models import DocumentOutput, TextChunk, TextChunks
from graphrag_sdk.core.providers import LLMInterface
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

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
    """Contextual chunking: chunking + LLM-generated context prefix.

    Each chunk gets a short LLM-generated context summary prepended to
    its text before storage and embedding.  This dramatically improves
    retrieval for questions that require cross-chunk co-reference
    resolution (e.g. pronouns, "the town", "the battle").

    Chunks are first produced by the ``base_chunker``, then enriched
    with LLM context in a single batched call.

    Args:
        llm: LLM provider used to generate context summaries.
        base_chunker: The underlying chunking strategy to use. If None,
            defaults to ``SentenceTokenCapChunking(max_tokens, overlap_sentences)``.
        max_tokens: Token cap per chunk (used if base_chunker is None). Default 512.
        overlap_sentences: Sentences shared between chunks (used if base_chunker is None).
            Default 2.
        encoding_name: tiktoken encoding for token counting. Default ``cl100k_base``.
        max_document_tokens: Maximum tokens of the source document included in each
            context prompt. Documents exceeding this are truncated before being sent
            to the LLM, preventing context-window overflows on large inputs. Default 16 000.

    Example::

        chunker = ContextualChunking(llm=my_llm, base_chunker=StructuralChunking())
        result = await chunker.chunk_document(doc, ctx)
    """

    def __init__(
        self,
        llm: LLMInterface,
        base_chunker: ChunkingStrategy | None = None,
        max_tokens: int = 512,
        overlap_sentences: int = 2,
        encoding_name: str = "cl100k_base",
        max_document_tokens: int = 16_000,
    ) -> None:
        super().__init__()
        self.llm = llm

        if base_chunker is None:
            from graphrag_sdk.ingestion.chunking_strategies.sentence_token_cap import (
                SentenceTokenCapChunking,
            )

            self.base_chunker = SentenceTokenCapChunking(
                max_tokens=max_tokens,
                overlap_sentences=overlap_sentences,
                encoding_name=encoding_name,
            )
        else:
            self.base_chunker = base_chunker

        self.encoding_name = encoding_name
        self.max_document_tokens = max_document_tokens

    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        ctx.log(f"Chunking text via base chunker: {self.base_chunker.__class__.__name__}")
        raw_chunks = await self.base_chunker.chunk(text, ctx)
        return await self._enrich_chunks(raw_chunks, text, ctx)

    async def chunk_document(self, document: DocumentOutput, ctx: Context) -> TextChunks:
        ctx.log(f"Chunking document via base chunker: {self.base_chunker.__class__.__name__}")
        raw_chunks = await self.base_chunker.chunk_document(document, ctx)
        return await self._enrich_chunks(raw_chunks, document.text, ctx)

    async def _enrich_chunks(
        self, raw_chunks: TextChunks, document_text: str, ctx: Context
    ) -> TextChunks:
        if not raw_chunks.chunks:
            return raw_chunks

        enc = tiktoken.get_encoding(self.encoding_name)

        # Truncate the document reference to avoid exceeding the LLM context window.
        doc_tokens = enc.encode(document_text)
        if len(doc_tokens) > self.max_document_tokens:
            ctx.log(
                f"Document ({len(doc_tokens)} tokens) exceeds max_document_tokens "
                f"({self.max_document_tokens}); truncating for context prompts."
            )
            document_ref = enc.decode(doc_tokens[: self.max_document_tokens])
        else:
            document_ref = document_text

        prompts = [
            _CONTEXT_PROMPT.replace("{full_document}", document_ref).replace(
                "{chunk_text}", chunk.text
            )
            for chunk in raw_chunks.chunks
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

        # Build final TextChunk list
        enriched_chunks: list[TextChunk] = []
        for i, chunk in enumerate(raw_chunks.chunks):
            context = context_map.get(i, "")
            enriched_text = f"{context}\n\n{chunk.text}" if context else chunk.text

            # Merge metadata
            new_metadata = dict(chunk.metadata)
            new_metadata.update(
                {
                    "contextual_enriched": bool(context),
                    "context_prefix": context,
                    "original_chunk": chunk.text,
                    "token_count": len(enc.encode(enriched_text)),
                    "char_count": len(enriched_text),
                }
            )

            enriched_chunks.append(
                TextChunk(
                    text=enriched_text,
                    index=chunk.index,
                    metadata=new_metadata,
                    uid=chunk.uid,  # Preserve the UID for graph provenance
                )
            )

        ctx.log(f"ContextualChunking produced {len(enriched_chunks)} enriched chunks")
        return TextChunks(chunks=enriched_chunks)
