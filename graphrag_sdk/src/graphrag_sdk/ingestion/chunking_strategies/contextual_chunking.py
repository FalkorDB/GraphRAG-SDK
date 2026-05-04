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

import tiktoken

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import DocumentOutput, TextChunk, TextChunks
from graphrag_sdk.core.providers import LLMInterface
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy

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
        base_chunker: The underlying chunking strategy to use. When supplied,
            ``max_tokens``, ``overlap_sentences``, and ``encoding_name`` must
            **not** be passed — configure those directly on the base chunker.
            If omitted, defaults to
            ``SentenceTokenCapChunking(max_tokens, overlap_sentences, encoding_name)``.
        max_tokens: Token cap per chunk for the default base chunker. Default 512.
            Ignored (and forbidden) when *base_chunker* is provided.
        overlap_sentences: Sentence overlap for the default base chunker. Default 2.
            Ignored (and forbidden) when *base_chunker* is provided.
        encoding_name: tiktoken encoding for token counting. Default ``cl100k_base``.
            Ignored (and forbidden) when *base_chunker* is provided.
        max_document_tokens: Maximum tokens of the source document included in each
            context prompt. Documents exceeding this are truncated before being sent
            to the LLM, preventing context-window overflows on large inputs. Default 16 000.

    Example::

        # No base chunker — shorthand kwargs configure the default one
        chunker = ContextualChunking(llm=my_llm, max_tokens=256, overlap_sentences=1)

        # Custom base chunker — configure it directly, pass no shorthand kwargs
        chunker = ContextualChunking(llm=my_llm, base_chunker=StructuralChunking(max_tokens=512))
        result = await chunker.chunk_document(doc, ctx)
    """

    _UNSET = object()  # sentinel for "caller did not pass this kwarg"

    def __init__(
        self,
        llm: LLMInterface,
        base_chunker: ChunkingStrategy | None = None,
        max_tokens: int | object = _UNSET,
        overlap_sentences: int | object = _UNSET,
        encoding_name: str | object = _UNSET,
        max_document_tokens: int = 16_000,
    ) -> None:
        super().__init__()
        self.llm = llm

        if base_chunker is not None:
            # Check that none of the shorthand kwargs were explicitly supplied
            _conflicts = [
                name
                for name, val in (
                    ("max_tokens", max_tokens),
                    ("overlap_sentences", overlap_sentences),
                    ("encoding_name", encoding_name),
                )
                if val is not ContextualChunking._UNSET
            ]
            if _conflicts:
                raise TypeError(
                    f"ContextualChunking: {', '.join(_conflicts)} "
                    "cannot be used together with 'base_chunker'. "
                    "Configure those parameters on the base_chunker directly."
                )
            self.base_chunker = base_chunker
            # Use encoding_name solely for document-truncation token counting
            _encoding_name = "cl100k_base"
        else:
            _max_tokens = max_tokens if max_tokens is not ContextualChunking._UNSET else 512
            _overlap = (
                overlap_sentences
                if overlap_sentences is not ContextualChunking._UNSET
                else 2
            )
            _encoding_name = (
                encoding_name
                if encoding_name is not ContextualChunking._UNSET
                else "cl100k_base"
            )

            from graphrag_sdk.ingestion.chunking_strategies.sentence_token_cap import (
                SentenceTokenCapChunking,
            )

            self.base_chunker = SentenceTokenCapChunking(
                max_tokens=_max_tokens,
                overlap_sentences=_overlap,
                encoding_name=_encoding_name,
            )

        self.encoding_name = _encoding_name
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
                    "strategy": "contextual_chunking",
                    "base_strategy": chunk.metadata.get("strategy"),
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
