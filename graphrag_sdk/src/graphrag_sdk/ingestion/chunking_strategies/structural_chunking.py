# GraphRAG SDK — Ingestion: Structural Chunking
#
# A format-agnostic chunker that groups parsed DocumentElements.
# Keeps paragraphs and their hierarchical context (headers) together.
#
# Delegates oversized blocks to a fallback chunker (e.g., SentenceTokenCapChunking).

from __future__ import annotations

import tiktoken

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import DocumentElement, DocumentOutput, TextChunk, TextChunks
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy


class StructuralChunking(ChunkingStrategy):
    """Groups document elements based on structure, constrained by a token cap.

    Reads ``document.elements`` (a tree/list of ``DocumentElement``).
    Groups them into chunks up to ``max_tokens``.
    If a single element exceeds ``max_tokens``, delegates it to ``fallback_chunker``.

    Args:
        fallback_chunker: Chunker to handle oversized elements. Default: SentenceTokenCapChunking.
        max_tokens: Maximum tokens per structural chunk. Default: 512.
        encoding_name: tiktoken encoding to use. Default ``cl100k_base``.
    """

    def __init__(
        self,
        fallback_chunker: ChunkingStrategy | None = None,
        max_tokens: int = 512,
        encoding_name: str = "cl100k_base",
    ) -> None:
        super().__init__()
        if fallback_chunker is None:
            from graphrag_sdk.ingestion.chunking_strategies.sentence_token_cap import (
                SentenceTokenCapChunking,
            )

            self.fallback_chunker = SentenceTokenCapChunking(
                max_tokens=max_tokens,
                encoding_name=encoding_name,
            )
        else:
            self.fallback_chunker = fallback_chunker

        self.max_tokens = max_tokens
        self.encoding_name = encoding_name

    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        ctx.log("StructuralChunking: No structural elements found. Delegating to fallback chunker.")
        return await self.fallback_chunker.chunk(text, ctx)

    async def chunk_document(self, document: DocumentOutput, ctx: Context) -> TextChunks:
        if not document.elements:
            return await self.chunk(document.text, ctx)

        ctx.log(f"StructuralChunking: processing {len(document.elements)} root elements.")
        enc = tiktoken.get_encoding(self.encoding_name)
        chunks: list[TextChunk] = []
        chunk_index = 0

        # Flatten elements
        flat_elements: list[DocumentElement] = []

        def _flatten(elements: list[DocumentElement]) -> None:
            for el in elements:
                if el.content and el.content.strip():
                    flat_elements.append(el)
                if el.children:
                    _flatten(el.children)

        _flatten(document.elements)

        if not flat_elements:
            return await self.chunk(document.text, ctx)

        buf: list[str] = []
        buf_tokens = 0

        def _flush_buffer() -> None:
            nonlocal chunk_index, buf, buf_tokens
            if not buf:
                return
            chunks.append(
                TextChunk(
                    text="\n\n".join(buf),
                    index=chunk_index,
                    metadata={
                        "strategy": "structural_chunking",
                        "token_count": buf_tokens,
                    },
                )
            )
            chunk_index += 1
            buf = []
            buf_tokens = 0

        for el in flat_elements:
            # Build element text with context prefix
            prefix = " > ".join(el.breadcrumbs) if el.breadcrumbs else ""
            # Don't duplicate the prefix if the element content already starts with it (e.g. headers)
            if prefix and el.type != "header":
                el_text = f"{prefix}\n{el.content}"
            else:
                el_text = el.content or ""

            el_tokens = len(enc.encode(el_text))

            if el_tokens > self.max_tokens:
                # Emit current buffer first
                _flush_buffer()

                # Delegate the huge element
                ctx.log(f"Delegating oversized element ({el_tokens} tokens) to fallback chunker")
                sub_chunks = await self.fallback_chunker.chunk(el.content or "", ctx)
                total_parts = len(sub_chunks.chunks)
                for idx, sc in enumerate(sub_chunks.chunks, start=1):
                    sc.index = chunk_index
                    part_suffix = f" [Parte {idx}/{total_parts}]" if total_parts > 1 else ""
                    
                    if prefix and el.type != "header":
                        sc.text = f"{prefix}{part_suffix}\n{sc.text}"
                    elif total_parts > 1:
                        sc.text = f"[Elemento Fragmentado - Parte {idx}/{total_parts}]\n{sc.text}"
                        
                    chunks.append(sc)
                    chunk_index += 1
                continue

            if buf_tokens + el_tokens <= self.max_tokens:
                buf.append(el_text)
                buf_tokens += el_tokens
            else:
                # Emit current buffer
                _flush_buffer()
                buf = [el_text]
                buf_tokens = el_tokens

        # Emit any remaining buffer
        _flush_buffer()

        ctx.log(f"StructuralChunking produced {len(chunks)} chunks")
        return TextChunks(chunks=chunks)

