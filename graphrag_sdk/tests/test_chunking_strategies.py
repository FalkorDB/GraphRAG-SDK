"""Unit tests for chunking strategies.

Strategies under test:
  1. FixedSizeChunking        (char baseline, zero deps)
  2. SentenceTokenCapChunking (tiktoken, no LLM)
  3. ContextualChunking       (tiktoken + LLM — mock LLM used)
  4. CallableChunking         (adapts any function — sync & async)
"""
from __future__ import annotations

import pytest

from graphrag_sdk.core.models import LLMResponse
from graphrag_sdk.core.providers import LLMBatchItem, LLMInterface
from graphrag_sdk.ingestion.chunking_strategies.callable_chunking import CallableChunking
from graphrag_sdk.ingestion.chunking_strategies.contextual_chunking import ContextualChunking
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
from graphrag_sdk.ingestion.chunking_strategies.sentence_token_cap import SentenceTokenCapChunking

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "Alice walked to the market on a bright Tuesday morning. "
    "She bought apples, bread, and a bunch of fresh herbs. "
    "The market was busy with vendors calling out their wares. "
    "Bob met her near the cheese stall and they talked for a while. "
    "He mentioned that the old library on Elm Street had finally reopened. "
    "Alice was delighted; she had been waiting months to return to that reading room. "
    "The librarian, an elderly woman named Clara, greeted them warmly. "
    "Clara had worked there for over thirty years and knew every shelf by heart. "
    "Bob borrowed three books on local history while Alice chose a novel. "
    "They left the library just as the afternoon rain began to fall."
)


# ---------------------------------------------------------------------------
# Mock LLM for ContextualChunking
# ---------------------------------------------------------------------------

class _ContextualMockLLM(LLMInterface):
    """Minimal LLM that satisfies ContextualChunking's abatch_invoke requirement."""

    def __init__(self) -> None:
        super().__init__(model_name="mock-contextual")

    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(content="[context summary]")

    async def abatch_invoke(self, prompts: list[str], **kwargs) -> list[LLMBatchItem]:
        return [
            LLMBatchItem(index=i, response=LLMResponse(content="[context summary]"))
            for i in range(len(prompts))
        ]


# ---------------------------------------------------------------------------
# 1. FixedSizeChunking
# ---------------------------------------------------------------------------

class TestFixedSizeChunking:
    async def test_produces_chunks(self, ctx):
        chunker = FixedSizeChunking(chunk_size=100, chunk_overlap=10)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1

    async def test_empty_input(self, ctx):
        chunker = FixedSizeChunking(chunk_size=100, chunk_overlap=0)
        result = await chunker.chunk("", ctx)
        assert result.chunks == []

    async def test_sequential_indices(self, ctx):
        chunker = FixedSizeChunking(chunk_size=50, chunk_overlap=0)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        for i, chunk in enumerate(result.chunks):
            assert chunk.index == i

    async def test_no_empty_chunks(self, ctx):
        chunker = FixedSizeChunking(chunk_size=50, chunk_overlap=5)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        for chunk in result.chunks:
            assert chunk.text.strip() != ""

    async def test_chunk_size_respected(self, ctx):
        chunk_size = 80
        chunker = FixedSizeChunking(chunk_size=chunk_size, chunk_overlap=0)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        # All chunks except possibly the last should be exactly chunk_size
        for chunk in result.chunks[:-1]:
            assert len(chunk.text) == chunk_size


# ---------------------------------------------------------------------------
# 2. SentenceTokenCapChunking
# ---------------------------------------------------------------------------

class TestSentenceTokenCapChunking:
    async def test_produces_chunks(self, ctx):
        chunker = SentenceTokenCapChunking(max_tokens=100, overlap_sentences=1)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1

    async def test_empty_input(self, ctx):
        chunker = SentenceTokenCapChunking(max_tokens=512, overlap_sentences=0)
        result = await chunker.chunk("", ctx)
        assert result.chunks == []

    async def test_sequential_indices(self, ctx):
        chunker = SentenceTokenCapChunking(max_tokens=80, overlap_sentences=1)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        for i, chunk in enumerate(result.chunks):
            assert chunk.index == i

    async def test_no_empty_chunk_text(self, ctx):
        chunker = SentenceTokenCapChunking(max_tokens=80, overlap_sentences=1)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        for chunk in result.chunks:
            assert chunk.text.strip() != ""

    async def test_metadata_has_strategy(self, ctx):
        chunker = SentenceTokenCapChunking(max_tokens=512, overlap_sentences=2)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1
        assert "strategy" in result.chunks[0].metadata

    async def test_small_cap_more_chunks_than_large(self, ctx):
        small = await SentenceTokenCapChunking(max_tokens=30, overlap_sentences=0).chunk(SAMPLE_TEXT, ctx)
        large = await SentenceTokenCapChunking(max_tokens=512, overlap_sentences=0).chunk(SAMPLE_TEXT, ctx)
        assert len(small.chunks) >= len(large.chunks)


# ---------------------------------------------------------------------------
# 3. ContextualChunking
# ---------------------------------------------------------------------------

class TestContextualChunking:
    async def test_produces_chunks(self, ctx):
        chunker = ContextualChunking(llm=_ContextualMockLLM(), max_tokens=200, overlap_sentences=1)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1

    async def test_empty_input(self, ctx):
        chunker = ContextualChunking(llm=_ContextualMockLLM(), max_tokens=512)
        result = await chunker.chunk("", ctx)
        assert result.chunks == []

    async def test_sequential_indices(self, ctx):
        chunker = ContextualChunking(llm=_ContextualMockLLM(), max_tokens=200, overlap_sentences=1)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        for i, chunk in enumerate(result.chunks):
            assert chunk.index == i

    async def test_context_prefix_prepended(self, ctx):
        chunker = ContextualChunking(llm=_ContextualMockLLM(), max_tokens=200, overlap_sentences=1)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        # Every chunk should contain the mock context string
        for chunk in result.chunks:
            assert "[context summary]" in chunk.text

    async def test_metadata_has_context_prefix(self, ctx):
        chunker = ContextualChunking(llm=_ContextualMockLLM(), max_tokens=512)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1
        assert "context_prefix" in result.chunks[0].metadata
        assert "original_chunk" in result.chunks[0].metadata


# ---------------------------------------------------------------------------
# 4. CallableChunking
# ---------------------------------------------------------------------------

class TestCallableChunking:
    """Tests for the framework-agnostic callable adapter."""

    async def test_sync_function(self, ctx):
        chunker = CallableChunking(lambda text: text.split(". "))
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1

    async def test_async_function(self, ctx):
        async def async_split(text: str) -> list[str]:
            return text.split(". ")

        chunker = CallableChunking(async_split)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1

    async def test_empty_input(self, ctx):
        chunker = CallableChunking(lambda text: text.split("\n\n"))
        result = await chunker.chunk("", ctx)
        assert result.chunks == []

    async def test_filters_whitespace_only_chunks(self, ctx):
        chunker = CallableChunking(lambda text: ["hello", "  ", "", "world"])
        result = await chunker.chunk("ignored", ctx)
        assert len(result.chunks) == 2
        assert result.chunks[0].text == "hello"
        assert result.chunks[1].text == "world"

    async def test_sequential_indices(self, ctx):
        chunker = CallableChunking(lambda text: text.split(". "))
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        for i, chunk in enumerate(result.chunks):
            assert chunk.index == i

    async def test_metadata_has_strategy(self, ctx):
        chunker = CallableChunking(
            lambda text: text.split(". "), strategy_name="sentence_split"
        )
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1
        assert result.chunks[0].metadata["strategy"] == "sentence_split"

    async def test_default_strategy_name(self, ctx):
        chunker = CallableChunking(lambda text: [text])
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert result.chunks[0].metadata["strategy"] == "custom"

    async def test_metadata_has_char_count(self, ctx):
        chunker = CallableChunking(lambda text: [text])
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert result.chunks[0].metadata["char_count"] == len(SAMPLE_TEXT)

    async def test_callable_class(self, ctx):
        """A class with __call__ should work too."""

        class ParagraphSplitter:
            def __call__(self, text: str) -> list[str]:
                return text.split("\n\n")

        chunker = CallableChunking(ParagraphSplitter())
        result = await chunker.chunk("Para one.\n\nPara two.", ctx)
        assert len(result.chunks) == 2
