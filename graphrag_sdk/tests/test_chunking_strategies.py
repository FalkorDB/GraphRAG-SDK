"""Unit tests for all 7 supported chunking strategies.

Strategies under test:
  1. FixedSizeChunking          (char baseline, zero deps)
  2. SentenceTokenCapChunking   (tiktoken, no LLM)
  3. ContextualChunking         (tiktoken + LLM — mock LLM used)
  4. LlamaSentenceChunking      (optional: requires graphrag-sdk[llama])
  5. LlamaSemanticChunking      (optional: requires graphrag-sdk[llama])
  6. LlamaSemanticDoubleChunking(optional: requires graphrag-sdk[llama])
  7. LlamaTopicChunking         (optional: requires graphrag-sdk[llama])

Llama tests are skipped automatically when the optional deps are absent.
"""
from __future__ import annotations

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import LLMResponse
from graphrag_sdk.core.providers import LLMBatchItem, LLMInterface
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
from graphrag_sdk.ingestion.chunking_strategies.sentence_token_cap import SentenceTokenCapChunking
from graphrag_sdk.ingestion.chunking_strategies.contextual_chunking import ContextualChunking

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


@pytest.fixture
def ctx() -> Context:
    return Context(tenant_id="test", latency_budget_ms=30_000.0)


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
# 4–7. Llama strategies (skipped when [llama] extra not installed)
# ---------------------------------------------------------------------------

import importlib as _importlib
import os as _os

_has_llama = _importlib.util.find_spec("llama_index") is not None
_needs_llama = pytest.mark.skipif(
    not _has_llama,
    reason="graphrag-sdk[llama] not installed",
)

_has_openai_key = bool(_os.getenv("OPENAI_API_KEY"))
_needs_openai_key = pytest.mark.skipif(
    not _has_openai_key,
    reason="OPENAI_API_KEY not set — skipping llama strategies that call OpenAI",
)


@_needs_llama
class TestLlamaSentenceChunking:
    """LlamaIndex sentence splitter — no API key needed."""

    async def test_produces_chunks(self, ctx):
        from graphrag_sdk.ingestion.chunking_strategies.llama_sentence import LlamaSentenceChunking
        chunker = LlamaSentenceChunking(chunk_size=256, chunk_overlap=20)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1

    async def test_no_empty_chunk_text(self, ctx):
        from graphrag_sdk.ingestion.chunking_strategies.llama_sentence import LlamaSentenceChunking
        chunker = LlamaSentenceChunking(chunk_size=256, chunk_overlap=20)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        for chunk in result.chunks:
            assert chunk.text.strip() != ""

    async def test_sequential_indices(self, ctx):
        from graphrag_sdk.ingestion.chunking_strategies.llama_sentence import LlamaSentenceChunking
        chunker = LlamaSentenceChunking(chunk_size=256, chunk_overlap=20)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        for i, chunk in enumerate(result.chunks):
            assert chunk.index == i

    async def test_metadata_has_strategy(self, ctx):
        from graphrag_sdk.ingestion.chunking_strategies.llama_sentence import LlamaSentenceChunking
        chunker = LlamaSentenceChunking(chunk_size=512, chunk_overlap=50)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1
        assert "strategy" in result.chunks[0].metadata


@_needs_llama
@_needs_openai_key
class TestLlamaSemanticChunking:
    """Requires OPENAI_API_KEY (calls OpenAI embeddings during chunking)."""

    async def test_produces_chunks(self, ctx):
        from graphrag_sdk.ingestion.chunking_strategies.llama_semantic import LlamaSemanticChunking
        chunker = LlamaSemanticChunking()
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1

    async def test_no_empty_chunk_text(self, ctx):
        from graphrag_sdk.ingestion.chunking_strategies.llama_semantic import LlamaSemanticChunking
        chunker = LlamaSemanticChunking()
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        for chunk in result.chunks:
            assert chunk.text.strip() != ""

    async def test_metadata_has_strategy(self, ctx):
        from graphrag_sdk.ingestion.chunking_strategies.llama_semantic import LlamaSemanticChunking
        chunker = LlamaSemanticChunking()
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1
        assert "strategy" in result.chunks[0].metadata


@_needs_llama
@_needs_openai_key
class TestLlamaSemanticDoubleChunking:
    """Requires OPENAI_API_KEY (calls OpenAI embeddings during chunking)."""

    async def test_produces_chunks(self, ctx):
        from graphrag_sdk.ingestion.chunking_strategies.llama_semantic_double import LlamaSemanticDoubleChunking
        chunker = LlamaSemanticDoubleChunking()
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1

    async def test_no_empty_chunk_text(self, ctx):
        from graphrag_sdk.ingestion.chunking_strategies.llama_semantic_double import LlamaSemanticDoubleChunking
        chunker = LlamaSemanticDoubleChunking()
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        for chunk in result.chunks:
            assert chunk.text.strip() != ""

    async def test_metadata_has_strategy(self, ctx):
        from graphrag_sdk.ingestion.chunking_strategies.llama_semantic_double import LlamaSemanticDoubleChunking
        chunker = LlamaSemanticDoubleChunking()
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1
        assert "strategy" in result.chunks[0].metadata


@_needs_llama
@_needs_openai_key
class TestLlamaTopicChunking:
    """Requires OPENAI_API_KEY (calls OpenAI LLM during chunking)."""

    async def test_produces_chunks(self, ctx):
        from graphrag_sdk.ingestion.chunking_strategies.llama_topic import LlamaTopicChunking
        chunker = LlamaTopicChunking()
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1

    async def test_no_empty_chunk_text(self, ctx):
        from graphrag_sdk.ingestion.chunking_strategies.llama_topic import LlamaTopicChunking
        chunker = LlamaTopicChunking()
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        for chunk in result.chunks:
            assert chunk.text.strip() != ""

    async def test_metadata_has_strategy(self, ctx):
        from graphrag_sdk.ingestion.chunking_strategies.llama_topic import LlamaTopicChunking
        chunker = LlamaTopicChunking()
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1
        assert "strategy" in result.chunks[0].metadata
