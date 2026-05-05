"""Unit tests for chunking strategies.

Strategies under test:
  1. FixedSizeChunking        (char baseline, zero deps)
  2. SentenceTokenCapChunking (tiktoken, no LLM)
  3. ContextualChunking       (tiktoken + LLM — mock LLM used)
  4. CallableChunking         (adapts any function — sync & async)
  5. StructuralChunking       (Groups document elements based on structure, constrained by a token cap) 
"""
from __future__ import annotations

import pytest

from graphrag_sdk.core.models import LLMResponse
from graphrag_sdk.core.providers import LLMBatchItem, LLMInterface
from graphrag_sdk.ingestion.chunking_strategies.callable_chunking import CallableChunking
from graphrag_sdk.ingestion.chunking_strategies.contextual_chunking import ContextualChunking
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
from graphrag_sdk.ingestion.chunking_strategies.sentence_token_cap import SentenceTokenCapChunking
from graphrag_sdk.ingestion.chunking_strategies.structural_chunking import StructuralChunking

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

    async def test_metadata_strategy_is_contextual_chunking(self, ctx):
        """metadata['strategy'] must always equal 'contextual_chunking'.

        Regression: a previous refactor silently overwrote the strategy key
        without setting it to the contextual value, breaking any downstream
        code that filters chunks by metadata['strategy'].
        """
        chunker = ContextualChunking(llm=_ContextualMockLLM(), max_tokens=512)
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1
        for chunk in result.chunks:
            assert chunk.metadata.get("strategy") == "contextual_chunking"


    async def test_metadata_base_strategy_preserved(self, ctx):
        """metadata['base_strategy'] must reflect the underlying chunker's strategy.

        When a caller wraps e.g. StructuralChunking inside
        ContextualChunking, the original strategy name must be kept so that
        provenance information is not silently discarded.
        """
        chunker = ContextualChunking(
            llm=_ContextualMockLLM(),
            base_chunker=StructuralChunking(max_tokens=512, overlap_sentences=2),
        )
        result = await chunker.chunk(SAMPLE_TEXT, ctx)
        assert len(result.chunks) >= 1
        for chunk in result.chunks:
            # base_strategy must be the value that StructuralChunking wrote
            assert "base_strategy" in chunk.metadata
            assert chunk.metadata["strategy"] == "contextual_chunking"

    def test_raises_on_conflicting_kwargs(self):
        """Passing shorthand kwargs alongside base_chunker must raise TypeError.

        The parameters max_tokens, overlap_sentences, and encoding_name are
        only meaningful when ContextualChunking builds the default base chunker.
        Accepting them silently when base_chunker is supplied would mislead
        callers into believing those values take effect.
        """
        with pytest.raises(TypeError, match="cannot be used together with 'base_chunker'"):
            ContextualChunking(
                llm=_ContextualMockLLM(),
                base_chunker=StructuralChunking(max_tokens=256),
                max_tokens=512,  # dead — should blow up
            )

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
        result = await chunker.chunk("Part one.\n\nPart two.", ctx)
        assert len(result.chunks) == 2


class TestStructuralChunking:
    async def test_chunk_with_elements(self, ctx):
        from graphrag_sdk.core.models import DocumentElement, DocumentInfo, DocumentOutput
        from graphrag_sdk.ingestion.chunking_strategies.structural_chunking import StructuralChunking

        # Create elements
        elements = [
            DocumentElement(type="header", content="# Title", breadcrumbs=["Title"]),
            DocumentElement(type="paragraph", content="Intro.", breadcrumbs=["Title"]),
            DocumentElement(type="paragraph", content="P2.", breadcrumbs=["Title"]),
        ]
        doc = DocumentOutput(
            text="# Title\n\nIntro.\n\nP2.",
            document_info=DocumentInfo(path="test.md"),
            elements=elements,
        )

        # max_tokens=100 ensures they group together
        chunker = StructuralChunking(max_tokens=100)
        result = await chunker.chunk_document(doc, ctx)

        assert len(result.chunks) == 1
        text = result.chunks[0].text
        # Header text has prefix but since it's a header we avoided duplicating.
        # Actually in our code, if prefix is present but it's not a header it gets added.
        assert "Title\nIntro." in result.chunks[0].text
        assert "Title\nP2." in result.chunks[0].text

    async def test_fallback_when_oversized(self, ctx):
        from graphrag_sdk.core.models import DocumentElement, DocumentInfo, DocumentOutput
        from graphrag_sdk.ingestion.chunking_strategies.structural_chunking import StructuralChunking

        # A very large paragraph
        large_content = "Word. " * 50
        elements = [
            DocumentElement(type="paragraph", content=large_content, breadcrumbs=["H1"])
        ]
        doc = DocumentOutput(
            text=large_content,
            document_info=DocumentInfo(path="test.md"),
            elements=elements,
        )

        # Cap tokens to a very small number to force fallback
        chunker = StructuralChunking(max_tokens=10)
        result = await chunker.chunk_document(doc, ctx)

        assert len(result.chunks) > 1
        total_parts = len(result.chunks)
        for i, chunk in enumerate(result.chunks, start=1):
            assert f"H1 [Part {i}/{total_parts}]" in chunk.text

    async def test_no_elements_fallback(self, ctx):
        from graphrag_sdk.core.models import DocumentInfo, DocumentOutput
        from graphrag_sdk.ingestion.chunking_strategies.structural_chunking import StructuralChunking

        doc = DocumentOutput(
            text="Just some text. More text.",
            document_info=DocumentInfo(path="test.txt"),
            elements=None,
        )

        chunker = StructuralChunking(max_tokens=50)
        result = await chunker.chunk_document(doc, ctx)

        assert len(result.chunks) == 1
        assert result.chunks[0].text == "Just some text. More text."

    async def test_metadata_breadcrumbs_in_normal_chunk(self, ctx):
        """Buffered chunks must carry the union of their elements' breadcrumbs.

        The lexical-graph step spreads chunk.metadata onto Chunk node properties,
        so breadcrumbs become graph-queryable without any extra pipeline work.
        """
        from graphrag_sdk.core.models import DocumentElement, DocumentInfo, DocumentOutput
        from graphrag_sdk.ingestion.chunking_strategies.structural_chunking import StructuralChunking

        elements = [
            DocumentElement(type="header",    content="Title",   breadcrumbs=["Title"]),
            DocumentElement(type="paragraph", content="Intro.",  breadcrumbs=["Title"]),
            DocumentElement(type="paragraph", content="Detail.", breadcrumbs=["Title", "Sub"]),
        ]
        doc = DocumentOutput(
            text="Title\n\nIntro.\n\nDetail.",
            document_info=DocumentInfo(path="test.md"),
            elements=elements,
        )

        chunker = StructuralChunking(max_tokens=200)
        result = await chunker.chunk_document(doc, ctx)

        # All three elements fit in a single buffer → one chunk
        assert len(result.chunks) == 1
        meta = result.chunks[0].metadata
        assert "breadcrumbs" in meta
        # Must contain every unique breadcrumb seen across all elements in the chunk
        assert "Title" in meta["breadcrumbs"]
        assert "Sub" in meta["breadcrumbs"]

    async def test_metadata_breadcrumbs_in_oversized_fallback(self, ctx):
        """Fallback chunks (oversized elements) must also carry breadcrumbs."""
        from graphrag_sdk.core.models import DocumentElement, DocumentInfo, DocumentOutput
        from graphrag_sdk.ingestion.chunking_strategies.structural_chunking import StructuralChunking

        large_content = "Word. " * 50
        elements = [
            DocumentElement(
                type="paragraph",
                content=large_content,
                breadcrumbs=["Chapter", "Section"],
            )
        ]
        doc = DocumentOutput(
            text=large_content,
            document_info=DocumentInfo(path="test.md"),
            elements=elements,
        )

        chunker = StructuralChunking(max_tokens=10)
        result = await chunker.chunk_document(doc, ctx)

        assert len(result.chunks) > 1
        for chunk in result.chunks:
            assert "breadcrumbs" in chunk.metadata
            assert chunk.metadata["breadcrumbs"] == ["Chapter", "Section"]

