"""Tests for ingestion/chunking_strategies/fixed_size.py."""
from __future__ import annotations

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking


class TestFixedSizeChunking:
    async def test_basic_chunking(self, ctx):
        chunker = FixedSizeChunking(chunk_size=20, chunk_overlap=5)
        text = "abcdefghij" * 5  # 50 chars
        result = await chunker.chunk(text, ctx)
        assert len(result.chunks) > 0
        # First chunk should be 20 chars
        assert len(result.chunks[0].text) == 20

    async def test_chunk_indices_sequential(self, ctx):
        chunker = FixedSizeChunking(chunk_size=10, chunk_overlap=0)
        text = "a" * 50
        result = await chunker.chunk(text, ctx)
        for i, chunk in enumerate(result.chunks):
            assert chunk.index == i

    async def test_chunk_metadata(self, ctx):
        chunker = FixedSizeChunking(chunk_size=10, chunk_overlap=2)
        text = "Hello World, this is a test."
        result = await chunker.chunk(text, ctx)
        first = result.chunks[0]
        assert "start_char" in first.metadata
        assert "end_char" in first.metadata
        assert first.metadata["start_char"] == 0
        assert first.metadata["end_char"] == 10
        assert first.metadata["chunk_size"] == 10
        assert first.metadata["chunk_overlap"] == 2

    async def test_no_empty_chunks(self, ctx):
        chunker = FixedSizeChunking(chunk_size=10, chunk_overlap=0)
        text = "Hello     "  # trailing whitespace
        result = await chunker.chunk(text, ctx)
        for chunk in result.chunks:
            assert chunk.text.strip() != ""

    async def test_overlap_creates_more_chunks(self, ctx):
        text = "a" * 100
        no_overlap = await FixedSizeChunking(chunk_size=20, chunk_overlap=0).chunk(text, ctx)
        with_overlap = await FixedSizeChunking(chunk_size=20, chunk_overlap=10).chunk(text, ctx)
        assert len(with_overlap.chunks) > len(no_overlap.chunks)

    async def test_text_shorter_than_chunk_size(self, ctx):
        chunker = FixedSizeChunking(chunk_size=1000, chunk_overlap=100)
        text = "Short text."
        result = await chunker.chunk(text, ctx)
        assert len(result.chunks) == 1
        assert result.chunks[0].text == "Short text."

    async def test_empty_text(self, ctx):
        chunker = FixedSizeChunking(chunk_size=10, chunk_overlap=0)
        result = await chunker.chunk("", ctx)
        assert len(result.chunks) == 0

    async def test_whitespace_only_text(self, ctx):
        chunker = FixedSizeChunking(chunk_size=10, chunk_overlap=0)
        result = await chunker.chunk("     ", ctx)
        assert len(result.chunks) == 0

    def test_overlap_must_be_smaller(self):
        with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
            FixedSizeChunking(chunk_size=10, chunk_overlap=10)

    def test_overlap_exceeds_size(self):
        with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
            FixedSizeChunking(chunk_size=10, chunk_overlap=15)

    async def test_unique_uids(self, ctx):
        chunker = FixedSizeChunking(chunk_size=10, chunk_overlap=0)
        text = "a" * 50
        result = await chunker.chunk(text, ctx)
        uids = [c.uid for c in result.chunks]
        assert len(uids) == len(set(uids))

    async def test_coverage(self, ctx):
        """All text covered by chunks (with overlap)."""
        text = "0123456789" * 10  # 100 chars
        chunker = FixedSizeChunking(chunk_size=30, chunk_overlap=5)
        result = await chunker.chunk(text, ctx)
        # First chunk starts at 0, last chunk covers the end
        assert result.chunks[0].metadata["start_char"] == 0
        last = result.chunks[-1]
        assert last.metadata["end_char"] == len(text)
