"""Unit tests for SentenceTokenCapChunking strategy."""
from __future__ import annotations

import pytest

from graphrag_sdk.ingestion.chunking_strategies.sentence_token_cap import SentenceTokenCapChunking

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
