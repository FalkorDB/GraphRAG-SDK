"""Tests for ingestion/extraction_strategies/schema_guided.py."""
from __future__ import annotations

import json

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import ExtractionError
from graphrag_sdk.core.models import (
    EntityType,
    GraphSchema,
    RelationType,
    SchemaPattern,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.ingestion.extraction_strategies.schema_guided import SchemaGuidedExtraction

from .conftest import MockLLM, MockLLMWithExtraction


class TestSchemaGuidedExtraction:
    async def test_extraction_basic(self, ctx, sample_schema, sample_chunks):
        llm = MockLLMWithExtraction()
        extractor = SchemaGuidedExtraction(llm=llm)
        result = await extractor.extract(sample_chunks, sample_schema, ctx)
        # Should process all 3 chunks — MockLLM returns same response each time
        assert len(result.nodes) > 0
        assert len(result.relationships) > 0

    async def test_extraction_adds_source_chunk_id(self, ctx, sample_schema):
        llm = MockLLMWithExtraction()
        extractor = SchemaGuidedExtraction(llm=llm)
        chunks = TextChunks(chunks=[
            TextChunk(text="Alice works at Acme.", index=0, uid="test-chunk-123"),
        ])
        result = await extractor.extract(chunks, sample_schema, ctx)
        for node in result.nodes:
            assert node.properties.get("source_chunk_id") == "test-chunk-123"
        for rel in result.relationships:
            assert rel.properties.get("source_chunk_id") == "test-chunk-123"

    async def test_extraction_empty_chunks(self, ctx, sample_schema):
        llm = MockLLM()
        extractor = SchemaGuidedExtraction(llm=llm)
        result = await extractor.extract(TextChunks(), sample_schema, ctx)
        assert len(result.nodes) == 0
        assert len(result.relationships) == 0

    async def test_extraction_invalid_json_skipped(self, ctx, sample_schema):
        """When LLM returns invalid JSON, chunk is skipped (logged)."""
        llm = MockLLM(responses=["this is not json at all"])
        extractor = SchemaGuidedExtraction(llm=llm)
        chunks = TextChunks(chunks=[
            TextChunk(text="Some text", index=0, uid="chunk-0"),
        ])
        # Should not raise, just skip the chunk
        result = await extractor.extract(chunks, sample_schema, ctx)
        assert len(result.nodes) == 0

    async def test_extraction_with_empty_schema(self, ctx, sample_chunks):
        """Open schema mode — no constraints."""
        llm = MockLLMWithExtraction()
        extractor = SchemaGuidedExtraction(llm=llm)
        result = await extractor.extract(sample_chunks, GraphSchema(), ctx)
        assert len(result.nodes) > 0

    async def test_extraction_respects_budget(self, ctx, sample_schema):
        """Should stop when latency budget is exceeded."""
        ctx_tight = Context(latency_budget_ms=0.0)
        import time
        time.sleep(0.001)  # ensure budget is exceeded

        llm = MockLLMWithExtraction()
        extractor = SchemaGuidedExtraction(llm=llm)
        chunks = TextChunks(chunks=[
            TextChunk(text=f"Chunk {i}", index=i, uid=f"c{i}") for i in range(10)
        ])
        result = await extractor.extract(chunks, sample_schema, ctx_tight)
        # With 0ms budget, should stop early (may process 0 chunks)
        assert len(result.nodes) < 30  # fewer than 10 * 3 nodes

    async def test_parse_response_valid(self, ctx, sample_schema):
        extractor = SchemaGuidedExtraction(llm=MockLLM())
        data = {
            "nodes": [{"id": "x", "label": "Thing", "properties": {"name": "X"}}],
            "relationships": [
                {
                    "start_node_id": "x",
                    "end_node_id": "y",
                    "type": "RELATES",
                    "properties": {},
                }
            ],
        }
        result = extractor._parse_response(json.dumps(data), "chunk-src")
        assert len(result.nodes) == 1
        assert result.nodes[0].properties["source_chunk_id"] == "chunk-src"
        assert len(result.relationships) == 1

    def test_parse_response_invalid_json(self):
        extractor = SchemaGuidedExtraction(llm=MockLLM())
        with pytest.raises(ExtractionError, match="invalid JSON"):
            extractor._parse_response("not json", "chunk-0")
