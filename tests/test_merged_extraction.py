"""Tests for MergedExtraction strategy."""
from __future__ import annotations

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphSchema, TextChunk, TextChunks
from graphrag_sdk.ingestion.extraction_strategies.merged_extraction import (
    MergedExtraction,
    compute_entity_id,
)

from .conftest import MockEmbedder, MockLLM, MockLLMWithMergedExtraction


# ── Helpers ────────────────────────────────────────────────────


def _make_chunks(*texts: str) -> TextChunks:
    return TextChunks(
        chunks=[
            TextChunk(text=t, index=i, uid=f"chunk-{i}")
            for i, t in enumerate(texts)
        ]
    )


# ── Tests ──────────────────────────────────────────────────────


class TestComputeEntityId:
    def test_lowercase_strip(self):
        assert compute_entity_id("  Alice  ") == "alice"

    def test_spaces_to_underscores(self):
        assert compute_entity_id("Acme Corp") == "acme_corp"

    def test_already_normalised(self):
        assert compute_entity_id("bob") == "bob"


class TestMergedExtractionBasic:
    @pytest.fixture
    def extractor(self):
        llm = MockLLMWithMergedExtraction()
        return MergedExtraction(llm=llm)

    @pytest.fixture
    def schema(self, sample_schema):
        return sample_schema

    async def test_produces_entities_with_names(self, extractor, schema, ctx):
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, schema, ctx)

        assert len(result.nodes) > 0
        names = {n.properties.get("name") for n in result.nodes}
        assert "Alice" in names

    async def test_produces_relationships(self, extractor, schema, ctx):
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, schema, ctx)

        assert len(result.relationships) > 0
        types = {r.type for r in result.relationships}
        assert "WORKS_AT" in types

    async def test_entities_have_descriptions(self, extractor, schema, ctx):
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, schema, ctx)

        for node in result.nodes:
            assert "description" in node.properties
            assert len(node.properties["description"]) > 0

    async def test_facts_generated_from_relations(self, extractor, schema, ctx):
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, schema, ctx)

        facts = getattr(result, "facts", [])
        assert len(facts) > 0
        for fact in facts:
            assert fact.subject
            assert fact.predicate
            assert fact.object

    async def test_mentions_link_chunks_to_entities(self, extractor, schema, ctx):
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, schema, ctx)

        mentions = getattr(result, "mentions", [])
        assert len(mentions) > 0
        for mention in mentions:
            assert mention.chunk_id == "chunk-0"
            assert mention.entity_id

    async def test_relationships_have_keywords(self, extractor, schema, ctx):
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, schema, ctx)

        for rel in result.relationships:
            assert "keywords" in rel.properties


class TestMergedExtractionGleaning:
    async def test_gleaning_adds_entities(self, ctx):
        first_response = (
            '("entity"<|#|>Alice<|#|>Person<|#|>Software engineer)##'
        )
        gleaning_response = (
            '("entity"<|#|>Acme Corp<|#|>Company<|#|>A tech company)##'
        )
        llm = MockLLM(responses=[first_response, gleaning_response])
        extractor = MergedExtraction(llm=llm, enable_gleaning=True)

        schema = GraphSchema()
        chunks = _make_chunks("Alice works at Acme Corp.")
        result = await extractor.extract(chunks, schema, ctx)

        names = {n.properties.get("name") for n in result.nodes}
        assert "Alice" in names
        assert "Acme Corp" in names

    async def test_gleaning_disabled_single_call(self, ctx):
        llm = MockLLMWithMergedExtraction()
        extractor = MergedExtraction(llm=llm, enable_gleaning=False)

        schema = GraphSchema()
        chunks = _make_chunks("Alice works at Acme Corp.")
        await extractor.extract(chunks, schema, ctx)

        # With gleaning off, should only have 1 call per chunk
        assert llm._call_index == 1


class TestMergedExtractionSchema:
    async def test_schema_types_in_prompt(self, sample_schema, ctx):
        """Schema entity/relation types should appear in the LLM prompt."""
        captured_prompts: list[str] = []

        class CaptureLLM(MockLLM):
            def invoke(self, prompt, **kwargs):
                captured_prompts.append(prompt)
                return super().invoke(prompt, **kwargs)

        llm = CaptureLLM(responses=['("entity"<|#|>Test<|#|>Person<|#|>desc)##'])
        extractor = MergedExtraction(llm=llm)
        chunks = _make_chunks("Test text")

        await extractor.extract(chunks, sample_schema, ctx)

        assert len(captured_prompts) > 0
        prompt = captured_prompts[0]
        assert "Person" in prompt
        assert "Company" in prompt
        assert "WORKS_AT" in prompt


class TestMergedExtractionBudget:
    async def test_budget_exceeded_stops_extraction(self):
        llm = MockLLMWithMergedExtraction()
        extractor = MergedExtraction(llm=llm)
        schema = GraphSchema()
        chunks = _make_chunks("Text 1", "Text 2", "Text 3")

        # Create context with expired budget
        ctx = Context(tenant_id="test", latency_budget_ms=0.0)
        # Force elapsed time > budget
        ctx._start_time = ctx._start_time - 1.0

        result = await extractor.extract(chunks, schema, ctx)
        # Should produce no results since budget expired
        assert llm._call_index == 0 or len(result.nodes) == 0


class TestMergedExtractionAggregation:
    async def test_cross_chunk_entity_aggregation(self, ctx):
        """Same entity from multiple chunks should accumulate source_chunk_ids."""
        response = (
            '("entity"<|#|>Alice<|#|>Person<|#|>A software engineer)##'
        )
        llm = MockLLM(responses=[response, response])
        extractor = MergedExtraction(llm=llm)
        schema = GraphSchema()

        chunks = _make_chunks("Alice is an engineer.", "Alice works hard.")
        result = await extractor.extract(chunks, schema, ctx)

        # Alice should appear once after aggregation
        alice_nodes = [n for n in result.nodes if n.properties.get("name") == "Alice"]
        assert len(alice_nodes) == 1
        assert "chunk-0" in alice_nodes[0].properties["source_chunk_ids"]
        assert "chunk-1" in alice_nodes[0].properties["source_chunk_ids"]


class TestMergedExtractionConcurrency:
    async def test_semaphore_limits_concurrency(self, ctx):
        """Verify max_concurrency parameter is respected."""
        llm = MockLLMWithMergedExtraction()
        extractor = MergedExtraction(llm=llm, max_concurrency=2)
        assert extractor._semaphore._value == 2

        schema = GraphSchema()
        chunks = _make_chunks("Text 1", "Text 2", "Text 3")
        result = await extractor.extract(chunks, schema, ctx)

        # All chunks should still be processed (just rate-limited)
        assert len(result.nodes) > 0
