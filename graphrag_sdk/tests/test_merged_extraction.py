"""Tests for MergedExtraction strategy."""
from __future__ import annotations

from typing import Any

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    ExtractedEntity,
    GraphSchema,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.ingestion.extraction_strategies.merged_extraction import (
    MergedExtraction,
    _normalize_type_label,
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

    def test_type_qualified_id(self):
        """Entity ID should include type suffix when entity_type is provided."""
        assert compute_entity_id("Paris", "Location") == "paris__location"
        assert compute_entity_id("Paris", "Person") == "paris__person"

    def test_cross_type_collision_prevented(self):
        """Same name with different types should produce different IDs."""
        id_person = compute_entity_id("Paris", "Person")
        id_location = compute_entity_id("Paris", "Location")
        assert id_person != id_location

    def test_no_type_backwards_compatible(self):
        """Without entity_type, should return just the normalized name."""
        assert compute_entity_id("Alice") == "alice"
        assert compute_entity_id("Alice", "") == "alice"


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
        # All relationships use the single RELATES edge type
        types = {r.type for r in result.relationships}
        assert "RELATES" in types
        # Original relationship type preserved in rel_type property
        rel_types = {r.properties.get("rel_type") for r in result.relationships}
        assert "WORKS_AT" in rel_types

    async def test_entities_have_descriptions(self, extractor, schema, ctx):
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, schema, ctx)

        for node in result.nodes:
            assert "description" in node.properties
            assert len(node.properties["description"]) > 0

    async def test_relationships_have_src_tgt_names(self, extractor, schema, ctx):
        """Relationships should carry src_name/tgt_name for edge embedding."""
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, schema, ctx)

        assert len(result.relationships) > 0
        for rel in result.relationships:
            assert "src_name" in rel.properties
            assert "tgt_name" in rel.properties
            assert rel.properties["src_name"]
            assert rel.properties["tgt_name"]

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
    async def test_max_concurrency_passed_through(self, ctx):
        """Verify max_concurrency parameter is stored for abatch_invoke."""
        llm = MockLLMWithMergedExtraction()
        extractor = MergedExtraction(llm=llm, max_concurrency=2)
        assert extractor._max_concurrency == 2

        schema = GraphSchema()
        chunks = _make_chunks("Text 1", "Text 2", "Text 3")
        result = await extractor.extract(chunks, schema, ctx)

        # All chunks should still be processed (just rate-limited)
        assert len(result.nodes) > 0


# ── Type Taxonomy Resolution Tests ────────────────────────────


class _ClusterableEmbedder(MockEmbedder):
    """Mock embedder that returns controllable vectors for clustering tests.

    Types containing the same root word get nearly identical vectors.
    Types with different roots get orthogonal vectors.
    """

    _ROOTS = {
        "function": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "method": [0.98, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # very similar to function
        "constant": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "parameter": [0.05, 0.98, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # very similar to constant
        "person": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "company": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "descriptor": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "option": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "setting": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    }

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        self.call_count += 1
        key = text.strip().lower()
        for root, vec in self._ROOTS.items():
            if root in key:
                return list(vec)
        # Fallback: hash-based for unknown types
        return super().embed_query(text, **kwargs)


class TestTypeTaxonomyResolution:
    """Tests for the _resolve_type_taxonomy() method."""

    async def test_surface_normalization(self, ctx):
        """Entities with types 'Data Type', 'data_type', 'DataType' collapse to one."""
        entities = [
            ExtractedEntity(name="X", type="Data Type", description="desc1", source_chunk_ids=["c0"]),
            ExtractedEntity(name="Y", type="data_type", description="desc2", source_chunk_ids=["c1"]),
            ExtractedEntity(name="Z", type="DataType", description="desc3", source_chunk_ids=["c2"]),
            ExtractedEntity(name="W", type="Data Type", description="desc4", source_chunk_ids=["c3"]),
        ]
        # No embedder: surface-only
        extractor = MergedExtraction(llm=MockLLM(), embedder=None)
        result = await extractor._resolve_type_taxonomy(entities, ctx)

        types = {ent.type for ent in result}
        assert len(types) == 1, f"Expected 1 canonical type, got {types}"
        # Should pick "Data Type" (most frequent: 2 occurrences)
        assert types == {"Data Type"}

    async def test_embedding_clustering(self, ctx):
        """Semantically similar types (Function/Method) merge above threshold."""
        entities = [
            ExtractedEntity(name="foo", type="Function", description="", source_chunk_ids=["c0"]),
            ExtractedEntity(name="bar", type="Function", description="", source_chunk_ids=["c1"]),
            ExtractedEntity(name="baz", type="Method", description="", source_chunk_ids=["c2"]),
        ]
        embedder = _ClusterableEmbedder(dimension=8)
        extractor = MergedExtraction(
            llm=MockLLM(), embedder=embedder, type_resolution_threshold=0.85
        )
        result = await extractor._resolve_type_taxonomy(entities, ctx)

        types = {ent.type for ent in result}
        assert len(types) == 1, f"Expected 1 type after clustering, got {types}"
        # Function has 2 occurrences vs Method's 1
        assert types == {"Function"}

    async def test_embedding_keeps_distant_types(self, ctx):
        """Types that are not similar (Person/Company) stay separate."""
        entities = [
            ExtractedEntity(name="Alice", type="Person", description="", source_chunk_ids=["c0"]),
            ExtractedEntity(name="Acme", type="Company", description="", source_chunk_ids=["c1"]),
        ]
        embedder = _ClusterableEmbedder(dimension=8)
        extractor = MergedExtraction(
            llm=MockLLM(), embedder=embedder, type_resolution_threshold=0.85
        )
        result = await extractor._resolve_type_taxonomy(entities, ctx)

        types = {ent.type for ent in result}
        assert types == {"Person", "Company"}

    async def test_no_embedder_fallback(self, ctx):
        """Without embedder, only surface normalization runs."""
        entities = [
            ExtractedEntity(name="foo", type="Function", description="", source_chunk_ids=["c0"]),
            ExtractedEntity(name="bar", type="Method", description="", source_chunk_ids=["c1"]),
            ExtractedEntity(name="baz", type="data_type", description="", source_chunk_ids=["c2"]),
            ExtractedEntity(name="qux", type="Data Type", description="", source_chunk_ids=["c3"]),
        ]
        extractor = MergedExtraction(llm=MockLLM(), embedder=None)
        result = await extractor._resolve_type_taxonomy(entities, ctx)

        types = {ent.type for ent in result}
        # Surface merges data_type/Data Type but leaves Function/Method separate
        assert "Function" in types
        assert "Method" in types
        assert len([t for t in types if _normalize_type_label(t) == "datatype"]) == 1

    async def test_skip_on_single_type(self, ctx):
        """Method returns immediately with no changes for single type."""
        entities = [
            ExtractedEntity(name="A", type="Person", description="desc", source_chunk_ids=["c0"]),
            ExtractedEntity(name="B", type="Person", description="desc", source_chunk_ids=["c1"]),
        ]
        extractor = MergedExtraction(llm=MockLLM(), embedder=MockEmbedder())
        result = await extractor._resolve_type_taxonomy(entities, ctx)

        assert len(result) == 2
        assert all(ent.type == "Person" for ent in result)

    async def test_preserves_entity_data(self, ctx):
        """Names and descriptions are untouched after type resolution."""
        entities = [
            ExtractedEntity(name="GrB_TRAN", type="Descriptor", description="A transpose descriptor", source_chunk_ids=["c0"]),
            ExtractedEntity(name="GrB_TRAN", type="descriptor", description="Transpose option", source_chunk_ids=["c1"]),
        ]
        extractor = MergedExtraction(llm=MockLLM(), embedder=None)
        result = await extractor._resolve_type_taxonomy(entities, ctx)

        names = [ent.name for ent in result]
        descs = [ent.description for ent in result]
        assert names == ["GrB_TRAN", "GrB_TRAN"]
        assert "A transpose descriptor" in descs
        assert "Transpose option" in descs

    async def test_name_consolidation(self, ctx):
        """Entities with same name but different types all get the dominant type."""
        entities = [
            ExtractedEntity(name="GrB_TRAN", type="Setting", description="desc1", source_chunk_ids=["c0"]),
            ExtractedEntity(name="GrB_TRAN", type="Setting", description="desc2", source_chunk_ids=["c1"]),
            ExtractedEntity(name="GrB_TRAN", type="Operator", description="desc3", source_chunk_ids=["c2"]),
            ExtractedEntity(name="GrB_TRAN", type="Constant", description="desc4", source_chunk_ids=["c3"]),
        ]
        extractor = MergedExtraction(llm=MockLLM(), embedder=None, consolidate_by_name=True)
        result = await extractor._resolve_type_taxonomy(entities, ctx)

        types = {ent.type for ent in result}
        assert types == {"Setting"}, f"Expected all types to be 'Setting', got {types}"

    async def test_name_consolidation_disabled(self, ctx):
        """With consolidate_by_name=False, types stay separate."""
        entities = [
            ExtractedEntity(name="GrB_TRAN", type="Setting", description="desc1", source_chunk_ids=["c0"]),
            ExtractedEntity(name="GrB_TRAN", type="Setting", description="desc2", source_chunk_ids=["c1"]),
            ExtractedEntity(name="GrB_TRAN", type="Operator", description="desc3", source_chunk_ids=["c2"]),
        ]
        extractor = MergedExtraction(llm=MockLLM(), embedder=None, consolidate_by_name=False)
        result = await extractor._resolve_type_taxonomy(entities, ctx)

        types = {ent.type for ent in result}
        assert types == {"Setting", "Operator"}, f"Expected types unchanged, got {types}"

    async def test_name_consolidation_picks_most_frequent(self, ctx):
        """Verifies frequency-based selection: 3x Setting > 2x Operator > 1x Constant."""
        entities = [
            ExtractedEntity(name="GrB_TRAN", type="Setting", description="", source_chunk_ids=["c0"]),
            ExtractedEntity(name="GrB_TRAN", type="Setting", description="", source_chunk_ids=["c1"]),
            ExtractedEntity(name="GrB_TRAN", type="Setting", description="", source_chunk_ids=["c2"]),
            ExtractedEntity(name="GrB_TRAN", type="Operator", description="", source_chunk_ids=["c3"]),
            ExtractedEntity(name="GrB_TRAN", type="Operator", description="", source_chunk_ids=["c4"]),
            ExtractedEntity(name="GrB_TRAN", type="Constant", description="", source_chunk_ids=["c5"]),
        ]
        extractor = MergedExtraction(llm=MockLLM(), embedder=None, consolidate_by_name=True)
        result = await extractor._resolve_type_taxonomy(entities, ctx)

        types = {ent.type for ent in result}
        assert types == {"Setting"}, f"Expected dominant type 'Setting', got {types}"
        assert all(ent.type == "Setting" for ent in result)

    async def test_full_pipeline_name_consolidation(self, ctx):
        """End-to-end: two chunks produce same entity with different types → one node."""
        resp_chunk0 = '("entity"<|#|>GrB_TRAN<|#|>Setting<|#|>A transpose setting)##'
        resp_chunk1 = '("entity"<|#|>GrB_TRAN<|#|>Operator<|#|>A transpose operator)##'
        llm = MockLLM(responses=[resp_chunk0, resp_chunk1])
        extractor = MergedExtraction(llm=llm, embedder=None, consolidate_by_name=True)

        schema = GraphSchema()
        chunks = _make_chunks("GrB_TRAN is a setting.", "GrB_TRAN is an operator.")
        result = await extractor.extract(chunks, schema, ctx)

        # Should produce exactly one node for GrB_TRAN
        grb_nodes = [n for n in result.nodes if "grb_tran" in n.id.lower()]
        assert len(grb_nodes) == 1, f"Expected 1 GrB_TRAN node, got {len(grb_nodes)}: {[n.id for n in grb_nodes]}"
        # The dominant type is Setting (1 vs 1, tie-break: alphabetical S > O... actually let's check)
        # Both have freq 1, tie-break: Title Case (both are), then alphabetical: "Operator" < "Setting"
        # So "Setting" wins because S > O alphabetically
        assert grb_nodes[0].label == "Setting"
        # Both chunk sources should be accumulated
        assert "chunk-0" in grb_nodes[0].properties["source_chunk_ids"]
        assert "chunk-1" in grb_nodes[0].properties["source_chunk_ids"]

    async def test_mention_ids_use_canonical_types(self, ctx):
        """After full extract(), mentions for type-inconsistent entities share the canonical entity_id."""
        # Two chunks: both mention GrB_TRAN but with different types
        resp_chunk0 = '("entity"<|#|>GrB_TRAN<|#|>Descriptor<|#|>A transpose descriptor)##'
        resp_chunk1 = '("entity"<|#|>GrB_TRAN<|#|>descriptor<|#|>Transpose option)##'
        llm = MockLLM(responses=[resp_chunk0, resp_chunk1])
        extractor = MergedExtraction(llm=llm, embedder=None)

        schema = GraphSchema()
        chunks = _make_chunks("GrB_TRAN is a descriptor.", "GrB_TRAN is a descriptor option.")
        result = await extractor.extract(chunks, schema, ctx)

        # Both mentions should reference the same entity_id
        mention_ids = {m.entity_id for m in result.mentions}
        assert len(mention_ids) == 1, f"Expected 1 unique entity_id, got {mention_ids}"
