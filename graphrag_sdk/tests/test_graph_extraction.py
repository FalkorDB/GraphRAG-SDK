"""Tests for GraphExtraction strategy."""

from __future__ import annotations

import json


import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    ExtractedEntity,
    GraphSchema,
    EntityType,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
    EntityExtractor,
    LLMExtractor,
)
from graphrag_sdk.ingestion.extraction_strategies.graph_extraction import (
    GraphExtraction,
    VERIFY_EXTRACT_RELS_PROMPT,
    _format_entity_types,
)

from .conftest import MockLLM, MockLLMWithGraphExtraction


# ── Helpers ────────────────────────────────────────────────────


def _make_chunks(*texts: str) -> TextChunks:
    return TextChunks(
        chunks=[TextChunk(text=t, index=i, uid=f"chunk-{i}") for i, t in enumerate(texts)]
    )


def _mock_hybrid_llm(
    step1_entities: list[dict] | None = None,
    step2_entities: list[dict] | None = None,
    step2_relationships: list[dict] | None = None,
) -> MockLLM:
    """Create a MockLLM with step-1 NER and step-2 verify+rels responses."""
    if step1_entities is None:
        step1_entities = [
            {"name": "Alice", "type": "Person", "description": "An engineer"},
            {"name": "Acme Corp", "type": "Organization", "description": "A company"},
        ]
    if step2_entities is None:
        step2_entities = step1_entities
    if step2_relationships is None:
        step2_relationships = [
            {"source": "Alice", "target": "Acme Corp", "type": "WORKS_AT",
             "description": "Alice is employed as an engineer at Acme Corp",
             "keywords": "employment, engineering",
             "weight": 0.9},
        ]

    step1 = json.dumps(step1_entities)
    step2 = json.dumps({"entities": step2_entities, "relationships": step2_relationships})
    return MockLLM(responses=[step1, step2])


# ── Tests ──────────────────────────────────────────────────────


class TestGraphExtractionSmoke:
    @pytest.fixture
    def extractor(self):
        llm = _mock_hybrid_llm()
        return GraphExtraction(llm=llm, entity_extractor=LLMExtractor(llm))

    async def test_produces_nodes(self, extractor, ctx):
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, GraphSchema(), ctx)
        assert len(result.nodes) > 0
        names = {n.properties.get("name") for n in result.nodes}
        assert "Alice" in names

    async def test_produces_relationships(self, extractor, ctx):
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, GraphSchema(), ctx)
        assert len(result.relationships) > 0
        types = {r.type for r in result.relationships}
        assert "RELATES" in types

    async def test_relationships_have_all_retrieval_properties(self, extractor, ctx):
        """Relationships must have all properties needed by retrieval."""
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, GraphSchema(), ctx)
        for rel in result.relationships:
            # Used by relationship vector search (Phase 0)
            assert "fact" in rel.properties
            assert rel.properties["fact"]  # non-empty
            assert "src_name" in rel.properties
            assert "tgt_name" in rel.properties
            # Used by PPR expansion (Phase 2)
            assert "rel_type" in rel.properties

    async def test_produces_mentions(self, extractor, ctx):
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, GraphSchema(), ctx)
        assert len(result.mentions) > 0
        for mention in result.mentions:
            assert mention.chunk_id == "chunk-0"
            assert mention.entity_id

    async def test_entities_have_descriptions(self, extractor, ctx):
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, GraphSchema(), ctx)
        for node in result.nodes:
            assert "description" in node.properties


class TestGraphExtractionWithMock:
    """Tests using the MockLLMWithGraphExtraction from conftest."""

    async def test_mock_llm_produces_output(self, ctx):
        llm = MockLLMWithGraphExtraction()
        extractor = GraphExtraction(llm=llm, entity_extractor=LLMExtractor(llm))
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, GraphSchema(), ctx)
        assert len(result.nodes) >= 2
        assert len(result.relationships) >= 1


class TestGraphExtractionSchemaTypes:
    async def test_schema_entity_types_used(self, ctx):
        """Entity types from schema override defaults."""
        captured_prompts: list[str] = []

        class CaptureLLM(MockLLM):
            def invoke(self, prompt, **kwargs):
                captured_prompts.append(prompt)
                return super().invoke(prompt, **kwargs)

        llm = CaptureLLM(responses=[
            json.dumps([{"name": "Test", "type": "Vehicle", "description": "A car"}]),
            json.dumps({"entities": [{"name": "Test", "type": "Vehicle", "description": "A car"}],
                        "relationships": []}),
        ])
        schema = GraphSchema(
            entities=[
                EntityType(label="Vehicle", description="A vehicle"),
                EntityType(label="Road", description="A road"),
            ],
        )
        extractor = GraphExtraction(llm=llm, entity_extractor=LLMExtractor(llm))
        chunks = _make_chunks("The car drove down the highway.")
        await extractor.extract(chunks, schema, ctx)

        # Prompts should contain schema entity types
        assert len(captured_prompts) > 0
        assert "Vehicle" in captured_prompts[0]
        assert "Road" in captured_prompts[0]


class TestGraphExtractionBudget:
    async def test_budget_exceeded_stops_extraction(self):
        llm = _mock_hybrid_llm()
        extractor = GraphExtraction(llm=llm, entity_extractor=LLMExtractor(llm))
        chunks = _make_chunks("Text 1", "Text 2", "Text 3")

        ctx = Context(tenant_id="test", latency_budget_ms=0.0)
        ctx._start_time = ctx._start_time - 1.0

        result = await extractor.extract(chunks, GraphSchema(), ctx)
        assert llm._call_index == 0 or len(result.nodes) == 0


class TestGraphExtractionAggregation:
    async def test_cross_chunk_entity_dedup(self, ctx):
        """Same entity from multiple chunks should be deduplicated."""
        step1 = json.dumps([{"name": "Alice", "type": "Person", "description": "An engineer"}])
        step2 = json.dumps({
            "entities": [{"name": "Alice", "type": "Person", "description": "An engineer"}],
            "relationships": [],
        })
        llm = MockLLM(responses=[step1, step1, step2, step2])
        extractor = GraphExtraction(llm=llm, entity_extractor=LLMExtractor(llm))

        chunks = _make_chunks("Alice is an engineer.", "Alice works hard.")
        result = await extractor.extract(chunks, GraphSchema(), ctx)

        alice_nodes = [n for n in result.nodes if n.properties.get("name") == "Alice"]
        assert len(alice_nodes) == 1
        assert "chunk-0" in alice_nodes[0].properties["source_chunk_ids"]
        assert "chunk-1" in alice_nodes[0].properties["source_chunk_ids"]


class TestGraphExtractionGraphOutput:
    async def test_node_ids_are_type_qualified(self, ctx):
        """Node IDs should include entity type for collision prevention."""
        llm = _mock_hybrid_llm()
        extractor = GraphExtraction(llm=llm, entity_extractor=LLMExtractor(llm))
        chunks = _make_chunks("Alice works at Acme Corp.")
        result = await extractor.extract(chunks, GraphSchema(), ctx)

        ids = {n.id for n in result.nodes}
        # IDs should contain __ separator for type qualification
        for nid in ids:
            assert "__" in nid

    async def test_relationship_edge_type_is_relates(self, ctx):
        """All relationships should use RELATES edge type."""
        llm = _mock_hybrid_llm()
        extractor = GraphExtraction(llm=llm, entity_extractor=LLMExtractor(llm))
        chunks = _make_chunks("Alice works at Acme Corp.")
        result = await extractor.extract(chunks, GraphSchema(), ctx)

        for rel in result.relationships:
            assert rel.type == "RELATES"
            assert rel.properties["rel_type"] == "WORKS_AT"


class TestGraphExtractionPluggableExtractor:
    async def test_custom_ner_model(self, ctx):
        """GraphExtraction works with a custom NER model."""

        class SimpleNERExtractor(EntityExtractor):
            async def extract_entities(self, text, entity_types, source_chunk_id):
                from graphrag_sdk.core.models import ExtractedEntity
                return [
                    ExtractedEntity(name="Alice", type="Person", description="",
                                    source_chunk_ids=[source_chunk_id]),
                    ExtractedEntity(name="Bob", type="Person", description="",
                                    source_chunk_ids=[source_chunk_id]),
                ]

        step2 = json.dumps({
            "entities": [
                {"name": "Alice", "type": "Person", "description": ""},
                {"name": "Bob", "type": "Person", "description": ""},
            ],
            "relationships": [
                {"source": "Alice", "target": "Bob", "type": "KNOWS",
                 "description": "They know each other", "weight": 0.8},
            ],
        })
        llm = MockLLM(responses=[step2])
        custom_extractor = SimpleNERExtractor()
        extractor = GraphExtraction(
            llm=llm,
            entity_extractor=custom_extractor,
        )

        chunks = _make_chunks("Alice and Bob are friends.")
        result = await extractor.extract(chunks, GraphSchema(), ctx)

        names = {n.properties.get("name") for n in result.nodes}
        assert "Alice" in names
        assert "Bob" in names
        assert len(result.relationships) >= 1


class TestGraphExtractionConcurrency:
    async def test_max_concurrency_stored(self, ctx):
        llm = _mock_hybrid_llm()
        extractor = GraphExtraction(
            llm=llm, entity_extractor=LLMExtractor(llm), max_concurrency=2,
        )
        assert extractor._max_concurrency == 2

        chunks = _make_chunks("Text 1")
        result = await extractor.extract(chunks, GraphSchema(), ctx)
        assert len(result.nodes) > 0


class TestGraphExtractionDefaults:
    def test_default_entity_types(self):
        llm = MockLLM()
        extractor = GraphExtraction(llm=llm, entity_extractor=LLMExtractor(llm))
        assert "Person" in extractor.entity_types
        assert "Organization" in extractor.entity_types
        assert "Location" in extractor.entity_types

    def test_custom_entity_types(self):
        llm = MockLLM()
        extractor = GraphExtraction(
            llm=llm,
            entity_extractor=LLMExtractor(llm),
            entity_types=["Vehicle", "Road"],
        )
        assert extractor.entity_types == ["Vehicle", "Road"]


class TestGraphExtractionStep2Parsing:
    def test_parse_valid_response(self):
        content = json.dumps({
            "entities": [
                {"name": "Alice", "type": "Person", "description": "Engineer"},
            ],
            "relationships": [
                {"source": "Alice", "target": "Bob", "type": "KNOWS",
                 "description": "Friends", "weight": 0.8},
            ],
        })
        ents, rels = GraphExtraction._parse_step2_response(
            content, ["Person"], "chunk-0"
        )
        assert len(ents) == 1
        assert ents[0].name == "Alice"
        assert len(rels) == 1
        assert rels[0].source == "Alice"
        assert rels[0].target == "Bob"

    def test_parse_invalid_json(self):
        ents, rels = GraphExtraction._parse_step2_response(
            "not json", ["Person"], "chunk-0"
        )
        assert ents == []
        assert rels == []

    def test_parse_filters_stoplist_endpoints(self):
        content = json.dumps({
            "entities": [
                {"name": "Alice", "type": "Person", "description": ""},
            ],
            "relationships": [
                {"source": "he", "target": "Alice", "type": "KNOWS",
                 "description": "", "weight": 0.5},
            ],
        })
        ents, rels = GraphExtraction._parse_step2_response(
            content, ["Person"], "chunk-0"
        )
        assert len(rels) == 0  # "he" is in stoplist

    def test_parse_rejects_slash_types(self):
        content = json.dumps({
            "entities": [
                {"name": "Horse", "type": "Animal/Concept", "description": "A horse"},
                {"name": "Alice", "type": "Person", "description": "Valid"},
            ],
            "relationships": [],
        })
        ents, rels = GraphExtraction._parse_step2_response(
            content, ["Person", "Animal"], "chunk-0"
        )
        names = {e.name for e in ents}
        assert "Alice" in names
        assert "Horse" not in names

    def test_parse_markdown_fences(self):
        content = '```json\n{"entities": [{"name": "Alice", "type": "Person", "description": ""}], "relationships": []}\n```'
        ents, rels = GraphExtraction._parse_step2_response(
            content, ["Person"], "chunk-0"
        )
        assert len(ents) == 1


class TestSpansMerging:
    def test_step1_metadata_merged_into_step2(self):
        """Step 1 spans/confidence should carry over to step 2 verified entities."""
        step1 = [
            ExtractedEntity(
                name="Alice", type="Person", description="",
                source_chunk_ids=["chunk-0"],
                spans={"chunk-0": [{"start": 0, "end": 5}]},
                confidence=0.95,
            ),
            ExtractedEntity(
                name="Bob", type="Person", description="",
                source_chunk_ids=["chunk-0"],
                spans={"chunk-0": [{"start": 10, "end": 13}]},
                confidence=0.88,
            ),
        ]
        verified = [
            ExtractedEntity(
                name="Alice", type="Person",
                description="A software engineer",
                source_chunk_ids=["chunk-0"],
            ),
            ExtractedEntity(
                name="Bob", type="Person",
                description="A product manager",
                source_chunk_ids=["chunk-0"],
            ),
        ]
        chunk_text = "Alice and Bob work together at Acme Corp."
        GraphExtraction._merge_step1_metadata(verified, step1, chunk_text, "chunk-0")

        alice = next(e for e in verified if e.name == "Alice")
        assert hasattr(alice, "spans")
        assert alice.spans["chunk-0"] == [{"start": 0, "end": 5}]
        assert alice.confidence == 0.95
        # Description should be the richer step 2 version
        assert alice.description == "A software engineer"

        bob = next(e for e in verified if e.name == "Bob")
        assert bob.spans["chunk-0"] == [{"start": 10, "end": 13}]
        assert bob.confidence == 0.88

    def test_llm_discovered_entity_gets_text_find_spans(self):
        """Entities found by LLM (not in step 1) get spans via text.find()."""
        step1 = [
            ExtractedEntity(
                name="Alice", type="Person", description="",
                source_chunk_ids=["chunk-0"],
                spans={"chunk-0": [{"start": 0, "end": 5}]},
                confidence=0.95,
            ),
        ]
        # LLM discovered "Acme Corp" which GLiNER missed
        verified = [
            ExtractedEntity(
                name="Alice", type="Person",
                description="A software engineer",
                source_chunk_ids=["chunk-0"],
            ),
            ExtractedEntity(
                name="Acme Corp", type="Organization",
                description="A tech company",
                source_chunk_ids=["chunk-0"],
            ),
        ]
        chunk_text = "Alice works at Acme Corp as an engineer."
        GraphExtraction._merge_step1_metadata(verified, step1, chunk_text, "chunk-0")

        # Alice should get GLiNER spans
        alice = next(e for e in verified if e.name == "Alice")
        assert alice.spans["chunk-0"] == [{"start": 0, "end": 5}]

        # Acme Corp should get text.find() spans
        acme = next(e for e in verified if e.name == "Acme Corp")
        assert hasattr(acme, "spans")
        assert acme.spans["chunk-0"] == [{"start": 15, "end": 24}]

    def test_llm_discovered_entity_case_insensitive_spans(self):
        """text.find() spans should be case-insensitive."""
        step1: list[ExtractedEntity] = []
        verified = [
            ExtractedEntity(
                name="ACME Corp", type="Organization",
                description="A tech company",
                source_chunk_ids=["chunk-0"],
            ),
        ]
        chunk_text = "Alice works at Acme Corp as an engineer."
        GraphExtraction._merge_step1_metadata(verified, step1, chunk_text, "chunk-0")

        acme = verified[0]
        assert hasattr(acme, "spans")
        assert acme.spans["chunk-0"] == [{"start": 15, "end": 24}]

    def test_aggregate_merges_spans_across_chunks(self):
        """Spans from different chunks should merge during aggregation."""
        ent1 = ExtractedEntity(
            name="Alice", type="Person", description="",
            source_chunk_ids=["chunk-0"],
            spans={"chunk-0": [{"start": 0, "end": 5}]},
        )
        ent2 = ExtractedEntity(
            name="Alice", type="Person", description="",
            source_chunk_ids=["chunk-1"],
            spans={"chunk-1": [{"start": 10, "end": 15}]},
        )
        merged = GraphExtraction._aggregate_entities([ent1, ent2])
        assert len(merged) == 1
        assert "chunk-0" in merged[0].spans
        assert "chunk-1" in merged[0].spans
        assert merged[0].spans["chunk-0"] == [{"start": 0, "end": 5}]
        assert merged[0].spans["chunk-1"] == [{"start": 10, "end": 15}]

    def test_aggregate_without_spans_works(self):
        """Entities without spans (LLM mode) should aggregate normally."""
        ent1 = ExtractedEntity(
            name="Alice", type="Person", description="desc1",
            source_chunk_ids=["chunk-0"],
        )
        ent2 = ExtractedEntity(
            name="Alice", type="Person", description="longer description",
            source_chunk_ids=["chunk-1"],
        )
        merged = GraphExtraction._aggregate_entities([ent1, ent2])
        assert len(merged) == 1
        assert merged[0].description == "longer description"
        assert "chunk-0" in merged[0].source_chunk_ids
        assert "chunk-1" in merged[0].source_chunk_ids

    def test_spans_propagated_to_node_properties(self):
        """Nodes with spans should include them in properties."""
        ent = ExtractedEntity(
            name="Alice", type="Person", description="",
            source_chunk_ids=["chunk-0"],
            spans={"chunk-0": [{"start": 0, "end": 5}]},
        )
        nodes = GraphExtraction._entities_to_nodes([ent])
        assert len(nodes) == 1
        assert "spans" in nodes[0].properties
        assert nodes[0].properties["spans"]["chunk-0"] == [{"start": 0, "end": 5}]

    def test_no_spans_not_in_node_properties(self):
        """Nodes without spans should not have spans key."""
        ent = ExtractedEntity(
            name="Alice", type="Person", description="",
            source_chunk_ids=["chunk-0"],
        )
        nodes = GraphExtraction._entities_to_nodes([ent])
        assert "spans" not in nodes[0].properties

    def test_relationship_spans_parsed(self):
        """Step 2 should extract span_start/span_end for relationships."""
        content = json.dumps({
            "entities": [
                {"name": "Alice", "type": "Person", "description": ""},
                {"name": "Bob", "type": "Person", "description": ""},
            ],
            "relationships": [
                {"source": "Alice", "target": "Bob", "type": "KNOWS",
                 "description": "Alice knows Bob", "keywords": "social",
                 "weight": 0.9, "span_start": 10, "span_end": 35},
            ],
        })
        ents, rels = GraphExtraction._parse_step2_response(
            content, ["Person"], "chunk-5"
        )
        assert len(rels) == 1
        assert hasattr(rels[0], "spans")
        assert "chunk-5" in rels[0].spans
        assert rels[0].spans["chunk-5"] == [{"start": 10, "end": 35}]

    def test_relationship_spans_propagated_to_properties(self):
        """Relationship spans should appear in GraphRelationship properties."""
        from graphrag_sdk.core.models import ExtractedRelation

        rel = ExtractedRelation(
            source="Alice", target="Bob", type="KNOWS",
            keywords="social", description="Alice knows Bob",
            weight=0.9, source_chunk_ids=["chunk-0"],
            spans={"chunk-0": [{"start": 10, "end": 35}]},
        )
        rels = GraphExtraction._relations_to_relationships([rel])
        assert len(rels) == 1
        assert "spans" in rels[0].properties
        assert rels[0].properties["spans"]["chunk-0"] == [{"start": 10, "end": 35}]

    def test_relationship_without_spans_no_property(self):
        """Relationships without spans should not have spans key."""
        from graphrag_sdk.core.models import ExtractedRelation

        rel = ExtractedRelation(
            source="Alice", target="Bob", type="KNOWS",
            keywords="", description="", weight=1.0,
            source_chunk_ids=["chunk-0"],
        )
        rels = GraphExtraction._relations_to_relationships([rel])
        assert "spans" not in rels[0].properties

    def test_relationship_spans_merge_across_chunks(self):
        """Same relationship from different chunks should merge spans."""
        from graphrag_sdk.core.models import ExtractedRelation

        rel1 = ExtractedRelation(
            source="Alice", target="Bob", type="KNOWS",
            keywords="", description="short",
            weight=0.9, source_chunk_ids=["chunk-0"],
            spans={"chunk-0": [{"start": 10, "end": 35}]},
        )
        rel2 = ExtractedRelation(
            source="Alice", target="Bob", type="KNOWS",
            keywords="", description="longer description",
            weight=0.9, source_chunk_ids=["chunk-1"],
            spans={"chunk-1": [{"start": 5, "end": 40}]},
        )
        merged = GraphExtraction._aggregate_relations([rel1, rel2])
        assert len(merged) == 1
        assert "chunk-0" in merged[0].spans
        assert "chunk-1" in merged[0].spans


class TestNoiseFilteringPrompt:
    """Bug 4: VERIFY_EXTRACT_RELS_PROMPT should contain noise-filtering instructions."""

    def test_prompt_contains_operator_filtering(self):
        assert "symbolic" in VERIFY_EXTRACT_RELS_PROMPT.lower()

    def test_prompt_contains_abbreviation_filtering(self):
        assert "non-domain-specific" in VERIFY_EXTRACT_RELS_PROMPT

    def test_prompt_contains_short_token_filtering(self):
        assert "1-2 characters" in VERIFY_EXTRACT_RELS_PROMPT


class TestEntityTypeDescriptions:
    """Bug 3: _format_entity_types should include descriptions when available."""

    def test_with_descriptions(self):
        types = ["Command", "Function", "Concept"]
        descs = {"Command": "FalkorDB GRAPH.* commands", "Function": "Cypher built-in functions"}
        result = _format_entity_types(types, descs)
        assert "Command\n  Description: FalkorDB GRAPH.* commands" in result
        assert "Function\n  Description: Cypher built-in functions" in result
        assert "\n" in result  # newline-separated when descriptions present
        # Concept has no description — should appear without suffix
        assert "Concept" in result
        assert "Concept\n  Description:" not in result

    def test_without_descriptions(self):
        types = ["Person", "Organization"]
        result = _format_entity_types(types)
        assert result == "Person, Organization"

    def test_empty_descriptions_dict(self):
        types = ["Person", "Organization"]
        result = _format_entity_types(types, {})
        assert result == "Person, Organization"

    def test_partial_descriptions(self):
        types = ["Person", "Vehicle"]
        descs = {"Vehicle": "A car, truck, or other transport"}
        result = _format_entity_types(types, descs)
        assert "Vehicle\n  Description: A car, truck, or other transport" in result
        assert "Person" in result

    async def test_descriptions_reach_step2_prompt(self, ctx):
        """Schema entity type descriptions should appear in the Step 2 LLM prompt."""
        captured_prompts: list[str] = []

        class CaptureLLM(MockLLM):
            def invoke(self, prompt, **kwargs):
                captured_prompts.append(prompt)
                return super().invoke(prompt, **kwargs)

        llm = CaptureLLM(responses=[
            json.dumps([{"name": "GRAPH.QUERY", "type": "Command", "description": "A cmd"}]),
            json.dumps({"entities": [{"name": "GRAPH.QUERY", "type": "Command", "description": "A cmd"}],
                        "relationships": []}),
        ])
        schema = GraphSchema(
            entities=[
                EntityType(label="Command", description="FalkorDB GRAPH.* commands"),
                EntityType(label="Function", description="Cypher built-in functions"),
            ],
        )
        extractor = GraphExtraction(llm=llm, entity_extractor=LLMExtractor(llm))
        chunks = _make_chunks("GRAPH.QUERY is a FalkorDB command.")
        await extractor.extract(chunks, schema, ctx)

        # Step 2 prompt (second call) should contain the descriptions
        assert len(captured_prompts) >= 2
        step2_prompt = captured_prompts[1]
        assert "Command\n  Description: FalkorDB GRAPH.* commands" in step2_prompt
        assert "Function\n  Description: Cypher built-in functions" in step2_prompt
