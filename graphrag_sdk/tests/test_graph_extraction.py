"""Tests for GraphExtraction strategy."""

from __future__ import annotations

import json

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    RESERVED_NODE_LABELS,
    Entity,
    ExtractedEntity,
    Ontology,
    Relation,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
    DEFAULT_ENTITY_TYPES,
    EntityExtractor,
    LLMExtractor,
    is_valid_entity_name,
)
from graphrag_sdk.ingestion.extraction_strategies.graph_extraction import (
    DEFAULT_RELATION_TYPES,
    VERIFY_EXTRACT_RELS_PROMPT,
    GraphExtraction,
    _format_entity_types,
    _format_relation_patterns,
    _reject_reserved_labels,
    _relationship_type_instruction,
)
from graphrag_sdk.storage.graph_store import GraphStore

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
        result = await extractor.extract(chunks, Ontology(), ctx)
        assert len(result.nodes) > 0
        names = {n.properties.get("name") for n in result.nodes}
        assert "Alice" in names

    async def test_produces_relationships(self, extractor, ctx):
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, Ontology(), ctx)
        assert len(result.relationships) > 0
        types = {r.type for r in result.relationships}
        assert "RELATES" in types

    async def test_relationships_have_all_retrieval_properties(self, extractor, ctx):
        """Relationships must have all properties needed by retrieval."""
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, Ontology(), ctx)
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
        result = await extractor.extract(chunks, Ontology(), ctx)
        assert len(result.mentions) > 0
        for mention in result.mentions:
            assert mention.chunk_id == "chunk-0"
            assert mention.entity_id

    async def test_entities_have_descriptions(self, extractor, ctx):
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, Ontology(), ctx)
        for node in result.nodes:
            assert "description" in node.properties


class TestGraphExtractionWithMock:
    """Tests using the MockLLMWithGraphExtraction from conftest."""

    async def test_mock_llm_produces_output(self, ctx):
        llm = MockLLMWithGraphExtraction()
        extractor = GraphExtraction(llm=llm, entity_extractor=LLMExtractor(llm))
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await extractor.extract(chunks, Ontology(), ctx)
        assert len(result.nodes) >= 2
        assert len(result.relationships) >= 1


class TestGraphExtractionSchemaTypes:
    async def test_schema_entity_types_used(self, ctx):
        """Entity types from ontology override defaults."""
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
        ontology = Ontology(
            entities=[
                Entity(label="Vehicle", description="A vehicle"),
                Entity(label="Road", description="A road"),
            ],
        )
        extractor = GraphExtraction(llm=llm, entity_extractor=LLMExtractor(llm))
        chunks = _make_chunks("The car drove down the highway.")
        await extractor.extract(chunks, ontology, ctx)

        # Prompts should contain ontology entity types
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

        result = await extractor.extract(chunks, Ontology(), ctx)
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
        result = await extractor.extract(chunks, Ontology(), ctx)

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
        result = await extractor.extract(chunks, Ontology(), ctx)

        ids = {n.id for n in result.nodes}
        # IDs should contain __ separator for type qualification
        for nid in ids:
            assert "__" in nid

    async def test_relationship_edge_type_is_relates(self, ctx):
        """All relationships should use RELATES edge type."""
        llm = _mock_hybrid_llm()
        extractor = GraphExtraction(llm=llm, entity_extractor=LLMExtractor(llm))
        chunks = _make_chunks("Alice works at Acme Corp.")
        result = await extractor.extract(chunks, Ontology(), ctx)

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
        result = await extractor.extract(chunks, Ontology(), ctx)

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
        result = await extractor.extract(chunks, Ontology(), ctx)
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


class TestNoiseFiltering:
    """Bug 4: operator/abbreviation/short-token noise must be filtered out.

    These rules used to live as instructions inside VERIFY_EXTRACT_RELS_PROMPT
    and were asserted by checking the prompt's wording. They now live in
    ``is_valid_entity_name`` instead, after measurement showed the LLM does not
    reliably act on verification instructions (RESULTS.md P2.10). Asserting the
    behaviour rather than the prompt text is also what these tests should have
    done in the first place: the old version passed whether or not anything was
    actually filtered.
    """

    @pytest.mark.parametrize("name", ["+=", "->", "++", "==", "!="])
    def test_operator_tokens_rejected(self, name):
        assert not is_valid_entity_name(name)

    @pytest.mark.parametrize("name", ["sh", "cd", "ls", "rm", "cp", "mv"])
    def test_shell_abbreviations_rejected(self, name):
        assert not is_valid_entity_name(name)

    @pytest.mark.parametrize("name", ["dt", "bg", "fn"])
    def test_generic_short_tokens_rejected(self, name):
        assert not is_valid_entity_name(name)

    @pytest.mark.parametrize("name", ["AI", "US", "UK", "Go", "EU", "UN", "IT"])
    def test_real_acronyms_kept(self, name):
        """The filter must not take widely-recognised acronyms with it."""
        assert is_valid_entity_name(name)

    @pytest.mark.parametrize(
        "name",
        ["CD", "LS", "RM", "PS", "CAT", "ENV", "ETC", "VAR", "BIN", "USR", "SED", "AWK"],
    )
    def test_uppercase_shell_tokens_still_rejected(self, name):
        """The acronym exemption excuses a token from the *pronoun* rows only.

        Before, `is_acronym` skipped the whole stoplist, so an all-caps heading
        or OCR'd text put `CD`/`ETC` straight into the graph and the shell-token
        filter was defeated for uppercase input.
        """
        assert not is_valid_entity_name(name)

    @pytest.mark.parametrize("name", ["ONE", "MAN", "BOY"])
    def test_uppercase_generic_nouns_still_rejected(self, name):
        assert not is_valid_entity_name(name)

    @pytest.mark.parametrize("name", ["us", "it", "he", "we"])
    def test_lowercase_pronouns_still_rejected(self, name):
        assert not is_valid_entity_name(name)

    @pytest.mark.parametrize("name", ["1823", "1957", "1003 ce", "14 january 1904"])
    def test_specific_dates_rejected_when_ontology_lacks_date_type(self, name):
        """A date pins down a moment; unless the ontology asks for Date nodes it
        is an attribute, not an entity."""
        assert not is_valid_entity_name(name, ["Person", "Location"])

    @pytest.mark.parametrize("name", ["1823", "1957", "1003 ce", "14 january 1904"])
    def test_specific_dates_kept_when_ontology_has_date(self, name):
        """An ontology that declares Date (the defaults do) keeps date nodes."""
        assert is_valid_entity_name(name, DEFAULT_ENTITY_TYPES)
        assert is_valid_entity_name(name, ["Person", "date"])

    @pytest.mark.parametrize("name", ["1823.", "(14 January 1904)", "1957,", "'1003 ce'"])
    def test_specific_dates_with_edge_punctuation_still_rejected(self, name):
        """Extraction leaves sentence punctuation on a date at a boundary;
        that must not let it past the gate as a "different" name."""
        assert not is_valid_entity_name(name, ["Person", "Location"])

    @pytest.mark.parametrize("name", ["Bronze Age", "1990s", "Victorian era", "Ming dynasty"])
    def test_periods_kept_when_ontology_lacks_date_type(self, name):
        """A span of time is a topic, not a moment, and stays an entity."""
        assert is_valid_entity_name(name, ["Person", "Location"])

    @pytest.mark.parametrize("word", ["baggage", "package", "opera", "camera"])
    def test_period_words_match_whole_words_only(self, word):
        """`age`/`era` need a leading boundary too, or any word ending in them
        is misread as a period and exempted from the date rule."""
        from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
            _DATE_PERIOD_RE,
        )

        assert _DATE_PERIOD_RE.search(word) is None

    @pytest.mark.parametrize("name", ["1823", "1984", "747"])
    def test_dates_kept_without_an_ontology(self, name):
        """No ontology, no date rule: the caller has not said dates are unwanted.

        This is what lets grounded discovery see "1984" and "747" -- it has no
        ontology yet, that is the point of discovery.
        """
        assert is_valid_entity_name(name)
        assert is_valid_entity_name(name, None)

    @pytest.mark.parametrize("name", ["1984", "747", "1969"])
    def test_ner_anchor_labels_do_not_drive_the_date_gate(self, name):
        """Grounded discovery hands NER its broad anchor labels, not an
        ontology. Those must not make the parser drop numeric-looking mentions
        before catalog linking, or `Book`/`Aircraft` never get discovered."""
        from graphrag_sdk.discovery.pipeline import _NER_ANCHOR_LABELS
        from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
            _parse_predictions,
        )

        preds = [{"text": name, "label": "event", "score": 0.9, "start": 0, "end": 4}]
        parsed = _parse_predictions(preds, _NER_ANCHOR_LABELS, "chunk-0", 0.5)
        assert [e.name for e in parsed] == [name]
        llm_parsed = LLMExtractor._parse_response(
            json.dumps([{"name": name, "type": "event"}]), _NER_ANCHOR_LABELS, "chunk-0"
        )
        assert [e.name for e in llm_parsed] == [name]

    async def test_extraction_applies_date_gate_to_step1_output(self, ctx):
        """Extractors no longer know the ontology, so GraphExtraction must
        apply the date rule itself -- including to step-1 entities that fall
        through unverified when step 2 fails."""

        class _DateExtractor(EntityExtractor):
            async def extract_entities(self, text, entity_types, source_chunk_id):
                return [
                    ExtractedEntity(name=n, type=t, source_chunk_ids=[source_chunk_id])
                    for n, t in (("Alice", "Person"), ("1823", "Date"))
                ]

        async def _names(ontology: Ontology) -> set[str]:
            # Step 2 returns garbage, so the step-1 entities are used as-is.
            strategy = GraphExtraction(
                llm=MockLLM(responses=["not json"]), entity_extractor=_DateExtractor()
            )
            result = await strategy.extract(_make_chunks("Alice was born in 1823."), ontology, ctx)
            return {n.properties["name"] for n in result.nodes}

        no_date = await _names(
            Ontology(entities=[Entity(label="Person"), Entity(label="Location")])
        )
        assert "Alice" in no_date and "1823" not in no_date

        with_date = await _names(Ontology(entities=[Entity(label="Person"), Entity(label="Date")]))
        assert {"Alice", "1823"} <= with_date

    @pytest.mark.parametrize("name", ["1820s", "19th century", "Abbasid era"])
    def test_periods_kept(self, name):
        """A period is something facts attach to, so it stays a node."""
        assert is_valid_entity_name(name)

    @pytest.mark.parametrize("name", ["Boeing 747", "COVID-19"])
    def test_numeric_names_not_mistaken_for_dates(self, name):
        assert is_valid_entity_name(name)

    def test_prompt_still_asks_the_llm_to_verify(self):
        """Removing this instruction was tried and reverted (RESULTS.md P2.12).

        Without it the LLM emitted 813 entities instead of 719 and entity
        precision fell 0.645 -> 0.551. The code-side rules above are a floor,
        not a replacement.
        """
        assert "REMOVE any entity" in VERIFY_EXTRACT_RELS_PROMPT
        assert "VERIFY the entities" in VERIFY_EXTRACT_RELS_PROMPT


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
        ontology = Ontology(
            entities=[
                Entity(label="Command", description="FalkorDB GRAPH.* commands"),
                Entity(label="Function", description="Cypher built-in functions"),
            ],
        )
        extractor = GraphExtraction(llm=llm, entity_extractor=LLMExtractor(llm))
        chunks = _make_chunks("GRAPH.QUERY is a FalkorDB command.")
        await extractor.extract(chunks, ontology, ctx)

        # Step 2 prompt (second call) should contain the descriptions
        assert len(captured_prompts) >= 2
        step2_prompt = captured_prompts[1]
        assert "Command\n  Description: FalkorDB GRAPH.* commands" in step2_prompt
        assert "Function\n  Description: Cypher built-in functions" in step2_prompt


class TestFormatRelationPatterns:
    def test_empty_relations(self):
        assert _format_relation_patterns([]) == ""

    def test_relation_with_patterns(self):
        rels = [
            Relation(
                label="WORKS_AT",
                description="Employment",
                patterns=[("Person", "Company")],
            ),
        ]
        result = _format_relation_patterns(rels)
        assert "## Allowed Relationships" in result
        assert "WORKS_AT (Person \u2192 Company): Employment" in result

    def test_relation_with_multiple_patterns(self):
        rels = [
            Relation(
                label="LOCATED_IN",
                patterns=[("Person", "Place"), ("Organization", "Place")],
            ),
        ]
        result = _format_relation_patterns(rels)
        assert "Person \u2192 Place" in result
        assert "Organization \u2192 Place" in result

    def test_open_relation(self):
        """Relations without declared patterns render as ``- LABEL`` — no ``(any)`` noise."""
        rels = [Relation(label="RELATED_TO")]
        result = _format_relation_patterns(rels)
        assert "- RELATED_TO" in result
        assert "(any)" not in result
        assert "→" not in result

    async def test_relation_patterns_in_prompt(self, ctx):
        """Relation patterns should appear in the Step 2 LLM prompt."""
        captured_prompts: list[str] = []

        class CaptureLLM(MockLLM):
            def invoke(self, prompt, **kwargs):
                captured_prompts.append(prompt)
                return super().invoke(prompt, **kwargs)

        llm = CaptureLLM(responses=[
            json.dumps([{"name": "Alice", "type": "Person", "description": "A person"}]),
            json.dumps({"entities": [{"name": "Alice", "type": "Person", "description": "A person"}],
                        "relationships": []}),
        ])
        ontology = Ontology(
            entities=[Entity(label="Person"), Entity(label="Company")],
            relations=[
                Relation(label="WORKS_AT", description="Works at", patterns=[("Person", "Company")]),
            ],
        )
        extractor = GraphExtraction(llm=llm, entity_extractor=LLMExtractor(llm))
        chunks = _make_chunks("Alice works at Acme.")
        await extractor.extract(chunks, ontology, ctx)

        assert len(captured_prompts) >= 2
        step2_prompt = captured_prompts[1]
        assert "## Allowed Relationships" in step2_prompt
        assert "WORKS_AT (Person \u2192 Company): Works at" in step2_prompt


class TestGraphExtractionSchemaAttributes:
    """Coverage for ontology-declared attributes flowing through extraction."""

    def test_parse_step2_coerces_declared_attributes(self):
        from graphrag_sdk.core.models import Attribute
        content = json.dumps({
            "entities": [
                {
                    "name": "Marie Curie",
                    "type": "Person",
                    "description": "Scientist",
                    "attributes": {"age": "56", "birth_date": "1867-11-07"},
                },
            ],
            "relationships": [],
        })
        ontology = Ontology(
            entities=[
                Entity(
                    label="Person",
                    properties=[
                        Attribute(name="age", type="INTEGER"),
                        Attribute(name="birth_date", type="DATE"),
                    ],
                )
            ]
        )
        ents, _ = GraphExtraction._parse_step2_response(
            content, ["Person"], "c1", ontology
        )
        assert len(ents) == 1
        assert ents[0].attributes == {"age": 56, "birth_date": "1867-11-07"}

    def test_parse_step2_missing_attribute_becomes_none(self):
        """Missing values are represented as ``None`` in attributes — the
        entity is NEVER dropped. Storage strips ``None`` before writing so
        the graph sees "key missing", which is the right null semantics."""
        from graphrag_sdk.core.models import Attribute
        content = json.dumps({
            "entities": [
                {
                    "name": "Marie Curie",
                    "type": "Person",
                    "description": "",
                    "attributes": {"age": 56},
                },
            ],
            "relationships": [],
        })
        ontology = Ontology(
            entities=[
                Entity(
                    label="Person",
                    properties=[
                        Attribute(name="age", type="INTEGER"),
                        Attribute(name="birth_date", type="DATE"),
                    ],
                )
            ]
        )
        ents, _ = GraphExtraction._parse_step2_response(
            content, ["Person"], "c1", ontology
        )
        assert len(ents) == 1
        assert ents[0].attributes == {"age": 56, "birth_date": None}

    def test_aggregator_carries_attributes_with_last_write_wins(self):
        from graphrag_sdk.core.models import ExtractedEntity
        e1 = ExtractedEntity(
            name="Marie Curie",
            type="Person",
            description="",
            source_chunk_ids=["c1"],
            attributes={"age": 56, "country": "France"},
        )
        e2 = ExtractedEntity(
            name="Marie Curie",
            type="Person",
            description="",
            source_chunk_ids=["c2"],
            attributes={"age": 58, "birth_year": 1867},
        )
        merged = GraphExtraction._aggregate_entities([e1, e2])
        assert len(merged) == 1
        # `age` is overwritten last-write-wins; the other keys are unioned.
        assert merged[0].attributes == {
            "age": 58,
            "country": "France",
            "birth_year": 1867,
        }

    def test_attributes_merged_into_node_properties(self):
        from graphrag_sdk.core.models import ExtractedEntity
        ent = ExtractedEntity(
            name="Marie Curie",
            type="Person",
            description="d",
            source_chunk_ids=["c1"],
            attributes={"age": 56, "birth_date": "1867-11-07"},
        )
        nodes = GraphExtraction._entities_to_nodes([ent])
        assert nodes[0].properties["age"] == 56
        assert nodes[0].properties["birth_date"] == "1867-11-07"
        # Reserved keys still present.
        assert nodes[0].properties["name"] == "Marie Curie"

    def test_property_less_schema_keeps_attributes_empty(self):
        """Deterministic-extractor regression: ontology declares no
        properties \u2192 records have empty attributes \u2192 storage gets only the
        reserved keys."""
        content = json.dumps({
            "entities": [
                {"name": "Alice", "type": "Person", "description": "A person"},
            ],
            "relationships": [],
        })
        ontology = Ontology(entities=[Entity(label="Person")])
        ents, _ = GraphExtraction._parse_step2_response(
            content, ["Person"], "c1", ontology
        )
        assert len(ents) == 1
        assert ents[0].attributes == {}


class TestFailedChunkReporting:
    """Finding #7: a broken extraction must not look like an empty document.

    Before the fix, per-chunk failures were swallowed and replaced with
    empty results, so a document whose chunks all failed returned byte
    identical output to a document that genuinely contained no entities.
    These three arms mirror the benchmark harness that proved it.
    """

    class _ScriptedExtractor(EntityExtractor):
        """Fails on the first ``n_fail`` chunks, returns [] for the rest."""

        def __init__(self, n_fail: int) -> None:
            self._n_fail = n_fail
            self.calls = 0

        async def extract_entities(
            self, text: str, entity_types: list[str], source_chunk_id: str
        ) -> list[ExtractedEntity]:
            index = self.calls
            self.calls += 1
            if index < self._n_fail:
                raise RuntimeError(f"simulated NER failure on chunk {index}")
            return []

    async def _run(self, n_fail: int, ctx, *, silent_llm: bool = False):
        extractor = self._ScriptedExtractor(n_fail)
        # silent_llm: step 2 also yields nothing, so the graph really is empty
        # in every arm and only the new fields can tell the arms apart.
        llm = (
            _mock_hybrid_llm(step1_entities=[], step2_entities=[], step2_relationships=[])
            if silent_llm
            else _mock_hybrid_llm()
        )
        strategy = GraphExtraction(llm=llm, entity_extractor=extractor)
        chunks = _make_chunks("one.", "two.", "three.", "four.")
        result = await strategy.extract(chunks, Ontology(), ctx)
        # Guard against the instrument silently not running: if the extractor
        # was never called, every assertion below is vacuously true.
        assert extractor.calls == 4, "extractor did not run on all 4 chunks"
        return result

    async def test_empty_document_is_not_reported_as_failed(self, ctx):
        result = await self._run(0, ctx)
        assert result.chunks_attempted == 4
        assert result.failed_chunks == []
        assert result.extraction_failed is False

    async def test_total_failure_is_reported(self, ctx):
        result = await self._run(4, ctx)
        assert result.chunks_attempted == 4
        assert len(result.failed_chunks) == 4
        assert result.extraction_failed is True

    async def test_partial_failure_reports_only_the_failed_chunks(self, ctx):
        result = await self._run(2, ctx)
        assert result.chunks_attempted == 4
        assert len(result.failed_chunks) == 2
        # Not a total failure: the surviving chunks did their job.
        assert result.extraction_failed is False

    async def test_empty_and_broken_are_distinguishable(self, ctx):
        """The finding itself: these two used to be identical."""
        empty = await self._run(0, ctx, silent_llm=True)
        broken = await self._run(4, ctx, silent_llm=True)
        assert empty.nodes == broken.nodes == []
        assert (empty.chunks_attempted, empty.failed_chunks, empty.extraction_failed) != (
            broken.chunks_attempted,
            broken.failed_chunks,
            broken.extraction_failed,
        )

    async def test_failed_chunks_are_retryable_ids(self, ctx):
        """A count is not enough; the caller must be able to re-ingest."""
        result = await self._run(2, ctx)
        chunks = _make_chunks("one.", "two.", "three.", "four.")
        known = {c.uid for c in chunks.chunks}
        assert set(result.failed_chunks) <= known
        assert all(isinstance(uid, str) and uid for uid in result.failed_chunks)

    async def test_no_chunks_reports_nothing_attempted(self, ctx):
        strategy = GraphExtraction(
            llm=_mock_hybrid_llm(), entity_extractor=self._ScriptedExtractor(0)
        )
        result = await strategy.extract(TextChunks(chunks=[]), Ontology(), ctx)
        assert result.chunks_attempted == 0
        assert result.failed_chunks == []
        assert result.extraction_failed is False

    async def test_successful_extraction_reports_no_failures(self, ctx):
        """Guard the happy path: normal ingests must stay clean."""
        strategy = GraphExtraction(
            llm=_mock_hybrid_llm(), entity_extractor=LLMExtractor(_mock_hybrid_llm())
        )
        chunks = _make_chunks("Alice is a software engineer at Acme Corp.")
        result = await strategy.extract(chunks, Ontology(), ctx)
        assert len(result.nodes) > 0
        assert result.chunks_attempted == 1
        assert result.chunks_skipped == 0
        assert result.failed_chunks == []
        assert result.relation_failed_chunks == []
        assert result.extraction_failed is False


class _FlakyBatchLLM(MockLLM):
    """MockLLM whose ``abatch_invoke`` fails (or returns nothing) for chosen
    prompt indices, the way a timed-out provider call surfaces through
    ``LLMBatchItem``."""

    def __init__(self, response: str, *, fail: tuple[int, ...] = (), none: tuple[int, ...] = ()):
        super().__init__(responses=[response])
        self._fail = set(fail)
        self._none = set(none)

    async def abatch_invoke(self, prompts, **kwargs):
        from graphrag_sdk.core.providers.base import LLMBatchItem

        items = []
        for i, prompt in enumerate(prompts):
            if i in self._fail:
                items.append(LLMBatchItem(index=i, error=RuntimeError(f"timeout on {i}")))
            elif i in self._none:
                items.append(LLMBatchItem(index=i, response=None))
            else:
                items.append(LLMBatchItem(index=i, response=self.invoke(prompt)))
        return items


_STEP1_JSON = json.dumps(
    [{"name": "Alice", "type": "Person", "description": "An engineer"}]
)
_STEP2_JSON = json.dumps(
    {
        "entities": [{"name": "Alice", "type": "Person", "description": "An engineer"}],
        "relationships": [],
    }
)
# Step 2 verifying nothing makes the strategy fall back to step-1 entities,
# so per-chunk node counts below reflect what step 1 produced.
_STEP2_EMPTY_JSON = json.dumps({"entities": [], "relationships": []})


class TestLLMExtractorStep1FailureIsRecorded:
    """Review finding: only the local-extractor branch recorded step-1
    failures. On the default ``LLMExtractor`` path a timed-out NER call was
    appended as ``[]`` and the chunk reported clean."""

    async def test_not_ok_item_lands_in_failed_chunks(self, ctx):
        step1_llm = _FlakyBatchLLM(_STEP1_JSON, fail=(1, 3))
        strategy = GraphExtraction(
            llm=MockLLM(responses=[_STEP2_JSON]),
            entity_extractor=LLMExtractor(step1_llm),
        )
        result = await strategy.extract(_make_chunks("a.", "b.", "c.", "d."), Ontology(), ctx)
        assert result.chunks_attempted == 4
        assert result.failed_chunks == ["chunk-1", "chunk-3"]
        assert result.extraction_failed is False

    async def test_none_response_lands_in_failed_chunks(self, ctx):
        step1_llm = _FlakyBatchLLM(_STEP1_JSON, none=(0,))
        strategy = GraphExtraction(
            llm=MockLLM(responses=[_STEP2_JSON]),
            entity_extractor=LLMExtractor(step1_llm),
        )
        result = await strategy.extract(_make_chunks("a.", "b."), Ontology(), ctx)
        assert result.failed_chunks == ["chunk-0"]

    async def test_all_llm_step1_failures_is_total_failure(self, ctx):
        step1_llm = _FlakyBatchLLM(_STEP1_JSON, fail=(0, 1))
        strategy = GraphExtraction(
            llm=MockLLM(responses=[_STEP2_JSON]),
            entity_extractor=LLMExtractor(step1_llm),
        )
        result = await strategy.extract(_make_chunks("a.", "b."), Ontology(), ctx)
        assert result.extraction_failed is True


class TestRelationFailureIsSeparateFromExtractionFailure:
    """Review finding: a step-2 failure used to land in ``failed_chunks``, so
    ``extraction_failed`` was True for a run that wrote a complete node set."""

    class _Entities(EntityExtractor):
        async def extract_entities(self, text, entity_types, source_chunk_id):
            return [
                ExtractedEntity(
                    name=f"E{source_chunk_id}",
                    type="Person",
                    description="",
                    source_chunk_ids=[source_chunk_id],
                )
            ]

    async def test_step2_failure_keeps_entities_and_is_not_extraction_failed(self, ctx):
        strategy = GraphExtraction(
            llm=_FlakyBatchLLM(_STEP2_EMPTY_JSON, fail=(0, 1, 2)),
            entity_extractor=self._Entities(),
        )
        result = await strategy.extract(_make_chunks("a.", "b.", "c."), Ontology(), ctx)
        # Every chunk lost its relations...
        assert result.relation_failed_chunks == ["chunk-0", "chunk-1", "chunk-2"]
        # ...but every chunk's entities are in the graph.
        assert len(result.nodes) == 3
        assert result.failed_chunks == []
        assert result.extraction_failed is False

    async def test_partial_step2_failure(self, ctx):
        strategy = GraphExtraction(
            llm=_FlakyBatchLLM(_STEP2_EMPTY_JSON, fail=(1,)),
            entity_extractor=self._Entities(),
        )
        result = await strategy.extract(_make_chunks("a.", "b.", "c."), Ontology(), ctx)
        assert result.relation_failed_chunks == ["chunk-1"]
        assert result.failed_chunks == []
        assert len(result.nodes) == 3

    async def test_step1_failure_is_not_double_counted_as_relation_failure(self, ctx):
        """A chunk that already failed step 1 and then fails step 2 belongs in
        ``failed_chunks`` only; the two lists stay disjoint."""
        step1_llm = _FlakyBatchLLM(_STEP1_JSON, fail=(0,))
        strategy = GraphExtraction(
            llm=_FlakyBatchLLM(_STEP2_JSON, fail=(0,)),
            entity_extractor=LLMExtractor(step1_llm),
        )
        result = await strategy.extract(_make_chunks("a.", "b."), Ontology(), ctx)
        assert result.failed_chunks == ["chunk-0"]
        assert result.relation_failed_chunks == []

    async def test_relation_failure_warning_does_not_claim_nothing_contributed(self, ctx, caplog):
        import logging

        strategy = GraphExtraction(
            llm=_FlakyBatchLLM(_STEP2_JSON, fail=(0,)),
            entity_extractor=self._Entities(),
        )
        with caplog.at_level(logging.WARNING):
            await strategy.extract(_make_chunks("a."), Ontology(), ctx)
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "contributed nothing" not in text
        assert "relationship" in text


class _TruncatingContext(Context):
    """Context whose budget runs out after ``allow`` ``budget_exceeded`` reads,
    so the truncation path can be exercised without wall-clock timing."""

    allow: int = 0

    @property
    def budget_exceeded(self) -> bool:  # type: ignore[override]
        if self.allow > 0:
            self.allow -= 1
            return False
        return True


class TestBudgetTruncationIsReported:
    """Review finding: ``chunks_attempted`` was the post-truncation count, so a
    40-chunk document cut to 3 by the budget looked like a healthy 3-chunk one."""

    async def test_truncated_run_reports_skipped_chunks(self):
        ctx = _TruncatingContext()
        ctx.allow = 3
        strategy = GraphExtraction(
            llm=_mock_hybrid_llm(), entity_extractor=LLMExtractor(_mock_hybrid_llm())
        )
        chunks = _make_chunks(*[f"chunk {i}." for i in range(10)])
        result = await strategy.extract(chunks, Ontology(), ctx)
        assert result.chunks_attempted == 3
        assert result.chunks_skipped == 7
        assert result.failed_chunks == []
        assert result.extraction_failed is False

    async def test_truncated_run_is_distinguishable_from_short_document(self, ctx):
        truncated_ctx = _TruncatingContext()
        truncated_ctx.allow = 3
        strategy = GraphExtraction(
            llm=_mock_hybrid_llm(), entity_extractor=LLMExtractor(_mock_hybrid_llm())
        )
        truncated = await strategy.extract(
            _make_chunks(*[f"c{i}." for i in range(10)]), Ontology(), truncated_ctx
        )
        healthy = await strategy.extract(_make_chunks("c0.", "c1.", "c2."), Ontology(), ctx)
        assert truncated.chunks_attempted == healthy.chunks_attempted == 3
        assert (truncated.chunks_skipped, healthy.chunks_skipped) == (7, 0)

    async def test_budget_exhausted_before_start_reports_everything_skipped(self):
        ctx = Context(latency_budget_ms=0.0)
        assert ctx.budget_exceeded
        strategy = GraphExtraction(
            llm=_mock_hybrid_llm(), entity_extractor=LLMExtractor(_mock_hybrid_llm())
        )
        result = await strategy.extract(_make_chunks("a.", "b.", "c."), Ontology(), ctx)
        assert result.nodes == []
        assert result.chunks_attempted == 0
        assert result.chunks_skipped == 3
        assert result.extraction_failed is False


class TestReservedNodeLabels:
    """`Document`/`Chunk` are the graph store's bookkeeping labels.

    Reusing one as an entity type used to corrupt the graph silently: document
    counts picked up extracted entities (an 11-document corpus reported 106),
    and `GraphStore._write_nodes` skips `__Entity__` for structural labels, so
    the entity vanished from dedup and retrieval without any error.
    """

    @pytest.mark.parametrize("bad", ["Document", "Chunk"])
    def test_reserved_entity_type_is_rejected(self, bad):
        llm = MockLLM()
        with pytest.raises(ValueError, match="reserved label"):
            GraphExtraction(
                llm=llm,
                entity_extractor=LLMExtractor(llm),
                entity_types=["Person", bad],
            )

    @pytest.mark.parametrize("ok", ["document", "chunk", "DOCUMENT"])
    def test_match_is_exact_like_the_store(self, ok):
        """FalkorDB labels are case-sensitive and `GraphStore._write_nodes`
        tests membership exactly, so `document` is a distinct label the store
        handles correctly (it gets `__Entity__`; `MATCH (d:Document)` never
        sees it). Rejecting it would break configs that used to work."""
        llm = MockLLM()
        extractor = GraphExtraction(
            llm=llm, entity_extractor=LLMExtractor(llm), entity_types=["Person", ok]
        )
        assert ok in extractor.entity_types

    def test_error_names_the_offending_label(self):
        llm = MockLLM()
        with pytest.raises(ValueError, match="Document"):
            GraphExtraction(
                llm=llm,
                entity_extractor=LLMExtractor(llm),
                entity_types=["Document"],
            )

    def test_non_reserved_types_still_allowed(self):
        llm = MockLLM()
        extractor = GraphExtraction(
            llm=llm,
            entity_extractor=LLMExtractor(llm),
            entity_types=["Publication", "TextSegment"],
        )
        assert extractor.entity_types == ["Publication", "TextSegment"]

    def test_ontology_labels_are_checked_too(self):
        """The ontology path assigns entity types without going through
        __init__, so it needs its own guard or the fix is bypassable."""
        with pytest.raises(ValueError, match="reserved label"):
            _reject_reserved_labels(["Person", "Document"])

    def test_store_and_extractor_share_one_definition(self):
        """Hardcoded copies would drift; the bug returns when they do."""
        from graphrag_sdk.retrieval.strategies import cypher_generation

        assert GraphStore._STRUCTURAL_LABELS is RESERVED_NODE_LABELS
        # The Cypher generator's allow-list is the third copy; it adds the
        # `__Entity__` marker on top of the store's structural labels.
        assert cypher_generation._STRUCTURAL_LABELS == RESERVED_NODE_LABELS | {"__Entity__"}


class TestDefaultRelationTypes:
    """The shipped relation vocabulary — the counterpart to DEFAULT_ENTITY_TYPES.

    Entity extraction always shipped a default type list; relations shipped
    nothing, so the prompt asked the model to invent a label per edge. Measured
    on an 11-document corpus that produced 447 distinct labels against 30 in
    gold. Supplying a default list doubled exact triple F1 (0.065 -> 0.134) with
    no loss of recall, and held across five unrelated Wikipedia domains
    (vocabulary 2.2-2.9x smaller, 10-18% -> 66-81% of edges on the list).
    """

    def test_default_is_applied_when_nothing_is_passed(self):
        ge = GraphExtraction(llm=MockLLM())
        assert ge.relation_types == list(DEFAULT_RELATION_TYPES)
        assert len(ge.relation_types) > 0

    def test_explicit_list_overrides_the_default(self):
        ge = GraphExtraction(llm=MockLLM(), relation_types=["eats", "owns"])
        assert ge.relation_types == ["eats", "owns"]

    def test_empty_list_restores_open_vocabulary(self):
        """``[]`` is a request, not an omission.

        Guards the ``is None`` check: a truthiness test would silently swap an
        explicit open-vocabulary request for the default list.
        """
        ge = GraphExtraction(llm=MockLLM(), relation_types=[])
        assert ge.relation_types == []

    def test_default_list_is_not_shared_between_instances(self):
        a = GraphExtraction(llm=MockLLM())
        b = GraphExtraction(llm=MockLLM())
        a.relation_types.append("mutated")
        assert "mutated" not in b.relation_types
        assert "mutated" not in DEFAULT_RELATION_TYPES

    def test_default_labels_are_well_formed(self):
        for label in DEFAULT_RELATION_TYPES:
            assert label == label.lower(), f"{label} is not lower_snake_case"
            assert " " not in label, f"{label} contains a space"
            assert label.replace("_", "").isalpha(), f"{label} has odd characters"
        assert len(set(DEFAULT_RELATION_TYPES)) == len(DEFAULT_RELATION_TYPES)

    def test_default_list_reaches_the_prompt(self):
        """The list is worthless if it never renders into the prompt."""
        rels = [Relation(label=lbl) for lbl in DEFAULT_RELATION_TYPES]
        block = _format_relation_patterns(rels)
        assert "## Allowed Relationships" in block
        for label in DEFAULT_RELATION_TYPES:
            assert label in block
        assert "MUST be one of" in _relationship_type_instruction(rels)

    def test_open_vocabulary_prompt_when_list_is_empty(self):
        assert _format_relation_patterns([]) == ""
        assert "UPPER_SNAKE_CASE" in _relationship_type_instruction([])
