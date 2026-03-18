"""Tests for entity_extractors.py — single EntityExtractor with multiple backends."""

from __future__ import annotations

import json
from typing import Any

import pytest

from graphrag_sdk.core.models import ExtractedEntity
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
    EntityExtractor,
)

from .conftest import MockLLM


# ── LLM mode tests ────────────────────────────────────────────


class TestEntityExtractorLLMMode:
    @pytest.fixture
    def extractor(self):
        llm = MockLLM(responses=[json.dumps([
            {"name": "Alice", "type": "Person", "description": "A software engineer"},
            {"name": "Acme Corp", "type": "Organization", "description": "A tech company"},
        ])])
        return EntityExtractor(llm=llm)

    def test_mode_is_llm(self, extractor):
        assert extractor.mode == "llm"

    async def test_basic_extraction(self, extractor):
        entities = await extractor.extract_entities(
            text="Alice works at Acme Corp.",
            entity_types=["Person", "Organization"],
            source_chunk_id="chunk-0",
        )
        assert len(entities) == 2
        names = {e.name for e in entities}
        assert "Alice" in names
        assert "Acme Corp" in names

    async def test_entities_have_source_chunk(self, extractor):
        entities = await extractor.extract_entities(
            text="Alice works at Acme Corp.",
            entity_types=["Person", "Organization"],
            source_chunk_id="chunk-42",
        )
        for ent in entities:
            assert "chunk-42" in ent.source_chunk_ids

    async def test_entities_have_types(self, extractor):
        entities = await extractor.extract_entities(
            text="Alice works at Acme Corp.",
            entity_types=["Person", "Organization"],
            source_chunk_id="chunk-0",
        )
        types = {e.type for e in entities}
        assert "Person" in types
        assert "Organization" in types

    async def test_invalid_json_returns_empty(self):
        llm = MockLLM(responses=["this is not json"])
        extractor = EntityExtractor(llm=llm)
        entities = await extractor.extract_entities(
            text="Some text", entity_types=["Person"], source_chunk_id="chunk-0"
        )
        assert entities == []

    async def test_filters_invalid_names(self):
        llm = MockLLM(responses=[json.dumps([
            {"name": "he", "type": "Person", "description": "A pronoun"},
            {"name": "A", "type": "Person", "description": "Single char"},
            {"name": "Alice", "type": "Person", "description": "Valid"},
        ])])
        extractor = EntityExtractor(llm=llm)
        entities = await extractor.extract_entities(
            text="He and Alice", entity_types=["Person"], source_chunk_id="c0"
        )
        assert len(entities) == 1
        assert entities[0].name == "Alice"

    async def test_markdown_fences_stripped(self):
        response = '```json\n[{"name": "Alice", "type": "Person", "description": "desc"}]\n```'
        llm = MockLLM(responses=[response])
        extractor = EntityExtractor(llm=llm)
        entities = await extractor.extract_entities(
            text="Alice", entity_types=["Person"], source_chunk_id="c0"
        )
        assert len(entities) == 1
        assert entities[0].name == "Alice"

    async def test_type_mapping(self):
        llm = MockLLM(responses=[json.dumps([
            {"name": "Tokyo", "type": "city", "description": "Capital of Japan"},
        ])])
        extractor = EntityExtractor(llm=llm)
        entities = await extractor.extract_entities(
            text="Tokyo", entity_types=["Location", "Person"], source_chunk_id="c0"
        )
        # "city" doesn't match any allowed type -> Unknown
        assert len(entities) == 1
        assert entities[0].type == "Unknown"

    async def test_dict_response_with_entities_key(self):
        """LLM wraps entities in a dict instead of returning bare array."""
        llm = MockLLM(responses=[json.dumps({
            "entities": [
                {"name": "Bob", "type": "Person", "description": "A person"},
            ]
        })])
        extractor = EntityExtractor(llm=llm)
        entities = await extractor.extract_entities(
            text="Bob", entity_types=["Person"], source_chunk_id="c0"
        )
        assert len(entities) == 1
        assert entities[0].name == "Bob"

    async def test_llm_confidence_and_spans(self):
        """LLM response with confidence and spans should be parsed."""
        llm = MockLLM(responses=[json.dumps([
            {"name": "Alice", "type": "Person", "description": "Engineer",
             "confidence": 0.95, "start": 0, "end": 5},
            {"name": "Paris", "type": "Location", "description": "City",
             "confidence": 0.88, "start": 20, "end": 25},
        ])])
        extractor = EntityExtractor(llm=llm)
        entities = await extractor.extract_entities(
            text="Alice went to Paris.", entity_types=["Person", "Location"],
            source_chunk_id="chunk-3",
        )
        assert len(entities) == 2

        alice = next(e for e in entities if e.name == "Alice")
        assert alice.confidence == 0.95
        assert "chunk-3" in alice.spans
        assert alice.spans["chunk-3"] == [{"start": 0, "end": 5}]

    async def test_llm_low_confidence_becomes_unknown(self):
        """LLM entities with confidence below threshold should be labeled Unknown."""
        llm = MockLLM(responses=[json.dumps([
            {"name": "Maybe", "type": "Person", "description": "Uncertain",
             "confidence": 0.3, "start": 0, "end": 5},
        ])])
        extractor = EntityExtractor(llm=llm, threshold=0.75)
        entities = await extractor.extract_entities(
            text="Maybe is here.", entity_types=["Person"], source_chunk_id="c0",
        )
        assert len(entities) == 1
        assert entities[0].type == "Unknown"

    async def test_llm_no_confidence_no_spans(self):
        """LLM response without confidence/spans should still work (no spans property)."""
        llm = MockLLM(responses=[json.dumps([
            {"name": "Alice", "type": "Person", "description": "Engineer"},
        ])])
        extractor = EntityExtractor(llm=llm)
        entities = await extractor.extract_entities(
            text="Alice", entity_types=["Person"], source_chunk_id="c0",
        )
        assert len(entities) == 1
        assert not hasattr(entities[0], "spans") or not entities[0].spans


# ── Custom model mode tests ──────────────────────────────────


class _MockNERModel:
    """Mock NER model implementing predict_entities protocol."""

    def predict_entities(
        self, text: str, labels: list[str], **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [
            {"text": "Alice", "label": "person"},
            {"text": "Paris", "label": "location"},
        ]


class TestEntityExtractorCustomMode:
    @pytest.fixture
    def extractor(self):
        return EntityExtractor(model=_MockNERModel())

    def test_mode_is_custom(self, extractor):
        assert extractor.mode == "custom"

    async def test_extraction(self, extractor):
        entities = await extractor.extract_entities(
            text="Alice went to Paris.",
            entity_types=["Person", "Location"],
            source_chunk_id="chunk-0",
        )
        assert len(entities) == 2
        names = {e.name for e in entities}
        assert "Alice" in names
        assert "Paris" in names


# ── GLiNER2 mode tests ───────────────────────────────────────


class TestEntityExtractorGLiNER2Mode:
    def test_default_mode_is_gliner2(self):
        extractor = EntityExtractor()
        assert extractor.mode == "gliner2"

    async def test_import_error_when_gliner_missing(self):
        """Default GLiNER2 mode raises ImportError if gliner not installed."""
        try:
            import gliner  # noqa: F401
            pytest.skip("gliner is installed")
        except ImportError:
            extractor = EntityExtractor()
            with pytest.raises(ImportError, match="GLiNER"):
                await extractor.extract_entities("text", ["Person"], "c0")


class TestGLiNER2ResponseParsing:
    """Test _parse_gliner2_response with score and spans."""

    def test_high_confidence_gets_typed(self):
        extractor = EntityExtractor(threshold=0.75)
        predictions = [
            {"text": "Alice", "label": "person", "score": 0.95, "start": 0, "end": 5},
        ]
        entities = extractor._parse_gliner2_response(
            predictions, ["Person", "Location"], "chunk-0"
        )
        assert len(entities) == 1
        assert entities[0].name == "Alice"
        assert entities[0].type == "Person"

    def test_low_confidence_gets_unknown(self):
        extractor = EntityExtractor(threshold=0.75)
        predictions = [
            {"text": "Bob", "label": "person", "score": 0.50, "start": 10, "end": 13},
        ]
        entities = extractor._parse_gliner2_response(
            predictions, ["Person"], "chunk-0"
        )
        assert len(entities) == 1
        assert entities[0].type == "Unknown"

    def test_spans_stored_per_chunk(self):
        extractor = EntityExtractor(threshold=0.5)
        predictions = [
            {"text": "Alice", "label": "person", "score": 0.95, "start": 0, "end": 5},
            {"text": "Paris", "label": "location", "score": 0.88, "start": 20, "end": 25},
        ]
        entities = extractor._parse_gliner2_response(
            predictions, ["Person", "Location"], "chunk-7"
        )
        assert len(entities) == 2

        alice = next(e for e in entities if e.name == "Alice")
        assert hasattr(alice, "spans")
        assert "chunk-7" in alice.spans
        assert alice.spans["chunk-7"] == [{"start": 0, "end": 5}]

        paris = next(e for e in entities if e.name == "Paris")
        assert paris.spans["chunk-7"] == [{"start": 20, "end": 25}]

    def test_confidence_stored_on_entity(self):
        extractor = EntityExtractor(threshold=0.5)
        predictions = [
            {"text": "Alice", "label": "person", "score": 0.95, "start": 0, "end": 5},
        ]
        entities = extractor._parse_gliner2_response(
            predictions, ["Person"], "chunk-0"
        )
        assert hasattr(entities[0], "confidence")
        assert entities[0].confidence == 0.95

    def test_invalid_names_filtered(self):
        extractor = EntityExtractor(threshold=0.5)
        predictions = [
            {"text": "he", "label": "person", "score": 0.99, "start": 0, "end": 2},
            {"text": "Alice", "label": "person", "score": 0.90, "start": 5, "end": 10},
        ]
        entities = extractor._parse_gliner2_response(
            predictions, ["Person"], "chunk-0"
        )
        assert len(entities) == 1
        assert entities[0].name == "Alice"

    def test_empty_response(self):
        extractor = EntityExtractor(threshold=0.5)
        entities = extractor._parse_gliner2_response(
            [], ["Person"], "chunk-0"
        )
        assert entities == []

    def test_multiple_entities_same_type(self):
        extractor = EntityExtractor(threshold=0.5)
        predictions = [
            {"text": "Alice", "label": "person", "score": 0.95, "start": 0, "end": 5},
            {"text": "Bob", "label": "person", "score": 0.88, "start": 10, "end": 13},
        ]
        entities = extractor._parse_gliner2_response(
            predictions, ["Person"], "chunk-0"
        )
        assert len(entities) == 2


# ── Mode precedence tests ────────────────────────────────────


class TestEntityExtractorModePrecedence:
    def test_llm_overrides_model(self):
        """When both llm and model are provided, llm takes precedence."""
        llm = MockLLM()
        model = _MockNERModel()
        extractor = EntityExtractor(llm=llm, model=model)
        assert extractor.mode == "llm"

    def test_model_over_default(self):
        """Custom model overrides default GLiNER2."""
        model = _MockNERModel()
        extractor = EntityExtractor(model=model)
        assert extractor.mode == "custom"
