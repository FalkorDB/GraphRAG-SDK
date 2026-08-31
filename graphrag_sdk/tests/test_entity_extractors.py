"""Tests for entity_extractors.py — EntityExtractor ABC + implementations."""

from __future__ import annotations

import json

import pytest

from graphrag_sdk.core.models import ExtractedEntity
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
    EntityExtractor,
    GLiNERExtractor,
    LLMExtractor,
    _parse_predictions,
)

from .conftest import MockLLM


# ── LLMExtractor Tests ────────────────────────────────────────


class TestLLMExtractor:
    @pytest.fixture
    def extractor(self):
        llm = MockLLM(responses=[json.dumps([
            {"name": "Alice", "type": "Person", "description": "A software engineer"},
            {"name": "Acme Corp", "type": "Organization", "description": "A tech company"},
        ])])
        return LLMExtractor(llm)

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

    async def test_invalid_json_returns_empty(self):
        extractor = LLMExtractor(MockLLM(responses=["this is not json"]))
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
        extractor = LLMExtractor(llm)
        entities = await extractor.extract_entities(
            text="He and Alice", entity_types=["Person"], source_chunk_id="c0"
        )
        assert len(entities) == 1
        assert entities[0].name == "Alice"

    async def test_markdown_fences_stripped(self):
        response = '```json\n[{"name": "Alice", "type": "Person", "description": "desc"}]\n```'
        extractor = LLMExtractor(MockLLM(responses=[response]))
        entities = await extractor.extract_entities(
            text="Alice", entity_types=["Person"], source_chunk_id="c0"
        )
        assert len(entities) == 1

    async def test_confidence_and_spans(self):
        llm = MockLLM(responses=[json.dumps([
            {"name": "Alice", "type": "Person", "description": "Engineer",
             "confidence": 0.95, "start": 0, "end": 5},
        ])])
        extractor = LLMExtractor(llm)
        entities = await extractor.extract_entities(
            text="Alice", entity_types=["Person"], source_chunk_id="chunk-3",
        )
        assert entities[0].confidence == 0.95
        assert entities[0].spans["chunk-3"] == [{"start": 0, "end": 5}]

    async def test_low_confidence_becomes_unknown(self):
        llm = MockLLM(responses=[json.dumps([
            {"name": "Maybe", "type": "Person", "description": "Uncertain",
             "confidence": 0.3, "start": 0, "end": 5},
        ])])
        extractor = LLMExtractor(llm, threshold=0.75)
        entities = await extractor.extract_entities(
            text="Maybe", entity_types=["Person"], source_chunk_id="c0",
        )
        assert entities[0].type == "Unknown"


# ── GLiNERExtractor Tests ────────────────────────────────────


class TestGLiNERExtractor:
    async def test_import_error_when_gliner_missing(self):
        try:
            import gliner  # noqa: F401
            pytest.skip("gliner is installed")
        except ImportError:
            extractor = GLiNERExtractor()
            with pytest.raises(ImportError, match="GLiNER"):
                await extractor.extract_entities("text", ["Person"], "c0")


# ── Shared parser tests ──────────────────────────────────────


class TestParsePredictions:
    def test_high_confidence_typed(self):
        preds = [{"text": "Alice", "label": "person", "score": 0.95, "start": 0, "end": 5}]
        ents = _parse_predictions(preds, ["Person"], "c0", 0.75)
        assert ents[0].type == "Person"

    def test_low_confidence_unknown(self):
        preds = [{"text": "Bob", "label": "person", "score": 0.50, "start": 0, "end": 3}]
        ents = _parse_predictions(preds, ["Person"], "c0", 0.75)
        assert ents[0].type == "Unknown"

    def test_spans_stored(self):
        preds = [{"text": "Alice", "label": "person", "score": 0.9, "start": 10, "end": 15}]
        ents = _parse_predictions(preds, ["Person"], "chunk-7", 0.5)
        assert ents[0].spans["chunk-7"] == [{"start": 10, "end": 15}]

    def test_invalid_names_filtered(self):
        preds = [
            {"text": "he", "label": "person", "score": 0.99, "start": 0, "end": 2},
            {"text": "Alice", "label": "person", "score": 0.9, "start": 5, "end": 10},
        ]
        ents = _parse_predictions(preds, ["Person"], "c0", 0.5)
        assert len(ents) == 1
        assert ents[0].name == "Alice"

    def test_empty(self):
        assert _parse_predictions([], ["Person"], "c0", 0.5) == []


# ── ABC Contract Test ────────────────────────────────────────


class TestEntityExtractorABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            EntityExtractor()  # type: ignore[abstract]

    def test_custom_subclass(self):
        class MyExtractor(EntityExtractor):
            async def extract_entities(self, text, entity_types, source_chunk_id):
                return [ExtractedEntity(
                    name="Test", type="Person", description="",
                    source_chunk_ids=[source_chunk_id],
                )]

        assert isinstance(MyExtractor(), EntityExtractor)


class TestGLiNERModelSharing:
    """Bugs #8 and #11 — one model copy per extractor, and a serialising lock.

    #11: with current (not peak) RSS, six extractors each loading their own
    model went 74.5 MB -> 2447 MB, ~395 MB marginal per copy, projecting
    ~11.6 GB for 30 concurrent documents. With the shared cache the same six
    sit at 1425.3 -> 1425.4 MB: 0.0 MB marginal, ~1.39 GB projected.

    #8: inference ran under a per-instance lock while the caller dispatched it
    via ``asyncio.to_thread``, so concurrent documents serialised. Counter-
    balanced over eight documents: locked 3.75/3.54 s, unlocked 2.21/2.46 s =
    1.56x. Removing it is only sound because GLiNER inference proved
    thread-safe — 40/40 documents byte-identical against the serialised run.
    """

    def _extractor(self, monkeypatch, loads):
        from graphrag_sdk.ingestion.extraction_strategies import entity_extractors as ee

        class FakeGLiNER:
            @staticmethod
            def from_pretrained(name):
                loads.append(name)
                return object()

        monkeypatch.setitem(__import__("sys").modules, "gliner",
                            type("m", (), {"GLiNER": FakeGLiNER}))
        monkeypatch.setattr(ee.GLiNERExtractor, "_MODEL_CACHE", {}, raising=False)
        return ee.GLiNERExtractor

    def test_model_loaded_once_across_instances(self, monkeypatch):
        loads = []
        cls = self._extractor(monkeypatch, loads)
        models = [cls()._load_model() for _ in range(5)]
        assert len(loads) == 1
        assert len({id(m) for m in models}) == 1

    def test_different_models_are_not_shared(self, monkeypatch):
        loads = []
        cls = self._extractor(monkeypatch, loads)
        a = cls(model_name="model-a", threshold=0.5)._load_model()
        b = cls(model_name="model-b", threshold=0.5)._load_model()
        assert loads == ["model-a", "model-b"]
        assert a is not b

    def test_inference_takes_no_lock(self):
        """Guards bug #8 against reintroduction."""
        import inspect

        from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
            GLiNERExtractor,
        )

        src = inspect.getsource(GLiNERExtractor._predict_sync)
        code = "\n".join(
            line for line in src.splitlines() if not line.strip().startswith("#")
        )
        assert "self._lock" not in code
