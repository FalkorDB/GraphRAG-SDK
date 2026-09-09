"""Tests for entity_extractors.py — EntityExtractor ABC + implementations."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from types import SimpleNamespace

import pytest

from graphrag_sdk.core.models import ExtractedEntity
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
    CompositeExtractor,
    EntityExtractor,
    GLiNERExtractor,
    LLMExtractor,
    SpacyExtractor,
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


class TestGLiNERCandidateBand:
    """The ``"Unknown"`` band is 25 % below the model threshold by default and
    follows whichever threshold is in effect."""

    def test_default_band_for_default_model(self):
        ex = GLiNERExtractor()
        assert ex._threshold == 0.75
        assert ex._candidate_threshold == pytest.approx(0.5625)

    def test_default_band_follows_model_threshold(self):
        ex = GLiNERExtractor(model_name="knowledgator/gliner-bi-small-v2.0")
        assert ex._threshold == 0.5
        assert ex._candidate_threshold == pytest.approx(0.375)

    def test_default_band_follows_explicit_threshold(self):
        ex = GLiNERExtractor(threshold=0.8)
        assert ex._candidate_threshold == pytest.approx(0.6)

    def test_none_disables_band(self):
        ex = GLiNERExtractor(candidate_threshold=None)
        assert ex._candidate_threshold is None

    def test_explicit_floor_wins(self):
        ex = GLiNERExtractor(candidate_threshold=0.7)
        assert ex._candidate_threshold == 0.7

    def test_floor_above_threshold_rejected(self):
        with pytest.raises(ValueError, match="candidate_threshold"):
            GLiNERExtractor(threshold=0.75, candidate_threshold=0.8)

    def test_band_spans_are_unknown_and_below_band_is_not_returned(self):
        # what _parse_predictions does with a score inside the band; anything
        # below the band never comes back from the model (queried at the floor)
        preds = [
            {"text": "Alice", "label": "person", "score": 0.80, "start": 0, "end": 5},
            {"text": "Bobby", "label": "person", "score": 0.60, "start": 6, "end": 11},
        ]
        ents = _parse_predictions(preds, ["Person"], "c0", 0.75)
        assert [(e.name, e.type) for e in ents] == [("Alice", "Person"), ("Bobby", "Unknown")]


# ── SpacyExtractor Tests ─────────────────────────────────────


TYPES = ["Person", "Organization", "Location", "Date"]


def _fake_spacy(monkeypatch, ents=(), fail=None):
    """Install a stub ``spacy`` module; returns the list of load calls."""
    loads: list[tuple[str, list[str]]] = []

    def load(name, disable=()):
        loads.append((name, list(disable)))
        if fail is not None:
            raise fail
        return lambda text: SimpleNamespace(ents=[SimpleNamespace(**e) for e in ents])

    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(load=load))
    monkeypatch.setattr(SpacyExtractor, "_MODEL_CACHE", {}, raising=False)
    return loads


def _ent(text, label, start, end):
    return {"text": text, "label_": label, "start_char": start, "end_char": end}


class TestSpacyExtractor:
    def test_labels_rejects_bare_string(self):
        with pytest.raises(TypeError, match="labels=\\['PERSON'\\]"):
            SpacyExtractor(labels="PERSON")

    def test_labels_accepts_iterables(self):
        assert SpacyExtractor(labels=["PERSON"])._labels == frozenset({"PERSON"})
        assert SpacyExtractor()._labels == SpacyExtractor.DEFAULT_LABELS

    def test_default_confidence_is_default_gliner_threshold(self):
        # Not 0.5: the default GLiNER model's threshold is 0.75, and a
        # ``confidence >= 0.75`` filter must not drop every spaCy entity.
        assert SpacyExtractor()._confidence == GLiNERExtractor()._threshold == 0.75
        assert SpacyExtractor(confidence=0.4)._confidence == 0.4

    def test_type_for_maps_label_onto_allowed_types(self):
        ex = SpacyExtractor()
        assert ex._type_for("GPE", TYPES) == "Location"
        assert ex._type_for("GPE", ["Place"]) == "Place"
        assert ex._type_for("ORG", ["Company", "Person"]) == "Company"
        assert ex._type_for("GPE", ["Person"]) is None
        assert ex._type_for("MONEY", TYPES) is None

    async def test_extract_filters_labels_and_maps_types(self, monkeypatch):
        _fake_spacy(monkeypatch, ents=[
            _ent("Alice", "PERSON", 0, 5),
            _ent("Paris", "GPE", 10, 15),
            _ent("Acme Corp", "ORG", 20, 29),
            _ent("2020", "DATE", 30, 34),        # DATE not in DEFAULT_LABELS
            _ent("Parisians", "NORP", 40, 49),   # NORP not in DEFAULT_LABELS
        ])
        ents = await SpacyExtractor(confidence=0.4).extract_entities("text", TYPES, "c0")
        assert [(e.name, e.type) for e in ents] == [
            ("Alice", "Person"), ("Paris", "Location"), ("Acme Corp", "Organization"),
        ]
        assert ents[0].spans["c0"] == [{"start": 0, "end": 5}]
        # A low recorded confidence never demotes: spaCy has no real scores.
        assert all(e.confidence == 0.4 for e in ents)

    async def test_label_without_home_in_schema_is_dropped(self, monkeypatch):
        _fake_spacy(monkeypatch, ents=[_ent("Paris", "GPE", 0, 5), _ent("Alice", "PERSON", 6, 11)])
        ents = await SpacyExtractor().extract_entities("text", ["Person"], "c0")
        assert [e.name for e in ents] == ["Alice"]

    async def test_model_loaded_once_across_instances(self, monkeypatch):
        loads = _fake_spacy(monkeypatch)
        a, b = SpacyExtractor(), SpacyExtractor()
        await asyncio.gather(*(x.extract_entities("t", TYPES, "c0") for x in (a, b, a)))
        assert len(loads) == 1
        assert a._nlp is b._nlp

    async def test_different_models_are_not_shared(self, monkeypatch):
        loads = _fake_spacy(monkeypatch)
        a = SpacyExtractor(model_name="model-a")
        b = SpacyExtractor(model_name="model-b")
        await a.extract_entities("t", TYPES, "c0")
        await b.extract_entities("t", TYPES, "c0")
        assert [name for name, _ in loads] == ["model-a", "model-b"]
        assert a._nlp is not b._nlp

    async def test_load_disables_everything_but_ner(self, monkeypatch):
        loads = _fake_spacy(monkeypatch)
        await SpacyExtractor().extract_entities("t", TYPES, "c0")
        (_, disabled), = loads
        assert {"tagger", "parser", "attribute_ruler", "lemmatizer"} <= set(disabled)
        assert not {"ner", "tok2vec"} & set(disabled)

    async def test_missing_model_is_raised_once_and_remembered(self, monkeypatch):
        loads = _fake_spacy(monkeypatch, fail=OSError("[E050] Can't find model"))
        ex = SpacyExtractor(model_name="en_core_web_xx")
        for _ in range(3):
            with pytest.raises(OSError, match="python -m spacy download en_core_web_xx"):
                await ex.extract_entities("t", TYPES, "c0")
        assert len(loads) == 1

    async def test_import_error_when_spacy_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "spacy", None)  # makes ``import spacy`` fail
        monkeypatch.setattr(SpacyExtractor, "_MODEL_CACHE", {}, raising=False)
        with pytest.raises(ImportError, match="graphrag-sdk\\[spacy\\]"):
            await SpacyExtractor().extract_entities("t", TYPES, "c0")


# ── CompositeExtractor Tests ─────────────────────────────────


class StaticExtractor(EntityExtractor):
    """Returns canned predictions, or raises."""

    def __init__(self, preds=(), error=None):
        self._preds = list(preds)
        self._error = error

    async def extract_entities(self, text, entity_types, source_chunk_id):
        if self._error is not None:
            raise self._error
        return _parse_predictions(self._preds, entity_types, source_chunk_id, 0.0)


def _pred(text, label, start=None, end=None):
    p = {"text": text, "label": label, "score": 0.9}
    if start is not None:
        p.update(start=start, end=end)
    return p


TEXT = "Paris Observatory and New York City; Fresnel lens, then the Paris Observatory again."


async def _run(*extractors, **kw):
    ents = await CompositeExtractor(list(extractors), **kw).extract_entities(TEXT, TYPES, "c0")
    return {e.name: e for e in ents}


def _offsets(ent):
    return [(sp["start"], sp["end"]) for sp in ent.spans["c0"]]


class TestCompositeExtractor:
    def test_requires_at_least_one_extractor(self):
        with pytest.raises(ValueError, match="at least one"):
            CompositeExtractor([])

    def test_rejects_llm_extractor_even_when_nested(self):
        llm = LLMExtractor(MockLLM(responses=[]))
        with pytest.raises(TypeError, match="abatch_invoke"):
            CompositeExtractor([StaticExtractor(), llm])
        with pytest.raises(TypeError, match="LLMExtractor"):
            CompositeExtractor([CompositeExtractor([llm])])

    async def test_earliest_extractor_wins_type(self):
        first = StaticExtractor([_pred("Fresnel", "Person", 0, 7)])
        second = StaticExtractor([_pred("fresnel", "Organization", 0, 7)])
        out = await _run(first, second)
        assert [(e.name, e.type) for e in out.values()] == [("Fresnel", "Person")]

    async def test_unknown_type_is_filled_by_a_later_extractor(self):
        first = StaticExtractor([_pred("Fresnel", "Widget", 0, 7)])  # -> Unknown
        second = StaticExtractor([_pred("Fresnel", "Person", 0, 7)])
        out = await _run(first, second)
        assert out["Fresnel"].type == "Person"

    async def test_fragment_of_a_claimed_span_is_dropped(self):
        gliner = StaticExtractor([_pred("Fresnel lens", "Organization", 0, 12)])
        spacy = StaticExtractor([_pred("Fresnel", "Organization", 0, 7)])
        out = await _run(gliner, spacy)
        assert set(out) == {"Fresnel lens"}

    async def test_article_only_extension_does_not_supersede(self):
        # The docstring's measured false positive: spaCy's ``The Paris
        # Observatory`` is the same entity as GLiNER's ``Paris Observatory``.
        text = "At the Paris Observatory. Acme Corp. was there."
        gliner = StaticExtractor([_pred("Paris Observatory", "Location", 7, 24),
                                 _pred("Acme Corp", "Organization", 26, 35)])
        spacy = StaticExtractor([_pred("the Paris Observatory", "Location", 3, 24),
                                 _pred("Acme Corp.", "Organization", 26, 36)])
        ents = await CompositeExtractor([gliner, spacy]).extract_entities(text, TYPES, "c0")
        assert {e.name for e in ents} == {"Paris Observatory", "Acme Corp"}

    async def test_content_extension_supersedes(self):
        text = "The Samarkand Expedition of 892 began."
        gliner = StaticExtractor([_pred("Samarkand", "Location", 4, 13)])
        spacy = StaticExtractor([_pred("Samarkand Expedition of 892", "Event", 4, 31)])
        ents = await CompositeExtractor([gliner, spacy]).extract_entities(
            text, TYPES + ["Event"], "c0"
        )
        assert [(e.name, e.type) for e in ents] == [("Samarkand Expedition of 892", "Event")]

    async def test_longer_nested_span_supersedes_earlier_fragment(self):
        # The PR's own motivating case: GLiNER emits the fragment, spaCy the
        # full name at the same position. The longer span must win.
        gliner = StaticExtractor([_pred("Paris", "Location", 0, 5)])
        spacy = StaticExtractor([_pred("Paris Observatory", "Location", 0, 17)])
        out = await _run(gliner, spacy)
        assert set(out) == {"Paris Observatory"}
        assert _offsets(out["Paris Observatory"]) == [(0, 17)]

    async def test_superseded_fragment_keeps_its_other_occurrences(self):
        gliner = StaticExtractor([_pred("Paris", "Location", 0, 5),
                                  _pred("Paris", "Location", 40, 45)])
        spacy = StaticExtractor([_pred("Paris Observatory", "Location", 0, 17)])
        out = await _run(gliner, spacy)
        assert set(out) == {"Paris", "Paris Observatory"}
        assert _offsets(out["Paris"]) == [(40, 45)]

    async def test_partial_overlap_keeps_the_earlier_extractor(self):
        first = StaticExtractor([_pred("New York", "Location", 0, 8)])
        second = StaticExtractor([_pred("York City", "Location", 4, 13)])
        out = await _run(first, second)
        assert set(out) == {"New York"}

    async def test_identical_span_under_another_name_keeps_the_earlier(self):
        first = StaticExtractor([_pred("Dr. Smith", "Person", 0, 9)])
        second = StaticExtractor([_pred("Dr Smith", "Person", 0, 9)])
        out = await _run(first, second)
        assert set(out) == {"Dr. Smith"}

    async def test_same_extractor_entities_never_suppress_each_other(self):
        one = StaticExtractor([_pred("Paris", "Location", 0, 5),
                               _pred("Paris Observatory", "Location", 0, 17)])
        out = await _run(one)
        assert set(out) == {"Paris", "Paris Observatory"}

    async def test_repeat_occurrences_keep_every_span(self):
        # Two mentions arrive as two entities with the same name; the merged
        # entity must carry both offsets, not just the first.
        one = StaticExtractor([_pred("Fresnel lens", "Organization", 0, 12),
                               _pred("Fresnel lens", "Organization", 50, 62)])
        out = await _run(one)
        assert _offsets(out["Fresnel lens"]) == [(0, 12), (50, 62)]

    async def test_repeat_occurrences_are_claimed_against_later_fragments(self):
        gliner = StaticExtractor([_pred("Fresnel lens", "Organization", 0, 12),
                                  _pred("Fresnel lens", "Organization", 50, 62)])
        spacy = StaticExtractor([_pred("Fresnel", "Organization", 50, 57)])
        out = await _run(gliner, spacy)
        assert set(out) == {"Fresnel lens"}

    async def test_later_extractor_adds_a_new_mention_of_a_known_name(self):
        gliner = StaticExtractor([_pred("Fresnel lens", "Organization", 0, 12)])
        spacy = StaticExtractor([_pred("Fresnel lens", "Organization", 0, 12),
                                 _pred("Fresnel lens", "Organization", 50, 62)])
        out = await _run(gliner, spacy)
        assert _offsets(out["Fresnel lens"]) == [(0, 12), (50, 62)]

    async def test_suppress_overlaps_false_keeps_fragments(self):
        gliner = StaticExtractor([_pred("Fresnel lens", "Organization", 0, 12)])
        spacy = StaticExtractor([_pred("Fresnel", "Organization", 0, 7)])
        out = await _run(gliner, spacy, suppress_overlaps=False)
        assert set(out) == {"Fresnel lens", "Fresnel"}

    async def test_entities_without_spans_are_always_kept(self):
        gliner = StaticExtractor([_pred("Fresnel lens", "Organization", 0, 12)])
        other = StaticExtractor([_pred("Fresnel", "Organization")])
        out = await _run(gliner, other)
        assert set(out) == {"Fresnel lens", "Fresnel"}

    async def test_one_failing_extractor_is_skipped_with_a_warning(self, caplog):
        ok = StaticExtractor([_pred("Alice", "Person", 0, 5)])
        bad = StaticExtractor(error=RuntimeError("boom"))
        with caplog.at_level(logging.WARNING):
            out = await _run(ok, bad)
        assert set(out) == {"Alice"}
        assert "StaticExtractor failed" in caplog.text and "boom" in caplog.text

    async def test_all_failing_reraises_the_first_error(self):
        with pytest.raises(RuntimeError, match="first"):
            await _run(StaticExtractor(error=RuntimeError("first")),
                       StaticExtractor(error=RuntimeError("second")))

    @pytest.mark.parametrize("error", [
        ImportError("SpacyExtractor requires the 'spacy' extra"),
        OSError("spaCy model 'en_core_web_lg' is not installed"),
    ])
    async def test_configuration_faults_propagate_immediately(self, error):
        # A missing dependency or model is not a bad chunk; it must not become
        # 157 identical warnings while the composite quietly degrades to GLiNER.
        ok = StaticExtractor([_pred("Alice", "Person", 0, 5)])
        with pytest.raises(type(error), match=str(error)):
            await _run(ok, StaticExtractor(error=error))

    async def test_cancellation_propagates(self):
        ok = StaticExtractor([_pred("Alice", "Person", 0, 5)])
        with pytest.raises(asyncio.CancelledError):
            await _run(ok, StaticExtractor(error=asyncio.CancelledError()))
