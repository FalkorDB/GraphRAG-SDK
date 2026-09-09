"""Tests for entity_extractors.py — EntityExtractor ABC + implementations."""

from __future__ import annotations

import json
import logging
import re

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

    def test_candidate_threshold_below_zero_rejected(self):
        # A negative floor makes the model return every candidate span, all of
        # which would reach step 2 as "Unknown".
        with pytest.raises(ValueError, match="candidate_threshold"):
            GLiNERExtractor(candidate_threshold=-1)

    def test_threshold_outside_unit_interval_rejected(self):
        with pytest.raises(ValueError, match="threshold"):
            GLiNERExtractor(threshold=1.5)
        with pytest.raises(ValueError, match="threshold"):
            GLiNERExtractor(threshold=-0.1)


# ── GLiNER windowing ──────────────────────────────────────────


class _StubGLiNER:
    """Stand-in for a loaded GLiNER model.

    Exposes ``config.max_len`` and ``data_processor.words_splitter`` the way
    the real model does, and a ``predict_entities`` that returns every
    occurrence of a known phrase inside the text it is given, with exclusive
    char offsets — the same contract as GLiNER. Records every call so tests
    can count windows and inspect what each one saw.
    """

    _WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")

    def __init__(self, phrases: dict[str, tuple[str, float]], max_len: int = 384):
        self.phrases = phrases
        self.config = type("Cfg", (), {"max_len": max_len})()
        self.data_processor = type("DP", (), {"words_splitter": self._split})()
        self.calls: list[str] = []

    def _split(self, text):
        for m in self._WORD.finditer(text):
            yield m.group(), m.start(), m.end()

    def predict_entities(self, text, labels, threshold=0.5, **_):
        self.calls.append(text)
        out = []
        for phrase, (label, score) in self.phrases.items():
            if label not in labels or score < threshold:
                continue
            for m in re.finditer(re.escape(phrase), text):
                out.append(
                    {"text": m.group(), "label": label, "score": score,
                     "start": m.start(), "end": m.end()}
                )
        return sorted(out, key=lambda p: p["start"])


def _long_text(n_words: int, entities: dict[int, str]) -> str:
    """``n_words`` filler words with multi-word ``entities`` spliced in at the
    given word indices (each phrase occupies as many word slots as it has
    words, so the total stays at ``n_words``)."""
    words = [f"w{i}" for i in range(n_words)]
    for idx, phrase in entities.items():
        parts = phrase.split()
        words[idx : idx + len(parts)] = parts
    return " ".join(words)


class TestGLiNERWindowing:
    """``_predict_body`` / ``_merge`` / ``_resolve_window`` against a stub
    model: offsets round-trip, windowing is invisible in the output, boundary
    entities are not duplicated, and degenerate configurations fail loudly."""

    PHRASES = {"Marie Curie": ("person", 0.95), "Nobel Prize": ("award", 0.9)}

    def _extractor(self, stub, **kw):
        ex = GLiNERExtractor(threshold=0.75, **kw)
        ex._model = stub  # bypass model loading
        return ex

    def test_short_text_takes_fast_path(self):
        stub = _StubGLiNER(self.PHRASES)
        ex = self._extractor(stub)
        text = "Marie Curie won the Nobel Prize."
        preds = ex._predict_sync(text, ["Person", "Award"])
        assert stub.calls == [text]
        assert [p["text"] for p in preds] == ["Marie Curie", "Nobel Prize"]

    def test_windowed_offsets_round_trip(self):
        stub = _StubGLiNER(self.PHRASES)
        ex = self._extractor(stub)
        # 349 puts "Marie Curie" on words 349-350, straddling the edge of the
        # first 350-word window.
        text = _long_text(1000, {5: "Marie Curie", 349: "Nobel Prize", 700: "Marie Curie",
                                 995: "Nobel Prize"})
        preds = ex._predict_sync(text, ["Person", "Award"])
        assert len(stub.calls) > 1, "text must have been windowed"
        for p in preds:
            assert text[p["start"] : p["end"]] == p["text"]

    def test_windowed_equals_unwindowed(self):
        stub_w = _StubGLiNER(self.PHRASES)
        stub_u = _StubGLiNER(self.PHRASES)
        text = _long_text(2000, {5: "Marie Curie", 349: "Nobel Prize", 651: "Marie Curie",
                                 700: "Nobel Prize", 1300: "Marie Curie", 1998: "Nobel Prize"})
        windowed = self._extractor(stub_w)._predict_sync(text, ["Person", "Award"])
        unwindowed = self._extractor(stub_u, window_tokens=10_000)._predict_sync(
            text, ["Person", "Award"]
        )
        assert len(stub_w.calls) == 7  # ceil((2000 - 350) / 302) + 1
        assert stub_u.calls == [text]
        assert windowed == unwindowed
        assert len(windowed) == 6

    def test_boundary_entity_not_duplicated(self):
        stub = _StubGLiNER(self.PHRASES)
        ex = self._extractor(stub, window_tokens=100, window_overlap=20)
        # words 99-100: seen whole by window 2 (80-179), and by window 1 only
        # if the phrase-level stub happens to see both words — either way it
        # must come out exactly once.
        text = _long_text(300, {99: "Marie Curie", 180: "Nobel Prize"})
        preds = ex._predict_sync(text, ["Person", "Award"])
        assert [p["text"] for p in preds] == ["Marie Curie", "Nobel Prize"]

    def test_merge_output_is_non_overlapping(self):
        stub = _StubGLiNER(self.PHRASES)
        ex = self._extractor(stub, window_tokens=60, window_overlap=30)
        text = _long_text(500, {i: "Marie Curie" for i in range(0, 500, 25)})
        preds = ex._predict_sync(text, ["Person"])
        assert len(preds) == 20
        for a, b in zip(preds, preds[1:]):
            assert a["end"] <= b["start"]

    def test_merge_drops_truncated_fragment_with_other_label(self):
        # Window A clipped "Acme Corporation Ltd" at its right edge and
        # returned the fragment under a different label; window B saw it whole.
        merged = GLiNERExtractor._merge([
            {"text": "Acme", "label": "product", "score": 0.6, "start": 100, "end": 104},
            {"text": "Acme Corporation Ltd", "label": "organization", "score": 0.9,
             "start": 100, "end": 120},
        ])
        assert [(p["text"], p["label"]) for p in merged] == [("Acme Corporation Ltd", "organization")]

    def test_merge_prefers_longer_span_even_at_lower_score(self):
        # The clipped fragment can score higher than the whole entity; the
        # failure mode is truncation, so length wins over score.
        merged = GLiNERExtractor._merge([
            {"text": "Corporation Ltd", "label": "organization", "score": 0.95,
             "start": 105, "end": 120},
            {"text": "Acme Corporation Ltd", "label": "organization", "score": 0.7,
             "start": 100, "end": 120},
        ])
        assert [p["text"] for p in merged] == ["Acme Corporation Ltd"]

    def test_merge_partial_overlap_resolved(self):
        # Partially overlapping spans (not nested) also collapse to one; the
        # longer span wins, the rest of the output is untouched.
        merged = GLiNERExtractor._merge([
            {"text": "New York", "label": "location", "score": 0.8, "start": 0, "end": 8},
            {"text": "York Times", "label": "organization", "score": 0.7, "start": 4, "end": 14},
            {"text": "Alice", "label": "person", "score": 0.9, "start": 20, "end": 25},
        ])
        assert [p["text"] for p in merged] == ["York Times", "Alice"]

    def test_merge_same_span_two_labels_keeps_best_score(self):
        merged = GLiNERExtractor._merge([
            {"text": "Paris", "label": "person", "score": 0.6, "start": 0, "end": 5},
            {"text": "Paris", "label": "location", "score": 0.9, "start": 0, "end": 5},
            {"text": "Paris", "label": "location", "score": 0.8, "start": 0, "end": 5},
        ])
        assert [(p["label"], p["score"]) for p in merged] == [("location", 0.9)]

    def test_merge_scales(self):
        # 20k predictions: must be O(n log n), not quadratic.
        import time

        preds = [
            {"text": "x", "label": "t", "score": 0.5, "start": i * 10, "end": i * 10 + 5}
            for i in range(20_000)
        ]
        t0 = time.perf_counter()
        assert len(GLiNERExtractor._merge(preds)) == 20_000
        assert time.perf_counter() - t0 < 2.0

    def test_window_count_in_log_matches_calls(self, caplog):
        stub = _StubGLiNER(self.PHRASES)
        ex = self._extractor(stub, window_tokens=350, window_overlap=300)
        text = _long_text(400, {})
        with caplog.at_level(logging.DEBUG, logger="graphrag_sdk.ingestion.extraction_strategies.entity_extractors"):
            ex._predict_sync(text, ["Person"])
        assert len(stub.calls) == 2
        assert "400 word-tokens -> 2 windows" in caplog.text

    # -- _resolve_window --------------------------------------------------

    def test_window_derived_from_model_max_len(self):
        ex = self._extractor(_StubGLiNER({}, max_len=384))
        assert ex._resolve_window(ex._model) == 350

    def test_window_derived_without_config_defaults_to_384(self):
        ex = self._extractor(object())
        assert ex._resolve_window(ex._model) == 350

    def test_explicit_window_wins(self):
        ex = self._extractor(_StubGLiNER({}, max_len=2048), window_tokens=200)
        assert ex._resolve_window(ex._model) == 200

    def test_derived_window_checked_against_overlap(self):
        # max_len 90 clamps to the 64-word floor; an overlap of 64 collides
        # with the derived window and must fail instead of stepping 1 word.
        ex = self._extractor(_StubGLiNER({}, max_len=90), window_overlap=64)
        with pytest.raises(ValueError, match="window_overlap"):
            ex._resolve_window(ex._model)

    # -- degenerate configurations fail in __init__ -------------------------

    @pytest.mark.parametrize("kw", [
        {"window_tokens": 0},
        {"window_tokens": -5},
        {"window_overlap": -1},
        {"window_tokens": 30},              # default overlap 48 >= window
        {"window_tokens": 30, "window_overlap": 30},
    ])
    def test_degenerate_window_config_rejected(self, kw):
        with pytest.raises(ValueError, match="window"):
            GLiNERExtractor(**kw)

    def test_zero_overlap_allowed(self):
        stub = _StubGLiNER(self.PHRASES)
        ex = self._extractor(stub, window_tokens=100, window_overlap=0)
        text = _long_text(250, {10: "Marie Curie", 200: "Nobel Prize"})
        preds = ex._predict_sync(text, ["Person", "Award"])
        assert len(stub.calls) == 3
        assert [p["text"] for p in preds] == ["Marie Curie", "Nobel Prize"]

    def test_no_splitter_cached_on_instance(self):
        # Inference runs unlocked from several threads; the extractor must not
        # mutate shared state while predicting.
        stub = _StubGLiNER(self.PHRASES)
        ex = self._extractor(stub)
        before = dict(vars(ex))
        ex._predict_sync(_long_text(800, {5: "Marie Curie"}), ["Person"])
        assert vars(ex) == before
