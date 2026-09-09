# GraphRAG SDK — Extraction: Entity Extractors
# ABC + built-in implementations for step 1 entity NER.
#
# Also exports shared entity utilities (constants, ID computation,
# name validation, type mapping) used by GraphExtraction.

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar

from graphrag_sdk.core.models import ExtractedEntity
from graphrag_sdk.core.providers import LLMInterface

logger = logging.getLogger(__name__)


# ── Default Entity Types ──────────────────────────────────────────

DEFAULT_ENTITY_TYPES: list[str] = [
    "Person",
    "Organization",
    "Technology",
    "Product",
    "Location",
    "Date",
    "Event",
    "Concept",
    "Law",
    "Dataset",
    "Method",
]

UNKNOWN_LABEL = "Unknown"

#: Sentinel for "use the class default" where ``None`` is itself a valid value.
_DEFAULT: Any = object()

MIN_NAME_LEN = 2  # single-char names are noise
MAX_NAME_LEN = 80  # descriptions masquerading as names

# Pronouns and generic references that should never become entities
_PRONOUNS: set[str] = {
    "he",
    "she",
    "they",
    "it",
    "him",
    "her",
    "his",
    "them",
    "who",
    "whom",
    "i",
    "we",
    "you",
    "one",
    "me",
    "us",
    "my",
    "our",
    "your",
    "their",
    "hers",
    "its",
}

# Shell and system abbreviations that read as entities to an NER model but name
# nothing in any document domain.  Ported here from the extraction prompt, which
# used to ask the LLM to remove them: a fixed list costs nothing, cannot vary
# between runs, and cannot be quietly skipped the way the prompt instruction was
# (see RESULTS.md P2.10).  Well-known two-letter acronyms - AI, US, UK, EU, UN,
# Go - are deliberately absent and stay.
_SHELL_TOKENS: frozenset[str] = frozenset(
    {
        "sh",
        "cd",
        "ls",
        "rm",
        "cp",
        "mv",
        "dt",
        "bg",
        "fg",
        "fn",
        "df",
        "du",
        "ps",
        "cat",
        "pwd",
        "echo",
        "mkdir",
        "rmdir",
        "chmod",
        "chown",
        "grep",
        "awk",
        "sed",
        "env",
        "sudo",
        "ssh",
        "tmp",
        "var",
        "usr",
        "bin",
        "etc",
    }
)

_ENTITY_STOPLIST: set[str] = (
    _PRONOUNS
    | _SHELL_TOKENS
    | {
        # Generic/anonymous references
        "narrator",
        "the narrator",
        "author",
        "the author",
        "reader",
        "the reader",
        "speaker",
        "the speaker",
        "listener",
        "the listener",
        "the man",
        "the woman",
        "the boy",
        "the girl",
        "the child",
        "man",
        "woman",
        "boy",
        "girl",
        "child",
        "people",
        "person",
        "someone",
        "somebody",
        "everyone",
        "everybody",
        "mistress",
        "master",
        # Meta-textual
        "story",
        "chapter",
        "passage",
        "book",
        "text",
        "narrative",
        "paragraph",
        "section",
        "document",
    }
)


# ── Entity Utility Functions ─────────────────────────────────────


def compute_entity_id(name: str, entity_type: str = "") -> str:
    """Deterministic entity ID from normalized name and optional type."""
    base = name.strip().lower().replace(" ", "_")
    if entity_type:
        return f"{base}__{entity_type.strip().lower()}"
    return base


def _normalize_type_label(raw: str) -> str:
    """Normalize type string by lowercasing and removing separators."""
    s = raw.strip().lower()
    return re.sub(r"[\s_\-/]+", "", s)


# A date that pins down a single moment is an *attribute* of an event ("the
# lamp was installed in 1823"), not something you can hold a conversation
# about.  A date that names a *period* ("the 1820s", "the Abbasid era") is a
# thing facts get attached to, so it stays.
#
# This only applies when ``Date`` is not an entity type in the ontology.  An
# ontology that declares ``Date`` (the defaults do) has asked for date nodes
# and gets them; one that leaves it out gets no date nodes.
#
# Measured on the 11-document benchmark: specific dates were 63 of 274 false
# positive entities (23%).  Removing them lifted entity precision 0.577 -> 0.642
# with recall unchanged at 0.644 (F1 0.609 -> 0.643) and cost nothing, because
# every gold Date entity is a decade.  This also stops the graph accumulating
# "X happened_in 1957" edges that carry no answerable content.
_SPECIFIC_DATE_RE = re.compile(
    r"""^(?:
        (?:c\.?\s*|circa\s+|ca\.?\s*)?\d{3,4}\s*(?:ce|bce|ad|bc)?   # 1823, 1003 ce, c. 1200
      | \d{1,2}\s+[a-z]+\s+\d{4}                                     # 14 january 1904
      | [a-z]+\s+\d{1,2},?\s+\d{4}                                    # january 14, 1904
      | \d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?                              # 1904-01-14
    )$""",
    re.IGNORECASE | re.VERBOSE,
)

# Overrides the rule above: these name a span of time, not a moment.
_DATE_PERIOD_RE = re.compile(r"\d{3,4}s\b|centur|era\b|dynasty|period|decade|age\b", re.IGNORECASE)


def is_specific_date(name: str) -> bool:
    """True if name pins down one moment in time rather than naming a period.

    Specific dates are rejected as entity names unless the ontology declares a
    ``Date`` type (see ``is_valid_entity_name``): by default they belong on the
    relation that mentions them, not as nodes of their own.  Note the deliberate gap --
    a product genuinely named for a number ("747", "1984") is indistinguishable
    from a year here and will be dropped.  That trade was worth 63 false
    positives against zero true positives on the benchmark corpus, but it is
    the first thing to revisit if a domain uses numeric product names.
    """
    stripped = name.strip()
    if _DATE_PERIOD_RE.search(stripped):
        return False
    return bool(_SPECIFIC_DATE_RE.match(stripped))


def _ontology_has_date_type(entity_types: list[str] | None) -> bool:
    if not entity_types:
        return False
    return any(_normalize_type_label(t) == "date" for t in entity_types)


def is_valid_entity_name(name: str, entity_types: list[str] | None = None) -> bool:
    """Return True if name passes quality gates for entity extraction.

    Specific dates ("1823") are rejected unless ``entity_types`` contains
    ``Date``: when the ontology asks for date nodes, dates are kept.
    """
    if not name or not name.strip():
        return False
    stripped = name.strip()
    if len(stripped) < MIN_NAME_LEN or len(stripped) > MAX_NAME_LEN:
        return False
    # "US" is a country, "us" is a pronoun, and casefolding the stoplist check
    # conflates them. An all-caps short token is an acronym, not a pronoun.
    is_acronym = len(stripped) <= 3 and stripped.isupper() and stripped.isalpha()
    if not is_acronym and stripped.lower() in _ENTITY_STOPLIST:
        return False
    if not _ontology_has_date_type(entity_types) and is_specific_date(stripped):
        return False
    # Operator and punctuation tokens (+=, ->, ==, !=). A name with no letter or
    # digit anywhere in it cannot be the name of anything.
    if not any(ch.isalnum() for ch in stripped):
        return False
    return True


def label_for_type(raw_type: str, allowed_types: list[str]) -> str:
    """Map a raw type string to the closest allowed type, or UNKNOWN_LABEL."""
    if not raw_type or not raw_type.strip():
        return UNKNOWN_LABEL
    norm = _normalize_type_label(raw_type)
    for allowed in allowed_types:
        if _normalize_type_label(allowed) == norm:
            return allowed
    return UNKNOWN_LABEL


def _parse_confidence(item: dict[str, Any]) -> float | None:
    """Extract confidence from a prediction dict.

    Checks ``score`` first (GLiNER), then ``confidence`` (LLM).
    Returns None if neither is present.
    """
    for key in ("score", "confidence"):
        val = item.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


def _build_spans(
    item: dict[str, Any],
    source_chunk_id: str,
    start_key: str = "start",
    end_key: str = "end",
) -> dict[str, list[dict[str, int]]]:
    """Build spans dict from a prediction item. Returns empty dict if no spans."""
    start, end = item.get(start_key), item.get(end_key)
    if start is not None and end is not None:
        try:
            return {source_chunk_id: [{"start": int(start), "end": int(end)}]}
        except (ValueError, TypeError):
            pass
    return {}


def _strip_markdown_fences(text: str) -> str:
    """Strip ```json ... ``` fences from LLM responses."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def _parse_predictions(
    predictions: list[dict[str, Any]],
    entity_types: list[str],
    source_chunk_id: str,
    threshold: float,
) -> list[ExtractedEntity]:
    """Parse NER predictions into ExtractedEntity objects.

    Shared by GLiNERExtractor and any custom extractor that returns
    the same format: ``[{"text": ..., "label": ..., "score": ...,
    "start": ..., "end": ...}]``.
    """
    entities: list[ExtractedEntity] = []
    for pred in predictions:
        if not isinstance(pred, dict):
            continue
        name = str(pred.get("text", "")).strip()
        if not is_valid_entity_name(name, entity_types):
            continue
        raw_type = str(pred.get("label", "")).strip()

        confidence = _parse_confidence(pred)
        if confidence is not None and confidence < threshold:
            etype = UNKNOWN_LABEL
        else:
            etype = label_for_type(raw_type, entity_types)

        extra: dict[str, Any] = {}
        spans = _build_spans(pred, source_chunk_id)
        if spans:
            extra["spans"] = spans
        if confidence is not None:
            extra["confidence"] = confidence

        entities.append(
            ExtractedEntity(
                name=name,
                type=etype,
                description=pred.get("description", ""),
                source_chunk_ids=[source_chunk_id],
                **extra,
            )
        )
    return entities


# ── NER Prompt (used by LLMExtractor) ────────────────────────────

NER_PROMPT = (
    "You are an expert named entity recognition system.\n"
    "Extract all entities from the text below.\n\n"
    "## Entity Types\n"
    "Only extract entities of these types: {entity_types}\n\n"
    "## Text\n"
    "{text}\n\n"
    "## Instructions\n"
    "- Extract ALL named entities present in the text.\n"
    "- Entity names MUST be specific, named references — proper nouns, named places, "
    "titled works, specific concepts, or named objects.\n"
    "- The text may contain tables, code, or API references. Treat each function name, "
    "method name, type name, class name, or API identifier (e.g. GrB_mxm, numpy.array, "
    "torch.nn.Linear, requests.get) as a named entity of the appropriate type "
    "(typically Method, Product, or Technology).\n"
    "- Do NOT extract pronouns (he, she, they, it, him, her, his, them, who, whom, "
    "I, we, you, one).\n"
    "- Do NOT extract generic references (narrator, the narrator, author, reader, "
    "the man, the woman, people, person, someone, story, chapter, book, text).\n"
    "- If a pronoun refers to a named entity, use the named entity's actual name.\n"
    "- For each entity, provide:\n"
    "  - name: the exact text span as it appears in the text\n"
    "  - type: one of the entity types above\n"
    "  - description: a brief description\n"
    "  - confidence: a float 0-1 indicating how confident you are\n"
    "  - start: the character offset where the entity starts in the text\n"
    "  - end: the character offset where the entity ends in the text\n\n"
    "Return ONLY a JSON array of objects:\n"
    '[{{"name": "<entity_name>", "type": "<entity_type>", '
    '"description": "<brief description>", "confidence": 0.95, '
    '"start": 0, "end": 5}}]\n\n'
    "Return ONLY valid JSON, nothing else."
)


# ── ABC ──────────────────────────────────────────────────────────


class EntityExtractor(ABC):
    """Abstract base for entity extractors (step 1 of GraphExtraction).

    Subclass this to build your own NER backend. Built-in implementations:
    ``GLiNERExtractor`` (default, local) and ``LLMExtractor`` (API-based).
    """

    @abstractmethod
    async def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        source_chunk_id: str,
    ) -> list[ExtractedEntity]:
        """Extract entities from a single chunk of text.

        Args:
            text: The chunk text to extract from.
            entity_types: Allowed entity type labels.
            source_chunk_id: UID of the source chunk for provenance.

        Returns:
            List of extracted entities.
        """
        ...


# ── GLiNER Extractor (default) ───────────────────────────────────


class GLiNERExtractor(EntityExtractor):
    """Entity extraction via GLiNER local transformer model.

    Default extractor — no API calls, fast. Returns entities with
    confidence scores and character spans.

    The model is loaded lazily on first use and protected by a lock
    so a single instance can be safely shared across concurrent
    ``asyncio.to_thread`` calls (e.g. parallel doc ingestion).

    GLiNER has a hard input limit (``config.max_len``) and truncates anything
    beyond it **silently** — no exception, no warning, the tail simply never
    reaches the model. Measured on ``gliner_medium-v2.1`` (``max_len`` 384): a
    probe entity placed at word-token 388 is always returned, at 389 never.
    The default model's limit is 2048, but real documents still exceed it —
    disabling windowing on our benchmark corpus dropped recall from 0.805 to
    0.621 — so windowing matters regardless of the model.

    To keep long chunks fully visible, text longer than ``window_tokens`` is
    processed as a series of overlapping windows and the results are merged.
    Short text takes a fast path and behaves exactly as before.

    **Confidence handling.** Spans scoring at or above ``threshold`` are typed.
    Spans in the band just below it — from ``candidate_threshold`` up to
    ``threshold`` — are kept but labelled ``"Unknown"``; anything below the band
    is discarded inside GLiNER and never reaches this SDK. The rest of the
    pipeline is built for the ``"Unknown"`` label: the step-2 LLM re-types or
    drops each one against the text, ontology filtering whitelists it so
    survivors are not pruned, and entity resolution prefers any specific type
    over ``"Unknown"`` when merging duplicates.

    By default the band is **25 % of the threshold** (``CANDIDATE_BAND``), so it
    follows whichever model threshold is in effect: 0.5625–0.75 for
    ``gliner_medium-v2.1``, 0.375–0.5 for the bi-encoders. Measured end to end on
    the 11-document benchmark against the same code with the band off: NER
    spans 1828 -> 2468, the step-2 LLM kept most of them under a real type (0
    ``Unknown`` nodes reached the graph), entity recall 0.535 -> 0.571 at
    precision 0.605 -> 0.572 (F1 0.568 -> 0.572), triple relaxed F1 0.228 ->
    0.236, QA 30.4 % -> 32.4 %, ingest cost +8 %. A much wider band (floor 0.30,
    earlier measurement) was net negative — recall +0.13 for precision -0.13,
    F1 -0.027 — which is why this is a fraction of the threshold and not a
    fixed low floor. Pass ``candidate_threshold=None`` to turn the band off, or
    a number below ``threshold`` to set the floor explicitly.

    **Thresholds are model-specific and are not comparable between models.**
    ``DEFAULT_THRESHOLDS`` records the measured operating point for each known
    model, and ``threshold=None`` (the default) looks it up. Passing an explicit
    number that was tuned for a different model is the single easiest way to
    break this class: the bi-encoder models return almost nothing at the 0.75
    that suits ``gliner_medium-v2.1`` — measured at **2 entities for an entire
    corpus**, with no error raised.

    Args:
        threshold: Confidence threshold (0-1). Below this → "Unknown".
            ``None`` (default) selects the value measured for ``model_name``.
        model_name: HuggingFace model name for GLiNER.
        window_tokens: Word-tokens per inference window. ``None`` derives it
            from the model's own ``config.max_len`` minus a safety margin.
        window_overlap: Word-tokens shared between consecutive windows. Must
            exceed the model's ``max_width`` (longest representable entity,
            12 words by default) or entities on a boundary are lost.
        candidate_threshold: Floor of the ``"Unknown"`` band. Default: 25 %
            below ``threshold`` (``CANDIDATE_BAND``). ``None`` disables the
            band (spans below ``threshold`` are discarded). Must not exceed
            ``threshold``.
    """

    # Safety margin under config.max_len; the label prompt and special tokens
    # share the window with the text.
    _WINDOW_MARGIN = 34

    #: Default model. ``gliner-bi-small-v2.0`` measured better on an 11-document
    #: benchmark (ceiling recall 0.805 vs 0.709, 432MB vs 781MB on disk, 1004MB
    #: vs 1532MB resident, ~14s vs ~29s, 2048-word window vs 384) but its score
    #: calibration collapsed to <= 0.03 for every span under gliner 0.2.27 /
    #: transformers 5.6 / torch 2.13 (a ``resize_embeddings`` warning on load),
    #: returning zero entities at its 0.5 threshold with no error. The
    #: uni-encoder ``gliner_medium-v2.1`` is unaffected and stays the default
    #: until the bi-encoder path is stable across library versions.
    DEFAULT_MODEL = "urchade/gliner_medium-v2.1"

    #: Confidence thresholds are on different scales per model and MUST be
    #: re-tuned when the model changes. Each value below is the measured best
    #: end-to-end operating point, not a guess.
    DEFAULT_THRESHOLDS: dict[str, float] = {
        "knowledgator/gliner-bi-small-v2.0": 0.5,
        "knowledgator/gliner-bi-base-v2.0": 0.5,
        "urchade/gliner_medium-v2.1": 0.75,
        "urchade/gliner_large-v2.1": 0.75,
        "gliner-community/gliner_medium-v2.5": 0.75,
    }

    #: Used when ``model_name`` is not in ``DEFAULT_THRESHOLDS``. The GLiNER
    #: library's own default, which is a safer guess than assuming our tuned
    #: value transfers to an unknown model.
    _FALLBACK_THRESHOLD = 0.5

    #: Default ``candidate_threshold`` as a fraction below ``threshold``:
    #: ``threshold * (1 - CANDIDATE_BAND)``. Relative, so it tracks the
    #: per-model threshold instead of assuming one score scale.
    CANDIDATE_BAND = 0.25

    def __init__(
        self,
        threshold: float | None = None,
        model_name: str | None = None,
        window_tokens: int | None = None,
        window_overlap: int = 48,
        candidate_threshold: float | None | object = _DEFAULT,
    ) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        if threshold is None:
            threshold = self.DEFAULT_THRESHOLDS.get(self._model_name, self._FALLBACK_THRESHOLD)
            if self._model_name not in self.DEFAULT_THRESHOLDS:
                logger.warning(
                    "No measured threshold for GLiNER model %r; falling back to "
                    "%.2f. Thresholds are not comparable between models, so "
                    "tune this before relying on the results.",
                    self._model_name,
                    self._FALLBACK_THRESHOLD,
                )
        self._threshold = threshold
        if candidate_threshold is _DEFAULT:
            candidate_threshold = round(threshold * (1.0 - self.CANDIDATE_BAND), 4)
        if candidate_threshold is not None and candidate_threshold > threshold:
            raise ValueError(
                f"candidate_threshold ({candidate_threshold}) must be <= "
                f"threshold ({threshold}); a candidate floor above the demotion "
                f"line would discard the very entities it is meant to keep"
            )
        self._candidate_threshold = candidate_threshold
        self._model: Any = None
        # Retained only so callers/tests that swap in a context manager to
        # A/B the removed inference lock keep working. Nothing in this class
        # acquires it any more; model loading uses the class-level
        # ``_CACHE_LOCK`` and inference deliberately takes no lock at all.
        # See ``_predict_sync`` for the thread-safety evidence.
        self._lock = threading.Lock()
        self._window_tokens = window_tokens
        self._window_overlap = window_overlap
        self._splitter: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            self._model = self._get_shared_model(self._model_name)
        return self._model

    # Process-wide cache of loaded GLiNER models, keyed on model name.
    #
    # Bug #11: nothing cached the model, so every ``GLiNERExtractor`` instance
    # loaded its own copy. Measured with current (not peak) RSS, creating six
    # extractors and forcing each to load:
    #
    #     baseline 74.5 MB -> 2447 MB after 6 copies = ~395 MB per copy
    #
    # which projects to ~11.6 GB for 30 concurrent documents. That matches the
    # customer report of large RSS under concurrent ingest. With this cache the
    # second and later extractors for the same model cost ~0.
    #
    # Keyed on model name rather than shared unconditionally because two
    # extractors may legitimately want different models; those must stay
    # separate or one would silently answer with the other's weights.
    _MODEL_CACHE: ClassVar[dict[str, Any]] = {}
    _CACHE_LOCK: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def _get_shared_model(cls, model_name: str) -> Any:
        cached = cls._MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached
        with cls._CACHE_LOCK:
            # Double-checked: another thread may have loaded it while we waited.
            cached = cls._MODEL_CACHE.get(model_name)
            if cached is None:
                try:
                    from gliner import GLiNER
                except ImportError:
                    raise ImportError(
                        "GLiNER is required for GLiNERExtractor. Install with: pip install gliner"
                    )
                cached = GLiNER.from_pretrained(model_name)
                cls._MODEL_CACHE[model_name] = cached
        return cached

    def _resolve_window(self, model: Any) -> int:
        """Window size in word-tokens, derived from the model if not set."""
        if self._window_tokens is not None:
            return self._window_tokens
        max_len = getattr(getattr(model, "config", None), "max_len", None)
        if not isinstance(max_len, int) or max_len <= 0:
            max_len = 384
        return max(64, max_len - self._WINDOW_MARGIN)

    def _word_spans(self, model: Any, text: str) -> list[tuple[str, int, int]]:
        """Split text the same way GLiNER does, keeping char offsets."""
        if self._splitter is None:
            splitter = getattr(getattr(model, "data_processor", None), "words_splitter", None)
            if splitter is None:
                from gliner.data_processing import WordsSplitter

                splitter = WordsSplitter()
            self._splitter = splitter
        return list(self._splitter(text))

    @staticmethod
    def _merge(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deduplicate predictions from overlapping windows.

        Identical spans found in two windows collapse to the highest-scoring
        copy. A span that is strictly contained in a longer span of the same
        label is dropped: it is the truncated remains of an entity clipped by a
        window edge, which the neighbouring window saw whole.
        """
        best: dict[tuple[int, int, str], dict[str, Any]] = {}
        for p in preds:
            key = (p["start"], p["end"], p["label"])
            prev = best.get(key)
            if prev is None or p.get("score", 0.0) > prev.get("score", 0.0):
                best[key] = p

        kept: list[dict[str, Any]] = []
        for p in best.values():
            contained = any(
                q is not p
                and q["label"] == p["label"]
                and q["start"] <= p["start"]
                and p["end"] <= q["end"]
                and (q["end"] - q["start"]) > (p["end"] - p["start"])
                for q in best.values()
            )
            if not contained:
                kept.append(p)
        kept.sort(key=lambda p: (p["start"], p["end"]))
        return kept

    def _predict_sync(self, text: str, entity_types: list[str]) -> list[dict[str, Any]]:
        model = self._load_model()
        labels = [t.lower() for t in entity_types]
        window = self._resolve_window(model)
        # What we ask the MODEL for. Anything between this floor and
        # ``self._threshold`` comes back and is demoted to ``UNKNOWN_LABEL`` by
        # ``_parse_predictions`` rather than being discarded. When no candidate
        # threshold is configured the two are equal and nothing is demoted.
        floor = self._threshold if self._candidate_threshold is None else self._candidate_threshold

        # No lock here, deliberately.
        #
        # Bug #8: this whole block used to run under ``self._lock`` while the
        # caller dispatched it through ``asyncio.to_thread`` — the SDK paid for
        # threads and then serialised them anyway. Concurrent documents queued
        # behind each other in NER.
        #
        # Removing it is only sound if GLiNER inference is genuinely
        # thread-safe, so that was tested rather than assumed: eight documents
        # extracted concurrently, unlocked, compared field-by-field against the
        # serialised result, five trials — 40/40 documents byte-identical.
        # Inference mutates no model state; only ``_load_model`` does, and that
        # is guarded separately by ``_CACHE_LOCK``.
        #
        # Measured on eight documents, counterbalanced (locked, unlocked,
        # unlocked, locked) so ordering and warm caches cannot explain it:
        # locked 3.75 s / 3.54 s versus unlocked 2.21 s / 2.46 s = **1.56x**.
        # Note issue #71 claimed 3.44x; the honest measured figure is 1.56x,
        # because torch's own intra-op threading already uses the cores.
        return self._predict_body(model, text, labels, window, floor)

    def _predict_body(
        self,
        model: Any,
        text: str,
        labels: list[str],
        window: int,
        floor: float,
    ) -> list[dict[str, Any]]:
        """Windowed inference. Split out from :meth:`_predict_sync` so the
        lock removal above could be A/B tested by wrapping one call site."""
        words = self._word_spans(model, text)

        # Fast path: fits in one window, identical to unwindowed behaviour.
        if len(words) <= window:
            return model.predict_entities(text, labels, threshold=floor)

        step = max(1, window - self._window_overlap)
        out: list[dict[str, Any]] = []
        for begin in range(0, len(words), step):
            span = words[begin : begin + window]
            if not span:
                break
            lo, hi = span[0][1], span[-1][2]
            for p in model.predict_entities(text[lo:hi], labels, threshold=floor):
                p = dict(p)
                p["start"] += lo
                p["end"] += lo
                p["text"] = text[p["start"] : p["end"]]
                out.append(p)
            if begin + window >= len(words):
                break

        merged = self._merge(out)
        logger.debug(
            "GLiNER windowed inference: %d word-tokens -> %d windows, "
            "%d raw predictions -> %d after merge",
            len(words),
            (len(words) - 1) // step + 1,
            len(out),
            len(merged),
        )
        return merged

    async def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        source_chunk_id: str,
    ) -> list[ExtractedEntity]:
        raw = await asyncio.to_thread(self._predict_sync, text, entity_types)
        return _parse_predictions(raw, entity_types, source_chunk_id, self._threshold)


# ── LLM Extractor ────────────────────────────────────────────────


class LLMExtractor(EntityExtractor):
    """Entity extraction via LLM using structured NER prompt.

    Uses ``NER_PROMPT`` to ask the LLM for entities with confidence
    and character spans.

    Args:
        llm: LLMInterface instance.
        threshold: Confidence threshold (0-1). Below this → "Unknown".
    """

    def __init__(self, llm: LLMInterface, threshold: float = 0.75) -> None:
        self._llm = llm
        self._threshold = threshold

    async def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        source_chunk_id: str,
    ) -> list[ExtractedEntity]:
        prompt = NER_PROMPT.format(
            entity_types=", ".join(entity_types),
            text=text,
        )
        response = await self._llm.ainvoke(prompt)
        return self._parse_response(
            response.content, entity_types, source_chunk_id, self._threshold
        )

    @staticmethod
    def _parse_response(
        content: str,
        entity_types: list[str],
        source_chunk_id: str,
        threshold: float = 0.75,
    ) -> list[ExtractedEntity]:
        """Parse JSON array of entities from LLM response."""
        text = _strip_markdown_fences(content)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("LLM NER returned invalid JSON, skipping chunk %s", source_chunk_id)
            return []

        if not isinstance(data, list):
            data = data.get("entities", []) if isinstance(data, dict) else []

        entities: list[ExtractedEntity] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not is_valid_entity_name(name, entity_types):
                continue
            raw_type = str(item.get("type", "")).strip()
            description = str(item.get("description", "")).strip()

            confidence = _parse_confidence(item)
            if confidence is not None and confidence < threshold:
                etype = UNKNOWN_LABEL
            else:
                etype = label_for_type(raw_type, entity_types)

            extra: dict[str, Any] = {}
            spans = _build_spans(item, source_chunk_id)
            if spans:
                extra["spans"] = spans
            if confidence is not None:
                extra["confidence"] = confidence

            entities.append(
                ExtractedEntity(
                    name=name,
                    type=etype,
                    description=description,
                    source_chunk_ids=[source_chunk_id],
                    **extra,
                )
            )
        return entities


# ── spaCy Extractor ──────────────────────────────────────────────


class SpacyExtractor(EntityExtractor):
    """Classic (non zero-shot) NER via spaCy, for well-known proper nouns.

    This exists to cover a measured blind spot rather than as a general
    extractor. Swapping the GLiNER default to ``gliner-bi-small-v2.0`` gained 49
    gold entities but *lost* 24, and the losses were textbook proper nouns —
    ``Baghdad``, ``Cairo``, ``Madrid``, ``Barcelona``, ``Constantinople``,
    ``Paris Observatory`` — which a supervised model trained on exactly those
    categories gets right trivially.

    Used alone it is a poor fit for GraphRAG: its label set is fixed, so it
    cannot represent custom types such as ``Method`` or ``Technology``. Its
    value is as the second half of a :class:`CompositeExtractor`.

    ``DEFAULT_LABELS`` is deliberately narrow. Measured on an 11-document
    benchmark (157 chunks), unioned with the default GLiNER extractor:

    ===========================  ======  =========  =====
    spaCy labels added           recall  precision  F1
    ===========================  ======  =========  =====
    none (GLiNER alone)          0.494   0.554      0.522
    PERSON/ORG/GPE/FAC           0.613   0.499      0.550
    + LOC                        0.616   0.497      0.550
    + NORP                       0.616   0.477      0.538
    all 12 usable labels         0.654   0.352      0.458
    ===========================  ======  =========  =====

    Widening past the four default labels buys recall by giving away
    precision, and F1 falls off a cliff at the wide setting. The narrow set
    recovers 18 of the 24 lost entities for ~5s per 157 chunks.

    Requires the ``spacy`` extra and a downloaded model::

        pip install "graphrag-sdk[spacy]"
        python -m spacy download en_core_web_lg
    """

    #: spaCy labels kept by default. Everything else is dropped rather than
    #: guessed at, because a wrong type is worse than a missing entity here.
    DEFAULT_LABELS = frozenset({"PERSON", "ORG", "GPE", "FAC"})

    #: spaCy label -> the generic type name we look for in ``entity_types``.
    #: Candidates are tried in order and the first one the caller allows wins,
    #: so this works whether a schema calls it ``Place``, ``Location`` or
    #: ``Organization``.
    LABEL_MAP: dict[str, tuple[str, ...]] = {
        "PERSON": ("Person",),
        "ORG": ("Organization", "Institution", "Company"),
        "GPE": ("Location", "Place", "GeographicLocation"),
        "LOC": ("Location", "Place", "GeographicLocation"),
        "FAC": ("Building", "Facility", "Location", "Place"),
        "NORP": ("Group", "Nationality", "Organization"),
        "EVENT": ("Event",),
        "PRODUCT": ("Product",),
        "WORK_OF_ART": ("Work", "Publication", "Product"),
        "LAW": ("Law",),
        "DATE": ("Date",),
        "LANGUAGE": ("Language",),
    }

    DEFAULT_MODEL = "en_core_web_lg"

    #: Confidence recorded on every emitted entity. Pinned to the threshold of
    #: the default :class:`GLiNERExtractor` model so that, in the documented
    #: ``CompositeExtractor([GLiNERExtractor(), SpacyExtractor()])`` pairing, a
    #: ``confidence >= threshold`` filter on the graph keeps spaCy's entities
    #: alongside GLiNER's instead of silently dropping them.
    DEFAULT_CONFIDENCE: float = GLiNERExtractor.DEFAULT_THRESHOLDS[GLiNERExtractor.DEFAULT_MODEL]

    #: Pipeline components that cost time and contribute nothing to ``ents``.
    #: Only ``tok2vec`` and ``ner`` stay active. spaCy ignores names that a
    #: given pipeline does not have, so this is safe for custom models too.
    DISABLED_COMPONENTS: tuple[str, ...] = ("tagger", "parser", "attribute_ruler", "lemmatizer")

    def __init__(
        self,
        model_name: str | None = None,
        labels: Iterable[str] | None = None,
        confidence: float | None = None,
    ) -> None:
        """Initialise the extractor.

        Args:
            model_name: spaCy pipeline to load. Defaults to ``en_core_web_lg``.
                ``en_core_web_sm`` is ~2.6x smaller and just as fast but
                measured 0.554 ceiling recall against 0.601 for ``lg``.
            labels: spaCy entity labels to keep. Defaults to
                :attr:`DEFAULT_LABELS`. Widening this reduces F1 — see the
                class docstring for the measured trade. A bare string is
                rejected rather than being iterated character by character.
            confidence: Score recorded on emitted entities. spaCy's ``ents``
                expose no per-span probability, so this is a fixed stand-in
                rather than a real confidence and has no effect on *which*
                entities are emitted. Defaults to :attr:`DEFAULT_CONFIDENCE`,
                the default GLiNER model's threshold (0.75), so the recorded
                value lines up with GLiNER's under the default pairing. Pass
                the matching threshold when pairing with a different GLiNER
                model.

        Raises:
            TypeError: If ``labels`` is a single string.
        """
        if isinstance(labels, str):
            raise TypeError(
                f"labels must be an iterable of spaCy labels, not a string; "
                f"got {labels!r}. Did you mean labels=[{labels!r}]?"
            )
        self._model_name = model_name or self.DEFAULT_MODEL
        self._labels = frozenset(labels) if labels is not None else self.DEFAULT_LABELS
        self._confidence = self.DEFAULT_CONFIDENCE if confidence is None else confidence
        self._nlp: Any = None
        self._load_error: Exception | None = None

    async def _get_nlp(self) -> Any:
        if self._nlp is not None:
            return self._nlp
        # A missing package or model is permanent for the life of this process;
        # remember it so every chunk does not re-attempt the load in a thread.
        if self._load_error is not None:
            raise self._load_error
        try:
            self._nlp = await asyncio.to_thread(self._get_shared_nlp, self._model_name)
        except (ImportError, OSError) as exc:
            self._load_error = exc
            raise
        return self._nlp

    # Process-wide cache of loaded spaCy pipelines, keyed on model name.
    #
    # Mirrors ``GLiNERExtractor._MODEL_CACHE`` for the same reason: extractors
    # are constructed ad hoc (one per ``GraphExtraction``, one per discovery
    # pipeline), and ``en_core_web_lg`` is several hundred MB resident, most of
    # it the 685k x 300 word-vector table. Without sharing, one pipeline per
    # document pays that per document.
    _MODEL_CACHE: ClassVar[dict[str, Any]] = {}
    _CACHE_LOCK: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def _get_shared_nlp(cls, model_name: str) -> Any:
        cached = cls._MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached
        with cls._CACHE_LOCK:
            # Double-checked: another thread may have loaded it while we waited.
            cached = cls._MODEL_CACHE.get(model_name)
            if cached is None:
                cached = cls._load(model_name)
                cls._MODEL_CACHE[model_name] = cached
        return cached

    @classmethod
    def _load(cls, model_name: str) -> Any:
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "SpacyExtractor requires the 'spacy' extra. Install with: "
                'pip install "graphrag-sdk[spacy]"'
            ) from exc
        try:
            return spacy.load(model_name, disable=list(cls.DISABLED_COMPONENTS))
        except OSError as exc:  # pragma: no cover - depends on env
            raise OSError(
                f"spaCy model '{model_name}' is not installed. Run: "
                f"python -m spacy download {model_name}"
            ) from exc

    def _type_for(self, label: str, entity_types: list[str]) -> str | None:
        """Map a spaCy label onto an allowed type, or None if unrepresentable."""
        for candidate in self.LABEL_MAP.get(label, ()):
            mapped = label_for_type(candidate, entity_types)
            if mapped != UNKNOWN_LABEL:
                return mapped
        return None

    async def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        source_chunk_id: str,
    ) -> list[ExtractedEntity]:
        nlp = await self._get_nlp()
        # Deliberately no lock: spaCy inference on a shared ``Language`` is
        # thread-safe in practice. Verified with 12 threads sharing one ``nlp``
        # over 400 docs (the pipeline's ``Semaphore(12)`` + ``to_thread`` shape)
        # against a serialised run: 0 mismatches.
        doc = await asyncio.to_thread(nlp, text)

        preds: list[dict[str, Any]] = []
        for ent in doc.ents:
            if ent.label_ not in self._labels:
                continue
            etype = self._type_for(ent.label_, entity_types)
            if etype is None:
                continue  # caller's schema has no home for this label
            preds.append(
                {
                    "text": ent.text,
                    "label": etype,
                    "score": self._confidence,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
            )
        # threshold=0.0: spaCy gives no real scores, so there is nothing to
        # demote on. The recorded score is a fixed stand-in (see __init__).
        return _parse_predictions(preds, entity_types, source_chunk_id, threshold=0.0)


# ── Composite Extractor ──────────────────────────────────────────


class CompositeExtractor(EntityExtractor):
    """Run several extractors over the same text and merge their entities.

    Built for the measured fact that no single extractor we tested wins
    everywhere: ``gliner-bi-small-v2.0`` is far stronger on multi-word,
    domain-specific entities (``Fresnel lens``, ``differential gear train
    mechanism``, ``Samarkand Expedition of 892``), while a supervised spaCy
    pipeline is stronger on plain proper nouns (``Baghdad``, ``Madrid``,
    ``Paris Observatory``). Combining them recovered 18 of the 24 entities lost
    in the model swap.

    Measured on an 11-document benchmark, 157 chunks, GLiNER default plus
    ``SpacyExtractor`` at its default labels: recall 0.494 -> 0.613, F1
    0.522 -> 0.550, for about 5 extra seconds. Precision falls 0.554 -> 0.499,
    so this trades some precision for a larger gain in recall.

    Extractors run concurrently. Duplicates are resolved by normalised name:
    the **earliest** extractor in the list wins the type, which makes ordering
    meaningful — put your most trusted or most schema-aware extractor first.
    Chunk provenance and character spans are unioned, so a merged entity keeps
    every chunk and every occurrence it came from, including repeat mentions
    the same extractor reported as separate entities.

    A failing extractor does not take the others down: the exception is logged
    and its results are skipped, on the grounds that degraded extraction beats
    a failed ingest. If *every* extractor fails the error is re-raised. Two
    kinds of failure are never demoted to a per-chunk warning, because they
    are configuration faults rather than bad input and would otherwise repeat
    silently on every chunk: ``ImportError`` (a missing optional dependency)
    and ``OSError`` (a model that is not installed). Cancellation and other
    process-level ``BaseException`` subclasses propagate as well.

    ``LLMExtractor`` members are rejected. ``GraphExtraction`` dispatches on
    the extractor type to batch NER prompts through ``abatch_invoke`` with the
    caller's ``max_concurrency`` and ontology type descriptions; wrapped in a
    composite it would silently fall back to one un-batched ``ainvoke`` per
    chunk with a poorer prompt and N-times the configured concurrency.

    Example::

        extractor = CompositeExtractor([
            GLiNERExtractor(),
            SpacyExtractor(),
        ])
    """

    def __init__(
        self,
        extractors: Sequence[EntityExtractor],
        suppress_overlaps: bool = True,
    ) -> None:
        """Initialise the extractor.

        Args:
            extractors: Extractors to run, in priority order. The first one to
                produce a given name decides its type.
            suppress_overlaps: Resolve entities whose character spans collide
                across extractors. Without this, merging two NER systems
                reliably produces near-duplicate fragments — measured on one
                sentence, spaCy contributed ``Fresnel`` (typed
                ``Organization``) next to GLiNER's ``Fresnel lens``, and ``The
                Paris Observatory`` next to ``Paris Observatory``. Those
                fragments are false positives and inflate the entity count
                without adding knowledge. When one span strictly contains the
                other and adds a content word, the longer span wins regardless
                of which extractor produced it, so a later extractor's ``Paris
                Observatory`` replaces an earlier ``Paris`` at the same
                position (but leaves that entity's other, non-nested
                occurrences alone). An extension that only adds an article —
                ``The Paris Observatory`` over ``Paris Observatory`` — is the
                same entity, not a better one, and the earlier extractor
                wins, as it does when spans merely partially overlap or are
                identical under different names. Entities without span
                information are always kept, since there is no evidence on
                which to drop them.

        Raises:
            ValueError: If ``extractors`` is empty.
            TypeError: If any member (or nested composite member) is an
                ``LLMExtractor``; see the class docstring for why.
        """
        if not extractors:
            raise ValueError("CompositeExtractor requires at least one extractor")
        for member in self._flatten(extractors):
            if isinstance(member, LLMExtractor):
                raise TypeError(
                    "CompositeExtractor does not accept LLMExtractor members: "
                    "GraphExtraction batches LLM NER through abatch_invoke only when "
                    "the extractor itself is an LLMExtractor, so wrapping one would "
                    "silently lose batching, max_concurrency and entity-type descriptions. "
                    "Use LLMExtractor directly, or compose local extractors only."
                )
        self._extractors = list(extractors)
        self._suppress_overlaps = suppress_overlaps

    @classmethod
    def _flatten(cls, extractors: Iterable[EntityExtractor]) -> Iterable[EntityExtractor]:
        for e in extractors:
            if isinstance(e, CompositeExtractor):
                yield from cls._flatten(e._extractors)
            else:
                yield e

    @staticmethod
    def _spans_dict(ent: ExtractedEntity) -> dict[str, list[dict[str, Any]]] | None:
        """The live ``{chunk_id: [{"start", "end"}, ...]}`` mapping of an entity.

        ``_parse_predictions`` passes ``spans`` as an extra model field rather
        than into ``attributes``, so check both.
        """
        spans = getattr(ent, "spans", None)
        if spans is None:
            spans = ent.attributes.get("spans")
        return spans if isinstance(spans, dict) else None

    @classmethod
    def _spans_of(cls, ent: ExtractedEntity) -> list[tuple[int, int]]:
        """Character spans claimed by an entity, flattened across chunks."""
        spans = cls._spans_dict(ent)
        out: list[tuple[int, int]] = []
        if spans:
            for items in spans.values():
                for sp in items or ():
                    try:
                        out.append((int(sp["start"]), int(sp["end"])))
                    except (KeyError, TypeError, ValueError):
                        continue
        return out

    @classmethod
    def _remove_span(cls, ent: ExtractedEntity, span: tuple[int, int]) -> None:
        """Drop one ``(start, end)`` occurrence from an entity's span mapping."""
        spans = cls._spans_dict(ent)
        if not spans:
            return
        start, end = span
        for chunk_id in list(spans):
            kept = [
                sp
                for sp in spans[chunk_id] or ()
                if not (isinstance(sp, dict) and sp.get("start") == start and sp.get("end") == end)
            ]
            if kept:
                spans[chunk_id] = kept
            else:
                del spans[chunk_id]

    @classmethod
    def _merge_spans(cls, into: ExtractedEntity, ent: ExtractedEntity) -> None:
        """Union ``ent``'s spans into ``into`` without duplicating offsets."""
        src = cls._spans_dict(ent)
        if not src:
            return
        dst = cls._spans_dict(into)
        if dst is None:
            dst = {}
            into.spans = dst  # type: ignore[attr-defined]
        for chunk_id, offsets in src.items():
            bucket = dst.setdefault(chunk_id, [])
            for sp in offsets or ():
                if sp not in bucket:
                    bucket.append(sp)

    async def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        source_chunk_id: str,
    ) -> list[ExtractedEntity]:
        results = await asyncio.gather(
            *(e.extract_entities(text, entity_types, source_chunk_id) for e in self._extractors),
            return_exceptions=True,
        )

        merged: dict[str, ExtractedEntity] = {}
        # Spans claimed by earlier extractors -> the merged key that owns each.
        claimed: dict[tuple[int, int], str] = {}
        failures: list[Exception] = []
        for extractor, result in zip(self._extractors, results, strict=True):
            if isinstance(result, BaseException):
                if not isinstance(result, Exception) or isinstance(result, (ImportError, OSError)):
                    # Cancellation / interpreter exit, or a configuration
                    # fault that would otherwise repeat on every chunk.
                    raise result
                failures.append(result)
                logger.warning(
                    "%s failed during entity extraction, skipping its results: %s",
                    type(extractor).__name__,
                    result,
                )
                continue
            fresh: dict[tuple[int, int], str] = {}
            for ent in result:
                key = ent.name.strip().casefold()
                if not key:
                    continue
                spans = self._spans_of(ent)
                if self._suppress_overlaps and spans:
                    admitted = self._admit_spans(ent, key, spans, claimed, merged, text, extractor)
                    if not admitted:
                        continue  # every occurrence was a fragment of something claimed
                    spans = admitted
                existing = merged.get(key)
                if existing is None:
                    merged[key] = ent
                    for sp in spans:
                        fresh[sp] = key
                    continue
                # Keep the earlier extractor's type; only fill genuine gaps.
                if existing.type == UNKNOWN_LABEL and ent.type != UNKNOWN_LABEL:
                    existing.type = ent.type
                if not existing.description and ent.description:
                    existing.description = ent.description
                for cid in ent.source_chunk_ids:
                    if cid not in existing.source_chunk_ids:
                        existing.source_chunk_ids.append(cid)
                # Repeat occurrences keep their offsets, and claim them, so a
                # later fragment overlapping the *second* mention is caught too.
                self._merge_spans(existing, ent)
                for sp in spans:
                    fresh[sp] = key
            # Only claim spans once the whole extractor is processed, so two
            # entities from the SAME extractor never suppress each other.
            claimed.update(fresh)

        if failures and len(failures) == len(self._extractors):
            raise failures[0]
        return list(merged.values())

    #: Tokens that, on their own, do not make a longer span a *better* entity.
    #: ``The Paris Observatory`` over ``Paris Observatory`` is the same entity
    #: with an article, not a recovered one, so it does not supersede.
    _TRIVIAL_EXTENSION_TOKENS: ClassVar[frozenset[str]] = frozenset({"the", "a", "an"})

    @classmethod
    def _extends_meaningfully(
        cls, text: str, outer: tuple[int, int], inner: tuple[int, int]
    ) -> bool:
        """True if ``outer`` adds at least one content token around ``inner``."""
        (s, e), (cs, ce) = outer, inner
        extra = f"{text[s:cs]} {text[ce:e]}"
        tokens = re.findall(r"\w+", extra.casefold())
        return any(tok not in cls._TRIVIAL_EXTENSION_TOKENS for tok in tokens)

    def _admit_spans(
        self,
        ent: ExtractedEntity,
        key: str,
        spans: list[tuple[int, int]],
        claimed: dict[tuple[int, int], str],
        merged: dict[str, ExtractedEntity],
        text: str,
        extractor: EntityExtractor,
    ) -> list[tuple[int, int]]:
        """Decide which of ``ent``'s spans survive against already-claimed ones.

        Spans owned by the same normalised name never conflict; they are simply
        further occurrences. Against other names, per span: dropped if a
        claimed span contains it, equals it, or partly overlaps it (earlier
        extractor wins); kept if it strictly contains one or more claimed spans
        *and* adds a content token beyond them, in which case those fragments
        are evicted from their owners (an owner left with no spans is removed
        entirely). Dropped spans are also removed from ``ent`` so its
        provenance matches what was actually admitted.
        """
        admitted: list[tuple[int, int]] = []
        for s, e in spans:
            blocked = False
            nested: list[tuple[int, int]] = []
            for (cs, ce), owner_key in claimed.items():
                if owner_key == key or not (s < ce and cs < e):
                    continue
                if cs <= s and e <= ce:
                    blocked = True  # contained in, or identical to, a claimed span
                    break
                if s <= cs and ce <= e and self._extends_meaningfully(text, (s, e), (cs, ce)):
                    nested.append((cs, ce))
                else:
                    blocked = True  # partial overlap, or an article-only extension
                    break
            if blocked:
                logger.debug(
                    "%s: dropping %r at %d-%d, overlaps an entity claimed earlier",
                    type(extractor).__name__,
                    ent.name,
                    s,
                    e,
                )
                self._remove_span(ent, (s, e))
                continue
            for frag in nested:
                owner_key = claimed.pop(frag)
                owner = merged.get(owner_key)
                if owner is None:
                    continue
                logger.debug(
                    "%s: %r at %d-%d supersedes fragment %r at %d-%d",
                    type(extractor).__name__,
                    ent.name,
                    s,
                    e,
                    owner.name,
                    frag[0],
                    frag[1],
                )
                self._remove_span(owner, frag)
                if not self._spans_of(owner):
                    del merged[owner_key]
            admitted.append((s, e))
        return admitted
