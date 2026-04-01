# GraphRAG SDK 2.0 — Extraction: Entity Extractors
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
from typing import Any

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

_ENTITY_STOPLIST: set[str] = _PRONOUNS | {
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


def is_valid_entity_name(name: str) -> bool:
    """Return True if name passes quality gates for entity extraction."""
    if not name or not name.strip():
        return False
    stripped = name.strip()
    if len(stripped) < MIN_NAME_LEN or len(stripped) > MAX_NAME_LEN:
        return False
    if stripped.lower() in _ENTITY_STOPLIST:
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
        if not is_valid_entity_name(name):
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

    Args:
        threshold: Confidence threshold (0-1). Below this → "Unknown".
        model_name: HuggingFace model name for GLiNER.
    """

    def __init__(
        self,
        threshold: float = 0.75,
        model_name: str = "urchade/gliner_medium-v2.1",
    ) -> None:
        self._threshold = threshold
        self._model_name = model_name
        self._model: Any = None
        self._lock = threading.Lock()

    def _load_model(self) -> Any:
        if self._model is None:
            with self._lock:
                # Double-check after acquiring lock
                if self._model is None:
                    import os
                    # Must be set before HuggingFace tokenizers initialise to
                    # prevent deadlocks when asyncio.to_thread is used later.
                    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                    try:
                        from gliner import GLiNER
                    except ImportError:
                        raise ImportError(
                            "GLiNER is required for GLiNERExtractor. "
                            "Install with: pip install gliner"
                        )
                    logger.info("Loading GLiNER model '%s'…", self._model_name)
                    self._model = GLiNER.from_pretrained(self._model_name)
                    logger.info("GLiNER model loaded.")
        return self._model

    def warm_up(self) -> None:
        """Pre-load the GLiNER model on the calling thread.

        Call this once on the main thread before launching concurrent
        ``extract_entities`` tasks so that model loading never races with
        HuggingFace tokenizer parallelism inside worker threads.
        """
        self._load_model()

    def _predict_sync(self, text: str, entity_types: list[str]) -> list[dict[str, Any]]:
        import time as _time
        model = self._load_model()
        labels = [t.lower() for t in entity_types]
        logger.debug("GLiNER predict start (text_len=%d)", len(text))
        t0 = _time.monotonic()
        with self._lock:
            result = model.predict_entities(text, labels, threshold=self._threshold)
        logger.debug(
            "GLiNER predict done (text_len=%d, entities=%d, elapsed=%.1fs)",
            len(text),
            len(result),
            _time.monotonic() - t0,
        )
        return result

    async def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        source_chunk_id: str,
    ) -> list[ExtractedEntity]:
        logger.info("GLiNER NER → chunk %s (text_len=%d)", source_chunk_id, len(text))
        raw = await asyncio.to_thread(self._predict_sync, text, entity_types)
        entities = _parse_predictions(raw, entity_types, source_chunk_id, self._threshold)
        logger.info(
            "GLiNER NER ✓ chunk %s → %d entities", source_chunk_id, len(entities)
        )
        return entities


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
            if not is_valid_entity_name(name):
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
