# GraphRAG SDK 2.0 — Extraction: Pluggable Entity Extractor
# Single class that handles multiple NER backends:
# - Default: GLiNER2 (local, fast, no API calls)
# - LLM: Uses NER_PROMPT template with any LLMInterface
# - Custom: Any model implementing predict_entities(text, labels)
#
# Also exports shared entity utilities (constants, ID computation,
# name validation, type mapping) used by TwoStepExtraction.

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Protocol, runtime_checkable

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

MIN_NAME_LEN = 2   # single-char names are noise
MAX_NAME_LEN = 80  # descriptions masquerading as names

# Pronouns and generic references that should never become entities
_PRONOUNS: set[str] = {
    "he", "she", "they", "it", "him", "her", "his", "them", "who", "whom",
    "i", "we", "you", "one", "me", "us", "my", "our", "your", "their",
    "hers", "its",
}

_ENTITY_STOPLIST: set[str] = _PRONOUNS | {
    # Generic/anonymous references
    "narrator", "the narrator", "author", "the author", "reader", "the reader",
    "speaker", "the speaker", "listener", "the listener",
    "the man", "the woman", "the boy", "the girl", "the child",
    "man", "woman", "boy", "girl", "child",
    "people", "person", "someone", "somebody", "everyone", "everybody",
    "mistress", "master",
    # Meta-textual
    "story", "chapter", "passage", "book", "text", "narrative",
    "paragraph", "section", "document",
}


# ── Entity Utility Functions ─────────────────────────────────────


def compute_entity_id(name: str, entity_type: str = "") -> str:
    """Deterministic entity ID from normalized name and optional type.

    When ``entity_type`` is provided, a ``__type`` suffix is appended to
    prevent cross-type collisions (e.g. Person "Paris" vs Location "Paris"
    produce different IDs: ``paris__person`` vs ``paris__location``).
    """
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
    """Map a raw type string to the closest allowed type, or UNKNOWN_LABEL.

    Performs case-insensitive matching against the normalized forms of
    allowed_types.
    """
    if not raw_type or not raw_type.strip():
        return UNKNOWN_LABEL

    norm = _normalize_type_label(raw_type)
    for allowed in allowed_types:
        if _normalize_type_label(allowed) == norm:
            return allowed
    return UNKNOWN_LABEL


# ── NER Prompt ───────────────────────────────────────────────────

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


# ── NER Model Protocol ───────────────────────────────────────────


@runtime_checkable
class NERModel(Protocol):
    """Protocol for custom NER models.

    Any model that implements ``predict_entities(text, labels)``
    returning a list of dicts with ``text`` and ``label`` keys.
    """

    def predict_entities(
        self, text: str, labels: list[str], **kwargs: Any
    ) -> list[dict[str, Any]]: ...


# ── Entity Extractor ─────────────────────────────────────────────


class EntityExtractor:
    """Single entity extractor supporting multiple backends.

    By default uses GLiNER2 for local NER (no API calls, fast).
    Pass an ``LLMInterface`` to use LLM-based NER prompts instead.
    Pass any model implementing ``predict_entities(text, labels)``
    for custom backends (spaCy, custom transformers, etc.).

    Args:
        model: A NER model implementing ``predict_entities(text, labels)``.
            Default: GLiNER2 (lazy-loaded).
        llm: An LLMInterface for LLM-based NER. Takes precedence if both
            ``model`` and ``llm`` are provided.
        threshold: Confidence threshold (0-1). Entities below this are
            labeled "Unknown". Applied to all backends (GLiNER2, LLM, custom).
        gliner_model_name: GLiNER2 model name (only used when neither
            ``model`` nor ``llm`` is provided).
    """

    def __init__(
        self,
        *,
        model: Any | None = None,
        llm: LLMInterface | None = None,
        threshold: float = 0.75,
        gliner_model_name: str = "urchade/gliner_medium-v2.1",
    ) -> None:
        self._llm = llm
        self._custom_model = model
        self._threshold = threshold
        self._gliner_model_name = gliner_model_name
        self._gliner_model: Any = None

        # Determine mode
        if llm is not None:
            self._mode = "llm"
        elif model is not None:
            self._mode = "custom"
        else:
            self._mode = "gliner2"

    @property
    def mode(self) -> str:
        """Return the active backend mode: 'llm', 'custom', or 'gliner2'."""
        return self._mode

    async def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        source_chunk_id: str,
    ) -> list[ExtractedEntity]:
        """Extract entities from a single chunk of text.

        Routes to the appropriate backend based on how the extractor
        was constructed.
        """
        if self._mode == "llm":
            return await self._extract_llm(text, entity_types, source_chunk_id)
        elif self._mode == "custom":
            return await self._extract_custom(text, entity_types, source_chunk_id)
        else:
            return await self._extract_gliner2(text, entity_types, source_chunk_id)

    # ── LLM backend ──────────────────────────────────────────────

    async def _extract_llm(
        self,
        text: str,
        entity_types: list[str],
        source_chunk_id: str,
    ) -> list[ExtractedEntity]:
        assert self._llm is not None
        prompt = NER_PROMPT.format(
            entity_types=", ".join(entity_types),
            text=text,
        )
        response = await self._llm.ainvoke(prompt)
        return self._parse_llm_response(
            response.content, entity_types, source_chunk_id, self._threshold
        )

    @staticmethod
    def _parse_llm_response(
        content: str,
        entity_types: list[str],
        source_chunk_id: str,
        threshold: float = 0.75,
    ) -> list[ExtractedEntity]:
        """Parse JSON array of entities from LLM response.

        Extracts confidence and character spans (start/end) when provided.
        Entities with confidence below ``threshold`` are labeled "Unknown".
        """
        text = content.strip()
        # Strip markdown fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

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

            # Confidence: low → Unknown
            confidence = None
            try:
                confidence = float(item["confidence"])
            except (KeyError, ValueError, TypeError):
                pass

            if confidence is not None and confidence < threshold:
                etype = UNKNOWN_LABEL
            else:
                etype = label_for_type(raw_type, entity_types)

            # Spans: character offsets in the chunk
            spans: dict[str, list[dict[str, int]]] = {}
            start = item.get("start")
            end = item.get("end")
            if start is not None and end is not None:
                try:
                    spans[source_chunk_id] = [{"start": int(start), "end": int(end)}]
                except (ValueError, TypeError):
                    pass

            extra: dict[str, Any] = {}
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

    # ── GLiNER2 backend ──────────────────────────────────────────

    def _load_gliner2(self) -> Any:
        if self._gliner_model is None:
            try:
                from gliner import GLiNER
            except ImportError:
                raise ImportError(
                    "GLiNER is required for the default entity extractor. "
                    "Install with: pip install gliner\n"
                    "Or pass llm= to use LLM-based extraction instead."
                )
            self._gliner_model = GLiNER.from_pretrained(self._gliner_model_name)
        return self._gliner_model

    def _predict_gliner2_sync(
        self, text: str, entity_types: list[str]
    ) -> list[dict[str, Any]]:
        model = self._load_gliner2()
        labels = [t.lower() for t in entity_types]
        return model.predict_entities(text, labels, threshold=self._threshold)

    async def _extract_gliner2(
        self,
        text: str,
        entity_types: list[str],
        source_chunk_id: str,
    ) -> list[ExtractedEntity]:
        raw = await asyncio.to_thread(self._predict_gliner2_sync, text, entity_types)
        return self._parse_gliner2_response(raw, entity_types, source_chunk_id)

    def _parse_gliner2_response(
        self,
        predictions: list[dict[str, Any]],
        entity_types: list[str],
        source_chunk_id: str,
    ) -> list[ExtractedEntity]:
        """Parse GLiNER predict_entities response.

        Each prediction: {"text": "...", "label": "...", "score": 0.95,
                          "start": 0, "end": 5}

        Entities below threshold are labeled "Unknown".
        Character spans stored as ``{chunk_id: [{"start": N, "end": M}]}``.
        """
        entities: list[ExtractedEntity] = []
        for pred in predictions:
            if not isinstance(pred, dict):
                continue
            name = str(pred.get("text", "")).strip()
            if not is_valid_entity_name(name):
                continue

            confidence = float(pred.get("score", 0.0))
            raw_type = str(pred.get("label", "")).strip()
            start = pred.get("start")
            end = pred.get("end")

            # Low confidence → Unknown
            if confidence >= self._threshold:
                etype = label_for_type(raw_type, entity_types)
            else:
                etype = UNKNOWN_LABEL

            # Build spans dict: {chunk_id: [{start, end}]}
            spans: dict[str, list[dict[str, int]]] = {}
            if start is not None and end is not None:
                spans[source_chunk_id] = [{"start": int(start), "end": int(end)}]

            entities.append(
                ExtractedEntity(
                    name=name,
                    type=etype,
                    description="",
                    source_chunk_ids=[source_chunk_id],
                    spans=spans,
                    confidence=confidence,
                )
            )
        return entities

    # ── Custom model backend ─────────────────────────────────────

    async def _extract_custom(
        self,
        text: str,
        entity_types: list[str],
        source_chunk_id: str,
    ) -> list[ExtractedEntity]:
        assert self._custom_model is not None
        labels = [t.lower() for t in entity_types]
        raw = await asyncio.to_thread(
            self._custom_model.predict_entities, text, labels
        )
        return self._parse_ner_predictions(raw, entity_types, source_chunk_id)

    # ── Shared NER prediction parser (custom models) ─────────────

    @staticmethod
    def _parse_ner_predictions(
        predictions: list[dict[str, Any]],
        entity_types: list[str],
        source_chunk_id: str,
    ) -> list[ExtractedEntity]:
        """Parse NER model predictions into ExtractedEntity objects.

        Expects predictions with 'text' and 'label' keys.
        Passes through 'confidence', 'start', 'end' if present.
        """
        entities: list[ExtractedEntity] = []
        for pred in predictions:
            name = str(pred.get("text", "")).strip()
            if not is_valid_entity_name(name):
                continue
            raw_type = str(pred.get("label", "")).strip()
            etype = label_for_type(raw_type, entity_types)

            extra: dict[str, Any] = {}
            start = pred.get("start")
            end = pred.get("end")
            if start is not None and end is not None:
                try:
                    extra["spans"] = {
                        source_chunk_id: [{"start": int(start), "end": int(end)}]
                    }
                except (ValueError, TypeError):
                    pass
            confidence = pred.get("confidence")
            if confidence is not None:
                try:
                    extra["confidence"] = float(confidence)
                except (ValueError, TypeError):
                    pass

            entities.append(
                ExtractedEntity(
                    name=name,
                    type=etype,
                    description="",
                    source_chunk_ids=[source_chunk_id],
                    **extra,
                )
            )
        return entities
