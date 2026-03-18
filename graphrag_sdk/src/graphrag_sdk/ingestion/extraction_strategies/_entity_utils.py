# GraphRAG SDK 2.0 — Extraction: Shared Entity Utilities
# Consolidated helpers for entity ID computation, name validation,
# and type label normalization used across all extraction strategies.

from __future__ import annotations

import re


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


def compute_entity_id(name: str, entity_type: str = "") -> str:
    """Deterministic entity ID from normalized name and optional type.

    When ``entity_type`` is provided, a ``__type`` suffix is appended to
    prevent cross-type collisions (e.g. Person "Paris" vs Location "Paris"
    produce different IDs: ``paris__person`` vs ``paris__location``).

    When ``entity_type`` is empty, returns just the normalized name for
    backwards compatibility.
    """
    base = name.strip().lower().replace(" ", "_")
    if entity_type:
        return f"{base}__{entity_type.strip().lower()}"
    return base


def _normalize_type_label(raw: str) -> str:
    """Normalize type string by lowercasing and removing separators.

    Collapses trivial formatting variants:
    "Data Type" / "DataType" / "data_type" / "data-type" -> "datatype"
    """
    s = raw.strip().lower()
    return re.sub(r"[\s_\-/]+", "", s)


def _to_title_case(raw: str) -> str:
    """Convert any type label variant to Title Case.

    Handles CamelCase, snake_case, kebab-case, and already-spaced labels:
    "celestialBody" -> "Celestial Body"
    "data_type" -> "Data Type"
    "PERSON" -> "Person"
    "literary work" -> "Literary Work"
    """
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw.strip())  # CamelCase -> spaced
    s = re.sub(r"[_\-]+", " ", s)                           # snake_case -> spaced
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


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
    allowed_types. Returns the original (properly cased) allowed type
    if matched.
    """
    if not raw_type or not raw_type.strip():
        return UNKNOWN_LABEL

    norm = _normalize_type_label(raw_type)
    for allowed in allowed_types:
        if _normalize_type_label(allowed) == norm:
            return allowed
    return UNKNOWN_LABEL
