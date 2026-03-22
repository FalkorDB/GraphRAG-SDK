# GraphRAG SDK 2.0 — Extraction: Pluggable Coreference Resolvers
# ABC + concrete implementations for coreference resolution
# as an optional pre-processing step in TwoStepExtraction.

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class CorefResolver(ABC):
    """Abstract base for pluggable coreference resolution backends."""

    @abstractmethod
    async def resolve(self, text: str) -> str:
        """Resolve coreferences in text, replacing pronouns with their referents.

        Args:
            text: Input text with potential pronoun references.

        Returns:
            Text with pronouns replaced by their canonical mentions.
        """
        ...


# Pronouns eligible for replacement
_COREF_PRONOUNS: set[str] = {
    "he", "she", "it", "they", "we", "i",
    "me", "him", "her", "us", "them",
    "his", "hers", "its", "their", "our", "my",
    "this", "that", "these", "those",
    "which", "who", "whom", "what",
}

# Possessive pronouns that need 's appended to replacement
_POSSESSIVE_PRONOUNS: set[str] = {
    "his", "her", "its", "their", "our", "my", "hers",
}


class FastCorefResolver(CorefResolver):
    """Coreference resolution via fastcoref (LingMessCoref).

    Requires: ``pip install graphrag-sdk[fastcoref]``

    Uses character-offset-based replacement (not str.replace) for
    correctness. Processes spans right-to-left to preserve offsets.
    Handles possessive pronoun conversion (e.g. "her" -> "Voss's").
    """

    def __init__(self, model_name: str = "biu-nlp/lingmess-coref") -> None:
        self._model_name = model_name
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from fastcoref import LingMessCoref  # noqa: F811
        except ImportError:
            raise ImportError(
                "fastcoref is required for FastCorefResolver. "
                "Install with: pip install graphrag-sdk[fastcoref]"
            )

        # Longformer compatibility patch for transformers >= 5.0
        try:
            import transformers
            _orig_sdpa = getattr(transformers.LongformerModel, "_sdpa_can_dispatch", None)
            if _orig_sdpa is not None:
                transformers.LongformerModel._sdpa_can_dispatch = (
                    lambda self, *a, **k: False
                )
            self._model = LingMessCoref(self._model_name)
            if _orig_sdpa is not None:
                transformers.LongformerModel._sdpa_can_dispatch = _orig_sdpa
        except Exception:
            # Fallback without patching
            self._model = LingMessCoref(self._model_name)

        return self._model

    def _resolve_sync(self, text: str) -> str:
        model = self._load_model()
        preds = model.predict(texts=[text])

        clusters = preds[0].get_clusters(as_strings=False)
        if not clusters:
            return text

        # Build list of (start, end, replacement) for pronoun spans
        replacements: list[tuple[int, int, str]] = []

        for cluster in clusters:
            # Find canonical mention: longest non-pronoun span
            canonical = None
            for start, end in cluster:
                mention = text[start:end]
                if mention.lower().strip() not in _COREF_PRONOUNS:
                    if canonical is None or (end - start) > (canonical[1] - canonical[0]):
                        canonical = (start, end)

            if canonical is None:
                continue

            canonical_text = text[canonical[0]:canonical[1]]

            for start, end in cluster:
                mention = text[start:end]
                mention_lower = mention.lower().strip()

                # Only replace pronouns, skip noun phrases
                if mention_lower not in _COREF_PRONOUNS:
                    continue

                # Handle possessive: "her" -> "Voss's"
                if mention_lower in _POSSESSIVE_PRONOUNS:
                    if canonical_text.endswith("s"):
                        replacement = f"{canonical_text}'"
                    else:
                        replacement = f"{canonical_text}'s"
                else:
                    replacement = canonical_text

                replacements.append((start, end, replacement))

        if not replacements:
            return text

        # Sort right-to-left to preserve earlier offsets
        replacements.sort(key=lambda x: x[0], reverse=True)

        result = text
        for start, end, replacement in replacements:
            result = result[:start] + replacement + result[end:]

        return result

    async def resolve(self, text: str) -> str:
        return await asyncio.to_thread(self._resolve_sync, text)
