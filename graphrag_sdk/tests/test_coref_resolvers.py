"""Tests for coref_resolvers.py — pluggable coreference resolution."""

from __future__ import annotations

import pytest

from graphrag_sdk.ingestion.extraction_strategies.coref_resolvers import (
    CorefResolver,
    FastCorefResolver,
    _COREF_PRONOUNS,
    _POSSESSIVE_PRONOUNS,
)


class TestCorefResolverABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            CorefResolver()  # type: ignore[abstract]


class TestCorefPronounSets:
    def test_common_pronouns_present(self):
        for p in ["he", "she", "it", "they", "him", "her", "them"]:
            assert p in _COREF_PRONOUNS

    def test_possessives_are_subset_of_pronouns(self):
        assert _POSSESSIVE_PRONOUNS.issubset(_COREF_PRONOUNS)


class TestFastCorefResolver:
    async def test_import_error_message(self):
        """FastCorefResolver raises clear ImportError if fastcoref not installed."""
        try:
            import fastcoref  # noqa: F401
            pytest.skip("fastcoref is installed")
        except ImportError:
            resolver = FastCorefResolver()
            with pytest.raises(ImportError, match="fastcoref"):
                await resolver.resolve("She went to the store.")

    def test_constructor_accepts_model_name(self):
        resolver = FastCorefResolver(model_name="custom-model")
        assert resolver._model_name == "custom-model"
