"""Tests for ingestion/ingestion_planner.py and GraphRAG.ingest(auto=...)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from graphrag_sdk.api.main import GraphRAG
from graphrag_sdk.core.exceptions import LatencyBudgetExceededError
from graphrag_sdk.ingestion.chunking_strategies.contextual_chunking import (
    ContextualChunking,
)
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
from graphrag_sdk.ingestion.chunking_strategies.sentence_token_cap import (
    SentenceTokenCapChunking,
)
from graphrag_sdk.ingestion.chunking_strategies.structural_chunking import (
    StructuralChunking,
)
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
    GLiNERExtractor,
    LLMExtractor,
)
from graphrag_sdk.ingestion.extraction_strategies.graph_extraction import (
    GraphExtraction,
)
from graphrag_sdk.ingestion.ingestion_planner import (
    CHUNKERS,
    EXTRACTORS,
    RESOLVERS,
    HeuristicIngestionPlanner,
    IngestionPlan,
    LLMIngestionPlanner,
    build_chunker,
    build_extractor,
    build_ingestion_strategies,
    build_resolver,
    clamp_params,
    default_plan,
    parse_plan,
)
from graphrag_sdk.ingestion.resolution_strategies.description_merge import (
    DescriptionMergeResolution,
)
from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
    ExactMatchResolution,
)
from graphrag_sdk.ingestion.resolution_strategies.llm_verified_resolution import (
    LLMVerifiedResolution,
)
from graphrag_sdk.ingestion.resolution_strategies.semantic_resolution import (
    SemanticResolution,
)

from .conftest import MockEmbedder, MockLLM

# -- IngestionPlan --


class TestIngestionPlan:
    def test_default_plan_uses_cheap_defaults(self):
        plan = default_plan()
        assert (plan.chunker, plan.extractor, plan.resolver) == (
            "sentence",
            "gliner",
            "exact",
        )

    def test_rejects_unknown_option(self):
        with pytest.raises(ValueError):
            IngestionPlan(chunker="bogus")
        with pytest.raises(ValueError):
            IngestionPlan(extractor="bogus")
        with pytest.raises(ValueError):
            IngestionPlan(resolver="bogus")

    def test_option_sets_are_expected(self):
        assert set(CHUNKERS) == {"sentence", "fixed", "structural", "contextual"}
        assert set(EXTRACTORS) == {"gliner", "llm"}
        assert set(RESOLVERS) == {
            "exact",
            "description_merge",
            "semantic",
            "llm_verified",
        }


# -- parse_plan --
class TestParsePlan:
    def test_parses_json(self):
        plan = parse_plan(
            '{"chunker":"structural","extractor":"llm","resolver":"semantic","reason":"markdown"}'
        )
        assert plan == IngestionPlan("structural", "llm", "semantic", "markdown")

    def test_parses_fenced_json(self):
        plan = parse_plan('```json\n{"chunker": "fixed"}\n```')
        assert plan is not None
        assert plan.chunker == "fixed"
        assert plan.extractor == "gliner"  # default-filled

    def test_parses_key_value(self):
        plan = parse_plan("chunker: contextual\nextractor = gliner\nresolver: exact")
        assert plan == IngestionPlan("contextual", "gliner", "exact")

    def test_partial_fills_defaults(self):
        plan = parse_plan('{"resolver":"llm_verified"}')
        assert plan == IngestionPlan("sentence", "gliner", "llm_verified")

    def test_unknown_values_drop_to_default(self):
        plan = parse_plan('{"chunker":"nope","extractor":"llm"}')
        assert plan is not None
        assert plan.chunker == "sentence"
        assert plan.extractor == "llm"

    def test_garbage_returns_none(self):
        assert parse_plan("hello world") is None
        assert parse_plan("") is None
        assert parse_plan("   ") is None


# -- HeuristicIngestionPlanner --
class TestTunableParams:
    def test_clamp_within_range(self):
        out = clamp_params("chunker", "sentence", {"max_tokens": 256, "overlap_sentences": 3})
        assert out == {"max_tokens": 256, "overlap_sentences": 3}

    def test_clamp_out_of_range(self):
        out = clamp_params("chunker", "sentence", {"max_tokens": 999999, "overlap_sentences": -5})
        assert out["max_tokens"] == 2048
        assert out["overlap_sentences"] == 0

    def test_clamp_drops_unknown_and_unparseable(self):
        out = clamp_params("extractor", "gliner", {"threshold": "high", "bogus": 1})
        assert out == {}

    def test_clamp_unknown_strategy(self):
        assert clamp_params("chunker", "nope", {"max_tokens": 100}) == {}

    def test_fixed_overlap_forced_below_size(self):
        out = clamp_params("chunker", "fixed", {"chunk_size": 200, "chunk_overlap": 500})
        assert out["chunk_overlap"] < out["chunk_size"]

    def test_llm_verified_inverted_thresholds_dropped(self):
        out = clamp_params(
            "resolver", "llm_verified", {"hard_threshold": 0.6, "soft_threshold": 0.9}
        )
        assert "hard_threshold" not in out and "soft_threshold" not in out

    def test_plan_clamps_params(self):
        plan = IngestionPlan(chunker="sentence", chunker_params={"max_tokens": 100000})
        assert plan.chunker_params == {"max_tokens": 2048}

    def test_parse_plan_reads_params(self):
        plan = parse_plan(
            '{"chunker":"fixed","chunker_params":{"chunk_size":1500,"chunk_overlap":150}}'
        )
        assert plan is not None
        assert plan.chunker == "fixed"
        assert plan.chunker_params == {"chunk_size": 1500, "chunk_overlap": 150}

    def test_build_chunker_applies_params(self):
        c = build_chunker("sentence", params={"max_tokens": 256, "overlap_sentences": 1})
        assert c.max_tokens == 256
        assert c.overlap_sentences == 1

    def test_build_extractor_applies_threshold(self):
        ext = build_extractor("gliner", llm=MockLLM(), params={"threshold": 0.4})
        assert ext.entity_extractor._threshold == 0.4

    def test_build_resolver_applies_params(self):
        r = build_resolver(
            "semantic",
            llm=MockLLM(),
            embedder=MockEmbedder(),
            params={"similarity_threshold": 0.88, "ann_top_k": 25},
        )
        assert isinstance(r, SemanticResolution)
        assert r.similarity_threshold == 0.88
        assert r.ann_top_k == 25
