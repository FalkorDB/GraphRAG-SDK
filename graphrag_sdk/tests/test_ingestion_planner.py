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


class TestHeuristicPlanner:
    async def test_markdown_source_picks_structural(self):
        plan = await HeuristicIngestionPlanner().plan("anything", source="notes.md")
        assert plan.chunker == "structural"

    async def test_markdown_body_picks_structural(self):
        plan = await HeuristicIngestionPlanner().plan(
            "# Heading\n\n- bullet one\n- bullet two", source="x.txt"
        )
        assert plan.chunker == "structural"

    async def test_plain_prose_picks_sentence(self):
        plan = await HeuristicIngestionPlanner().plan(
            "Just a paragraph of ordinary prose with no structure at all.",
            source="x.txt",
        )
        assert plan.chunker == "sentence"

    async def test_never_upgrades_cost_options(self):
        plan = await HeuristicIngestionPlanner().plan("# H", source="x.md")
        assert plan.extractor == "gliner"
        assert plan.resolver == "exact"


# -- LLMIngestionPlanner --


class TestLLMPlanner:
    async def test_returns_parsed_plan(self):
        llm = MockLLM(responses=['{"chunker":"fixed","extractor":"llm","resolver":"semantic"}'])
        plan = await LLMIngestionPlanner(llm).plan("some text", source="x.txt")
        assert plan == IngestionPlan("fixed", "llm", "semantic")

    async def test_empty_response_falls_back_to_default(self):
        plan = await LLMIngestionPlanner(MockLLM(responses=[""])).plan("t")
        assert plan == default_plan()

    async def test_unparseable_response_falls_back_to_default(self):
        plan = await LLMIngestionPlanner(MockLLM(responses=["I think markdown"])).plan("t")
        assert plan == default_plan()

    async def test_llm_error_falls_back_to_default(self):
        class BoomLLM:
            async def ainvoke(self, *a, **k):
                raise RuntimeError("boom")

        plan = await LLMIngestionPlanner(BoomLLM()).plan("t")
        assert plan == default_plan()

    async def test_budget_error_propagates(self):
        class BudgetLLM:
            async def ainvoke(self, *a, **k):
                raise LatencyBudgetExceededError("over budget")

        with pytest.raises(LatencyBudgetExceededError):
            await LLMIngestionPlanner(BudgetLLM()).plan("t")


# -- builders --


class TestBuilders:
    def test_build_chunker_variants(self):
        llm = MockLLM()
        assert isinstance(build_chunker("sentence"), SentenceTokenCapChunking)
        assert isinstance(build_chunker("fixed"), FixedSizeChunking)
        assert isinstance(build_chunker("structural"), StructuralChunking)
        assert isinstance(build_chunker("contextual", llm=llm), ContextualChunking)

    def test_contextual_chunker_requires_llm(self):
        with pytest.raises(ValueError):
            build_chunker("contextual")

    def test_build_extractor_backends(self):
        llm = MockLLM()
        gliner = build_extractor("gliner", llm=llm)
        llm_ext = build_extractor("llm", llm=llm)
        assert isinstance(gliner, GraphExtraction)
        assert isinstance(gliner.entity_extractor, GLiNERExtractor)
        assert isinstance(llm_ext.entity_extractor, LLMExtractor)

    def test_build_resolver_variants(self):
        llm = MockLLM()
        emb = MockEmbedder()
        assert isinstance(build_resolver("exact"), ExactMatchResolution)
        assert isinstance(build_resolver("description_merge", llm=llm), DescriptionMergeResolution)
        assert isinstance(build_resolver("semantic", llm=llm, embedder=emb), SemanticResolution)
        assert isinstance(
            build_resolver("llm_verified", llm=llm, embedder=emb), LLMVerifiedResolution
        )

    def test_build_ingestion_strategies_triple(self):
        chunker, extractor, resolver = build_ingestion_strategies(
            IngestionPlan("structural", "llm", "semantic"),
            llm=MockLLM(),
            embedder=MockEmbedder(),
            entity_types=["Person"],
        )
        assert isinstance(chunker, StructuralChunking)
        assert isinstance(extractor, GraphExtraction)
        assert isinstance(resolver, SemanticResolution)


# -- GraphRAG._plan_ingestion_strategies merge logic --
#
# The method only touches self.llm / self.embedder / self.ontology, so a light
# stub stands in for a full GraphRAG instance.


def _fake_rag(llm, *, entities=None):
    return SimpleNamespace(
        llm=llm,
        embedder=MockEmbedder(),
        ontology=SimpleNamespace(entities=entities or []),
    )


class TestPlanIngestionStrategiesMerge:
    async def test_fills_all_when_none_passed(self):
        llm = MockLLM(responses=['{"chunker":"fixed","extractor":"llm","resolver":"exact"}'])
        rag = _fake_rag(llm)
        chunker, extractor, resolver, preloaded_document = (
            await GraphRAG._plan_ingestion_strategies(
                rag,
                text="hello",
                source="x.txt",
                chunker=None,
                extractor=None,
                resolver=None,
                planner=None,
                ctx=None,
            )
        )
        assert isinstance(chunker, FixedSizeChunking)
        assert isinstance(extractor, GraphExtraction)
        assert isinstance(resolver, ExactMatchResolution)
        # text mode: no file was loaded, so there's nothing to preload for reuse.
        assert preloaded_document is None

    async def test_explicit_override_wins(self):
        llm = MockLLM(responses=['{"chunker":"fixed","extractor":"llm","resolver":"semantic"}'])
        rag = _fake_rag(llm)
        explicit_chunker = SentenceTokenCapChunking()
        chunker, extractor, resolver, _preloaded_document = (
            await GraphRAG._plan_ingestion_strategies(
                rag,
                text="hello",
                source="x.txt",
                chunker=explicit_chunker,
                extractor=None,
                resolver=None,
                planner=None,
                ctx=None,
            )
        )
        # Caller's chunker is preserved; only the unset slots come from the plan.
        assert chunker is explicit_chunker
        assert isinstance(resolver, SemanticResolution)

    async def test_custom_heuristic_planner_used(self):
        rag = _fake_rag(MockLLM())
        chunker, _, _, _ = await GraphRAG._plan_ingestion_strategies(
            rag,
            text="# Title\n- a\n- b",
            source="notes.md",
            chunker=None,
            extractor=None,
            resolver=None,
            planner=HeuristicIngestionPlanner(),
            ctx=None,
        )
        assert isinstance(chunker, StructuralChunking)

    async def test_planner_failure_leaves_slots_none(self):
        class BoomPlanner:
            async def plan(self, *a, **k):
                raise RuntimeError("boom")

        rag = _fake_rag(MockLLM())
        result = await GraphRAG._plan_ingestion_strategies(
            rag,
            text="hello",
            source="x.txt",
            chunker=None,
            extractor=None,
            resolver=None,
            planner=BoomPlanner(),
            ctx=None,
        )
        assert result == (None, None, None, None)

    async def test_none_plan_falls_back_to_defaults(self):
        class NonePlanner:
            async def plan(self, *a, **k):
                return None

        rag = _fake_rag(MockLLM())
        result = await GraphRAG._plan_ingestion_strategies(
            rag,
            text="hello",
            source="x.txt",
            chunker=None,
            extractor=None,
            resolver=None,
            planner=NonePlanner(),
            ctx=None,
        )
        assert result == (None, None, None, None)

    async def test_file_mode_loads_content_sample_for_planner(self):
        seen: dict[str, str | None] = {}

        class RecordingPlanner:
            async def plan(self, text, *, source=None, ctx=None):
                seen["text"] = text
                return IngestionPlan(chunker="fixed", extractor="gliner", resolver="exact")

        class FakeLoader:
            async def load(self, source, ctx):
                return SimpleNamespace(text="LOADED CONTENT SAMPLE")

        rag = _fake_rag(MockLLM())
        _chunker, _extractor, _resolver, preloaded_document = (
            await GraphRAG._plan_ingestion_strategies(
                rag,
                text=None,
                source="x.pdf",
                loader=FakeLoader(),
                chunker=None,
                extractor=None,
                resolver=None,
                planner=RecordingPlanner(),
                ctx=None,
            )
        )
        # File mode: the planner sees loaded content, not just the path.
        assert seen["text"] == "LOADED CONTENT SAMPLE"
        # The full loaded document is returned too, so the caller (ingest())
        # can reuse it instead of loading the source a second time.
        assert preloaded_document is not None
        assert preloaded_document.text == "LOADED CONTENT SAMPLE"

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

    def test_llm_verified_lone_hard_below_default_soft_dropped(self):
        # hard=0.6 vs constructor default soft=0.80 would make the resolver raise.
        out = clamp_params("resolver", "llm_verified", {"hard_threshold": 0.6})
        assert "hard_threshold" not in out

    def test_llm_verified_lone_soft_above_default_hard_dropped(self):
        # soft=0.97 vs constructor default hard=0.95 would make the resolver raise.
        out = clamp_params("resolver", "llm_verified", {"soft_threshold": 0.97})
        assert "soft_threshold" not in out

    def test_llm_verified_lone_valid_threshold_kept(self):
        out = clamp_params("resolver", "llm_verified", {"hard_threshold": 0.9})
        assert out == {"hard_threshold": 0.9}

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
