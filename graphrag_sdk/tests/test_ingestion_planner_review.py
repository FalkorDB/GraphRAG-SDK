"""Regression tests for the ingestion planner review findings (second round)."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock

import pytest

from graphrag_sdk import GraphRAG, HeuristicIngestionPlanner
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import IngestionError
from graphrag_sdk.core.models import DocumentInfo, DocumentOutput, DocumentRecord
from graphrag_sdk.ingestion.chunking_strategies.structural_chunking import StructuralChunking
from graphrag_sdk.ingestion.extraction_strategies.graph_extraction import GraphExtraction
from graphrag_sdk.ingestion.ingestion_planner import (
    CHEAP_PLAN_LIMITS,
    ChunkBudget,
    ChunkBudgetExceededError,
    IngestionPlan,
    LLMIngestionPlanner,
    RuleBasedIngestionPlanner,
    _guide,
    build_chunker,
    build_resolver,
    parse_plan,
    restrict_plan,
)
from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution

from .conftest import MockLLM
from .test_facade import mock_conn  # noqa: F401 - fixture

LONG_PROSE = DocumentOutput(
    text=" ".join(f"Sentence number {i} talks about topic {i % 7} here." for i in range(1500)),
    document_info=DocumentInfo(path="x"),
)
LONG_RUN = DocumentOutput(text="word " * 20000, document_info=DocumentInfo(path="x"))


async def _chunks(chunker, doc) -> int:
    return len((await chunker.chunk_document(doc, Context())).chunks)


# ── Chunk count (the cost driver) is bounded ────────────────────


class TestChunkCountIsBounded:
    async def test_fixed_worst_case(self):
        default = await _chunks(build_chunker("fixed"), LONG_RUN)
        worst = await _chunks(
            build_chunker("fixed", params={"chunk_size": 1, "chunk_overlap": 10_000}), LONG_RUN
        )
        assert worst <= 3 * default  # was ~900x (a 1-character step)

    async def test_sentence_worst_case(self):
        default = await _chunks(build_chunker("sentence"), LONG_PROSE)
        worst = await _chunks(
            build_chunker("sentence", params={"max_tokens": 1, "overlap_sentences": 99}),
            LONG_PROSE,
        )
        assert worst <= 2 * default  # was ~30x

    async def test_tune_params_false_keeps_every_default(self):
        reply = (
            '{"chunker": "fixed", "chunker_params": {"chunk_size": 500}, '
            '"resolver": "llm_verified", "resolver_params": {"ann_top_k": 5}}'
        )
        planner = LLMIngestionPlanner(MockLLM(responses=[reply]), tune_params=False)
        plan = await planner.plan("text")
        assert plan.chunker == "fixed" and plan.chunker_params == {}
        assert plan.resolver_params == {}
        assert "You MAY also tune" not in _guide(tune_params=False)
        assert "You MAY also tune" in _guide()


# ── max_llm_pairs ───────────────────────────────────────────────


class TestMaxLlmPairs:
    def test_zero_disallows_llm_verified(self):
        plan = restrict_plan(IngestionPlan(resolver="llm_verified"), max_llm_pairs=0)
        assert plan.resolver == "exact"

    @pytest.mark.parametrize("bad", [5, -1, 6000, 1.5, float("inf"), float("-inf"), float("nan")])
    def test_out_of_range_is_refused(self, bad):
        with pytest.raises(ValueError):
            restrict_plan(IngestionPlan(resolver="llm_verified"), max_llm_pairs=bad)

    def test_whole_float_is_accepted(self):
        assert LLMIngestionPlanner(MockLLM(), max_llm_pairs=500.0).max_llm_pairs == 500

    def test_in_range_caps_the_budget(self):
        plan = IngestionPlan(resolver="llm_verified", resolver_params={"max_llm_pairs": 900})
        assert restrict_plan(plan, max_llm_pairs=40).resolver_params == {"max_llm_pairs": 40}

    async def test_cap_survives_tune_params_false(self):
        # tune_params=False drops the plan's own params, but the caller's
        # pair ceiling must still reach the built resolver (it would
        # otherwise default to 500 pairs).
        reply = '{"resolver": "llm_verified", "resolver_params": {"max_llm_pairs": 900}}'
        planner = LLMIngestionPlanner(
            MockLLM(responses=[reply]), max_llm_pairs=20, tune_params=False
        )
        plan = await planner.plan("text")
        assert plan.resolver == "llm_verified"
        assert plan.resolver_params == {"max_llm_pairs": 20}
        resolver = build_resolver(plan.resolver, params=plan.resolver_params)
        assert resolver.max_llm_pairs == 20

    async def test_rule_planner_cap_survives_tune_params_false(self):
        planner = RuleBasedIngestionPlanner(
            [{"match": "*", "resolver": "llm_verified"}], max_llm_pairs=20, tune_params=False
        )
        plan = await planner.plan(None, source="a.md")
        resolver = build_resolver(plan.resolver, params=plan.resolver_params)
        assert resolver.max_llm_pairs == 20

    def test_cap_above_the_default_keeps_the_default_without_tuning(self):
        plan = restrict_plan(
            IngestionPlan(resolver="llm_verified"), max_llm_pairs=5000, tune_params=False
        )
        assert plan.resolver_params == {}  # the resolver's own 500

    def test_planner_with_zero(self):
        assert "llm_verified" not in LLMIngestionPlanner(MockLLM(), max_llm_pairs=0).resolvers
        with pytest.raises(ValueError):
            LLMIngestionPlanner(MockLLM(), resolvers=("llm_verified",), max_llm_pairs=0)


# ── IngestionPlan is hashable ───────────────────────────────────


def test_plan_is_hashable():
    a = IngestionPlan(chunker="fixed", chunker_params={"chunk_size": 1500}, reason="x")
    b = IngestionPlan(chunker="fixed", chunker_params={"chunk_size": 1500}, reason="y")
    assert a == b and hash(a) == hash(b)
    assert len({a, b, IngestionPlan()}) == 2


def test_plan_params_are_read_only_and_the_hash_is_stable():
    import copy
    import json
    import pickle

    plan = IngestionPlan(chunker="fixed", chunker_params={"chunk_size": 1000})
    h = hash(plan)
    for edit in (
        lambda d: d.__setitem__("chunk_size", 10**9),
        lambda d: d.update(chunk_size=1),
        lambda d: d.pop("chunk_size"),
        lambda d: d.clear(),
        lambda d: d.setdefault("x", 1),
    ):
        with pytest.raises(TypeError):
            edit(plan.chunker_params)
    assert hash(plan) == h and plan in {plan}
    # Still a plain dict for every reader: JSON, copies, pickling, replace.
    assert json.dumps(plan.chunker_params) == '{"chunk_size": 1000}'
    assert copy.deepcopy(plan) == plan and pickle.loads(pickle.dumps(plan)) == plan
    editable = dict(plan.chunker_params)
    editable["chunk_size"] = 2000
    assert IngestionPlan(chunker="fixed", chunker_params=editable).chunker_params == {
        "chunk_size": 2000
    }
    # The builders still clamp a hand-made params dict.
    assert build_chunker("fixed", params={"chunk_size": 10**9}).chunk_size == 8000


# ── parse_plan ──────────────────────────────────────────────────


def test_brace_in_prose_before_the_json_keeps_params():
    plan = parse_plan(
        "Options considered {sentence, fixed}. "
        '{"chunker":"fixed","chunker_params":{"chunk_size":1500}}'
    )
    assert plan.chunker == "fixed" and plan.chunker_params == {"chunk_size": 1500}


def test_first_object_naming_a_strategy_wins():
    plan = parse_plan('{"note": 1} then {"chunker": "fixed"} and {"chunker": "contextual"}')
    assert plan.chunker == "fixed"


# ── Heuristic planner: structural only where it can take effect ─


@pytest.mark.parametrize(
    ("source", "expected"),
    [("a.md", "structural"), ("a.markdown", "structural"), ("a.html", "sentence")],
)
async def test_heuristic_structural_by_extension(source, expected):
    plan = await HeuristicIngestionPlanner().plan(None, source=source)
    assert plan.chunker == expected


# ── Through ingest() ────────────────────────────────────────────


class Structural:
    async def plan(self, text, *, source=None, ctx=None):
        return IngestionPlan(chunker="structural")


async def test_wrong_plan_type_falls_back_to_defaults(mock_conn, embedder, llm):  # noqa: F811
    class DictPlanner:
        async def plan(self, text, *, source=None, ctx=None):
            return {"chunker": "fixed"}

    g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder, embedding_dimension=8)
    result = await g.ingest(text="Alice works at Acme.", document_id="d1", planner=DictPlanner())
    report = result.metadata["plan"]
    assert report["plan"] is None and "expected IngestionPlan" in report["notes"][0]


async def test_planner_without_auto_turns_auto_on(mock_conn, embedder, llm):  # noqa: F811
    g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder, embedding_dimension=8)
    result = await g.ingest(
        text="Alice works at Acme.", document_id="d2", planner=HeuristicIngestionPlanner()
    )
    assert result.metadata["plan"]["planner"] == "HeuristicIngestionPlanner"


async def test_markdown_in_a_txt_file_is_chunked_structurally(
    mock_conn,  # noqa: F811
    embedder,
    llm,
    tmp_path,
    monkeypatch,
):
    used: list[object] = []
    original = StructuralChunking.chunk_document

    async def spy(self, document, ctx):
        used.append(document.elements)
        return await original(self, document, ctx)

    monkeypatch.setattr(StructuralChunking, "chunk_document", spy)
    notes = tmp_path / "notes.txt"
    notes.write_text("# Intro\n\nAlice works at Acme.\n\n# Later\n\n- one\n- two\n- three\n")
    g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder, embedding_dimension=8)
    result = await g.ingest(str(notes), planner=Structural())
    assert result.metadata["plan"]["plan"]["chunker"] == "structural"
    assert used and used[0]  # the chunker got the parsed elements


async def test_structural_text_mode_keeps_identity(mock_conn, embedder, llm):  # noqa: F811
    text = "# Intro\n\nAlice works at Acme.\n\n# Later\n\nBob lives in Berlin.\n"
    g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder, embedding_dimension=8)
    planned = await g.ingest(text=text, planner=Structural())
    g2 = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder, embedding_dimension=8)
    plain = await g2.ingest(text=text)
    assert planned.document_info.uid == plain.document_info.uid
    assert planned.metadata["plan"]["plan"]["chunker"] == "structural"


class TestUnloadableFile:
    @staticmethod
    def _setup(monkeypatch):
        from graphrag_sdk.ingestion.loaders.text_loader import TextLoader

        loads: list[int] = []
        plans: list[int] = []

        async def broken(self, source, ctx):
            loads.append(1)
            raise ValueError("corrupt file")

        monkeypatch.setattr(TextLoader, "load", broken)

        class P:
            async def plan(self, text, *, source=None, ctx=None):
                plans.append(1)
                return IngestionPlan()

        return loads, plans, P()

    async def test_reingest_is_not_planned(
        self,
        mock_conn,  # noqa: F811
        embedder,
        llm,
        tmp_path,
        monkeypatch,
    ):
        loads, plans, planner = self._setup(monkeypatch)
        f = tmp_path / "a.txt"
        f.write_text("x")
        g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder, embedding_dimension=8)
        record = DocumentRecord(path=str(f), content_hash="old")
        g._graph_store.get_document_record = AsyncMock(return_value=record)
        with pytest.raises(IngestionError, match="corrupt file"):
            await g.ingest(str(f), planner=planner)
        assert plans == [] and len(loads) == 1  # was 3 loads and a planner call

    async def test_first_ingest_is_not_planned(
        self,
        mock_conn,  # noqa: F811
        embedder,
        llm,
        tmp_path,
        monkeypatch,
    ):
        loads, plans, planner = self._setup(monkeypatch)
        f = tmp_path / "a.txt"
        f.write_text("x")
        g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder, embedding_dimension=8)
        with pytest.raises(IngestionError, match="corrupt file"):
            await g.ingest(str(f), planner=planner)
        assert plans == [] and len(loads) == 1


# ── One definition of the content hash and the entity types ─────


def test_content_hash_has_one_definition():
    from graphrag_sdk.api import main
    from graphrag_sdk.ingestion import pipeline

    assert main.content_hash is pipeline.content_hash
    assert pipeline.content_hash("abc") == hashlib.sha256(b"abc").hexdigest()


async def test_planned_gliner_extractor_matches_the_default(mock_conn, embedder, llm):  # noqa: F811
    from graphrag_sdk.core.models import Entity, Ontology

    g = GraphRAG(
        connection=mock_conn,
        llm=llm,
        embedder=embedder,
        embedding_dimension=8,
        ontology=Ontology(entities=[Entity(label="Person"), Entity(label="Company")]),
    )
    _, extractor, _, _ = await g._plan_ingestion_strategies(
        text="Alice works at Acme.",
        source="t",
        chunker=None,
        extractor=None,
        resolver=None,
        planner=HeuristicIngestionPlanner(),
        ctx=None,
    )
    assert extractor.entity_types == g._default_extractor().entity_types == ["Person", "Company"]


def test_markdown_parse_failure_means_no_elements(monkeypatch):
    from graphrag_sdk.ingestion.loaders.markdown_loader import MarkdownLoader

    def boom(self, text):
        raise RuntimeError("parser broke")

    monkeypatch.setattr(MarkdownLoader, "_parse_markdown", boom)
    assert GraphRAG._markdown_elements("# A\n\n# B\n", "x.md") == []


# ── max_chunks: a per-document chunk budget, checked before extraction ──


class TestMaxChunks:
    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "10", float("inf")])
    def test_invalid_budget_is_refused(self, bad):
        with pytest.raises(ValueError, match="max_chunks"):
            RuleBasedIngestionPlanner(max_chunks=bad)
        with pytest.raises(ValueError, match="max_chunks"):
            LLMIngestionPlanner(MockLLM(), max_chunks=bad)

    def test_cheap_limits_carry_a_budget(self):
        assert CHEAP_PLAN_LIMITS["max_chunks"] == 1000
        assert RuleBasedIngestionPlanner(**CHEAP_PLAN_LIMITS).max_chunks == 1000
        assert RuleBasedIngestionPlanner().max_chunks is None

    async def test_over_budget_is_refused_before_any_extraction_call(
        self,
        mock_conn,  # noqa: F811
        embedder,
    ):
        llm = MockLLM()
        planner = RuleBasedIngestionPlanner(
            [{"match": "*", "chunker": "fixed", "extractor": "llm"}], max_chunks=2
        )
        g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder, embedding_dimension=8)
        mock_conn.query.reset_mock()
        with pytest.raises(ChunkBudgetExceededError, match="over max_chunks=2"):
            await g.ingest(text=LONG_RUN.text, document_id="big", planner=planner)
        assert llm._call_index == 0  # no extraction (or any other) LLM call
        writes = [c for c in mock_conn.query.call_args_list if "MERGE" in str(c.args[0])]
        assert writes == []  # nothing stored either

    async def test_within_budget_ingests(self, mock_conn, embedder):  # noqa: F811
        planner = RuleBasedIngestionPlanner([{"match": "*", "extractor": "llm"}], max_chunks=5)
        g = GraphRAG(connection=mock_conn, llm=MockLLM(), embedder=embedder, embedding_dimension=8)
        mock_conn.query.reset_mock()
        result = await g.ingest(text="Alice works at Acme.", document_id="small", planner=planner)
        assert result.chunks_indexed == 1
        assert any("MERGE" in str(c.args[0]) for c in mock_conn.query.call_args_list)
        assert result.metadata["plan"]["max_chunks"] == 5

    async def test_budget_holds_when_all_strategies_are_passed(
        self,
        mock_conn,  # noqa: F811
        embedder,
    ):
        llm = MockLLM()
        g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder, embedding_dimension=8)
        with pytest.raises(ChunkBudgetExceededError, match="over max_chunks=2"):
            await g.ingest(
                text=LONG_RUN.text,
                document_id="big",
                planner=RuleBasedIngestionPlanner(max_chunks=2),
                chunker=build_chunker("fixed"),
                extractor=GraphExtraction(llm=llm),
                resolver=ExactMatchResolution(),
            )
        assert llm._call_index == 0

    async def test_all_strategies_passed_reports_the_budget(
        self,
        mock_conn,  # noqa: F811
        embedder,
    ):
        llm = MockLLM()
        g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder, embedding_dimension=8)
        result = await g.ingest(
            text="Alice works at Acme.",
            document_id="small",
            planner=RuleBasedIngestionPlanner(max_chunks=5),
            chunker=build_chunker("fixed"),
            extractor=GraphExtraction(llm=llm),
            resolver=ExactMatchResolution(),
        )
        plan = result.metadata["plan"]
        assert plan["max_chunks"] == 5 and plan["planner"] is None
        assert set(plan["slots"].values()) == {"caller"}

    async def test_wraps_whatever_chunker_is_used(self):
        inner = build_chunker("fixed")
        budget = ChunkBudget(inner, 3)
        with pytest.raises(ChunkBudgetExceededError, match="makes \\d+ chunks"):
            await budget.chunk_document(LONG_RUN, Context())
        assert len((await ChunkBudget(inner, 10**6).chunk(LONG_RUN.text, Context())).chunks) > 3
