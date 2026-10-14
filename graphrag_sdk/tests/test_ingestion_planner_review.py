"""Regression tests for the ingestion planner review findings (second round)."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock

import pytest

from graphrag_sdk import GraphRAG, HeuristicIngestionPlanner
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import DocumentInfo, DocumentOutput, DocumentRecord
from graphrag_sdk.ingestion.chunking_strategies.structural_chunking import StructuralChunking
from graphrag_sdk.ingestion.ingestion_planner import (
    IngestionPlan,
    LLMIngestionPlanner,
    _guide,
    build_chunker,
    parse_plan,
    restrict_plan,
)

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

    @pytest.mark.parametrize("bad", [5, -1, 6000, 1.5])
    def test_out_of_range_is_refused(self, bad):
        with pytest.raises(ValueError):
            restrict_plan(IngestionPlan(resolver="llm_verified"), max_llm_pairs=bad)

    def test_whole_float_is_accepted(self):
        assert LLMIngestionPlanner(MockLLM(), max_llm_pairs=500.0).max_llm_pairs == 500

    def test_in_range_caps_the_budget(self):
        plan = IngestionPlan(resolver="llm_verified", resolver_params={"max_llm_pairs": 900})
        assert restrict_plan(plan, max_llm_pairs=40).resolver_params == {"max_llm_pairs": 40}

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


def test_an_edited_param_is_clamped_again_when_built():
    plan = IngestionPlan(chunker="fixed", chunker_params={"chunk_size": 1000})
    plan.chunker_params["chunk_size"] = 10**9  # the dict stays plain
    assert build_chunker(plan.chunker, params=plan.chunker_params).chunk_size == 8000


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
        with pytest.raises(Exception, match="corrupt file"):
            await g.ingest(str(f), planner=planner)
        assert plans == [] and len(loads) == 2  # was 3 loads and a planner call

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
        with pytest.raises(Exception, match="corrupt file"):
            await g.ingest(str(f), planner=planner)
        assert plans == [] and len(loads) == 2


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
