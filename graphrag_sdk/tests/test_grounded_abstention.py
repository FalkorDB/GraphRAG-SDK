"""Tests for the grounded-answer / abstention example.

Locks in the behavior the ``grounded_answers_with_abstention.py`` example
advertises: a question with no supporting context in the graph returns an
explicit evidence-insufficient response instead of a fabricated answer.
The example module is imported directly so example and test cannot drift.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.api.main import GraphRAG
from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.models import RagResult, RetrieverResult, RetrieverResultItem
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy

from .conftest import MockLLM

_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "grounded_answers_with_abstention.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("grounded_answers_with_abstention", _EXAMPLE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load example module from {_EXAMPLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves annotations via sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


example = _load_example()


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_conn():
    conn = MagicMock(spec=FalkorDBConnection)
    result_mock = MagicMock()
    result_mock.result_set = []
    conn.query = AsyncMock(return_value=result_mock)
    conn.config = ConnectionConfig()
    ontology_graph = MagicMock()
    ontology_graph.query = AsyncMock(return_value=result_mock)
    conn._driver = MagicMock()
    conn._driver.select_graph = MagicMock(return_value=ontology_graph)
    conn._ensure_client = MagicMock()
    return conn


def _rag_with_context(mock_conn, embedder, items, *, answer="Alice Johnson founded Acme Corp."):
    """A GraphRAG whose retrieval always returns ``items``."""
    llm = MockLLM(responses=[answer])
    rag = GraphRAG(
        connection=mock_conn,
        llm=llm,
        embedder=embedder,
        embedding_dimension=8,
    )
    strategy = MagicMock(spec=RetrievalStrategy)
    strategy.search = AsyncMock(return_value=RetrieverResult(items=items))
    rag._retrieval_strategy = strategy
    return rag, llm


# ── Abstention ──────────────────────────────────────────────────


class TestAbstention:
    async def test_no_context_returns_evidence_insufficient(self, mock_conn, embedder):
        """No supporting context → explicit refusal, no generation."""
        rag, llm = _rag_with_context(mock_conn, embedder, items=[])

        retrieved = await rag.retrieve("What was Acme Corp's revenue in 2024?")
        assert retrieved.items == []

        result = await example.answer_or_abstain(rag, "What was Acme Corp's revenue in 2024?")

        assert result.grounded is False
        assert result.answer == example.INSUFFICIENT_EVIDENCE
        assert result.citations == []
        # The LLM was never asked to guess.
        assert llm._call_index == 0

    async def test_hint_only_context_is_not_evidence(self, mock_conn, embedder):
        """A MultiPath answer-format hint carries no evidence → abstain."""
        items = [
            RetrieverResultItem(
                content="Answer format: Name the place.", metadata={"section": "hint"}
            ),
            RetrieverResultItem(content="   ", metadata={"chunk_id": "c1"}),
        ]
        rag, llm = _rag_with_context(mock_conn, embedder, items=items)

        result = await example.answer_or_abstain(rag, "Where is Acme Corp's Tokyo office?")

        assert result.grounded is False
        assert result.answer == example.INSUFFICIENT_EVIDENCE
        assert llm._call_index == 0

    async def test_below_threshold_context_abstains(self, mock_conn, embedder):
        """Retrieved but low-scoring context is treated as insufficient."""
        items = [
            RetrieverResultItem(content="Weak match.", score=0.05, metadata={"chunk_id": "c1"})
        ]
        rag, llm = _rag_with_context(mock_conn, embedder, items=items)

        result = await example.answer_or_abstain(rag, "Who founded Acme Corp?", min_score=0.5)

        assert result.grounded is False
        assert result.answer == example.INSUFFICIENT_EVIDENCE
        assert llm._call_index == 0

    async def test_custom_refusal_string(self, mock_conn, embedder):
        rag, _ = _rag_with_context(mock_conn, embedder, items=[])

        result = await example.answer_or_abstain(rag, "Unknown?", refusal="No evidence.")

        assert result.answer == "No evidence."
        assert result.grounded is False


# ── Positive control ────────────────────────────────────────────


class TestGroundedAnswer:
    async def test_retrieval_options_match_generation(self):
        item = RetrieverResultItem(content="Evidence.", metadata={"chunk_id": "c1"})
        retriever_result = RetrieverResult(items=[item])
        rag = MagicMock(spec=GraphRAG)
        rag.retrieve = AsyncMock(return_value=retriever_result)
        rag.completion = AsyncMock(
            return_value=RagResult(answer="Grounded.", retriever_result=retriever_result)
        )
        strategy = MagicMock()
        reranker = MagicMock()
        ctx = MagicMock()

        await example.answer_or_abstain(
            rag,
            "Q?",
            strategy=strategy,
            reranker=reranker,
            ctx=ctx,
            return_context=False,
        )

        rag.retrieve.assert_awaited_once_with("Q?", strategy=strategy, reranker=reranker, ctx=ctx)
        rag.completion.assert_awaited_once_with(
            "Q?",
            strategy=strategy,
            reranker=reranker,
            ctx=ctx,
            return_context=True,
        )

    async def test_supported_question_returns_grounded_answer_with_citations(
        self, mock_conn, embedder
    ):
        items = [
            RetrieverResultItem(
                content="Acme Corp was founded in 2015 by Alice Johnson.",
                score=0.9,
                metadata={"chunk_id": "acme_facts:0"},
            )
        ]
        rag, llm = _rag_with_context(mock_conn, embedder, items=items)

        result = await example.answer_or_abstain(rag, "Who founded Acme Corp?")

        assert result.grounded is True
        assert result.answer == "Alice Johnson founded Acme Corp."
        assert result.citations == ["chunk_id=acme_facts:0"]
        assert result.context[0].content.startswith("Acme Corp was founded")
        assert llm._call_index == 1

    async def test_min_items_gate_requires_multiple_sources(self, mock_conn, embedder):
        items = [RetrieverResultItem(content="Single source.", metadata={"chunk_id": "c1"})]
        rag, llm = _rag_with_context(mock_conn, embedder, items=items)

        assert (await example.answer_or_abstain(rag, "Q?", min_items=2)).grounded is False
        assert (await example.answer_or_abstain(rag, "Q?", min_items=1)).grounded is True
        assert llm._call_index == 1


# ── Helper units ────────────────────────────────────────────────


class TestEvidenceHelpers:
    def test_is_evidence_rules(self):
        scored = RetrieverResultItem(content="text", score=0.4, metadata={"chunk_id": "c1"})
        assert example.is_evidence(scored, None) is True
        assert example.is_evidence(scored, 0.3) is True
        assert example.is_evidence(scored, 0.9) is False

        unscored = RetrieverResultItem(content="text")
        assert example.is_evidence(unscored, None) is True
        # A threshold cannot be satisfied by an item with no score.
        assert example.is_evidence(unscored, 0.1) is False

        assert example.is_evidence(RetrieverResultItem(content=""), None) is False

    def test_citation_of_falls_back_through_metadata_keys(self):
        assert (
            example.citation_of(RetrieverResultItem(content="x", metadata={"chunk_id": "c1"}), 0)
            == "chunk_id=c1"
        )
        assert (
            example.citation_of(RetrieverResultItem(content="x", metadata={"section": "facts"}), 0)
            == "section=facts"
        )
        assert example.citation_of(RetrieverResultItem(content="x"), 3) == "item[3]"
