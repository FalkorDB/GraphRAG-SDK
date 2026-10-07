"""Tests for agentic answers: citations, the grounding gate and facade integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.api.main import GraphRAG
from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    LLMResponse,
    RetrieverResult,
    RetrieverResultItem,
    ToolCall,
)
from graphrag_sdk.retrieval.agentic import (
    DEFAULT_UNGROUNDED_ANSWER,
    AgenticRetrieval,
    EvidenceLedger,
    Tool,
    ToolRegistry,
    ground_answer,
)
from graphrag_sdk.retrieval.agentic.graph_tools import make_lookup_entity_tool, make_search_tool
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy

from .conftest import MockEmbedder, MockLLM


def ledger(n: int) -> EvidenceLedger:
    out = EvidenceLedger()
    for i in range(n):
        out.add(tool="search", kind="fact", content=f"fact {i + 1}")
    return out


# ── ground_answer ────────────────────────────────────────────────


class TestGroundAnswer:
    def test_valid_citations(self):
        g = ground_answer("Alice works at Acme [1] in Berlin [2].", ledger(2), tools_called=True)
        assert g.grounded and g.reason == "cited"
        assert g.cited == [1, 2] and g.invalid == []
        assert g.answer == "Alice works at Acme [1] in Berlin [2]."
        assert [e.content for e in g.citations(ledger(2))] == ["fact 1", "fact 2"]

    def test_invalid_markers_are_removed(self):
        g = ground_answer("Acme [1, 9] is big [7].", ledger(2), tools_called=True)
        assert g.grounded
        assert g.answer == "Acme [1] is big."
        assert g.invalid == [9, 7]

    def test_only_invented_citations_is_ungrounded(self):
        g = ground_answer("Acme is big [7].", ledger(2), tools_called=True)
        assert not g.grounded and g.reason == "invalid_citations"
        assert g.answer == DEFAULT_UNGROUNDED_ANSWER
        assert g.raw_answer == "Acme is big [7]."

    def test_uncited_answer_after_tools_is_ungrounded(self):
        g = ground_answer("Acme is big.", ledger(2), tools_called=True)
        assert not g.grounded and g.reason == "no_citations"

    def test_short_conversation_needs_no_citation(self):
        g = ground_answer("Hello! How can I help?", EvidenceLedger(), tools_called=False)
        assert g.grounded and g.reason == "conversation"

    def test_long_uncited_answer_without_tools_is_ungrounded(self):
        g = ground_answer("x " * 200, EvidenceLedger(), tools_called=False)
        assert not g.grounded

    def test_annotate_keeps_the_answer(self):
        g = ground_answer("Acme is big [7].", ledger(1), tools_called=True, policy="annotate")
        assert not g.grounded and g.answer == "Acme is big."

    def test_off_skips_the_check(self):
        g = ground_answer("Acme [9].", ledger(0), tools_called=True, policy="off")
        assert g.grounded and g.answer == "Acme [9]." and g.reason == "not_checked"

    def test_custom_fallback(self):
        g = ground_answer("x", ledger(1), tools_called=True, ungrounded_answer="Not found.")
        assert g.answer == "Not found."


# ── Loop integration ─────────────────────────────────────────────


class Strategy:
    async def search(self, query: str, ctx: Context, **kwargs: Any) -> RetrieverResult:
        return RetrieverResult(items=[RetrieverResultItem(content="Alice works at Acme.")])


class LLM:
    supports_tool_calling = True
    model_name = "m"

    def __init__(self, *responses: LLMResponse):
        self.responses = list(responses)

    async def ainvoke_with_tools(self, messages, tools, **kwargs):
        return self.responses.pop(0)


def searching(answer: str) -> LLM:
    return LLM(
        LLMResponse(
            content="", tool_calls=[ToolCall(id="c", name="search", arguments={"query": "a"})]
        ),
        LLMResponse(content=answer),
    )


class TestLoopGrounding:
    async def test_invented_citation_is_replaced(self, ctx):
        result = await AgenticRetrieval(
            searching("Bob runs Acme [5]."), strategy=Strategy()
        ).search("q", ctx)
        md = result.metadata
        assert md["answer"] == DEFAULT_UNGROUNDED_ANSWER
        assert md["raw_answer"] == "Bob runs Acme [5]."
        assert md["grounded"] is False and md["invalid_citations"] == [5]
        assert md["answer_is_final"] is True

    async def test_annotate_policy(self, ctx):
        agent = AgenticRetrieval(searching("Uncited."), strategy=Strategy(), grounding="annotate")
        md = (await agent.search("q", ctx)).metadata
        assert md["answer"] == "Uncited." and md["grounded"] is False

    def test_invalid_policy(self):
        with pytest.raises(ValueError):
            AgenticRetrieval(LLM(), strategy=Strategy(), grounding="lenient")  # type: ignore[arg-type]

    async def test_plain_tool_output_gets_one_citation_number(self, ctx):
        async def weather(args, tctx):
            return "Sunny in Berlin."

        registry = ToolRegistry()
        registry.register(make_search_tool(Strategy()))
        registry.register(Tool("weather", "Weather report.", weather))
        llm = LLM(
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="a", name="search", arguments={"query": "x"}),
                    ToolCall(id="b", name="weather", arguments={}),
                ],
            ),
            LLMResponse(content="Alice is at Acme [1]; it is sunny [2]."),
        )
        result = await AgenticRetrieval(llm, registry=registry).search("q", ctx)
        steps = result.metadata["agent_trace"]["steps"]
        assert steps[1]["observation"] == "[2] Sunny in Berlin."
        assert [c["tool"] for c in result.metadata["citations"]] == ["search", "weather"]

    async def test_lookup_results_are_not_citable(self, ctx):
        class Store:
            async def query_raw(self, cypher, params=None, **kw):
                return MagicMock(result_set=[["a", "Alice", "Person", ""]])

        registry = ToolRegistry()
        registry.register(make_lookup_entity_tool(Store()))
        llm = LLM(
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="a", name="lookup_entity", arguments={"name": "ali"})],
            ),
            LLMResponse(content="The entity is called Alice."),
        )
        result = await AgenticRetrieval(llm, registry=registry).search("q", ctx)
        assert result.metadata["evidence"] == []
        assert result.metadata["grounded"] is False


# ── Facade integration ───────────────────────────────────────────


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


class RecordingAgent(RetrievalStrategy):
    """A strategy that behaves like AgenticRetrieval from the facade's view."""

    accepts_history = True

    def __init__(self):
        super().__init__()
        self.seen: list[tuple[str, Any]] = []
        self.ontology = None

    def set_ontology(self, ontology):
        self.ontology = ontology

    async def _execute(self, query, ctx, **kwargs):
        from graphrag_sdk.core.models import RawSearchResult

        self.seen.append((query, kwargs.get("history")))
        return RawSearchResult(
            records=[{"n": 1, "content": "Alice works at Acme.", "tool": "search"}],
            metadata={
                "answer": "Alice works at Acme [1].",
                "answer_is_final": True,
                "grounded": True,
                "grounding_reason": "cited",
                "citations": [{"n": 1, "content": "Alice works at Acme."}],
                "agent_mode": "native",
                "stop_reason": "final_answer",
                "num_steps": 1,
                "generated_cypher": [],
            },
        )

    def _format(self, raw):
        return RetrieverResult(
            items=[RetrieverResultItem(content="[1] Alice works at Acme.")], metadata=raw.metadata
        )


class TestFacade:
    def rag(self, mock_conn, llm=None):
        return GraphRAG(
            connection=mock_conn,
            llm=llm or MockLLM(responses=["SECOND ANSWER"]),
            embedder=MockEmbedder(dimension=8),
            embedding_dimension=8,
        )

    async def test_completion_returns_the_agent_answer(self, mock_conn):
        llm = MockLLM(responses=["SECOND ANSWER"])
        rag = self.rag(mock_conn, llm)
        agent = RecordingAgent()
        result = await rag.completion("Where does Alice work?", strategy=agent, return_context=True)
        assert result.answer == "Alice works at Acme [1]."
        assert llm.last_messages is None  # no second generation
        assert result.metadata["grounded"] is True
        assert result.metadata["citations"][0]["n"] == 1
        assert result.metadata["agent"]["stop_reason"] == "final_answer"
        assert result.metadata["strategy"] == "RecordingAgent"
        assert result.retriever_result is not None
        assert agent.ontology is not None  # per-call strategy got the ontology

    async def test_history_goes_to_the_agent_and_rewrite_is_skipped(self, mock_conn):
        llm = MockLLM(responses=["REWRITTEN"])
        rag = self.rag(mock_conn, llm)
        agent = RecordingAgent()
        history = [
            {"role": "user", "content": "Who is Alice?"},
            {"role": "assistant", "content": "An engineer."},
        ]
        await rag.completion(
            "Where does she work?",
            strategy=agent,
            history=history,
            rewrite_question_with_history=True,
        )
        query, seen_history = agent.seen[0]
        assert query == "Where does she work?"
        assert [m.content for m in seen_history] == ["Who is Alice?", "An engineer."]
        assert llm._call_index == 0  # no rewrite call

    async def test_non_agentic_strategy_gets_no_history(self, mock_conn):
        rag = self.rag(mock_conn)
        strategy = MagicMock(spec=RetrievalStrategy)
        strategy.search = AsyncMock(return_value=RetrieverResult(items=[]))
        await rag.retrieve("q", strategy=strategy, history=[{"role": "user", "content": "x"}])
        strategy.search.assert_awaited_once()
        assert "history" not in strategy.search.await_args.kwargs

    async def test_regular_completion_still_generates(self, mock_conn):
        llm = MockLLM(responses=["Generated."])
        rag = self.rag(mock_conn, llm)
        strategy = MagicMock(spec=RetrievalStrategy)
        strategy.search = AsyncMock(
            return_value=RetrieverResult(items=[RetrieverResultItem(content="ctx")])
        )
        result = await rag.completion("q", strategy=strategy)
        assert result.answer == "Generated."

    def test_agentic_retrieval_builder(self, mock_conn):
        rag = self.rag(mock_conn)
        agent = rag.agentic_retrieval(max_steps=3)
        assert isinstance(agent, AgenticRetrieval)
        assert agent._max_steps == 3
        assert agent._strategy is rag._retrieval_strategy
        assert "query_graph" in agent.registry.names()
        assert agent._ontology is rag._global_ontology
