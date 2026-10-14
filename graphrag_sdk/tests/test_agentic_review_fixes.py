"""Regression tests for the second round of agentic retrieval review findings."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from graphrag_sdk.core.connection import ConnectionConfig
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import RetrieverError
from graphrag_sdk.core.models import (
    Entity,
    LLMResponse,
    Ontology,
    Relation,
    RetrieverResult,
    RetrieverResultItem,
)
from graphrag_sdk.retrieval.agentic import AgenticRetrieval, EvidenceLedger, ground_answer
from graphrag_sdk.retrieval.agentic.citations import DEFAULT_TIMEOUT_ANSWER, is_small_talk
from graphrag_sdk.retrieval.agentic.limits import AgentLimits
from graphrag_sdk.retrieval.agentic.loop import compact_observations
from graphrag_sdk.retrieval.agentic.tools import is_read_only_cypher
from graphrag_sdk.retrieval.graph_walk import DynamicGraphWalk
from graphrag_sdk.retrieval.strategies.path_router import HeuristicPathRouter
from graphrag_sdk.storage.graph_store import GraphStore

from .test_agentic_retrieval import ScriptedLLM, ToolCallingLLM, call
from .test_facade import mock_conn  # noqa: F401 - fixture


def ledger(n: int) -> EvidenceLedger:
    out = EvidenceLedger()
    for i in range(n):
        out.add(tool="search", kind="fact", content=f"fact {i}")
    return out


class EmptyStrategy:
    async def search(self, query: str, ctx: Context, **kwargs: Any) -> RetrieverResult:
        return RetrieverResult(items=[])


class BigStrategy:
    """Every search returns one large passage."""

    def __init__(self, size: int) -> None:
        self._size = size

    async def search(self, query: str, ctx: Context, **kwargs: Any) -> RetrieverResult:
        return RetrieverResult(items=[RetrieverResultItem(content="x" * self._size)])


# ── "No results" is not evidence ────────────────────────────────


async def test_no_results_message_is_not_citable():
    llm = ScriptedLLM(
        [
            'Thought: s\nAction: search\nAction Input: {"query": "Acme founder"}',
            "Final Answer: Acme was founded by Jane Doe in 1999 [1].",
        ]
    )
    agent = AgenticRetrieval(llm, strategy=EmptyStrategy(), mode="react", repair_citations=False)
    md = (await agent.search("Who founded Acme?", Context())).metadata
    assert md["grounded"] is False and md["grounding_reason"] == "invalid_citations"
    assert md["evidence"] == []


# ── Strict mode: only small talk may go uncited ─────────────────


class TestSmallTalk:
    @pytest.mark.parametrize(
        "message", ["hi", "Hello!", "thanks a lot", "ok", "thank you very much"]
    )
    def test_small_talk(self, message):
        assert is_small_talk(message)

    @pytest.mark.parametrize(
        "message", ["Who is the CEO of Acme?", "Acme CEO", "what can you do", "", "hi who is Bob"]
    )
    def test_not_small_talk(self, message):
        assert not is_small_talk(message)

    def test_factual_question_answered_from_memory_is_ungrounded(self):
        g = ground_answer(
            "Jane Doe.", EvidenceLedger(), tools_called=False, question="Who is the CEO of Acme?"
        )
        assert not g.grounded and g.reason == "no_citations"

    def test_greeting_reply_is_conversation(self):
        g = ground_answer(
            "Hello! How can I help?", EvidenceLedger(), tools_called=False, question="hi"
        )
        assert g.grounded and g.reason == "conversation"

    async def test_agent_passes_the_question(self):
        llm = ToolCallingLLM([LLMResponse(content="Jane Doe.")])
        agent = AgenticRetrieval(llm, strategy=EmptyStrategy(), repair_citations=False)
        md = (await agent.search("Who is the CEO of Acme?", Context())).metadata
        assert md["grounded"] is False and md["answer"] != "Jane Doe."


# ── Citation markers and cleanup ────────────────────────────────


class TestMarkers:
    @pytest.mark.parametrize(
        "answer", ["Acme was founded in 1998[2].", "Acme's (CEO)[2] is Jane.", "Berlin [1][2]."]
    )
    def test_marker_after_digit_or_paren_counts(self, answer):
        g = ground_answer(answer, ledger(2), tools_called=True)
        assert g.grounded and 2 in g.cited

    def test_markers_inside_code_are_untouched(self):
        g = ground_answer(
            "Use `arr[9]` here [1].\n```\nx = y[7]\n```", ledger(1), tools_called=True
        )
        assert g.answer == "Use `arr[9]` here [1].\n```\nx = y[7]\n```"
        assert g.invalid == []

    def test_cleanup_keeps_formatting_where_nothing_was_removed(self):
        answer = "Steps:\n- top [1]\n    - nested  item\n```\n    code\n```"
        assert ground_answer(answer, ledger(1), tools_called=True).answer == answer

    @pytest.mark.parametrize(
        ("answer", "shown"),
        [
            ("Acme [1] is big [9].", "Acme [1] is big."),
            ("word [9] [1].", "word [1]."),
            ("a [9] b [1].", "a b [1]."),
            ("plant[9] next [1].", "plant next [1]."),
            ("x [9]\nnext [1].", "x\nnext [1]."),
            ("Berlin [1][9].", "Berlin [1]."),
        ],
    )
    def test_cleanup_around_removed_markers(self, answer, shown):
        assert ground_answer(answer, ledger(1), tools_called=True).answer == shown


# ── A latency stop is reported as such ──────────────────────────


async def test_budget_stop_without_answer_says_so():
    llm = ScriptedLLM(["Action: search\nAction Input: {}"])
    agent = AgenticRetrieval(llm, strategy=EmptyStrategy(), max_steps=4)
    md = (await agent.search("q", Context(latency_budget_ms=0.0))).metadata
    assert md["stop_reason"] == "budget_exceeded"
    assert md["answer"] == DEFAULT_TIMEOUT_ANSWER
    assert md["grounded"] is False and md["grounding_reason"] == "budget_exceeded"


# ── Native tool calling falls back to ReAct ─────────────────────


class NoToolsLLM:
    """Claims tool calling, but the model rejects tools (e.g. ollama/llama2)."""

    supports_tool_calling = True
    model_name = "no-tools"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke_with_tools(self, messages, tools, **kwargs: Any) -> LLMResponse:
        raise RuntimeError("400: this model does not support tools")

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(content="Final Answer: Hello!")


async def test_auto_mode_falls_back_to_react_when_tools_are_rejected():
    agent = AgenticRetrieval(NoToolsLLM(), strategy=EmptyStrategy())
    md = (await agent.search("hi", Context())).metadata
    assert md["agent_mode"] == "react" and md["answer"] == "Hello!"


async def test_forced_native_mode_does_not_fall_back():
    agent = AgenticRetrieval(NoToolsLLM(), strategy=EmptyStrategy(), mode="native")
    with pytest.raises(RetrieverError, match="does not support tools"):
        await agent.search("hi", Context())


async def test_failure_after_tools_ran_does_not_fall_back():
    class LaterFailure(ToolCallingLLM):
        async def ainvoke_with_tools(self, messages, tools, **kwargs):
            if self._i == 1:
                raise RuntimeError("provider down")
            return await super().ainvoke_with_tools(messages, tools, **kwargs)

    llm = LaterFailure([LLMResponse(content="", tool_calls=[call("search", query="x")])])
    with pytest.raises(RetrieverError, match="provider down"):
        await AgenticRetrieval(llm, strategy=EmptyStrategy()).search("q", Context())


def test_litellm_flag_follows_the_model_map():
    from graphrag_sdk.core.providers.litellm import LiteLLM

    fake = MagicMock()
    with patch.dict("sys.modules", {"litellm": fake}):
        fake.get_model_info.return_value = {"supports_function_calling": False}
        assert LiteLLM(model="ollama/llama2").supports_tool_calling is False
        fake.get_model_info.return_value = {"supports_function_calling": None}
        assert LiteLLM(model="ollama/llama2").supports_tool_calling is True
        fake.get_model_info.side_effect = Exception("not mapped")
        assert LiteLLM(model="azure/my-deployment").supports_tool_calling is True


# ── Total tool output kept in the conversation ──────────────────


def test_compact_observations_shortens_oldest_first():
    entries = [{"full": c * 1000, "content": c * 1000} for c in "abc"]
    changed = compact_observations(entries, 2000)
    assert changed == [0, 1]
    assert sum(len(e["content"]) for e in entries) <= 2000 + 200  # truncation notes
    assert entries[-1]["content"] == "c" * 1000  # newest kept whole
    assert entries[0]["content"].startswith("a" * 400)


async def test_native_conversation_stays_within_the_total_budget():
    limits = AgentLimits(max_observation_chars=5000, max_total_observation_chars=6000)
    calls = [
        LLMResponse(content="", tool_calls=[call("search", f"c{i}", query=f"q{i}")])
        for i in range(4)
    ]
    llm = ToolCallingLLM([*calls, LLMResponse(content="Done [1].")])
    await AgenticRetrieval(llm, strategy=BigStrategy(4000), limits=limits).search("q", Context())
    last = llm.requests[-1]["messages"]
    tool_text = sum(len(m.content) for m in last if m.role == "tool")
    assert tool_text <= 6000 + 400  # within the budget, plus truncation notes
    assert last[-1].role == "tool" and "x" * 3900 in last[-1].content  # newest whole


async def test_react_scratchpad_stays_within_the_total_budget():
    limits = AgentLimits(max_observation_chars=5000, max_total_observation_chars=6000)
    turns = [f'Action: search\nAction Input: {{"query": "q{i}"}}' for i in range(4)]
    llm = ScriptedLLM([*turns, "Final Answer: Done [1]."])
    agent = AgenticRetrieval(llm, strategy=BigStrategy(4000), limits=limits, mode="react")
    await agent.search("q", Context())
    assert llm.prompts[-1].count("x") <= 6000 + 400
    assert "x" * 3900 in llm.prompts[-1]


# ── The facade's agent sees the current ontology ────────────────


async def test_agentic_retrieval_reads_the_current_ontology(mock_conn, embedder, llm):  # noqa: F811
    from graphrag_sdk import GraphRAG

    g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder, embedding_dimension=8)
    agent = g.agentic_retrieval()
    later = Ontology(entities=[Entity(label="Person")], relations=[Relation(label="WORKS_AT")])
    g._global_ontology = later
    assert agent._ontology is later
    # An explicit ontology is kept, and set_ontology replaces the getter.
    fixed = Ontology(entities=[Entity(label="Robot")])
    assert g.agentic_retrieval(ontology=fixed)._ontology is fixed
    agent.set_ontology(fixed)
    g._global_ontology = later
    assert agent._ontology is fixed


async def test_agent_as_the_facade_default_strategy_reloads(mock_conn, embedder, llm):  # noqa: F811
    from graphrag_sdk import GraphRAG

    agent = AgenticRetrieval(llm, strategy=EmptyStrategy())
    g = GraphRAG(
        connection=mock_conn,
        llm=llm,
        embedder=embedder,
        embedding_dimension=8,
        retrieval_strategy=agent,
    )
    later = Ontology(entities=[Entity(label="Person")])
    agent._ontology = later  # what get_ontology()/refresh_ontology() do
    assert agent._ontology is later
    g._global_ontology = later
    assert agent._ontology is later


def test_retrieve_sync_forwards_history(mock_conn, embedder, llm):  # noqa: F811
    from graphrag_sdk import GraphRAG

    g = GraphRAG(connection=mock_conn, llm=llm, embedder=embedder, embedding_dimension=8)
    seen: dict[str, Any] = {}

    async def fake_retrieve(question, **kwargs):
        seen.update(kwargs)
        return RetrieverResult(items=[])

    g.retrieve = fake_retrieve  # type: ignore[method-assign]
    history = [{"role": "user", "content": "hi"}]
    g.retrieve_sync("q", history=history)
    assert seen["history"] is history


# ── is_read_only_cypher shares the guard's tokenizer ────────────


@pytest.mark.parametrize(
    ("cypher", "ok"),
    [
        ("MATCH (n) // don't\nCREATE (m) // isn't\nRETURN n", False),
        ("MATCH (n) /* it's */ DELETE n", False),
        ("MATCH (n) FOREACH (x IN [1] | CREATE (:A)) RETURN n", False),
        ("MATCH (n) WHERE n.name = 'call center' RETURN n", True),
        ('MATCH (n) WHERE n.name CONTAINS "delete" RETURN n // don\'t', True),
        ("MATCH (n) RETURN 'unterminated", False),
    ],
)
def test_is_read_only_cypher(cypher, ok):
    assert is_read_only_cypher(cypher) is ok


# ── Path router entity detection ────────────────────────────────


@pytest.mark.parametrize(
    ("question", "entity"),
    [
        ("Who founded IBM?", True),
        ("Microsoft acquired which companies?", True),
        ("who makes the iPhone", True),
        ('find "acme corp" details', True),
        ("tell me about 'big co'", True),
        ("Tell me about Alice", True),
        ("what's the revenue of acme", False),
        ("What colour is the sky?", False),
        ("How many people work there?", False),
    ],
)
async def test_router_entity_detection(question, entity):
    plan = await HeuristicPathRouter().plan(question)
    assert ({"entity_cypher", "entity_fulltext"} <= plan) is entity


# ── Graph walk performance ──────────────────────────────────────


async def test_beam_round_fetches_neighbours_concurrently():
    active = 0
    peak = 0

    async def neighbours(node: str):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        if node == "s":
            return [(f"n{i}", 1.0, "R") for i in range(4)]
        return [(f"{node}x", 1.0, "R")] if len(node) < 4 else []

    paths = await DynamicGraphWalk(neighbours, beam_width=4, max_depth=2).beam_search("s")
    assert peak == 4
    assert sorted(p.nodes[-1] for p in paths) == ["n0x", "n1x", "n2x", "n3x"]


async def test_pagerank_is_cached_briefly():
    conn = MagicMock()
    calls = 0

    async def query(cypher, params=None, **kwargs):
        nonlocal calls
        calls += 1
        return MagicMock(result_set=[["a", 0.5]])

    conn.query = query
    store = GraphStore(conn)
    assert await store.pagerank() == {"a": 0.5}
    first = await store.pagerank()
    first["a"] = 99.0  # callers get a copy
    assert await store.pagerank() == {"a": 0.5}
    assert calls == 1
    await store.pagerank(max_age_seconds=0)
    assert calls == 2


async def test_pagerank_failure_is_not_cached():
    conn = MagicMock()
    results = [Exception("procedure missing"), MagicMock(result_set=[["a", 1.0]])]

    async def query(cypher, params=None, **kwargs):
        r = results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    conn.query = query
    store = GraphStore(conn)
    assert await store.pagerank() == {}
    assert await store.pagerank() == {"a": 1.0}


# ── Write keywords used as names are not writes ─────────────────


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n) RETURN n.delete",
        "MATCH (n {set: 1}) RETURN n.remove AS x",
        "MATCH (n) RETURN n.`create`",
        "MATCH (n) WHERE n.merge = $set RETURN n",
        "MATCH (n) RETURN n.call",
    ],
)
def test_write_keyword_as_a_name_is_read_only(cypher):
    from graphrag_sdk.retrieval.agentic.cypher_guard import validate_read_query

    assert validate_read_query(cypher) == [] and is_read_only_cypher(cypher)


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n) SET n.x = 1 RETURN n",
        "MATCH (n) SET n:Label RETURN n",
        "MATCH (n)\nSET\nn.a = 1 RETURN n",
        "MATCH (n) DETACH DELETE n RETURN 1",
        "MATCH (n) REMOVE n:L RETURN n",
        "MATCH (n) FOREACH (x IN [1] | SET n.a = 1) RETURN n",
        "MATCH (n) CALL db.labels() YIELD label RETURN label",
        "MATCH (n) WITH n MERGE (m:A) RETURN m",
        "MATCH (n) // x\nCREATE (m) RETURN n",
        'MATCH (n) LOAD CSV FROM "x" AS l RETURN l',
    ],
)
def test_real_writes_are_still_refused(cypher):
    from graphrag_sdk.retrieval.agentic.cypher_guard import validate_read_query

    assert validate_read_query(cypher) and not is_read_only_cypher(cypher)


# ── Third round ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Is that right?", False),
        ("yes it is", False),
        ("Is it fine?", False),
        ("how are you?", True),
        ("Thanks a lot, bye!", True),
        ("Hi there, how’s it going?", True),
        ("good morning", True),
    ],
)
def test_small_talk_is_whole_phrases(message, expected):
    assert is_small_talk(message) is expected


def test_tilde_fenced_code_is_untouched():
    g = ground_answer("Example [1]:\n~~~\nx = y[9]\n~~~", ledger(1), tools_called=True)
    assert g.answer == "Example [1]:\n~~~\nx = y[9]\n~~~" and g.invalid == []


async def test_budget_running_out_mid_run_says_so():
    from graphrag_sdk.core.exceptions import LatencyBudgetExceededError

    class OutOfTime(ScriptedLLM):
        async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
            if self._i == 1:
                raise LatencyBudgetExceededError("out of time")
            return await super().ainvoke(prompt, **kwargs)

    llm = OutOfTime(['Action: search\nAction Input: {"query": "q"}'])
    agent = AgenticRetrieval(llm, strategy=BigStrategy(10), mode="react")
    md = (await agent.search("q", Context())).metadata
    assert md["num_steps"] == 1  # a tool ran before the budget ran out
    assert md["answer"] == DEFAULT_TIMEOUT_ANSWER
    assert md["stop_reason"] == "budget_exceeded"


async def test_timeout_answer_reports_budget_exceeded_as_stop_reason():
    # The budget runs out just after the last allowed tool call: the loop
    # stops on max_tool_calls with no answer; the reason must say why.
    class Clock(Context):
        calls = 0

        @property
        def budget_exceeded(self) -> bool:
            return Clock.calls >= 1

    class CountingStrategy(BigStrategy):
        async def search(self, query, ctx, **kwargs):
            Clock.calls += 1
            return await super().search(query, ctx, **kwargs)

    limits = AgentLimits(max_tool_calls=1)
    llm = ToolCallingLLM([LLMResponse(content="", tool_calls=[call("search", query="q")])])
    agent = AgenticRetrieval(llm, strategy=CountingStrategy(10), limits=limits)
    md = (await agent.search("q", Clock())).metadata
    assert md["answer"] == DEFAULT_TIMEOUT_ANSWER
    assert md["stop_reason"] == "budget_exceeded"


@pytest.mark.parametrize(
    ("question", "entity"),
    [
        ("Amazon's revenue", True),
        ("Amazon’s revenue", True),
        ("OpenAI products", True),
        ("who founded GitHub", True),
        ("WHAT is weather?", False),
        ("HOW many people?", False),
    ],
)
async def test_router_entity_detection_round_three(question, entity):
    plan = await HeuristicPathRouter().plan(question)
    assert ({"entity_cypher", "entity_fulltext"} <= plan) is entity


async def test_walk_limits_concurrent_lookups():
    active = 0
    peak = 0

    async def neighbours(node: str):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.005)
        active -= 1
        return [(f"{node}.{i}", 1.0, "R") for i in range(40)] if node == "s" else []

    walk = DynamicGraphWalk(neighbours, beam_width=40, max_depth=2, max_concurrency=5)
    await walk.beam_search("s")
    assert peak == 5


async def test_impact_reports_more_than_fifty_direct_neighbours():
    from graphrag_sdk.skills.impact_analysis import ImpactAnalysisSkill

    class Store:
        async def weighted_neighbors(self, node_id, *, limit=50):
            hub = [(f"n{i:03d}", 1.0, "R") for i in range(120)]
            return hub[:limit] if node_id == "root" else []

        async def pagerank(self, **kwargs):
            return {}

    skill = ImpactAnalysisSkill(Store())

    async def resolve(ref):
        return ref

    skill._resolve_entity = resolve  # type: ignore[method-assign]
    skill._summarize = AsyncMockSummary()  # type: ignore[method-assign]
    result = await skill.run(Context(), entity="root", max_depth=1)
    assert result.data["num_impacted"] == 120


class AsyncMockSummary:
    async def __call__(self, ctx, prompt):
        return ""


async def test_pagerank_cache_is_dropped_when_the_graph_changes_shape():
    from graphrag_sdk.core.connection import FalkorDBConnection

    conn = FalkorDBConnection.__new__(FalkorDBConnection)
    conn.topology_version = 0
    calls = 0

    async def query(cypher, params=None, **kwargs):
        nonlocal calls
        calls += 1
        return MagicMock(result_set=[["a", 0.5]])

    conn.query = query  # type: ignore[method-assign]
    store = GraphStore(conn)
    await store.pagerank()
    await store.pagerank()
    assert calls == 1
    conn.topology_version += 1  # a write created or deleted nodes/edges
    await store.pagerank()
    assert calls == 2


def test_connection_counts_topology_changes():
    from graphrag_sdk.core.connection import _changes_topology

    assert _changes_topology(
        MagicMock(
            nodes_created=1, nodes_deleted=0, relationships_created=0, relationships_deleted=0
        )
    )
    assert _changes_topology(
        MagicMock(
            nodes_created=0, nodes_deleted=0, relationships_created=0, relationships_deleted=2
        )
    )
    assert not _changes_topology(
        MagicMock(
            nodes_created=0, nodes_deleted=0, relationships_created=0, relationships_deleted=0
        )
    )


async def test_queries_that_change_the_graph_shape_bump_the_version():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from graphrag_sdk.core.connection import FalkorDBConnection

    def result(**stats):
        base = dict.fromkeys(
            ("nodes_created", "nodes_deleted", "relationships_created", "relationships_deleted"), 0
        )
        return MagicMock(**{**base, **stats})

    conn = FalkorDBConnection(ConnectionConfig())
    graph = MagicMock()
    conn._graph = graph
    conn._ensure_client = lambda: None  # type: ignore[method-assign]
    conn.config = SimpleNamespace(query_timeout_ms=1000, retry_count=1, retry_delay=0)

    graph.query = AsyncMock(return_value=result(nodes_created=2))
    await conn.query("CREATE (:A)")
    assert conn.topology_version == 1
    graph.query = AsyncMock(return_value=result())  # SET / MATCH: shape unchanged
    await conn.query("MATCH (n) SET n.x = 1")
    assert conn.topology_version == 1
    graph.ro_query = AsyncMock(return_value=result(nodes_created=5))
    await conn.query("MATCH (n) RETURN n", read_only=True)
    assert conn.topology_version == 1
    graph.query = AsyncMock(return_value=result(relationships_deleted=1))
    await conn.query("MATCH ()-[r]->() DELETE r")
    assert conn.topology_version == 2


async def test_impact_works_with_a_store_without_a_limit_argument():
    from graphrag_sdk.skills.impact_analysis import ImpactAnalysisSkill

    class Store:
        async def weighted_neighbors(self, node_id):
            return [("a", 1.0, "R")] if node_id == "root" else []

        async def pagerank(self, **kwargs):
            return {}

    skill = ImpactAnalysisSkill(Store())

    async def resolve(ref):
        return ref

    skill._resolve_entity = resolve  # type: ignore[method-assign]
    skill._summarize = AsyncMockSummary()  # type: ignore[method-assign]
    result = await skill.run(Context(), entity="root", max_depth=2)
    assert result.data["num_impacted"] == 1
