"""Regression tests for the agentic hardening fixes (review findings)."""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import DatabaseError
from graphrag_sdk.core.models import LLMResponse, RetrieverResult, RetrieverResultItem, ToolCall
from graphrag_sdk.retrieval.agentic import (
    AgenticRetrieval,
    EvidenceLedger,
    ToolRefusal,
    enforce_row_cap,
    ground_answer,
    validate_read_query,
)
from graphrag_sdk.retrieval.agentic.graph_tools import (
    _resolve_start,
    format_value,
    make_traverse_tool,
)
from graphrag_sdk.retrieval.agentic.tools import ToolContext

# ── H2: guard bypasses ───────────────────────────────────────────


class TestGuardBypasses:
    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (n) // '\nCALL db.labels() YIELD label // '\nRETURN label",
            "MATCH (c:Chunk) // RETURN '\nRETURN c.embedding AS v // '\n",
            "MATCH (n) /* ' */ DELETE n /* ' */ RETURN 1",
            "MATCH (c:Chunk) RETURN c['embedding']",
            "MATCH (c:Chunk) RETURN [k IN keys(c) | c[k]]",
            "MATCH (c:Chunk) RETURN properties(c)['embedding']",
            "MATCH (n) RETURN n[$p]",
            "MATCH (n) WHERE '__GraphRAGConfig__' IN labels(n) RETURN n",
            "MATCH (n) RETURN n /* unterminated",
            "MATCH (n) RETURN 'unterminated",
            "MATCH (n) RETURN `unterminated",
            "MATCH (c:Chunk) RETURN `c`['embedding']",
            "MATCH (c:Chunk) WITH c AS `IN` RETURN `IN`['embedding']",
            "MATCH (c:Chunk) RETURN vec.euclideanDistance(`c`['embedding'], vecf32([0.1]))",
        ],
    )
    def test_rejected(self, cypher):
        assert validate_read_query(cypher), cypher

    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (n) RETURN labels(n)[0] AS l",
            "MATCH (n) RETURN collect(n.name)[0..3] AS first",
            "MATCH (n) WHERE n.name IN ['a', 'b'] RETURN [x IN collect(n) WHERE x:Person | x.name]",
            "MATCH (n) // a comment with a 'quote\nRETURN n.name",
            "MATCH (p:Person) WHERE p.name CONTAINS 'call; delete /* x' RETURN p.name",
            "MATCH (n) RETURN [node IN [n] WHERE node:Person | node.name] AS x",
            "MATCH p = (a)-[*1..2]-(b) RETURN nodes(p)[1].name",
            "MATCH (n) WHERE n.text CONTAINS 'embedding' RETURN n.name",
        ],
    )
    def test_still_accepted(self, cypher):
        assert validate_read_query(cypher) == [], cypher

    def test_list_comprehension_label_is_not_mangled(self):
        from graphrag_sdk.retrieval.agentic import GraphSchema

        schema = GraphSchema(
            labels=["Person"], relationship_types=["RELATES"], property_keys=["name"]
        )
        cypher = "MATCH (n) RETURN [node IN [n] WHERE node:Person | node.name] AS x"
        assert validate_read_query(cypher, schema) == []


# ── M1: row cap ──────────────────────────────────────────────────


class TestRowCap:
    def test_comment_cannot_hide_a_missing_limit(self):
        assert enforce_row_cap("MATCH (n) RETURN n.name // LIMIT 5", 25) == (
            "MATCH (n) RETURN n.name\nLIMIT 25"
        )

    def test_limit_inside_a_string_is_data(self):
        out = enforce_row_cap("MATCH (n) WHERE n.name CONTAINS 'LIMIT 100' RETURN n LIMIT 500", 25)
        assert out == "MATCH (n) WHERE n.name CONTAINS 'LIMIT 100' RETURN n LIMIT 25"

    def test_comments_are_dropped_from_what_runs(self):
        out = enforce_row_cap("MATCH (n) /* hi */ RETURN n LIMIT 3", 25)
        assert "/*" not in out and out.endswith("LIMIT 3")

    def test_unterminated_input_raises(self):
        with pytest.raises(ValueError):
            enforce_row_cap("MATCH (n) RETURN 'x", 25)


def test_internal_config_node_is_never_rendered():
    node = SimpleNamespace(labels=["__GraphRAGConfig__"], properties={"sdk_version": "1"})
    assert format_value(node) == "[internal node omitted]"
    props = {"id": "default", "sdk_version": "1", "embedding_dimension": 8, "updated_at": "x"}
    assert format_value(props) == "[internal node omitted]"
    assert format_value({"name": "Acme"}) == {"name": "Acme"}


def test_vectors_are_never_rendered():
    assert format_value([0.1] * 64) == "[vector of 64 numbers omitted]"
    assert format_value([1, 2, 3]) == [1, 2, 3]
    assert format_value({"v": [0.5] * 100})["v"] == "[vector of 100 numbers omitted]"


# ── H1: read-only queries never trip the breaker ─────────────────


class TestReadOnlyFailFast:
    def conn(self, side_effect):
        from graphrag_sdk.core.connection import FalkorDBConnection

        conn = FalkorDBConnection.__new__(FalkorDBConnection)
        graph = MagicMock()
        graph.ro_query = AsyncMock(side_effect=side_effect)
        graph.query = AsyncMock(side_effect=side_effect)
        conn._graph = graph
        conn._ensure_client = lambda: None  # type: ignore[method-assign]
        breaker = MagicMock()
        breaker.allow_request = AsyncMock(return_value=True)
        breaker.record_success = AsyncMock()
        breaker.record_failure = AsyncMock()
        conn._breaker = breaker
        conn.config = SimpleNamespace(query_timeout_ms=1000, retry_count=3, retry_delay=0)
        return conn, graph, breaker

    async def test_server_rejection_fails_once_without_breaker(self):
        from redis.exceptions import ResponseError

        conn, graph, breaker = self.conn(ResponseError("Variable `m` not defined"))
        with pytest.raises(DatabaseError, match="not defined"):
            await conn.query("MATCH (n) RETURN m", read_only=True)
        assert graph.ro_query.await_count == 1
        breaker.record_failure.assert_not_awaited()

    async def test_network_errors_are_still_retried(self):
        conn, graph, breaker = self.conn(ConnectionError("reset"))
        with pytest.raises(DatabaseError):
            await conn.query("MATCH (n) RETURN n", read_only=True)
        assert graph.ro_query.await_count == 3
        assert breaker.record_failure.await_count == 3

    async def test_cluster_redirects_are_still_retried(self):
        from redis.exceptions import TryAgainError

        conn, graph, breaker = self.conn(TryAgainError("TRYAGAIN"))
        with pytest.raises(DatabaseError):
            await conn.query("MATCH (n) RETURN n", read_only=True)
        assert graph.ro_query.await_count == 3

    async def test_regular_queries_keep_their_retry_behaviour(self):
        from redis.exceptions import ResponseError

        conn, graph, breaker = self.conn(ResponseError("Variable `m` not defined"))
        with pytest.raises(DatabaseError):
            await conn.query("MATCH (n) RETURN m")
        assert graph.query.await_count == 3


# ── M2: grounding ────────────────────────────────────────────────


def ledger(n: int) -> EvidenceLedger:
    out = EvidenceLedger()
    for i in range(n):
        out.add(tool="search", kind="fact", content=f"fact {i}")
    return out


class TestGrounding:
    def test_figures_without_tools_are_not_conversation(self):
        g = ground_answer("Acme was founded in 1999.", EvidenceLedger(), tools_called=False)
        assert not g.grounded and g.reason == "no_citations"

    def test_empty_answer_is_ungrounded(self):
        g = ground_answer("", EvidenceLedger(), tools_called=False)
        assert not g.grounded and g.reason == "empty"
        assert ground_answer("", ledger(0), tools_called=False, policy="annotate").answer == ""

    def test_years_and_code_are_not_citations(self):
        g = ground_answer("In [2024] use items[0] and f(x)[1] [1].", ledger(1), tools_called=True)
        assert g.answer == "In [2024] use items[0] and f(x)[1] [1]."
        assert g.cited == [1] and g.invalid == []

    def test_glued_and_chained_markers_are_citations(self):
        g = ground_answer("Acme ships to the Berlin plant[1].", ledger(1), tools_called=True)
        assert g.grounded and g.cited == [1]
        g = ground_answer("Acme ships to Berlin [1][9].", ledger(1), tools_called=True)
        assert g.answer == "Acme ships to Berlin [1]." and g.invalid == [9]


# ── M4: Python 3.10 asyncio.TimeoutError ─────────────────────────


async def test_asyncio_timeout_is_a_time_budget_stop():
    class Store:
        async def query_raw(self, cypher, params=None, **kw):
            if "e.id = $k" in cypher:
                return SimpleNamespace(result_set=[["a", "A"]])
            raise asyncio.TimeoutError()

    out = await make_traverse_tool(Store()).handler({"start": "a"}, ToolContext(ctx=Context()))
    assert "stopped: time_budget" in out.content


# ── L3: start resolution respects the walk's deadline ────────────


async def test_resolution_uses_the_remaining_budget():
    seen: list[int] = []

    class Store:
        async def query_raw(self, cypher, params=None, *, read_only=False, timeout=None):
            seen.append(timeout)
            return SimpleNamespace(result_set=[["a", "A"]])

    await _resolve_start(Store(), "a", time.monotonic() + 0.2)
    assert seen and seen[0] <= 200
    with pytest.raises(ToolRefusal, match="time budget"):
        await _resolve_start(Store(), "a", time.monotonic() - 1)


# ── M5: OpenAI-shaped tool calls ─────────────────────────────────


def test_llm_response_accepts_openai_tool_call_dicts():
    resp = LLMResponse(
        content="",
        tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "s", "arguments": "{bad"}}
        ],
    )
    call = resp.tool_calls[0]
    assert call.name == "s" and call.parse_error and call.raw_arguments == "{bad"


# ── L7: bad JSON is not echoed back to the provider ──────────────


async def test_bad_json_arguments_are_echoed_as_empty_object(ctx):
    class Strategy:
        async def search(self, q, ctx, **kw):
            return RetrieverResult(items=[RetrieverResultItem(content="x")])

    class LLM:
        supports_tool_calling = True
        model_name = "m"

        def __init__(self):
            self.requests: list[Any] = []

        async def ainvoke_with_tools(self, messages, tools, **kw):
            self.requests.append(list(messages))
            if len(self.requests) == 1:
                bad = ToolCall(id="c", name="search", raw_arguments="{oops", parse_error="bad")
                return LLMResponse(content="", tool_calls=[bad])
            return LLMResponse(content="done")

    llm = LLM()
    await AgenticRetrieval(llm, strategy=Strategy()).search("q", ctx)
    assistant = [m for m in llm.requests[1] if m.role == "assistant"][0]
    assert assistant.to_dict()["tool_calls"][0]["function"]["arguments"] == "{}"


# ── L1 / L4 / L5 / M3 ────────────────────────────────────────────


def test_reserved_skill_names_are_refused():
    from graphrag_sdk.skills import GapAnalysisSkill, register_skill

    for name in ("search", "query_graph", "traverse", "lookup_entity"):
        with pytest.raises(ValueError, match="reserved"):
            register_skill(type("S", (GapAnalysisSkill,), {"name": name}))


def test_sse_security_settings():
    pytest.importorskip("mcp.server.transport_security")
    from graphrag_sdk.mcp.server import sse_security_kwargs

    local = sse_security_kwargs("127.0.0.1", 8080)["security_settings"]
    assert local.enable_dns_rebinding_protection is True
    assert local.allowed_hosts == ["127.0.0.1:8080", "localhost:8080"]
    assert sse_security_kwargs("::1", 1)["security_settings"].allowed_hosts[1] == "[::1]:1"
    # Off loopback the bearer token protects the server; clients use any name.
    assert sse_security_kwargs("0.0.0.0", 8080) == {}


class TestFacadeEdges:
    @pytest.fixture
    def rag(self):
        from graphrag_sdk.api.main import GraphRAG
        from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection

        from .conftest import MockEmbedder, MockLLM

        conn = MagicMock(spec=FalkorDBConnection)
        result = MagicMock()
        result.result_set = []
        conn.query = AsyncMock(return_value=result)
        conn.config = ConnectionConfig()
        graph = MagicMock()
        graph.query = AsyncMock(return_value=result)
        conn._driver = MagicMock()
        conn._driver.select_graph = MagicMock(return_value=graph)
        conn._ensure_client = MagicMock()
        return GraphRAG(
            connection=conn,
            llm=MockLLM(),
            embedder=MockEmbedder(dimension=8),
            embedding_dimension=8,
        )

    async def test_caller_ontology_is_kept(self, rag):
        from graphrag_sdk.core.models import Ontology

        strategy = MagicMock()
        strategy._ontology = Ontology()
        strategy.search = AsyncMock(return_value=RetrieverResult(items=[]))
        await rag.retrieve("q", strategy=strategy)
        strategy.set_ontology.assert_not_called()

    async def test_prompt_template_with_agent_warns(self, rag, caplog):
        from graphrag_sdk.core.models import RawSearchResult
        from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy

        class Agent(RetrievalStrategy):
            accepts_history = True

            async def _execute(self, query, ctx, **kw):
                return RawSearchResult(
                    records=[], metadata={"answer": "a", "answer_is_final": True}
                )

            def _format(self, raw):
                return RetrieverResult(items=[], metadata=raw.metadata)

        with caplog.at_level(logging.WARNING):
            result = await rag.completion(
                "q", strategy=Agent(), prompt_template="{context}{question}"
            )
        assert result.answer == "a"
        assert "not used with an agentic strategy" in caplog.text
