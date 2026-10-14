"""Tests for the MCP surface: GraphRAGToolset, the SSE auth guard and the CLI."""

from __future__ import annotations

import json
from typing import Any

import pytest

from graphrag_sdk.core.models import RagResult, RetrieverResult, RetrieverResultItem
from graphrag_sdk.mcp.server import bearer_token_guard, check_sse_binding
from graphrag_sdk.mcp.tools import GraphRAGToolset


class FakeResult:
    def __init__(self, rows):
        self.result_set = rows


class FakeGraphStore:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    @property
    def last_cypher(self) -> str | None:
        return self.calls[-1]["cypher"] if self.calls else None

    async def query_raw(self, cypher: str, params: dict | None = None, **kwargs: Any):
        self.calls.append({"cypher": cypher, "params": params, **kwargs})
        return FakeResult([["alice", 1]] * 200)

    async def pagerank(self, **kwargs: Any):
        return {}

    async def weighted_neighbors(self, node_id: str, *, limit: int = 50):
        return [("acme", 1.0, "WORKS_AT")]


class FakeRAG:
    def __init__(self):
        self.graph_store = FakeGraphStore()
        self.llm = None
        self.completion_calls: list[dict[str, Any]] = []
        self.agent_options: dict[str, Any] | None = None

    async def ingest(self, source=None, *, text: str, document_id: str | None = None):
        class R:
            nodes_created = 3
            relationships_created = 2
            chunks_indexed = 1

        return R()

    async def retrieve(self, question: str):
        return RetrieverResult(items=[RetrieverResultItem(content="ctx snippet")])

    def agentic_retrieval(self, **options: Any) -> str:
        self.agent_options = options
        return "AGENT"

    async def completion(self, question: str, **kwargs: Any):
        self.completion_calls.append({"question": question, **kwargs})
        if kwargs.get("strategy") == "AGENT":
            return RagResult(
                answer="Alice works at Acme [1].",
                metadata={
                    "grounded": True,
                    "citations": [
                        {"n": 1, "tool": "search", "source": "hr.pdf", "content": "x" * 900}
                    ],
                    "agent": {"stop_reason": "final_answer", "num_steps": 1},
                },
            )
        return RagResult(answer="the answer", metadata={"model": "fake"})

    async def get_statistics(self):
        return {"node_count": 5, "edge_count": 4}

    async def get_ontology(self):
        class FakeOntology:
            def model_dump(self):
                return {"entities": []}

        return FakeOntology()


@pytest.fixture
def rag() -> FakeRAG:
    return FakeRAG()


@pytest.fixture
def toolset(rag) -> GraphRAGToolset:
    return GraphRAGToolset(rag)


class TestToolset:
    def test_tool_names(self, toolset: GraphRAGToolset):
        assert {t.name for t in toolset.tools} == {
            "ingest",
            "retrieve",
            "answer",
            "ask_agent",
            "cypher_query",
            "graph_walk",
            "run_skill",
            "get_statistics",
            "get_ontology",
        }

    def test_specs_have_schema(self, toolset: GraphRAGToolset):
        for spec in toolset.specs():
            assert "name" in spec
            assert "description" in spec
            assert spec["inputSchema"]["type"] == "object"

    def test_run_skill_lists_the_registered_skills(self, toolset: GraphRAGToolset):
        spec = toolset.by_name("run_skill").to_spec()
        assert "gap_analysis" in spec["inputSchema"]["properties"]["skill"]["enum"]
        assert "impact_analysis:" in spec["description"]

    def test_by_name_unknown_is_none(self, toolset: GraphRAGToolset):
        assert toolset.by_name("missing") is None

    def test_private_graph_store_still_works(self):
        class Old(FakeRAG):
            def __init__(self):
                super().__init__()
                self._graph_store = self.graph_store
                self.graph_store = None

        assert GraphRAGToolset(Old()).by_name("cypher_query") is not None


class TestHandlers:
    async def test_ingest(self, toolset: GraphRAGToolset):
        out = await toolset.by_name("ingest").handler({"text": "hi"})
        assert json.loads(out)["nodes_created"] == 3

    async def test_retrieve(self, toolset: GraphRAGToolset):
        out = await toolset.by_name("retrieve").handler({"question": "q"})
        assert "ctx snippet" in json.loads(out)["items"]

    async def test_answer(self, toolset: GraphRAGToolset):
        out = await toolset.by_name("answer").handler({"question": "q"})
        assert json.loads(out)["answer"] == "the answer"

    async def test_ask_agent(self, toolset: GraphRAGToolset, rag: FakeRAG):
        out = json.loads(
            await toolset.by_name("ask_agent").handler(
                {
                    "question": "Where does Alice work?",
                    "history": [{"role": "user", "content": "hi"}],
                    "max_steps": 99,
                }
            )
        )
        assert out["answer"] == "Alice works at Acme [1]."
        assert out["grounded"] is True and out["stop_reason"] == "final_answer"
        assert len(out["citations"][0]["content"]) == 500
        assert rag.agent_options == {"max_steps": 12}
        assert rag.completion_calls[-1]["history"] == [{"role": "user", "content": "hi"}]

    async def test_cypher_query_is_read_only_and_capped(self, toolset, rag):
        out = json.loads(
            await toolset.by_name("cypher_query").handler({"cypher": "MATCH (n) RETURN n.name"})
        )
        call = rag.graph_store.calls[-1]
        assert call["read_only"] is True and call["timeout"] == 5000
        assert call["cypher"].endswith("LIMIT 25")
        assert len(out["rows"]) == 25 and out["row_cap"] == 25

        await toolset.by_name("cypher_query").handler(
            {"cypher": "MATCH (n) RETURN n LIMIT 100000", "max_rows": 5000}
        )
        assert "LIMIT 100" in rag.graph_store.last_cypher

    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (n) DELETE n",
            "CALL algo.pageRank('x', null) YIELD node RETURN node",
            "MATCH (n) RETURN n.embedding",
            "MATCH (c:__GraphRAGConfig__) RETURN c",
        ],
    )
    async def test_cypher_query_rejects(self, toolset, rag, cypher):
        out = await toolset.by_name("cypher_query").handler({"cypher": cypher})
        assert out.startswith("Error: only read-only Cypher")
        assert rag.graph_store.calls == []

    async def test_graph_walk(self, toolset: GraphRAGToolset):
        out = await toolset.by_name("graph_walk").handler(
            {"start": "alice", "beam_width": 10_000, "max_depth": 999}
        )
        assert "paths" in json.loads(out)

    async def test_get_statistics(self, toolset: GraphRAGToolset):
        out = await toolset.by_name("get_statistics").handler({})
        assert json.loads(out)["node_count"] == 5

    async def test_run_skill(self, toolset: GraphRAGToolset):
        out = await toolset.by_name("run_skill").handler({"skill": "gap_analysis", "params": {}})
        assert json.loads(out)["skill"] == "gap_analysis"

    async def test_run_skill_errors_are_reported(self, toolset: GraphRAGToolset, rag: FakeRAG):
        class NoEntities(FakeGraphStore):
            async def query_raw(self, cypher, params=None, **kwargs):
                return FakeResult([])

        rag.graph_store = NoEntities()
        unknown = await toolset.by_name("run_skill").handler({"skill": "nope"})
        assert unknown.startswith("Error: Unknown skill 'nope'")
        bad = await toolset.by_name("run_skill").handler(
            {"skill": "gap_analysis", "params": {"colour": "red"}}
        )
        assert "unknown argument" in bad
        missing = await toolset.by_name("run_skill").handler(
            {"skill": "impact_analysis", "params": {"entity": "nobody"}}
        )
        assert missing == "Error: 'entity': no entity is called 'nobody'"


class TestAuth:
    async def run_app(self, app, headers):
        sent: list[dict] = []

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.request"}

        await app({"type": "http", "headers": headers}, receive, send)
        return sent

    async def test_bearer_guard(self):
        reached = []

        async def app(scope, receive, send):
            reached.append(scope)

        guarded = bearer_token_guard(app, "s3cret")
        denied = await self.run_app(guarded, [(b"authorization", b"Bearer wrong")])
        assert denied[0]["status"] == 401 and not reached
        assert (await self.run_app(guarded, []))[0]["status"] == 401
        await self.run_app(guarded, [(b"authorization", b"Bearer s3cret")])
        assert len(reached) == 1

    async def test_guard_passes_non_http_scopes(self):
        reached = []

        async def app(scope, receive, send):
            reached.append(scope["type"])

        await bearer_token_guard(app, "t")({"type": "lifespan"}, None, None)
        assert reached == ["lifespan"]

    def test_guard_needs_a_token(self):
        with pytest.raises(ValueError):
            bearer_token_guard(lambda *a: None, "")  # type: ignore[arg-type]

    def test_sse_binding_rules(self):
        check_sse_binding("127.0.0.1", None)
        check_sse_binding("0.0.0.0", "token")
        with pytest.raises(ValueError, match="without an auth token"):
            check_sse_binding("0.0.0.0", None)


class TestCLI:
    def test_build_rag_uses_the_right_constructor_arguments(self, monkeypatch):
        import graphrag_sdk.api.main as api_main
        import graphrag_sdk.core.providers as providers
        from graphrag_sdk.mcp import __main__ as cli

        seen: dict[str, Any] = {}
        monkeypatch.setenv("GRAPHRAG_MODEL", "openai/gpt-4o")
        monkeypatch.setenv("GRAPHRAG_EMBED_DIM", "3072")
        monkeypatch.setattr(api_main, "GraphRAG", lambda **kw: seen.update(kw) or "RAG")
        rag = cli._build_rag()
        assert rag == "RAG"
        assert isinstance(seen["llm"], providers.LiteLLM)
        assert seen["llm"].model_name == "openai/gpt-4o"
        assert seen["embedder"].model_name == "text-embedding-3-small"
        assert seen["embedding_dimension"] == 3072
