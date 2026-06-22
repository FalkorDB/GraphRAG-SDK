"""Tests for mcp/tools.py — GraphRAGToolset (Phase 3.2)."""
from __future__ import annotations

import json
from typing import Any

import pytest

from graphrag_sdk.core.models import RagResult, RetrieverResult, RetrieverResultItem
from graphrag_sdk.mcp.tools import GraphRAGToolset


class FakeResult:
    def __init__(self, rows):
        self.result_set = rows


class FakeGraphStore:
    def __init__(self):
        self.last_cypher: str | None = None

    async def query_raw(self, cypher: str, params: dict | None = None):
        self.last_cypher = cypher
        return FakeResult([["alice", 1]])

    async def pagerank(self, **kwargs: Any):
        return {}

    async def weighted_neighbors(self, node_id: str, *, limit: int = 50):
        return [("acme", 1.0, "WORKS_AT")]


class FakeRAG:
    def __init__(self):
        self._graph_store = FakeGraphStore()
        self.llm = None

    async def ingest(self, text: str, document_id: str | None = None):
        class R:
            nodes_created = 3
            relationships_created = 2
            chunks_indexed = 1

        return R()

    async def retrieve(self, question: str):
        return RetrieverResult(items=[RetrieverResultItem(content="ctx snippet")])

    async def completion(self, question: str):
        return RagResult(answer="the answer", metadata={"model": "fake"})

    async def get_statistics(self):
        return {"node_count": 5, "edge_count": 4}

    async def get_ontology(self):
        class FakeOntology:
            def model_dump(self):
                return {"entities": []}

        return FakeOntology()


@pytest.fixture
def toolset() -> GraphRAGToolset:
    return GraphRAGToolset(FakeRAG())


class TestToolset:
    def test_exposes_eight_tools(self, toolset: GraphRAGToolset):
        assert len(toolset.tools) == 8

    def test_tool_names(self, toolset: GraphRAGToolset):
        names = {t.name for t in toolset.tools}
        assert names == {
            "ingest",
            "retrieve",
            "answer",
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

    def test_by_name_unknown_is_none(self, toolset: GraphRAGToolset):
        assert toolset.by_name("missing") is None


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

    async def test_cypher_query_read_only_passes(self, toolset: GraphRAGToolset):
        out = await toolset.by_name("cypher_query").handler(
            {"cypher": "MATCH (n) RETURN n"}
        )
        assert "rows" in json.loads(out)

    async def test_cypher_query_rejects_writes(self, toolset: GraphRAGToolset):
        out = await toolset.by_name("cypher_query").handler(
            {"cypher": "MATCH (n) DELETE n"}
        )
        assert "read-only" in out

    async def test_graph_walk(self, toolset: GraphRAGToolset):
        out = await toolset.by_name("graph_walk").handler({"start": "alice"})
        assert "paths" in json.loads(out)

    async def test_get_statistics(self, toolset: GraphRAGToolset):
        out = await toolset.by_name("get_statistics").handler({})
        assert json.loads(out)["node_count"] == 5

    async def test_run_skill(self, toolset: GraphRAGToolset):
        out = await toolset.by_name("run_skill").handler(
            {"skill": "gap_analysis", "params": {}}
        )
        assert json.loads(out)["skill"] == "gap_analysis"
