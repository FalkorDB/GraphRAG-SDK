"""Shared test fixtures for GraphRAG SDK v2 tests."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Type
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    DocumentInfo,
    DocumentOutput,
    EntityType,
    GraphData,
    GraphNode,
    GraphRelationship,
    GraphSchema,
    LLMResponse,
    RelationType,
    ResolutionResult,
    RetrieverResult,
    RetrieverResultItem,
    SchemaPattern,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.core.providers import Embedder, LLMInterface


# ── Mock Providers ──────────────────────────────────────────────


class MockEmbedder(Embedder):
    """Deterministic embedder that returns a fixed-length vector derived from text hash."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.call_count = 0

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        self.call_count += 1
        h = hash(text) % (10**9)
        return [(h >> i & 0xFF) / 255.0 for i in range(self.dimension)]


class MockLLM(LLMInterface):
    """Mock LLM that returns pre-configured responses."""

    def __init__(
        self,
        responses: list[str] | None = None,
        model_name: str = "mock-llm",
    ) -> None:
        super().__init__(model_name=model_name)
        self._responses = responses or ['{"nodes": [], "relationships": []}']
        self._call_index = 0

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        response = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return LLMResponse(content=response)


class MockLLMWithExtraction(MockLLM):
    """Mock LLM that returns a realistic extraction response."""

    def __init__(self) -> None:
        extraction_response = json.dumps({
            "nodes": [
                {"id": "alice", "label": "Person", "properties": {"name": "Alice"}},
                {"id": "bob", "label": "Person", "properties": {"name": "Bob"}},
                {"id": "acme", "label": "Company", "properties": {"name": "Acme Corp"}},
            ],
            "relationships": [
                {
                    "start_node_id": "alice",
                    "end_node_id": "acme",
                    "type": "WORKS_AT",
                    "properties": {},
                },
                {
                    "start_node_id": "bob",
                    "end_node_id": "acme",
                    "type": "WORKS_AT",
                    "properties": {},
                },
            ],
        })
        super().__init__(responses=[extraction_response])


class MockLLMWithMergedExtraction(MockLLM):
    """Mock LLM that returns a realistic merged-extraction (delimiter) response."""

    def __init__(self) -> None:
        merged_response = (
            '("entity"<|#|>Alice<|#|>Person<|#|>A software engineer)##'
            '("entity"<|#|>Acme Corp<|#|>Company<|#|>A tech company)##'
            '("relationship"<|#|>Alice<|#|>Acme Corp<|#|>WORKS_AT'
            "<|#|>employment<|#|>Alice works at Acme Corp<|#|>0.9)##"
        )
        super().__init__(responses=[merged_response])


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def ctx() -> Context:
    """A fresh Context with default settings."""
    return Context(tenant_id="test-tenant", latency_budget_ms=5000.0)


@pytest.fixture
def embedder() -> MockEmbedder:
    return MockEmbedder(dimension=8)


@pytest.fixture
def llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def llm_with_extraction() -> MockLLMWithExtraction:
    return MockLLMWithExtraction()


@pytest.fixture
def sample_schema() -> GraphSchema:
    return GraphSchema(
        entities=[
            EntityType(label="Person", description="A human being"),
            EntityType(label="Company", description="A business organization"),
        ],
        relations=[
            RelationType(label="WORKS_AT", description="Employment relationship"),
            RelationType(label="KNOWS", description="Social relationship"),
        ],
        patterns=[
            SchemaPattern(source="Person", relationship="WORKS_AT", target="Company"),
            SchemaPattern(source="Person", relationship="KNOWS", target="Person"),
        ],
    )


@pytest.fixture
def sample_text() -> str:
    return (
        "Alice is a software engineer at Acme Corp. "
        "Bob is a product manager at Acme Corp. "
        "Alice and Bob work together on the GraphRAG project. "
        "They have been collaborating for about two years."
    )


@pytest.fixture
def sample_chunks() -> TextChunks:
    return TextChunks(
        chunks=[
            TextChunk(text="Alice is a software engineer at Acme Corp.", index=0, uid="chunk-0"),
            TextChunk(text="Bob is a product manager at Acme Corp.", index=1, uid="chunk-1"),
            TextChunk(
                text="Alice and Bob work together on the GraphRAG project.",
                index=2,
                uid="chunk-2",
            ),
        ]
    )


@pytest.fixture
def sample_graph_data() -> GraphData:
    return GraphData(
        nodes=[
            GraphNode(id="alice", label="Person", properties={"name": "Alice"}),
            GraphNode(id="bob", label="Person", properties={"name": "Bob"}),
            GraphNode(id="acme", label="Company", properties={"name": "Acme Corp"}),
        ],
        relationships=[
            GraphRelationship(
                start_node_id="alice",
                end_node_id="acme",
                type="WORKS_AT",
                properties={},
            ),
            GraphRelationship(
                start_node_id="bob",
                end_node_id="acme",
                type="WORKS_AT",
                properties={},
            ),
        ],
    )


@pytest.fixture
def sample_graph_data_with_duplicates() -> GraphData:
    return GraphData(
        nodes=[
            GraphNode(id="alice-1", label="Person", properties={"name": "Alice", "role": "engineer"}),
            GraphNode(id="alice-2", label="Person", properties={"name": "Alice", "age": 30}),
            GraphNode(id="bob", label="Person", properties={"name": "Bob"}),
            GraphNode(id="acme", label="Company", properties={"name": "Acme Corp"}),
        ],
        relationships=[
            GraphRelationship(
                start_node_id="alice-1", end_node_id="acme", type="WORKS_AT", properties={},
            ),
            GraphRelationship(
                start_node_id="alice-2", end_node_id="acme", type="WORKS_AT", properties={},
            ),
            GraphRelationship(
                start_node_id="bob", end_node_id="acme", type="WORKS_AT", properties={},
            ),
        ],
    )


@pytest.fixture
def mock_connection() -> MagicMock:
    """A mocked FalkorDBConnection that records queries."""
    conn = MagicMock(spec=FalkorDBConnection)
    result_mock = MagicMock()
    result_mock.result_set = []
    conn.query = AsyncMock(return_value=result_mock)
    return conn


@pytest.fixture
def mock_graph_store(mock_connection: MagicMock) -> MagicMock:
    """A mocked GraphStore."""
    from graphrag_sdk.storage.graph_store import GraphStore

    store = MagicMock(spec=GraphStore)
    store.upsert_nodes = AsyncMock(return_value=0)
    store.upsert_relationships = AsyncMock(return_value=0)
    store.get_connected_entities = AsyncMock(return_value=[])
    store.query_raw = AsyncMock(return_value=MagicMock(result_set=[]))
    store.delete_all = AsyncMock()
    store.get_statistics = AsyncMock(return_value={
        "node_count": 0, "edge_count": 0, "entity_types": [],
        "relationship_types": [], "graph_density": 0,
        "fact_node_count": 0, "synonym_edge_count": 0, "mention_edge_count": 0,
    })
    return store


@pytest.fixture
def mock_vector_store(embedder: MockEmbedder) -> MagicMock:
    """A mocked VectorStore."""
    from graphrag_sdk.storage.vector_store import VectorStore

    store = MagicMock(spec=VectorStore)
    store.index_chunks = AsyncMock(return_value=0)
    store.index_facts = AsyncMock(return_value=0)
    store.search = AsyncMock(return_value=[])
    store.search_entities = AsyncMock(return_value=[])
    store.search_facts = AsyncMock(return_value=[])
    store.fulltext_search = AsyncMock(return_value=[])
    store.create_vector_index = AsyncMock()
    store.create_entity_vector_index = AsyncMock()
    store.create_fact_vector_index = AsyncMock()
    store.create_fulltext_index = AsyncMock()
    store.ensure_indices = AsyncMock(return_value={})
    store.backfill_entity_embeddings = AsyncMock(return_value=0)
    return store
