"""Shared test fixtures for GraphRAG SDK v2 tests."""
from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from graphrag_sdk.core.connection import FalkorDBConnection
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    EntityType,
    GraphData,
    GraphNode,
    GraphRelationship,
    GraphSchema,
    LLMResponse,
    RelationType,
)
from graphrag_sdk.core.providers import Embedder, LLMInterface


# ── Mock Providers ──────────────────────────────────────────────


class MockEmbedder(Embedder):
    """Deterministic embedder that returns a fixed-length vector derived from text hash."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-embedder"

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
        self.last_messages: list | None = None  # track ainvoke_messages calls

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        response = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return LLMResponse(content=response)

    async def ainvoke_messages(self, messages, *, max_retries=3, **kwargs):
        self.last_messages = messages
        return self.invoke("")


class MockLLMWithGraphExtraction(MockLLM):
    """Mock LLM that returns step-1 NER JSON then step-2 verify+rels JSON."""

    def __init__(self) -> None:
        step1_response = json.dumps([
            {"name": "Alice", "type": "Person", "description": "A software engineer"},
            {"name": "Acme Corp", "type": "Organization", "description": "A tech company"},
        ])
        step2_response = json.dumps({
            "entities": [
                {"name": "Alice", "type": "Person",
                 "description": "A software engineer who builds GraphRAG systems"},
                {"name": "Acme Corp", "type": "Organization",
                 "description": "A technology company specializing in AI products"},
            ],
            "relationships": [
                {
                    "source": "Alice",
                    "target": "Acme Corp",
                    "type": "WORKS_AT",
                    "description": "Alice is employed as a software engineer at Acme Corp",
                    "keywords": "employment, engineering, career",
                    "weight": 0.9,
                },
            ],
        })
        super().__init__(responses=[step1_response, step2_response])


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
def sample_schema() -> GraphSchema:
    return GraphSchema(
        entities=[
            EntityType(label="Person", description="A human being"),
            EntityType(label="Company", description="A business organization"),
        ],
        relations=[
            RelationType(
                label="WORKS_AT",
                description="Employment relationship",
                patterns=[("Person", "Company")],
            ),
            RelationType(
                label="KNOWS",
                description="Social relationship",
                patterns=[("Person", "Person")],
            ),
        ],
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
        "mention_edge_count": 0,
    })
    return store


@pytest.fixture
def mock_vector_store(embedder: MockEmbedder) -> MagicMock:
    """A mocked VectorStore."""
    from graphrag_sdk.storage.vector_store import VectorStore

    store = MagicMock(spec=VectorStore)
    store.index_chunks = AsyncMock(return_value=0)
    store.search_chunks = AsyncMock(return_value=[])
    store.search_entities = AsyncMock(return_value=[])
    store.search_relationships = AsyncMock(return_value=[])
    store.fulltext_search_chunks = AsyncMock(return_value=[])
    store.fulltext_search_entities = AsyncMock(return_value=[])
    store.create_chunk_vector_index = AsyncMock()
    store.create_entity_vector_index = AsyncMock()
    store.create_relates_vector_index = AsyncMock()
    store.create_chunk_fulltext_index = AsyncMock()
    store.create_entity_fulltext_index = AsyncMock()
    store.ensure_indices = AsyncMock(return_value={})
    store.backfill_entity_embeddings = AsyncMock(return_value=0)
    store.embed_relationships = AsyncMock(return_value=0)
    return store


# ── Real-FalkorDB integration helpers ───────────────────────────


def _scripted_extraction_llm(*per_doc_entities: list[tuple[str, str, str]]):
    """Build a MockLLM whose responses script the two-step graph_extraction
    flow once per ingest/update call.

    Each ``per_doc_entities`` arg is a list of ``(name, type, description)``
    tuples for ONE source. The helper produces 2 LLM responses per source:
    one step-1 NER response and one step-2 verify+rels response containing
    the same entity set with no relationships.

    The integration tests assume each source produces exactly one chunk
    (small sentence under FixedSizeChunking's default 1000-char limit), so
    one set of two LLM calls per source is enough.
    """
    responses: list[str] = []
    for entities in per_doc_entities:
        step1 = json.dumps(
            [{"name": n, "type": t, "description": d} for (n, t, d) in entities]
        )
        step2 = json.dumps(
            {
                "entities": [
                    {"name": n, "type": t, "description": d} for (n, t, d) in entities
                ],
                "relationships": [],
            }
        )
        responses.extend([step1, step2])
    return MockLLM(responses=responses)


@pytest.fixture
def scripted_llm():
    """Test-author-supplied scripted LLM. Use as
    ``llm = scripted_llm([("Alice", "Person", "..."), ...], [...])``.
    Returns a callable so each test can pass its own per-doc entity sets."""
    return _scripted_extraction_llm


@pytest.fixture
async def real_falkordb_rag_factory(embedder):
    """Factory that builds a fresh ``GraphRAG`` against a real FalkorDB
    with a unique graph_name (so parallel tests don't collide) and a
    caller-supplied LLM + resolver. Tests must call ``await rag.close()``
    or rely on the auto-cleanup at fixture teardown.

    Skipped unless ``RUN_INTEGRATION=1`` is set. Connects to FalkorDB at
    ``$FALKOR_HOST:$FALKOR_PORT`` (defaults ``localhost:6379``).
    """
    if not os.getenv("RUN_INTEGRATION"):
        pytest.skip("Set RUN_INTEGRATION=1 to run real-FalkorDB integration tests")

    from graphrag_sdk.api.main import GraphRAG
    from graphrag_sdk.core.connection import ConnectionConfig

    created: list[Any] = []

    def _make(*, llm, resolver, schema=None):
        config = ConnectionConfig(
            host=os.getenv("FALKOR_HOST", "localhost"),
            port=int(os.getenv("FALKOR_PORT", "6379")),
            username=os.getenv("FALKOR_USERNAME") or None,
            password=os.getenv("FALKOR_PASSWORD") or None,
            graph_name=f"test_{uuid4().hex[:8]}",
        )
        kwargs = dict(
            connection=config,
            llm=llm,
            embedder=embedder,
            embedding_dimension=embedder.dimension,
        )
        if schema is not None:
            kwargs["schema"] = schema
        rag = GraphRAG(**kwargs)
        # Per-call resolver injection (apply_changes / update / ingest don't
        # accept a default-resolver kwarg on the facade — but each call does).
        rag._test_resolver = resolver  # marker, not used by SDK
        created.append(rag)
        return rag

    yield _make

    # Teardown — drop every test graph so reruns start clean.
    for rag in created:
        try:
            await rag._graph_store.delete_all()
        except Exception:
            pass
        try:
            await rag.close()
        except Exception:
            pass
