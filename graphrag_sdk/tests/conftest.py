"""Shared test fixtures for GraphRAG SDK v2 tests."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

_logger = logging.getLogger(__name__)

from graphrag_sdk.core.connection import FalkorDBConnection
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    Entity,
    GraphData,
    GraphNode,
    GraphRelationship,
    Ontology,
    LLMResponse,
    Relation,
)
from graphrag_sdk.core.providers import Embedder, LLMInterface


# ── Mock Providers ──────────────────────────────────────────────


class MockEmbedder(Embedder):
    """Deterministic embedder that returns a fixed-length vector derived from a text digest.

    A stable digest rather than ``hash()``, which Python salts per process: the
    same text embeds the same across runs. Bytes are centred on zero so two
    unrelated texts are near-orthogonal rather than all-positive — with
    ``finalize()`` judging close pairs by default, an all-positive vector space
    made unrelated names score as duplicates on some hash seeds and not others.
    """

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-embedder"

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        self.call_count += 1
        # blake2b digests are at most 64 bytes; a wider vector is several
        # digests of the same text under successive salts.
        data = text.encode("utf-8")
        digest = b"".join(
            hashlib.blake2b(data, digest_size=64, salt=str(block).encode()).digest()
            for block in range(-(-self.dimension // 64))
        )
        return [byte / 127.5 - 1.0 for byte in digest[: self.dimension]]


class MockLLM(LLMInterface):
    """Mock LLM that returns pre-configured responses.

    By default, calls beyond the configured response list clamp to the
    LAST response (so a default empty MockLLM can be called any number
    of times). Pass ``strict=True`` to raise instead — useful for
    scripted-extraction tests where over-call indicates the script no
    longer matches the production code path (e.g. chunking now produces
    multiple chunks so the LLM is invoked more times than scripted).
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        model_name: str = "mock-llm",
        *,
        strict: bool = False,
    ) -> None:
        super().__init__(model_name=model_name)
        self._responses = responses or ['{"nodes": [], "relationships": []}']
        self._call_index = 0
        self._strict = strict
        self.last_messages: list | None = None  # track ainvoke_messages calls

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        if self._strict and self._call_index >= len(self._responses):
            raise AssertionError(
                f"MockLLM(strict=True): call #{self._call_index + 1} exceeds "
                f"{len(self._responses)} scripted responses. The production "
                "code is calling the LLM more times than the test scripted "
                "for — typically because chunking produced more chunks than "
                "expected. Update the script or fix the chunking assumption."
            )
        response = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return LLMResponse(content=response)

    async def ainvoke_messages(self, messages, *, max_retries=3, **kwargs):
        self.last_messages = messages
        return self.invoke("")


class MockLLMWithGraphExtraction(MockLLM):
    """Mock LLM that returns step-1 NER JSON then step-2 verify+rels JSON."""

    def __init__(self) -> None:
        step1_response = json.dumps(
            [
                {"name": "Alice", "type": "Person", "description": "A software engineer"},
                {"name": "Acme Corp", "type": "Organization", "description": "A tech company"},
            ]
        )
        step2_response = json.dumps(
            {
                "entities": [
                    {
                        "name": "Alice",
                        "type": "Person",
                        "description": "A software engineer who builds GraphRAG systems",
                    },
                    {
                        "name": "Acme Corp",
                        "type": "Organization",
                        "description": "A technology company specializing in AI products",
                    },
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
            }
        )
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
def sample_ontology() -> Ontology:
    return Ontology(
        entities=[
            Entity(label="Person", description="A human being"),
            Entity(label="Company", description="A business organization"),
        ],
        relations=[
            Relation(
                label="WORKS_AT",
                description="Employment relationship",
                patterns=[("Person", "Company")],
            ),
            Relation(
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
            GraphNode(
                id="alice-1", label="Person", properties={"name": "Alice", "role": "engineer"}
            ),
            GraphNode(id="alice-2", label="Person", properties={"name": "Alice", "age": 30}),
            GraphNode(id="bob", label="Person", properties={"name": "Bob"}),
            GraphNode(id="acme", label="Company", properties={"name": "Acme Corp"}),
        ],
        relationships=[
            GraphRelationship(
                start_node_id="alice-1",
                end_node_id="acme",
                type="WORKS_AT",
                properties={},
            ),
            GraphRelationship(
                start_node_id="alice-2",
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
    # Report every item as written, like the real store does on success —
    # the pipeline treats a short count as a partial failure and withholds
    # the Document's content_hash.
    store.upsert_nodes = AsyncMock(side_effect=len)
    store.upsert_relationships = AsyncMock(side_effect=len)
    store.get_connected_entities = AsyncMock(return_value=[])
    store.query_raw = AsyncMock(return_value=MagicMock(result_set=[]))
    store.delete_all = AsyncMock()
    store.get_statistics = AsyncMock(
        return_value={
            "node_count": 0,
            "edge_count": 0,
            "entity_types": [],
            "relationship_types": [],
            "graph_density": 0,
            "mention_edge_count": 0,
        }
    )
    return store


@pytest.fixture
def mock_vector_store(embedder: MockEmbedder) -> MagicMock:
    """A mocked VectorStore."""
    from graphrag_sdk.storage.vector_store import VectorStore

    store = MagicMock(spec=VectorStore)
    store.index_chunks = AsyncMock(side_effect=lambda chunks: len(chunks.chunks))
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

_DEFAULT_PORT = 6379
_DEFAULT_PORT_OPT_IN = "FALKOR_TEST_ALLOW_6379"


def falkordb_test_server() -> dict[str, Any]:
    """The FalkorDB server the integration tests run against, as the ``host``,
    ``port``, ``username`` and ``password`` keyword arguments of
    ``ConnectionConfig`` or a ``Redis`` client. Read from ``$FALKOR_HOST``,
    ``$FALKOR_PORT``, ``$FALKOR_USERNAME`` and ``$FALKOR_PASSWORD``.

    The tests create and drop ``test_*`` graphs on that server, and 6379, the
    default port, is where a local FalkorDB with real data usually listens. So
    with ``RUN_INTEGRATION`` set, port 6379 stops the run before anything
    connects, unless ``FALKOR_TEST_ALLOW_6379=1`` says the server there is
    disposable.
    """
    port_value = os.getenv("FALKOR_PORT", str(_DEFAULT_PORT))
    try:
        port = int(port_value)
    except ValueError:
        pytest.exit(
            f"FALKOR_PORT={port_value!r} is not a port number.",
            returncode=pytest.ExitCode.USAGE_ERROR,
        )
    if (
        os.getenv("RUN_INTEGRATION")
        and port == _DEFAULT_PORT
        and os.getenv(_DEFAULT_PORT_OPT_IN) != "1"
    ):
        pytest.exit(
            f"Refusing to run the integration tests on port {_DEFAULT_PORT}: they "
            "create and drop graphs on the server, and a local FalkorDB with real "
            "data usually listens there. Point FALKOR_PORT (and FALKOR_HOST if "
            f"needed) at a disposable server, or set {_DEFAULT_PORT_OPT_IN}=1 if "
            f"the server on port {_DEFAULT_PORT} is disposable.",
            returncode=pytest.ExitCode.USAGE_ERROR,
        )
    return {
        "host": os.getenv("FALKOR_HOST", "localhost"),
        "port": port,
        "username": os.getenv("FALKOR_USERNAME") or None,
        "password": os.getenv("FALKOR_PASSWORD") or None,
    }


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """Refuse the server before collection, not at the first test that connects."""
    if os.getenv("RUN_INTEGRATION"):
        falkordb_test_server()


def pytest_report_header(config):
    """Name the FalkorDB server and engine the integration tests run against.

    FalkorDB 4.x (C engine) and 6.x (Rust engine) differ in index DDL, error
    text and row order, so a result means little without knowing which ran.
    """
    if not os.getenv("RUN_INTEGRATION"):
        return None
    server = falkordb_test_server()
    host, port = server["host"], server["port"]
    try:
        from redis import Redis

        client = Redis(**server, socket_timeout=5)
        modules = client.module_list()
        client.close()
        ver = int(next(m for m in modules if m[b"name"] == b"graph")[b"ver"])
    except Exception as exc:  # the header must never stop the run
        return f"falkordb: {host}:{port} not readable ({exc})"
    version = f"{ver // 10000}.{ver // 100 % 100}.{ver % 100}"
    engine = "C engine" if ver < 50000 else "Rust engine"
    return f"falkordb: {host}:{port} — {version} ({engine})"


def _scripted_extraction_llm(*per_doc_entities: list[tuple[str, str, str]]):
    """Build a MockLLM whose responses script the verify+relations step
    of ``GraphExtraction`` once per ingest/update call.

    Each ``per_doc_entities`` arg is a list of ``(name, type, description)``
    tuples for ONE source. The default ``GraphExtraction`` runs step-1
    NER locally via GLiNER (no LLM call) and step-2 verify+rels via the
    injected LLM — so we script exactly ONE response per source: the
    step-2 ``{"entities": [...], "relationships": []}`` JSON, which the
    pipeline merges with GLiNER's step-1 output to produce the final
    entity set.

    Tests assume each source produces exactly one chunk (small sentence
    under FixedSizeChunking's default 1000-char limit), so a single LLM
    call per source is enough.
    """
    responses: list[str] = []
    for entities in per_doc_entities:
        step2 = json.dumps(
            {
                "entities": [{"name": n, "type": t, "description": d} for (n, t, d) in entities],
                "relationships": [],
            }
        )
        responses.append(step2)
    # strict=True so an unexpected extra LLM call (e.g. a future change
    # makes chunking produce multiple chunks per source, or restores
    # LLM-based step-1 NER) fails loudly instead of silently re-using
    # the last scripted response.
    return MockLLM(responses=responses, strict=True)


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

    Skipped unless ``RUN_INTEGRATION=1`` is set. Connects to the server
    ``falkordb_test_server()`` names.
    """
    if not os.getenv("RUN_INTEGRATION"):
        pytest.skip("Set RUN_INTEGRATION=1 to run real-FalkorDB integration tests")

    from graphrag_sdk.api.main import GraphRAG
    from graphrag_sdk.core.connection import ConnectionConfig

    created: list[Any] = []

    def _make(*, llm, resolver, ontology=None, **kwargs):
        config = ConnectionConfig(
            **falkordb_test_server(),
            graph_name=f"test_{uuid4().hex[:8]}",
        )
        options = dict(
            connection=config,
            llm=llm,
            embedder=embedder,
            embedding_dimension=embedder.dimension,
        )
        if ontology is not None:
            options["ontology"] = ontology
        # Anything else the test asked for (e.g. enable_cypher) goes straight
        # through, so a new GraphRAG option does not need a fixture change.
        options.update(kwargs)
        rag = GraphRAG(**options)
        # Per-call resolver injection (apply_changes / update / ingest don't
        # accept a default-resolver kwarg on the facade — but each call does).
        rag._test_resolver = resolver  # marker, not used by SDK
        created.append(rag)
        return rag

    yield _make

    # Teardown — drop every test graph so reruns start clean.
    # Failures here are logged at WARNING (not silently swallowed):
    # if delete_all or close is broken, test graphs accumulate forever
    # in the FalkorDB instance and CI-side cleanup catches it from logs.
    for rag in created:
        graph_name = getattr(getattr(rag, "_conn", None), "graph_name", "<unknown>")
        try:
            await rag._graph_store.delete_all()
        except Exception:
            _logger.warning(
                "real_falkordb_rag_factory teardown: delete_all failed for "
                "graph %r — test graph may persist in FalkorDB",
                graph_name,
                exc_info=True,
            )
        try:
            await rag.close()
        except Exception:
            _logger.warning(
                "real_falkordb_rag_factory teardown: close() failed for "
                "graph %r — connection may leak",
                graph_name,
                exc_info=True,
            )
