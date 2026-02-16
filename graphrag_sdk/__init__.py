# GraphRAG SDK 2.0
# A modular, async-first Graph RAG framework for FalkorDB.
#
# Core Principles:
#   Strategy Modularity — Swap any algorithmic concern via strategy ABCs.
#   Zero-Loss Data — Full traceability from raw text to graph nodes.
#   Production Latency — Async-first, pooled connections, batched writes.
#   Simplicity — One entry point, flat structure, no meta-programming.
#   Credibility — Schema-guided extraction + mandatory provenance.
#   Accuracy — Multi-hop reasoning across the knowledge graph.
#   Adaptability — Optimization-ready core, strategies are swappable.
#   Velocity — Production-grade throughput.

__version__ = "2.0.0a1"

# ── API Surface (Facade) ────────────────────────────────────────
from graphrag_sdk.api.main import GraphRAG

# ── Core Contracts ───────────────────────────────────────────────
from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import GraphRAGError
from graphrag_sdk.core.models import (
    DataModel,
    DocumentInfo,
    DocumentOutput,
    EntityType,
    GraphData,
    GraphNode,
    GraphRelationship,
    GraphSchema,
    IngestionResult,
    RagResult,
    RelationType,
    ResolutionResult,
    RetrieverResult,
    RetrieverResultItem,
    SchemaPattern,
    SearchType,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.core.providers import Embedder, LLMInterface

# ── Ingestion Strategies ────────────────────────────────────────
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy
from graphrag_sdk.ingestion.pipeline import IngestionPipeline
from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy

# ── Retrieval Strategies ────────────────────────────────────────
from graphrag_sdk.retrieval.reranking_strategies.base import RerankingStrategy
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy

# ── Storage ─────────────────────────────────────────────────────
from graphrag_sdk.storage.graph_store import GraphStore
from graphrag_sdk.storage.vector_store import VectorStore

__all__ = [
    # Version
    "__version__",
    # API
    "GraphRAG",
    # Core
    "ConnectionConfig",
    "Context",
    "DataModel",
    "DocumentInfo",
    "DocumentOutput",
    "Embedder",
    "EntityType",
    "FalkorDBConnection",
    "GraphData",
    "GraphNode",
    "GraphRAGError",
    "GraphRelationship",
    "GraphSchema",
    "IngestionResult",
    "LLMInterface",
    "RagResult",
    "RelationType",
    "ResolutionResult",
    "RetrieverResult",
    "RetrieverResultItem",
    "SchemaPattern",
    "SearchType",
    "TextChunk",
    "TextChunks",
    # Ingestion
    "ChunkingStrategy",
    "ExtractionStrategy",
    "IngestionPipeline",
    "LoaderStrategy",
    "ResolutionStrategy",
    # Retrieval
    "RerankingStrategy",
    "RetrievalStrategy",
    # Storage
    "GraphStore",
    "VectorStore",
]
