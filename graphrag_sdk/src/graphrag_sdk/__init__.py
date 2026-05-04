# GraphRAG SDK
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

__version__ = "1.0.2"

# ── API Surface (Facade) ────────────────────────────────────────
from graphrag_sdk.api.main import GraphRAG

# ── Core Contracts ───────────────────────────────────────────────
from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import GraphRAGError
from graphrag_sdk.core.models import (
    ChatMessage,
    DataModel,
    DocumentInfo,
    DocumentOutput,
    EntityType,
    FinalizeResult,
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
    SearchType,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.core.providers import (
    Embedder,
    LiteLLM,
    LiteLLMEmbedder,
    LLMBatchItem,
    LLMInterface,
    OpenRouterEmbedder,
    OpenRouterLLM,
)

# ── Ingestion Strategies ────────────────────────────────────────
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy
from graphrag_sdk.ingestion.chunking_strategies.callable_chunking import (
    CallableChunking,
)
from graphrag_sdk.ingestion.chunking_strategies.contextual_chunking import (
    ContextualChunking,
)
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
from graphrag_sdk.ingestion.chunking_strategies.sentence_token_cap import (
    SentenceTokenCapChunking,
)
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
from graphrag_sdk.ingestion.extraction_strategies.coref_resolvers import (
    CorefResolver,
    FastCorefResolver,
)
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
    EntityExtractor,
    GLiNERExtractor,
    LLMExtractor,
)
from graphrag_sdk.ingestion.extraction_strategies.graph_extraction import (
    GraphExtraction,
)
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy
from graphrag_sdk.ingestion.pipeline import IngestionPipeline
from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy
from graphrag_sdk.ingestion.resolution_strategies.description_merge import (
    DescriptionMergeResolution,
)
from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
    ExactMatchResolution,
)
from graphrag_sdk.ingestion.resolution_strategies.llm_verified_resolution import (
    LLMVerifiedResolution,
)
from graphrag_sdk.ingestion.resolution_strategies.semantic_resolution import (
    SemanticResolution,
)

# ── Retrieval Strategies ────────────────────────────────────────
from graphrag_sdk.retrieval.reranking_strategies.base import RerankingStrategy
from graphrag_sdk.retrieval.reranking_strategies.cosine import CosineReranker
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval

# ── Storage ─────────────────────────────────────────────────────
from graphrag_sdk.storage.graph_store import GraphStore
from graphrag_sdk.storage.vector_store import VectorStore

__all__ = [
    # Version
    "__version__",
    # API
    "GraphRAG",
    # Core
    "ChatMessage",
    "ConnectionConfig",
    "Context",
    "DataModel",
    "DocumentInfo",
    "DocumentOutput",
    "Embedder",
    "EntityType",
    "FalkorDBConnection",
    "FinalizeResult",
    "GraphData",
    "GraphNode",
    "GraphRAGError",
    "GraphRelationship",
    "GraphSchema",
    "IngestionResult",
    "LLMBatchItem",
    "LLMInterface",
    "LiteLLM",
    "LiteLLMEmbedder",
    "OpenRouterEmbedder",
    "OpenRouterLLM",
    "RagResult",
    "RelationType",
    "ResolutionResult",
    "RetrieverResult",
    "RetrieverResultItem",
    "SearchType",
    "TextChunk",
    "TextChunks",
    # Ingestion
    "ChunkingStrategy",
    "CallableChunking",
    "ContextualChunking",
    "FixedSizeChunking",
    "SentenceTokenCapChunking",
    "ExtractionStrategy",
    "GraphExtraction",
    "EntityExtractor",
    "GLiNERExtractor",
    "LLMExtractor",
    "CorefResolver",
    "FastCorefResolver",
    "IngestionPipeline",
    "LoaderStrategy",
    "ResolutionStrategy",
    "DescriptionMergeResolution",
    "ExactMatchResolution",
    "LLMVerifiedResolution",
    "SemanticResolution",
    # Retrieval
    "CosineReranker",
    "MultiPathRetrieval",
    "RerankingStrategy",
    "RetrievalStrategy",
    # Storage
    "GraphStore",
    "VectorStore",
]
