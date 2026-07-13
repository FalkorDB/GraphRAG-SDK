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

__version__ = "1.2.0"

# ── API Surface (Facade) ────────────────────────────────────────
from graphrag_sdk.api.main import GraphRAG

# ── Core Contracts ───────────────────────────────────────────────
from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import (
    DocumentNotFoundError,
    GraphRAGError,
    LatencyBudgetExceededError,
)
from graphrag_sdk.core.models import (
    AgentStep,
    AgentTrace,
    ApplyChangesResult,
    Attribute,
    BatchEntry,
    ChatMessage,
    DataModel,
    DeleteDocumentResult,
    DocumentInfo,
    DocumentOutput,
    DocumentRecord,
    Entity,
    FinalizeResult,
    GraphData,
    GraphNode,
    GraphRelationship,
    IngestionResult,
    Ontology,
    RagResult,
    Relation,
    ResolutionResult,
    RetrieverResult,
    RetrieverResultItem,
    ScoredPath,
    SearchType,
    SkillResult,
    TextChunk,
    TextChunks,
    UpdateResult,
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
from graphrag_sdk.discovery import OntologyDiscoveryError, SchemaExtensionProposal

# ── Ingestion Strategies ────────────────────────────────────────
from graphrag_sdk.ingestion.backfill import (
    BackfillExecutor,
    BackfillMergeStats,
    BackfillResult,
    ChunkContext,
    EvolutionResult,
    OntologyEvolutionError,
)
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
from graphrag_sdk.ingestion.ingestion_planner import (
    HeuristicIngestionPlanner,
    IngestionPlan,
    IngestionPlanner,
    LLMIngestionPlanner,
    build_ingestion_strategies,
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
from graphrag_sdk.retrieval.agentic import AgenticRetrieval, ToolRegistry
from graphrag_sdk.retrieval.graph_walk import (
    DynamicGraphWalk,
    GraphWalkRetrieval,
    score_path,
)

# ── Retrieval Strategies ────────────────────────────────────────
from graphrag_sdk.retrieval.reranking_strategies.base import RerankingStrategy
from graphrag_sdk.retrieval.reranking_strategies.cosine import CosineReranker
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval
from graphrag_sdk.retrieval.strategies.path_router import (
    RETRIEVAL_PATHS,
    HeuristicPathRouter,
    LLMPathRouter,
)

# ── Skills ──────────────────────────────────────────────────────
from graphrag_sdk.skills import (
    SKILL_REGISTRY,
    ContradictionDetectionSkill,
    EntityComparisonSkill,
    GapAnalysisSkill,
    ImpactAnalysisSkill,
    Skill,
    TimelineReconstructionSkill,
    build_skill,
)

# ── Storage ─────────────────────────────────────────────────────
from graphrag_sdk.storage.graph_store import GraphStore
from graphrag_sdk.storage.ontology_store import (
    OntologyContradictionError,
    OntologyModificationNotAllowedError,
    OntologyStore,
)
from graphrag_sdk.storage.vector_store import VectorStore

__all__ = [
    # Version
    "__version__",
    # API
    "GraphRAG",
    # Core
    "ApplyChangesResult",
    "AgentStep",
    "AgentTrace",
    "BatchEntry",
    "ChatMessage",
    "ConnectionConfig",
    "Context",
    "DataModel",
    "DeleteDocumentResult",
    "DocumentInfo",
    "DocumentNotFoundError",
    "Attribute",
    "DocumentOutput",
    "DocumentRecord",
    "Embedder",
    "Entity",
    "FalkorDBConnection",
    "FinalizeResult",
    "GraphData",
    "GraphNode",
    "GraphRAGError",
    "GraphRelationship",
    "Ontology",
    "IngestionResult",
    "LatencyBudgetExceededError",
    "LLMBatchItem",
    "LLMInterface",
    "LiteLLM",
    "LiteLLMEmbedder",
    "OpenRouterEmbedder",
    "OpenRouterLLM",
    "RagResult",
    "Relation",
    "ResolutionResult",
    "RetrieverResult",
    "RetrieverResultItem",
    "ScoredPath",
    "SearchType",
    "SkillResult",
    "TextChunk",
    "TextChunks",
    "UpdateResult",
    # Ingestion
    "BackfillExecutor",
    "BackfillMergeStats",
    "BackfillResult",
    "ChunkContext",
    "ChunkingStrategy",
    "EvolutionResult",
    "OntologyDiscoveryError",
    "OntologyEvolutionError",
    "SchemaExtensionProposal",
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
    "HeuristicIngestionPlanner",
    "IngestionPlan",
    "IngestionPlanner",
    "LLMIngestionPlanner",
    "build_ingestion_strategies",
    # Retrieval
    "AgenticRetrieval",
    "CosineReranker",
    "DynamicGraphWalk",
    "GraphWalkRetrieval",
    "HeuristicPathRouter",
    "LLMPathRouter",
    "MultiPathRetrieval",
    "RETRIEVAL_PATHS",
    "RerankingStrategy",
    "RetrievalStrategy",
    "ToolRegistry",
    "score_path",
    # Skills
    "Skill",
    "SKILL_REGISTRY",
    "build_skill",
    "ContradictionDetectionSkill",
    "EntityComparisonSkill",
    "GapAnalysisSkill",
    "ImpactAnalysisSkill",
    "TimelineReconstructionSkill",
    # Storage
    "GraphStore",
    "OntologyContradictionError",
    "OntologyModificationNotAllowedError",
    "OntologyStore",
    "VectorStore",
]


# ── Deprecation aliases ──────────────────────────────────────────
#
# Older names from before the v1.2.x ontology rename (commit 363a53d).
# Importing the old names still works but emits a ``DeprecationWarning``.

_LEGACY_TOP_LEVEL_ALIASES: dict[str, str] = {
    "GraphSchema": "Ontology",
    "EntityType": "Entity",
    "RelationType": "Relation",
    "PropertyType": "Attribute",
    "SchemaModificationNotAllowedError": "OntologyModificationNotAllowedError",
}


def __getattr__(name: str):  # PEP 562
    if name in _LEGACY_TOP_LEVEL_ALIASES:
        import warnings

        new_name = _LEGACY_TOP_LEVEL_ALIASES[name]
        warnings.warn(
            f"`graphrag_sdk.{name}` has been renamed to "
            f"`graphrag_sdk.{new_name}` (v1.2+). Update your imports — "
            f"the alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        if name == "SchemaModificationNotAllowedError":
            from graphrag_sdk.storage.ontology_store import (
                OntologyModificationNotAllowedError,
            )

            return OntologyModificationNotAllowedError
        from graphrag_sdk.core import models as _models

        return getattr(_models, new_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
