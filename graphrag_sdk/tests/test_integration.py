"""Integration tests — verify top-level imports and cross-module interactions."""
from __future__ import annotations

import pytest


class TestTopLevelImports:
    """Verify all public API exports are importable."""

    def test_version(self):
        from graphrag_sdk import __version__
        assert __version__ == "2.0.0a1"

    def test_facade(self):
        from graphrag_sdk import GraphRAG
        assert GraphRAG is not None

    def test_core_models(self):
        from graphrag_sdk import (
            DataModel, GraphNode, GraphRelationship, TextChunk, TextChunks,
            DocumentInfo, DocumentOutput, EntityType, RelationType,
            SchemaPattern, GraphSchema, GraphData, ResolutionResult,
            RetrieverResult, RetrieverResultItem, RagResult, IngestionResult,
            SearchType,
        )
        # All should be importable
        assert DataModel is not None
        assert SearchType.VECTOR == "vector"

    def test_core_contracts(self):
        from graphrag_sdk import (
            Embedder, LLMInterface, Context, ConnectionConfig,
            FalkorDBConnection, GraphRAGError,
        )
        assert Embedder is not None
        assert LLMInterface is not None

    def test_strategy_abcs(self):
        from graphrag_sdk import (
            LoaderStrategy, ChunkingStrategy, ExtractionStrategy,
            ResolutionStrategy, RetrievalStrategy, RerankingStrategy,
        )
        assert LoaderStrategy is not None

    def test_pipeline(self):
        from graphrag_sdk import IngestionPipeline
        assert IngestionPipeline is not None

    def test_storage(self):
        from graphrag_sdk import GraphStore, VectorStore
        assert GraphStore is not None
        assert VectorStore is not None


class TestCrossCuttingConcerns:
    """Test interactions across module boundaries."""

    def test_context_flows_through_chunking(self):
        """Context is accepted by every strategy interface."""
        from graphrag_sdk.core.context import Context
        from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy
        import inspect
        sig = inspect.signature(ChunkingStrategy.chunk)
        assert "ctx" in sig.parameters

    def test_context_flows_through_extraction(self):
        from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
        import inspect
        sig = inspect.signature(ExtractionStrategy.extract)
        assert "ctx" in sig.parameters

    def test_context_flows_through_resolution(self):
        from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy
        import inspect
        sig = inspect.signature(ResolutionStrategy.resolve)
        assert "ctx" in sig.parameters

    def test_context_flows_through_retrieval(self):
        from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy
        import inspect
        sig = inspect.signature(RetrievalStrategy.search)
        assert "ctx" in sig.parameters

    def test_context_flows_through_reranking(self):
        from graphrag_sdk.retrieval.reranking_strategies.base import RerankingStrategy
        import inspect
        sig = inspect.signature(RerankingStrategy.rerank)
        assert "ctx" in sig.parameters


class TestSubmoduleImports:
    """Verify concrete implementations are importable."""

    def test_text_loader(self):
        from graphrag_sdk.ingestion.loaders.text_loader import TextLoader
        assert TextLoader is not None

    def test_pdf_loader(self):
        from graphrag_sdk.ingestion.loaders.pdf_loader import PdfLoader
        assert PdfLoader is not None

    def test_fixed_size_chunking(self):
        from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
        assert FixedSizeChunking is not None

    def test_schema_guided_extraction(self):
        from graphrag_sdk.ingestion.extraction_strategies.schema_guided import SchemaGuidedExtraction
        assert SchemaGuidedExtraction is not None

    def test_exact_match_resolution(self):
        from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution
        assert ExactMatchResolution is not None

    def test_local_retrieval(self):
        from graphrag_sdk.retrieval.strategies.local import LocalRetrieval
        assert LocalRetrieval is not None

    def test_semantic_router(self):
        from graphrag_sdk.retrieval.router import SemanticRouter
        assert SemanticRouter is not None

    def test_tracer(self):
        from graphrag_sdk.telemetry.tracer import Tracer, Span
        assert Tracer is not None
        assert Span is not None

    def test_graph_visualizer(self):
        from graphrag_sdk.utils.graph_viz import GraphVisualizer
        assert GraphVisualizer is not None
