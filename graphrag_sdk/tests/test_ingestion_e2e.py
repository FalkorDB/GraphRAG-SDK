"""End-to-end integration tests: MarkdownLoader → StructuralChunking → IngestionPipeline.

Each component (loader, chunker, pipeline) is unit-tested in isolation.
These tests verify the *wiring*: that real instances compose correctly and
that metadata (breadcrumbs, strategy, chunk provenance) flows end-to-end
without being lost at any hand-off point.

External services (graph store, vector store) are provided by the shared
mock_graph_store / mock_vector_store conftest fixtures.  Extraction and
resolution are stubbed inline so each test controls only what it cares about.
"""
from __future__ import annotations

import pytest

_MARKDOWN = """\
# Project Overview

This document describes the GraphRAG SDK ingestion pipeline.

## Installation

Install the package with pip.

### Prerequisites

Python 3.10 or later is required.

## Usage

Import the pipeline and run it against your documents.
"""


class TestMarkdownLoaderStructuralChunkingPipeline:
    """End-to-end wiring: real MarkdownLoader + real StructuralChunking."""

    async def test_pipeline_runs_and_indexes_chunks(
        self, ctx, tmp_path, mock_graph_store, mock_vector_store
    ):
        """Pipeline must complete without error and index at least one chunk."""
        from graphrag_sdk.core.models import GraphData, GraphSchema, ResolutionResult
        from graphrag_sdk.ingestion.chunking_strategies.structural_chunking import StructuralChunking
        from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
        from graphrag_sdk.ingestion.loaders.markdown_loader import MarkdownLoader
        from graphrag_sdk.ingestion.pipeline import IngestionPipeline
        from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy

        class _NullExtractor(ExtractionStrategy):
            async def extract(self, chunks, schema, ctx):
                return GraphData(nodes=[], relationships=[])

        class _NullResolver(ResolutionStrategy):
            async def resolve(self, graph_data, ctx):
                return ResolutionResult(nodes=[], relationships=[], merged_count=0)

        md_file = tmp_path / "doc.md"
        md_file.write_text(_MARKDOWN)

        pipeline = IngestionPipeline(
            loader=MarkdownLoader(),
            chunker=StructuralChunking(max_tokens=512),
            extractor=_NullExtractor(),
            resolver=_NullResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            schema=GraphSchema(),
        )
        result = await pipeline.run(str(md_file), ctx)

        assert result.chunks_indexed >= 1
        assert mock_vector_store.index_chunks.called

    async def test_structural_strategy_marker_set_on_chunks(
        self, ctx, tmp_path, mock_graph_store, mock_vector_store
    ):
        """Chunks passed to the vector store must carry strategy='structural_chunking'.

        Verifies chunk_document() is reached (not the plain chunk() fallback).
        """
        from graphrag_sdk.core.models import GraphData, GraphSchema, ResolutionResult
        from graphrag_sdk.ingestion.chunking_strategies.structural_chunking import StructuralChunking
        from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
        from graphrag_sdk.ingestion.loaders.markdown_loader import MarkdownLoader
        from graphrag_sdk.ingestion.pipeline import IngestionPipeline
        from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy

        class _NullExtractor(ExtractionStrategy):
            async def extract(self, chunks, schema, ctx):
                return GraphData(nodes=[], relationships=[])

        class _NullResolver(ResolutionStrategy):
            async def resolve(self, graph_data, ctx):
                return ResolutionResult(nodes=[], relationships=[], merged_count=0)

        md_file = tmp_path / "doc.md"
        md_file.write_text(_MARKDOWN)

        pipeline = IngestionPipeline(
            loader=MarkdownLoader(),
            chunker=StructuralChunking(max_tokens=512),
            extractor=_NullExtractor(),
            resolver=_NullResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            schema=GraphSchema(),
        )
        await pipeline.run(str(md_file), ctx)

        chunks = mock_vector_store.index_chunks.call_args[0][0]
        for chunk in chunks.chunks:
            assert chunk.metadata.get("strategy") == "structural_chunking"

    async def test_breadcrumbs_propagate_into_chunks(
        self, ctx, tmp_path, mock_graph_store, mock_vector_store
    ):
        """Breadcrumbs produced by MarkdownLoader must survive into chunk metadata.

        Validates the full hand-off:
          MarkdownLoader (elements with breadcrumbs)
          → StructuralChunking._flush_buffer (writes breadcrumbs to metadata)
          → IngestionPipeline (passes chunks to vector store)
        """
        from graphrag_sdk.core.models import GraphData, GraphSchema, ResolutionResult
        from graphrag_sdk.ingestion.chunking_strategies.structural_chunking import StructuralChunking
        from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
        from graphrag_sdk.ingestion.loaders.markdown_loader import MarkdownLoader
        from graphrag_sdk.ingestion.pipeline import IngestionPipeline
        from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy

        class _NullExtractor(ExtractionStrategy):
            async def extract(self, chunks, schema, ctx):
                return GraphData(nodes=[], relationships=[])

        class _NullResolver(ResolutionStrategy):
            async def resolve(self, graph_data, ctx):
                return ResolutionResult(nodes=[], relationships=[], merged_count=0)

        md_file = tmp_path / "doc.md"
        md_file.write_text(_MARKDOWN)

        pipeline = IngestionPipeline(
            loader=MarkdownLoader(),
            chunker=StructuralChunking(max_tokens=512),
            extractor=_NullExtractor(),
            resolver=_NullResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            schema=GraphSchema(),
        )
        await pipeline.run(str(md_file), ctx)

        chunks = mock_vector_store.index_chunks.call_args[0][0]
        all_breadcrumbs = []
        for chunk in chunks.chunks:
            crumbs = chunk.metadata.get("breadcrumbs")
            assert crumbs is not None, (
                f"Chunk {chunk.index!r} missing 'breadcrumbs': {chunk.metadata}"
            )
            all_breadcrumbs.extend(crumbs)

        assert "Project Overview" in all_breadcrumbs

    async def test_chunk_nodes_in_graph_carry_breadcrumbs(
        self, ctx, tmp_path, mock_graph_store, mock_vector_store
    ):
        """Chunk nodes written to the graph store must include breadcrumbs as a property.

        _build_lexical_graph spreads chunk.metadata onto Chunk node properties,
        so breadcrumbs become graph-queryable at zero extra cost.
        """
        from graphrag_sdk.core.models import GraphData, GraphSchema, ResolutionResult
        from graphrag_sdk.ingestion.chunking_strategies.structural_chunking import StructuralChunking
        from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
        from graphrag_sdk.ingestion.loaders.markdown_loader import MarkdownLoader
        from graphrag_sdk.ingestion.pipeline import IngestionPipeline
        from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy

        class _NullExtractor(ExtractionStrategy):
            async def extract(self, chunks, schema, ctx):
                return GraphData(nodes=[], relationships=[])

        class _NullResolver(ResolutionStrategy):
            async def resolve(self, graph_data, ctx):
                return ResolutionResult(nodes=[], relationships=[], merged_count=0)

        md_file = tmp_path / "doc.md"
        md_file.write_text(_MARKDOWN)

        pipeline = IngestionPipeline(
            loader=MarkdownLoader(),
            chunker=StructuralChunking(max_tokens=512),
            extractor=_NullExtractor(),
            resolver=_NullResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            schema=GraphSchema(),
        )
        await pipeline.run(str(md_file), ctx)

        all_nodes = [
            n
            for call in mock_graph_store.upsert_nodes.call_args_list
            for n in call[0][0]
        ]
        chunk_nodes = [n for n in all_nodes if n.label == "Chunk"]
        assert chunk_nodes, "No Chunk nodes were written to the graph store"
        for node in chunk_nodes:
            assert "breadcrumbs" in node.properties, (
                f"Chunk node {node.id!r} missing 'breadcrumbs': {node.properties}"
            )

    async def test_document_node_path_matches_source(
        self, ctx, tmp_path, mock_graph_store, mock_vector_store
    ):
        """A Document node must be written with the source file path."""
        from graphrag_sdk.core.models import GraphData, GraphSchema, ResolutionResult
        from graphrag_sdk.ingestion.chunking_strategies.structural_chunking import StructuralChunking
        from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
        from graphrag_sdk.ingestion.loaders.markdown_loader import MarkdownLoader
        from graphrag_sdk.ingestion.pipeline import IngestionPipeline
        from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy

        class _NullExtractor(ExtractionStrategy):
            async def extract(self, chunks, schema, ctx):
                return GraphData(nodes=[], relationships=[])

        class _NullResolver(ResolutionStrategy):
            async def resolve(self, graph_data, ctx):
                return ResolutionResult(nodes=[], relationships=[], merged_count=0)

        md_file = tmp_path / "readme.md"
        md_file.write_text(_MARKDOWN)

        pipeline = IngestionPipeline(
            loader=MarkdownLoader(),
            chunker=StructuralChunking(max_tokens=512),
            extractor=_NullExtractor(),
            resolver=_NullResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            schema=GraphSchema(),
        )
        await pipeline.run(str(md_file), ctx)

        all_nodes = [
            n
            for call in mock_graph_store.upsert_nodes.call_args_list
            for n in call[0][0]
        ]
        doc_nodes = [n for n in all_nodes if n.label == "Document"]
        assert len(doc_nodes) == 1
        assert doc_nodes[0].properties.get("path") == str(md_file)

    async def test_header_markup_stripped_from_chunk_text(
        self, ctx, tmp_path, mock_graph_store, mock_vector_store
    ):
        """Markdown '#' sigils must not appear in any indexed chunk text."""
        from graphrag_sdk.core.models import GraphData, GraphSchema, ResolutionResult
        from graphrag_sdk.ingestion.chunking_strategies.structural_chunking import StructuralChunking
        from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
        from graphrag_sdk.ingestion.loaders.markdown_loader import MarkdownLoader
        from graphrag_sdk.ingestion.pipeline import IngestionPipeline
        from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy

        class _NullExtractor(ExtractionStrategy):
            async def extract(self, chunks, schema, ctx):
                return GraphData(nodes=[], relationships=[])

        class _NullResolver(ResolutionStrategy):
            async def resolve(self, graph_data, ctx):
                return ResolutionResult(nodes=[], relationships=[], merged_count=0)

        md_file = tmp_path / "doc.md"
        md_file.write_text("# Main Title\n\nSome content here.\n")

        pipeline = IngestionPipeline(
            loader=MarkdownLoader(),
            chunker=StructuralChunking(max_tokens=20),
            extractor=_NullExtractor(),
            resolver=_NullResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            schema=GraphSchema(),
        )
        await pipeline.run(str(md_file), ctx)

        chunks = mock_vector_store.index_chunks.call_args[0][0]
        full_text = " ".join(c.text for c in chunks.chunks)
        assert "Main Title" in full_text
        assert "# Main Title" not in full_text
