"""Integration tests for the 10-step ingestion pipeline."""
from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, call

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    DocumentInfo,
    DocumentOutput,
    EntityMention,
    FactTriple,
    GraphData,
    GraphNode,
    GraphRelationship,
    GraphSchema,
    ResolutionResult,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.ingestion.pipeline import IngestionPipeline

from .conftest import MockEmbedder


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def pipeline_components(mock_graph_store, mock_vector_store, embedder):
    """Build a pipeline with mock strategies and stores."""
    loader = MagicMock()
    loader.load = AsyncMock(
        return_value=DocumentOutput(
            text="Alice works at Acme Corp. Bob also works there.",
            document_info=DocumentInfo(path="test.txt", uid="doc-1"),
        )
    )

    chunker = MagicMock()
    chunker.chunk = AsyncMock(
        return_value=TextChunks(
            chunks=[
                TextChunk(text="Alice works at Acme Corp.", index=0, uid="chunk-0"),
                TextChunk(text="Bob also works there.", index=1, uid="chunk-1"),
            ]
        )
    )

    # Extractor returns graph data with attached facts and mentions
    extracted = GraphData(
        nodes=[
            GraphNode(
                id="alice",
                label="Person",
                properties={"name": "Alice", "description": "An engineer"},
            ),
            GraphNode(
                id="bob",
                label="Person",
                properties={"name": "Bob", "description": "A manager"},
            ),
            GraphNode(
                id="acme_corp",
                label="Company",
                properties={"name": "Acme Corp", "description": "A company"},
            ),
        ],
        relationships=[
            GraphRelationship(
                start_node_id="alice",
                end_node_id="acme_corp",
                type="WORKS_AT",
                properties={"keywords": "employment", "description": "works at"},
            ),
            GraphRelationship(
                start_node_id="bob",
                end_node_id="acme_corp",
                type="WORKS_AT",
                properties={"keywords": "employment", "description": "works at"},
            ),
        ],
    )
    extracted.facts = [  # type: ignore[attr-defined]
        FactTriple(
            subject="Alice", predicate="WORKS_AT", object="Acme Corp",
            source_chunk_id="chunk-0",
        ),
        FactTriple(
            subject="Bob", predicate="WORKS_AT", object="Acme Corp",
            source_chunk_id="chunk-1",
        ),
    ]
    extracted.mentions = [  # type: ignore[attr-defined]
        EntityMention(chunk_id="chunk-0", entity_id="alice"),
        EntityMention(chunk_id="chunk-0", entity_id="acme_corp"),
        EntityMention(chunk_id="chunk-1", entity_id="bob"),
        EntityMention(chunk_id="chunk-1", entity_id="acme_corp"),
    ]

    extractor = MagicMock()
    extractor.extract = AsyncMock(return_value=extracted)

    resolved = ResolutionResult(
        nodes=extracted.nodes,
        relationships=extracted.relationships,
        merged_count=0,
    )
    resolver = MagicMock()
    resolver.resolve = AsyncMock(return_value=resolved)

    return {
        "loader": loader,
        "chunker": chunker,
        "extractor": extractor,
        "resolver": resolver,
        "graph_store": mock_graph_store,
        "vector_store": mock_vector_store,
        "embedder": embedder,
        "extracted": extracted,
    }


@pytest.fixture
def pipeline(pipeline_components):
    return IngestionPipeline(
        loader=pipeline_components["loader"],
        chunker=pipeline_components["chunker"],
        extractor=pipeline_components["extractor"],
        resolver=pipeline_components["resolver"],
        graph_store=pipeline_components["graph_store"],
        vector_store=pipeline_components["vector_store"],
        embedder=pipeline_components["embedder"],
    )


# ── Tests ──────────────────────────────────────────────────────


class TestPipeline10StepExecution:
    async def test_all_10_steps_execute(self, pipeline, pipeline_components, ctx):
        """Verify all pipeline steps execute in order."""
        result = await pipeline.run("test.txt", ctx)

        # Step 1: Loader called
        pipeline_components["loader"].load.assert_called_once()
        # Step 2: Chunker called
        pipeline_components["chunker"].chunk.assert_called_once()
        # Step 3: Lexical graph written (upsert_nodes called for doc + chunks)
        # Step 4: Extractor called
        pipeline_components["extractor"].extract.assert_called_once()
        # Step 6: Resolver called
        pipeline_components["resolver"].resolve.assert_called_once()
        # Step 9: Chunks indexed
        pipeline_components["vector_store"].index_chunks.assert_called_once()
        # Step 10: Facts indexed
        pipeline_components["vector_store"].index_facts.assert_called_once()

        # Result has expected fields
        assert result.nodes_created == 3
        assert result.chunks_indexed == 2

    async def test_synonym_edges_written(self, pipeline, pipeline_components, ctx):
        """Synonym edges should be upserted to graph store."""
        result = await pipeline.run("test.txt", ctx)

        # Check metadata for synonym count
        assert "synonym_edges_created" in result.metadata

    async def test_mention_edges_written(self, pipeline, pipeline_components, ctx):
        """MENTIONED_IN edges should be upserted to graph store."""
        result = await pipeline.run("test.txt", ctx)

        assert "mention_edges_created" in result.metadata
        assert result.metadata["mention_edges_created"] > 0

        # Graph store should have multiple upsert_relationships calls
        # (lexical, entity rels, synonyms, mentions)
        assert pipeline_components["graph_store"].upsert_relationships.call_count >= 2

    async def test_facts_indexed(self, pipeline, pipeline_components, ctx):
        """Facts should be passed to vector_store.index_facts."""
        result = await pipeline.run("test.txt", ctx)

        assert "facts_indexed" in result.metadata
        pipeline_components["vector_store"].index_facts.assert_called_once()

    async def test_metadata_includes_new_fields(self, pipeline, ctx):
        """IngestionResult.metadata should include all new counters."""
        result = await pipeline.run("test.txt", ctx)

        assert "synonym_edges_created" in result.metadata
        assert "facts_indexed" in result.metadata
        assert "mention_edges_created" in result.metadata
        assert "merged_entities" in result.metadata

    async def test_relationships_count_includes_all_types(
        self, pipeline, pipeline_components, ctx
    ):
        """relationships_created should include entity rels + synonym + mention."""
        result = await pipeline.run("test.txt", ctx)

        # relationships_created = entity rels + synonym edges + mention edges
        entity_rels = 2
        mention_edges = result.metadata["mention_edges_created"]
        synonym_edges = result.metadata["synonym_edges_created"]
        assert result.relationships_created == entity_rels + synonym_edges + mention_edges


class TestPipelineWithTextInput:
    async def test_text_input_skips_loader(self, pipeline, pipeline_components, ctx):
        """Providing text= should skip the loader step."""
        result = await pipeline.run(
            "unused",
            ctx,
            text="Alice works at Acme Corp.",
        )

        pipeline_components["loader"].load.assert_not_called()
        pipeline_components["chunker"].chunk.assert_called_once()


class TestPipelineEmptyChunks:
    async def test_empty_chunks_early_return(self, ctx, mock_graph_store, mock_vector_store):
        """If chunker produces no chunks, pipeline returns early."""
        loader = MagicMock()
        loader.load = AsyncMock(
            return_value=DocumentOutput(
                text="",
                document_info=DocumentInfo(uid="doc-1"),
            )
        )
        chunker = MagicMock()
        chunker.chunk = AsyncMock(return_value=TextChunks(chunks=[]))

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=chunker,
            extractor=MagicMock(),
            resolver=MagicMock(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
        )

        result = await pipeline.run("test.txt", ctx)

        assert result.nodes_created == 0
        assert result.relationships_created == 0
        assert result.chunks_indexed == 0


class TestPipelineNoEmbedder:
    async def test_no_embedder_skips_synonymy(
        self, pipeline_components, ctx
    ):
        """Pipeline without embedder should skip synonymy detection gracefully."""
        pipeline = IngestionPipeline(
            loader=pipeline_components["loader"],
            chunker=pipeline_components["chunker"],
            extractor=pipeline_components["extractor"],
            resolver=pipeline_components["resolver"],
            graph_store=pipeline_components["graph_store"],
            vector_store=pipeline_components["vector_store"],
            embedder=None,  # No embedder
        )

        result = await pipeline.run("test.txt", ctx)

        assert result.metadata["synonym_edges_created"] == 0
