"""Tests for ingestion/pipeline.py — the sequential orchestrator."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import IngestionError
from graphrag_sdk.core.models import (
    DocumentInfo,
    DocumentOutput,
    EntityType,
    GraphData,
    GraphNode,
    GraphRelationship,
    GraphSchema,
    RelationType,
    ResolutionResult,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.ingestion.pipeline import IngestionPipeline
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy
from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy


# ── Stub strategies ─────────────────────────────────────────────


class StubLoader(LoaderStrategy):
    def __init__(self, text: str = "Test content for pipeline.") -> None:
        self._text = text

    async def load(self, source: str, ctx: Context) -> DocumentOutput:
        return DocumentOutput(
            text=self._text,
            document_info=DocumentInfo(path=source),
        )


class StubChunker(ChunkingStrategy):
    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        # Split by sentence
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        return TextChunks(
            chunks=[
                TextChunk(text=s, index=i, uid=f"chunk-{i}")
                for i, s in enumerate(sentences)
            ]
        )


class StubExtractor(ExtractionStrategy):
    async def extract(self, chunks, schema, ctx):
        return GraphData(
            nodes=[GraphNode(id="e1", label="Entity", properties={"name": "Test"})],
            relationships=[],
        )


class StubResolver(ResolutionStrategy):
    async def resolve(self, graph_data, ctx):
        return ResolutionResult(
            nodes=graph_data.nodes,
            relationships=graph_data.relationships,
            merged_count=0,
        )


# ── Tests ───────────────────────────────────────────────────────


class TestIngestionPipeline:
    def _make_pipeline(
        self,
        mock_graph_store,
        mock_vector_store,
        text="Alice works at Acme Corp. Bob is her colleague.",
        schema=None,
    ):
        return IngestionPipeline(
            loader=StubLoader(text),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            schema=schema or GraphSchema(),
        )

    async def test_full_run(self, ctx, mock_graph_store, mock_vector_store):
        pipeline = self._make_pipeline(mock_graph_store, mock_vector_store)
        result = await pipeline.run("test.txt", ctx)
        assert result.nodes_created >= 1
        assert result.chunks_indexed >= 1
        # Verify graph_store was called
        assert mock_graph_store.upsert_nodes.called
        assert mock_graph_store.upsert_relationships.called
        assert mock_vector_store.index_chunks.called

    async def test_run_with_text_param(self, ctx, mock_graph_store, mock_vector_store):
        """When text= is passed, loader is skipped."""
        pipeline = self._make_pipeline(mock_graph_store, mock_vector_store)
        result = await pipeline.run("ignored.txt", ctx, text="Direct text input.")
        assert result.chunks_indexed >= 1

    async def test_run_creates_lexical_graph(self, ctx, mock_graph_store, mock_vector_store):
        """Mandatory lexical graph creates Document + Chunk nodes + PART_OF rels."""
        pipeline = self._make_pipeline(mock_graph_store, mock_vector_store)
        await pipeline.run("test.txt", ctx)
        # upsert_nodes called multiple times: doc, chunks, extracted entities
        assert mock_graph_store.upsert_nodes.call_count >= 2
        # Check that rels include PART_OF
        rel_calls = mock_graph_store.upsert_relationships.call_args_list
        all_rels = []
        for call in rel_calls:
            all_rels.extend(call[0][0])
        part_of_rels = [r for r in all_rels if r.type == "PART_OF"]
        assert len(part_of_rels) > 0

    async def test_run_creates_next_chunk_links(self, ctx, mock_graph_store, mock_vector_store):
        """Pipeline creates NEXT_CHUNK between sequential chunks."""
        pipeline = self._make_pipeline(
            mock_graph_store, mock_vector_store,
            text="First. Second. Third.",
        )
        await pipeline.run("test.txt", ctx)
        rel_calls = mock_graph_store.upsert_relationships.call_args_list
        all_rels = []
        for call in rel_calls:
            all_rels.extend(call[0][0])
        next_chunk_rels = [r for r in all_rels if r.type == "NEXT_CHUNK"]
        assert len(next_chunk_rels) == 2  # 3 chunks → 2 NEXT_CHUNK

    async def test_empty_chunks_short_circuits(self, ctx, mock_graph_store, mock_vector_store):
        """If chunker produces nothing, pipeline returns early."""
        pipeline = self._make_pipeline(mock_graph_store, mock_vector_store, text="")
        result = await pipeline.run("empty.txt", ctx)
        assert result.nodes_created == 0
        assert result.chunks_indexed == 0

    async def test_default_context(self, mock_graph_store, mock_vector_store):
        pipeline = self._make_pipeline(mock_graph_store, mock_vector_store)
        result = await pipeline.run("test.txt")  # no ctx → default
        assert result.nodes_created >= 0

    async def test_schema_pruning(self, ctx, mock_graph_store, mock_vector_store):
        """Pruning step filters nodes/rels by schema."""
        schema = GraphSchema(
            entities=[EntityType(label="Person")],
            relations=[RelationType(label="KNOWS")],
        )

        class ExtractorWithWrongLabels(ExtractionStrategy):
            async def extract(self, chunks, schema, ctx):
                return GraphData(
                    nodes=[
                        GraphNode(id="p1", label="Person", properties={"name": "Alice"}),
                        GraphNode(id="x1", label="Unknown", properties={"name": "???"}),
                    ],
                    relationships=[
                        GraphRelationship(
                            start_node_id="p1", end_node_id="x1",
                            type="WRONG", properties={},
                        ),
                    ],
                )

        pipeline = IngestionPipeline(
            loader=StubLoader("Test"),
            chunker=StubChunker(),
            extractor=ExtractorWithWrongLabels(),
            resolver=StubResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            schema=schema,
        )
        result = await pipeline.run("test.txt", ctx)
        # Only Person nodes survive, Unknown gets pruned
        # The WRONG relationship refs Unknown, so it also gets pruned
        assert result.nodes_created == 1
        assert result.relationships_created == 0

    async def test_pipeline_wraps_exception(self, ctx, mock_graph_store, mock_vector_store):
        """Non-IngestionError exceptions get wrapped."""
        class FailingLoader(LoaderStrategy):
            async def load(self, source, ctx):
                raise RuntimeError("unexpected!")

        pipeline = IngestionPipeline(
            loader=FailingLoader(),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            schema=GraphSchema(),
        )
        with pytest.raises(IngestionError, match="Pipeline failed"):
            await pipeline.run("test.txt", ctx)


class TestPruneMethod:
    def test_prune_open_schema(self):
        """Empty schema = open mode, nothing pruned."""
        pipeline = IngestionPipeline(
            loader=StubLoader(),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=MagicMock(),
            vector_store=MagicMock(),
            schema=GraphSchema(),
        )
        data = GraphData(
            nodes=[GraphNode(id="a", label="Anything")],
            relationships=[
                GraphRelationship(start_node_id="a", end_node_id="a", type="SELF"),
            ],
        )
        result = pipeline._prune(data, GraphSchema())
        assert len(result.nodes) == 1
        assert len(result.relationships) == 1

    def test_prune_removes_invalid_labels(self):
        pipeline = IngestionPipeline(
            loader=StubLoader(),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=MagicMock(),
            vector_store=MagicMock(),
        )
        schema = GraphSchema(entities=[EntityType(label="Person")])
        data = GraphData(
            nodes=[
                GraphNode(id="p", label="Person"),
                GraphNode(id="x", label="Unknown"),
            ],
            relationships=[],
        )
        result = pipeline._prune(data, schema)
        assert len(result.nodes) == 1
        assert result.nodes[0].label == "Person"

    def test_prune_removes_orphaned_rels(self):
        pipeline = IngestionPipeline(
            loader=StubLoader(),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=MagicMock(),
            vector_store=MagicMock(),
        )
        schema = GraphSchema(
            entities=[EntityType(label="A")],
            relations=[RelationType(label="LINK")],
        )
        data = GraphData(
            nodes=[
                GraphNode(id="a", label="A"),
                GraphNode(id="b", label="B"),  # will be pruned
            ],
            relationships=[
                GraphRelationship(start_node_id="a", end_node_id="b", type="LINK"),
            ],
        )
        result = pipeline._prune(data, schema)
        assert len(result.nodes) == 1
        assert len(result.relationships) == 0  # rel removed because 'b' is pruned
