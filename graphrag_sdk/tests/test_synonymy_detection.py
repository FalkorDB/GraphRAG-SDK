"""Tests for synonymy detection in the pipeline."""
from __future__ import annotations

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphNode
from graphrag_sdk.ingestion.pipeline import IngestionPipeline

from .conftest import MockEmbedder


# ── Helpers ────────────────────────────────────────────────────


def _make_entity_node(
    id: str,
    label: str = "Person",
    name: str = "",
    description: str = "",
) -> GraphNode:
    return GraphNode(
        id=id,
        label=label,
        properties={"name": name or id, "description": description},
    )


async def _detect_synonyms(
    nodes: list[GraphNode],
    embedder: MockEmbedder,
    ctx: Context,
    **kwargs,
) -> list:
    """Helper to call _detect_synonymy_edges without a full pipeline."""
    from unittest.mock import MagicMock, AsyncMock

    pipeline = IngestionPipeline(
        loader=MagicMock(),
        chunker=MagicMock(),
        extractor=MagicMock(),
        resolver=MagicMock(),
        graph_store=MagicMock(),
        vector_store=MagicMock(),
        embedder=embedder,
    )
    return await pipeline._detect_synonymy_edges(nodes, ctx, **kwargs)


# ── Tests ──────────────────────────────────────────────────────


class TestSynonymyDetection:
    async def test_similar_entities_get_synonym_edges(self, ctx):
        """Identical entities should get SYNONYM edges with similarity ~1.0."""
        embedder = MockEmbedder(dimension=8)
        nodes = [
            _make_entity_node("alice", name="Alice", description="An engineer"),
            _make_entity_node("alice2", name="Alice", description="An engineer"),
        ]
        edges = await _detect_synonyms(
            nodes, embedder, ctx, similarity_threshold=0.5
        )

        assert len(edges) >= 1
        assert edges[0].type == "SYNONYM"
        assert "similarity" in edges[0].properties

    async def test_no_self_edges(self, ctx):
        """An entity should never have a synonym edge to itself."""
        embedder = MockEmbedder(dimension=8)
        nodes = [
            _make_entity_node("alice", name="Alice"),
            _make_entity_node("bob", name="Bob"),
        ]
        edges = await _detect_synonyms(
            nodes, embedder, ctx, similarity_threshold=0.0
        )

        for edge in edges:
            assert edge.start_node_id != edge.end_node_id

    async def test_deduplicated_pairs(self, ctx):
        """No A→B and B→A — only one direction per pair."""
        embedder = MockEmbedder(dimension=8)
        nodes = [
            _make_entity_node("a", name="Alice"),
            _make_entity_node("b", name="Alice"),
            _make_entity_node("c", name="Alice"),
        ]
        edges = await _detect_synonyms(
            nodes, embedder, ctx, similarity_threshold=0.0
        )

        seen_pairs = set()
        for edge in edges:
            pair = tuple(sorted([edge.start_node_id, edge.end_node_id]))
            assert pair not in seen_pairs, f"Duplicate pair: {pair}"
            seen_pairs.add(pair)

    async def test_empty_for_single_entity(self, ctx):
        """Less than 2 entities → no synonym edges."""
        embedder = MockEmbedder(dimension=8)
        nodes = [_make_entity_node("alice", name="Alice")]
        edges = await _detect_synonyms(nodes, embedder, ctx)

        assert edges == []

    async def test_empty_for_no_entities(self, ctx):
        embedder = MockEmbedder(dimension=8)
        edges = await _detect_synonyms([], embedder, ctx)
        assert edges == []

    async def test_respects_threshold(self, ctx):
        """High threshold should produce fewer edges."""
        embedder = MockEmbedder(dimension=8)
        nodes = [
            _make_entity_node("a", name="Alice", description="Engineer"),
            _make_entity_node("b", name="Bob", description="Manager"),
            _make_entity_node("c", name="Charlie", description="Designer"),
        ]

        edges_low = await _detect_synonyms(
            nodes, embedder, ctx, similarity_threshold=0.0
        )
        edges_high = await _detect_synonyms(
            nodes, embedder, ctx, similarity_threshold=0.99
        )

        assert len(edges_high) <= len(edges_low)

    async def test_document_chunk_nodes_excluded(self, ctx):
        """Document and Chunk nodes should not participate in synonymy."""
        embedder = MockEmbedder(dimension=8)
        nodes = [
            GraphNode(id="doc-1", label="Document", properties={"name": "doc"}),
            GraphNode(id="chunk-1", label="Chunk", properties={"name": "chunk"}),
            _make_entity_node("alice", name="Alice"),
        ]
        edges = await _detect_synonyms(
            nodes, embedder, ctx, similarity_threshold=0.0
        )

        # Only 1 entity node → no edges possible
        assert edges == []

    async def test_no_embedder_returns_empty(self, ctx):
        """If no embedder configured, return empty list."""
        from unittest.mock import MagicMock

        pipeline = IngestionPipeline(
            loader=MagicMock(),
            chunker=MagicMock(),
            extractor=MagicMock(),
            resolver=MagicMock(),
            graph_store=MagicMock(),
            vector_store=MagicMock(),
            embedder=None,
        )
        nodes = [
            _make_entity_node("a", name="Alice"),
            _make_entity_node("b", name="Bob"),
        ]
        edges = await pipeline._detect_synonymy_edges(nodes, ctx)
        assert edges == []
