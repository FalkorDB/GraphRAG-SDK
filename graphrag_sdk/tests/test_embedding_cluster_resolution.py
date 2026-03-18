"""Tests for EmbeddingClusterResolution strategy."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    GraphData,
    GraphNode,
    GraphRelationship,
)
from graphrag_sdk.ingestion.resolution_strategies.embedding_cluster import (
    EmbeddingClusterResolution,
)


def _make_node(node_id: str, label: str, name: str, description: str = "") -> GraphNode:
    return GraphNode(
        id=node_id,
        label=label,
        properties={"name": name, "description": description},
    )


def _make_rel(start: str, end: str, rel_type: str = "RELATES") -> GraphRelationship:
    return GraphRelationship(start_node_id=start, end_node_id=end, type=rel_type)


class TestExactNameDedup:
    """Phase 1: exact-name merging."""

    def test_no_duplicates(self):
        resolver = EmbeddingClusterResolution(
            llm=MagicMock(), embedder=MagicMock()
        )
        nodes = [
            _make_node("a", "Person", "Alice"),
            _make_node("b", "Person", "Bob"),
        ]
        ctx = Context(tenant_id="test")
        remap, deduped, merged = resolver._exact_name_dedup(nodes, ctx)
        assert len(deduped) == 2
        assert merged == 0
        assert remap == {}

    def test_exact_name_merge(self):
        resolver = EmbeddingClusterResolution(
            llm=MagicMock(), embedder=MagicMock()
        )
        nodes = [
            _make_node("alice_1", "Person", "Alice", "A girl"),
            _make_node("alice_2", "Person", "alice", "Another description"),
        ]
        ctx = Context(tenant_id="test")
        remap, deduped, merged = resolver._exact_name_dedup(nodes, ctx)
        assert len(deduped) == 1
        assert merged == 1
        assert "alice_2" in remap
        assert remap["alice_2"] == "alice_1"

    def test_cross_label_kept_separate(self):
        """Same name but different labels should NOT merge."""
        resolver = EmbeddingClusterResolution(
            llm=MagicMock(), embedder=MagicMock()
        )
        nodes = [
            _make_node("paris_person", "Person", "Paris"),
            _make_node("paris_location", "Location", "Paris"),
        ]
        ctx = Context(tenant_id="test")
        remap, deduped, merged = resolver._exact_name_dedup(nodes, ctx)
        assert len(deduped) == 2
        assert merged == 0

    def test_description_merge(self):
        resolver = EmbeddingClusterResolution(
            llm=MagicMock(), embedder=MagicMock()
        )
        nodes = [
            _make_node("a1", "Person", "Alice", "Lives in Wonderland"),
            _make_node("a2", "Person", "Alice", "Follows the rabbit"),
            _make_node("a3", "Person", "alice", "Has a cat"),
        ]
        ctx = Context(tenant_id="test")
        remap, deduped, merged = resolver._exact_name_dedup(nodes, ctx)
        assert len(deduped) == 1
        assert merged == 2
        desc = deduped[0].properties["description"]
        assert "Wonderland" in desc
        assert "rabbit" in desc
        assert "cat" in desc


class TestParseDedupResponse:
    """Test LLM response parsing."""

    def test_parse_merge_line(self):
        resolver = EmbeddingClusterResolution(
            llm=MagicMock(), embedder=MagicMock()
        )
        nodes = [
            _make_node("a", "Person", "Gen. Grant"),
            _make_node("b", "Person", "Grant"),
            _make_node("c", "Person", "U.S. Grant"),
            _make_node("d", "Person", "Sherman"),
        ]
        names = ["Gen. Grant", "Grant", "U.S. Grant", "Sherman"]
        cluster_members = [0, 1, 2, 3]

        # LLM picks "Gen. Grant" (exact match) as canonical
        content = "Gen. Grant | Grant, U.S. Grant\n"
        groups = resolver._parse_dedup_response(content, cluster_members, nodes, names)
        assert len(groups) == 1
        survivor_idx, dup_indices = groups[0]
        assert survivor_idx == 0  # Gen. Grant (exact match)
        assert 1 in dup_indices  # Grant
        assert 2 in dup_indices  # U.S. Grant
        assert 3 not in dup_indices  # Sherman stays separate

    def test_parse_none_response(self):
        resolver = EmbeddingClusterResolution(
            llm=MagicMock(), embedder=MagicMock()
        )
        groups = resolver._parse_dedup_response("NONE", [0, 1], [], [])
        assert groups == []

    def test_parse_empty_response(self):
        resolver = EmbeddingClusterResolution(
            llm=MagicMock(), embedder=MagicMock()
        )
        groups = resolver._parse_dedup_response("", [0, 1], [], [])
        assert groups == []

    def test_parse_multiple_groups(self):
        resolver = EmbeddingClusterResolution(
            llm=MagicMock(), embedder=MagicMock()
        )
        nodes = [
            _make_node("a", "Person", "Angeline Hall"),
            _make_node("b", "Person", "Angeline"),
            _make_node("c", "Place", "New York"),
            _make_node("d", "Place", "NYC"),
        ]
        names = ["Angeline Hall", "Angeline", "New York", "NYC"]
        cluster_members = [0, 1, 2, 3]

        content = (
            "Angeline Hall | Angeline\n"
            "New York | NYC\n"
        )
        groups = resolver._parse_dedup_response(content, cluster_members, nodes, names)
        assert len(groups) == 2


class TestRelationshipRemap:
    """Test relationship endpoint remapping."""

    def test_remap_endpoints(self):
        rels = [
            _make_rel("a", "b"),
            _make_rel("c", "d"),
        ]
        remap = {"c": "a"}
        result = EmbeddingClusterResolution._remap_relationships(rels, remap)
        assert len(result) == 2
        assert result[1].start_node_id == "a"  # c remapped to a

    def test_self_loops_removed(self):
        """Merging endpoints that creates self-loop should be dropped."""
        rels = [_make_rel("a", "b")]
        remap = {"b": "a"}
        result = EmbeddingClusterResolution._remap_relationships(rels, remap)
        assert len(result) == 0  # self-loop removed

    def test_dedup_relationships(self):
        """Duplicate relationships after remap should be deduplicated."""
        rels = [
            _make_rel("a", "c"),
            _make_rel("b", "c"),  # b→a after remap, same as first rel
        ]
        remap = {"b": "a"}
        result = EmbeddingClusterResolution._remap_relationships(rels, remap)
        assert len(result) == 1


class TestMergeNodeProperties:
    def test_merge_descriptions(self):
        survivor = _make_node("a", "Person", "Alice", "Lives in Wonderland")
        dup = _make_node("b", "Person", "Alice", "Follows the rabbit")
        EmbeddingClusterResolution._merge_node_properties(survivor, dup)
        assert "Wonderland" in survivor.properties["description"]
        assert "rabbit" in survivor.properties["description"]

    def test_merge_source_chunk_ids(self):
        survivor = _make_node("a", "Person", "Alice")
        survivor.properties["source_chunk_ids"] = ["c1"]
        dup = _make_node("b", "Person", "Alice")
        dup.properties["source_chunk_ids"] = ["c2", "c3"]
        EmbeddingClusterResolution._merge_node_properties(survivor, dup)
        assert set(survivor.properties["source_chunk_ids"]) == {"c1", "c2", "c3"}


class TestFullResolve:
    """Integration test for the full resolve pipeline."""

    @pytest.mark.asyncio
    async def test_resolve_with_no_clusters(self):
        """When embeddings are too different, only exact-name dedup happens."""
        # Embedder returns very different vectors
        embedder = AsyncMock()
        embedder.aembed_documents = AsyncMock(
            return_value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        llm = AsyncMock()

        resolver = EmbeddingClusterResolution(
            llm=llm, embedder=embedder, cluster_threshold=0.85
        )
        graph_data = GraphData(
            nodes=[
                _make_node("a", "Person", "Alice"),
                _make_node("b", "Person", "Bob"),
                _make_node("c", "Person", "Charlie"),
            ],
            relationships=[_make_rel("a", "b"), _make_rel("b", "c")],
        )
        ctx = Context(tenant_id="test")
        result = await resolver.resolve(graph_data, ctx)
        assert len(result.nodes) == 3
        assert len(result.relationships) == 2
        assert result.merged_count == 0

    @pytest.mark.asyncio
    async def test_resolve_with_llm_confirmed_merge(self):
        """LLM confirms merge of similar entities."""
        # Embedder returns similar vectors for first two, different for third
        embedder = AsyncMock()
        embedder.aembed_documents = AsyncMock(
            return_value=[[1.0, 0.1, 0.0], [0.99, 0.12, 0.01], [0.0, 0.0, 1.0]]
        )

        # LLM confirms merge of first two
        llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Gen. Grant | Grant"
        mock_batch_item = MagicMock()
        mock_batch_item.ok = True
        mock_batch_item.response = mock_response
        mock_batch_item.index = 0
        llm.abatch_invoke = AsyncMock(return_value=[mock_batch_item])

        resolver = EmbeddingClusterResolution(
            llm=llm, embedder=embedder, cluster_threshold=0.85
        )
        graph_data = GraphData(
            nodes=[
                _make_node("grant_1", "Person", "Gen. Grant", "Union general"),
                _make_node("grant_2", "Person", "Grant", "Led Union forces"),
                _make_node("sherman", "Person", "Sherman", "March to sea"),
            ],
            relationships=[
                _make_rel("grant_1", "sherman"),
                _make_rel("grant_2", "sherman"),
            ],
        )
        ctx = Context(tenant_id="test")
        result = await resolver.resolve(graph_data, ctx)
        # Grant merged into Gen. Grant
        assert len(result.nodes) == 2
        assert result.merged_count == 1
        # Both rels now point from same entity, so deduped
        assert len(result.relationships) == 1
