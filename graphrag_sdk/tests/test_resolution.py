"""Tests for ingestion/resolution_strategies/exact_match.py."""
from __future__ import annotations

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, GraphNode, GraphRelationship, ResolutionResult
from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution


class TestExactMatchResolution:
    async def test_no_duplicates(self, ctx, sample_graph_data):
        """No duplicates — all nodes survive, merged_count = 0."""
        resolver = ExactMatchResolution()
        result = await resolver.resolve(sample_graph_data, ctx)
        assert result.merged_count == 0
        assert len(result.nodes) == 3
        assert len(result.relationships) == 2

    async def test_duplicate_by_name(self, ctx, sample_graph_data_with_duplicates):
        """alice-1 and alice-2 share name='Alice' so should merge."""
        resolver = ExactMatchResolution(resolve_property="name")
        result = await resolver.resolve(sample_graph_data_with_duplicates, ctx)
        assert result.merged_count == 1
        assert len(result.nodes) == 3  # Alice(merged), Bob, Acme

        # The survived Alice should have merged properties
        alice = next(n for n in result.nodes if n.properties.get("name") == "Alice")
        assert "role" in alice.properties or "age" in alice.properties

    async def test_duplicate_relationships_deduped(self, ctx):
        """When two nodes merge, their duplicate rels should collapse."""
        data = GraphData(
            nodes=[
                GraphNode(id="a1", label="X", properties={"name": "A"}),
                GraphNode(id="a2", label="X", properties={"name": "A"}),
                GraphNode(id="b", label="Y", properties={"name": "B"}),
            ],
            relationships=[
                GraphRelationship(start_node_id="a1", end_node_id="b", type="REL"),
                GraphRelationship(start_node_id="a2", end_node_id="b", type="REL"),
            ],
        )
        resolver = ExactMatchResolution(resolve_property="name")
        result = await resolver.resolve(data, ctx)
        assert result.merged_count == 1
        # Both rels point (a->b REL), should deduplicate
        assert len(result.relationships) == 1

    async def test_property_merging(self, ctx):
        """Survivor inherits properties from duplicates."""
        data = GraphData(
            nodes=[
                GraphNode(
                    id="x1", label="T", properties={"name": "X", "color": "red"}
                ),
                GraphNode(
                    id="x2", label="T", properties={"name": "X", "size": "large"}
                ),
            ],
            relationships=[],
        )
        resolver = ExactMatchResolution(resolve_property="name")
        result = await resolver.resolve(data, ctx)
        assert len(result.nodes) == 1
        merged = result.nodes[0]
        assert merged.properties["color"] == "red"
        assert merged.properties["size"] == "large"

    async def test_empty_input(self, ctx):
        data = GraphData()
        resolver = ExactMatchResolution()
        result = await resolver.resolve(data, ctx)
        assert result.merged_count == 0
        assert len(result.nodes) == 0
        assert len(result.relationships) == 0

    async def test_default_resolve_property_is_id(self, ctx):
        """Default resolves by 'id' in properties, falling back to node.id."""
        data = GraphData(
            nodes=[
                GraphNode(id="same", label="T", properties={"x": 1}),
                GraphNode(id="same", label="T", properties={"y": 2}),
            ],
            relationships=[],
        )
        resolver = ExactMatchResolution()  # resolve_property="id"
        result = await resolver.resolve(data, ctx)
        # Both have id="same" → should merge
        assert result.merged_count == 1
        assert len(result.nodes) == 1

    async def test_different_labels_not_merged(self, ctx):
        """Nodes with same name but different labels stay separate."""
        data = GraphData(
            nodes=[
                GraphNode(id="a1", label="Person", properties={"name": "Alice"}),
                GraphNode(id="a2", label="Company", properties={"name": "Alice"}),
            ],
            relationships=[],
        )
        resolver = ExactMatchResolution(resolve_property="name")
        result = await resolver.resolve(data, ctx)
        assert result.merged_count == 0
        assert len(result.nodes) == 2

    async def test_relationship_remapping(self, ctx):
        """Relationships pointing to merged nodes get remapped."""
        data = GraphData(
            nodes=[
                GraphNode(id="a1", label="T", properties={"name": "A"}),
                GraphNode(id="a2", label="T", properties={"name": "A"}),
                GraphNode(id="b", label="T", properties={"name": "B"}),
                GraphNode(id="c", label="T", properties={"name": "C"}),
            ],
            relationships=[
                GraphRelationship(start_node_id="b", end_node_id="a1", type="LINK"),
                GraphRelationship(start_node_id="c", end_node_id="a2", type="LINK"),
            ],
        )
        resolver = ExactMatchResolution(resolve_property="name")
        result = await resolver.resolve(data, ctx)
        # Both rels should now point to the survivor's id
        survivor_id = next(n.id for n in result.nodes if n.properties["name"] == "A")
        for rel in result.relationships:
            if rel.type == "LINK":
                assert rel.end_node_id == survivor_id
