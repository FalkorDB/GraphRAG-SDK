"""Tests for ingestion/resolution_strategies/exact_match.py and base.py."""
from __future__ import annotations

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, GraphNode, GraphRelationship, ResolutionResult
from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution
from graphrag_sdk.ingestion.resolution_strategies.base import exact_match_merge

from .conftest import MockLLM


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


class TestCrossLabelMerge:
    """Bug 2: exact_match_merge with cross_label_merge=True groups by name
    only, merging same-name entities across labels. Uses enhanced summary
    prompt for type selection when 3+ descriptions; heuristic otherwise."""

    async def test_cross_label_merge_heuristic(self):
        """Two cross-label nodes (below summary threshold) → heuristic picks label."""
        nodes = [
            GraphNode(id="fdb__tech", label="Technology",
                      properties={"name": "FalkorDB", "description": "A graph database engine"}),
            GraphNode(id="fdb__org", label="Organization",
                      properties={"name": "FalkorDB", "description": "The company behind FalkorDB"}),
        ]
        deduped, remap, count = await exact_match_merge(
            nodes, None, cross_label_merge=True,
        )
        assert len(deduped) == 1
        assert count == 1
        # Heuristic: both have count 1, tie broken lexicographically
        assert deduped[0].label == "Organization"  # O < T

    async def test_cross_label_merge_prefers_non_unknown(self):
        """Heuristic picks specific type over Unknown."""
        nodes = [
            GraphNode(id="fdb__tech", label="Technology",
                      properties={"name": "FalkorDB", "description": "A graph DB"}),
            GraphNode(id="fdb__unk", label="Unknown",
                      properties={"name": "FalkorDB", "description": "FalkorDB system"}),
        ]
        deduped, remap, count = await exact_match_merge(
            nodes, None, cross_label_merge=True,
        )
        assert len(deduped) == 1
        assert deduped[0].label == "Technology"
        assert count == 1

    async def test_cross_label_merge_disabled_by_default(self):
        """Without cross_label_merge=True, same-name cross-label nodes stay separate."""
        nodes = [
            GraphNode(id="fdb__tech", label="Technology",
                      properties={"name": "FalkorDB", "description": "A graph database"}),
            GraphNode(id="fdb__org", label="Organization",
                      properties={"name": "FalkorDB", "description": "The company"}),
        ]
        deduped, remap, count = await exact_match_merge(nodes, None)
        assert len(deduped) == 2
        assert count == 0

    async def test_cross_label_merge_three_labels_with_summary(self):
        """Three labels (3+ descriptions) → enhanced summary prompt picks type."""
        nodes = [
            GraphNode(id="fdb__tech", label="Technology",
                      properties={"name": "FalkorDB", "description": "Graph database engine"}),
            GraphNode(id="fdb__org", label="Organization",
                      properties={"name": "FalkorDB", "description": "The company behind it"}),
            GraphNode(id="fdb__unk", label="Unknown",
                      properties={"name": "FalkorDB", "description": "FalkorDB system"}),
        ]
        # LLM returns: first line = chosen type, second line = summary
        llm = MockLLM(responses=["Technology\nFalkorDB is a graph database engine."])
        deduped, remap, count = await exact_match_merge(
            nodes, llm, cross_label_merge=True,
        )
        assert len(deduped) == 1
        assert deduped[0].label == "Technology"
        assert count == 2

    async def test_cross_label_all_in_one_group(self):
        """All same-name nodes land in one group regardless of label."""
        nodes = [
            GraphNode(id="fdb__tech_1", label="Technology",
                      properties={"name": "FalkorDB", "description": "Graph DB v1"}),
            GraphNode(id="fdb__tech_2", label="Technology",
                      properties={"name": "FalkorDB", "description": "Graph DB v2"}),
            GraphNode(id="fdb__org", label="Organization",
                      properties={"name": "FalkorDB", "description": "The company"}),
        ]
        # 3 descriptions → summary prompt fires, LLM picks type
        llm = MockLLM(responses=["Technology\nFalkorDB is a graph database."])
        deduped, remap, count = await exact_match_merge(
            nodes, llm, cross_label_merge=True,
        )
        assert len(deduped) == 1
        assert count == 2
        # All duplicates remap to the survivor
        survivor_id = deduped[0].id
        for node in nodes:
            if node.id != survivor_id:
                assert remap[node.id] == survivor_id

    async def test_cross_label_merge_preserves_descriptions(self):
        """Merged descriptions from cross-label nodes should be combined."""
        nodes = [
            GraphNode(id="fdb__tech", label="Technology",
                      properties={"name": "FalkorDB", "description": "A graph database",
                                  "source_chunk_ids": ["c1"]}),
            GraphNode(id="fdb__org", label="Organization",
                      properties={"name": "FalkorDB", "description": "The company behind FalkorDB",
                                  "source_chunk_ids": ["c2"]}),
        ]
        deduped, remap, count = await exact_match_merge(
            nodes, None, cross_label_merge=True,
        )
        survivor = deduped[0]
        # Descriptions pipe-joined (below summary threshold)
        assert "graph database" in survivor.properties["description"]
        assert "company" in survivor.properties["description"]
        assert "c1" in survivor.properties["source_chunk_ids"]
        assert "c2" in survivor.properties["source_chunk_ids"]

    async def test_cross_label_same_label_unaffected(self):
        """Same-label groups still merge normally with cross_label_merge=True."""
        nodes = [
            GraphNode(id="a1", label="Person", properties={"name": "Alice", "description": "Engineer"}),
            GraphNode(id="a2", label="Person", properties={"name": "Alice", "description": "Developer"}),
            GraphNode(id="b1", label="Person", properties={"name": "Bob", "description": "Manager"}),
        ]
        deduped, remap, count = await exact_match_merge(
            nodes, None, cross_label_merge=True,
        )
        assert len(deduped) == 2  # Alice (merged) + Bob
        assert count == 1
        names = {n.properties["name"] for n in deduped}
        assert names == {"Alice", "Bob"}
