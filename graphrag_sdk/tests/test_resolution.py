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
    """Bug 2: exact_match_merge with cross_label_merge=True should merge
    same-name entities across different labels when LLM confirms."""

    async def test_cross_label_merge_yes(self):
        """LLM confirms same entity → merged into canonical type."""
        nodes = [
            GraphNode(id="fdb__tech", label="Technology",
                      properties={"name": "FalkorDB", "description": "A graph database engine"}),
            GraphNode(id="fdb__org", label="Organization",
                      properties={"name": "FalkorDB", "description": "The company behind FalkorDB"}),
        ]
        llm = MockLLM(responses=["YES Technology\nSame entity, technology is more accurate."])
        deduped, remap, count = await exact_match_merge(
            nodes, llm, cross_label_merge=True,
        )
        assert len(deduped) == 1
        assert deduped[0].label == "Technology"
        assert count == 1
        assert "fdb__org" in remap
        assert remap["fdb__org"] == "fdb__tech"

    async def test_cross_label_merge_no(self):
        """LLM says different entities → kept separate."""
        nodes = [
            GraphNode(id="paris__person", label="Person",
                      properties={"name": "Paris", "description": "A character named Paris"}),
            GraphNode(id="paris__loc", label="Location",
                      properties={"name": "Paris", "description": "Capital of France"}),
        ]
        llm = MockLLM(responses=["NO\nDifferent entities — person vs city."])
        deduped, remap, count = await exact_match_merge(
            nodes, llm, cross_label_merge=True,
        )
        assert len(deduped) == 2
        assert count == 0
        assert len(remap) == 0

    async def test_cross_label_merge_disabled_by_default(self):
        """Without cross_label_merge=True, same-name cross-label nodes stay separate."""
        nodes = [
            GraphNode(id="fdb__tech", label="Technology",
                      properties={"name": "FalkorDB", "description": "A graph database"}),
            GraphNode(id="fdb__org", label="Organization",
                      properties={"name": "FalkorDB", "description": "The company"}),
        ]
        llm = MockLLM(responses=["YES Technology\nSame entity."])
        deduped, remap, count = await exact_match_merge(nodes, llm)
        assert len(deduped) == 2
        assert count == 0

    async def test_cross_label_merge_no_llm(self):
        """cross_label_merge=True but no LLM → no crash, no merge."""
        nodes = [
            GraphNode(id="fdb__tech", label="Technology",
                      properties={"name": "FalkorDB", "description": "DB"}),
            GraphNode(id="fdb__org", label="Organization",
                      properties={"name": "FalkorDB", "description": "Company"}),
        ]
        deduped, remap, count = await exact_match_merge(
            nodes, None, cross_label_merge=True,
        )
        assert len(deduped) == 2
        assert count == 0

    async def test_cross_label_merge_three_labels(self):
        """Three labels for one name → LLM picks canonical, all merge."""
        nodes = [
            GraphNode(id="fdb__tech", label="Technology",
                      properties={"name": "FalkorDB", "description": "Graph database engine"}),
            GraphNode(id="fdb__org", label="Organization",
                      properties={"name": "FalkorDB", "description": "The company"}),
            GraphNode(id="fdb__unk", label="Unknown",
                      properties={"name": "FalkorDB", "description": "FalkorDB system"}),
        ]
        llm = MockLLM(responses=["YES Technology\nAll refer to the graph database."])
        deduped, remap, count = await exact_match_merge(
            nodes, llm, cross_label_merge=True,
        )
        assert len(deduped) == 1
        assert deduped[0].label == "Technology"
        assert count == 2
        assert remap["fdb__org"] == "fdb__tech"
        assert remap["fdb__unk"] == "fdb__tech"

    async def test_cross_label_transitive_remap(self):
        """Phase 1 same-label merge + cross-label merge chains resolve transitively."""
        nodes = [
            GraphNode(id="fdb__tech_1", label="Technology",
                      properties={"name": "FalkorDB", "description": "Graph DB v1"}),
            GraphNode(id="fdb__tech_2", label="Technology",
                      properties={"name": "FalkorDB", "description": "Graph DB v2"}),
            GraphNode(id="fdb__org", label="Organization",
                      properties={"name": "FalkorDB", "description": "The company"}),
        ]
        llm = MockLLM(responses=["YES Technology\nSame entity."])
        deduped, remap, count = await exact_match_merge(
            nodes, llm, cross_label_merge=True,
        )
        # fdb__tech_1 and fdb__tech_2 merge in Phase 1 (same label+name)
        # Then cross-label merges fdb__org into the tech survivor
        assert len(deduped) == 1
        assert deduped[0].label == "Technology"
        # fdb__org should transitively point to the tech survivor
        tech_survivor = deduped[0].id
        assert remap.get("fdb__org") == tech_survivor

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
        llm = MockLLM(responses=["YES Technology\nSame entity."])
        deduped, remap, count = await exact_match_merge(
            nodes, llm, cross_label_merge=True,
        )
        survivor = deduped[0]
        assert "graph database" in survivor.properties["description"]
        assert "company" in survivor.properties["description"]
        assert "c1" in survivor.properties["source_chunk_ids"]
        assert "c2" in survivor.properties["source_chunk_ids"]
