"""Tests for ingestion/resolution_strategies/semantic_resolution.py."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    GraphData,
    GraphNode,
    GraphRelationship,
    ResolutionResult,
)
from graphrag_sdk.ingestion.resolution_strategies.semantic_resolution import (
    SemanticResolution,
)

from .conftest import MockEmbedder, MockLLM


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def ctx() -> Context:
    return Context(tenant_id="test-tenant", latency_budget_ms=5000.0)


@pytest.fixture
def graph_with_same_name_diff_label() -> GraphData:
    """Two entities named 'Paris' but different labels: Person vs Location."""
    return GraphData(
        nodes=[
            GraphNode(
                id="paris_person",
                label="Person",
                properties={"name": "Paris", "description": "A character in the story"},
            ),
            GraphNode(
                id="paris_location",
                label="Location",
                properties={"name": "Paris", "description": "Capital of France"},
            ),
        ],
        relationships=[
            GraphRelationship(
                start_node_id="paris_person",
                end_node_id="paris_location",
                type="LIVES_IN",
                properties={},
            ),
        ],
    )


@pytest.fixture
def graph_with_duplicates() -> GraphData:
    """Two nodes with same name AND same label — should be merged."""
    return GraphData(
        nodes=[
            GraphNode(
                id="alice-1",
                label="Person",
                properties={"name": "Alice", "description": "An engineer"},
            ),
            GraphNode(
                id="alice-2",
                label="Person",
                properties={"name": "Alice", "description": "A software developer"},
            ),
            GraphNode(
                id="acme",
                label="Company",
                properties={"name": "Acme Corp"},
            ),
        ],
        relationships=[
            GraphRelationship(
                start_node_id="alice-1",
                end_node_id="acme",
                type="WORKS_AT",
                properties={},
            ),
            GraphRelationship(
                start_node_id="alice-2",
                end_node_id="acme",
                type="WORKS_AT",
                properties={},
            ),
        ],
    )


# ── Tests ───────────────────────────────────────────────────────


class TestSemanticResolutionCrossTypeSafety:
    """Entities with same name but different labels must NOT be merged."""

    async def test_same_name_diff_label_kept_separate(
        self, ctx, graph_with_same_name_diff_label
    ):
        resolver = SemanticResolution()
        result = await resolver.resolve(graph_with_same_name_diff_label, ctx)

        assert len(result.nodes) == 2
        assert result.merged_count == 0
        labels = {n.label for n in result.nodes}
        assert labels == {"Person", "Location"}

    async def test_relationship_preserved(
        self, ctx, graph_with_same_name_diff_label
    ):
        resolver = SemanticResolution()
        result = await resolver.resolve(graph_with_same_name_diff_label, ctx)

        assert len(result.relationships) == 1
        rel = result.relationships[0]
        assert rel.start_node_id == "paris_person"
        assert rel.end_node_id == "paris_location"


class TestSemanticResolutionSameTypeMerge:
    """Entities with same name AND same label should be merged."""

    async def test_same_name_same_label_merged(
        self, ctx, graph_with_duplicates
    ):
        resolver = SemanticResolution()
        result = await resolver.resolve(graph_with_duplicates, ctx)

        # Alice-1 and Alice-2 merged into one
        assert len(result.nodes) == 2  # Alice + Acme
        assert result.merged_count == 1

        # Both WORKS_AT relationships should point to the same Alice
        alice_ids = {n.id for n in result.nodes if n.label == "Person"}
        assert len(alice_ids) == 1

    async def test_relationships_remapped_and_deduped(
        self, ctx, graph_with_duplicates
    ):
        resolver = SemanticResolution()
        result = await resolver.resolve(graph_with_duplicates, ctx)

        # Two WORKS_AT rels should be deduped to one (same source after merge)
        works_at = [r for r in result.relationships if r.type == "WORKS_AT"]
        assert len(works_at) == 1

    async def test_descriptions_merged(
        self, ctx, graph_with_duplicates
    ):
        resolver = SemanticResolution()
        result = await resolver.resolve(graph_with_duplicates, ctx)

        alice = [n for n in result.nodes if n.label == "Person"][0]
        desc = alice.properties.get("description", "")
        # Both descriptions should be present (concatenated)
        assert "engineer" in desc.lower() or "developer" in desc.lower()


class TestSemanticResolutionWithLLM:
    """LLM summarisation for groups at or above threshold."""

    async def test_llm_summary_for_large_groups(self, ctx):
        llm = MockLLM(responses=["A versatile software professional"])

        nodes = [
            GraphNode(
                id=f"alice-{i}",
                label="Person",
                properties={
                    "name": "Alice",
                    "description": f"Description {i}",
                },
            )
            for i in range(4)
        ]
        gd = GraphData(nodes=nodes, relationships=[])

        resolver = SemanticResolution(
            llm=llm,
            force_summary_threshold=3,
        )
        result = await resolver.resolve(gd, ctx)

        assert len(result.nodes) == 1
        assert result.merged_count == 3
        # LLM should have been called for summary
        assert llm._call_index >= 1


class TestSemanticResolutionFuzzyMerge:
    """Embedding-based fuzzy merge within same label."""

    async def test_fuzzy_merge_near_duplicates(self, ctx):
        embedder = MockEmbedder(dimension=8)

        # Two nodes with slightly different names but same label
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="tolkien_1",
                    label="Person",
                    properties={"name": "J.R.R. Tolkien"},
                ),
                GraphNode(
                    id="tolkien_2",
                    label="Person",
                    properties={"name": "Tolkien"},
                ),
                GraphNode(
                    id="acme",
                    label="Company",
                    properties={"name": "Acme Corp"},
                ),
            ],
            relationships=[],
        )

        # With a very low threshold, no merge happens (names are different enough)
        resolver = SemanticResolution(
            embedder=embedder,
            similarity_threshold=0.99999,  # Very high — won't merge
        )
        result = await resolver.resolve(gd, ctx)
        # Tolkien_1 and Tolkien_2 have different normalized names, so they stay as 2
        assert len(result.nodes) == 3

    async def test_no_fuzzy_merge_without_embedder(self, ctx):
        gd = GraphData(
            nodes=[
                GraphNode(id="a", label="Person", properties={"name": "Alice"}),
                GraphNode(id="b", label="Person", properties={"name": "Alice B"}),
            ],
            relationships=[],
        )

        resolver = SemanticResolution(embedder=None)
        result = await resolver.resolve(gd, ctx)

        # No fuzzy merge without embedder — both nodes remain
        assert len(result.nodes) == 2


class TestSemanticResolutionEdgeCases:
    async def test_empty_graph(self, ctx):
        resolver = SemanticResolution()
        result = await resolver.resolve(
            GraphData(nodes=[], relationships=[]), ctx
        )
        assert len(result.nodes) == 0
        assert len(result.relationships) == 0
        assert result.merged_count == 0

    async def test_single_node(self, ctx):
        resolver = SemanticResolution()
        result = await resolver.resolve(
            GraphData(
                nodes=[GraphNode(id="a", label="X", properties={"name": "A"})],
                relationships=[],
            ),
            ctx,
        )
        assert len(result.nodes) == 1
        assert result.merged_count == 0

    async def test_case_insensitive_grouping(self, ctx):
        """'alice' and 'Alice' should be grouped together."""
        gd = GraphData(
            nodes=[
                GraphNode(id="a1", label="Person", properties={"name": "Alice"}),
                GraphNode(id="a2", label="Person", properties={"name": "alice"}),
            ],
            relationships=[],
        )
        resolver = SemanticResolution()
        result = await resolver.resolve(gd, ctx)
        assert len(result.nodes) == 1
        assert result.merged_count == 1
