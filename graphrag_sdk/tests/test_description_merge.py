"""Tests for DescriptionMergeResolution strategy."""
from __future__ import annotations

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    GraphData,
    GraphNode,
    GraphRelationship,
)
from graphrag_sdk.ingestion.resolution_strategies.description_merge import (
    DescriptionMergeResolution,
)

from .conftest import MockLLM


# ── Helpers ────────────────────────────────────────────────────


def _make_node(
    id: str,
    label: str = "Person",
    name: str | None = None,
    description: str = "",
    source_chunk_ids: list[str] | None = None,
) -> GraphNode:
    props: dict = {"name": name or id, "description": description}
    if source_chunk_ids:
        props["source_chunk_ids"] = source_chunk_ids
    return GraphNode(id=id, label=label, properties=props)


# ── Tests ──────────────────────────────────────────────────────


class TestSingleNodeGroups:
    async def test_single_node_passes_through(self, ctx):
        resolver = DescriptionMergeResolution()
        data = GraphData(
            nodes=[_make_node("alice", description="Engineer")],
            relationships=[],
        )
        result = await resolver.resolve(data, ctx)

        assert len(result.nodes) == 1
        assert result.merged_count == 0
        assert result.nodes[0].properties["description"] == "Engineer"


class TestMultiNodeBelowThreshold:
    async def test_descriptions_concatenated(self, ctx):
        resolver = DescriptionMergeResolution(force_summary_threshold=3)
        data = GraphData(
            nodes=[
                _make_node("alice-1", name="Alice", description="An engineer"),
                _make_node("alice-2", name="Alice", description="Works at Acme"),
            ],
            relationships=[],
        )
        result = await resolver.resolve(data, ctx)

        assert len(result.nodes) == 1
        assert result.merged_count == 1
        desc = result.nodes[0].properties["description"]
        assert "An engineer" in desc
        assert "Works at Acme" in desc
        assert " | " in desc

    async def test_source_ids_merged(self, ctx):
        resolver = DescriptionMergeResolution()
        data = GraphData(
            nodes=[
                _make_node("a-1", name="Alice", source_chunk_ids=["c1"]),
                _make_node("a-2", name="Alice", source_chunk_ids=["c2"]),
            ],
            relationships=[],
        )
        result = await resolver.resolve(data, ctx)

        src_ids = result.nodes[0].properties["source_chunk_ids"]
        assert "c1" in src_ids
        assert "c2" in src_ids


class TestMultiNodeAboveThreshold:
    async def test_llm_summarization_triggered(self, ctx):
        summary = "Alice is a versatile professional."
        llm = MockLLM(responses=[summary])
        resolver = DescriptionMergeResolution(
            llm=llm, force_summary_threshold=2
        )
        data = GraphData(
            nodes=[
                _make_node("a-1", name="Alice", description="An engineer"),
                _make_node("a-2", name="Alice", description="Works at Acme"),
                _make_node("a-3", name="Alice", description="Likes GraphRAG"),
            ],
            relationships=[],
        )
        result = await resolver.resolve(data, ctx)

        assert len(result.nodes) == 1
        assert result.nodes[0].properties["description"] == summary

    async def test_no_llm_falls_back_to_concat(self, ctx):
        resolver = DescriptionMergeResolution(
            llm=None, force_summary_threshold=2
        )
        data = GraphData(
            nodes=[
                _make_node("a-1", name="Alice", description="desc A"),
                _make_node("a-2", name="Alice", description="desc B"),
                _make_node("a-3", name="Alice", description="desc C"),
            ],
            relationships=[],
        )
        result = await resolver.resolve(data, ctx)

        desc = result.nodes[0].properties["description"]
        assert " | " in desc


class TestIdRemapping:
    async def test_relationship_endpoints_remapped(self, ctx):
        resolver = DescriptionMergeResolution()
        data = GraphData(
            nodes=[
                _make_node("alice-1", name="Alice"),
                _make_node("alice-2", name="Alice"),
                _make_node("acme", label="Company", name="Acme Corp"),
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
        result = await resolver.resolve(data, ctx)

        # Both alice nodes merged → relationships should point to surviving alice
        assert len(result.nodes) == 2  # alice + acme
        assert len(result.relationships) == 1  # deduplicated
        assert result.relationships[0].start_node_id == "alice-1"
        assert result.relationships[0].end_node_id == "acme"


class TestRelationshipDeduplication:
    async def test_duplicate_rels_removed_after_remap(self, ctx):
        resolver = DescriptionMergeResolution()
        data = GraphData(
            nodes=[
                _make_node("a-1", name="Alice"),
                _make_node("a-2", name="Alice"),
                _make_node("b", label="Company", name="Bob Corp"),
            ],
            relationships=[
                GraphRelationship(
                    start_node_id="a-1", end_node_id="b", type="KNOWS", properties={}
                ),
                GraphRelationship(
                    start_node_id="a-2", end_node_id="b", type="KNOWS", properties={}
                ),
            ],
        )
        result = await resolver.resolve(data, ctx)

        assert len(result.relationships) == 1


class TestNormalization:
    async def test_case_insensitive_grouping(self, ctx):
        resolver = DescriptionMergeResolution()
        data = GraphData(
            nodes=[
                _make_node("a-1", name="Alice"),
                _make_node("a-2", name="alice"),
            ],
            relationships=[],
        )
        result = await resolver.resolve(data, ctx)

        assert len(result.nodes) == 1
        assert result.merged_count == 1

    async def test_whitespace_stripped_for_grouping(self, ctx):
        resolver = DescriptionMergeResolution()
        data = GraphData(
            nodes=[
                _make_node("a-1", name="Alice"),
                _make_node("a-2", name="  Alice  "),
            ],
            relationships=[],
        )
        result = await resolver.resolve(data, ctx)

        assert len(result.nodes) == 1
        assert result.merged_count == 1
