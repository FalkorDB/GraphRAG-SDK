"""Tests for fact indexing in the pipeline."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    EntityMention,
    FactTriple,
    GraphData,
    GraphNode,
    GraphRelationship,
)
from graphrag_sdk.ingestion.pipeline import IngestionPipeline


# ── Helpers ────────────────────────────────────────────────────


def _make_pipeline(mock_vector_store) -> IngestionPipeline:
    return IngestionPipeline(
        loader=MagicMock(),
        chunker=MagicMock(),
        extractor=MagicMock(),
        resolver=MagicMock(),
        graph_store=MagicMock(),
        vector_store=mock_vector_store,
    )


# ── Tests ──────────────────────────────────────────────────────


class TestIndexFactsFromAttachedFacts:
    async def test_indexes_facts_from_graph_data(self, ctx, mock_vector_store):
        """When graph_data has .facts attached, use those."""
        pipeline = _make_pipeline(mock_vector_store)

        graph_data = GraphData(nodes=[], relationships=[])
        graph_data.facts = [  # type: ignore[attr-defined]
            FactTriple(
                subject="Alice",
                predicate="WORKS_AT",
                object="Acme",
                source_chunk_id="chunk-0",
            ),
            FactTriple(
                subject="Bob",
                predicate="KNOWS",
                object="Alice",
                source_chunk_id="chunk-1",
            ),
        ]

        count = await pipeline._index_facts(graph_data, ctx)

        mock_vector_store.index_facts.assert_called_once()
        args = mock_vector_store.index_facts.call_args
        fact_strings = args[0][0]
        facts = args[0][1]
        assert len(fact_strings) == 2
        assert len(facts) == 2
        assert "(Alice, WORKS_AT, Acme)" in fact_strings[0]


class TestIndexFactsFromRelationships:
    async def test_generates_facts_from_relationships(self, ctx, mock_vector_store):
        """When no .facts attached, generate from relationships."""
        pipeline = _make_pipeline(mock_vector_store)

        graph_data = GraphData(
            nodes=[],
            relationships=[
                GraphRelationship(
                    start_node_id="alice",
                    end_node_id="acme",
                    type="WORKS_AT",
                    properties={"source_chunk_id": "c0"},
                ),
                GraphRelationship(
                    start_node_id="bob",
                    end_node_id="acme",
                    type="WORKS_AT",
                    properties={"source_chunk_id": "c1"},
                ),
            ],
        )

        await pipeline._index_facts(graph_data, ctx)

        mock_vector_store.index_facts.assert_called_once()
        args = mock_vector_store.index_facts.call_args
        facts = args[0][1]
        assert len(facts) == 2

    async def test_skips_synonym_relationships(self, ctx, mock_vector_store):
        """SYNONYM relationships should NOT become facts."""
        pipeline = _make_pipeline(mock_vector_store)

        graph_data = GraphData(
            nodes=[],
            relationships=[
                GraphRelationship(
                    start_node_id="a",
                    end_node_id="b",
                    type="SYNONYM",
                    properties={"similarity": 0.9},
                ),
                GraphRelationship(
                    start_node_id="alice",
                    end_node_id="acme",
                    type="WORKS_AT",
                    properties={},
                ),
            ],
        )

        await pipeline._index_facts(graph_data, ctx)

        args = mock_vector_store.index_facts.call_args
        facts = args[0][1]
        assert len(facts) == 1
        assert facts[0].predicate == "WORKS_AT"


class TestIndexFactsNoFacts:
    async def test_no_facts_no_rels_returns_zero(self, ctx, mock_vector_store):
        """No facts and no relationships → 0 facts indexed."""
        pipeline = _make_pipeline(mock_vector_store)
        graph_data = GraphData(nodes=[], relationships=[])

        count = await pipeline._index_facts(graph_data, ctx)

        assert count == 0
        mock_vector_store.index_facts.assert_not_called()


class TestIndexFactsVectorStoreCall:
    async def test_correct_arguments_passed(self, ctx, mock_vector_store):
        """Verify index_facts receives correct fact_strings and facts."""
        pipeline = _make_pipeline(mock_vector_store)

        fact = FactTriple(
            subject="X",
            predicate="REL",
            object="Y",
            source_chunk_id="c0",
            weight=0.5,
        )
        graph_data = GraphData(nodes=[], relationships=[])
        graph_data.facts = [fact]  # type: ignore[attr-defined]

        await pipeline._index_facts(graph_data, ctx)

        args = mock_vector_store.index_facts.call_args
        assert args[0][0] == ["(X, REL, Y)"]
        assert args[0][1] == [fact]
