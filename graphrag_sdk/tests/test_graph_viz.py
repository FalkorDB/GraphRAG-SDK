"""Tests for utils/graph_viz.py — graph visualization helpers."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.utils.graph_viz import GraphVisualizer


@pytest.fixture
def visualizer(mock_connection):
    return GraphVisualizer(mock_connection)


class TestGraphVisualizer:
    async def test_get_stats(self, visualizer, mock_connection):
        results = [
            MagicMock(result_set=[[42]]),          # node count
            MagicMock(result_set=[[15]]),           # rel count
            MagicMock(result_set=[                  # labels
                [["Person"], 20],
                [["Company"], 22],
            ]),
            MagicMock(result_set=[                  # rel types
                ["WORKS_AT", 10],
                ["KNOWS", 5],
            ]),
        ]
        mock_connection.query = AsyncMock(side_effect=results)
        stats = await visualizer.get_stats()
        assert stats["node_count"] == 42
        assert stats["relationship_count"] == 15
        assert "labels" in stats
        assert "relationship_types" in stats

    async def test_get_stats_error(self, visualizer, mock_connection):
        mock_connection.query = AsyncMock(side_effect=Exception("db down"))
        stats = await visualizer.get_stats()
        # Should return partial/empty stats, not raise
        assert isinstance(stats, dict)

    async def test_describe(self, visualizer, mock_connection):
        results = [
            MagicMock(result_set=[[10]]),
            MagicMock(result_set=[[5]]),
            MagicMock(result_set=[[["Person"], 10]]),
            MagicMock(result_set=[["KNOWS", 5]]),
        ]
        mock_connection.query = AsyncMock(side_effect=results)
        desc = await visualizer.describe()
        assert "Knowledge Graph Summary" in desc
        assert "Nodes:" in desc
        assert "10" in desc

    async def test_sample_nodes(self, visualizer, mock_connection):
        result_mock = MagicMock()
        result_mock.result_set = [["node1"], ["node2"]]
        mock_connection.query = AsyncMock(return_value=result_mock)
        samples = await visualizer.sample_nodes(limit=2)
        assert len(samples) == 2

    async def test_sample_nodes_with_label(self, visualizer, mock_connection):
        result_mock = MagicMock()
        result_mock.result_set = [["p1"]]
        mock_connection.query = AsyncMock(return_value=result_mock)
        samples = await visualizer.sample_nodes(label="Person", limit=1)
        cypher = mock_connection.query.call_args[0][0]
        assert "Person" in cypher

    async def test_sample_nodes_error(self, visualizer, mock_connection):
        mock_connection.query = AsyncMock(side_effect=Exception("fail"))
        result = await visualizer.sample_nodes()
        assert result == []
