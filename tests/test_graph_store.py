"""Tests for storage/graph_store.py — Repository pattern for graph operations."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.connection import FalkorDBConnection
from graphrag_sdk.core.exceptions import DatabaseError
from graphrag_sdk.core.models import GraphNode, GraphRelationship
from graphrag_sdk.storage.graph_store import GraphStore


@pytest.fixture
def graph_store(mock_connection):
    return GraphStore(mock_connection)


class TestGraphStoreUpsertNodes:
    async def test_upsert_single_node(self, graph_store, mock_connection):
        nodes = [GraphNode(id="n1", label="Person", properties={"name": "Alice"})]
        result = await graph_store.upsert_nodes(nodes)
        assert result == 1
        mock_connection.query.assert_called_once()
        cypher = mock_connection.query.call_args[0][0]
        assert "UNWIND" in cypher
        assert "MERGE" in cypher
        assert "Person" in cypher
        assert "__Entity__" in cypher

    async def test_upsert_multiple_nodes(self, graph_store, mock_connection):
        nodes = [
            GraphNode(id="n1", label="Person", properties={"name": "Alice"}),
            GraphNode(id="n2", label="Company", properties={"name": "Acme"}),
        ]
        result = await graph_store.upsert_nodes(nodes)
        assert result == 2
        assert mock_connection.query.call_count == 2

    async def test_upsert_empty_list(self, graph_store, mock_connection):
        result = await graph_store.upsert_nodes([])
        assert result == 0
        mock_connection.query.assert_not_called()

    async def test_upsert_raises_on_error(self, graph_store, mock_connection):
        mock_connection.query = AsyncMock(side_effect=Exception("db error"))
        with pytest.raises(DatabaseError, match="Node upsert failed"):
            await graph_store.upsert_nodes(
                [GraphNode(id="x", label="T", properties={})]
            )

    async def test_upsert_passes_id_in_batch_param(self, graph_store, mock_connection):
        await graph_store.upsert_nodes(
            [GraphNode(id="test-id", label="X", properties={})]
        )
        params = mock_connection.query.call_args[0][1]
        assert params["batch"][0]["id"] == "test-id"


class TestGraphStoreUpsertRelationships:
    async def test_upsert_relationship(self, graph_store, mock_connection):
        rels = [
            GraphRelationship(
                start_node_id="a", end_node_id="b", type="KNOWS", properties={"since": 2020}
            )
        ]
        result = await graph_store.upsert_relationships(rels)
        assert result == 1
        cypher = mock_connection.query.call_args[0][0]
        assert "MERGE" in cypher
        assert "KNOWS" in cypher

    async def test_upsert_empty_relationships(self, graph_store, mock_connection):
        result = await graph_store.upsert_relationships([])
        assert result == 0

    async def test_upsert_rel_error_continues(self, graph_store, mock_connection):
        """Relationship upsert logs warning but continues.

        With UNWIND batching, rels are grouped by type. Each type group
        gets one UNWIND call. If that fails, per-item fallback runs.
        R1 type: batch fails → individual fallback also fails → logged
        R2 type: batch succeeds.
        """
        mock_connection.query = AsyncMock(
            side_effect=[Exception("fail"), Exception("fail again"), MagicMock()]
        )
        rels = [
            GraphRelationship(start_node_id="a", end_node_id="b", type="R1"),
            GraphRelationship(start_node_id="c", end_node_id="d", type="R2"),
        ]
        # R1 batch fails (call 1), R1 individual fails (call 2), R2 batch succeeds (call 3)
        result = await graph_store.upsert_relationships(rels)
        assert result == 1  # only R2 batch succeeded


class TestGraphStoreGetConnectedEntities:
    async def test_get_entities(self, graph_store, mock_connection):
        result_mock = MagicMock()
        result_mock.result_set = [
            ["e1", ["Person"], {"name": "Alice"}],
            ["e2", ["Company"], {"name": "Acme"}],
        ]
        mock_connection.query = AsyncMock(return_value=result_mock)
        entities = await graph_store.get_connected_entities("chunk-1")
        assert len(entities) == 2
        assert entities[0]["id"] == "e1"
        assert entities[1]["labels"] == ["Company"]

    async def test_get_entities_empty(self, graph_store, mock_connection):
        result_mock = MagicMock()
        result_mock.result_set = []
        mock_connection.query = AsyncMock(return_value=result_mock)
        entities = await graph_store.get_connected_entities("nonexistent")
        assert entities == []

    async def test_get_entities_error_returns_empty(self, graph_store, mock_connection):
        mock_connection.query = AsyncMock(side_effect=Exception("db error"))
        entities = await graph_store.get_connected_entities("chunk-1")
        assert entities == []

    async def test_custom_max_hops(self, graph_store, mock_connection):
        result_mock = MagicMock()
        result_mock.result_set = []
        mock_connection.query = AsyncMock(return_value=result_mock)
        await graph_store.get_connected_entities("c1", max_hops=3)
        cypher = mock_connection.query.call_args[0][0]
        assert "*1..3" in cypher


class TestGraphStoreQueryRaw:
    async def test_raw_query(self, graph_store, mock_connection):
        await graph_store.query_raw("MATCH (n) RETURN n LIMIT 10")
        mock_connection.query.assert_called_once_with("MATCH (n) RETURN n LIMIT 10", None)

    async def test_raw_query_with_params(self, graph_store, mock_connection):
        await graph_store.query_raw("MATCH (n {id: $id})", {"id": "test"})
        mock_connection.query.assert_called_once_with("MATCH (n {id: $id})", {"id": "test"})


class TestGraphStoreDeleteAll:
    async def test_delete_all(self, graph_store, mock_connection):
        await graph_store.delete_all()
        cypher = mock_connection.query.call_args[0][0]
        assert "DETACH DELETE" in cypher


class TestCleanProperties:
    def test_removes_none(self):
        result = GraphStore._clean_properties({"a": 1, "b": None, "c": "ok"})
        assert "b" not in result
        assert result["a"] == 1
        assert result["c"] == "ok"

    def test_preserves_primitives(self):
        result = GraphStore._clean_properties(
            {"s": "text", "i": 42, "f": 3.14, "b": True}
        )
        assert result == {"s": "text", "i": 42, "f": 3.14, "b": True}

    def test_preserves_lists(self):
        result = GraphStore._clean_properties({"tags": ["a", "b"]})
        assert result["tags"] == ["a", "b"]

    def test_converts_objects_to_str(self):
        result = GraphStore._clean_properties({"obj": {"nested": True}})
        assert isinstance(result["obj"], str)

    def test_empty_dict(self):
        assert GraphStore._clean_properties({}) == {}
