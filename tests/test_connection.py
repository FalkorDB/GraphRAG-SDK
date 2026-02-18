"""Tests for core/connection.py — async-only FalkorDB connection."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection


class TestConnectionConfig:
    def test_defaults(self):
        cfg = ConnectionConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 6379
        assert cfg.username is None
        assert cfg.password is None
        assert cfg.graph_name == "knowledge_graph"
        assert cfg.max_connections == 16
        assert cfg.retry_count == 3
        assert cfg.retry_delay == 1.0

    def test_custom_values(self):
        cfg = ConnectionConfig(
            host="db.example.com",
            port=6380,
            username="admin",
            password="secret",
            graph_name="my_graph",
        )
        assert cfg.host == "db.example.com"
        assert cfg.port == 6380
        assert cfg.username == "admin"
        assert cfg.graph_name == "my_graph"


class TestFalkorDBConnection:
    def test_default_config(self):
        conn = FalkorDBConnection()
        assert conn.config.host == "localhost"
        assert conn._driver is None
        assert conn._graph is None
        assert conn._pool is None

    def test_custom_config(self):
        cfg = ConnectionConfig(host="remote", port=9999)
        conn = FalkorDBConnection(cfg)
        assert conn.config.host == "remote"
        assert conn.config.port == 9999

    async def test_close_resets(self):
        conn = FalkorDBConnection()
        conn._pool = MagicMock()
        conn._pool.aclose = AsyncMock()
        conn._driver = MagicMock()
        conn._graph = MagicMock()
        await conn.close()
        assert conn._driver is None
        assert conn._graph is None
        assert conn._pool is None

    def test_import_error(self):
        conn = FalkorDBConnection()
        with patch.dict("sys.modules", {"falkordb.asyncio": None, "falkordb": None}):
            with pytest.raises(ImportError):
                conn._ensure_client()

    async def test_query_retries(self):
        conn = FalkorDBConnection(ConnectionConfig(retry_count=3, retry_delay=0.0))

        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(
            side_effect=[Exception("fail 1"), Exception("fail 2"), "success"]
        )
        conn._graph = mock_graph
        conn._driver = MagicMock()  # mark as initialised

        result = await conn.query("MATCH (n) RETURN n")
        assert result == "success"
        assert mock_graph.query.call_count == 3

    async def test_query_exhausts_retries(self):
        conn = FalkorDBConnection(ConnectionConfig(retry_count=2, retry_delay=0.0))

        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=Exception("always fails"))
        conn._graph = mock_graph
        conn._driver = MagicMock()

        with pytest.raises(Exception, match="always fails"):
            await conn.query("MATCH (n) RETURN n")
        assert mock_graph.query.call_count == 2

    async def test_query_with_params(self):
        conn = FalkorDBConnection(ConnectionConfig(retry_count=1))

        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(return_value="ok")
        conn._graph = mock_graph
        conn._driver = MagicMock()

        result = await conn.query("MATCH (n {id: $id}) RETURN n", {"id": "test"})
        assert result == "ok"
        mock_graph.query.assert_called_once_with(
            "MATCH (n {id: $id}) RETURN n", params={"id": "test"}, timeout=None
        )

    async def test_graph_property_lazy_init(self):
        conn = FalkorDBConnection()
        mock_graph = MagicMock()
        mock_driver = MagicMock()
        mock_driver.select_graph = MagicMock(return_value=mock_graph)

        with patch("graphrag_sdk.core.connection.FalkorDBConnection._ensure_client") as mock_init:
            conn._driver = mock_driver
            conn._graph = mock_graph
            assert conn.graph is mock_graph
