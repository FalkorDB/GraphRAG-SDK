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
        # TLS off by default; verification strict if enabled.
        assert cfg.ssl is False
        assert cfg.ssl_cert_reqs == "required"
        assert cfg.ssl_check_hostname is True

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


class TestConnectionConfigFromURL:
    def test_redis_scheme_keeps_ssl_off(self):
        cfg = ConnectionConfig.from_url("redis://user:pw@host:6380/0")
        assert cfg.host == "host"
        assert cfg.port == 6380
        assert cfg.username == "user"
        assert cfg.password == "pw"
        assert cfg.ssl is False

    def test_rediss_scheme_enables_ssl(self):
        cfg = ConnectionConfig.from_url("rediss://host:6380")
        assert cfg.ssl is True

    def test_explicit_kwargs_override_scheme(self):
        cfg = ConnectionConfig.from_url("rediss://host", ssl=False)
        assert cfg.ssl is False

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            ConnectionConfig.from_url("https://host")


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
        # Default query_timeout_ms is 10_000
        mock_graph.query.assert_called_once_with(
            "MATCH (n {id: $id}) RETURN n", params={"id": "test"}, timeout=10_000
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


class TestFalkorDBConnectionTLS:
    def test_pool_omits_ssl_kwargs_when_ssl_disabled(self):
        from redis.asyncio import BlockingConnectionPool

        conn = FalkorDBConnection(ConnectionConfig(host="h", port=1, ssl=False))
        with patch("redis.asyncio.BlockingConnectionPool") as mock_pool, \
             patch("falkordb.asyncio.FalkorDB") as mock_falkor:
            mock_falkor.return_value.select_graph = MagicMock()
            conn._ensure_client()
            kwargs = mock_pool.call_args.kwargs
            assert "connection_class" not in kwargs
            assert "ssl_cert_reqs" not in kwargs

    def test_pool_passes_ssl_kwargs_when_enabled(self):
        from redis.asyncio.connection import SSLConnection

        cfg = ConnectionConfig(
            host="h",
            port=1,
            ssl=True,
            ssl_cert_reqs="required",
            ssl_ca_certs="/etc/ca.pem",
            ssl_certfile="/etc/client.pem",
            ssl_keyfile="/etc/client.key",
            ssl_check_hostname=True,
        )
        conn = FalkorDBConnection(cfg)
        with patch("redis.asyncio.BlockingConnectionPool") as mock_pool, \
             patch("falkordb.asyncio.FalkorDB") as mock_falkor:
            mock_falkor.return_value.select_graph = MagicMock()
            conn._ensure_client()
            kwargs = mock_pool.call_args.kwargs
            assert kwargs["connection_class"] is SSLConnection
            assert kwargs["ssl_cert_reqs"] == "required"
            assert kwargs["ssl_ca_certs"] == "/etc/ca.pem"
            assert kwargs["ssl_certfile"] == "/etc/client.pem"
            assert kwargs["ssl_keyfile"] == "/etc/client.key"
            assert kwargs["ssl_check_hostname"] is True
