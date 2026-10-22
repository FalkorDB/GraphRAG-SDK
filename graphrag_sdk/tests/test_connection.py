"""Tests for core/connection.py — async-only FalkorDB connection."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import ResponseError

from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.exceptions import (
    DatabaseError,
    DatabaseUnavailableError,
    FalkorDBVersionError,
)


def _skip_version_check(conn: FalkorDBConnection) -> None:
    """Treat the server version as checked: ``TestServerVersion`` covers it."""
    conn._version_checked = True


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
        _skip_version_check(conn)
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
        _skip_version_check(conn)

        result = await conn.query("MATCH (n) RETURN n")
        assert result == "success"
        assert mock_graph.query.call_count == 3

    async def test_query_exhausts_retries(self, caplog):
        conn = FalkorDBConnection(ConnectionConfig(retry_count=2, retry_delay=0.0))

        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=Exception("always fails"))
        conn._graph = mock_graph
        conn._driver = MagicMock()
        _skip_version_check(conn)

        with caplog.at_level("ERROR", logger="graphrag_sdk.core.connection"):
            with pytest.raises(Exception, match="always fails"):
                await conn.query("MATCH (n) RETURN n")
        assert "FalkorDB query failed after 2 attempts" in caplog.text
        assert mock_graph.query.call_count == 2

    async def test_query_non_transient_failure_is_typed_and_logged(self, caplog):
        conn = FalkorDBConnection(ConnectionConfig(retry_count=3, retry_delay=0.0))

        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=Exception("syntax error at offset 3"))
        conn._graph = mock_graph
        conn._driver = MagicMock()
        _skip_version_check(conn)

        with caplog.at_level("ERROR", logger="graphrag_sdk.core.connection"):
            with pytest.raises(Exception, match="syntax error"):
                await conn.query("MATCH (n RETURN n")
        assert "Non-transient FalkorDB query failure" in caplog.text
        assert mock_graph.query.call_count == 1

    @pytest.mark.parametrize(
        "message",
        ["Attribute 'embedding' is already indexed", "Index already exists"],
    )
    async def test_expected_already_indexed_is_logged_at_debug(self, caplog, message):
        """An idempotent CREATE INDEX on an indexed graph is not a failure.

        finalize() re-creates every index each call; FalkorDB answers
        'already indexed' and VectorStore treats that as success. Logging it
        at ERROR printed three spurious failures per finalize(). The caller
        declares the reply expected; the connection logs it at DEBUG exactly.
        """
        conn = FalkorDBConnection(ConnectionConfig(retry_count=3, retry_delay=0.0))

        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=Exception(message))
        conn._graph = mock_graph
        conn._driver = MagicMock()
        _skip_version_check(conn)

        with caplog.at_level("DEBUG", logger="graphrag_sdk.core.connection"):
            with pytest.raises(Exception, match="already"):
                await conn.query(
                    "CREATE VECTOR INDEX FOR (n:Chunk) ON (n.embedding)",
                    expected_errors=("already indexed", "already exists"),
                )
        failures = [
            r
            for r in caplog.records
            if r.getMessage().startswith("Non-transient FalkorDB query failure:")
        ]
        assert [r.levelname for r in failures] == ["DEBUG"]
        assert message in failures[0].getMessage()
        assert mock_graph.query.call_count == 1

    async def test_already_exists_stays_an_error_unless_the_caller_expects_it(self, caplog):
        """``query()`` cannot tell an idempotent CREATE INDEX from any other
        statement whose error says "already exists", so it never downgrades
        on its own -- that would hide real failures from ERROR-based monitoring."""
        conn = FalkorDBConnection(ConnectionConfig(retry_count=3, retry_delay=0.0))

        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=Exception("Constraint already exists"))
        conn._graph = mock_graph
        conn._driver = MagicMock()
        _skip_version_check(conn)

        with caplog.at_level("DEBUG", logger="graphrag_sdk.core.connection"):
            with pytest.raises(Exception, match="already exists"):
                await conn.query("CREATE CONSTRAINT ...")
        failures = [
            r
            for r in caplog.records
            if r.getMessage().startswith("Non-transient FalkorDB query failure:")
        ]
        assert [r.levelname for r in failures] == ["ERROR"]

    async def test_expected_errors_do_not_cover_other_failures(self, caplog):
        conn = FalkorDBConnection(ConnectionConfig(retry_count=3, retry_delay=0.0))

        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=Exception("syntax error at offset 3"))
        conn._graph = mock_graph
        conn._driver = MagicMock()
        _skip_version_check(conn)

        with caplog.at_level("DEBUG", logger="graphrag_sdk.core.connection"):
            with pytest.raises(Exception, match="syntax error"):
                await conn.query("CREATE INDEX ...", expected_errors=("already indexed",))
        failures = [
            r
            for r in caplog.records
            if r.getMessage().startswith("Non-transient FalkorDB query failure:")
        ]
        assert [r.levelname for r in failures] == ["ERROR"]

    async def test_query_with_params(self):
        conn = FalkorDBConnection(ConnectionConfig(retry_count=1))

        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(return_value="ok")
        conn._graph = mock_graph
        conn._driver = MagicMock()
        _skip_version_check(conn)

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
        with (
            patch("redis.asyncio.BlockingConnectionPool") as mock_pool,
            patch("falkordb.asyncio.FalkorDB") as mock_falkor,
        ):
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
        with (
            patch("redis.asyncio.BlockingConnectionPool") as mock_pool,
            patch("falkordb.asyncio.FalkorDB") as mock_falkor,
        ):
            mock_falkor.return_value.select_graph = MagicMock()
            conn._ensure_client()
            kwargs = mock_pool.call_args.kwargs
            assert kwargs["connection_class"] is SSLConnection
            assert kwargs["ssl_cert_reqs"] == "required"
            assert kwargs["ssl_ca_certs"] == "/etc/ca.pem"
            assert kwargs["ssl_certfile"] == "/etc/client.pem"
            assert kwargs["ssl_keyfile"] == "/etc/client.key"
            assert kwargs["ssl_check_hostname"] is True


class TestDriverErrorsBecomeDatabaseError:
    """Driver-level failures reach callers as ``DatabaseError``.

    Callers decide what to do about an unreachable database — the chunk
    extraction cache, for one, stops instead of paying for an LLM run whose
    result it could not store. That decision needs a database failure to be
    recognisable, so it cannot be left as a ``redis`` exception that only
    reads as a generic ``Exception``.
    """

    def test_unreachable_server_on_connect(self):
        """The driver contacts the server while being constructed, so an
        unreachable database surfaces here rather than at query time."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        conn = FalkorDBConnection(ConnectionConfig(host="h", port=1))
        with (
            patch("redis.asyncio.BlockingConnectionPool"),
            patch("falkordb.asyncio.FalkorDB") as mock_falkor,
        ):
            mock_falkor.side_effect = RedisConnectionError("Connection refused")
            with pytest.raises(DatabaseError, match="Could not connect to FalkorDB"):
                conn._ensure_client()

    def test_failed_connect_leaves_no_half_built_client(self):
        """A later attempt must reconnect rather than reuse the wreckage."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        conn = FalkorDBConnection(ConnectionConfig(host="h", port=1))
        with (
            patch("redis.asyncio.BlockingConnectionPool"),
            patch("falkordb.asyncio.FalkorDB") as mock_falkor,
        ):
            mock_falkor.side_effect = RedisConnectionError("Connection refused")
            with pytest.raises(DatabaseError):
                conn._ensure_client()

        assert conn._driver is None
        assert conn._graph is None

    def test_failed_connect_keeps_the_pool_closeable(self):
        """Only close() can release a pool, so a failed attempt must not
        strand one — and a retry must not overwrite it with a second."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        conn = FalkorDBConnection(ConnectionConfig(host="h", port=1))
        with (
            patch("redis.asyncio.BlockingConnectionPool") as mock_pool,
            patch("falkordb.asyncio.FalkorDB") as mock_falkor,
        ):
            mock_falkor.side_effect = RedisConnectionError("Connection refused")
            for _ in range(3):
                with pytest.raises(DatabaseError):
                    conn._ensure_client()

            assert mock_pool.call_count == 1, "each retry allocated another pool"
            assert conn._pool is mock_pool.return_value

    def test_missing_falkordb_package_still_raises_import_error(self):
        """A packaging problem is not a database problem."""
        conn = FalkorDBConnection()
        with patch.dict("sys.modules", {"falkordb.asyncio": None, "falkordb": None}):
            with pytest.raises(ImportError):
                conn._ensure_client()

    async def test_query_wraps_error_once_retries_are_spent(self):
        from redis.exceptions import ConnectionError as RedisConnectionError

        conn = FalkorDBConnection(ConnectionConfig(retry_count=2, retry_delay=0.001))
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=RedisConnectionError("Connection reset"))
        conn._driver = MagicMock()
        _skip_version_check(conn)
        conn._graph = mock_graph

        with pytest.raises(DatabaseError) as excinfo:
            await conn.query("MATCH (n) RETURN n")

        assert mock_graph.query.await_count == 2
        assert isinstance(excinfo.value.__cause__, RedisConnectionError)

    async def test_query_wraps_permanent_error_without_retrying(self):
        conn = FalkorDBConnection(ConnectionConfig(retry_count=3, retry_delay=0.001))
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=ResponseError("Attribute already indexed"))
        conn._driver = MagicMock()
        _skip_version_check(conn)
        conn._graph = mock_graph

        with pytest.raises(DatabaseError):
            await conn.query("CREATE INDEX ...")

        assert mock_graph.query.await_count == 1

    async def test_a_parameter_falkordb_cannot_parse_is_not_retried(self):
        """FalkorDB refuses NaN and infinity as parameters. Retried as an
        outage, the same query failed three times and opened the circuit
        breaker for every query after it."""
        conn = FalkorDBConnection(ConnectionConfig(retry_count=3, retry_delay=0.001))
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(
            side_effect=ResponseError("Failed to parse query parameter 'v' value")
        )
        conn._driver = MagicMock()
        _skip_version_check(conn)
        conn._graph = mock_graph

        with pytest.raises(DatabaseError) as excinfo:
            await conn.query("CREATE (:X {v: $v})", {"v": float("nan")})
        assert not isinstance(excinfo.value, DatabaseUnavailableError)
        assert mock_graph.query.await_count == 1
        assert await conn._breaker.allow_request()

    @pytest.mark.parametrize(
        "reply",
        [
            "Missing parameters",
            "Cannot merge node using null property value",
            "Property values can only be of primitive types or arrays of primitive types",
            "Division by zero",
            "Attempted to access undefined attribute",
        ],
    )
    async def test_an_error_the_server_answered_is_not_retried(self, reply):
        """The server answered: the query itself is wrong and will fail the
        same way again. Retried as an outage, two of them opened the circuit
        breaker and refused every query after them."""
        conn = FalkorDBConnection(ConnectionConfig(retry_count=3, retry_delay=0.001))
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=ResponseError(reply))
        conn._driver = MagicMock()
        _skip_version_check(conn)
        conn._graph = mock_graph

        with pytest.raises(DatabaseError) as excinfo:
            await conn.query("RETURN $x")
        assert not isinstance(excinfo.value, DatabaseUnavailableError)
        assert mock_graph.query.await_count == 1
        assert await conn._breaker.allow_request()

    @pytest.mark.parametrize(
        "reply",
        [
            "LOADING FalkorDB is loading the dataset",
            "BUSY",
            "Query timed out",
            "Max pending queries exceeded",
            "OOM command not allowed when used memory > 'maxmemory'.",
            "CLUSTERDOWN The cluster is down",
            "NOREPLICAS Not enough good replicas to write.",
        ],
    )
    async def test_a_reply_from_a_busy_server_is_retried(self, reply):
        conn = FalkorDBConnection(ConnectionConfig(retry_count=3, retry_delay=0.001))
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(
            side_effect=[ResponseError(reply), MagicMock(result_set=[[1]])]
        )
        conn._driver = MagicMock()
        _skip_version_check(conn)
        conn._graph = mock_graph

        result = await conn.query("RETURN 1")
        assert result.result_set == [[1]] and mock_graph.query.await_count == 2

    async def test_wrapped_message_keeps_the_driver_text(self):
        """Index creation tolerates 'already indexed' by reading the message,
        so wrapping must not hide what the driver said."""
        conn = FalkorDBConnection(ConnectionConfig(retry_count=1, retry_delay=0.001))
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=ResponseError("Attribute already indexed"))
        conn._driver = MagicMock()
        _skip_version_check(conn)
        conn._graph = mock_graph

        with pytest.raises(DatabaseError, match="already indexed"):
            await conn.query("CREATE INDEX ...")

    async def test_open_circuit_breaker(self):
        conn = FalkorDBConnection()
        conn._driver = MagicMock()
        _skip_version_check(conn)
        conn._graph = MagicMock()
        conn._breaker.allow_request = AsyncMock(return_value=False)

        with pytest.raises(DatabaseError, match="Circuit breaker is open"):
            await conn.query("MATCH (n) RETURN n")

    async def test_ping_reports_unreachable_instead_of_raising(self):
        """Liveness checks read the bool; an unreachable server is the
        case ping() exists to report, not one it should raise on."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        conn = FalkorDBConnection(ConnectionConfig(host="h", port=1))
        with (
            patch("redis.asyncio.BlockingConnectionPool"),
            patch("falkordb.asyncio.FalkorDB") as mock_falkor,
        ):
            mock_falkor.side_effect = RedisConnectionError("Connection refused")
            assert await conn.ping() is False


class TestServerVersion:
    """A connection refuses a FalkorDB server older than 4.20, once, before
    its first query: nothing is sent to it."""

    @staticmethod
    def _conn(**mock_kw):
        conn = FalkorDBConnection(ConnectionConfig(retry_count=1, retry_delay=0.0))
        conn._graph = MagicMock()
        conn._graph.query = AsyncMock(return_value="ok")
        conn._driver = MagicMock()
        conn._driver.execute_command = AsyncMock(**mock_kw)
        return conn

    async def test_an_older_server_is_refused_and_nothing_is_sent(self):
        conn = self._conn(return_value=[["name", "graph", "ver", 41802]])
        for _ in range(2):  # every query, not only the first
            with pytest.raises(FalkorDBVersionError, match="is 4.18.2; GraphRAG SDK needs"):
                await conn.query("CREATE (:X)")
        conn._graph.query.assert_not_awaited()
        assert issubclass(FalkorDBVersionError, DatabaseError)

    async def test_a_current_server_is_asked_once(self):
        conn = self._conn(return_value=[{"name": "graph", "ver": 42004}])
        assert await conn.server_version() == (4, 20, 4)
        conn._driver.execute_command.reset_mock()
        await asyncio.gather(*(conn.query("RETURN 1") for _ in range(50)))
        assert conn._driver.execute_command.await_count == 1
        assert conn._graph.query.await_count == 50

    async def test_a_server_that_does_not_say_is_let_through_with_one_warning(self, caplog):
        conn = self._conn(side_effect=ResponseError("NOPERM this user has no permissions"))
        with caplog.at_level("WARNING", logger="graphrag_sdk.core.connection"):
            assert await conn.query("RETURN 1") == "ok"
            assert await conn.query("RETURN 1") == "ok"
        assert caplog.text.count("Could not read the FalkorDB server version") == 1

    @pytest.mark.parametrize(
        "reply",
        [
            "CLUSTERDOWN The cluster is down",
            "TRYAGAIN Multiple keys request during rehashing of slot",
            "OOM command not allowed when used memory > 'maxmemory'",
        ],
        ids=["clusterdown", "tryagain", "oom"],
    )
    async def test_an_error_that_passes_is_not_read_as_a_refusal(self, reply):
        """Read as "will not say", one such reply left the connection checked
        for good, with a warning; it is retried like a query instead."""
        conn = self._conn(side_effect=[ResponseError(reply), [["name", "graph", "ver", 41803]]])
        conn.config.retry_count = 2
        with pytest.raises(FalkorDBVersionError, match="is 4.18.3"):
            await conn.query("CREATE (:X)")
        conn._graph.query.assert_not_awaited()

    async def test_a_closed_connection_asks_again(self):
        """``close()`` drops the client and the next query reconnects, maybe to
        a server replaced meanwhile at the same address."""
        conn = self._conn(return_value=[{"name": "graph", "ver": 42004}])
        assert await conn.query("RETURN 1") == "ok"
        await conn.close()
        reopened = self._conn(return_value=[["name", "graph", "ver", 41802]])
        conn._graph, conn._driver = reopened._graph, reopened._driver  # the reconnect
        with pytest.raises(FalkorDBVersionError, match="is 4.18.2"):
            await conn.query("CREATE (:X)")
        conn._graph.query.assert_not_awaited()

    async def test_a_server_that_cannot_be_asked_fails_the_query_and_is_asked_again(self):
        """No query runs unchecked: a failed version question fails the query,
        and the next query asks again."""
        conn = self._conn(side_effect=[ConnectionError("down"), [{"name": "graph", "ver": 42004}]])
        with pytest.raises(DatabaseUnavailableError, match="version check failed"):
            await conn.query("CREATE (:X)")
        conn._graph.query.assert_not_awaited()
        assert await conn.query("RETURN 1") == "ok"
        assert conn._driver.execute_command.await_count == 2

    async def test_the_version_question_is_retried_like_a_query(self):
        """A stale pooled connection — after a server restart, or in a fresh
        ``asyncio.run`` — fails the first ask; the retry reaches the server,
        and an old one is still refused."""
        conn = self._conn(side_effect=[ConnectionError("stale"), [["name", "graph", "ver", 41803]]])
        conn.config.retry_count = 2
        with pytest.raises(FalkorDBVersionError, match="is 4.18.3"):
            await conn.query("CREATE (:X)")
        conn._graph.query.assert_not_awaited()

    def test_the_check_holds_across_event_loops(self):
        """The ``*_sync`` wrappers run every call in a fresh ``asyncio.run``; a
        lock bound to the first loop raised ``RuntimeError`` in the next, in
        place of the version error."""
        conn = self._conn(return_value=[["name", "graph", "ver", 41802]])

        async def three_at_once():
            return await asyncio.gather(
                *(conn.query("CREATE (:X)") for _ in range(3)), return_exceptions=True
            )

        for _ in range(2):
            assert all(isinstance(r, FalkorDBVersionError) for r in asyncio.run(three_at_once()))
        conn._graph.query.assert_not_awaited()


async def test_the_ontology_graph_is_not_written_on_a_server_too_old():
    """The ontology graph is queried through the driver, not conn.query: on a
    server older than 4.20 it was created before the first refusal."""
    from graphrag_sdk.core.exceptions import FalkorDBVersionError
    from graphrag_sdk.storage.ontology_store import OntologyStore

    conn = FalkorDBConnection(ConnectionConfig(retry_count=1, retry_delay=0.001))
    conn._driver = MagicMock()
    graph = MagicMock()
    graph.query = AsyncMock()
    conn._driver.select_graph = MagicMock(return_value=graph)

    async def old():
        return (4, 18, 3)

    conn.server_version = old
    store = OntologyStore(conn, "g")
    with pytest.raises(FalkorDBVersionError):
        await store._query("CREATE (:X)")
    graph.query.assert_not_called()


async def test_a_server_too_old_is_not_loaded_as_an_empty_ontology():
    from graphrag_sdk.core.exceptions import FalkorDBVersionError
    from graphrag_sdk.storage.ontology_store import OntologyStore

    conn = FalkorDBConnection(ConnectionConfig(retry_count=1, retry_delay=0.001))
    conn._driver = MagicMock()
    conn._driver.select_graph = MagicMock(return_value=MagicMock(query=AsyncMock()))

    async def old():
        return (4, 18, 3)

    conn.server_version = old
    with pytest.raises(FalkorDBVersionError):
        await OntologyStore(conn, "g").load()
