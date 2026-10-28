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

    @pytest.mark.parametrize(
        ("query_timeout_ms", "socket_timeout"),
        [(10_000, 15.0), (2_000, 7.0), (None, None)],
    )
    def test_pool_waits_past_the_server_query_timeout(self, query_timeout_ms, socket_timeout):
        """redis-py's 5 s default fired before the server's 10 s query
        timeout, so a 7 s read failed as a socket error, three times over."""
        conn = FalkorDBConnection(
            ConnectionConfig(host="h", port=1, query_timeout_ms=query_timeout_ms)
        )
        with (
            patch("redis.asyncio.BlockingConnectionPool") as mock_pool,
            patch("falkordb.asyncio.FalkorDB") as mock_falkor,
        ):
            mock_falkor.return_value.select_graph = MagicMock()
            conn._ensure_client()
            kwargs = mock_pool.call_args.kwargs
        assert kwargs["socket_timeout"] == socket_timeout
        assert kwargs["socket_connect_timeout"] == 5.0

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
        "message",
        [
            # FalkorDB 6.0.x (Rust engine)
            (
                "Received 2 arguments to procedure 'db.idx.fulltext.createNodeIndex', "
                + "expected at most 1"
            ),
            "Cannot merge a map with a non-map value",
            # FalkorDB 4.x (C engine)
            "Procedure `db.nope` is not registered",
            "Unable to resolve filtered alias '(n)-[:R]->()'",
            "Attempted to access undefined attribute",
            # Both engines
            "'zz' not defined",
            "Division by zero",
            "Unable to drop index on :Nope(x): no such index.",
            "Cannot merge node using null property value for key 'id'",
            # A rejection that quotes a word a temporary reply also uses
            "'timeout' not defined",
            "'busy' not defined",
            # The query used its whole time budget: a retry costs as much again
            "Query timed out",
        ],
    )
    async def test_server_rejection_is_raised_once_and_spares_the_breaker(self, message):
        """A query the server rejected fails the same way on every attempt.
        Retrying it, and counting it against the breaker, turned one bad
        statement into an outage for every caller."""
        from redis.exceptions import ResponseError

        conn = FalkorDBConnection(ConnectionConfig(retry_count=3, retry_delay=0.001))
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=ResponseError(message))
        conn._driver = MagicMock()
        conn._graph = mock_graph
        _skip_version_check(conn)

        for _ in range(conn._breaker.failure_threshold + 1):
            with pytest.raises(DatabaseError) as excinfo:
                await conn.query("RETURN 1")
            assert not isinstance(excinfo.value, DatabaseUnavailableError)

        assert mock_graph.query.await_count == conn._breaker.failure_threshold + 1
        assert await conn._breaker.allow_request()

    async def test_a_rejected_half_open_probe_closes_the_breaker(self):
        """After an outage the breaker lets one probe through. A rejection
        proves the server answered; reporting nothing left the breaker
        HALF_OPEN and every later query was refused for good."""
        from redis.exceptions import ConnectionError as RedisConnectionError
        from redis.exceptions import ResponseError

        from graphrag_sdk.core.circuit_breaker import CircuitState

        conn = FalkorDBConnection(ConnectionConfig(retry_count=1, retry_delay=0.001))
        conn._breaker.recovery_timeout = 0.0
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=RedisConnectionError("down"))
        conn._driver = MagicMock()
        conn._graph = mock_graph
        _skip_version_check(conn)
        for _ in range(conn._breaker.failure_threshold):
            with pytest.raises(DatabaseError):
                await conn.query("RETURN 1")
        assert conn._breaker._state == CircuitState.OPEN

        mock_graph.query = AsyncMock(side_effect=ResponseError("Division by zero"))
        with pytest.raises(DatabaseError):
            await conn.query("RETURN 1 / 0")  # the half-open probe

        assert conn._breaker.state == CircuitState.CLOSED
        mock_graph.query = AsyncMock(return_value="ok")
        assert await conn.query("RETURN 1") == "ok"

    @pytest.mark.parametrize(
        ("error_type", "message"),
        [
            ("ResponseError", "Max pending queries exceeded"),
            ("ResponseError", "BUSY Redis is busy running a script."),
            ("ResponseError", "NOREPLICAS Not enough good replicas to write."),
            ("ResponseError", "MISCONF Redis is configured to save RDB snapshots, but ..."),
            ("BusyLoadingError", "Redis is loading the dataset in memory"),  # LOADING
            ("TryAgainError", "TRYAGAIN"),
            ("ReadOnlyError", "READONLY"),
            ("OutOfMemoryError", "OOM"),
        ],
    )
    async def test_temporary_server_replies_are_still_retried(self, error_type, message):
        from redis import exceptions as rx

        exc = getattr(rx, error_type)(message)
        conn = FalkorDBConnection(ConnectionConfig(retry_count=3, retry_delay=0.001))
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=exc)
        conn._driver = MagicMock()
        conn._graph = mock_graph
        _skip_version_check(conn)

        with pytest.raises(DatabaseUnavailableError):
            await conn.query("RETURN 1")

        assert mock_graph.query.await_count == 3

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

    async def test_ping_reports_a_hung_server_within_the_probe_bound(self):
        """PING went through the pool's socket timeout, sized for slow queries
        (15 s by default, unbounded with query_timeout_ms=None), so a liveness
        check on a hung server took that long or never answered."""
        import asyncio

        conn = FalkorDBConnection(ConnectionConfig(host="h", port=1, query_timeout_ms=None))
        conn._PROBE_TIMEOUT_S = 0.05
        conn._driver = MagicMock()

        async def hang():
            await asyncio.sleep(60)

        with patch("redis.asyncio.Redis") as mock_redis:
            mock_redis.return_value.ping = hang
            assert await asyncio.wait_for(conn.ping(), timeout=2) is False

    @pytest.mark.parametrize("fails", [False, True])
    def test_driver_probes_with_the_short_bound_then_restores_the_query_one(self, fails):
        """The driver's constructor probes the server with a blocking call that
        copies the pool's socket timeout: a hung server stalled the event loop
        for 11 x 15 s, or for good with query_timeout_ms=None."""
        conn = FalkorDBConnection(ConnectionConfig(host="h", port=1))
        seen: list[float | None] = []

        def construct(connection_pool):
            seen.append(connection_pool.connection_kwargs["socket_timeout"])
            if fails:
                raise ConnectionError("hung")
            return MagicMock()

        with patch("falkordb.asyncio.FalkorDB", side_effect=construct):
            if fails:
                with pytest.raises(DatabaseUnavailableError):
                    conn._ensure_client()
            else:
                conn._ensure_client()

        assert seen == [FalkorDBConnection._PROBE_TIMEOUT_S]
        assert conn._pool.connection_kwargs["socket_timeout"] == 15.0


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
        ("error_type", "reply"),
        [
            ("ClusterDownError", "The cluster is down"),
            ("TryAgainError", "Multiple keys request during rehashing of slot"),
            ("OutOfMemoryError", "command not allowed when used memory > 'maxmemory'"),
        ],
        ids=["clusterdown", "tryagain", "oom"],
    )
    async def test_an_error_that_passes_is_not_read_as_a_refusal(self, error_type, reply):
        """Read as "will not say", one such reply left the connection checked
        for good, with a warning; it is retried like a query instead. (redis-py
        raises each as its own ``ResponseError`` subclass.)"""
        from redis import exceptions as rx

        exc = getattr(rx, error_type)(reply)
        conn = self._conn(side_effect=[exc, [["name", "graph", "ver", 41803]]])
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
