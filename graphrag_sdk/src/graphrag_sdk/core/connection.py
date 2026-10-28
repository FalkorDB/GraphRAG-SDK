# GraphRAG SDK — Core: FalkorDB Connection
# Async-only FalkorDB client using native ``falkordb.asyncio``.
# Origin: User design — production resilience.

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from graphrag_sdk.core.exceptions import (
    DatabaseError,
    DatabaseUnavailableError,
    FalkorDBVersionError,
)

logger = logging.getLogger(__name__)

#: The oldest FalkorDB server the SDK works with. Before 4.20 a write to a node
#: holding a vector dropped vector-index entries (fixed by FalkorDB PR #2197):
#: measured on 4.18, ``queryNodes`` returned 2-3 of 6 neighbours after one such
#: write, so search and dedup miss entities. Ingest, update and dedup all write
#: to such nodes.
MIN_SERVER_VERSION = (4, 20, 0)

#: Replies that mean a server will not answer ``MODULE LIST``: an ACL without
#: the command, or a managed server that disabled or renamed it (which answers
#: as an unknown command). ``CLUSTERDOWN``, ``TRYAGAIN``, ``OOM`` and the like
#: are error replies too, and they pass.
_MODULE_LIST_REFUSALS = ("noperm", "no permission", "unknown command")


@dataclass
class ConnectionConfig:
    """FalkorDB connection configuration.

    TLS-related fields mirror ``redis-py``'s naming. They are no-ops unless
    ``ssl=True``. When TLS is enabled, certificates are verified by default
    (``ssl_cert_reqs="required"``) — relax explicitly only if you understand
    the implications.
    """

    host: str = "localhost"
    port: int = 6379
    username: str | None = None
    password: str | None = field(default=None, repr=False)
    graph_name: str = "knowledge_graph"
    max_connections: int = 16
    retry_count: int = 3
    retry_delay: float = 1.0
    pool_timeout: float = 30.0
    query_timeout_ms: int | None = 10_000
    # TLS / SSL
    ssl: bool = False
    ssl_cert_reqs: str = "required"
    ssl_ca_certs: str | None = None
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None
    ssl_check_hostname: bool = True

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> ConnectionConfig:
        """Create a ConnectionConfig from a ``redis://`` or ``rediss://`` URL.

        Supports ``redis://[user:pass@]host[:port][/db]`` for plaintext and
        ``rediss://[user:pass@]host[:port][/db]`` for TLS. The scheme drives
        the ``ssl`` field; explicit kwargs always win.

        Extra keyword arguments override parsed values.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("redis", "rediss", ""):
            raise ValueError(
                f"Unsupported URL scheme: {parsed.scheme!r} "
                "(use 'redis://' for plaintext or 'rediss://' for TLS)"
            )
        if parsed.scheme == "rediss":
            kwargs.setdefault("ssl", True)
        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            username=parsed.username or None,
            password=parsed.password or None,
            **kwargs,
        )


def _changes_topology(result: Any) -> bool:
    """Whether a query result reports nodes or relationships created or deleted."""
    for stat in (
        "nodes_created",
        "nodes_deleted",
        "relationships_created",
        "relationships_deleted",
    ):
        try:
            if int(getattr(result, stat, 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


class FalkorDBConnection:
    """Async-only FalkorDB connection using the native async client.

    Wraps ``falkordb.asyncio.FalkorDB`` with:
    - ``redis.asyncio.BlockingConnectionPool`` for bounded pooling
    - Automatic retries with exponential backoff on transient failures
    - Lazy initialisation — no I/O until the first query
    - A server older than ``MIN_SERVER_VERSION`` refused at the first query

    Example::

        conn = FalkorDBConnection(ConnectionConfig(host="localhost"))
        result = await conn.query("RETURN 1")
        await conn.close()
    """

    # Bound for liveness checks (PING, the driver's startup probe), which need
    # no query budget: redis-py's own default, which the SDK used to inherit.
    _PROBE_TIMEOUT_S = 5.0

    def __init__(self, config: ConnectionConfig | None = None) -> None:
        from graphrag_sdk.core.circuit_breaker import CircuitBreaker

        self.config = config or ConnectionConfig()
        self._pool: Any | None = None
        self._driver: Any | None = None
        self._graph: Any | None = None
        self._breaker = CircuitBreaker()
        #: Bumped by every query through this connection that creates or
        #: deletes nodes or relationships (and by ``delete_graph``), so
        #: caches of graph-shape results (PageRank) know when they are stale.
        self.topology_version = 0
        # Set once the server's version passed, or was not reported (see
        # _check_server_version); cleared by close(), so a reconnect asks again.
        self._version_checked = False
        self._version_check_lock: asyncio.Lock | None = None
        self._version_check_loop: asyncio.AbstractEventLoop | None = None

    def _ensure_client(self) -> None:
        """Lazy-init the async FalkorDB driver and graph handle."""
        if self._driver is not None:
            return

        try:
            from falkordb.asyncio import FalkorDB
            from redis.asyncio import BlockingConnectionPool
        except ImportError:
            raise ImportError("falkordb package is required. Install with: pip install falkordb")

        pool_kwargs: dict[str, Any] = {
            "host": self.config.host,
            "port": self.config.port,
            "username": self.config.username,
            "password": self.config.password,
            "max_connections": self.config.max_connections,
            "timeout": self.config.pool_timeout,
            "decode_responses": True,
            # redis-py stops waiting for a reply after 5 s unless told
            # otherwise, before the server's own query timeout (10 s by
            # default) can answer. A slower query then failed as a socket
            # timeout: retried, counted against the breaker, and left running
            # on the server. Wait for the server's answer plus a margin. For a
            # read, a reply later than that means the server missed its own
            # deadline. FalkorDB applies no deadline to a write, so a write
            # slower than this is still cut here and re-sent by the retry loop.
            "socket_timeout": (
                self.config.query_timeout_ms / 1000 + 5.0 if self.config.query_timeout_ms else None
            ),
            # Otherwise redis-py connects with the socket timeout above, which
            # can be None: keep connecting to a dead host bounded.
            "socket_connect_timeout": 5.0,
        }
        if self.config.ssl:
            from redis.asyncio.connection import SSLConnection

            pool_kwargs["connection_class"] = SSLConnection
            pool_kwargs["ssl_cert_reqs"] = self.config.ssl_cert_reqs
            pool_kwargs["ssl_ca_certs"] = self.config.ssl_ca_certs
            pool_kwargs["ssl_certfile"] = self.config.ssl_certfile
            pool_kwargs["ssl_keyfile"] = self.config.ssl_keyfile
            pool_kwargs["ssl_check_hostname"] = self.config.ssl_check_hostname

        self._pool = self._pool or BlockingConnectionPool(**pool_kwargs)
        # The driver's constructor probes the server with a blocking call that
        # copies the pool's socket timeout and stalls the event loop while it
        # waits. Probe with the short bound; queries get the long one. Safe to
        # swap: the constructor is synchronous, so no pool connection can be
        # created from the lowered value.
        connection_kwargs = self._pool.connection_kwargs
        query_socket_timeout = connection_kwargs.get("socket_timeout")
        connection_kwargs["socket_timeout"] = self._PROBE_TIMEOUT_S
        try:
            self._driver = FalkorDB(connection_pool=self._pool)
            self._graph = self._driver.select_graph(self.config.graph_name)
        except Exception as exc:
            # The driver probes the server during construction, so an
            # unreachable database surfaces here rather than at query time.
            # The pool is kept: it connects lazily and is reusable, and
            # only close() can release it. Dropping it here would strand
            # a pool per failed attempt with no way to aclose() it.
            self._driver = None
            self._graph = None
            raise DatabaseUnavailableError(
                f"Could not connect to FalkorDB at {self.config.host}:{self.config.port}: {exc}"
            ) from exc
        finally:
            connection_kwargs["socket_timeout"] = query_socket_timeout

        logger.info(
            "Connected to FalkorDB (async) at %s:%s (tls=%s)",
            self.config.host,
            self.config.port,
            self.config.ssl,
        )

    @property
    def graph(self) -> Any:
        """Return the ``AsyncGraph`` handle (lazy-created)."""
        self._ensure_client()
        return self._graph

    # ── Query ────────────────────────────────────────────────────

    async def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: int | None = None,
        expected_errors: tuple[str, ...] = (),
        read_only: bool = False,
    ) -> Any:
        """Execute a Cypher query with retry logic.

        Args:
            cypher: The Cypher query string.
            params: Optional query parameters.
            timeout: Optional per-query timeout (ms) forwarded to FalkorDB.
                The client stops waiting at ``query_timeout_ms`` + 5 s whatever
                this says; a reply later than that is a socket timeout, which
                is retried (re-sending the query while the earlier attempt may
                still run) and ends in ``DatabaseUnavailableError``.
            read_only: Run the query with ``GRAPH.RO_QUERY``, which FalkorDB
                rejects if the query would write. Use it for any query whose
                text came from a model or a user.
            expected_errors: Lower-case substrings identifying non-transient
                failures the caller anticipates and handles itself (e.g.
                ``"already indexed"`` for an idempotent ``CREATE INDEX``). Such
                a failure still raises ``DatabaseError`` but is logged at DEBUG
                rather than ERROR. Only the caller knows which failures are
                expected; ``query()`` cannot tell an idempotent index creation
                from any other statement whose error happens to say "already
                exists", so it never downgrades on its own.

        Returns:
            ``QueryResult`` from the async FalkorDB driver.

        Raises:
            DatabaseUnavailableError: The circuit breaker is open, or a
                transient failure (connection loss, full query queue, ``BUSY``,
                failover, ...) outlasted ``retry_count`` attempts.
            DatabaseError: The server rejected the query, or answered that it
                ran past its timeout. Raised on the first attempt, not retried. Driver-level
                exceptions are wrapped and kept as ``__cause__``.
            FalkorDBVersionError: The server is older than FalkorDB 4.20 (a
                ``DatabaseError``). Checked once per connection, before its
                first query.
        """
        self._ensure_client()
        assert self._graph is not None  # for type-checkers

        if not await self._breaker.allow_request():
            raise DatabaseUnavailableError(
                "Circuit breaker is open — FalkorDB connection is unhealthy. "
                "Requests will resume after recovery timeout."
            )
        await self._check_server_version()

        effective_timeout = timeout if timeout is not None else self.config.query_timeout_ms
        graph = self._graph
        run = graph.ro_query if read_only else graph.query
        return await self._with_retries(
            lambda: run(cypher, params=params, timeout=effective_timeout),
            "query",
            expected_errors=expected_errors,
            read_only=read_only,
        )

    async def _with_retries(
        self,
        call: Callable[[], Awaitable[Any]],
        action: str,
        *,
        expected_errors: tuple[str, ...] = (),
        read_only: bool = False,
    ) -> Any:
        """Await ``call()``, retrying transient failures with exponential
        backoff and jitter, and feeding the circuit breaker. A non-transient
        failure raises ``DatabaseError`` at once; retries spent raise
        ``DatabaseUnavailableError``. ``action`` names the call in logs and
        errors; ``expected_errors`` is as for :meth:`query`."""
        last_exc: Exception | None = None
        for attempt in range(self.config.retry_count):
            try:
                result = await call()
                await self._breaker.record_success()
                if not read_only and _changes_topology(result):
                    self.topology_version = getattr(self, "topology_version", 0) + 1
                return result
            except Exception as exc:
                last_exc = exc
                # A read-only query's text came from a model or a user. When the
                # server answers it with an error (unknown variable, bad type,
                # a write under RO_QUERY, its own timeout), the query is wrong,
                # not the database: fail once, without retrying and without
                # counting it against the circuit breaker, so bad generated
                # Cypher can never take the connection down for everyone.
                if read_only and self._is_server_rejection(exc):
                    logger.debug("Read-only query rejected by FalkorDB: %s", exc)
                    # The server answered, so it is reachable: close the
                    # breaker. Without this a rejected query that happens to
                    # be the half-open probe leaves the breaker HALF_OPEN for
                    # good and every later query is refused.
                    await self._breaker.record_success()
                    raise DatabaseError(f"FalkorDB query failed: {exc}") from exc
                # Don't retry non-transient errors (e.g. schema/index conflicts)
                if self._is_non_transient(exc):
                    msg = str(exc).lower()
                    expected = any(marker in msg for marker in expected_errors)
                    logger.log(
                        logging.DEBUG if expected else logging.ERROR,
                        "Non-transient FalkorDB %s failure: %s: %s",
                        action,
                        type(exc).__name__,
                        exc,
                    )
                    logger.debug("Non-transient FalkorDB %s failure details", action, exc_info=True)
                    # The server answered, so it is reachable: close the
                    # breaker. Without this a rejected query that happens to be
                    # the half-open probe leaves the breaker HALF_OPEN for good,
                    # and every later query is refused.
                    await self._breaker.record_success()
                    raise DatabaseError(f"FalkorDB {action} failed: {exc}") from exc
                await self._breaker.record_failure()
                logger.warning(
                    "%s attempt %d/%d failed: %s",
                    action.capitalize(),
                    attempt + 1,
                    self.config.retry_count,
                    exc,
                )
                if attempt < self.config.retry_count - 1:
                    if not await self._breaker.allow_request():
                        break
                    base_delay = self.config.retry_delay * (2**attempt)
                    await asyncio.sleep(base_delay * (0.5 + random.random()))
        logger.error(
            "FalkorDB %s failed after %d attempts: %s: %s",
            action,
            self.config.retry_count,
            type(last_exc).__name__ if last_exc is not None else "UnknownError",
            last_exc,
        )
        if last_exc is not None:
            logger.debug(
                "FalkorDB %s failure details",
                action,
                exc_info=(type(last_exc), last_exc, last_exc.__traceback__),
            )
            raise DatabaseUnavailableError(
                f"FalkorDB {action} failed after {self.config.retry_count} attempts: {last_exc}"
            ) from last_exc
        raise DatabaseError(f"FalkorDB {action} failed without an exception")

    # Substrings that indicate a non-transient (permanent) error —
    # retrying will never succeed.
    _NON_TRANSIENT_MARKERS = (
        "already indexed",
        "already exists",
        "unknown index",
        # Malformed or unsupported query: the server answered and rejected
        # it. Retrying re-sends the same bytes, so it can only fail again —
        # and exhausting the budget would misreport a rejection as an outage.
        "syntax error",
        "invalid input",
        "unknown function",
        "type mismatch",
        "procedure not found",
    )

    # How a server reply (``ResponseError``) starts when the server is
    # temporarily unable to run the query, rather than saying the query is
    # wrong. FalkorDB 4.x and 6.0.x word "max pending queries" the same way.
    # ``BUSY``, ``NOREPLICAS`` and ``MISCONF`` are Redis replies that redis-py
    # leaves as a plain ``ResponseError`` with the code kept in the text;
    # GRAPH.QUERY is a write command, so the last two refuse reads too.
    # ``LOADING`` never gets here: redis-py raises ``BusyLoadingError``, a
    # ``ConnectionError``. "Query timed out" is left out on purpose: the query
    # used its whole time budget, so a retry costs as much again and usually
    # times out too, and counting it against the breaker let two slow queries
    # refuse every caller for the recovery timeout. Matched as prefixes, not
    # substrings: a rejection quotes the query ("'timeout' not defined"), and a
    # substring match would retry it and count it against the breaker.
    _TRANSIENT_REPLY_PREFIXES = (
        "max pending queries",
        "busy ",
        "noreplicas ",
        "misconf ",
    )

    @classmethod
    def _is_server_rejection(cls, exc: Exception) -> bool:
        """True when FalkorDB rejected the query itself (not a network or cluster fault).

        Cluster / failover replies (``TRYAGAIN``, ``MOVED``, ``CLUSTERDOWN``,
        a read-only replica, ...) are also ``ResponseError`` subclasses but
        are transient, so they are not counted as a rejection.
        """
        try:
            from redis import exceptions as rx
        except ImportError:  # pragma: no cover - redis ships with falkordb
            return False
        transient = tuple(
            getattr(rx, name)
            for name in (
                "AskError",
                "ClusterDownError",
                "MasterDownError",
                "MovedError",
                "TryAgainError",
                "ReadOnlyError",
                "OutOfMemoryError",
            )
            if hasattr(rx, name)
        )
        return isinstance(exc, rx.ResponseError) and not isinstance(exc, transient)

    @classmethod
    def _is_non_transient(cls, exc: Exception) -> bool:
        msg = str(exc).lower()
        if any(marker in msg for marker in cls._NON_TRANSIENT_MARKERS):
            return True
        # Any other reply from the server is a rejection of the query itself —
        # unknown variable, division by zero, a MERGE on a null key, a missing
        # index, a procedure called with the wrong arity. The wording differs
        # between the C engine (4.x) and the Rust engine (6.0.x), so a list of
        # messages can never be complete. Retrying re-sends the same bytes and
        # can only fail again, and counting it against the circuit breaker lets
        # one bad statement lock every caller out. Cluster redirects, failover,
        # read-only replicas and out-of-memory are ResponseError subclasses that
        # can succeed later, so they stay transient.
        try:
            from redis import exceptions as rx
        except ImportError:  # pragma: no cover - redis ships with falkordb
            return False
        if not isinstance(exc, rx.ResponseError):
            return False
        transient_types = tuple(
            getattr(rx, name)
            for name in (
                "AskError",
                "ClusterDownError",
                "MasterDownError",
                "MovedError",
                "TryAgainError",
                "ReadOnlyError",
                "OutOfMemoryError",
            )
            if hasattr(rx, name)
        )
        if isinstance(exc, transient_types):
            return False
        return not msg.strip().startswith(cls._TRANSIENT_REPLY_PREFIXES)

    # ── Health & Admin ────────────────────────────────────────────

    async def ping(self) -> bool:
        """Send a Redis PING to verify the connection is alive.

        Returns ``False`` rather than raising when the server cannot be
        reached — reporting "not alive" is the point of the call. An
        ``ImportError`` from a missing driver still propagates: that is a
        broken install, not a dead server.
        """
        try:
            self._ensure_client()

            from redis.asyncio import Redis

            redis: Redis = Redis(connection_pool=self._pool)
            # PING needs no query budget; the pool's socket timeout is sized
            # for slow queries and is unbounded when query_timeout_ms is None.
            return await asyncio.wait_for(redis.ping(), timeout=self._PROBE_TIMEOUT_S)
        except ImportError:
            # The driver isn't installed. Reporting that as "server down"
            # sends operators to inspect a healthy database instead of the
            # broken install, so let it surface.
            raise
        except Exception:
            logger.debug("Ping failed", exc_info=True)
            return False

    async def server_version(self) -> tuple[int, int, int] | None:
        """The FalkorDB module version, e.g. ``(4, 20, 4)``; ``None`` when the
        server does not say (``MODULE LIST`` refused, or no graph module). A
        server that cannot be reached, or answers with an error that passes,
        raises: read as "will not say", one ``TRYAGAIN`` left the connection
        unchecked for good."""
        self._ensure_client()
        assert self._driver is not None  # for type-checkers
        from redis.exceptions import NoPermissionError, ResponseError

        try:
            modules = await self._driver.execute_command("MODULE", "LIST")
        except ResponseError as exc:
            msg = str(exc).lower()
            if not isinstance(exc, NoPermissionError) and not any(
                marker in msg for marker in _MODULE_LIST_REFUSALS
            ):
                raise
            logger.debug("MODULE LIST refused", exc_info=True)
            return None
        for module in modules or []:
            info = module if isinstance(module, dict) else dict(zip(module[::2], module[1::2]))
            info = {str(k).lower(): v for k, v in info.items()}
            if str(info.get("name", "")).lower() == "graph":
                try:
                    ver = int(info.get("ver"))
                except (TypeError, ValueError):
                    return None
                return ver // 10000, ver // 100 % 100, ver % 100
        return None

    async def _check_server_version(self) -> None:
        """Refuse a server older than :data:`MIN_SERVER_VERSION`, once per
        connection and before its first query, so nothing is written to it.

        The version is asked with the retries of any query. A server that
        refuses to say (``MODULE LIST`` refused) is let through with a warning:
        refusing it would lock out a managed server that may be current. One
        that cannot be asked fails the query with ``DatabaseUnavailableError``
        — no query runs unchecked — and is asked again by the next one."""
        if self._version_checked:
            return
        async with self._version_lock():
            if self._version_checked:
                return
            version = await self._with_retries(self.server_version, "version check")
            needed = ".".join(str(v) for v in MIN_SERVER_VERSION[:2])
            if version is None:
                logger.warning(
                    "Could not read the FalkorDB server version; GraphRAG SDK needs %s or later",
                    needed,
                )
            elif version < MIN_SERVER_VERSION:
                found = ".".join(str(v) for v in version)
                raise FalkorDBVersionError(
                    f"This FalkorDB server is {found}; GraphRAG SDK needs FalkorDB {needed} or "
                    "later. Earlier versions drop vector-index entries on every write to a node "
                    "that holds a vector, so search and dedup miss entities. Upgrade the "
                    "server, e.g. docker pull falkordb/falkordb:v4.22.2"
                )
            self._version_checked = True

    def _version_lock(self) -> asyncio.Lock:
        """The version check's lock, bound to the running loop and remade when
        that changes: the ``*_sync`` wrappers run each call under a fresh
        ``asyncio.run()``, and a lock that once waited on one loop raises when
        acquired from another."""
        loop = asyncio.get_running_loop()
        if self._version_check_lock is None or self._version_check_loop is not loop:
            self._version_check_lock = asyncio.Lock()
            self._version_check_loop = loop
        return self._version_check_lock

    async def delete_graph(self) -> None:
        """Delete the entire graph using ``GRAPH.DELETE`` (fast).

        Prefer this over ``MATCH (n) DETACH DELETE n`` which hangs on
        large graphs with many indexes.
        """
        self._ensure_client()
        from redis.asyncio import Redis

        redis: Redis = Redis(connection_pool=self._pool)
        self.topology_version = getattr(self, "topology_version", 0) + 1
        try:
            await redis.execute_command("GRAPH.DELETE", self.config.graph_name)
            logger.info("Deleted graph '%s' via GRAPH.DELETE", self.config.graph_name)
        except Exception as exc:
            if "empty" in str(exc).lower() or "invalid" in str(exc).lower():
                logger.debug("Graph '%s' already deleted or empty", self.config.graph_name)
            else:
                raise

    # ── Async context manager ──────────────────────────────────

    async def __aenter__(self) -> FalkorDBConnection:
        return self

    async def __aexit__(self, *exc: object) -> None:
        try:
            await self.close()
        except Exception:
            if exc[0] is None:
                raise
            logger.warning("Error closing connection during __aexit__", exc_info=True)

    # ── Lifecycle ────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying connection pool."""
        if self._pool is not None:
            await self._pool.aclose()
        self._pool = None
        self._driver = None
        self._graph = None
        # The next query reconnects, perhaps to a server replaced meanwhile.
        self._version_checked = False
