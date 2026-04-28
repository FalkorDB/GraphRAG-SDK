# GraphRAG SDK — Core: FalkorDB Connection
# Async-only FalkorDB client using native ``falkordb.asyncio``.
# Origin: User design — production resilience.

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


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


class FalkorDBConnection:
    """Async-only FalkorDB connection using the native async client.

    Wraps ``falkordb.asyncio.FalkorDB`` with:
    - ``redis.asyncio.BlockingConnectionPool`` for bounded pooling
    - Automatic retries with exponential backoff on transient failures
    - Lazy initialisation — no I/O until the first query

    Example::

        conn = FalkorDBConnection(ConnectionConfig(host="localhost"))
        result = await conn.query("RETURN 1")
        await conn.close()
    """

    def __init__(self, config: ConnectionConfig | None = None) -> None:
        from graphrag_sdk.core.circuit_breaker import CircuitBreaker

        self.config = config or ConnectionConfig()
        self._pool: Any | None = None
        self._driver: Any | None = None
        self._graph: Any | None = None
        self._breaker = CircuitBreaker()

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
        }
        if self.config.ssl:
            from redis.asyncio.connection import SSLConnection

            pool_kwargs["connection_class"] = SSLConnection
            pool_kwargs["ssl_cert_reqs"] = self.config.ssl_cert_reqs
            pool_kwargs["ssl_ca_certs"] = self.config.ssl_ca_certs
            pool_kwargs["ssl_certfile"] = self.config.ssl_certfile
            pool_kwargs["ssl_keyfile"] = self.config.ssl_keyfile
            pool_kwargs["ssl_check_hostname"] = self.config.ssl_check_hostname

        self._pool = BlockingConnectionPool(**pool_kwargs)
        self._driver = FalkorDB(connection_pool=self._pool)
        self._graph = self._driver.select_graph(self.config.graph_name)

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
    ) -> Any:
        """Execute a Cypher query with retry logic.

        Args:
            cypher: The Cypher query string.
            params: Optional query parameters.
            timeout: Optional per-query timeout (ms) forwarded to FalkorDB.

        Returns:
            ``QueryResult`` from the async FalkorDB driver.
        """
        self._ensure_client()
        assert self._graph is not None  # for type-checkers

        if not await self._breaker.allow_request():
            raise ConnectionError(
                "Circuit breaker is open — FalkorDB connection is unhealthy. "
                "Requests will resume after recovery timeout."
            )

        effective_timeout = timeout if timeout is not None else self.config.query_timeout_ms

        last_exc: Exception | None = None
        for attempt in range(self.config.retry_count):
            try:
                result = await self._graph.query(cypher, params=params, timeout=effective_timeout)
                await self._breaker.record_success()
                return result
            except Exception as exc:
                last_exc = exc
                # Don't retry non-transient errors (e.g. schema/index conflicts)
                if self._is_non_transient(exc):
                    raise
                await self._breaker.record_failure()
                logger.warning(
                    "Query attempt %d/%d failed: %s",
                    attempt + 1,
                    self.config.retry_count,
                    exc,
                )
                if attempt < self.config.retry_count - 1:
                    if not await self._breaker.allow_request():
                        break
                    base_delay = self.config.retry_delay * (2**attempt)
                    await asyncio.sleep(base_delay * (0.5 + random.random()))
        raise last_exc  # type: ignore[misc]

    # Substrings that indicate a non-transient (permanent) error —
    # retrying will never succeed.
    _NON_TRANSIENT_MARKERS = (
        "already indexed",
        "already exists",
        "unknown index",
    )

    @classmethod
    def _is_non_transient(cls, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(marker in msg for marker in cls._NON_TRANSIENT_MARKERS)

    # ── Health & Admin ────────────────────────────────────────────

    async def ping(self) -> bool:
        """Send a Redis PING to verify the connection is alive."""
        self._ensure_client()
        try:
            from redis.asyncio import Redis

            redis: Redis = Redis(connection_pool=self._pool)
            return await redis.ping()
        except Exception:
            logger.debug("Ping failed", exc_info=True)
            return False

    async def delete_graph(self) -> None:
        """Delete the entire graph using ``GRAPH.DELETE`` (fast).

        Prefer this over ``MATCH (n) DETACH DELETE n`` which hangs on
        large graphs with many indexes.
        """
        self._ensure_client()
        from redis.asyncio import Redis

        redis: Redis = Redis(connection_pool=self._pool)
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
