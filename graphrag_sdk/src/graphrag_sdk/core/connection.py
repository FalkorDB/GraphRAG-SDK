# GraphRAG SDK 2.0 — Core: FalkorDB Connection
# Async-only FalkorDB client using native ``falkordb.asyncio``.
# Origin: User design — production resilience.

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """FalkorDB connection configuration."""

    host: str = "localhost"
    port: int = 6379
    username: str | None = None
    password: str | None = None
    graph_name: str = "knowledge_graph"
    max_connections: int = 16
    retry_count: int = 3
    retry_delay: float = 1.0
    pool_timeout: float = 30.0
    query_timeout_ms: int | None = 10_000

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> "ConnectionConfig":
        """Create a ConnectionConfig from a ``redis://`` URL.

        Supports ``redis://[user:pass@]host[:port][/db]``.
        Extra keyword arguments override parsed values.
        """
        parsed = urlparse(url)
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
        self.config = config or ConnectionConfig()
        self._pool: Any | None = None
        self._driver: Any | None = None
        self._graph: Any | None = None

    def _ensure_client(self) -> None:
        """Lazy-init the async FalkorDB driver and graph handle."""
        if self._driver is not None:
            return

        try:
            from falkordb.asyncio import FalkorDB
            from redis.asyncio import BlockingConnectionPool
        except ImportError:
            raise ImportError(
                "falkordb package is required. Install with: pip install falkordb"
            )

        self._pool = BlockingConnectionPool(
            host=self.config.host,
            port=self.config.port,
            username=self.config.username,
            password=self.config.password,
            max_connections=self.config.max_connections,
            timeout=self.config.pool_timeout,
            decode_responses=True,
        )
        self._driver = FalkorDB(connection_pool=self._pool)
        self._graph = self._driver.select_graph(self.config.graph_name)

        logger.info(
            "Connected to FalkorDB (async) at %s:%s",
            self.config.host,
            self.config.port,
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

        effective_timeout = timeout if timeout is not None else self.config.query_timeout_ms

        last_exc: Exception | None = None
        for attempt in range(self.config.retry_count):
            try:
                return await self._graph.query(
                    cypher, params=params, timeout=effective_timeout
                )
            except Exception as exc:
                last_exc = exc
                # Don't retry non-transient errors (e.g. schema/index conflicts)
                if self._is_non_transient(exc):
                    raise
                logger.warning(
                    "Query attempt %d/%d failed: %s",
                    attempt + 1,
                    self.config.retry_count,
                    exc,
                )
                if attempt < self.config.retry_count - 1:
                    await asyncio.sleep(
                        self.config.retry_delay * (attempt + 1)
                    )
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

    # ── Lifecycle ────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying connection pool."""
        if self._pool is not None:
            await self._pool.aclose()
        self._pool = None
        self._driver = None
        self._graph = None
