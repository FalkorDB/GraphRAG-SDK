---
title: "Connection"
nav_order: 5
parent: "API Reference"
grand_parent: "GraphRAG-SDK"
description: "ConnectionConfig and FalkorDBConnection — async pooled connection to FalkorDB."
---

# Connection

Module: `graphrag_sdk`  ·  Import: `from graphrag_sdk import ConnectionConfig, FalkorDBConnection`

Async-only FalkorDB connection wrapping `falkordb.asyncio` with bounded pooling, retries, and a circuit breaker.

---

## `ConnectionConfig`

```python
from dataclasses import dataclass

@dataclass
class ConnectionConfig:
    host: str = "localhost"
    port: int = 6379
    username: str | None = None
    password: str | None = None     # repr=False
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
```

| Field | Type | Default | Description |
|---|---|---|---|
| `host` | `str` | `"localhost"` | Database host. |
| `port` | `int` | `6379` | Database port. |
| `username` | `str \| None` | `None` | ACL username. |
| `password` | `str \| None` | `None` | ACL password. Never serialised in `repr()`. |
| `graph_name` | `str` | `"knowledge_graph"` | Target graph name. Provides multi-tenant isolation — one instance, many graphs. |
| `max_connections` | `int` | `16` | Pool size. |
| `retry_count` | `int` | `3` | Max attempts per `query()` before giving up. |
| `retry_delay` | `float` | `1.0` | Base backoff (seconds); doubled per attempt with ±50% jitter. |
| `pool_timeout` | `float` | `30.0` | Seconds to wait for a free pool connection. |
| `query_timeout_ms` | `int \| None` | `10_000` | Per-query timeout forwarded to FalkorDB. `None` disables. |
| `ssl` | `bool` | `False` | Enable TLS. Other `ssl_*` fields are no-ops unless `True`. |
| `ssl_cert_reqs` | `str` | `"required"` | TLS certificate requirements (`required`, `optional`, `none`). Mirrors `redis-py`. |
| `ssl_ca_certs` | `str \| None` | `None` | CA bundle path. |
| `ssl_certfile` | `str \| None` | `None` | Client cert path. |
| `ssl_keyfile` | `str \| None` | `None` | Client key path. |
| `ssl_check_hostname` | `bool` | `True` | Verify hostname against the server cert SAN. |

### `ConnectionConfig.from_url`

```python
@classmethod
def from_url(cls, url: str, **kwargs: Any) -> ConnectionConfig
```

Parse a `redis://` or `rediss://` URL. `rediss://` defaults `ssl=True`. Extra kwargs override parsed values.

```python
cfg = ConnectionConfig.from_url("rediss://user:pass@host:6380", graph_name="prod")
```

#### Raises

- `ValueError` — scheme is not `redis://` or `rediss://`.

---

## `FalkorDBConnection`

```python
class FalkorDBConnection:
    def __init__(self, config: ConnectionConfig | None = None) -> None
```

Lazy-init pooled connection. No I/O until the first `query()`.

### Methods

| Method | Signature | Purpose |
|---|---|---|
| `query` | `async (cypher, params=None, *, timeout=None) -> Any` | Execute a Cypher query with retry. Returns the FalkorDB `QueryResult`. |
| `ping` | `async () -> bool` | Redis PING. `False` on any failure. |
| `delete_graph` | `async () -> None` | `GRAPH.DELETE` the entire graph. Fast — preferred over `MATCH (n) DETACH DELETE n` on large graphs. |
| `close` | `async () -> None` | Close the underlying pool. Idempotent. |

### `query`

```python
async def query(
    self,
    cypher: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: int | None = None,
) -> Any
```

#### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `cypher` | `str` | — required — | Cypher query. |
| `params` | `dict[str, Any] \| None` | `None` | Bound parameters. |
| `timeout` | `int \| None` | `config.query_timeout_ms` | Per-query timeout in ms. |

#### Returns

`QueryResult` from the async FalkorDB driver. Inspect `.result_set` for rows.

#### Raises

- `ConnectionError` — circuit breaker is open (FalkorDB is unhealthy).
- Any FalkorDB driver exception that isn't classified as transient. Non-transient errors are surfaced immediately without retries.

### Async context manager

```python
async with FalkorDBConnection(ConnectionConfig(host="localhost")) as conn:
    result = await conn.query("RETURN 1")
```

`__aexit__` calls `close()`.

## See also

- [API Reference → GraphRAG](./graphrag) — pass a `ConnectionConfig` or a `FalkorDBConnection` to the facade.
