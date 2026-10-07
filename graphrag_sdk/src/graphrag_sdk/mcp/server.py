# GraphRAG SDK — MCP: Server (Phase 3.2)
# Adapts the transport-agnostic GraphRAGToolset into a live MCP server
# over stdio or SSE. The `mcp` package is an optional dependency, imported
# lazily so the rest of the SDK works without it.

from __future__ import annotations

import hmac
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from graphrag_sdk.mcp.tools import GraphRAGToolset

logger = logging.getLogger(__name__)

Transport = Literal["stdio", "sse"]

ASGIApp = Callable[
    [dict[str, Any], Callable[..., Awaitable[Any]], Callable[..., Awaitable[Any]]], Awaitable[None]
]


def bearer_token_guard(app: ASGIApp, token: str) -> ASGIApp:
    """Wrap an ASGI app so every HTTP request must send ``Authorization: Bearer <token>``.

    Requests without the right token get ``401`` before reaching the MCP
    endpoints. The comparison is constant-time.
    """
    if not token:
        raise ValueError("token must be a non-empty string")
    expected = f"Bearer {token}".encode()

    async def guarded(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            headers = dict(scope.get("headers") or [])
            supplied = headers.get(b"authorization", b"")
            if not hmac.compare_digest(supplied, expected):
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"text/plain"),
                            (b"www-authenticate", b"Bearer"),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": b"Unauthorized"})
                return
        await app(scope, receive, send)

    return guarded


def _require_mcp() -> Any:
    try:
        import mcp  # noqa: F401
        import mcp.server  # noqa: F401

        return mcp
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "The MCP server requires the `mcp` package. Install with: pip install graphrag-sdk[mcp]"
        ) from exc


class GraphRAGMCPServer:
    """MCP server exposing a GraphRAG instance as MCP tools.

    Args:
        rag: A constructed ``GraphRAG`` facade.
        name: Server name advertised to MCP clients.
        auth_token: For the SSE transport: require
            ``Authorization: Bearer <auth_token>`` on every request. Without
            it the SSE server only accepts unauthenticated local use (binding
            a non-loopback host without a token is refused).
    """

    def __init__(
        self, rag: Any, *, name: str = "graphrag-sdk", auth_token: str | None = None
    ) -> None:
        self._rag = rag
        self._name = name
        self._auth_token = auth_token or None
        self._toolset = GraphRAGToolset(rag)

    @property
    def toolset(self) -> GraphRAGToolset:
        return self._toolset

    def _build_server(self) -> Any:
        """Construct the underlying ``mcp.server.Server`` with handlers."""
        _require_mcp()
        from mcp import types  # type: ignore
        from mcp.server import Server  # type: ignore

        server = Server(self._name)
        toolset = self._toolset

        @server.list_tools()  # type: ignore[misc]
        async def _list_tools() -> list[Any]:
            return [
                types.Tool(
                    name=t.name,
                    description=t.description,
                    inputSchema=t.input_schema,
                )
                for t in toolset.tools
            ]

        @server.call_tool()  # type: ignore[misc]
        async def _call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
            tool = toolset.by_name(name)
            if tool is None:
                return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
            try:
                text = await tool.handler(arguments or {})
            except Exception as exc:  # noqa: BLE001 - surface as tool-level error
                logger.warning("MCP tool %s failed: %s", name, exc)
                return [types.TextContent(type="text", text=f"Error running tool '{name}': {exc}")]
            return [types.TextContent(type="text", text=text)]

        return server

    async def run(self, transport: Transport = "stdio", **kwargs: Any) -> None:
        """Run the server over the given transport until the client disconnects."""
        _require_mcp()
        server = self._build_server()
        init_options = server.create_initialization_options()

        if transport == "stdio":
            from mcp.server.stdio import stdio_server  # type: ignore

            async with stdio_server() as (read, write):
                await server.run(read, write, init_options)
        elif transport == "sse":
            await self._run_sse(server, init_options, **kwargs)
        else:  # pragma: no cover - guarded by typing
            raise ValueError(f"Unsupported transport: {transport!r}")

    async def _run_sse(
        self,
        server: Any,
        init_options: Any,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:  # pragma: no cover - network server
        check_sse_binding(host, self._auth_token)
        import uvicorn  # type: ignore
        from mcp.server.sse import SseServerTransport  # type: ignore
        from starlette.applications import Starlette  # type: ignore
        from starlette.routing import Mount, Route  # type: ignore

        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Any) -> None:
            async with sse.connect_sse(request.scope, request.receive, request._send) as (
                read,
                write,
            ):
                await server.run(read, write, init_options)

        app: Any = Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ]
        )
        if self._auth_token:
            app = bearer_token_guard(app, self._auth_token)
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        await uvicorn.Server(config).serve()


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def check_sse_binding(host: str, auth_token: str | None) -> None:
    """Refuse to serve SSE on a non-loopback host without an auth token.

    The tools can ingest data and query the whole graph; exposing them on a
    network interface without authentication would hand both to anyone who
    can reach the port.
    """
    if host not in _LOOPBACK_HOSTS and not auth_token:
        raise ValueError(
            f"Refusing to serve MCP over SSE on {host!r} without an auth token. "
            "Set GRAPHRAG_MCP_TOKEN (or pass --token / auth_token=), or bind 127.0.0.1."
        )
