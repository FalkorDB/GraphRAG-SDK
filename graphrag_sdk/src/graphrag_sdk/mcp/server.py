# GraphRAG SDK — MCP server
# Serves GraphRAGToolkit's tools to an MCP client (Claude Code, Cursor, ...)
# over stdio or SSE. The client's agent plans and answers; the server runs
# each tool call and returns its JSON result. Works with mcp 1.x and 2.x;
# the `mcp` package is an optional extra, imported lazily.

from __future__ import annotations

import copy
import hmac
import inspect
import json
import logging
import os
import weakref
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal

from graphrag_sdk.core.exceptions import ReadOnlyViolation
from graphrag_sdk.tools.errors import ToolError
from graphrag_sdk.tools.toolkit import FinalizePolicy, GraphRAGToolkit

logger = logging.getLogger(__name__)

Transport = Literal["stdio", "sse"]

ASGIApp = Callable[
    [dict[str, Any], Callable[..., Awaitable[Any]], Callable[..., Awaitable[Any]]], Awaitable[None]
]

#: Sent to the client as the server's instructions: how to use the tools.
GUIDE = """\
GraphRAG knowledge-graph tools. You plan, call the tools and write the answer.
The question-answering tools only read the graph, and number every piece of
evidence they return.

To answer a question:
1. retrieve first, with the user's question word for word (do not shorten
   it to keywords) and top_k 8 or more: hybrid search returning entities,
   relations and the source passages that usually hold the answer. It runs
   every search path; name a few in paths only when you know the question
   needs just those, and run it again without paths if nothing comes back.
2. lookup_entity: turn names in the question into exact entity names and ids.
3. traverse (all neighbours, or one relation path with via) or graph_walk
   (strongest paths; how A connects to B) to follow relations from them.
4. status or get_ontology: what the graph holds (entity labels, relation
   types, counts), for questions about the graph itself.
5. run_skill: built-in analyses (compare entities, impact, contradictions,
   gaps, timeline).
6. cypher_query for counts, sums, sorting and property filters. When the
   question is only that, call get_ontology and cypher_query and skip
   retrieve; otherwise use it last. When a result says truncated: true, do
   not count its rows; aggregate in Cypher.

Answer what was asked, briefly, and leave out evidence that does not answer
it. Cite evidence as [n] after each claim, using only numbers the tools
returned. Always call check_citations on the final draft and fix what it
reports before replying. If the evidence does not answer the question, say
so. When a call fails, read its hint: it names the tool to call or the exact
names to use.
"""

#: Added to :data:`GUIDE` when the ingestion tools are served.
INGEST_GUIDE = """\

To build or extend the graph from documents:
1. Ask the user whether they have an ontology (the entity and relation types
   to extract). If they do, declare it with set_ontology and go to step 3.
2. If not, design one: call preview_files and follow next_offset until it is
   null, so you have seen one sample from every file. Choose ONE ontology for
   the whole corpus (types that recur across files, each with a short
   description, and relations with their [source, target] patterns), and
   declare it with set_ontology. Declare a specific relation type for each
   kind of link the samples show (owns, causes, supplies, depends on, ...):
   with only a few broad types, extraction files facts under the wrong one.
   Give each relation every [source, target] pair it can join; a relation
   between undeclared types is dropped.
3. Plan per file type from what preview_files showed, e.g. structural chunking
   for Markdown with headings. Call ingest(files=[...], plan=[rules]) in
   batches of at most 50 files; a glob such as 'reports/*.pdf' is fine.
4. Call finalize once, after the last batch.
5. Read each result's pruned and hint. Relations dropped for an undeclared
   [source, target] pair are often real facts. Add a pair only when the
   relation's meaning fits it (SUPPLIES Company->Product, not
   Company->Event); otherwise declare a better relation type for those
   facts. Then call ingest(files=[the affected files], reextract=true) and
   finalize again. Do the same for undeclared types that matter (dropped, or
   stored as Unknown).
To add a single text instead, call ingest(text=..., document_id=...).
"""

#: Added to :data:`GUIDE` when only text ingestion is served.
TEXT_INGEST_GUIDE = """\

To add documents: optionally declare the ontology with set_ontology first,
then ingest(text=..., document_id=...) per document, then finalize once.
Check each result's pruned for types the ontology does not declare
(dropped, or stored as Unknown); after adding them with set_ontology,
ingest the same text and document_id with reextract=true.
"""


def guide_for(toolkit: GraphRAGToolkit) -> str:
    """:data:`GUIDE`, plus the ingestion steps for the tools ``toolkit`` serves."""
    names = {spec.name for spec in toolkit.tool_specs()}
    if "preview_files" in names and "ingest" in names:
        return GUIDE + INGEST_GUIDE
    if "ingest" in names:
        return GUIDE + TEXT_INGEST_GUIDE
    return GUIDE


#: Longest JSON text returned for one tool call.
MAX_RESULT_CHARS = 40_000
# Evidence ledgers kept for HTTP sessions identified only by id.
MAX_SESSIONS = 1024
_LONG_STRING_CHARS = 1000


def fit_json(data: dict[str, Any], max_chars: int = MAX_RESULT_CHARS) -> str:
    """``data`` as JSON of at most about ``max_chars`` characters.

    An oversized result loses the back half of its longest list until it
    fits (then long strings are shortened), and says so with
    ``truncated: true`` and a hint.
    """
    text = json.dumps(data, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    data = copy.deepcopy(data)
    while len(text) > max_chars:
        lists = [k for k, v in data.items() if isinstance(v, list) and len(v) > 1]
        if not lists:
            break
        key = max(lists, key=lambda k: len(json.dumps(data[k], ensure_ascii=False, default=str)))
        data[key] = data[key][: len(data[key]) // 2]
        text = json.dumps(data, ensure_ascii=False, default=str)
    if len(text) > max_chars:

        def shorten(value: Any) -> Any:
            if isinstance(value, str) and len(value) > _LONG_STRING_CHARS:
                return value[:_LONG_STRING_CHARS] + "…"
            if isinstance(value, list):
                return [shorten(v) for v in value]
            if isinstance(value, dict):
                return {k: shorten(v) for k, v in value.items()}
            return value

        data = shorten(data)
    data["truncated"] = True
    note = "The result was too large and was cut; narrow the call (fewer rows, smaller top_k)."
    data["hint"] = f"{data['hint']} {note}" if data.get("hint") else note
    return json.dumps(data, ensure_ascii=False, default=str)[: max_chars + 2000]


class _ToolFailed(Exception):
    """Raised inside an mcp 1.x handler so the client receives isError."""


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


def _stateless_versions() -> frozenset[str]:
    """MCP protocol versions with no connection identity between requests."""
    try:
        from mcp_types.version import MODERN_PROTOCOL_VERSIONS  # type: ignore
    except ImportError:  # mcp 1.x
        return frozenset()
    return frozenset(MODERN_PROTOCOL_VERSIONS)


def _require_mcp() -> Any:
    try:
        import mcp  # noqa: F401
        import mcp.server  # noqa: F401

        return mcp
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "The MCP server requires the `mcp` package. Install with: pip install graphrag-sdk[mcp]"
        ) from exc


def _is_mcp_v2(server_cls: type) -> bool:
    """mcp 2.x takes handlers as constructor arguments; 1.x uses decorators."""
    return "on_call_tool" in inspect.signature(server_cls.__init__).parameters


class GraphRAGMCPServer:
    """MCP server exposing a GraphRAG graph as :class:`GraphRAGToolkit` tools.

    Each MCP client session gets its own evidence ledger
    (:meth:`GraphRAGToolkit.new_session`), so ``[n]`` numbers never mix
    between conversations. Results are JSON; failures come back with
    ``isError`` and a hint naming the fix.

    Args:
        rag: A constructed ``GraphRAG`` facade.
        name: Server name advertised to MCP clients.
        auth_token: For the SSE transport: require
            ``Authorization: Bearer <token>`` on every request. Without
            it the SSE server only accepts unauthenticated local use (binding
            a non-loopback host without a token is refused).
        read_only: Serve only the read tools (no ingestion tools).
        include: Optional subset of tool names to serve.
        finalize_policy: Passed to :class:`GraphRAGToolkit`.
        files_root: Folder ``preview_files`` and ``ingest(files=...)`` read
            from; ``None`` (default) serves text ingestion only.
        plan_limits: The strategies an agent's ingest plan may use; passed
            to :class:`GraphRAGToolkit`.
        instructions: Server instructions sent to the client; defaults to
            :func:`guide_for` the served tools.
    """

    def __init__(
        self,
        rag: Any,
        *,
        name: str = "graphrag-sdk",
        auth_token: str | None = None,
        read_only: bool = False,
        include: Sequence[str] | None = None,
        finalize_policy: FinalizePolicy = "manual",
        files_root: str | os.PathLike[str] | None = None,
        plan_limits: Mapping[str, Any] | None = None,
        instructions: str | None = None,
    ) -> None:
        self._rag = rag
        self._name = name
        self._auth_token = auth_token or None
        self._toolkit = GraphRAGToolkit(
            rag,
            read_only=read_only,
            include=include,
            finalize_policy=finalize_policy,
            files_root=files_root,
            plan_limits=plan_limits,
        )
        self._instructions = guide_for(self._toolkit) if instructions is None else instructions
        self._sessions: weakref.WeakKeyDictionary[Any, GraphRAGToolkit] = (
            weakref.WeakKeyDictionary()
        )
        self._sessions_by_id: OrderedDict[str, GraphRAGToolkit] = OrderedDict()
        self._connection_toolkit: ContextVar[GraphRAGToolkit | None] = ContextVar(
            "graphrag_mcp_connection_toolkit", default=None
        )

    @contextmanager
    def connection_scope(self) -> Iterator[GraphRAGToolkit]:
        """Give the requests of one transport connection their own ledger.

        Requests handled inside the block (the server copies the context into
        each request task) use one fresh toolkit session, whatever the MCP
        version says about sessions. The SSE transport wraps each client
        stream in this, so stateless clients never share evidence.
        """
        toolkit = self._toolkit.new_session()
        token = self._connection_toolkit.set(toolkit)
        try:
            yield toolkit
        finally:
            self._connection_toolkit.reset(token)

    @property
    def toolkit(self) -> GraphRAGToolkit:
        """The base toolkit; each client session uses a copy of it."""
        return self._toolkit

    @property
    def instructions(self) -> str:
        return self._instructions

    def _session_toolkit(self, session: Any) -> GraphRAGToolkit:
        """The toolkit (and evidence ledger) for one client connection.

        mcp 1.x hands every request of a connection the same
        ``ServerSession``; mcp 2.x builds one per request around a
        connection object, so key on that, or on its HTTP session id. The
        stateless 2026-07-28 protocol gives each request a fresh connection
        and no id: over SSE the connection scope still keeps one ledger per
        client stream; over stdio they share the server's ledger (one stdio
        process serves one client).
        """
        scoped = self._connection_toolkit.get()
        if scoped is not None:
            return scoped
        if session is None:
            return self._toolkit
        key = getattr(session, "_connection", None) or session
        if getattr(key, "protocol_version", None) in _stateless_versions() and not getattr(
            key, "session_id", None
        ):
            return self._toolkit
        session_id = getattr(key, "session_id", None)
        if isinstance(session_id, str) and session_id:
            toolkit = self._sessions_by_id.get(session_id)
            if toolkit is None:
                toolkit = self._toolkit.new_session()
                self._sessions_by_id[session_id] = toolkit
                if len(self._sessions_by_id) > MAX_SESSIONS:
                    self._sessions_by_id.popitem(last=False)
            else:
                self._sessions_by_id.move_to_end(session_id)
            return toolkit
        try:
            toolkit = self._sessions.get(key)
            if toolkit is None:
                toolkit = self._toolkit.new_session()
                self._sessions[key] = toolkit
            return toolkit
        except TypeError:  # not weak-referenceable
            return self._toolkit

    def tool_dicts(self) -> list[dict[str, Any]]:
        """The tool list as MCP ``Tool`` dicts (camelCase keys)."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "inputSchema": spec.input_schema,
                "annotations": {
                    "readOnlyHint": spec.read_only,
                    "destructiveHint": spec.destructive,
                    "openWorldHint": False,
                },
            }
            for spec in self._toolkit.tool_specs()
        ]

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None, session: Any = None
    ) -> tuple[str, bool]:
        """Run tool ``name`` for ``session``; returns ``(text, is_error)``."""
        toolkit = self._session_toolkit(session)
        try:
            result = await toolkit.call(name, arguments or {})
        except (ToolError, ReadOnlyViolation) as exc:
            return str(exc), True
        except Exception as exc:  # noqa: BLE001 - surface as a tool-level error
            logger.warning("MCP tool %s failed: %s", name, exc, exc_info=True)
            return f"Error running {name}: {type(exc).__name__}: {exc}", True
        return fit_json(result.model_dump(mode="json", exclude_none=True)), False

    def _build_server(self) -> Any:
        """Construct the underlying ``mcp.server.Server`` with handlers."""
        _require_mcp()
        from mcp import types  # type: ignore
        from mcp.server import Server  # type: ignore

        def tools() -> list[Any]:
            return [types.Tool.model_validate(d) for d in self.tool_dicts()]

        def text_result(text: str, is_error: bool) -> Any:
            return types.CallToolResult.model_validate(
                {"content": [{"type": "text", "text": text}], "isError": is_error}
            )

        if _is_mcp_v2(Server):

            async def on_list_tools(ctx: Any, params: Any) -> Any:
                return types.ListToolsResult(tools=tools())

            async def on_call_tool(ctx: Any, params: Any) -> Any:
                session = getattr(ctx, "session", None)
                text, is_error = await self.call_tool(params.name, params.arguments, session)
                return text_result(text, is_error)

            return Server(
                self._name,
                instructions=self._instructions,
                on_list_tools=on_list_tools,
                on_call_tool=on_call_tool,
            )

        server = Server(self._name, instructions=self._instructions)

        @server.list_tools()  # type: ignore[misc]
        async def _list_tools() -> list[Any]:
            return tools()

        @server.call_tool()  # type: ignore[misc]
        async def _call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
            try:
                session = server.request_context.session
            except LookupError:  # pragma: no cover - outside a request
                session = None
            text, is_error = await self.call_tool(name, arguments, session)
            if is_error:
                # mcp 1.x turns a raised exception into isError=True.
                raise _ToolFailed(text)
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

        sse = SseServerTransport("/messages/", **sse_security_kwargs(host, port))

        async def handle_sse(request: Any) -> None:
            with self.connection_scope():
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


def sse_security_kwargs(host: str, port: int) -> dict[str, Any]:
    """DNS-rebinding protection for a loopback SSE server, when ``mcp`` supports it.

    A server bound to 127.0.0.1 / localhost accepts only requests whose
    ``Host`` (and ``Origin``, when sent) name it, so a web page cannot reach
    it through a rebound domain. A server on any other interface requires a
    bearer token instead (see :func:`check_sse_binding`), and clients reach it
    by whatever name or address they use, so no Host list is imposed.
    ``mcp`` releases without transport security get no settings.
    """
    if host not in _LOOPBACK_HOSTS:
        return {}
    try:
        from mcp.server.transport_security import TransportSecuritySettings
    except ImportError:  # pragma: no cover - older mcp
        logger.warning("This mcp version has no DNS-rebinding protection; upgrade mcp")
        return {}
    names = {host, "127.0.0.1", "localhost"}
    # IPv6 literals appear bracketed in a Host header.
    hosts = sorted(f"[{n}]:{port}" if ":" in n else f"{n}:{port}" for n in names)
    return {
        "security_settings": TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=hosts,
            allowed_origins=[f"http://{h}" for h in hosts] + [f"https://{h}" for h in hosts],
        )
    }
