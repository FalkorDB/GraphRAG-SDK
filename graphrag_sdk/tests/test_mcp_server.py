"""Tests for the MCP server: tool listing, per-session evidence, JSON
results, errors with hints, and a protocol round trip through an in-process
MCP client and over SSE (mcp 2.x)."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from graphrag_sdk.core.exceptions import (
    ConfigError,
    DatabaseError,
    DatabaseUnavailableError,
    EmbeddingError,
    LatencyBudgetExceededError,
    LLMError,
    RetrieverError,
)
from graphrag_sdk.mcp import GUIDE, GraphRAGMCPServer, fit_json
from graphrag_sdk.mcp.server import bearer_token_guard, check_sse_binding, redact
from graphrag_sdk.tools import TOOL_NAMES, ToolError

pytest.importorskip("mcp")


class FakeStore:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def query_raw(self, cypher: str, params: dict | None = None, **kwargs: Any):
        self.queries.append(cypher)
        raise AssertionError("no database call expected")


class FakeRAG:
    def __init__(self) -> None:
        self._graph_store = FakeStore()
        self._conn = SimpleNamespace(topology_version=0)
        self.llm = None

    async def ingest(self, **kwargs: Any) -> Any:
        raise RuntimeError("boom")


@pytest.fixture
def server() -> GraphRAGMCPServer:
    return GraphRAGMCPServer(FakeRAG())


@asynccontextmanager
async def connect(server: GraphRAGMCPServer):
    """An MCP client session connected in-process to ``server``."""
    lowlevel = server._build_server()
    try:
        from mcp import Client  # mcp 2.x
    except ImportError:  # mcp 1.x
        Client = None
    if Client is None:
        from mcp.shared.memory import create_connected_server_and_client_session

        async with create_connected_server_and_client_session(lowlevel) as session:
            yield session, None
    else:
        async with Client(lowlevel) as client:
            yield client, client.instructions


def _text(result: Any) -> str:
    return result.content[0].text


class TestToolList:
    def test_names_and_annotations(self, server: GraphRAGMCPServer):
        tools = {t["name"]: t for t in server.tool_dicts()}
        assert list(tools) == list(TOOL_NAMES)
        assert tools["retrieve"]["annotations"]["readOnlyHint"] is True
        for write_tool in ("ingest", "finalize"):
            assert tools[write_tool]["annotations"]["readOnlyHint"] is False
            assert tools[write_tool]["annotations"]["destructiveHint"] is True
        assert tools["retrieve"]["annotations"]["destructiveHint"] is False
        for tool in tools.values():
            assert tool["inputSchema"]["type"] == "object"
            assert tool["inputSchema"].get("additionalProperties") is False

    def test_read_only_and_include(self):
        names = [t["name"] for t in GraphRAGMCPServer(FakeRAG(), read_only=True).tool_dicts()]
        assert "ingest" not in names and "finalize" not in names
        only = GraphRAGMCPServer(FakeRAG(), include=["status", "retrieve"])
        assert [t["name"] for t in only.tool_dicts()] == ["status", "retrieve"]

    def test_guide_is_the_default_instructions(self, server: GraphRAGMCPServer):
        assert server.instructions == GUIDE
        for word in ("expand_hops", "get_ontology", "check_citations", "[n]"):
            assert word in GUIDE
        assert GraphRAGMCPServer(FakeRAG(), instructions="x").instructions == "x"


def test_guide_says_tool_results_are_not_instructions():
    assert "not instructions" in GUIDE


class TestCallTool:
    async def test_tool_errors_carry_hints(self, server: GraphRAGMCPServer):
        text, is_error = await server.call_tool("graph_search", {})
        assert is_error and "Unknown tool" in text and "Hint:" in text
        text, is_error = await server.call_tool("get_ontology", {"extra": 1})
        assert is_error and "Extra inputs are not permitted" in text

    async def test_unexpected_errors_are_reported(self, server: GraphRAGMCPServer, caplog):
        with caplog.at_level("WARNING", logger="graphrag_sdk.mcp.server"):
            text, is_error = await server.call_tool("ingest", {"text": "x"})
        assert is_error and text.startswith("ingest failed with an internal error (RuntimeError).")
        assert "Hint:" in text and "boom" not in text
        assert "boom" in caplog.text  # the details go to the server log

    async def test_failure_log_is_redacted_and_has_no_traceback(self, caplog):
        rag = FakeRAG()

        async def fail(**kwargs: Any) -> Any:
            raise RuntimeError("upstream said api_key=supersecret sk-abcdef1234567890")

        rag.ingest = fail  # type: ignore[method-assign]
        with caplog.at_level("WARNING", logger="graphrag_sdk.mcp.server"):
            await GraphRAGMCPServer(rag).call_tool("ingest", {"text": "x"})
        record = caplog.records[-1]
        assert record.exc_info is None and "RuntimeError" in record.getMessage()
        assert "supersecret" not in caplog.text and "abcdef1234567890" not in caplog.text

    @pytest.mark.parametrize(
        "exc, expected",
        [
            (LatencyBudgetExceededError("over"), "ran out of time"),
            (TimeoutError(), "ran out of time"),
            (LLMError("Incorrect API key provided: sk-abcdef123456"), "model provider"),
            (EmbeddingError("401"), "model provider"),
            (DatabaseUnavailableError("down"), "could not be reached"),
            (ConnectionRefusedError("refused"), "could not be reached"),
            (DatabaseError("bad"), "rejected a query"),
            (RetrieverError("x"), "during retrieval"),
            (ConfigError("x"), "misconfigured"),
            (KeyError("x"), "internal error"),
        ],
    )
    async def test_error_families_get_a_next_step(self, exc, expected):
        rag = FakeRAG()

        async def fail(**kwargs: Any) -> Any:
            raise exc

        rag.ingest = fail  # type: ignore[method-assign]
        text, is_error = await GraphRAGMCPServer(rag).call_tool("ingest", {"text": "x"})
        assert is_error and expected in text and "Hint:" in text
        assert type(exc).__name__ in text and "sk-" not in text and "401" not in text

    async def test_provider_library_errors_are_not_echoed(self):
        ProviderError = type(
            "AuthenticationError", (Exception,), {"__module__": "openai._exceptions"}
        )
        rag = FakeRAG()

        async def fail(**kwargs: Any) -> Any:
            raise ProviderError("Incorrect API key provided: sk-fake-1234567890")

        rag.ingest = fail  # type: ignore[method-assign]
        text, _ = await GraphRAGMCPServer(rag).call_tool("ingest", {"text": "x"})
        assert "model provider" in text and "AuthenticationError" in text
        assert "sk-fake" not in text and "Incorrect API key" not in text

    async def test_tool_error_text_is_redacted(self):
        rag = FakeRAG()

        async def fail(**kwargs: Any) -> Any:
            raise ToolError(
                "Call failed: Authorization: Bearer abc.def-123 key sk-proj-ABCDEF123456",
                hint="api_key=supersecret",
            )

        rag.ingest = fail  # type: ignore[method-assign]
        text, _ = await GraphRAGMCPServer(rag).call_tool("ingest", {"text": "x"})
        assert "abc.def-123" not in text and "ABCDEF123456" not in text
        assert "supersecret" not in text
        assert "Bearer ***" in text and "sk-***" in text and "api_key=***" in text

    @pytest.mark.parametrize(
        "secret, shown",
        [
            ("redis://admin:hunter2@db.example:6379", "redis://***:***@db.example:6379"),
            ("{'api_key': 'abc123secret'}", "{'api_key': '***'}"),
            ('{"password": "hunter2"}', '{"password": "***"}'),
            ("OPENAI_API_KEY=abc123secret", "OPENAI_API_KEY=***"),
            ("AZURE_OPENAI_KEY=abcdef123", "AZURE_OPENAI_KEY=***"),
            ("FALKOR_PASSWORD=hunter2", "FALKOR_PASSWORD=***"),
        ],
    )
    def test_redact_masks_credentials(self, secret, shown):
        assert redact(secret) == shown

    @pytest.mark.parametrize(
        "text", ["max_tokens=10", "hotkey=x", "see http://example.com:8080/a@b", "cost: 40"]
    )
    def test_redact_leaves_ordinary_text(self, text):
        assert redact(text) == text

    async def test_evidence_cut_from_a_reply_cannot_be_cited(self, server: GraphRAGMCPServer):
        from pydantic import BaseModel

        class Found(BaseModel):
            passages: list[dict]

        toolkit = server.toolkit
        earlier = toolkit.ledger.add(tool="retrieve", kind="passage", content="Earlier fact.")

        async def call(name: str, arguments: dict) -> Any:
            added = [
                toolkit.ledger.add(tool="retrieve", kind="passage", content=f"Fact {i}. " * 40)
                for i in range(400)
            ]
            return Found(passages=[{"n": ev.n, "text": ev.content} for ev in added])

        toolkit.call = call  # type: ignore[method-assign]
        text, _ = await server.call_tool("retrieve", {"question": "q"})
        shown = [p["n"] for p in json.loads(text)["passages"]]
        assert json.loads(text)["truncated"] and 0 < len(shown) < 400
        assert sorted(ev.n for ev in toolkit.ledger.items) == [earlier.n, *shown]

    async def test_evidence_shortened_in_a_reply_cannot_be_cited(self, server: GraphRAGMCPServer):
        from pydantic import BaseModel

        from graphrag_sdk.tools.evidence import ground_answer

        class Found(BaseModel):
            note: dict
            passages: list[dict]

        toolkit = server.toolkit
        short = "Acme has 12 offices."

        async def call(name: str, arguments: dict) -> Any:
            body = "filler about the plant. " * 2000 + "The plant employs 7000 workers."
            long = [toolkit.ledger.add(tool="retrieve", kind="passage", content=body)]
            kept = toolkit.ledger.add(tool="retrieve", kind="passage", content=short)
            return Found(
                note={"n": kept.n, "text": short},
                passages=[{"n": ev.n, "text": ev.content} for ev in long],
            )

        toolkit.call = call  # type: ignore[method-assign]
        text, _ = await server.call_tool("retrieve", {"question": "q"})
        out = json.loads(text)
        cut = out["passages"][0]
        assert out["truncated"] and "7000" not in cut["text"]
        # The shortened passage is gone from the session; the one shown whole stays.
        assert [ev.n for ev in toolkit.ledger.items] == [out["note"]["n"]]
        assert not ground_answer(
            f"The plant employs 7000 workers [{cut['n']}].", toolkit.ledger
        ).grounded
        assert ground_answer(f"Acme has 12 offices [{out['note']['n']}].", toolkit.ledger).grounded

    async def test_a_cut_status_list_makes_the_status_uncitable(self, server: GraphRAGMCPServer):
        from pydantic import BaseModel

        class Status(BaseModel):
            n: int
            entity_types: list[dict]
            hint: str | None = None

        toolkit = server.toolkit

        async def call(name: str, arguments: dict) -> Any:
            types = [{"label": f"Type{i}" * 20, "count": i} for i in range(400)]
            ev = toolkit.ledger.add(tool="status", kind="status", content=str(types))
            return Status(n=ev.n, entity_types=types, hint="Graph is ready.")

        toolkit.call = call  # type: ignore[method-assign]
        text, _ = await server.call_tool("status", {})
        assert json.loads(text)["truncated"] and len(toolkit.ledger) == 0

    async def test_a_reply_that_fits_keeps_its_evidence(self, server: GraphRAGMCPServer):
        from pydantic import BaseModel

        class Status(BaseModel):
            n: int
            hint: str | None = None

        toolkit = server.toolkit

        async def call(name: str, arguments: dict) -> Any:
            ev = toolkit.ledger.add(tool="status", kind="status", content="3 entities")
            return Status(n=ev.n, hint="Graph is ready.")

        toolkit.call = call  # type: ignore[method-assign]
        text, _ = await server.call_tool("status", {})
        assert "truncated" not in json.loads(text) and len(toolkit.ledger) == 1

    async def test_read_only_write_is_an_error(self):
        server = GraphRAGMCPServer(FakeRAG(), read_only=True)
        text, is_error = await server.call_tool("ingest", {"text": "x"})
        assert is_error and "read-only" in text

    async def test_results_are_json(self, server: GraphRAGMCPServer):
        text, is_error = await server.call_tool("check_citations", {"answer": "No sources."})
        data = json.loads(text)
        assert not is_error and data["reason"] == "no_citations" and "hint" in data

    async def test_sessions_have_separate_ledgers(self, server: GraphRAGMCPServer):
        class Session:
            pass

        a, b = Session(), Session()
        server._session_toolkit(a).ledger.add(tool="t", kind="k", content="fact")
        text, _ = await server.call_tool("check_citations", {"answer": "fact [1]"}, a)
        assert json.loads(text)["grounded"] is True
        text, _ = await server.call_tool("check_citations", {"answer": "fact [1]"}, b)
        assert json.loads(text)["grounded"] is False
        assert server._session_toolkit(a) is server._session_toolkit(a)
        assert server._session_toolkit(None) is server.toolkit

    def test_mcp2_sessions_key_on_their_connection(self, server: GraphRAGMCPServer):
        class Connection:
            def __init__(self, protocol_version, session_id=None):
                self.protocol_version = protocol_version
                self.session_id = session_id

        class Session:  # mcp 2.x: a new ServerSession per request
            def __init__(self, connection):
                self._connection = connection

        loop = Connection("2025-06-18")
        assert server._session_toolkit(Session(loop)) is server._session_toolkit(Session(loop))
        assert server._session_toolkit(Session(loop)) is not server.toolkit

        http_a = server._session_toolkit(Session(Connection("2025-06-18", "sid-1")))
        assert http_a is server._session_toolkit(Session(Connection("2025-06-18", "sid-1")))
        assert http_a is not server._session_toolkit(Session(Connection("2025-06-18", "sid-2")))

    def test_stateless_requests_share_the_server_ledger(self, server, monkeypatch):
        from graphrag_sdk.mcp import server as server_module

        monkeypatch.setattr(server_module, "_stateless_versions", lambda: frozenset({"next"}))

        class Connection:
            protocol_version = "next"
            session_id = None

        class Session:
            def __init__(self):
                self._connection = Connection()

        assert server._session_toolkit(Session()) is server.toolkit

    async def test_each_connection_scope_has_its_own_ledger(self, server, monkeypatch):
        """SSE wraps each client stream in a scope, so stateless clients never share."""
        from graphrag_sdk.mcp import server as server_module

        monkeypatch.setattr(server_module, "_stateless_versions", lambda: frozenset({"next"}))

        class Connection:
            protocol_version = "next"
            session_id = None

        class Session:
            def __init__(self):
                self._connection = Connection()

        async def handler() -> Any:
            return server._session_toolkit(Session())

        async def request() -> Any:
            # The server runs each request in a task; tasks copy the context.
            return await asyncio.create_task(handler())

        with server.connection_scope() as first:
            assert await request() is first
            assert await request() is first
        with server.connection_scope() as second:
            assert await request() is second
        assert first is not second and server.toolkit not in (first, second)
        assert server._session_toolkit(Session()) is server.toolkit


class TestFitJson:
    def test_small_results_unchanged(self):
        data = {"rows": [1, 2, 3], "hint": None}
        assert json.loads(fit_json(data)) == data

    def test_longest_list_is_halved_until_it_fits(self):
        data = {"rows": [{"text": "x" * 100} for _ in range(400)], "columns": ["a"]}
        out = json.loads(fit_json(data, max_chars=5000))
        assert out["truncated"] is True and "narrow" in out["hint"]
        assert 0 < len(out["rows"]) < 400 and out["columns"] == ["a"]
        assert len(data["rows"]) == 400  # input untouched

    def test_existing_hint_is_kept(self):
        data = {"rows": ["y" * 100] * 200, "hint": "More rows matched."}
        out = json.loads(fit_json(data, max_chars=2000))
        assert out["hint"].startswith("More rows matched.")

    def test_long_strings_are_shortened(self):
        out = fit_json({"text": "z" * 100_000}, max_chars=5000)
        assert len(out) <= 5000 and json.loads(out)["truncated"] is True

    def test_nested_big_list_gives_valid_json_within_budget(self):
        # A nested result: the big list sits under data, not at the top.
        data = {
            "name": "report",
            "n": 1,
            "data": {
                "entity": "acme",
                "impacted": [
                    {"entity": f"e{i}", "distance": 2, "score": 0.5, "via": ["acme", f"e{i}"]}
                    for i in range(3000)
                ],
            },
        }
        for budget in (40_000, 5_000, 1_000):
            text = fit_json(data, max_chars=budget)
            assert len(text) <= budget
            out = json.loads(text)
            assert out["truncated"] is True and "data.impacted (was 3000 items)" in out["hint"]
            assert 0 < len(out["data"]["impacted"]) < 3000
            assert out["data"]["impacted"][0]["entity"] == "e0"
            assert out["name"] == "report" and out["n"] == 1
        assert len(data["data"]["impacted"]) == 3000  # input untouched

    def test_unshrinkable_result_keeps_its_short_fields(self):
        data = {"name": "x", "data": {f"k{i}": i for i in range(5000)}}
        text = fit_json(data, max_chars=1000)
        out = json.loads(text)
        assert len(text) <= 1000 and out["truncated"] is True and out["name"] == "x"


class TestProtocol:
    async def test_round_trip(self, server: GraphRAGMCPServer, monkeypatch):
        used: list[Any] = []
        pick = server._session_toolkit

        def spy(session: Any) -> Any:
            used.append(pick(session))
            return used[-1]

        monkeypatch.setattr(server, "_session_toolkit", spy)
        async with connect(server) as (client, instructions):
            listed = await client.list_tools()
            assert [t.name for t in listed.tools] == list(TOOL_NAMES)
            if instructions is not None:
                assert instructions == GUIDE

            rejected = await client.call_tool("get_ontology", {"extra": 1})
            assert rejected.is_error if hasattr(rejected, "is_error") else rejected.isError
            assert "Extra inputs are not permitted" in _text(rejected)

            checked = await client.call_tool("check_citations", {"answer": "x [3]"})
            assert json.loads(_text(checked))["invalid"] == [3]

            failed = await client.call_tool("ingest", {"text": "x"})
            assert "internal error (RuntimeError)" in _text(failed)
            assert "boom" not in _text(failed)

        # One client keeps one evidence ledger across its calls.
        assert len(used) == 3 and len({id(t) for t in used}) == 1
        assert used[0].ledger.items == []
        assert server._rag._graph_store.queries == []


class TestAuth:
    async def run_app(self, app, headers):
        sent: list[dict] = []

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.request"}

        await app({"type": "http", "headers": headers}, receive, send)
        return sent

    async def test_bearer_guard(self):
        reached = []

        async def app(scope, receive, send):
            reached.append(scope)

        guarded = bearer_token_guard(app, "s3cret")
        denied = await self.run_app(guarded, [(b"authorization", b"Bearer wrong")])
        assert denied[0]["status"] == 401 and not reached
        assert (await self.run_app(guarded, []))[0]["status"] == 401
        await self.run_app(guarded, [(b"authorization", b"Bearer s3cret")])
        assert len(reached) == 1

    async def test_bearer_scheme_is_case_insensitive(self):
        reached = []

        async def app(scope, receive, send):
            reached.append(scope)

        guarded = bearer_token_guard(app, "s3cret")
        for scheme in (b"bearer", b"BEARER"):
            await self.run_app(guarded, [(b"authorization", scheme + b" " + b"s3cret")])
        denied = await self.run_app(guarded, [(b"authorization", b"bearer " + b"wrong")])
        assert len(reached) == 2 and denied[0]["status"] == 401

    async def test_guard_passes_non_http_scopes(self):
        reached = []

        async def app(scope, receive, send):
            reached.append(scope["type"])

        await bearer_token_guard(app, "t")({"type": "lifespan"}, None, None)
        assert reached == ["lifespan"]

    def test_guard_needs_a_token(self):
        with pytest.raises(ValueError):
            bearer_token_guard(lambda *a: None, "")  # type: ignore[arg-type]

    def test_sse_binding_rules(self):
        check_sse_binding("127.0.0.1", None)
        check_sse_binding("0.0.0.0", "token")
        with pytest.raises(ValueError, match="without an auth token"):
            check_sse_binding("0.0.0.0", None)


class TestCLI:
    def test_build_rag_uses_the_right_constructor_arguments(self, monkeypatch):
        import graphrag_sdk.api.main as api_main
        import graphrag_sdk.core.providers as providers
        from graphrag_sdk.mcp import __main__ as cli

        seen: dict[str, Any] = {}
        monkeypatch.setenv("GRAPHRAG_MODEL", "openai/gpt-4o")
        monkeypatch.setenv("GRAPHRAG_EMBED_DIM", "3072")
        monkeypatch.setattr(api_main, "GraphRAG", lambda **kw: seen.update(kw) or "RAG")
        rag = cli._build_rag()
        assert rag == "RAG"
        assert isinstance(seen["llm"], providers.LiteLLM)
        assert seen["llm"].model_name == "openai/gpt-4o"
        assert seen["embedder"].model_name == "text-embedding-3-small"
        assert seen["embedding_dimension"] == 3072
        assert seen["embedder"]._extra["dimensions"] == 3072

    def test_embedder_dimensions_are_left_to_the_model_by_default(self, monkeypatch):
        import graphrag_sdk.api.main as api_main
        from graphrag_sdk.mcp import __main__ as cli

        seen: dict[str, Any] = {}
        monkeypatch.delenv("GRAPHRAG_EMBED_DIM", raising=False)
        monkeypatch.setattr(api_main, "GraphRAG", lambda **kw: seen.update(kw) or "RAG")
        cli._build_rag()
        assert seen["embedding_dimension"] == 1536
        assert "dimensions" not in seen["embedder"]._extra

    def test_embedder_dimensions_are_not_sent_to_older_models(self, monkeypatch):
        import graphrag_sdk.api.main as api_main
        from graphrag_sdk.mcp import __main__ as cli

        seen: dict[str, Any] = {}
        monkeypatch.setenv("GRAPHRAG_EMBED_MODEL", "text-embedding-ada-002")
        monkeypatch.setenv("GRAPHRAG_EMBED_DIM", "1536")
        monkeypatch.delenv("GRAPHRAG_EMBED_SEND_DIMENSIONS", raising=False)
        monkeypatch.setattr(api_main, "GraphRAG", lambda **kw: seen.update(kw) or "RAG")
        cli._build_rag()
        assert seen["embedding_dimension"] == 1536
        assert "dimensions" not in seen["embedder"]._extra

    def test_large_embedding_model_defaults_to_its_native_dimension(self, monkeypatch):
        import graphrag_sdk.api.main as api_main
        from graphrag_sdk.mcp import __main__ as cli

        seen: dict[str, Any] = {}
        monkeypatch.setenv("GRAPHRAG_EMBED_MODEL", "text-embedding-3-large")
        monkeypatch.delenv("GRAPHRAG_EMBED_DIM", raising=False)
        monkeypatch.delenv("GRAPHRAG_EMBED_SEND_DIMENSIONS", raising=False)
        monkeypatch.setattr(api_main, "GraphRAG", lambda **kw: seen.update(kw) or "RAG")
        cli._build_rag()
        assert seen["embedding_dimension"] == 3072
        assert "dimensions" not in seen["embedder"]._extra

    def test_embedder_dimensions_can_be_sent_to_an_aliased_model(self, monkeypatch):
        import graphrag_sdk.api.main as api_main
        from graphrag_sdk.mcp import __main__ as cli

        seen: dict[str, Any] = {}
        monkeypatch.setenv("GRAPHRAG_EMBED_MODEL", "azure/my-embedding-deployment")
        monkeypatch.setenv("GRAPHRAG_EMBED_DIM", "256")
        monkeypatch.setenv("GRAPHRAG_EMBED_SEND_DIMENSIONS", "true")
        monkeypatch.setattr(api_main, "GraphRAG", lambda **kw: seen.update(kw) or "RAG")
        cli._build_rag()
        assert seen["embedding_dimension"] == 256
        assert seen["embedder"]._extra["dimensions"] == 256

    @pytest.mark.parametrize(
        ("env", "argv", "expected"),
        [
            ("TRUE ", [], True),
            ("on", [], True),
            ("y", [], True),
            ("0", [], False),
            ("", [], False),
            ("", ["--read-only"], True),
            ("yes", ["--no-read-only"], False),
        ],
    )
    def test_read_only_flag_and_environment(self, monkeypatch, env, argv, expected):
        from graphrag_sdk.mcp import __main__ as cli

        seen: dict[str, Any] = {}

        class FakeServer:
            def __init__(self, rag: Any, *, auth_token: Any, read_only: bool) -> None:
                seen["read_only"] = read_only

        def fake_run(coro: Any) -> None:
            coro.close()

        monkeypatch.setenv("GRAPHRAG_MCP_READ_ONLY", env)
        monkeypatch.setattr("sys.argv", ["graphrag-mcp", *argv])
        monkeypatch.setattr(cli, "_build_rag", lambda: "RAG")
        monkeypatch.setattr("graphrag_sdk.mcp.server.GraphRAGMCPServer", FakeServer)
        monkeypatch.setattr(cli.asyncio, "run", fake_run)
        cli.main()
        assert seen["read_only"] is expected


class TestSSESecurity:
    def test_loopback_server_gets_dns_rebinding_protection(self):
        pytest.importorskip("mcp.server.transport_security")
        from graphrag_sdk.mcp.server import sse_security_kwargs

        local = sse_security_kwargs("127.0.0.1", 8080)["security_settings"]
        assert local.enable_dns_rebinding_protection is True
        assert local.allowed_hosts == ["127.0.0.1:8080", "localhost:8080"]
        assert "http://localhost:8080" in local.allowed_origins
        assert "https://127.0.0.1:8080" in local.allowed_origins

    def test_ipv6_loopback_is_bracketed(self):
        pytest.importorskip("mcp.server.transport_security")
        from graphrag_sdk.mcp.server import sse_security_kwargs

        hosts = sse_security_kwargs("::1", 1)["security_settings"].allowed_hosts
        assert hosts[1] == "[::1]:1"

    def test_non_loopback_relies_on_the_bearer_token(self):
        from graphrag_sdk.mcp.server import sse_security_kwargs

        # Off loopback the bearer token protects the server; clients use any name.
        assert sse_security_kwargs("0.0.0.0", 8080) == {}


class TestSSEConnections:
    """Each SSE stream is its own connection, even for stateless mcp 2.x clients."""

    @asynccontextmanager
    async def serve(self, server: GraphRAGMCPServer):
        import socket

        uvicorn = pytest.importorskip("uvicorn")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        lowlevel = server._build_server()
        app = server._sse_app(
            lowlevel, lowlevel.create_initialization_options(), host="127.0.0.1", port=port
        )
        # log_config=None: uvicorn's loggers propagate, so caplog sees ASGI errors.
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None)
        http = uvicorn.Server(config)
        task = asyncio.create_task(http.serve())
        try:
            while not http.started:
                await asyncio.sleep(0.01)
            yield f"http://127.0.0.1:{port}"
        finally:
            http.should_exit = True
            await task

    @pytest.fixture
    def counted(self, server: GraphRAGMCPServer, monkeypatch) -> GraphRAGMCPServer:
        from graphrag_sdk.tools import graph_ops as ops

        async def graph_counts(store):
            return {"entities": 3, "relations": 0, "documents": 1, "chunks": 1}

        async def schema_counts(store):
            return {"Person": 3}, {}

        async def graph_exists(store):
            return True

        monkeypatch.setattr(ops, "graph_counts", graph_counts)
        monkeypatch.setattr(ops, "schema_counts", schema_counts)
        monkeypatch.setattr(ops, "graph_exists", graph_exists)
        return server

    async def test_clients_do_not_share_evidence(self, counted: GraphRAGMCPServer, caplog):
        from mcp import Client
        from mcp.client.session import ClientSession
        from mcp.client.sse import sse_client

        answer = {"answer": "The graph has 3 Person entities [1]."}
        async with self.serve(counted) as base:
            async with Client(sse_client(f"{base}/sse")) as a:
                async with Client(sse_client(f"{base}/sse")) as b:
                    assert json.loads(_text(await a.call_tool("status", {})))["n"] == 1
                    mine = json.loads(_text(await a.call_tool("check_citations", answer)))
                    theirs = json.loads(_text(await b.call_tool("check_citations", answer)))
            assert mine["grounded"] is True
            assert theirs["grounded"] is False and theirs["invalid"] == [1]
            # A client that opens with the classic initialize handshake works too.
            async with sse_client(f"{base}/sse") as (read, write):
                async with ClientSession(read, write) as c:
                    await c.initialize()
                    assert json.loads(_text(await c.call_tool("status", {})))["n"] == 1
            await asyncio.sleep(0.1)  # let the closed streams' handlers finish
        assert counted.toolkit.ledger.items == []
        assert not [r for r in caplog.records if r.exc_info or r.levelname == "ERROR"]

    async def test_a_rejected_host_is_answered_without_a_traceback(self, server, caplog):
        async with self.serve(server) as base:
            host, port = base.removeprefix("http://").split(":")
            with caplog.at_level("DEBUG"):
                reader, writer = await asyncio.open_connection(host, int(port))
                request = b"GET /sse HTTP/1.1\r\nHost: evil.example\r\nConnection: close\r\n\r\n"
                writer.write(request)
                await writer.drain()
                status_line = await reader.readline()
                writer.close()
                await writer.wait_closed()
            await asyncio.sleep(0.1)  # let the handler finish
        assert status_line.split()[1] == b"421"
        assert not [r for r in caplog.records if r.exc_info or r.levelname == "ERROR"]


def test_the_all_extra_installs_the_mcp_server():
    from pathlib import Path

    tomllib = pytest.importorskip("tomllib")
    extras = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    assert set(extras["mcp"]) <= set(extras["all"])
    # CI installs the dev extra; without mcp these tests would be skipped.
    mcp_spec = next(r for r in extras["mcp"] if r.startswith("mcp"))
    assert mcp_spec in extras["dev"]
