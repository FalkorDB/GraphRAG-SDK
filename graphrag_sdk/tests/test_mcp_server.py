"""Tests for the MCP server: tool listing, per-session evidence, JSON
results, errors with hints, and a protocol round trip through an in-process
MCP client (mcp 1.x or 2.x)."""

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
from graphrag_sdk.mcp.server import TEXT_INGEST_GUIDE, bearer_token_guard, check_sse_binding
from graphrag_sdk.tools import TOOL_NAMES, ToolError

# Without a files root the server does not offer preview_files.
SERVED = [n for n in TOOL_NAMES if n != "preview_files"]

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
        from mcp.shared.memory import create_connected_server_and_client_session

        async with create_connected_server_and_client_session(lowlevel) as session:
            yield session, None
        return
    async with Client(lowlevel) as client:
        yield client, client.instructions


def _text(result: Any) -> str:
    return result.content[0].text


class TestToolList:
    def test_names_and_annotations(self, server: GraphRAGMCPServer):
        tools = {t["name"]: t for t in server.tool_dicts()}
        assert list(tools) == SERVED
        assert tools["cypher_query"]["annotations"]["readOnlyHint"] is True
        for write_tool in ("ingest", "finalize", "set_ontology"):
            assert tools[write_tool]["annotations"]["readOnlyHint"] is False
            assert tools[write_tool]["annotations"]["destructiveHint"] is True
        assert tools["cypher_query"]["annotations"]["destructiveHint"] is False
        for tool in tools.values():
            assert tool["inputSchema"]["type"] == "object"
            assert tool["inputSchema"].get("additionalProperties") is False

    def test_read_only_and_include(self):
        names = [t["name"] for t in GraphRAGMCPServer(FakeRAG(), read_only=True).tool_dicts()]
        assert "ingest" not in names and "finalize" not in names
        only = GraphRAGMCPServer(FakeRAG(), include=["status", "cypher_query"])
        assert [t["name"] for t in only.tool_dicts()] == ["status", "cypher_query"]

    def test_guide_is_the_default_instructions(self, server: GraphRAGMCPServer):
        assert server.instructions == GUIDE + TEXT_INGEST_GUIDE
        for word in ("lookup_entity", "cypher_query", "truncated", "check_citations", "[n]"):
            assert word in GUIDE
        assert GraphRAGMCPServer(FakeRAG(), instructions="x").instructions == "x"


def test_guide_says_tool_results_are_not_instructions():
    assert "not instructions" in GUIDE


class TestCallTool:
    async def test_tool_errors_carry_hints(self, server: GraphRAGMCPServer):
        text, is_error = await server.call_tool("graph_search", {})
        assert is_error and "Unknown tool" in text and "Hint:" in text
        text, is_error = await server.call_tool("lookup_entity", {"name": "x", "extra": 1})
        assert is_error and "Extra inputs are not permitted" in text
        text, is_error = await server.call_tool("cypher_query", {"cypher": "MATCH (n) DELETE n"})
        assert is_error and "rejected" in text and "Hint:" in text

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
        # A run_skill result: the big list sits under data, not at the top.
        data = {
            "skill": "impact_analysis",
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
            assert out["skill"] == "impact_analysis" and out["n"] == 1
        assert len(data["data"]["impacted"]) == 3000  # input untouched

    def test_unshrinkable_result_keeps_its_short_fields(self):
        data = {"skill": "x", "data": {f"k{i}": i for i in range(5000)}}
        text = fit_json(data, max_chars=1000)
        out = json.loads(text)
        assert len(text) <= 1000 and out["truncated"] is True and out["skill"] == "x"


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
            assert [t.name for t in listed.tools] == SERVED
            if instructions is not None:
                assert instructions == GUIDE + TEXT_INGEST_GUIDE

            rejected = await client.call_tool("cypher_query", {"cypher": "CREATE (n) RETURN n"})
            assert rejected.is_error if hasattr(rejected, "is_error") else rejected.isError
            assert "rejected" in _text(rejected)

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
        monkeypatch.setattr(api_main, "GraphRAG", lambda **kw: seen.update(kw) or "RAG")
        cli._build_rag()
        assert seen["embedding_dimension"] == 1536
        assert "dimensions" not in seen["embedder"]._extra

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
        import graphrag_sdk.mcp.server as server_mod
        from graphrag_sdk.mcp import __main__ as cli

        seen: dict[str, Any] = {}

        class FakeServer:
            def __init__(self, rag: Any, *, auth_token: Any, read_only: bool, **_: Any) -> None:
                seen["read_only"] = read_only

        def fake_run(coro: Any) -> None:
            coro.close()

        monkeypatch.setenv("GRAPHRAG_MCP_READ_ONLY", env)
        monkeypatch.setattr("sys.argv", ["graphrag-mcp", *argv])
        monkeypatch.setattr(cli, "_build_rag", lambda: "RAG")
        monkeypatch.setattr(server_mod, "GraphRAGMCPServer", FakeServer)
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
