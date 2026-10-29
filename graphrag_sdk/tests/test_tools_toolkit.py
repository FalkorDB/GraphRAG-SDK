"""Unit tests for GraphRAGToolkit: dispatch, gates, sessions, evidence and
tool behaviour over a fake GraphRAG. The same tools run against a real
FalkorDB in test_tools_integration.py."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from graphrag_sdk.core.connection import ConnectionConfig
from graphrag_sdk.core.exceptions import ConfigError, DatabaseError, ReadOnlyViolation
from graphrag_sdk.core.models import (
    DocumentInfo,
    FinalizeResult,
    IngestionResult,
    RetrieverResult,
    ScoredPath,
)
from graphrag_sdk.tools import TOOL_NAMES, GraphRAGToolkit, ToolError
from graphrag_sdk.tools import graph_ops as ops
from graphrag_sdk.tools.cypher_guard import GraphSchema


class FakeResult:
    def __init__(self, rows: list[list[Any]], header: list[str] | None = None):
        self.result_set = rows
        self.header = [[1, h] for h in (header or [])]


class FakeStore:
    def __init__(self, *, rows=None, header=None, error: Exception | None = None):
        self.rows = rows or []
        self.header = header or []
        self.error = error
        self.queries: list[tuple[str, dict]] = []
        self.neighbors: Any = {}
        self.neighbor_calls: list[tuple[str, bool]] = []

    async def query_raw(self, cypher: str, params: dict | None = None, **kwargs: Any):
        self.queries.append((cypher, params or {}))
        if self.error is not None:
            raise self.error
        return FakeResult(self.rows, self.header)

    async def pagerank(self, **kwargs: Any):
        return {}

    async def weighted_neighbors(
        self, node_id: str, *, limit: int = 50, with_direction: bool = False
    ):
        self.neighbor_calls.append((node_id, with_direction))
        if isinstance(self.neighbors, Exception):
            raise self.neighbors
        found = self.neighbors.get(node_id, [])
        return found if with_direction else [t[:3] for t in found]


class FakeRAG:
    def __init__(self, store: FakeStore | None = None):
        self._graph_store = store or FakeStore()
        self._conn = SimpleNamespace(topology_version=1)
        self._vector_store = None
        self._global_ontology = None
        self.embedder = None
        self.llm = None
        self.ingested: list[dict] = []
        self.finalized = 0
        self.closed = False
        self.retrieve_result = RetrieverResult(items=[], metadata={})

    async def get_ontology(self):
        raise RuntimeError("no ontology in this fake")

    async def ingest(self, **kwargs: Any) -> IngestionResult:
        self.ingested.append(kwargs)
        return IngestionResult(
            document_info=DocumentInfo(uid=kwargs.get("document_id") or "doc-1"),
            nodes_created=3,
            relationships_created=2,
            chunks_indexed=1,
        )

    async def finalize(self) -> FinalizeResult:
        self.finalized += 1
        return FinalizeResult(entities_deduplicated=1, entities_embedded=2)

    async def retrieve(self, question: str, **kwargs: Any) -> RetrieverResult:
        self.retrieve_kwargs = kwargs
        return self.retrieve_result

    async def close(self) -> None:
        self.closed = True


async def _const(value: Any) -> Any:
    return value


SCHEMA = GraphSchema(
    labels=["Person", "Company", "Chunk", "Document"],
    relationship_types=["RELATES", "MENTIONED_IN", "PART_OF"],
    property_keys=["id", "name", "description", "rel_type", "fact", "text"],
    semantic_relations=["WORKS_AT", "FOUNDED"],
)


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def rag(store: FakeStore) -> FakeRAG:
    return FakeRAG(store)


@pytest.fixture
def toolkit(rag: FakeRAG, monkeypatch: pytest.MonkeyPatch) -> GraphRAGToolkit:
    tk = GraphRAGToolkit(rag)  # type: ignore[arg-type]

    async def schema() -> GraphSchema:
        return SCHEMA

    monkeypatch.setattr(tk, "_schema", schema)
    return tk


# ── Construction, specs, dispatch ──────────────────────────────────


class TestSpecsAndDispatch:
    def test_all_tools_advertised(self, toolkit: GraphRAGToolkit):
        assert [s.name for s in toolkit.tool_specs()] == list(TOOL_NAMES)
        assert len(TOOL_NAMES) == 11

    def test_read_only_drops_writes(self, rag: FakeRAG):
        tk = GraphRAGToolkit(rag, read_only=True)  # type: ignore[arg-type]
        names = [s.name for s in tk.tool_specs()]
        assert "ingest" not in names and "finalize" not in names
        assert "cypher_query" in names

    def test_finalize_only_advertised_when_manual(self, rag: FakeRAG):
        for policy in ("on_write", "never"):
            tk = GraphRAGToolkit(rag, finalize_policy=policy)  # type: ignore[arg-type]
            assert "finalize" not in [s.name for s in tk.tool_specs()]

    def test_include_and_validation(self, rag: FakeRAG):
        tk = GraphRAGToolkit(rag, include=["status", "lookup_entity"])  # type: ignore[arg-type]
        assert [s.name for s in tk.tool_specs()] == ["status", "lookup_entity"]
        with pytest.raises(ValueError, match="Unknown tool names"):
            GraphRAGToolkit(rag, include=["status", "graph_search"])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="finalize_policy"):
            GraphRAGToolkit(rag, finalize_policy="always")  # type: ignore[arg-type]

    async def test_call_unknown_tool(self, toolkit: GraphRAGToolkit):
        with pytest.raises(ToolError, match="Unknown tool 'graph_search'") as exc:
            await toolkit.call("graph_search", {})
        assert "lookup_entity" in (exc.value.hint or "")
        assert "Hint:" in str(exc.value)

    async def test_call_disabled_tool(self, rag: FakeRAG):
        tk = GraphRAGToolkit(rag, include=["status"])  # type: ignore[arg-type]
        with pytest.raises(ToolError, match="not enabled"):
            await tk.call("lookup_entity", {"name": "x"})

    async def test_call_write_on_read_only(self, rag: FakeRAG):
        tk = GraphRAGToolkit(rag, read_only=True)  # type: ignore[arg-type]
        with pytest.raises(ReadOnlyViolation):
            await tk.call("ingest", {"text": "hello"})
        with pytest.raises(ReadOnlyViolation):
            await tk.ingest("hello")
        assert rag.ingested == []

    async def test_call_rejects_bad_arguments(self, toolkit: GraphRAGToolkit):
        with pytest.raises(ToolError, match="bogus: Extra inputs are not permitted"):
            await toolkit.call("lookup_entity", {"name": "x", "bogus": 1})
        with pytest.raises(ToolError, match="name"):
            await toolkit.call("lookup_entity", {})
        with pytest.raises(ToolError, match="max_rows"):
            await toolkit.call("cypher_query", {"cypher": "MATCH (n) RETURN n", "max_rows": 101})

    async def test_call_dispatches(self, toolkit: GraphRAGToolkit, monkeypatch):
        async def find(store, name, *, limit):
            return [{"id": "e1", "name": name, "label": "Person", "description": ""}]

        monkeypatch.setattr(ops, "find_entities", find)
        result = await toolkit.call("lookup_entity", {"name": "Alice"})
        assert result.matches[0].id == "e1"


# ── Sessions and lifecycle ─────────────────────────────────────────


class TestSessions:
    async def test_new_session_has_its_own_ledger(self, toolkit: GraphRAGToolkit):
        toolkit.ledger.add(tool="t", kind="k", content="x")
        other = toolkit.new_session()
        assert len(toolkit.ledger.items) == 1
        assert other.ledger.items == []
        other.ledger.add(tool="t", kind="k", content="y")
        assert toolkit.ledger.get(1).content == "x"
        assert other.ledger.get(1).content == "y"

    async def test_new_session_shares_schema_cache(self, rag: FakeRAG):
        tk = GraphRAGToolkit(rag)  # type: ignore[arg-type]
        assert tk.new_session()._shared is tk._shared

    async def test_aclose_only_when_owned(self, rag: FakeRAG):
        await GraphRAGToolkit(rag).aclose()  # type: ignore[arg-type]
        assert not rag.closed
        owner = GraphRAGToolkit(rag, owns_rag=True)  # type: ignore[arg-type]
        await owner.new_session().aclose()
        assert not rag.closed
        async with owner:
            pass
        assert rag.closed

    def test_for_tenant_names_the_graph(self):
        tk = GraphRAGToolkit.for_tenant(
            ConnectionConfig(graph_name="kb"),
            "acme_1",
            llm=SimpleNamespace(),  # type: ignore[arg-type]
            embedder=SimpleNamespace(),  # type: ignore[arg-type]
            read_only=True,
        )
        assert tk.rag._conn.config.graph_name == "kb__acme_1"
        assert tk._owns_rag and tk._read_only and tk._tenant_id == "acme_1"

    @pytest.mark.parametrize("bad", ["", "a b", "x" * 65, "../g", "a:b"])
    def test_for_tenant_rejects_bad_ids(self, bad: str):
        with pytest.raises(ValueError, match="tenant_id"):
            GraphRAGToolkit.for_tenant(
                ConnectionConfig(),
                bad,
                llm=SimpleNamespace(),  # type: ignore[arg-type]
                embedder=SimpleNamespace(),  # type: ignore[arg-type]
            )


# ── cypher_query ───────────────────────────────────────────────────


class TestCypherQuery:
    @pytest.mark.parametrize(
        "cypher",
        [
            "CREATE (n:Person {name: 'x'}) RETURN n",
            "MATCH (n) DETACH DELETE n",
            "MATCH (n) SET n.name = 'x' RETURN n",
            "CALL db.labels()",
            "MATCH (n) RETURN n; MATCH (m) DELETE m",
            "MATCH (n) // comment\nMERGE (m) RETURN m",
        ],
    )
    async def test_writes_rejected_before_the_database(self, rag: FakeRAG, cypher: str):
        tk = GraphRAGToolkit(rag)  # type: ignore[arg-type]
        with pytest.raises(ToolError, match="rejected"):
            await tk.cypher_query(cypher)
        assert rag._graph_store.queries == []

    async def test_unknown_names_rejected_with_schema_hint(self, toolkit, store):
        with pytest.raises(ToolError, match="Employee") as exc:
            await toolkit.cypher_query("MATCH (n:Employee) RETURN n.name")
        hint = exc.value.hint or ""
        assert "Person" in hint and "WORKS_AT" in hint
        assert "Chunk" not in hint.split("Relations")[0]
        # Only the read-only existence check reached the database.
        assert store.queries == [("RETURN 1", {})]

    async def test_rows_are_evidence_and_limit_is_capped(self, toolkit, store):
        store.rows = [["Alice", 3], ["Bob", 1]]
        store.header = ["name", "n"]
        result = await toolkit.cypher_query(
            "MATCH (p:Person) RETURN p.name AS name, 1 AS n LIMIT 500", max_rows=5
        )
        assert result.columns == ["name", "n"]
        assert [r.values for r in result.rows] == [["Alice", 3], ["Bob", 1]]
        assert [r.n for r in result.rows] == [1, 2]
        assert not result.truncated and result.row_count == 2
        cypher, _ = store.queries[-1]
        assert "LIMIT 6" in cypher and "500" not in cypher
        await toolkit.cypher_query(
            "MATCH (p:Person) RETURN p.name AS name LIMIT $k", {"k": 100000}, max_rows=5
        )
        cypher, params = store.queries[-1]
        assert cypher.endswith("LIMIT 6") and params == {"k": 100000}
        assert toolkit.ledger.get(1).tool == "cypher_query"
        assert '"name": "Alice"' in toolkit.ledger.get(1).content

    async def test_truncated_when_more_rows_than_max(self, toolkit, store):
        store.rows = [[i] for i in range(4)]
        store.header = ["i"]
        result = await toolkit.cypher_query("MATCH (p:Person) RETURN p.name", max_rows=3)
        assert result.truncated and result.row_count == 3
        assert "do not count" in (result.hint or "")

    async def test_inner_limit_is_kept_so_aggregates_stay_correct(self, toolkit, store):
        store.rows = [[100]]
        store.header = ["c"]
        result = await toolkit.cypher_query(
            "MATCH (n:Person) WITH n ORDER BY n.name LIMIT 100 RETURN count(n) AS c"
        )
        cypher, _ = store.queries[-1]
        assert "LIMIT 100 RETURN" in cypher and cypher.endswith("LIMIT 26")
        assert result.rows[0].values == [100] and not result.truncated

    async def test_parameter_limit_above_the_cap_is_clamped(self, toolkit, store):
        store.rows = [[i] for i in range(50)]
        store.header = ["i"]
        result = await toolkit.cypher_query(
            "MATCH (p:Person) RETURN p.name LIMIT $n", {"n": 50}, max_rows=10
        )
        cypher, params = store.queries[-1]
        assert cypher.endswith("LIMIT 11") and cypher.count("LIMIT") == 1
        assert params == {"n": 50}
        assert result.row_count == 10 and result.truncated

    async def test_untyped_relates_is_allowed(self, toolkit, store):
        await toolkit.cypher_query("MATCH (a:Person)-[r:RELATES]->(b) RETURN r.rel_type, count(*)")
        assert store.queries

    async def test_unknown_rel_type_is_rejected(self, toolkit):
        with pytest.raises(ToolError, match="EMPLOYED_BY"):
            await toolkit.cypher_query(
                "MATCH (a)-[r:RELATES {rel_type: 'EMPLOYED_BY'}]->(b) RETURN b.name"
            )

    async def test_database_errors_become_tool_errors(self, toolkit, store):
        store.error = DatabaseError("Type mismatch: expected Integer")
        with pytest.raises(ToolError, match="Type mismatch") as exc:
            await toolkit.cypher_query("MATCH (p:Person) RETURN p.name")
        assert "Entity labels" in (exc.value.hint or "")

    async def test_timeouts_become_tool_errors(self, toolkit, store):
        store.error = asyncio.TimeoutError()
        with pytest.raises(ToolError, match="timed out") as exc:
            await toolkit.cypher_query("MATCH (p:Person) RETURN p.name", timeout_ms=100)
        assert "aggregate" in (exc.value.hint or "")

    async def test_vectors_never_returned(self, toolkit, store):
        store.rows = [[{"name": "Alice", "embedding": [0.1] * 128}, [0.2] * 128]]
        store.header = ["p", "v"]
        result = await toolkit.cypher_query("MATCH (p:Person) RETURN p.name AS p, 1 AS v")
        text = str(result.rows[0].values)
        assert "0.1" not in text and "0.2" not in text

    async def test_empty_hint(self, toolkit):
        result = await toolkit.cypher_query("MATCH (p:Person) RETURN p.name")
        assert result.rows == [] and "lookup_entity" in (result.hint or "")


# ── Other read tools ───────────────────────────────────────────────


class TestReadTools:
    async def test_status_hints_empty_graph(self, toolkit, monkeypatch):
        async def counts(store):
            return {"entities": 0, "relations": 0, "documents": 0, "chunks": 0}

        async def schema_counts(store):
            return {}, {}

        monkeypatch.setattr(ops, "graph_counts", counts)
        monkeypatch.setattr(ops, "schema_counts", schema_counts)
        result = await toolkit.status()
        assert result.entities == 0 and "empty" in (result.hint or "")

    async def test_lookup_entity_is_not_evidence(self, toolkit, monkeypatch):
        async def find(store, name, *, limit):
            return []

        monkeypatch.setattr(ops, "find_entities", find)
        result = await toolkit.lookup_entity("Nobody")
        assert result.matches == [] and "retrieve" in (result.hint or "")
        assert toolkit.ledger.items == []

    async def test_lookup_failure_is_a_tool_error(self, toolkit, monkeypatch):
        async def find(store, name, *, limit):
            return None

        monkeypatch.setattr(ops, "find_entities", find)
        with pytest.raises(ToolError, match="lookup failed"):
            await toolkit.lookup_entity("x")

    async def test_traverse_edges_are_evidence(self, toolkit, monkeypatch):
        async def resolve(store, ref, deadline):
            return "a", "Alice"

        async def expand(store, sid, sname, depth, exp, remaining):
            nodes = [{"id": "c", "name": "Acme", "depth": 1, "relation": "WORKS_AT"}]
            edges = [
                {
                    "source": "Alice",
                    "source_id": "a",
                    "relation": "WORKS_AT",
                    "target": "Acme",
                    "target_id": "c",
                    "fact": "Alice joined Acme in 2020",
                }
            ]
            return nodes, edges, "expansion_budget"

        monkeypatch.setattr(ops, "resolve_start", resolve)
        monkeypatch.setattr(ops, "expand", expand)
        result = await toolkit.traverse("alice")
        assert result.start == "Alice" and result.edges[0].n == 1
        assert result.truncated and "max_expansions" in (result.hint or "")
        expected = "Alice -[WORKS_AT]-> Acme: Alice joined Acme in 2020"
        assert toolkit.ledger.get(1).content == expected

    async def test_traverse_via_resolves_case_and_rejects_unknown(self, toolkit, monkeypatch):
        seen = {}

        async def resolve(store, ref, deadline):
            return "a", "Alice"

        async def typed_path(store, sid, path, ms):
            seen["path"] = path
            return [{"id": "c", "name": "Acme", "depth": 1, "relation": "WORKS_AT"}], "complete"

        monkeypatch.setattr(ops, "resolve_start", resolve)
        monkeypatch.setattr(ops, "typed_path", typed_path)
        result = await toolkit.traverse("Alice", via=["works_at"])
        assert seen["path"] == ["WORKS_AT"] and result.nodes[0].n == 1
        with pytest.raises(ToolError, match="Unknown relation") as exc:
            await toolkit.traverse("Alice", via=["EMPLOYS"])
        assert "WORKS_AT" in (exc.value.hint or "")

    async def test_traverse_unknown_start_is_a_tool_error(self, toolkit, monkeypatch):
        async def resolve(store, ref, deadline):
            raise ToolError("No entity is called 'Zed'.", hint="Use lookup_entity.")

        monkeypatch.setattr(ops, "resolve_start", resolve)
        with pytest.raises(ToolError, match="Zed"):
            await toolkit.call("traverse", {"start": "Zed"})

    async def test_graph_walk_paths_are_evidence(self, toolkit, monkeypatch):
        async def resolve(store, ref, deadline):
            return ref.lower(), ref

        async def names(store, ids):
            return {"alice": "Alice", "acme": "Acme"}

        async def beam(self, start, goal=None, ctx=None):
            return [
                ScoredPath(
                    nodes=["alice", "acme"], edges=["WORKS_AT"], edge_forward=[True], score=0.75
                )
            ]

        monkeypatch.setattr(ops, "resolve_start", resolve)
        monkeypatch.setattr(ops, "entity_names", names)
        monkeypatch.setattr("graphrag_sdk.tools.toolkit.DynamicGraphWalk.beam_search", beam)
        result = await toolkit.graph_walk("Alice")
        assert result.paths[0].nodes == ["Alice", "Acme"] and result.paths[0].n == 1
        assert toolkit.ledger.get(1).content == "Alice -[WORKS_AT]-> Acme"

    async def test_graph_walk_draws_edges_in_their_stored_direction(
        self, toolkit, store, monkeypatch
    ):
        # Stored: alice-[WORKS_AT]->acme, acme-[IN]->berlin.
        store.neighbors = {
            "acme": [("alice", 1.0, "WORKS_AT", False)],
            "alice": [("acme", 1.0, "WORKS_AT", True)],
            "berlin": [("acme", 1.0, "IN", False)],
        }
        store.neighbors["acme"].append(("berlin", 1.0, "IN", True))

        async def resolve(store, ref, deadline):
            return ref.lower(), ref

        async def names(store, ids):
            return {"alice": "Alice", "acme": "Acme", "berlin": "Berlin"}

        monkeypatch.setattr(ops, "resolve_start", resolve)
        monkeypatch.setattr(ops, "entity_names", names)
        out = await toolkit.graph_walk("Acme", max_depth=1)
        texts = {toolkit.ledger.get(p.n).content for p in out.paths}
        assert texts == {"Acme <-[WORKS_AT]- Alice", "Acme -[IN]-> Berlin"}
        assert all(with_dir for _, with_dir in store.neighbor_calls)
        goal = await toolkit.graph_walk("Berlin", goal="Alice", max_depth=2)
        assert toolkit.ledger.get(goal.paths[0].n).content == (
            "Berlin <-[IN]- Acme <-[WORKS_AT]- Alice"
        )
        assert "Berlin <-[IN]- Acme <-[WORKS_AT]- Alice" in goal.to_llm_text()

    async def test_graph_walk_goal_depth_is_the_whole_path(self, toolkit, store, monkeypatch):
        store.neighbors = {
            "a": [("b", 1.0, "R", True)],
            "b": [("a", 1.0, "R", False), ("c", 1.0, "R", True)],
            "c": [("b", 1.0, "R", False)],
        }

        async def resolve(store, ref, deadline):
            return ref, ref

        async def names(store, ids):
            return {}

        monkeypatch.setattr(ops, "resolve_start", resolve)
        monkeypatch.setattr(ops, "entity_names", names)
        out = await toolkit.graph_walk("a", goal="c", max_depth=1)
        assert out.paths == [] and not out.truncated
        assert "at most 1 hop " in out.hint
        two = await toolkit.graph_walk("a", goal="c", max_depth=2)
        assert two.paths[0].nodes == ["a", "b", "c"]

    async def test_graph_walk_reports_a_search_cut_short(self, toolkit, store, monkeypatch):
        import graphrag_sdk.tools.toolkit as tk_mod

        store.neighbors = {"a": [("b", 1.0, "R", True)], "b": [("c", 1.0, "R", True)]}

        async def resolve(store, ref, deadline):
            return ref, ref

        async def names(store, ids):
            return {}

        monkeypatch.setattr(ops, "resolve_start", resolve)
        monkeypatch.setattr(ops, "entity_names", names)
        monkeypatch.setattr(tk_mod, "_WALK_BUDGET_SECONDS", 0.0)
        out = await toolkit.graph_walk("a", goal="z", max_depth=4)
        assert out.truncated and out.paths == []
        assert "time budget" in out.hint

    async def test_traverse_via_with_no_time_left_sends_no_query(self, toolkit, monkeypatch):
        import graphrag_sdk.tools.toolkit as tk_mod

        calls = []
        clock = iter([100.0, 200.0, 200.0, 200.0])

        async def resolve(store, ref, deadline):
            return "a", "Alice"

        async def typed_path(store, sid, path, ms):
            calls.append(ms)
            return [], "complete"

        monkeypatch.setattr(ops, "resolve_start", resolve)
        monkeypatch.setattr(ops, "typed_path", typed_path)
        monkeypatch.setattr(tk_mod.time, "monotonic", lambda: next(clock, 200.0))
        result = await toolkit.traverse("Alice", via=["WORKS_AT"], timeout_ms=1000)
        assert calls == [] and result.stopped == "time_budget" and result.truncated
        assert "timed out" in (result.hint or "")

    async def test_graph_walk_slow_beam_search_is_a_tool_error(self, toolkit, monkeypatch):
        import graphrag_sdk.tools.toolkit as tk_mod

        async def resolve(store, ref, deadline):
            return ref, ref

        async def beam(self, start, goal=None, ctx=None):
            await asyncio.sleep(5)
            return []

        monkeypatch.setattr(ops, "resolve_start", resolve)
        monkeypatch.setattr("graphrag_sdk.tools.toolkit.DynamicGraphWalk.beam_search", beam)
        monkeypatch.setattr(tk_mod, "_WALK_BUDGET_SECONDS", 0.05)
        with pytest.raises(ToolError, match="time budget") as exc:
            await toolkit.graph_walk("a")
        assert "beam_width" in (exc.value.hint or "")

    async def test_graph_walk_slow_pagerank_falls_back_to_no_weights(
        self, toolkit, store, monkeypatch
    ):
        import graphrag_sdk.tools.toolkit as tk_mod

        store.neighbors = {"a": [("b", 1.0, "R", True)]}

        async def resolve(store, ref, deadline):
            return ref, ref

        async def names(store, ids):
            return {}

        async def slow_pagerank(**kwargs):
            await asyncio.sleep(5)
            return {"a": 1.0}

        monkeypatch.setattr(ops, "resolve_start", resolve)
        monkeypatch.setattr(ops, "entity_names", names)
        monkeypatch.setattr(store, "pagerank", slow_pagerank)
        monkeypatch.setattr(tk_mod, "_WALK_BUDGET_SECONDS", 0.05)
        out = await asyncio.wait_for(toolkit.graph_walk("a", max_depth=1), 2)
        assert [p.nodes for p in out.paths] == [["a", "b"]]

    async def test_graph_walk_database_errors_are_not_hidden(self, toolkit, store, monkeypatch):
        store.neighbors = RuntimeError("connection refused")

        async def resolve(store, ref, deadline):
            return ref, ref

        monkeypatch.setattr(ops, "resolve_start", resolve)
        with pytest.raises(RuntimeError, match="connection refused"):
            await toolkit.graph_walk("a")

    async def test_retrieve_numbers_every_item(self, toolkit, rag, monkeypatch):
        rag.retrieve_result = RetrieverResult(
            items=[],
            metadata={
                "provenance": {
                    "entities": [{"id": "a", "name": "Alice", "description": "an engineer"}],
                    "facts": ["Alice works at Acme", " "],
                    "chunks": [{"id": "c1", "text": "Alice works at Acme.", "document_path": "p"}],
                }
            },
        )

        async def enrich(store, ids):
            return {
                "a": {
                    "id": "a",
                    "name": "Alice",
                    "label": "Person",
                    "description": "an engineer",
                    "properties": {},
                }
            }

        async def triples(store, ids, *, hops, cap):
            return [{"source": "Alice", "type": "WORKS_AT", "target": "Acme", "fact": ""}]

        async def docs(store, ids):
            return {"c1": ("d1", "/docs/a.txt")}

        monkeypatch.setattr(ops, "enrich_entities", enrich)
        monkeypatch.setattr(ops, "expand_triples", triples)
        monkeypatch.setattr(ops, "chunk_documents", docs)
        result = await toolkit.retrieve("Where does Alice work?")
        assert [result.entities[0].n, result.relations[0].n, result.facts[0].n] == [1, 2, 3]
        assert len(result.facts) == 1
        assert result.chunks[0].n == 4 and result.chunks[0].document_path == "/docs/a.txt"
        assert result.documents[0].document_id == "d1"
        assert toolkit.ledger.get(4).source == "/docs/a.txt"

    async def test_retrieve_trims_repeated_text(self, toolkit, rag, monkeypatch):
        rag.retrieve_result = RetrieverResult(
            items=[],
            metadata={
                "provenance": {
                    "entities": [{"id": "a", "name": "Alice"}],
                    "facts": [
                        "Alice —[WORKS_AT]→ Acme: (Alice, WORKS_AT, Acme): since 2020",
                        "Acme —[BASED_IN]→ Paris: (Acme, BASED_IN, Paris): head office",
                        "Acme —[OWNS]→ Lab: (Acme, OWNS, Lab)",
                    ],
                    "chunks": [{"id": "c1", "text": "Alice works at Acme."}],
                }
            },
        )

        async def enrich(store, ids):
            return {}

        async def triples(store, ids, *, hops, cap):
            return [{"source": "alice", "type": "WORKS_AT", "target": "ACME", "fact": "since 2020"}]

        async def docs(store, ids):
            return {"c1": ("/docs/a.txt", "/docs/a.txt")}

        monkeypatch.setattr(ops, "enrich_entities", enrich)
        monkeypatch.setattr(ops, "expand_triples", triples)
        monkeypatch.setattr(ops, "chunk_documents", docs)
        result = await toolkit.retrieve("Where does Alice work?")
        assert [f.text for f in result.facts] == [
            "Acme —[BASED_IN]→ Paris: head office",
            "Acme —[OWNS]→ Lab",
        ]
        dumped = result.chunks[0].model_dump(exclude_none=True)
        assert dumped["document_id"] == "/docs/a.txt" and "document_path" not in dumped
        assert toolkit.ledger.get(result.chunks[0].n).source == "/docs/a.txt"
        assert "[/docs/a.txt]" in result.to_llm_text()

    async def test_retrieve_nothing_found_hint(self, toolkit, monkeypatch):
        async def none(*a, **k):
            return {}

        async def no_triples(*a, **k):
            return []

        monkeypatch.setattr(ops, "enrich_entities", none)
        monkeypatch.setattr(ops, "expand_triples", no_triples)
        monkeypatch.setattr(ops, "chunk_documents", none)
        result = await toolkit.retrieve("anything")
        assert "lookup_entity" in (result.hint or "")

    @pytest.fixture
    def found(self, rag, monkeypatch):
        """One entity, relation, fact and passage; records whether links ran."""
        rag.retrieve_result = RetrieverResult(
            items=[],
            metadata={
                "provenance": {
                    "entities": [{"id": "a", "name": "Alice", "description": ""}],
                    "facts": ["Alice works at Acme"],
                    "chunks": [{"id": "c1", "text": "Alice works at Acme.", "document_path": "p"}],
                }
            },
        )
        walked: list[list[str]] = []

        async def triples(store, ids, *, hops, cap):
            walked.append(ids)
            return [{"source": "Alice", "type": "WORKS_AT", "target": "Acme", "fact": ""}]

        monkeypatch.setattr(ops, "enrich_entities", lambda *a, **k: _const({}))
        monkeypatch.setattr(ops, "expand_triples", triples)
        monkeypatch.setattr(ops, "chunk_documents", lambda *a, **k: _const({}))
        return walked

    @staticmethod
    async def _plan(rag) -> list[str]:
        strategy = rag.retrieve_kwargs["strategy"]
        return await strategy._router.plan("", None)

    async def test_retrieve_runs_every_path_by_default(self, toolkit, rag, found):
        result = await toolkit.retrieve("Where does Alice work?")
        # MultiPath's own expansion is skipped: retrieve walks the links itself.
        assert await self._plan(rag) == ["chunks", "entity_cypher", "entity_fulltext", "relates"]
        assert found == [["a"]] and result.relations and result.chunks

    async def test_retrieve_runs_only_the_paths_named(self, toolkit, rag, found):
        result = await toolkit.call("retrieve", {"question": "Who?", "paths": ["passages"]})
        assert await self._plan(rag) == ["chunks"]
        assert found == [] and result.relations == []

    async def test_links_alone_still_finds_entities_to_walk_from(self, toolkit, rag, found):
        await toolkit.retrieve("Who?", paths=["links"])
        assert await self._plan(rag) == ["entity_cypher"]
        assert found == [["a"]]

    async def test_include_chunks_false_skips_the_passages_path(self, toolkit, rag, found):
        result = await toolkit.retrieve("Who?", include_chunks=False)
        assert "chunks" not in await self._plan(rag)
        assert result.chunks == []

    async def test_passages_only_without_chunks_is_refused(self, toolkit, found):
        with pytest.raises(ToolError, match="include_chunks") as exc:
            await toolkit.retrieve("Who?", paths=["passages"], include_chunks=False)
        assert "Drop include_chunks" in (exc.value.hint or "")

    async def test_unknown_path_is_refused(self, toolkit):
        with pytest.raises(ToolError, match="paths"):
            await toolkit.call("retrieve", {"question": "Who?", "paths": ["vectors"]})

    async def test_nothing_found_on_chosen_paths_says_retry_without(self, toolkit, monkeypatch):
        monkeypatch.setattr(ops, "enrich_entities", lambda *a, **k: _const({}))
        monkeypatch.setattr(ops, "expand_triples", lambda *a, **k: _const([]))
        monkeypatch.setattr(ops, "chunk_documents", lambda *a, **k: _const({}))
        result = await toolkit.retrieve("anything", paths=["facts"])
        assert "without paths" in (result.hint or "")

    async def test_retrieve_on_an_uninitialised_rag_writes_no_ontology(
        self, toolkit, rag, monkeypatch
    ):
        """GraphRAG.retrieve would register the default types first; the tool must not."""
        from graphrag_sdk.core.models import EntityType, Ontology

        declared = Ontology(entities=[EntityType(label="Person")])
        rag._ontology_initialized = False
        rag.ontology = None
        rag._ontology_store = SimpleNamespace(load=lambda: _const(declared))
        validated: list[bool] = []

        async def validate(*, ctx=None):
            validated.append(True)

        async def facade_retrieve(*a, **k):
            raise AssertionError("GraphRAG.retrieve initialises (writes) the ontology")

        rag._validate_graph_config = validate
        rag.retrieve = facade_retrieve
        seen: dict[str, Any] = {}

        class Strategy:
            def __init__(self, ontology):
                seen["ontology"] = ontology

            async def search(self, question, ctx):
                seen["question"] = question
                return RetrieverResult(items=[], metadata={})

        monkeypatch.setattr(
            toolkit, "_make_strategy", lambda top_k, onto=None, plan=None: Strategy(onto)
        )
        for name in ("enrich_entities", "chunk_documents"):
            monkeypatch.setattr(ops, name, lambda *a, **k: _const({}))
        monkeypatch.setattr(ops, "expand_triples", lambda *a, **k: _const([]))
        await toolkit.retrieve("Who is Alice?")
        assert validated == [True]
        assert seen == {"ontology": declared, "question": "Who is Alice?"}


# ── check_citations ────────────────────────────────────────────────


class TestMissingGraph:
    """A graph that was never written: read tools say so instead of
    answering "nothing found", and never create it."""

    @pytest.fixture
    def missing(self) -> GraphRAGToolkit:
        store = FakeStore(error=DatabaseError("Invalid graph operation on empty key"))
        rag = FakeRAG(store)
        rag._conn = SimpleNamespace(topology_version=1, config=SimpleNamespace(graph_name="nope"))
        return GraphRAGToolkit(rag)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "tool, args",
        [
            ("cypher_query", {"cypher": "MATCH (n) RETURN count(n) AS c"}),
            ("retrieve", {"question": "Who founded Acme?"}),
            ("lookup_entity", {"name": "Alice"}),
            ("traverse", {"start": "Alice"}),
            ("graph_walk", {"start": "Alice"}),
            ("run_skill", {"skill": "gap_analysis"}),
        ],
    )
    async def test_read_tools_name_the_missing_graph(self, missing, tool, args):
        with pytest.raises(ToolError) as info:
            await missing.call(tool, args)
        assert "Graph 'nope' does not exist" in str(info.value)
        assert "Ingest" in info.value.hint and "graph name" in info.value.hint
        # Only read-only queries ran.
        assert missing._store.queries and all(
            "CREATE" not in q and "MERGE" not in q for q, _ in missing._store.queries
        )

    async def test_status_says_the_graph_does_not_exist(self, missing):
        status = await missing.call("status", {})
        assert status.entities == 0 and "Graph 'nope' does not exist" in status.hint

    async def test_existence_is_checked_once(self, toolkit: GraphRAGToolkit, store: FakeStore):
        await toolkit.call("lookup_entity", {"name": "Alice"})
        await toolkit.call("lookup_entity", {"name": "Alice"})
        assert [q for q, _ in store.queries].count("RETURN 1") == 1

    async def test_other_database_errors_are_left_to_the_tool(self, rag: FakeRAG):
        rag._graph_store.error = DatabaseError("Unknown function 'foo'")
        tk = GraphRAGToolkit(rag)  # type: ignore[arg-type]
        with pytest.raises(ToolError, match="The query failed"):
            await tk.cypher_query("MATCH (n) RETURN n.name")


class TestCheckCitations:
    async def test_cited_invalid_and_missing(self, toolkit: GraphRAGToolkit):
        toolkit.ledger.add(tool="traverse", kind="relation", content="Alice -[WORKS_AT]-> Acme")
        ok = await toolkit.check_citations("Alice works at Acme [1].")
        assert ok.grounded and ok.reason == "cited" and ok.cited == [1]
        assert ok.sources[0].tool == "traverse" and ok.hint is None

        mixed = await toolkit.check_citations("Alice works at Acme [1][7].")
        assert mixed.grounded and mixed.invalid == [7] and "[7]" not in mixed.answer
        assert "removed" in (mixed.hint or "")

        bad = await toolkit.check_citations("Alice works at Acme [9].")
        assert not bad.grounded and bad.reason == "invalid_citations"

        none = await toolkit.check_citations("Alice works at Acme.")
        assert not none.grounded and none.reason == "no_citations" and "[n]" in (none.hint or "")

    async def test_uncited_and_unsupported_claims(self, toolkit: GraphRAGToolkit):
        toolkit.ledger.add(tool="traverse", kind="relation", content="Alice -[WORKS_AT]-> Acme")
        result = await toolkit.check_citations(
            "Alice works at Globex [1]. Alice joined Acme in 2019."
        )
        assert result.grounded and result.reason == "cited"
        assert result.uncited == ["Alice joined Acme in 2019."]
        assert [(c.sentence, c.missing) for c in result.unsupported] == [
            ("Alice works at Globex [1].", ["Globex"])
        ]
        assert "drop it" in (result.hint or "") and "correct the claim" in (result.hint or "")
        text = result.to_llm_text()
        assert "No citation (1):" in text and "missing: Globex" in text

    async def test_citations_are_checked_against_the_text_the_agent_saw(
        self, toolkit: GraphRAGToolkit, store: FakeStore
    ):
        # A row longer than one evidence item used to be cut at 2,000
        # characters, so a figure the agent read near its end was flagged.
        store.header = ["bio", "note"]
        store.rows = [["x" * 1500, "y" * 800 + " founded in 1987"]]
        out = await toolkit.cypher_query("MATCH (p:Person) RETURN p.name AS bio, p.name AS note")
        assert "1987" in json.dumps(out.rows[0].values)
        assert "1987" in toolkit.ledger.get(out.rows[0].n).content
        result = await toolkit.check_citations("Acme was founded in 1987 [1].")
        assert result.grounded and result.unsupported == []

    async def test_sessions_do_not_share_numbers(self, toolkit: GraphRAGToolkit):
        toolkit.ledger.add(tool="t", kind="k", content="x")
        other = toolkit.new_session()
        result = await other.check_citations("x [1]")
        assert not result.grounded and result.invalid == [1]


# ── Write tools ────────────────────────────────────────────────────


class TestWriteTools:
    async def test_ingest_manual_hints_finalize(self, toolkit, rag):
        result = await toolkit.call("ingest", {"text": "Alice works at Acme.", "document_id": "d"})
        assert result.document_id == "d" and result.nodes_created == 3
        assert not result.finalized and "finalize" in (result.hint or "")
        assert rag.finalized == 0 and rag.ingested[0]["text"] == "Alice works at Acme."

    async def test_ingest_on_write_finalizes(self, rag):
        tk = GraphRAGToolkit(rag, finalize_policy="on_write")  # type: ignore[arg-type]
        result = await tk.ingest("x")
        assert result.finalized and rag.finalized == 1
        assert (await tk.finalize()).finalized and rag.finalized == 1

    async def test_finalize_reports_counts(self, toolkit, rag):
        report = await toolkit.call("finalize", {})
        assert report.finalized and report.entities_deduplicated == 1
        assert report.entities_embedded == 2 and rag.finalized == 1

    async def test_finalize_never_policy(self, rag):
        tk = GraphRAGToolkit(rag, finalize_policy="never")  # type: ignore[arg-type]
        with pytest.raises(ConfigError):
            await tk.finalize()
        with pytest.raises(ToolError, match="not enabled"):
            await tk.call("finalize", {})
