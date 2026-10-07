"""Tests for the agentic graph tools: Cypher guard, search, query_graph, lookup, traverse."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    LLMResponse,
    Ontology,
    Relation,
    RetrieverResult,
    RetrieverResultItem,
    ToolCall,
)
from graphrag_sdk.retrieval.agentic import (
    AgenticRetrieval,
    GraphSchema,
    ToolContext,
    build_default_registry,
    enforce_row_cap,
    validate_read_query,
)
from graphrag_sdk.retrieval.agentic.cypher_guard import introspect_schema, mask_internal_labels
from graphrag_sdk.retrieval.agentic.graph_tools import (
    format_value,
    make_lookup_entity_tool,
    make_query_graph_tool,
    make_search_tool,
    make_traverse_tool,
    split_search_items,
)


def result(rows: list[list[Any]], header: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(result_set=rows, header=[[1, h] for h in (header or [])])


class FakeGraphStore:
    """Answers query_raw by the first matching (substring, responder) rule."""

    def __init__(self, rules: list[tuple[str, Callable[[str, dict], Any]]] | None = None):
        self.rules = rules or []
        self.calls: list[dict[str, Any]] = []

    async def query_raw(self, cypher, params=None, *, read_only=False, timeout=None):
        self.calls.append(
            {"cypher": cypher, "params": params or {}, "read_only": read_only, "timeout": timeout}
        )
        for needle, responder in self.rules:
            if needle in cypher:
                out = responder(cypher, params or {})
                if isinstance(out, Exception):
                    raise out
                return out
        return result([])


SCHEMA_RULES = [
    ("db.labels()", lambda c, p: result([["__Entity__"], ["Person"], ["Company"], ["Chunk"]])),
    ("db.relationshipTypes()", lambda c, p: result([["RELATES"], ["MENTIONED_IN"]])),
    ("db.propertyKeys()", lambda c, p: result([["id"], ["name"], ["description"], ["age"]])),
]

ONTOLOGY = Ontology(relations=[Relation(label="WORKS_AT")])


class QueueLLM:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(content=self.replies.pop(0) if self.replies else "")


def tctx() -> ToolContext:
    return ToolContext(ctx=Context(latency_budget_ms=10_000.0))


# ── Guard ────────────────────────────────────────────────────────

SCHEMA = GraphSchema(
    labels=["__Entity__", "Person", "Company"],
    relationship_types=["RELATES", "MENTIONED_IN"],
    property_keys=["id", "name", "age"],
    semantic_relations=["WORKS_AT"],
)


class TestGuard:
    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (p:Person)-[r:RELATES]->(c:Company) WHERE r.rel_type = 'WORKS_AT' "
            "RETURN c.name, count(p) AS n",
            "MATCH (p:Person) WHERE p.name = 'call center: delete' RETURN p.name",
            "MATCH (n) RETURN [x IN collect(n) WHERE x:Person | x.name] AS names",
            "MATCH (n)-[r:RELATES]-(m) WHERE r.rel_type IN ['WORKS_AT'] RETURN m.name",
            "MATCH (n) RETURN n {.name, .age} AS m",
            "UNWIND [1, 2] AS x RETURN x",
        ],
    )
    def test_accepts_valid_reads(self, cypher):
        assert validate_read_query(cypher, SCHEMA) == []

    @pytest.mark.parametrize(
        ("cypher", "fragment"),
        [
            ("CREATE (n:Person) RETURN n", "must start with"),
            ("MATCH (n) DETACH DELETE n", "'DETACH' is not allowed"),
            ("MATCH (n) SET n.age = 1 RETURN n", "'SET' is not allowed"),
            ("MATCH (n) CALL db.labels() YIELD label RETURN label", "'CALL' is not allowed"),
            ("MATCH (n) RETURN n; MATCH (m) RETURN m", "one statement"),
            ("MATCH (n:Person)", "RETURN clause"),
            ("MATCH (n) RETURN n.embedding", "embedding"),
            ("MATCH (n) RETURN n.description_embedding", "embedding"),
            ("MATCH (c:__GraphRAGConfig__) RETURN c", "internal"),
            ("MATCH (p:Robot) RETURN p.name", "unknown label or relationship type 'Robot'"),
            ("MATCH (p:Person) RETURN p.salary", "unknown property 'salary'"),
            ("MATCH (p:Person)-[r:RELATES]->(c) RETURN c.name", "must constrain r.rel_type"),
            ("MATCH (p)-[:RELATES]->(c) RETURN c.name", "must bind a variable"),
            ("MATCH (p)-[:WORKS_AT]->(c) RETURN c.name", "stored as RELATES"),
            (
                "MATCH (p)-[r:RELATES]->(c) WHERE r.rel_type = 'OWNS' RETURN c.name",
                "unknown relation 'OWNS'",
            ),
        ],
    )
    def test_rejects(self, cypher, fragment):
        errors = validate_read_query(cypher, SCHEMA)
        assert any(fragment in e for e in errors), errors

    def test_without_schema_only_structural_checks_apply(self):
        assert validate_read_query("MATCH (p:Robot) RETURN p.salary") == []
        assert validate_read_query("MATCH (n) DELETE n")

    def test_row_cap(self):
        assert enforce_row_cap("MATCH (n) WITH n LIMIT 100000 RETURN n LIMIT 5", 25) == (
            "MATCH (n) WITH n LIMIT 25 RETURN n LIMIT 5"
        )
        assert enforce_row_cap("MATCH (n) RETURN n;", 10) == "MATCH (n) RETURN n\nLIMIT 10"
        assert "shortestPath" not in enforce_row_cap(
            "MATCH p = shortestPath((a)-[*]-(b)) RETURN p", 5
        )
        with pytest.raises(ValueError):
            enforce_row_cap("MATCH (n) RETURN n", 0)

    def test_mask_and_format(self):
        assert mask_internal_labels("(e:__Entity__)") == "(e:[internal])"
        node = SimpleNamespace(labels=["Person"], properties={"name": "A", "embedding": [0.1]})
        assert format_value(node) == {"labels": ["Person"], "name": "A"}
        assert format_value({"name": "A", "name_embedding": [1]}) == {"name": "A"}

    async def test_introspection(self):
        store = FakeGraphStore(SCHEMA_RULES)
        schema = await introspect_schema(store, ONTOLOGY)
        assert schema.labels == ["__Entity__", "Person", "Company", "Chunk"]
        assert schema.semantic_relations == ["WORKS_AT"]
        assert all(c["read_only"] for c in store.calls)

    async def test_introspection_failure_leaves_list_empty(self):
        store = FakeGraphStore([("db.labels()", lambda c, p: RuntimeError("down"))])
        schema = await introspect_schema(store)
        assert schema.labels == [] and schema.relationship_types == []


# ── search ───────────────────────────────────────────────────────


class SectionStrategy:
    async def search(self, query: str, ctx: Context, **kwargs: Any) -> RetrieverResult:
        return RetrieverResult(
            items=[
                RetrieverResultItem(
                    content="Answer format: a number.", metadata={"section": "hint"}
                ),
                RetrieverResultItem(
                    content="## Key Entities\n- Alice: an engineer\n- Acme: a company",
                    metadata={"section": "entities"},
                ),
                RetrieverResultItem(
                    content="## Knowledge Graph Facts\n- Alice works at Acme",
                    metadata={"section": "facts"},
                ),
                RetrieverResultItem(
                    content="## Source Document Passages\n[Source: hr.pdf]\nAlice joined Acme in "
                    "2020.\n---\nAcme is in Berlin.",
                    metadata={"section": "passages"},
                ),
            ]
        )


class TestSearch:
    def test_split_search_items(self):
        items = asyncio.run(SectionStrategy().search("q", Context())).items
        assert split_search_items(items) == [
            ("entity", "Alice: an engineer", ""),
            ("entity", "Acme: a company", ""),
            ("fact", "Alice works at Acme", ""),
            ("passage", "Alice joined Acme in 2020.", "hr.pdf"),
            ("passage", "Acme is in Berlin.", ""),
        ]

    async def test_numbers_continue_across_calls(self):
        tool = make_search_tool(SectionStrategy())
        state = tctx()
        first = await tool.handler({"query": "alice"}, state)
        second = await tool.handler({"query": "acme"}, state)
        assert first.content.splitlines()[0] == "[1] Alice: an engineer"
        assert "[4] (source: hr.pdf) Alice joined Acme in 2020." in first.content
        assert second.content.splitlines()[0].startswith("[6] ")
        assert len(state.evidence) == 10
        assert state.evidence.get(4).source == "hr.pdf"

    async def test_no_results(self):
        class Empty:
            async def search(self, q, ctx, **kw):
                return RetrieverResult(items=[])

        out = await make_search_tool(Empty()).handler({"query": "x"}, tctx())
        assert out.content.startswith("No results")


# ── query_graph ──────────────────────────────────────────────────

GOOD = (
    "```cypher\nMATCH (p:Person)-[r:RELATES]->(c:Company) WHERE r.rel_type = 'WORKS_AT' "
    "RETURN c.name AS company, count(p) AS people LIMIT 1000\n```"
)


class TestQueryGraph:
    def store(self, rows=None, exc=None):
        def answer(c, p):
            return exc if exc is not None else result(rows or [], ["company", "people"])

        return FakeGraphStore([*SCHEMA_RULES, ("MATCH (p:Person)", answer)])

    async def test_runs_a_valid_query_read_only_and_capped(self):
        store = self.store([["Acme", 3], ["__Entity__ Corp", 1]])
        llm = QueueLLM([GOOD])
        tool = make_query_graph_tool(store, llm, ontology_getter=lambda: ONTOLOGY)
        state = tctx()
        out = await tool.handler({"question": "how many people per company?"}, state)

        run = store.calls[-1]
        assert run["read_only"] is True and run["timeout"] == 5000
        assert "LIMIT 25" in run["cypher"] and "LIMIT 1000" not in run["cypher"]
        assert out.content.startswith("[1] Graph query result for: how many people per company?")
        assert "Columns: company, people" in out.content
        assert "Acme | 3" in out.content and "[internal] Corp | 1" in out.content
        assert state.evidence.get(1).kind == "graph_query"
        assert state.state["generated_cypher"][0]["rows"] == 2
        assert "WORKS_AT" in llm.prompts[0] and "Person" in llm.prompts[0]

    async def test_retries_with_the_rejection_reason(self):
        store = self.store([["Acme", 3]])
        llm = QueueLLM(["```cypher\nMATCH (n) DELETE n\n```", GOOD])
        out = await make_query_graph_tool(store, llm).handler({"question": "q"}, tctx())
        assert out.status == "ok"
        assert "previous attempt was rejected" in llm.prompts[1]
        assert "'DELETE' is not allowed" in llm.prompts[1]

    async def test_database_error_is_retried(self):
        calls = {"n": 0}

        def flaky(c, p):
            calls["n"] += 1
            return RuntimeError("syntax error") if calls["n"] == 1 else result([["Acme", 1]])

        store = FakeGraphStore([*SCHEMA_RULES, ("MATCH (p:Person)", flaky)])
        out = await make_query_graph_tool(store, QueueLLM([GOOD, GOOD])).handler(
            {"question": "q"}, tctx()
        )
        assert out.status == "ok" and calls["n"] == 2

    async def test_gives_up_with_a_refusal_and_logs_the_last_query(self, ctx):
        store = self.store([])
        llm = QueueLLM(["```cypher\nMATCH (p:Robot) RETURN p.name\n```"] * 3)
        registry = build_default_registry(graph_store=store, llm=llm, include_skills=False)
        state = ToolContext(ctx=ctx)
        out = await registry.run("query_graph", {"question": "q"}, state)
        assert out.status == "refused"
        assert "no valid graph query" in out.content and "Robot" in out.content
        assert state.state["generated_cypher"][0]["error"]
        assert len(llm.prompts) == 3

    async def test_schema_is_introspected_once_per_run(self):
        store = self.store([["Acme", 1]])
        tool = make_query_graph_tool(store, QueueLLM([GOOD, GOOD]))
        state = tctx()
        await tool.handler({"question": "a"}, state)
        await tool.handler({"question": "b"}, state)
        assert sum("db.labels()" in c["cypher"] for c in store.calls) == 1


# ── lookup_entity / traverse ─────────────────────────────────────


def lookup_rows(c, p):
    if "CONTAINS" in c:
        return result([["control_board_cx10", "Control Board CX-10", "Part", "A board"]])
    return result([])


class TestLookup:
    async def test_lists_candidates(self):
        store = FakeGraphStore([("CONTAINS", lookup_rows)])
        out = await make_lookup_entity_tool(store).handler({"name": "CX-10 board"}, tctx())
        assert '"Control Board CX-10" (Part, id control_board_cx10): A board' in out.content
        params = store.calls[0]["params"]
        assert params["needle"] == "cx-10 board" and "board" in params["tokens"]

    async def test_no_match(self):
        out = await make_lookup_entity_tool(FakeGraphStore()).handler({"name": "zzz"}, tctx())
        assert out.content == "No entity name matches 'zzz'."


def walk_store(neighbours: dict[str, list[tuple[str, str, str]]], **extra) -> FakeGraphStore:
    def resolve(c, p):
        return result([[p["k"], p["k"].title()]]) if p["k"] in neighbours else result([])

    def expand(c, p):
        rows = []
        for s in p["frontier"]:
            for t, name, rel in neighbours.get(s, []):
                rows.append([s, t, name, rel, True])
        return result(rows[: p["remaining"]])

    return FakeGraphStore(
        [
            ("WHERE e.id = $k OR e.name = $k", resolve),
            ("s.id IN $frontier", expand),
            *extra.get("rules", []),
            ("CONTAINS", lookup_rows),
        ]
    )


GRAPH = {
    "alice": [("acme", "Acme", "WORKS_AT"), ("bob", "Bob", "KNOWS")],
    "acme": [("alice", "Alice", "WORKS_AT"), ("berlin", "Berlin", "LOCATED_IN")],
    "bob": [("alice", "Alice", "KNOWS")],
    "berlin": [],
}


class TestTraverse:
    async def test_breadth_first_walk(self):
        store = walk_store(GRAPH)
        out = await make_traverse_tool(store).handler({"start": "alice", "max_depth": 3}, tctx())
        assert out.status == "ok"
        # Berlin has no further neighbours, so the third hop ends the walk.
        assert "reached 3 entities; stopped: complete" in out.content
        assert out.data["nodes"][-1]["id"] == "berlin" and out.data["nodes"][-1]["depth"] == 2
        expand_call = next(c for c in store.calls if "$frontier" in c["cypher"])
        assert expand_call["read_only"] and "SAME_AS" in expand_call["params"]["structural"]

    async def test_depth_and_expansion_budgets(self):
        out = await make_traverse_tool(walk_store(GRAPH)).handler(
            {"start": "alice", "max_depth": 1}, tctx()
        )
        assert "stopped: depth_budget" in out.content
        out = await make_traverse_tool(walk_store(GRAPH)).handler(
            {"start": "alice", "max_expansions": 1}, tctx()
        )
        assert "stopped: expansion_budget" in out.content and len(out.data["nodes"]) == 1

    async def test_bounds_are_clamped(self):
        store = walk_store(GRAPH)
        await make_traverse_tool(store).handler(
            {"start": "alice", "max_expansions": 10_000, "max_depth": 99}, tctx()
        )
        expand_call = next(c for c in store.calls if "$frontier" in c["cypher"])
        assert expand_call["params"]["remaining"] == 200

    async def test_unknown_start_is_refused_with_suggestions(self, ctx):
        registry = build_default_registry(graph_store=walk_store(GRAPH), include_skills=False)
        out = await registry.run("traverse", {"start": "CX-10"}, ctx)
        assert out.status == "refused"
        assert "no entity is called 'CX-10'" in out.content
        assert '"Control Board CX-10"' in out.content

    async def test_ambiguous_start_is_refused(self, ctx):
        store = FakeGraphStore(
            [
                (
                    "WHERE e.id = $k OR e.name = $k",
                    lambda c, p: result([["a1", "Apex"], ["a2", "Apex"]]),
                )
            ]
        )
        registry = build_default_registry(graph_store=store, include_skills=False)
        out = await registry.run("traverse", {"start": "Apex"}, ctx)
        assert out.status == "refused" and "matches 2 entities" in out.content

    async def test_case_insensitive_fallback(self):
        store = FakeGraphStore(
            [
                (
                    "toLower(toString(e.name)) = toLower($k)",
                    lambda c, p: result([["alice", "Alice"]]),
                ),
                ("s.id IN $frontier", lambda c, p: result([])),
            ]
        )
        out = await make_traverse_tool(store).handler({"start": "ALICE"}, tctx())
        assert "Walk from 'Alice' (id alice)" in out.content

    async def test_typed_path(self):
        path = ("RETURN DISTINCT n1.id", lambda c, p: result([["berlin", "Berlin"]]))
        store = walk_store(GRAPH, rules=[path])
        out = await make_traverse_tool(store).handler(
            {"start": "alice", "via": ["WORKS_AT", "LOCATED_IN"]}, tctx()
        )
        call = next(c for c in store.calls if "RETURN DISTINCT" in c["cypher"])
        assert call["params"] == {"start": "alice", "t0": "WORKS_AT", "t1": "LOCATED_IN"}
        assert "WORKS_AT" not in call["cypher"]  # relation names are parameters
        assert (
            "along WORKS_AT -> LOCATED_IN" in out.content and "- Berlin (id berlin)" in out.content
        )

    async def test_too_many_hops_refused(self, ctx):
        registry = build_default_registry(graph_store=walk_store(GRAPH), include_skills=False)
        out = await registry.run("traverse", {"start": "alice", "via": ["A"] * 6}, ctx)
        assert out.status == "refused" and "at most 5" in out.content


# ── Registry / loop wiring ───────────────────────────────────────


class TestWiring:
    def test_default_registry_tools(self):
        reg = build_default_registry(
            strategy=SectionStrategy(), graph_store=FakeGraphStore(), llm=QueueLLM([])
        )
        assert reg.names()[:4] == ["search", "query_graph", "traverse", "lookup_entity"]
        assert "entity_comparison" in reg.names()
        assert "cypher" not in reg.names()

    def test_no_query_graph_without_llm(self):
        reg = build_default_registry(graph_store=FakeGraphStore(), include_skills=False)
        assert reg.names() == ["traverse", "lookup_entity"]

    async def test_loop_exposes_evidence_and_generated_cypher(self, ctx):
        store = FakeGraphStore(
            [
                *SCHEMA_RULES,
                ("MATCH (p:Person)", lambda c, p: result([["Acme", 2]], ["company", "n"])),
            ]
        )

        class LLM:
            supports_tool_calling = True
            model_name = "m"

            def __init__(self):
                self.turn = 0

            async def ainvoke(self, prompt, **kw):
                return LLMResponse(content=GOOD)

            async def ainvoke_with_tools(self, messages, tools, **kw):
                self.turn += 1
                if self.turn == 1:
                    return LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(id="c1", name="query_graph", arguments={"question": "q"})
                        ],
                    )
                return LLMResponse(content="Acme has 2 people [1].")

        agent = AgenticRetrieval(LLM(), graph_store=store, ontology=ONTOLOGY)
        out = await agent.search("how many people work at Acme?", ctx)
        assert out.metadata["evidence"][0]["kind"] == "graph_query"
        assert out.metadata["generated_cypher"][0]["rows"] == 1

    def test_set_ontology_reaches_query_graph_and_inner_strategy(self):
        inner = MagicMock()
        agent = AgenticRetrieval(QueueLLM([]), strategy=inner, graph_store=FakeGraphStore())
        agent.set_ontology(ONTOLOGY)
        inner.set_ontology.assert_called_once_with(ONTOLOGY)
        assert agent._ontology is ONTOLOGY


# ── Storage read-only plumbing ───────────────────────────────────


class TestReadOnlyPlumbing:
    async def test_graph_store_forwards_read_only(self):
        from graphrag_sdk.storage.graph_store import GraphStore

        conn = MagicMock()
        conn.query = AsyncMock(return_value="r")
        store = GraphStore(conn)
        await store.query_raw("MATCH (n) RETURN n")
        conn.query.assert_awaited_with("MATCH (n) RETURN n", None)
        await store.query_raw("MATCH (n) RETURN n", {"a": 1}, read_only=True, timeout=50)
        conn.query.assert_awaited_with("MATCH (n) RETURN n", {"a": 1}, timeout=50, read_only=True)

    async def test_connection_uses_ro_query(self):
        from graphrag_sdk.core.connection import FalkorDBConnection

        conn = FalkorDBConnection.__new__(FalkorDBConnection)
        graph = MagicMock()
        graph.query = AsyncMock(return_value="rw")
        graph.ro_query = AsyncMock(return_value="ro")
        conn._graph = graph
        conn._ensure_client = lambda: None  # type: ignore[method-assign]
        breaker = MagicMock()
        breaker.allow_request = AsyncMock(return_value=True)
        breaker.record_success = AsyncMock()
        conn._breaker = breaker
        conn.config = SimpleNamespace(query_timeout_ms=1000, retry_count=1, retry_delay=0)
        assert await conn.query("MATCH (n) RETURN n", read_only=True) == "ro"
        assert await conn.query("MATCH (n) RETURN n") == "rw"
        graph.ro_query.assert_awaited_once()
