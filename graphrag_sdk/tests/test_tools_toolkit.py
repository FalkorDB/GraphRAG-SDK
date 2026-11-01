"""Unit tests for GraphRAGToolkit: dispatch, gates, sessions, evidence and
tool behaviour over a fake GraphRAG. The same tools run against a real
FalkorDB in test_tools_integration.py."""

from __future__ import annotations

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
)
from graphrag_sdk.tools import TOOL_NAMES, GraphRAGToolkit, ToolError
from graphrag_sdk.tools import graph_ops as ops


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
        self.read_only: list[bool] = []

    async def query_raw(self, cypher: str, params: dict | None = None, **kwargs: Any):
        self.queries.append((cypher, params or {}))
        self.read_only.append(kwargs.get("read_only", False))
        if self.error is not None:
            raise self.error
        return FakeResult(self.rows, self.header)

    async def get_document_record(self, document_id: str):
        return None  # no stored document: text ingest is never refused


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


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def rag(store: FakeStore) -> FakeRAG:
    return FakeRAG(store)


@pytest.fixture
def toolkit(rag: FakeRAG) -> GraphRAGToolkit:
    return GraphRAGToolkit(rag)  # type: ignore[arg-type]


# ── Construction, specs, dispatch ──────────────────────────────────


class TestSpecsAndDispatch:
    def test_all_tools_advertised(self, toolkit: GraphRAGToolkit):
        # preview_files needs a files root, which this toolkit has none of.
        assert [s.name for s in toolkit.tool_specs()] == [
            n for n in TOOL_NAMES if n != "preview_files"
        ]
        assert len(TOOL_NAMES) == 8

    def test_read_only_drops_writes(self, rag: FakeRAG):
        tk = GraphRAGToolkit(rag, read_only=True)  # type: ignore[arg-type]
        names = [s.name for s in tk.tool_specs()]
        assert "ingest" not in names and "finalize" not in names
        assert "retrieve" in names

    def test_finalize_only_advertised_when_manual(self, rag: FakeRAG):
        for policy in ("on_write", "never"):
            tk = GraphRAGToolkit(rag, finalize_policy=policy)  # type: ignore[arg-type]
            assert "finalize" not in [s.name for s in tk.tool_specs()]

    def test_include_and_validation(self, rag: FakeRAG):
        tk = GraphRAGToolkit(rag, include=["status", "retrieve"])  # type: ignore[arg-type]
        assert [s.name for s in tk.tool_specs()] == ["status", "retrieve"]
        with pytest.raises(ValueError, match="Unknown tool names"):
            GraphRAGToolkit(rag, include=["status", "graph_search"])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="finalize_policy"):
            GraphRAGToolkit(rag, finalize_policy="always")  # type: ignore[arg-type]

    async def test_call_unknown_tool(self, toolkit: GraphRAGToolkit):
        with pytest.raises(ToolError, match="Unknown tool 'graph_search'") as exc:
            await toolkit.call("graph_search", {})
        assert "retrieve" in (exc.value.hint or "")
        assert "Hint:" in str(exc.value)

    async def test_call_disabled_tool(self, rag: FakeRAG):
        tk = GraphRAGToolkit(rag, include=["status"])  # type: ignore[arg-type]
        with pytest.raises(ToolError, match="not enabled"):
            await tk.call("retrieve", {"question": "Who?"})

    async def test_call_write_on_read_only(self, rag: FakeRAG):
        tk = GraphRAGToolkit(rag, read_only=True)  # type: ignore[arg-type]
        with pytest.raises(ReadOnlyViolation):
            await tk.call("ingest", {"text": "hello"})
        with pytest.raises(ReadOnlyViolation):
            await tk.ingest("hello")
        assert rag.ingested == []

    async def test_call_rejects_bad_arguments(self, toolkit: GraphRAGToolkit):
        with pytest.raises(ToolError, match="bogus: Extra inputs are not permitted"):
            await toolkit.call("retrieve", {"question": "Who?", "bogus": 1})
        with pytest.raises(ToolError, match="question"):
            await toolkit.call("retrieve", {})
        with pytest.raises(ToolError, match="top_k"):
            await toolkit.call("retrieve", {"question": "Who?", "top_k": 26})

    async def test_call_dispatches(self, toolkit: GraphRAGToolkit, rag: FakeRAG):
        result = await toolkit.call("retrieve", {"question": "Who?"})
        assert result.question == "Who?" and rag.retrieve_kwargs is not None


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

    async def test_new_session_shares_state(self, rag: FakeRAG):
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

    @pytest.mark.parametrize("bad", ["", "a b", "x" * 65, "../g", "a:b", "acme\n"])
    def test_for_tenant_rejects_bad_ids(self, bad: str):
        with pytest.raises(ValueError, match="tenant_id"):
            GraphRAGToolkit.for_tenant(
                ConnectionConfig(),
                bad,
                llm=SimpleNamespace(),  # type: ignore[arg-type]
                embedder=SimpleNamespace(),  # type: ignore[arg-type]
            )


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

    async def test_status_and_ontology_are_citable(self, toolkit, monkeypatch):
        async def counts(store):
            return {"entities": 40, "relations": 2, "documents": 1, "chunks": 4}

        async def schema_counts(store):
            return {"Person": 40}, {"WORKS_AT": 2}

        async def declared():
            from graphrag_sdk.core.models import Ontology

            return Ontology()

        monkeypatch.setattr(ops, "graph_counts", counts)
        monkeypatch.setattr(ops, "schema_counts", schema_counts)
        monkeypatch.setattr(toolkit, "_declared_ontology", declared)
        status = await toolkit.status()
        ontology = await toolkit.get_ontology()
        assert (status.n, ontology.n) == (1, 2)
        assert status.to_llm_text().startswith("[1] Entities: 40 |")
        assert toolkit.ledger.get(2).tool == "get_ontology"
        check = await toolkit.check_citations("The graph holds 40 Person entities [1][2].")
        assert check.grounded and check.unsupported == []

    async def test_entities_the_graph_does_not_store_are_dropped(self, toolkit, rag, monkeypatch):
        # The facts path names entities by a made-up id (the lower-cased name).
        rag.retrieve_result = RetrieverResult(
            items=[],
            metadata={
                "provenance": {
                    "entities": [
                        {"id": "bob smith", "name": "Bob Smith"},
                        {"id": "a", "name": "Alice"},
                    ]
                }
            },
        )
        card = {"id": "a", "name": "Alice", "label": "", "description": None, "properties": {}}
        walked: list[list[str]] = []

        async def enrich(store, ids):
            assert ids == ["bob smith", "a"]
            return {"a": card}

        async def triples(store, ids, *, hops, cap):
            walked.append(ids)
            return []

        monkeypatch.setattr(ops, "enrich_entities", enrich)
        monkeypatch.setattr(ops, "expand_triples", triples)
        monkeypatch.setattr(ops, "chunk_documents", lambda *a, **k: _const({}))
        result = await toolkit.retrieve("Who?")
        assert [e.id for e in result.entities] == ["a"] and walked == [["a"]]
        assert len(toolkit.ledger) == 1

    @pytest.mark.parametrize("failing", ["expand_triples", "chunk_documents"])
    async def test_a_failed_retrieve_leaves_nothing_citable(
        self, toolkit, rag, monkeypatch, failing
    ):
        rag.retrieve_result = RetrieverResult(
            items=[],
            metadata={
                "provenance": {
                    "entities": [{"id": "a", "name": "Acme"}],
                    "facts": ["Acme has 4321 staff"],
                    "chunks": [{"id": "c1", "text": "Acme has 4321 staff."}],
                }
            },
        )
        card = {"id": "a", "name": "Acme", "label": "", "description": "4321 staff"}

        async def boom(*a, **k):
            raise RuntimeError("Query timed out")

        monkeypatch.setattr(ops, "enrich_entities", lambda *a, **k: _const({"a": card}))
        monkeypatch.setattr(ops, "expand_triples", lambda *a, **k: _const([]))
        monkeypatch.setattr(ops, "chunk_documents", lambda *a, **k: _const({}))
        monkeypatch.setattr(ops, failing, boom)
        with pytest.raises(RuntimeError):
            await toolkit.retrieve("How many staff does Acme have?")
        assert len(toolkit.ledger) == 0
        assert not (await toolkit.check_citations("Acme has 4321 staff [1].")).grounded

    async def test_an_empty_passage_gets_no_number(self, toolkit, rag, monkeypatch):
        rag.retrieve_result = RetrieverResult(
            items=[],
            metadata={"provenance": {"chunks": [{"id": "c1", "text": "  "}, {"id": "c2"}]}},
        )
        monkeypatch.setattr(ops, "chunk_documents", lambda *a, **k: _const({}))
        result = await toolkit.retrieve("Who?", paths=["passages"])
        assert [c.n for c in result.chunks] == [None, None] and len(toolkit.ledger) == 0
        check = await toolkit.check_citations("Alice works at Acme [1].")
        assert not check.grounded

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

    async def test_entity_properties_are_citable(self, toolkit, rag, monkeypatch):
        rag.retrieve_result = RetrieverResult(
            items=[], metadata={"provenance": {"entities": [{"id": "a", "name": "Alice"}]}}
        )

        async def enrich(store, ids):
            card = {"id": "a", "name": "Alice", "label": "Person", "description": None}
            return {"a": {**card, "properties": {"age": 40, "aliases": ["Al", "Ally"]}}}

        async def nothing(*a, **k):
            return [] if "hops" in k else {}

        monkeypatch.setattr(ops, "enrich_entities", enrich)
        monkeypatch.setattr(ops, "expand_triples", nothing)
        monkeypatch.setattr(ops, "chunk_documents", nothing)
        result = await toolkit.retrieve("How old is Alice?")
        n = result.entities[0].n
        assert "age: 40; aliases: Al, Ally" in toolkit.ledger.get(n).content
        assert "(age: 40; aliases: Al, Ally)" in result.to_llm_text()
        check = await toolkit.check_citations(f"Alice is 40 years old [{n}].")
        assert check.grounded and not check.unsupported

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
        assert "Rephrase the question" in (result.hint or "")
        assert "fuzzy_names" not in result.hint

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

        card = {"id": "a", "name": "Alice", "label": "", "description": None, "properties": {}}
        monkeypatch.setattr(
            ops, "enrich_entities", lambda store, ids: _const({"a": card} if "a" in ids else {})
        )
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
        """GraphRAG.retrieve would register the default types first; the tool must not.
        Nor does it load the ontology: retrieve never runs text-to-Cypher."""
        rag._ontology_initialized = False
        rag.ontology = None

        def load():
            raise AssertionError("the ontology is not needed to retrieve")

        rag._ontology_store = SimpleNamespace(load=load)
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
        assert seen == {"ontology": None, "question": "Who is Alice?"}


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
            ("retrieve", {"question": "Who founded Acme?"}),
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

    async def test_existence_is_checked_read_only_on_every_call(
        self, toolkit: GraphRAGToolkit, store: FakeStore
    ):
        await toolkit.call("retrieve", {"question": "Who?"})
        await toolkit.call("retrieve", {"question": "Who?"})
        probes = [i for i, (q, _) in enumerate(store.queries) if q == "RETURN 1"]
        assert len(probes) == 2
        assert all(store.read_only[i] is True for i in probes)

    async def test_a_graph_deleted_later_is_reported_missing(
        self, toolkit: GraphRAGToolkit, store: FakeStore
    ):
        await toolkit.call("retrieve", {"question": "Who?"})
        store.error = DatabaseError("Invalid graph operation on empty key")
        with pytest.raises(ToolError, match="does not exist"):
            await toolkit.call("retrieve", {"question": "Who?"})

    async def test_an_unreachable_store_is_an_error_not_an_empty_graph(
        self, toolkit: GraphRAGToolkit, store: FakeStore
    ):
        store.error = DatabaseError("Connection refused")
        calls = (("retrieve", {"question": "Who?"}), ("get_ontology", {}), ("status", {}))
        for tool, args in calls:
            with pytest.raises(ToolError, match="Could not reach the graph store") as info:
                await toolkit.call(tool, args)
            assert "running and reachable" in info.value.hint
        assert store.queries == [("RETURN 1", {})] * 3  # nothing else was attempted

    async def test_get_ontology_names_the_missing_graph(self, missing):
        with pytest.raises(ToolError, match="Graph 'nope' does not exist"):
            await missing.call("get_ontology", {})


class TestCheckCitations:
    async def test_saying_the_graph_lacks_something_needs_no_citation(self, toolkit):
        toolkit.ledger.add(tool="retrieve", kind="passage", content="Initech was acquired in 2019.")
        gap = await toolkit.check_citations("The graph does not say who founded Globex.")
        assert not gap.grounded and gap.reason == "no_citations"
        assert "needs no citation" in gap.hint and "after each claim" not in gap.hint
        mixed = await toolkit.check_citations(
            "Initech was acquired in 2019 [1]. The graph does not say who founded Globex."
        )
        assert mixed.grounded and mixed.uncited == [] and mixed.hint is None

    async def test_cited_invalid_and_missing(self, toolkit: GraphRAGToolkit):
        toolkit.ledger.add(tool="retrieve", kind="relation", content="Alice -[WORKS_AT]-> Acme")
        ok = await toolkit.check_citations("Alice works at Acme [1].")
        assert ok.grounded and ok.reason == "cited" and ok.cited == [1]
        assert ok.sources[0].tool == "retrieve" and ok.hint is None

        mixed = await toolkit.check_citations("Alice works at Acme [1][7].")
        assert mixed.grounded and mixed.invalid == [7] and "[7]" not in mixed.answer
        assert "removed" in (mixed.hint or "")

        bad = await toolkit.check_citations("Alice works at Acme [9].")
        assert not bad.grounded and bad.reason == "invalid_citations"

        none = await toolkit.check_citations("Alice works at Acme.")
        assert not none.grounded and none.reason == "no_citations" and "[n]" in (none.hint or "")

    async def test_uncited_and_unsupported_claims(self, toolkit: GraphRAGToolkit):
        toolkit.ledger.add(tool="retrieve", kind="relation", content="Alice -[WORKS_AT]-> Acme")
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
