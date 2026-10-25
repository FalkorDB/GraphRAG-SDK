"""Tests for retrieval/strategies/multi_path.py — MultiPathRetrieval (Lean)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LatencyBudgetExceededError
from graphrag_sdk.core.models import (
    LLMResponse,
    RawSearchResult,
    RetrieverResult,
)
from graphrag_sdk.retrieval.strategies.entity_discovery import (
    expand_sibling_entities,
    is_enumeration_query,
    search_relates_edges,
)
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval
from graphrag_sdk.retrieval.strategies.result_assembly import rerank_chunks

from .conftest import MockEmbedder, MockLLM


# -- Fixtures --


@pytest.fixture
def mp_embedder() -> MockEmbedder:
    return MockEmbedder(dimension=8)


@pytest.fixture
def mp_llm() -> MockLLM:
    return MockLLM(responses=["Alice, Bob, Acme Corp"])


@pytest.fixture
def mp_graph_store():
    store = MagicMock()
    store.query_raw = AsyncMock(return_value=MagicMock(result_set=[]))
    return store


@pytest.fixture
def mp_vector_store():
    store = MagicMock()
    store.search_chunks = AsyncMock(return_value=[])
    store.search_entities = AsyncMock(return_value=[])
    store.search_relationships = AsyncMock(return_value=[])
    store.fulltext_search_chunks = AsyncMock(return_value=[])
    store.fulltext_search_entities = AsyncMock(return_value=[])
    return store


@pytest.fixture
def strategy(mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
    return MultiPathRetrieval(
        graph_store=mp_graph_store,
        vector_store=mp_vector_store,
        embedder=mp_embedder,
        llm=mp_llm,
    )


# -- Tests --


class TestMultiPathRetrieval:
    async def test_search_returns_retriever_result(self, strategy):
        result = await strategy.search("Who is Alice?")
        assert isinstance(result, RetrieverResult)

    async def test_keyword_extraction(self, strategy, mp_llm):
        """LLM should be called for keyword extraction."""
        await strategy.search("Who is Alice?")
        assert mp_llm._call_index >= 1

    async def test_question_embedding(self, strategy, mp_embedder):
        """Embedding should be called for the question via aembed_query."""
        mp_embedder.call_count = 0
        await strategy.search("Who is Alice?")
        assert mp_embedder.call_count > 0

    async def test_latency_budget_checked_before_first_phase(self, strategy, mp_embedder, mp_llm):
        ctx = Context(latency_budget_ms=0.0)

        with pytest.raises(LatencyBudgetExceededError, match="keyword extraction"):
            await strategy.search("Who is Alice?", ctx)

        assert mp_llm._call_index == 0
        assert mp_embedder.call_count == 0

    async def test_relates_search_budget_error_propagates(self):
        vector = MagicMock()
        vector.search_relationships = AsyncMock(
            side_effect=LatencyBudgetExceededError("budget exhausted")
        )

        with pytest.raises(LatencyBudgetExceededError, match="budget exhausted"):
            await search_relates_edges(vector, [0.1], rel_top_k=3)

    async def test_relates_search_checks_budget_before_vector_call(self):
        vector = MagicMock()
        vector.search_relationships = AsyncMock(return_value=[])

        with pytest.raises(LatencyBudgetExceededError, match="RELATES vector search"):
            await search_relates_edges(
                vector,
                [0.1],
                rel_top_k=3,
                ctx=Context(latency_budget_ms=0.0),
            )

        vector.search_relationships.assert_not_awaited()

    async def test_rerank_budget_error_propagates(self, mp_embedder):
        mp_embedder.aembed_documents = AsyncMock(
            side_effect=LatencyBudgetExceededError("budget exhausted")
        )

        with pytest.raises(LatencyBudgetExceededError, match="budget exhausted"):
            await rerank_chunks(
                mp_embedder,
                [0.1],
                {"c1": "text"},
                stored_embeddings={},
            )

    async def test_latency_budget_checked_before_question_embedding(
        self, strategy, mp_embedder, mp_llm, monkeypatch
    ):
        ctx = Context(latency_budget_ms=1000.0)

        def exhaust_budget(self: Context, operation: str) -> None:
            if operation == "MultiPath question embedding":
                raise LatencyBudgetExceededError("budget exhausted before embedding")

        monkeypatch.setattr(Context, "ensure_budget", exhaust_budget)
        with pytest.raises(LatencyBudgetExceededError, match="before embedding"):
            await strategy.search("Who is Alice?", ctx)

        assert mp_llm._call_index == 1
        assert mp_embedder.call_count == 0

    async def test_latency_budget_propagates_from_keyword_llm(
        self, strategy, mp_embedder, mp_llm, monkeypatch
    ):
        ctx = Context(latency_budget_ms=1000.0)

        def exhaust_budget(self: Context, operation: str) -> None:
            if operation == "MultiPath keyword extraction LLM call":
                raise LatencyBudgetExceededError("budget exhausted before keyword LLM")

        monkeypatch.setattr(Context, "ensure_budget", exhaust_budget)
        with pytest.raises(LatencyBudgetExceededError, match="keyword LLM"):
            await strategy.search("Who is Alice?", ctx)

        assert mp_llm._call_index == 0
        assert mp_embedder.call_count == 0

    async def test_relates_edge_vector_search(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """RELATES edge vector search should be called and return facts + entities."""
        mp_vector_store.search_relationships = AsyncMock(
            return_value=[
                {
                    "src_name": "Alice",
                    "type": "WORKS_AT",
                    "tgt_name": "Acme Corp",
                    "fact": "Alice is a senior engineer at Acme Corp",
                    "score": 0.95,
                },
                {
                    "src_name": "Bob",
                    "type": "MANAGES",
                    "tgt_name": "Alice",
                    "fact": "Bob manages Alice's team",
                    "score": 0.88,
                },
            ]
        )

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        result = await s.search("Who works at Acme Corp?")
        assert isinstance(result, RetrieverResult)
        # search_relationships should have been called
        assert mp_vector_store.search_relationships.call_count == 1

        # Check that facts section appears in output
        sections = {item.metadata.get("section") for item in result.items}
        assert "facts" in sections

        # Verify fact content
        for item in result.items:
            if item.metadata.get("section") == "facts":
                assert "Alice" in item.content
                assert "WORKS_AT" in item.content
                assert "Acme Corp" in item.content

    async def test_relates_entities_merged_into_discovery(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """Entities from RELATES edges should be merged into entity discovery."""
        mp_vector_store.search_relationships = AsyncMock(
            return_value=[
                {
                    "src_name": "RelEntity1",
                    "type": "KNOWS",
                    "tgt_name": "RelEntity2",
                    "fact": "they know each other",
                    "score": 0.9,
                },
            ]
        )

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        result = await s.search("What about RelEntity1?")
        assert isinstance(result, RetrieverResult)

        # Check that entities from RELATES edges appear in the output
        for item in result.items:
            if item.metadata.get("section") == "entities":
                assert "RelEntity1" in item.content or "RelEntity2" in item.content

    async def test_entity_discovery_2_paths(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """Entity discovery should use Cypher CONTAINS and fulltext (2 paths)."""
        queries_seen = []

        async def capture_query(cypher, params=None):
            queries_seen.append(cypher)
            result = MagicMock()
            result.result_set = []
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=capture_query)

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        await s.search("Who is Alice at Acme Corp?")

        # Cypher CONTAINS should use UNWIND $keywords
        unwind_kw_queries = [q for q in queries_seen if "UNWIND $keywords" in q]
        assert len(unwind_kw_queries) >= 1

        # Fulltext should be called on __Entity__
        assert mp_vector_store.fulltext_search_entities.call_count >= 1

    async def test_chunk_retrieval_2_paths(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """Chunk retrieval should use fulltext + vector (2 paths only)."""
        mp_vector_store.fulltext_search_chunks = AsyncMock(
            return_value=[
                {"id": "c1", "text": "Alice is an engineer.", "score": 0.8},
            ]
        )
        mp_vector_store.search_chunks = AsyncMock(
            return_value=[
                {"id": "c2", "text": "Bob works with Alice.", "score": 0.7},
            ]
        )

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        result = await s.search("Who is Alice?")
        assert isinstance(result, RetrieverResult)
        # Both fulltext and vector search called
        assert mp_vector_store.fulltext_search_chunks.call_count >= 1
        assert mp_vector_store.search_chunks.call_count >= 1

    async def test_mentioned_in_and_2hop_chunk_paths(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """Retrieval should use MENTIONED_IN and 2-hop chunk paths when entities are discovered."""
        mp_vector_store.search_relationships = AsyncMock(
            return_value=[
                {
                    "src_name": "Alice",
                    "type": "WORKS_AT",
                    "tgt_name": "Acme",
                    "fact": "engineer",
                    "score": 0.9,
                },
            ]
        )

        queries_seen = []

        async def capture_query(cypher, params=None):
            queries_seen.append(cypher)
            result = MagicMock()
            result.result_set = []
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=capture_query)

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        await s.search("Who is Alice?")

        # MENTIONED_IN chunk queries should be present
        mention_chunk_queries = [q for q in queries_seen if "MENTIONED_IN" in q and "Chunk" in q]
        assert len(mention_chunk_queries) >= 1

        # 2-hop chunk queries (entity→neighbor→MENTIONED_IN→Chunk) should be present
        twohop_chunk_queries = [
            q for q in queries_seen if "neighbor" in q.lower() and "MENTIONED_IN" in q
        ]
        assert len(twohop_chunk_queries) >= 1

    async def test_mentioned_in_ranks_chunks_by_cosine(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """The MENTIONED_IN path must rank chunks per entity by cosine
        distance to the query embedding — not pick an arbitrary 3.

        Hub entities can be MENTIONED_IN hundreds of chunks; arbitrary
        selection almost never surfaces the chunks relevant to the
        current query (regression: see PR referencing issue #258).
        """
        # Seed entity discovery so Path C runs (mirrors the setup in
        # ``test_mentioned_in_and_2hop_chunk_paths``).
        mp_vector_store.search_relationships = AsyncMock(
            return_value=[
                {
                    "src_name": "Alice",
                    "type": "WORKS_AT",
                    "tgt_name": "Acme",
                    "fact": "engineer",
                    "score": 0.9,
                },
            ]
        )

        captured: list[tuple[str, dict]] = []

        async def capture_query(cypher, params=None):
            captured.append((cypher, params or {}))
            result = MagicMock()
            result.result_set = []
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=capture_query)

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        await s.search("Who is Alice?")

        # Find the direct MENTIONED_IN query (entity -> chunk, not 2-hop)
        direct_mention = [
            (q, p)
            for q, p in captured
            if "MENTIONED_IN" in q and "Chunk" in q and "neighbor" not in q.lower()
        ]
        assert direct_mention, "expected at least one direct MENTIONED_IN chunk query"

        cypher, params = direct_mention[0]
        # The fix: rank by cosine distance to the query vector before
        # COLLECT, so per-entity chunk selection is query-relevant.
        assert "vec.cosineDistance" in cypher, (
            "MENTIONED_IN chunk query must rank by cosine distance to the "
            "query vector (regression of issue #258)"
        )
        assert "ORDER BY" in cypher, "expected ORDER BY to make COLLECT[..3] meaningful"
        assert "qv" in params, "query vector must be passed as a parameter"

    async def test_format_produces_sections(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """Output should include structured sections when data is available."""
        mp_vector_store.search_relationships = AsyncMock(
            return_value=[
                {
                    "src_name": "Alice",
                    "type": "WORKS_AT",
                    "tgt_name": "Acme",
                    "fact": "engineer",
                    "score": 0.9,
                },
            ]
        )
        mp_vector_store.fulltext_search_chunks = AsyncMock(
            return_value=[
                {"id": "c1", "text": "Alice is a software engineer at Acme Corp.", "score": 0.8},
            ]
        )

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        result = await s.search("Who is Alice?")
        if result.items:
            sections = {item.metadata.get("section") for item in result.items}
            assert len(sections) >= 1

    async def test_2hop_relationship_expansion(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """The walk expands the far end of an edge, reaching a second link."""
        mp_vector_store.search_relationships = AsyncMock(
            return_value=[
                {
                    "src_name": "Alice",
                    "type": "WORKS_AT",
                    "tgt_name": "Acme",
                    "fact": "",
                    "score": 0.9,
                    "src_id": "alice__person",
                    "tgt_id": "acme__org",
                },
            ]
        )

        expanded: list[list[str]] = []

        async def mock_query_raw(cypher, params=None):
            result = MagicMock()
            result.result_set = []
            if "RETURN r, b.id AS other" in cypher:
                eids = params["eids"]
                expanded.append(eids)
                # src, rel, tgt, description, fact, distance, far end id
                if "alice__person" in eids:
                    result.result_set = [["Alice", "WORKS_AT", "Bob", "", "", 0.3, "bob__person"]]
                elif "bob__person" in eids:
                    result.result_set = [
                        ["Bob", "LIVES_IN", "NYC", "Bob lives in NYC.", "", 0.1, "nyc__location"]
                    ]
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=mock_query_raw)

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        result = await s.search("Where does Alice's colleague live?")
        assert isinstance(result, RetrieverResult)
        # Round 2 expanded Bob, reached only through Alice's edge.
        assert any("bob__person" in eids for eids in expanded[1:])
        rels = [i.content for i in result.items if i.metadata.get("section") == "relationships"]
        assert rels and "Bob —[LIVES_IN]→ NYC: Bob lives in NYC." in rels[0]

    async def test_source_document_tags(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """Passages should include [Source: <path>] with the full Document.path verbatim.

        Citation consumers (link builders, source-attribution UIs) need
        the full relative path to disambiguate files that share a
        basename — e.g. `operations/index.md` vs `commands/index.md`.
        """
        mp_vector_store.fulltext_search_chunks = AsyncMock(
            return_value=[
                {"id": "c1", "text": "Alice is an engineer at Acme Corp.", "score": 0.8},
            ]
        )

        async def mock_query_raw(cypher, params=None):
            if "PART_OF" in cypher:
                result = MagicMock()
                result.result_set = [["c1", "docs/fiction/novel.txt"]]
                return result
            result = MagicMock()
            result.result_set = []
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=mock_query_raw)

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        result = await s.search("Who is Alice?")
        for item in result.items:
            if item.metadata.get("section") == "passages":
                assert "[Source: docs/fiction/novel.txt]" in item.content
                break

    async def test_question_type_yesno(self):
        """Yes/no questions should produce a hint."""
        hint = MultiPathRetrieval._detect_question_type("Is Alice an engineer?")
        assert "yes/no" in hint.lower()

    async def test_question_type_who(self):
        """Who questions should produce a person hint."""
        hint = MultiPathRetrieval._detect_question_type("Who founded Acme Corp?")
        assert "person" in hint.lower() or "character" in hint.lower()

    async def test_question_type_where(self):
        """Where questions should produce a location hint."""
        hint = MultiPathRetrieval._detect_question_type("Where is Acme located?")
        assert "place" in hint.lower() or "location" in hint.lower()

    async def test_question_type_open(self):
        """Open-ended questions should produce no hint."""
        hint = MultiPathRetrieval._detect_question_type("Describe Alice's role at Acme.")
        assert hint == ""

    async def test_question_type_hint_in_output(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """Yes/no question should have a hint section in the output."""
        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        result = await s.search("Is Alice an engineer?")
        sections = {item.metadata.get("section") for item in result.items}
        assert "hint" in sections

    async def test_empty_results_graceful(self, strategy):
        """Should handle no results without errors."""
        result = await strategy.search("Completely unknown topic XYZ123")
        assert isinstance(result, RetrieverResult)

    async def test_constructor_defaults(self):
        """Verify new constructor defaults for lean retrieval."""
        store = MagicMock()
        store.query_raw = AsyncMock(return_value=MagicMock(result_set=[]))
        vec = MagicMock()
        vec.search_chunks = AsyncMock(return_value=[])
        vec.search_relationships = AsyncMock(return_value=[])
        vec.fulltext_search_chunks = AsyncMock(return_value=[])
        vec.fulltext_search_entities = AsyncMock(return_value=[])

        s = MultiPathRetrieval(
            graph_store=store,
            vector_store=vec,
            embedder=MockEmbedder(dimension=8),
            llm=MockLLM(responses=["test"]),
        )
        assert s._chunk_top_k == 20  # sized for the default 384-token chunks
        assert s._max_entities == 30
        assert s._max_relationships == 20
        assert s._rel_top_k == 15

    async def test_no_batch_embed_method(self):
        """_batch_embed should no longer exist (replaced by single aembed_query)."""
        assert not hasattr(MultiPathRetrieval, "_batch_embed")

    async def test_no_llm_rerank_method(self):
        """_llm_rerank_passages should no longer exist."""
        assert not hasattr(MultiPathRetrieval, "_llm_rerank_passages")


class TestMultiPathUnwindQueries:
    """Tests for UNWIND batched Cypher queries."""

    async def test_cypher_contains_uses_unwind(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """Entity CONTAINS search should use UNWIND $keywords."""
        queries_seen = []

        async def capture_query(cypher, params=None):
            queries_seen.append(cypher)
            result = MagicMock()
            result.result_set = []
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=capture_query)

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        await s.search("Who is Alice at Acme Corp?")

        unwind_kw_queries = [q for q in queries_seen if "UNWIND $keywords" in q]
        assert len(unwind_kw_queries) >= 1

    async def test_1hop_relationships_uses_unwind(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """1-hop relationship expansion should use UNWIND $eids."""
        mp_vector_store.search_relationships = AsyncMock(
            return_value=[
                {
                    "src_name": "Alice",
                    "type": "WORKS_AT",
                    "tgt_name": "Acme",
                    "fact": "",
                    "score": 0.9,
                },
            ]
        )

        queries_seen = []

        async def capture_query(cypher, params=None):
            queries_seen.append(cypher)
            result = MagicMock()
            result.result_set = []
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=capture_query)

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        await s.search("Who is Alice?")

        unwind_queries = [q for q in queries_seen if "UNWIND $eids" in q]
        assert len(unwind_queries) >= 1


class TestSearchRelatesEdges:
    """Tests for the _search_relates_edges method."""

    async def test_returns_facts_and_entities(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """_search_relates_edges should return fact strings and entity dict."""
        mp_vector_store.search_relationships = AsyncMock(
            return_value=[
                {
                    "src_name": "Alice",
                    "type": "WORKS_AT",
                    "tgt_name": "Acme Corp",
                    "fact": "Alice is a senior engineer",
                    "score": 0.95,
                },
            ]
        )

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        facts, entities = await s._search_relates_edges([0.1] * 8)

        assert len(facts) == 1
        fact_text, fact_score = facts[0]
        assert "Alice" in fact_text
        assert "WORKS_AT" in fact_text
        assert "Acme Corp" in fact_text
        assert "senior engineer" in fact_text
        assert fact_score == 0.95

        assert "alice" in entities
        assert "acme_corp" in entities
        assert entities["alice"]["name"] == "Alice"
        assert entities["acme_corp"]["name"] == "Acme Corp"

    async def test_handles_missing_fact(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """Facts without fact text should still produce a line."""
        mp_vector_store.search_relationships = AsyncMock(
            return_value=[
                {
                    "src_name": "Bob",
                    "type": "KNOWS",
                    "tgt_name": "Carol",
                    "fact": "",
                    "score": 0.8,
                },
            ]
        )

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        facts, entities = await s._search_relates_edges([0.1] * 8)

        assert len(facts) == 1
        fact_text, fact_score = facts[0]
        assert "Bob" in fact_text
        assert "KNOWS" in fact_text
        assert "Carol" in fact_text
        assert ":" not in fact_text  # no colon appended when fact is empty
        assert fact_score == 0.8

    async def test_handles_search_failure(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """Should gracefully handle search_relationships failure."""
        mp_vector_store.search_relationships = AsyncMock(side_effect=Exception("connection error"))

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        facts, entities = await s._search_relates_edges([0.1] * 8)

        assert facts == []
        assert entities == {}

    async def test_deduplicates_entities(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """Same entity appearing in multiple edges should only appear once."""
        mp_vector_store.search_relationships = AsyncMock(
            return_value=[
                {
                    "src_name": "Alice",
                    "type": "WORKS_AT",
                    "tgt_name": "Acme",
                    "fact": "f1",
                    "score": 0.9,
                },
                {
                    "src_name": "Alice",
                    "type": "KNOWS",
                    "tgt_name": "Bob",
                    "fact": "f2",
                    "score": 0.8,
                },
            ]
        )

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        facts, entities = await s._search_relates_edges([0.1] * 8)

        assert len(facts) == 2
        # "alice" should appear only once in entities
        alice_count = sum(1 for eid in entities if eid == "alice")
        assert alice_count == 1


class TestIsEnumerationQuery:
    """Tests for is_enumeration_query heuristic."""

    @pytest.mark.parametrize(
        "query",
        [
            "What are ALL the FalkorDB-specific list functions?",
            "List every function in the API",
            "Name all the sorting algorithms",
            "Give me the complete list of endpoints",
            "Enumerate each parameter",
            "What are all the methods?",
            "Show me all of the supported types",
        ],
    )
    def test_positive(self, query):
        assert is_enumeration_query(query)

    @pytest.mark.parametrize(
        "query",
        [
            "What is the default parameter for list.remove?",
            "How does FalkorDB handle indexing?",
            "Tell me all about Alice",
            "Who is Alice?",
            "Where is Acme located?",
        ],
    )
    def test_negative(self, query):
        assert not is_enumeration_query(query)


class TestExpandSiblingEntities:
    """Tests for expand_sibling_entities."""

    async def test_expands_siblings_via_shared_hub(self):
        """Should find sibling entities through a shared hub entity."""
        graph_store = MagicMock()
        graph_store.query_raw = AsyncMock(
            return_value=MagicMock(
                result_set=[
                    ["id_remove", "list.remove", "Removes elements from a list"],
                ]
            )
        )

        found = {
            "id_dedup": {"name": "list.dedup", "description": ""},
            "id_insert": {"name": "list.insert", "description": ""},
            "id_sort": {"name": "list.sort", "description": ""},
        }
        sources: dict[str, str] = {}

        added = await expand_sibling_entities(graph_store, found, sources)

        assert added == 1
        assert "id_remove" in found
        assert found["id_remove"]["name"] == "list.remove"
        assert sources["id_remove"] == "sibling_expansion"

    async def test_skips_when_fewer_than_2_entities(self):
        """Should not query the graph when fewer than 2 entities are found."""
        graph_store = MagicMock()
        graph_store.query_raw = AsyncMock()

        found = {"id_only": {"name": "only_entity", "description": ""}}
        sources: dict[str, str] = {}

        added = await expand_sibling_entities(graph_store, found, sources)

        assert added == 0
        graph_store.query_raw.assert_not_called()

    async def test_handles_query_failure(self):
        """Should return 0 and not raise on graph query failure."""
        graph_store = MagicMock()
        graph_store.query_raw = AsyncMock(side_effect=Exception("connection error"))

        found = {
            "id_a": {"name": "A", "description": ""},
            "id_b": {"name": "B", "description": ""},
        }
        sources: dict[str, str] = {}

        added = await expand_sibling_entities(graph_store, found, sources)
        assert added == 0

    async def test_does_not_duplicate_found_entities(self):
        """Should not re-add entities already in the found set."""
        graph_store = MagicMock()
        graph_store.query_raw = AsyncMock(
            return_value=MagicMock(
                result_set=[
                    ["id_dedup", "list.dedup", "Already found"],
                    ["id_new", "list.new", "New entity"],
                ]
            )
        )

        found = {
            "id_dedup": {"name": "list.dedup", "description": ""},
            "id_insert": {"name": "list.insert", "description": ""},
        }
        sources: dict[str, str] = {}

        added = await expand_sibling_entities(graph_store, found, sources)
        assert added == 1
        assert "id_new" in found


class TestSiblingExpansionIntegration:
    """Integration: sibling expansion triggers only for enumeration queries."""

    async def test_enumeration_query_triggers_expansion(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """An enumeration query should issue a sibling-expansion Cypher query."""
        queries_seen = []

        async def capture_query(cypher, params=None):
            queries_seen.append(cypher)
            result = MagicMock()
            result.result_set = []
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=capture_query)
        mp_vector_store.search_relationships = AsyncMock(
            return_value=[
                {
                    "src_name": "list.dedup",
                    "type": "BELONGS_TO",
                    "tgt_name": "FalkorDB",
                    "fact": "",
                    "score": 0.9,
                },
                {
                    "src_name": "list.insert",
                    "type": "BELONGS_TO",
                    "tgt_name": "FalkorDB",
                    "fact": "",
                    "score": 0.85,
                },
            ]
        )

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        await s.search("What are all the FalkorDB list functions?")

        sibling_queries = [q for q in queries_seen if "hub" in q.lower() and "sibling" in q.lower()]
        assert len(sibling_queries) >= 1

    async def test_non_enumeration_query_skips_expansion(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """A non-enumeration query should NOT issue a sibling-expansion query."""
        queries_seen = []

        async def capture_query(cypher, params=None):
            queries_seen.append(cypher)
            result = MagicMock()
            result.result_set = []
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=capture_query)

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        await s.search("What is the default parameter for list.remove?")

        sibling_queries = [q for q in queries_seen if "hub" in q.lower() and "sibling" in q.lower()]
        assert len(sibling_queries) == 0


# -- Retrieval aligned with what 1.5.0 ingestion writes --


class TestRelationLines:
    def test_triple_with_parentheses_in_a_name_is_not_repeated(self):
        from graphrag_sdk.retrieval.strategies.result_assembly import format_relation_line

        line = format_relation_line(
            "Python (language)", "USES", "Java", "", "(Python (language), USES, Java)"
        )
        assert "(Python (language), USES, Java)" not in line
        assert line.startswith("Python (language) —[USES]→ Java")

    def test_description_preferred_and_triple_not_repeated(self):
        from graphrag_sdk.retrieval.strategies.result_assembly import format_relation_line

        line = format_relation_line(
            "Alice", "WORKS_AT", "Acme", "Alice is an engineer.", "(Alice, WORKS_AT, Acme): x"
        )
        assert line == "Alice —[WORKS_AT]→ Acme: Alice is an engineer."

    def test_fact_used_without_its_stored_triple(self):
        from graphrag_sdk.retrieval.strategies.result_assembly import format_relation_line

        assert (
            format_relation_line("A", "R", "B", "", "(Old A, R, B): evidence")
            == "A —[R]→ B: evidence"
        )
        assert format_relation_line("A", "R", "B", "", "(A, R, B)") == "A —[R]→ B"

    def test_long_merged_description_is_capped(self):
        from graphrag_sdk.retrieval.strategies.result_assembly import assemble_raw_result

        long_desc = " | ".join(["Alice is a senior engineer at Acme"] * 40)
        raw = assemble_raw_result(
            [("alice__person", {"name": "Alice", "description": long_desc})], [], [], []
        )
        content = raw.records[0]["content"]
        assert content.startswith("## Key Entities\n- Alice: ")
        assert len(content) < 500 and content.endswith("…")


class TestEdgeEndpoints:
    async def test_relates_hits_use_real_ids_and_descriptions(self):
        vector = MagicMock()
        vector.search_relationships = AsyncMock(
            return_value=[
                {
                    "src_name": "Alice",
                    "type": "WORKS_AT",
                    "tgt_name": "Acme Corp",
                    "fact": "(Al, WORKS_AT, Acme Corp): works there",
                    "description": "works there",
                    "src_id": "alice__person",
                    "tgt_id": "acme_corp__organization",
                    "src_description": "An engineer.",
                    "tgt_description": "A company.",
                    "score": 0.9,
                }
            ]
        )
        facts, entities = await search_relates_edges(vector, [0.1], rel_top_k=3)

        assert facts == [("Alice —[WORKS_AT]→ Acme Corp: works there", 0.9)]
        # Real ids, so expansion and chunk lookups by id find these entities;
        # no descriptions, the fact line already places them.
        assert entities == {
            "alice__person": {"name": "Alice", "description": ""},
            "acme_corp__organization": {"name": "Acme Corp", "description": ""},
        }

    async def test_hits_without_ids_fall_back_to_name_ids(self):
        vector = MagicMock()
        vector.search_relationships = AsyncMock(
            return_value=[
                {"src_name": "Bob", "type": "KNOWS", "tgt_name": "Eve", "fact": "", "score": 0.5}
            ]
        )
        _, entities = await search_relates_edges(vector, [0.1])
        assert set(entities) == {"bob", "eve"}


class TestRelationshipExpansion:
    async def test_equal_distances_break_on_the_line_not_arrival_order(self):
        from graphrag_sdk.retrieval.strategies.relationship_expansion import expand_relationships

        calls = {"n": 0}

        async def query_raw(cypher, params=None):
            calls["n"] += 1
            result = MagicMock()
            result.result_set = []
            if calls["n"] == 1:
                result.result_set = [
                    ["Acme", "EMPLOYS", "Zoe", "", "", 0.2, "zoe__person"],
                    ["Acme", "EMPLOYS", "Ann", "", "", 0.2, "ann__person"],
                ]
            return result

        store = MagicMock()
        store.query_raw = AsyncMock(side_effect=query_raw)
        lines = await expand_relationships(
            store, [("acme__organization", {"name": "Acme"})], query_vector=[0.1]
        )
        assert lines == ["Acme —[EMPLOYS]→ Ann", "Acme —[EMPLOYS]→ Zoe"]

    async def test_matches_both_directions_and_ranks_by_question(self):
        from graphrag_sdk.retrieval.strategies.relationship_expansion import expand_relationships

        seen: list[tuple[str, dict]] = []

        async def query_raw(cypher, params=None):
            seen.append((cypher, params or {}))
            result = MagicMock()
            result.result_set = []
            if len(seen) == 1:
                # src, rel, tgt, description, fact, distance, far end (unordered on purpose)
                result.result_set = [
                    ["Bob", "FOUNDED", "Acme", "Bob founded Acme.", "", 0.40, "bob__person"],
                    ["Acme", "LOCATED_IN", "Paris", "Acme is in Paris.", "", 0.10, "paris__loc"],
                    ["Acme", "EMPLOYS", "Eve", "", "", None, "eve__person"],
                ]
            return result

        store = MagicMock()
        store.query_raw = AsyncMock(side_effect=query_raw)
        lines = await expand_relationships(
            store, [("acme__organization", {"name": "Acme"})], query_vector=[0.1, 0.2]
        )

        first = seen[0][0]
        assert "-[r:RELATES]-(b:__Entity__)" in first  # undirected: incoming edges count
        assert "vec.cosineDistance(r.embedding, vecf32($qv))" in first
        assert seen[0][1]["qv"] == [0.1, 0.2]
        # Round 2 expands the far ends, closest edge first.
        assert seen[1][1]["eids"] == ["paris__loc", "bob__person", "eve__person"]
        assert lines == [
            "Acme —[LOCATED_IN]→ Paris: Acme is in Paris.",
            "Bob —[FOUNDED]→ Acme: Bob founded Acme.",
            "Acme —[EMPLOYS]→ Eve",  # no embedding: kept, ranked last
        ]

    async def test_walk_stops_and_never_reexpands(self):
        from graphrag_sdk.retrieval.strategies.relationship_expansion import expand_relationships

        calls: list[list[str]] = []

        async def query_raw(cypher, params=None):
            calls.append(params["eids"])
            result = MagicMock()
            # Every entity points back at "a": nothing new to expand after round 1.
            result.result_set = [["A", "R", "B", "", "", 0.2, "a"]]
            return result

        store = MagicMock()
        store.query_raw = AsyncMock(side_effect=query_raw)
        await expand_relationships(store, [("a", {"name": "A"})], query_vector=[0.1])
        assert calls == [["a"]]

    async def test_without_a_vector_keeps_graph_order(self):
        from graphrag_sdk.retrieval.strategies.relationship_expansion import expand_relationships

        result = MagicMock()
        result.result_set = [["X", "R", "Y", "", "", 0.0, "y"], ["Y", "S", "Z", "", "", 0.0, "z"]]
        store = MagicMock()
        store.query_raw = AsyncMock(return_value=result)
        lines = await expand_relationships(store, [("x", {"name": "X"})], max_relationships=1)
        assert lines == ["X —[R]→ Y"]
        assert "cosineDistance" not in store.query_raw.call_args_list[0][0][0]

    async def test_relationship_lines_already_in_facts_are_not_repeated(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        mp_vector_store.search_relationships = AsyncMock(
            return_value=[
                {
                    "src_name": "A",
                    "type": "R",
                    "tgt_name": "B",
                    "description": "d",
                    "score": 0.9,
                    "src_id": "a",
                    "tgt_id": "b",
                },
            ]
        )

        async def query_raw(cypher, params=None):
            result = MagicMock()
            result.result_set = []
            if "RETURN r, b.id AS other" in cypher and "a" in params["eids"]:
                result.result_set = [
                    ["A", "R", "B", "d", "", 0.1, "b"],
                    ["A", "S", "C", "e", "", 0.2, "c"],
                ]
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=query_raw)
        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        result = await s.search("What is A related to?")
        by_section = {i.metadata.get("section"): i.content for i in result.items}
        assert "A —[R]→ B: d" in by_section["facts"]
        assert "A —[R]→ B: d" not in by_section["relationships"]
        assert "A —[S]→ C: e" in by_section["relationships"]


class TestAliasLookup:
    async def test_exact_and_contains_passes_read_aliases(self, mp_graph_store, mp_vector_store):
        from graphrag_sdk.retrieval.strategies.entity_discovery import discover_entities

        queries: list[str] = []

        async def capture(cypher, params=None):
            queries.append(cypher)
            return MagicMock(result_set=[])

        mp_graph_store.query_raw = AsyncMock(side_effect=capture)
        await discover_entities(mp_graph_store, mp_vector_store, ["Real Jardín Botánico"], [])
        exact, contains = queries[0], queries[1]
        assert "e.aliases" in exact and "toLower(al) = toLower(kw)" in exact
        assert "e.aliases" in contains and "CONTAINS toLower(kw)" in contains


class TestChunksAlignedWithIngestion:
    async def test_chunks_of_pending_or_deleting_documents_are_left_out(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        mp_vector_store.search_chunks = AsyncMock(
            return_value=[
                {"id": "live", "text": "The live passage.", "score": 0.9},
                {"id": "pending", "text": "The half-written new version.", "score": 0.95},
            ]
        )
        seen: list[str] = []

        async def query_raw(cypher, params=None):
            seen.append(cypher)
            result = MagicMock()
            result.result_set = []
            if "__pending__" in cypher and "pending_delete" in cypher:
                result.result_set = [["pending"]]
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=query_raw)
        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        result = await s.search("What does the passage say?")
        passages = [i.content for i in result.items if i.metadata.get("section") == "passages"]
        assert passages and "The live passage." in passages[0]
        assert "half-written" not in passages[0]

    async def test_same_text_under_two_ids_is_one_passage(self, mp_graph_store, mp_vector_store):
        from graphrag_sdk.retrieval.strategies.chunk_retrieval import retrieve_chunks

        mp_vector_store.search_chunks = AsyncMock(
            return_value=[
                {"id": "a", "text": "Same text.", "score": 0.9},
                {"id": "b", "text": "Same text.", "score": 0.9},
            ]
        )
        chunks, _, _ = await retrieve_chunks(
            mp_vector_store, mp_graph_store, "q", [0.1], [], [], []
        )
        assert chunks == {"a": "Same text."}

    async def test_mentioned_in_keeps_unembedded_record_chunks(
        self, mp_graph_store, mp_vector_store
    ):
        from graphrag_sdk.retrieval.strategies.chunk_retrieval import retrieve_chunks

        captured: list[str] = []

        async def query_raw(cypher, params=None):
            captured.append(cypher)
            return MagicMock(result_set=[])

        mp_graph_store.query_raw = AsyncMock(side_effect=query_raw)
        await retrieve_chunks(
            mp_vector_store, mp_graph_store, "q", [0.1], [], [], [("acme__org", {"name": "Acme"})]
        )
        mention = next(c for c in captured if "MENTIONED_IN]->(c:Chunk)" in c and "eid, c" in c)
        assert "WHERE c.embedding IS NOT NULL" not in mention
        assert "CASE WHEN c.embedding IS NULL THEN 3.0" in mention
        # Unembedded record chunks get slots of their own, so three prose
        # chunks of the same entity cannot crowd its table row out.
        assert "[x IN cs WHERE x.embedding IS NOT NULL][..3]" in mention
        assert "[x IN cs WHERE x.embedding IS NULL][..1]" in mention
        two_hop = next(c for c in captured if "neighbor:__Entity__" in c)
        assert "ORDER BY dist ASC, c.id ASC" in two_hop
        assert "[x IN cs WHERE x.embedding IS NOT NULL][..20]" in two_hop
        assert "[x IN cs WHERE x.embedding IS NULL][..5]" in two_hop

    async def test_a_pending_copy_reached_first_does_not_hide_the_live_passage(
        self, mp_graph_store, mp_vector_store
    ):
        from graphrag_sdk.retrieval.strategies.chunk_retrieval import retrieve_chunks

        mp_vector_store.search_chunks = AsyncMock(
            return_value=[
                {"id": "pending", "text": "Same text.", "score": 0.95},
                {"id": "live", "text": "Same text.", "score": 0.9},
            ]
        )

        async def query_raw(cypher, params=None):
            result = MagicMock()
            result.result_set = []
            if "__pending__" in cypher and "pending_delete" in cypher:
                result.result_set = [["pending"]]
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=query_raw)
        chunks, sources, _ = await retrieve_chunks(
            mp_vector_store, mp_graph_store, "q", [0.1], [], [], []
        )
        assert chunks == {"live": "Same text."}
        assert list(sources) == ["live"]

    async def test_vector_search_fetches_as_many_candidates_as_passages_kept(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        await s.search("What does the passage say?")
        assert mp_vector_store.search_chunks.call_args.kwargs["top_k"] == 20

    async def test_rerank_embeds_only_candidates_without_a_stored_vector(self):
        embedder = MagicMock()
        embedder.aembed_documents = AsyncMock(return_value=[[1.0, 0.0]])
        candidates = {f"c{i}": f"text {i}" for i in range(10)}
        stored = {f"c{i}": [0.0, 1.0] for i in range(9)}  # c9 is a record chunk

        ranked = await rerank_chunks(embedder, [1.0, 0.0], candidates, 3, stored_embeddings=stored)

        embedder.aembed_documents.assert_awaited_once()
        assert embedder.aembed_documents.call_args[0][0] == ["text 9"]
        assert ranked[0] == "text 9"  # closest to the question, not ranked last


class TestEntityLookupDetails:
    async def test_short_acronyms_skip_the_contains_pass(self, mp_graph_store, mp_vector_store):
        from graphrag_sdk.retrieval.strategies.entity_discovery import discover_entities

        calls: list[tuple[str, dict]] = []

        async def capture(cypher, params=None):
            calls.append((cypher, params or {}))
            return MagicMock(result_set=[])

        mp_graph_store.query_raw = AsyncMock(side_effect=capture)
        await discover_entities(mp_graph_store, mp_vector_store, ["US", "Acme"], [])
        exact_params, contains_params = calls[0][1], calls[1][1]
        assert exact_params["keywords"] == ["US", "Acme"]
        assert contains_params["keywords"] == ["Acme"]
        assert "ORDER BY rank ASC, id ASC" in calls[0][0]

    async def test_only_short_acronyms_skip_the_contains_query(
        self, mp_graph_store, mp_vector_store
    ):
        from graphrag_sdk.retrieval.strategies.entity_discovery import discover_entities

        calls: list[str] = []

        async def capture(cypher, params=None):
            calls.append(cypher)
            return MagicMock(result_set=[])

        mp_graph_store.query_raw = AsyncMock(side_effect=capture)
        await discover_entities(mp_graph_store, mp_vector_store, ["US", "UK"], [])
        assert not any("CONTAINS toLower(kw)" in c for c in calls)


class TestCypherPromptKnowsTheGraph:
    def test_prompt_lists_aliases_and_common_relation_types(self):
        from graphrag_sdk.retrieval.strategies.cypher_generation import (
            build_ontology_prompt,
            render_ontology_block,
        )

        block = render_ontology_block(None)
        assert "aliases (LIST)" in block
        assert "Common `rel_type` values" in block and "LOCATED_IN" in block
        assert "startNode(r).name" in block  # src_name is marked as possibly outdated
        prompt = build_ontology_prompt(None, "Who?")
        assert "coalesce(e.aliases, [])" in prompt
