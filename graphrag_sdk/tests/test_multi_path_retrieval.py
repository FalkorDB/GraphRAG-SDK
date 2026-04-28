"""Tests for retrieval/strategies/multi_path.py — MultiPathRetrieval (Lean)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    LLMResponse,
    RawSearchResult,
    RetrieverResult,
)
from graphrag_sdk.retrieval.strategies.entity_discovery import (
    expand_sibling_entities,
    is_enumeration_query,
)
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval

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

    async def test_relates_edge_vector_search(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """RELATES edge vector search should be called and return facts + entities."""
        mp_vector_store.search_relationships = AsyncMock(return_value=[
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
        ])

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

    async def test_relates_entities_merged_into_discovery(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """Entities from RELATES edges should be merged into entity discovery."""
        mp_vector_store.search_relationships = AsyncMock(return_value=[
            {
                "src_name": "RelEntity1",
                "type": "KNOWS",
                "tgt_name": "RelEntity2",
                "fact": "they know each other",
                "score": 0.9,
            },
        ])

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

    async def test_entity_discovery_2_paths(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
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

    async def test_chunk_retrieval_2_paths(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """Chunk retrieval should use fulltext + vector (2 paths only)."""
        mp_vector_store.fulltext_search_chunks = AsyncMock(return_value=[
            {"id": "c1", "text": "Alice is an engineer.", "score": 0.8},
        ])
        mp_vector_store.search_chunks = AsyncMock(return_value=[
            {"id": "c2", "text": "Bob works with Alice.", "score": 0.7},
        ])

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

    async def test_mentioned_in_and_2hop_chunk_paths(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """Retrieval should use MENTIONED_IN and 2-hop chunk paths when entities are discovered."""
        mp_vector_store.search_relationships = AsyncMock(return_value=[
            {"src_name": "Alice", "type": "WORKS_AT", "tgt_name": "Acme", "fact": "engineer", "score": 0.9},
        ])

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
        mention_chunk_queries = [
            q for q in queries_seen
            if "MENTIONED_IN" in q and "Chunk" in q
        ]
        assert len(mention_chunk_queries) >= 1

        # 2-hop chunk queries (entity→neighbor→MENTIONED_IN→Chunk) should be present
        twohop_chunk_queries = [
            q for q in queries_seen
            if "neighbor" in q.lower() and "MENTIONED_IN" in q
        ]
        assert len(twohop_chunk_queries) >= 1

    async def test_format_produces_sections(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """Output should include structured sections when data is available."""
        mp_vector_store.search_relationships = AsyncMock(return_value=[
            {"src_name": "Alice", "type": "WORKS_AT", "tgt_name": "Acme", "fact": "engineer", "score": 0.9},
        ])
        mp_vector_store.fulltext_search_chunks = AsyncMock(return_value=[
            {"id": "c1", "text": "Alice is a software engineer at Acme Corp.", "score": 0.8},
        ])

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

    async def test_2hop_relationship_expansion(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """2-hop relationship expansion should be attempted for top entities."""
        mp_vector_store.search_relationships = AsyncMock(return_value=[
            {"src_name": "Alice", "type": "WORKS_AT", "tgt_name": "Acme", "fact": "", "score": 0.9},
        ])

        call_count = {"two_hop_rel": 0}

        async def mock_query_raw(cypher, params=None):
            if "r1" in cypher and "r2" in cypher:
                call_count["two_hop_rel"] += 1
                result = MagicMock()
                result.result_set = [
                    ["Alice", "WORKS_AT", "Acme", "LOCATED_IN", "NYC"],
                ]
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
        result = await s.search("Where is Alice's company located?")
        assert isinstance(result, RetrieverResult)
        assert call_count["two_hop_rel"] >= 1

    async def test_source_document_tags(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """Passages should include [Source: filename] when document info is available."""
        mp_vector_store.fulltext_search_chunks = AsyncMock(return_value=[
            {"id": "c1", "text": "Alice is an engineer at Acme Corp.", "score": 0.8},
        ])

        async def mock_query_raw(cypher, params=None):
            if "PART_OF" in cypher:
                result = MagicMock()
                result.result_set = [["c1", "/docs/novel.txt"]]
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
                assert "[Source: novel.txt]" in item.content
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

    async def test_question_type_hint_in_output(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
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
        assert s._chunk_top_k == 15
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

    async def test_cypher_contains_uses_unwind(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
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

    async def test_1hop_relationships_uses_unwind(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """1-hop relationship expansion should use UNWIND $eids."""
        mp_vector_store.search_relationships = AsyncMock(return_value=[
            {"src_name": "Alice", "type": "WORKS_AT", "tgt_name": "Acme", "fact": "", "score": 0.9},
        ])

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

    async def test_returns_facts_and_entities(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """_search_relates_edges should return fact strings and entity dict."""
        mp_vector_store.search_relationships = AsyncMock(return_value=[
            {
                "src_name": "Alice",
                "type": "WORKS_AT",
                "tgt_name": "Acme Corp",
                "fact": "Alice is a senior engineer",
                "score": 0.95,
            },
        ])

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
        mp_vector_store.search_relationships = AsyncMock(return_value=[
            {
                "src_name": "Bob",
                "type": "KNOWS",
                "tgt_name": "Carol",
                "fact": "",
                "score": 0.8,
            },
        ])

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

    async def test_handles_search_failure(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
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

    async def test_deduplicates_entities(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """Same entity appearing in multiple edges should only appear once."""
        mp_vector_store.search_relationships = AsyncMock(return_value=[
            {"src_name": "Alice", "type": "WORKS_AT", "tgt_name": "Acme", "fact": "f1", "score": 0.9},
            {"src_name": "Alice", "type": "KNOWS", "tgt_name": "Bob", "fact": "f2", "score": 0.8},
        ])

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
                {"src_name": "list.dedup", "type": "BELONGS_TO", "tgt_name": "FalkorDB", "fact": "", "score": 0.9},
                {"src_name": "list.insert", "type": "BELONGS_TO", "tgt_name": "FalkorDB", "fact": "", "score": 0.85},
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
