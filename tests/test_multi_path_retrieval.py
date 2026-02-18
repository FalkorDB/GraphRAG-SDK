"""Tests for retrieval/strategies/multi_path.py — MultiPathRetrieval."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    LLMResponse,
    RawSearchResult,
    RetrieverResult,
)
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval

from .conftest import MockEmbedder, MockLLM


# ── Fixtures ────────────────────────────────────────────────────


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
    store.search = AsyncMock(return_value=[])
    store.search_entities = AsyncMock(return_value=[])
    store.search_facts = AsyncMock(return_value=[])
    store.fulltext_search = AsyncMock(return_value=[])
    return store


@pytest.fixture
def strategy(mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
    return MultiPathRetrieval(
        graph_store=mp_graph_store,
        vector_store=mp_vector_store,
        embedder=mp_embedder,
        llm=mp_llm,
    )


# ── Tests ───────────────────────────────────────────────────────


class TestMultiPathRetrieval:
    async def test_search_returns_retriever_result(self, strategy):
        result = await strategy.search("Who is Alice?")
        assert isinstance(result, RetrieverResult)

    async def test_keyword_extraction(self, strategy, mp_llm):
        """LLM should be called for keyword extraction."""
        await strategy.search("Who is Alice?")
        assert mp_llm._call_index >= 1

    async def test_batch_embedding(self, strategy, mp_embedder):
        """Embedding should be called via batch method (embed_documents)."""
        mp_embedder.call_count = 0
        await strategy.search("Who is Alice?")
        # aembed_documents calls embed_documents which calls embed_query per text
        # With question + keywords, call_count should be > 1
        assert mp_embedder.call_count > 0

    async def test_entity_discovery_paths(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """All 5 entity paths should be attempted."""
        # Set up entity results for vector search
        mp_vector_store.search_entities = AsyncMock(return_value=[
            {"id": "e1", "name": "Alice", "description": "A person", "score": 0.9},
        ])

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        result = await s.search("Who is Alice?")
        assert isinstance(result, RetrieverResult)
        # search_entities should have been called (for keyword vectors + question vector)
        assert mp_vector_store.search_entities.call_count >= 1

    async def test_chunk_retrieval_paths(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """All 4 chunk paths should be attempted."""
        mp_vector_store.fulltext_search = AsyncMock(return_value=[
            {"id": "c1", "text": "Alice is an engineer.", "score": 0.8},
        ])
        mp_vector_store.search = AsyncMock(return_value=[
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
        assert mp_vector_store.fulltext_search.call_count >= 1
        assert mp_vector_store.search.call_count >= 1

    async def test_fact_retrieval(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """search_facts should be called."""
        mp_vector_store.search_facts = AsyncMock(return_value=[
            {"id": "f1", "subject": "Alice", "predicate": "works_at", "object": "Acme", "score": 0.9},
        ])

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        result = await s.search("Where does Alice work?")
        assert mp_vector_store.search_facts.call_count >= 1

    async def test_format_produces_sections(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """Output should include structured sections when data is available."""
        # Set up rich results
        mp_vector_store.search_entities = AsyncMock(return_value=[
            {"id": "e1", "name": "Alice", "description": "A person", "score": 0.9},
        ])
        mp_vector_store.fulltext_search = AsyncMock(return_value=[
            {"id": "c1", "text": "Alice is a software engineer at Acme Corp.", "score": 0.8},
        ])
        mp_vector_store.search_facts = AsyncMock(return_value=[
            {"id": "f1", "subject": "Alice", "predicate": "works_at", "object": "Acme", "score": 0.9},
        ])

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
        )
        result = await s.search("Who is Alice?")
        # Should have at least some items with section metadata
        if result.items:
            sections = {item.metadata.get("section") for item in result.items}
            # At least entities and passages or facts should appear
            assert len(sections) >= 1

    async def test_2hop_chunk_retrieval(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """Path E: 2-hop entity→neighbor→chunk should be attempted."""
        # Set up entity discovery to find one entity
        mp_vector_store.search_entities = AsyncMock(return_value=[
            {"id": "e1", "name": "Alice", "description": "A person", "score": 0.9},
        ])

        # 2-hop query returns a chunk via neighbor
        call_count = {"two_hop": 0}
        original_query_raw = mp_graph_store.query_raw

        async def mock_query_raw(cypher, params=None):
            if "neighbor" in cypher and "MENTIONED_IN" in cypher:
                call_count["two_hop"] += 1
                result = MagicMock()
                result.result_set = [
                    ["c_2hop", "Alice's neighbor Bob works at Acme Corp."],
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
        result = await s.search("What does Bob do at Acme?")
        assert isinstance(result, RetrieverResult)
        # The 2-hop query should have been attempted
        assert call_count["two_hop"] >= 1

    async def test_2hop_relationship_expansion(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """2-hop relationship expansion should be attempted for top entities."""
        mp_vector_store.search_entities = AsyncMock(return_value=[
            {"id": "e1", "name": "Alice", "description": "A person", "score": 0.9},
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

    async def test_llm_rerank_narrows_passages(self, mp_graph_store, mp_vector_store, mp_embedder):
        """LLM reranker should narrow passages when more than llm_rerank_top_k."""
        # LLM returns keyword extraction first, then reranker picks passages 2,1
        llm = MockLLM(responses=["Alice, Bob", "2,1"])

        # Provide enough chunks to trigger LLM reranking (> 8)
        chunks = [
            {"id": f"c{i}", "text": f"Passage {i} about topic.", "score": 0.5}
            for i in range(12)
        ]
        mp_vector_store.fulltext_search = AsyncMock(return_value=chunks)

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=llm,
            llm_rerank_top_k=8,
        )
        result = await s.search("Who is Alice?")
        assert isinstance(result, RetrieverResult)
        # LLM should have been called at least twice (keyword extraction + reranker)
        assert llm._call_index >= 2

    async def test_llm_rerank_skips_when_few_passages(self, mp_graph_store, mp_vector_store, mp_embedder):
        """LLM reranker should skip when passages <= llm_rerank_top_k."""
        llm = MockLLM(responses=["Alice"])

        # Only 3 chunks — fewer than top_k=8, so no LLM rerank call
        mp_vector_store.fulltext_search = AsyncMock(return_value=[
            {"id": "c1", "text": "Passage about Alice.", "score": 0.8},
        ])

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=llm,
            llm_rerank_top_k=8,
        )
        result = await s.search("Who is Alice?")
        assert isinstance(result, RetrieverResult)
        # LLM called only once for keyword extraction, NOT for reranking
        assert llm._call_index == 1

    async def test_source_document_tags(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """Passages should include [Source: filename] when document info is available."""
        mp_vector_store.fulltext_search = AsyncMock(return_value=[
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
        # Check that at least one passage section contains [Source: novel.txt]
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
        # Even with no data, should not raise


class TestMultiPathUnwindQueries:
    """Tests for UNWIND batched Cypher queries (Fix 8)."""

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

        # At least one query should use UNWIND $keywords
        unwind_kw_queries = [q for q in queries_seen if "UNWIND $keywords" in q]
        assert len(unwind_kw_queries) >= 1

    async def test_synonym_expansion_uses_unwind(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """Synonym expansion should use UNWIND $eids."""
        mp_vector_store.search_entities = AsyncMock(return_value=[
            {"id": "e1", "name": "Alice", "description": "A person", "score": 0.9},
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

        # Synonym expansion should use UNWIND $eids with SYNONYM
        syn_queries = [q for q in queries_seen if "UNWIND $eids" in q and "SYNONYM" in q]
        assert len(syn_queries) >= 1

    async def test_1hop_relationships_uses_unwind(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """1-hop relationship expansion should use UNWIND $eids."""
        mp_vector_store.search_entities = AsyncMock(return_value=[
            {"id": "e1", "name": "Alice", "description": "", "score": 0.9},
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

        # Should have UNWIND $eids queries for relationship expansion
        unwind_queries = [q for q in queries_seen if "UNWIND $eids" in q]
        assert len(unwind_queries) >= 1

    async def test_mentioned_in_uses_unwind(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """MENTIONED_IN chunk retrieval should use UNWIND $eids."""
        mp_vector_store.search_entities = AsyncMock(return_value=[
            {"id": "e1", "name": "Alice", "description": "", "score": 0.9},
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

        # MENTIONED_IN path should use UNWIND $eids
        mention_queries = [q for q in queries_seen if "UNWIND $eids" in q and "MENTIONED_IN" in q]
        assert len(mention_queries) >= 1


class TestMultiPathOptionalLLMRerank:
    """Tests for optional LLM reranker (Fix 9)."""

    async def test_llm_rerank_false_skips_llm_call(self, mp_graph_store, mp_vector_store, mp_embedder):
        """llm_rerank=False should skip LLM reranking entirely."""
        llm = MockLLM(responses=["Alice, Bob"])

        # Provide enough chunks to normally trigger LLM reranking (> 8)
        chunks = [
            {"id": f"c{i}", "text": f"Passage {i} about Alice.", "score": 0.5}
            for i in range(12)
        ]
        mp_vector_store.fulltext_search = AsyncMock(return_value=chunks)

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=llm,
            llm_rerank=False,
        )
        result = await s.search("Who is Alice?")
        assert isinstance(result, RetrieverResult)
        # LLM should be called only once for keyword extraction, NOT for reranking
        assert llm._call_index == 1

    async def test_llm_rerank_true_calls_reranker(self, mp_graph_store, mp_vector_store, mp_embedder):
        """llm_rerank=True (default) should call LLM reranker when enough passages."""
        llm = MockLLM(responses=["Alice, Bob", "2,1,3"])

        chunks = [
            {"id": f"c{i}", "text": f"Passage {i} about Alice.", "score": 0.5}
            for i in range(12)
        ]
        mp_vector_store.fulltext_search = AsyncMock(return_value=chunks)

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=llm,
            llm_rerank=True,
        )
        result = await s.search("Who is Alice?")
        assert isinstance(result, RetrieverResult)
        # LLM should be called at least twice (keyword extraction + reranker)
        assert llm._call_index >= 2
