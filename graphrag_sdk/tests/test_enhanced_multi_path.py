"""Tests for enhanced MultiPathRetrieval features (PPR, RRF, budget)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import RetrieverResult
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
    store.search = AsyncMock(return_value=[])
    store.search_entities = AsyncMock(return_value=[])
    store.search_relationships = AsyncMock(return_value=[])
    store.fulltext_search = AsyncMock(return_value=[])
    return store


# -- Enhancement 1: 4-Path Entity Discovery --


class TestEntityDiscoveryPaths:
    async def test_path_c_entity_vector_search(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """Path C: Entity vector search should be called."""
        mp_vector_store.search_entities = AsyncMock(
            return_value=[
                {
                    "id": "vec_ent1",
                    "name": "VecEntity",
                    "description": "found via vector",
                    "score": 0.85,
                },
            ]
        )

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
            enable_ppr=False,
        )
        await s.search("Who is VecEntity?")
        assert mp_vector_store.search_entities.call_count >= 1

    async def test_path_c_threshold(self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm):
        """Path C: Entities below 0.75 threshold should be filtered out."""
        mp_vector_store.search_entities = AsyncMock(
            return_value=[
                {"id": "good", "name": "Good", "description": "", "score": 0.80},
                {"id": "bad", "name": "Bad", "description": "", "score": 0.50},
            ]
        )

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
            enable_ppr=False,
        )
        result = await s.search("test query")

        # Check entities section — "Good" should be present, "Bad" should not
        for item in result.items:
            if item.metadata.get("section") == "entities":
                assert "Good" in item.content
                assert "Bad" not in item.content
                break

    async def test_3_discovery_paths_all_called(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """All 3 entity discovery paths should be attempted."""
        queries_seen = []

        async def capture_query(cypher, params=None):
            queries_seen.append(cypher)
            result = MagicMock()
            if "CONTAINS" in cypher:
                result.result_set = [["ent1", "Alice", "desc"]]
            else:
                result.result_set = []
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=capture_query)

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
            enable_ppr=False,
        )
        await s.search("Who is Alice?")

        # Path A: CONTAINS
        assert any("CONTAINS" in q for q in queries_seen)
        # Path B: Fulltext
        assert mp_vector_store.fulltext_search.call_count >= 1
        # Path C: Entity vector
        assert mp_vector_store.search_entities.call_count >= 1


# -- Enhancement 2: PPR Expansion --


class TestPPRExpansion:
    async def test_ppr_uses_bfs_algorithm(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """PPR subgraph extraction should attempt algo.bfs() first."""
        queries_seen = []

        async def capture_query(cypher, params=None):
            queries_seen.append(cypher)
            result = MagicMock()
            if "CONTAINS" in cypher:
                result.result_set = [["ent1", "Alice", "a person"]]
            else:
                result.result_set = []
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=capture_query)
        mp_vector_store.search_relationships = AsyncMock(
            return_value=[
                {"src_name": "Alice", "type": "KNOWS", "tgt_name": "Bob", "fact": "", "score": 0.9},
            ]
        )

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
            enable_ppr=True,
        )
        await s.search("Who is Alice?")

        # algo.bfs should be attempted (may fall back to Cypher patterns)
        bfs_queries = [q for q in queries_seen if "algo.bfs" in q]
        cypher_fallback = [q for q in queries_seen if "RELATES*1.." in q]
        # Either BFS or Cypher fallback should be present
        assert len(bfs_queries) >= 1 or len(cypher_fallback) >= 1

    async def test_ppr_bfs_fallback_to_cypher_patterns(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """When algo.bfs() fails, should fall back to Cypher variable-length patterns."""
        queries_seen = []
        call_count = {"n": 0}

        async def capture_query(cypher, params=None):
            queries_seen.append(cypher)
            call_count["n"] += 1
            result = MagicMock()
            if "CONTAINS" in cypher:
                result.result_set = [["ent1", "Alice", "a person"]]
            elif "algo.bfs" in cypher:
                raise Exception("algo.bfs not available")
            else:
                result.result_set = []
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=capture_query)
        mp_vector_store.search_relationships = AsyncMock(
            return_value=[
                {"src_name": "Alice", "type": "KNOWS", "tgt_name": "Bob", "fact": "", "score": 0.9},
            ]
        )

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
            enable_ppr=True,
        )
        result = await s.search("Who is Alice?")
        assert isinstance(result, RetrieverResult)

        # After BFS failure, Cypher fallback should have been attempted
        cypher_fallback = [q for q in queries_seen if "RELATES*1.." in q]
        assert len(cypher_fallback) >= 1

    async def test_ppr_enabled_by_default(self):
        """PPR should be enabled by default."""
        store = MagicMock()
        store.query_raw = AsyncMock(return_value=MagicMock(result_set=[]))
        vec = MagicMock()
        vec.search = AsyncMock(return_value=[])
        vec.search_entities = AsyncMock(return_value=[])
        vec.search_relationships = AsyncMock(return_value=[])
        vec.fulltext_search = AsyncMock(return_value=[])

        s = MultiPathRetrieval(
            graph_store=store,
            vector_store=vec,
            embedder=MockEmbedder(dimension=8),
            llm=MockLLM(responses=["test"]),
        )
        assert s._enable_ppr is True

    async def test_ppr_disabled_fallback(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """When PPR is disabled, hop-based expansion should be used."""
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
            enable_ppr=False,
        )
        await s.search("Who is Alice?")

        # Should use 1-hop UNWIND pattern (not PPR subgraph extraction)
        hop_queries = [q for q in queries_seen if "UNWIND $eids" in q and "RELATES" in q]
        assert len(hop_queries) >= 1

    async def test_ppr_graceful_fallback_on_empty_subgraph(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """PPR should fall back to hop-based when subgraph is empty."""
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

        # All queries return empty — PPR subgraph extraction will fail
        mp_graph_store.query_raw = AsyncMock(return_value=MagicMock(result_set=[]))

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
            enable_ppr=True,
        )
        result = await s.search("Who is Alice?")
        assert isinstance(result, RetrieverResult)  # No crash

    async def test_ppr_constructor_params(self):
        """Constructor should accept PPR tuning parameters."""
        store = MagicMock()
        vec = MagicMock()
        s = MultiPathRetrieval(
            graph_store=store,
            vector_store=vec,
            embedder=MockEmbedder(dimension=8),
            llm=MockLLM(responses=["test"]),
            ppr_damping=0.3,
            ppr_max_hops=2,
            ppr_max_nodes=300,
        )
        assert s._ppr_damping == 0.3
        assert s._ppr_max_hops == 2
        assert s._ppr_max_nodes == 300


# -- Enhancement 3: Budget Enforcement --


class TestBudgetEnforcement:
    async def test_budget_exceeded_returns_partial(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """When budget is exceeded, should return partial results without error."""
        # Budget of 1ms — will be exceeded almost immediately
        ctx = Context(latency_budget_ms=1)
        # Add a small delay to guarantee budget expiry
        import asyncio

        await asyncio.sleep(0.01)

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
            enable_ppr=False,
        )
        result = await s.search("Who is Alice?", ctx)
        assert isinstance(result, RetrieverResult)

    async def test_ample_budget_runs_full_pipeline(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """With ample budget, full pipeline should execute."""
        ctx = Context(latency_budget_ms=60_000)  # 60 seconds

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
            enable_ppr=False,
        )
        result = await s.search("Who is Alice?", ctx)
        assert isinstance(result, RetrieverResult)


# -- Enhancement 4: RRF Reranking --


class TestRRFReranking:
    async def test_rrf_reranking_produces_results(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """RRF reranking should produce passage results."""
        mp_vector_store.search = AsyncMock(
            return_value=[
                {"id": "c1", "text": "Alice is an engineer.", "score": 0.9},
                {"id": "c2", "text": "Bob works at Acme.", "score": 0.8},
            ]
        )

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
            enable_ppr=False,
        )
        result = await s.search("Who is Alice?")
        assert isinstance(result, RetrieverResult)

    async def test_deep_mode_flag(self):
        """deep_mode constructor param should be stored."""
        store = MagicMock()
        vec = MagicMock()
        s = MultiPathRetrieval(
            graph_store=store,
            vector_store=vec,
            embedder=MockEmbedder(dimension=8),
            llm=MockLLM(responses=["test"]),
            deep_mode=True,
        )
        assert s._deep_mode is True

    async def test_rrf_queries_entity_ids_for_coverage(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """RRF should query entity IDs per chunk for entity-coverage signal."""
        queries_seen = []

        async def capture_query(cypher, params=None):
            queries_seen.append(cypher)
            result = MagicMock()
            if "CONTAINS" in cypher:
                # Need entities so RRF entity_ids is non-empty
                result.result_set = [["ent1", "Alice", "a person"]]
            else:
                result.result_set = []
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=capture_query)
        mp_vector_store.search = AsyncMock(
            return_value=[
                {"id": "c1", "text": "Alice is an engineer.", "score": 0.9},
            ]
        )

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
            enable_ppr=False,
        )
        await s.search("Who is Alice?")

        # RRF query should collect entity IDs per chunk for coverage scoring
        entity_queries = [q for q in queries_seen if "COLLECT" in q and "eids" in q]
        assert len(entity_queries) >= 1


# -- Deep Mode: Shortest Path Chains --


class TestShortestPathChains:
    async def test_deep_mode_attempts_spaths(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """In deep_mode, algo.SPpaths() should be attempted for entity pairs."""
        queries_seen = []

        async def capture_query(cypher, params=None):
            queries_seen.append(cypher)
            result = MagicMock()
            if "CONTAINS" in cypher:
                result.result_set = [
                    ["ent1", "Alice", "person"],
                    ["ent2", "Bob", "person"],
                ]
            elif "algo.SPpaths" in cypher:
                result.result_set = [
                    [["Alice", "Acme", "Bob"], ["WORKS_AT", "EMPLOYS"]],
                ]
            else:
                result.result_set = []
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=capture_query)
        mp_vector_store.search_relationships = AsyncMock(return_value=[])

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
            enable_ppr=False,
            deep_mode=True,
        )
        await s.search("How is Alice related to Bob?")

        spaths_queries = [q for q in queries_seen if "algo.SPpaths" in q]
        assert len(spaths_queries) >= 1

    async def test_deep_mode_off_no_spaths(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """Without deep_mode, algo.SPpaths() should NOT be called."""
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
            enable_ppr=False,
            deep_mode=False,
        )
        await s.search("Who is Alice?")

        spaths_queries = [q for q in queries_seen if "algo.SPpaths" in q]
        assert len(spaths_queries) == 0

    async def test_spaths_graceful_on_unavailable(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """When algo.SPpaths() is unavailable, should degrade gracefully."""
        async def capture_query(cypher, params=None):
            result = MagicMock()
            if "CONTAINS" in cypher:
                result.result_set = [
                    ["ent1", "Alice", "person"],
                    ["ent2", "Bob", "person"],
                ]
            elif "algo.SPpaths" in cypher:
                raise Exception("algo.SPpaths not available")
            else:
                result.result_set = []
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=capture_query)
        mp_vector_store.search_relationships = AsyncMock(return_value=[])

        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
            enable_ppr=False,
            deep_mode=True,
        )
        result = await s.search("How is Alice related to Bob?")
        assert isinstance(result, RetrieverResult)  # No crash


# -- Retrieval Attribution --


class TestRetrievalAttribution:
    async def test_attribution_includes_ppr_info(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """Retrieval attribution should include PPR-related metadata."""
        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
            enable_ppr=True,
        )
        result = await s.search("Who is Alice?")
        attribution = result.metadata.get("retrieval_attribution", {})
        assert "ppr_enabled" in attribution
        assert "ppr_nodes_scored" in attribution
        assert "shortest_path_chains" in attribution


# -- QC-PPR + Graph TF-IDF --


class TestQCPPRIntegration:
    async def test_edge_embedder_parameter_accepted(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """MultiPathRetrieval should accept edge_embedder parameter."""
        edge_emb = MockEmbedder(dimension=4)
        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
            edge_embedder=edge_emb,
        )
        assert s._edge_embedder is edge_emb

    async def test_qc_ppr_fetches_edge_embeddings(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """When edge_embedder is set, PPR query should request r.embedding."""
        queries_seen = []

        async def capture_query(cypher, params=None):
            queries_seen.append(cypher)
            result = MagicMock()
            if "CONTAINS" in cypher:
                result.result_set = [["ent1", "Alice", "a person"]]
            elif "algo.bfs" in cypher:
                # BFS returns subgraph node IDs so PPR proceeds
                result.result_set = [["ent1"], ["ent2"]]
            elif "r.embedding" in cypher:
                # Edge query with embeddings
                # (src_id, src_name, rel_type, fact, tgt_id, tgt_name, a_pr, b_pr, edge_emb)
                result.result_set = [
                    ["ent1", "Alice", "KNOWS", "Alice knows Bob", "ent2", "Bob", 0.1, 0.05, [0.1, 0.2, 0.3, 0.4]],
                ]
            else:
                result.result_set = []
            return result

        mp_graph_store.query_raw = AsyncMock(side_effect=capture_query)
        mp_vector_store.search = AsyncMock(
            return_value=[
                {"id": "c1", "text": "Alice is an engineer.", "score": 0.9},
            ]
        )

        edge_emb = MockEmbedder(dimension=4)
        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
            edge_embedder=edge_emb,
            enable_ppr=True,
        )
        await s.search("Who is Alice?")

        # PPR edge query should fetch edge embeddings
        emb_queries = [q for q in queries_seen if "r.embedding" in q]
        assert len(emb_queries) >= 1

    async def test_no_edge_embedder_still_works(
        self, mp_graph_store, mp_vector_store, mp_embedder, mp_llm
    ):
        """Without edge_embedder, PPR should still work (uniform transitions)."""
        s = MultiPathRetrieval(
            graph_store=mp_graph_store,
            vector_store=mp_vector_store,
            embedder=mp_embedder,
            llm=mp_llm,
            edge_embedder=None,
            enable_ppr=True,
        )
        result = await s.search("Who is Alice?")
        assert isinstance(result, RetrieverResult)
