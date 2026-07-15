"""Tests for IncrementalResolution — resolve a batch against the existing graph."""
from __future__ import annotations

import json

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, GraphNode
from graphrag_sdk.core.providers import Embedder
from graphrag_sdk.ingestion.resolution_strategies.incremental_resolution import (
    IncrementalResolution,
    normalize_name,
)

from .conftest import MockLLM


class WordEmbedder(Embedder):
    """Deterministic bag-of-words embedder — shared words → high cosine."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    @property
    def model_name(self) -> str:
        return "word-embedder"

    def embed_query(self, text: str, **kw):
        toks = str(text).lower().split()
        vec = [0.0] * 64
        for t in toks:
            self._vocab.setdefault(t, len(self._vocab) % 64)
            vec[self._vocab[t] % 64] += 1.0
        return vec or [0.0] * 64


def _ctx():
    return Context(tenant_id="t", latency_budget_ms=5000.0)


# Existing graph nodes (candidates the retriever can return).
GAL_SH = GraphNode(id="gal_sh__person", label="Person",
                   properties={"name": "Gal Sh", "description": "Engineer at FalkorDB."})
GAL_BR = GraphNode(id="gal_br__person", label="Person",
                   properties={"name": "Gal Br", "description": "Designer at another firm."})


def make_retriever(returns):
    async def retriever(name, description, k):
        return returns
    return retriever


class TestIncrementalResolution:
    async def test_links_new_entities_into_existing_node_and_rejects_lookalike(self):
        """gal / gal.sh / Gal Shubeli → merge INTO existing Gal Sh;
        Gal Kurland → new; Gal Br (candidate) → rejected."""
        batch = [
            GraphNode(id="gal__person", label="Person",
                      properties={"name": "gal", "description": "works at FalkorDB"}),
            GraphNode(id="gal.sh__person", label="Person",
                      properties={"name": "gal.sh", "description": "FalkorDB engineer"}),
            GraphNode(id="gal_shubeli__person", label="Person",
                      properties={"name": "Gal Shubeli", "description": "engineer at FalkorDB"}),
            GraphNode(id="gal_kurland__person", label="Person",
                      properties={"name": "Gal Kurland", "description": "researcher elsewhere"}),
        ]
        # LLM sees refs 1..4 = new (gal, gal.sh, Gal Shubeli, Gal Kurland),
        # refs 5,6 = graph (Gal Sh, Gal Br).
        decision = json.dumps({"groups": [
            {"members": [1, 2, 3, 5], "target": 5,
             "canonical": "Gal Shubeli", "type": "Person", "description": "Engineer at FalkorDB."},
            {"members": [4], "target": "new",
             "canonical": "Gal Kurland", "type": "Person", "description": "A researcher."},
        ]})
        resolver = IncrementalResolution(
            llm=MockLLM(responses=[decision]),
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([GAL_SH, GAL_BR]),
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())

        ids = {n.id for n in res.nodes}
        # Merged entity carries the EXISTING graph id, and Gal Kurland is new.
        assert "gal_sh__person" in ids, "new mentions should link into existing Gal Sh"
        assert "gal_kurland__person" in ids, "Gal Kurland stays a new entity"
        assert len(res.nodes) == 2
        # Gal Br was a candidate only — never written as a batch node.
        assert "gal_br__person" not in ids
        # All three gal-variants remap onto the existing node.
        for old in ("gal__person", "gal.sh__person", "gal_shubeli__person"):
            assert res.remap.get(old) == "gal_sh__person"
        merged = next(n for n in res.nodes if n.id == "gal_sh__person")
        assert merged.properties["name"] == "Gal Shubeli"

    async def test_no_candidates_means_new_entity_no_llm(self):
        """A survivor with no graph candidates is created as new — LLM untouched."""
        batch = [GraphNode(id="novel__person", label="Person",
                           properties={"name": "Nadia Q", "description": "brand new person"})]

        class BoomLLM(MockLLM):
            def invoke(self, *a, **k):
                raise AssertionError("LLM must not be called when there are no candidates")

        resolver = IncrementalResolution(
            llm=BoomLLM(responses=[""]),
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([]),  # nothing similar in graph
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())
        assert len(res.nodes) == 1
        assert res.nodes[0].id == "novel__person"

    async def test_free_merge_collapses_same_name_before_llm(self):
        """Same-name/different-type homograph merges for free in stage 1."""
        batch = [
            GraphNode(id="graphrag__concept", label="Concept",
                      properties={"name": "GraphRAG", "description": "graph based RAG technique"}),
            GraphNode(id="graphrag__technology", label="Technology",
                      properties={"name": "GraphRAG", "description": "graph based RAG technique"}),
        ]
        resolver = IncrementalResolution(
            llm=MockLLM(responses=[""]),
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([]),  # no graph yet
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())
        assert len(res.nodes) == 1, "GraphRAG Concept+Technology should free-merge"
        assert res.remap.get("graphrag__technology") == "graphrag__concept"

    async def test_immutable_conflict_flags_review(self):
        """Merging nodes that disagree on an immutable prop flags _needs_review."""
        batch = [
            GraphNode(id="acme_a__org", label="Org",
                      properties={"name": "Acme", "description": "a company", "founded": "2001"}),
            GraphNode(id="acme_b__org", label="Org",
                      properties={"name": "Acme", "description": "a company", "founded": "1998"}),
        ]
        resolver = IncrementalResolution(
            llm=MockLLM(responses=[""]),
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([]),
            immutable_props=("founded",),
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())
        assert len(res.nodes) == 1
        assert res.nodes[0].properties.get("_needs_review") is True
        assert res.nodes[0].properties.get("_merge_conflicts")

    async def test_genuine_homograph_stays_separate(self):
        """Same name, different type, divergent descriptions → NOT free-merged."""
        batch = [
            GraphNode(id="paris__location", label="Location",
                      properties={"name": "Paris", "description": "capital city france europe"}),
            GraphNode(id="paris__person", label="Person",
                      properties={"name": "Paris",
                                  "description": "american media personality celebrity"}),
        ]
        resolver = IncrementalResolution(
            llm=MockLLM(responses=[""]),
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([]),
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())
        assert len(res.nodes) == 2, "distinct homographs must stay separate"

    async def test_malformed_llm_response_leaves_pile_untouched(self):
        """An unparseable partition → fail-safe: no merges applied."""
        batch = [GraphNode(id="xylo__t", label="T",
                           properties={"name": "Xylo", "description": "d"})]
        resolver = IncrementalResolution(
            llm=MockLLM(responses=["not valid json"]),  # consulted (has a candidate)
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([GAL_SH]),
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())
        assert len(res.nodes) == 1
        assert res.nodes[0].id == "xylo__t"

    def test_normalize_name_folds_case_and_separators(self):
        assert normalize_name("GraphRAG-SDK") == "graphrag sdk"
        assert normalize_name("graphrag_sdk") == "graphrag sdk"
        assert normalize_name("  GraphRAG   SDK ") == "graphrag sdk"
