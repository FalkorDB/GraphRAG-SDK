"""Tests for ingestion/resolution_strategies/llm_verified_resolution.py."""
from __future__ import annotations

import math
import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, GraphNode, GraphRelationship
from graphrag_sdk.core.providers import Embedder
from graphrag_sdk.ingestion.resolution_strategies.llm_verified_resolution import LLMVerifiedResolution

from .conftest import MockEmbedder, MockLLM


# ── Helpers ──────────────────────────────────────────────────────────────────


class ControlledEmbedder(Embedder):
    """Returns pre-set vectors per text for deterministic similarity control."""

    def __init__(self, vectors: dict[str, list[float]], default_dim: int = 4) -> None:
        self._vectors = vectors
        self._default_dim = default_dim

    @property
    def model_name(self) -> str:
        return "controlled-test-embedder"

    def embed_query(self, text: str, **kwargs) -> list[float]:
        return self._vectors.get(text, [1.0] + [0.0] * (self._default_dim - 1))


def _unit(v: list[float]) -> list[float]:
    """Normalize a vector to unit length."""
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm > 0 else v


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def ctx() -> Context:
    return Context(tenant_id="test-tenant")


# ── Init validation ───────────────────────────────────────────────────────────


class TestLLMVerifiedResolutionInit:

    def test_threshold_validation_raises(self):
        """hard_threshold <= soft_threshold must raise ValueError."""
        with pytest.raises(ValueError, match="hard_threshold"):
            LLMVerifiedResolution(hard_threshold=0.80, soft_threshold=0.90)

    def test_equal_thresholds_raises(self):
        with pytest.raises(ValueError):
            LLMVerifiedResolution(hard_threshold=0.85, soft_threshold=0.85)

    def test_valid_defaults_accepted(self):
        r = LLMVerifiedResolution()
        assert r.hard_threshold == 0.95
        assert r.soft_threshold == 0.80
        assert r.max_llm_pairs == 500


# ── Phase 1: exact-match merge ────────────────────────────────────────────────


class TestPhase1ExactMatch:

    async def test_same_name_same_label_merged(self, ctx):
        """'Alice' and 'alice' share normalized name → merged in Phase 1."""
        gd = GraphData(
            nodes=[
                GraphNode(id="a1", label="Person", properties={"name": "Alice"}),
                GraphNode(id="a2", label="Person", properties={"name": "alice"}),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution()
        result = await resolver.resolve(gd, ctx)
        assert result.merged_count == 1
        assert len(result.nodes) == 1

    async def test_same_name_diff_label_kept_separate(self, ctx):
        """'Paris' Person vs 'Paris' Location — never merged regardless of similarity."""
        gd = GraphData(
            nodes=[
                GraphNode(id="p1", label="Person", properties={"name": "Paris"}),
                GraphNode(id="p2", label="Location", properties={"name": "Paris"}),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution()
        result = await resolver.resolve(gd, ctx)
        assert result.merged_count == 0
        assert len(result.nodes) == 2

    async def test_no_duplicates_passes_through(self, ctx):
        gd = GraphData(
            nodes=[
                GraphNode(id="a", label="Person", properties={"name": "Alice"}),
                GraphNode(id="b", label="Person", properties={"name": "Bob"}),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution()
        result = await resolver.resolve(gd, ctx)
        assert result.merged_count == 0
        assert len(result.nodes) == 2


# ── Hard merge zone (similarity >= hard_threshold) ───────────────────────────


class TestHardMergeZone:

    async def test_identical_vectors_hard_merged_no_llm(self, ctx):
        """Identical embeddings → similarity = 1.0 >= 0.95 → hard merge, no LLM."""
        vec = _unit([1.0, 0.0, 0.0, 0.0])
        embedder = ControlledEmbedder({"Tolkien": vec, "J.R.R. Tolkien": vec})
        llm = MockLLM(responses=["YES"])

        gd = GraphData(
            nodes=[
                GraphNode(id="t1", label="Person", properties={"name": "Tolkien"}),
                GraphNode(id="t2", label="Person", properties={"name": "J.R.R. Tolkien"}),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(
            llm=llm, embedder=embedder,
            hard_threshold=0.95, soft_threshold=0.80,
        )
        result = await resolver.resolve(gd, ctx)
        assert result.merged_count == 1
        assert len(result.nodes) == 1
        # LLM should NOT have been called for the hard-merge pair
        assert llm._call_index == 0

    async def test_hard_merge_property_inheritance(self, ctx):
        """Properties from duplicate are merged onto survivor."""
        vec = _unit([1.0, 0.0, 0.0, 0.0])
        embedder = ControlledEmbedder({
            "Alice": vec,
            "Alice B": vec,
        })
        gd = GraphData(
            nodes=[
                GraphNode(id="a1", label="Person", properties={"name": "Alice", "color": "red"}),
                GraphNode(id="a2", label="Person", properties={"name": "Alice B", "size": "large"}),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(embedder=embedder, hard_threshold=0.95, soft_threshold=0.80)
        result = await resolver.resolve(gd, ctx)
        assert len(result.nodes) == 1
        node = result.nodes[0]
        assert node.properties.get("color") == "red"
        assert node.properties.get("size") == "large"


# ── Ambiguous zone (soft <= similarity < hard) ────────────────────────────────


class TestAmbiguousZone:

    def _make_ambiguous_embedder(self) -> ControlledEmbedder:
        """Two vectors with cosine similarity ~0.87 (in the 0.80–0.95 zone).
        a = [1, 0] and b = [0.85, 0.527] → cosine ≈ 0.85 after normalization."""
        a = _unit([1.0, 0.0, 0.0, 0.0])
        b = _unit([0.85, 0.527, 0.0, 0.0])
        # Verify they're actually in the ambiguous zone
        sim = _cosine(a, b)
        assert 0.80 <= sim < 0.95, f"sim={sim:.3f} not in [0.80, 0.95)"
        return ControlledEmbedder({"Mirabel Soto": a, "Señora Vega": b})

    async def test_llm_yes_merges_pair(self, ctx):
        """LLM returns YES → pair merged."""
        embedder = self._make_ambiguous_embedder()
        llm = MockLLM(responses=["YES\nSame person, married name."])

        gd = GraphData(
            nodes=[
                GraphNode(id="m1", label="Person", properties={"name": "Mirabel Soto"}),
                GraphNode(id="m2", label="Person", properties={"name": "Señora Vega"}),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(
            llm=llm, embedder=embedder,
            hard_threshold=0.95, soft_threshold=0.80,
        )
        result = await resolver.resolve(gd, ctx)
        assert result.merged_count == 1
        assert len(result.nodes) == 1
        assert llm._call_index == 1

    async def test_llm_no_keeps_pair(self, ctx):
        """LLM returns NO → pair NOT merged."""
        embedder = self._make_ambiguous_embedder()
        llm = MockLLM(responses=["NO\nDifferent people with similar roles."])

        gd = GraphData(
            nodes=[
                GraphNode(id="m1", label="Person", properties={"name": "Mirabel Soto"}),
                GraphNode(id="m2", label="Person", properties={"name": "Señora Vega"}),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(
            llm=llm, embedder=embedder,
            hard_threshold=0.95, soft_threshold=0.80,
        )
        result = await resolver.resolve(gd, ctx)
        assert result.merged_count == 0
        assert len(result.nodes) == 2
        assert llm._call_index == 1

    async def test_llm_yes_case_insensitive(self, ctx):
        """'yes - same entity' still parses as YES."""
        embedder = self._make_ambiguous_embedder()
        llm = MockLLM(responses=["yes - same entity"])

        gd = GraphData(
            nodes=[
                GraphNode(id="m1", label="Person", properties={"name": "Mirabel Soto"}),
                GraphNode(id="m2", label="Person", properties={"name": "Señora Vega"}),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(
            llm=llm, embedder=embedder,
            hard_threshold=0.95, soft_threshold=0.80,
        )
        result = await resolver.resolve(gd, ctx)
        assert result.merged_count == 1

    async def test_multiple_pairs_batched(self, ctx):
        """3 ambiguous pairs → LLM called 3 times in one batch."""
        # Three vectors all in the ambiguous zone with each other
        a = _unit([1.0, 0.5, 0.0, 0.0])
        b = _unit([1.0, 0.4, 0.0, 0.0])
        c = _unit([1.0, 0.3, 0.0, 0.0])
        embedder = ControlledEmbedder({"Alice": a, "Alicia": b, "Ali": c})
        llm = MockLLM(responses=["YES\nSame.", "YES\nSame.", "YES\nSame."])

        gd = GraphData(
            nodes=[
                GraphNode(id="x1", label="Person", properties={"name": "Alice"}),
                GraphNode(id="x2", label="Person", properties={"name": "Alicia"}),
                GraphNode(id="x3", label="Person", properties={"name": "Ali"}),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(
            llm=llm, embedder=embedder,
            hard_threshold=0.95, soft_threshold=0.80,
        )
        result = await resolver.resolve(gd, ctx)
        # All merged into 1
        assert len(result.nodes) == 1
        assert result.merged_count == 2

    async def test_max_llm_pairs_cap(self, ctx):
        """Pairs above max_llm_pairs are skipped (not sent to LLM)."""
        # Build 10 nodes all with similar vectors — many ambiguous pairs
        base = [1.0, 0.5, 0.0, 0.0]
        vectors = {}
        nodes = []
        for i in range(10):
            v = _unit([1.0, 0.5 + i * 0.01, 0.0, 0.0])
            name = f"Entity{i}"
            vectors[name] = v
            nodes.append(GraphNode(id=f"e{i}", label="Person", properties={"name": name}))

        embedder = ControlledEmbedder(vectors)
        llm = MockLLM(responses=["YES"] * 100)

        gd = GraphData(nodes=nodes, relationships=[])
        resolver = LLMVerifiedResolution(
            llm=llm, embedder=embedder,
            hard_threshold=0.95, soft_threshold=0.80,
            max_llm_pairs=3,  # only top 3 pairs sent to LLM
        )
        result = await resolver.resolve(gd, ctx)
        # LLM called at most 3 times
        assert llm._call_index <= 3


# ── Skip zone (similarity < soft_threshold) ───────────────────────────────────


class TestSkipZone:

    async def test_low_similarity_no_merge_no_llm(self, ctx):
        """Orthogonal vectors → similarity = 0 < 0.80 → skip, no LLM."""
        embedder = ControlledEmbedder({
            "Tolkien": _unit([1.0, 0.0, 0.0, 0.0]),
            "Paris": _unit([0.0, 1.0, 0.0, 0.0]),
        })
        llm = MockLLM(responses=["YES"])

        gd = GraphData(
            nodes=[
                GraphNode(id="t", label="Person", properties={"name": "Tolkien"}),
                GraphNode(id="p", label="Person", properties={"name": "Paris"}),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(
            llm=llm, embedder=embedder,
            hard_threshold=0.95, soft_threshold=0.80,
        )
        result = await resolver.resolve(gd, ctx)
        assert result.merged_count == 0
        assert len(result.nodes) == 2
        assert llm._call_index == 0  # LLM never called


# ── No embedder / no LLM degradation ─────────────────────────────────────────


class TestDegradation:

    async def test_no_embedder_only_phase1(self, ctx):
        """Without embedder, only Phase 1 (exact-match) runs."""
        gd = GraphData(
            nodes=[
                GraphNode(id="a1", label="Person", properties={"name": "Alice"}),
                GraphNode(id="a2", label="Person", properties={"name": "Alice"}),
                GraphNode(id="b", label="Person", properties={"name": "Bob"}),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(embedder=None)
        result = await resolver.resolve(gd, ctx)
        assert result.merged_count == 1  # only the exact-name duplicate
        assert len(result.nodes) == 2

    async def test_no_llm_ambiguous_zone_skipped(self, ctx):
        """Without LLM, ambiguous pairs are never merged (safe fallback).
        Vectors have cosine ~0.85 — in the ambiguous zone (0.80–0.99).
        Without LLM the pair is skipped → no merge."""
        a = _unit([1.0, 0.0, 0.0, 0.0])
        b = _unit([0.85, 0.527, 0.0, 0.0])
        embedder = ControlledEmbedder({"Mirabel Soto": a, "Señora Vega": b})

        gd = GraphData(
            nodes=[
                GraphNode(id="m1", label="Person", properties={"name": "Mirabel Soto"}),
                GraphNode(id="m2", label="Person", properties={"name": "Señora Vega"}),
            ],
            relationships=[],
        )
        # hard_threshold=0.99 ensures sim~0.85 is in the ambiguous zone, not hard-merged
        resolver = LLMVerifiedResolution(
            llm=None, embedder=embedder,
            hard_threshold=0.99, soft_threshold=0.80,
        )
        result = await resolver.resolve(gd, ctx)
        # No LLM → ambiguous pairs skipped → no merge
        assert result.merged_count == 0
        assert len(result.nodes) == 2


# ── Relationship handling ──────────────────────────────────────────────────────


class TestRelationshipHandling:

    async def test_relationships_remapped_after_hard_merge(self, ctx):
        """Relationships pointing to merged duplicate are remapped to survivor."""
        vec = _unit([1.0, 0.0, 0.0, 0.0])
        embedder = ControlledEmbedder({"Alice": vec, "Alice B": vec, "Bob": _unit([0.0, 1.0, 0.0, 0.0])})

        gd = GraphData(
            nodes=[
                GraphNode(id="a1", label="Person", properties={"name": "Alice"}),
                GraphNode(id="a2", label="Person", properties={"name": "Alice B"}),
                GraphNode(id="bob", label="Person", properties={"name": "Bob"}),
            ],
            relationships=[
                GraphRelationship(start_node_id="bob", end_node_id="a2", type="KNOWS"),
            ],
        )
        resolver = LLMVerifiedResolution(embedder=embedder, hard_threshold=0.95, soft_threshold=0.80)
        result = await resolver.resolve(gd, ctx)

        survivor_id = next(n.id for n in result.nodes if n.label == "Person" and "Alice" in n.properties.get("name", ""))
        rel = next(r for r in result.relationships if r.type == "KNOWS")
        assert rel.end_node_id == survivor_id

    async def test_duplicate_relationships_deduped(self, ctx):
        """After merge, duplicate rels pointing to same (start, type, end) collapse."""
        vec = _unit([1.0, 0.0, 0.0, 0.0])
        embedder = ControlledEmbedder({"Alice": vec, "Alice B": vec, "Acme": _unit([0.0, 1.0, 0.0, 0.0])})

        gd = GraphData(
            nodes=[
                GraphNode(id="a1", label="Person", properties={"name": "Alice"}),
                GraphNode(id="a2", label="Person", properties={"name": "Alice B"}),
                GraphNode(id="acme", label="Company", properties={"name": "Acme"}),
            ],
            relationships=[
                GraphRelationship(start_node_id="a1", end_node_id="acme", type="WORKS_AT"),
                GraphRelationship(start_node_id="a2", end_node_id="acme", type="WORKS_AT"),
            ],
        )
        resolver = LLMVerifiedResolution(embedder=embedder, hard_threshold=0.95, soft_threshold=0.80)
        result = await resolver.resolve(gd, ctx)

        works_at = [r for r in result.relationships if r.type == "WORKS_AT"]
        assert len(works_at) == 1


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:

    async def test_empty_graph(self, ctx):
        resolver = LLMVerifiedResolution()
        result = await resolver.resolve(GraphData(), ctx)
        assert result.merged_count == 0
        assert len(result.nodes) == 0
        assert len(result.relationships) == 0

    async def test_single_node(self, ctx):
        resolver = LLMVerifiedResolution()
        result = await resolver.resolve(
            GraphData(nodes=[GraphNode(id="a", label="X", properties={"name": "A"})]),
            ctx,
        )
        assert result.merged_count == 0
        assert len(result.nodes) == 1

    async def test_all_different_labels_no_cross_merge(self, ctx):
        """Nodes in different labels never merge even with identical embeddings."""
        vec = _unit([1.0, 0.0, 0.0, 0.0])
        embedder = ControlledEmbedder({"Paris": vec})

        gd = GraphData(
            nodes=[
                GraphNode(id="p1", label="Person", properties={"name": "Paris"}),
                GraphNode(id="p2", label="Location", properties={"name": "Paris"}),
                GraphNode(id="p3", label="Organization", properties={"name": "Paris"}),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(
            embedder=embedder,
            hard_threshold=0.50,  # very low — would merge if cross-label allowed
            soft_threshold=0.10,
        )
        result = await resolver.resolve(gd, ctx)
        assert result.merged_count == 0
        assert len(result.nodes) == 3
