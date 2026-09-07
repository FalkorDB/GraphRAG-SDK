"""Tests for ingestion/resolution_strategies/llm_verified_resolution.py."""

from __future__ import annotations

import logging
import math

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    GraphData,
    GraphNode,
    GraphRelationship,
    LLMResponse,
)
from graphrag_sdk.core.providers import Embedder
from graphrag_sdk.ingestion.resolution_strategies.base import (
    flatten_remap,
    remap_relationships,
)
from graphrag_sdk.ingestion.resolution_strategies.llm_verified_resolution import (
    _RULES,
    LLMVerifiedResolution,
    _pair_distance,
    _is_acronym,
    _name_tokens,
    _parse_batch_verdicts,
)

from .conftest import MockLLM

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


def _angle(theta: float) -> list[float]:
    """Unit vector at a fixed angle, so pairwise cosines are exactly known."""
    return _unit([math.cos(theta), math.sin(theta), 0.0, 0.0])


class PairScriptedLLM(MockLLM):
    """Answers YES/NO by looking at which two names are in the prompt.

    ``abatch_invoke`` runs prompts concurrently, so a positional response list
    is not deterministic — the Nth call is not reliably the Nth pair. Keying on
    prompt content removes that race from the test.
    """

    def __init__(self, verdicts: dict[frozenset[str], bool]) -> None:
        super().__init__(responses=["NO\nunscripted pair"])
        self._verdicts = verdicts
        self.asked: list[frozenset[str]] = []

    def invoke(self, prompt: str, **kwargs):
        for names, verdict in self._verdicts.items():
            if all(f"Name: {n}\n" in prompt for n in names):
                self.asked.append(names)
                self._call_index += 1
                return LLMResponse(content="YES\nsame" if verdict else "NO\ndifferent")
        self._call_index += 1
        return LLMResponse(content="NO\nunscripted")


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
        # 0.65, not 0.80: the 0.65-0.80 band held 31% of real duplicate pairs
        # that no stage asked about. See the sweep in RESULTS.md P3.25.
        assert r.soft_threshold == 0.65
        assert r.max_llm_pairs == 500

    def test_default_soft_threshold_admits_the_previously_dead_band(self):
        """A same-label pair at 0.70 must reach the LLM under the default.

        Under the old 0.80 default this pair was dropped before any LLM call
        and could never merge. Guards the P3.25 change against a silent revert.
        """
        r = LLMVerifiedResolution()
        assert r.soft_threshold <= 0.70 < r.hard_threshold


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

    async def test_same_name_diff_label_preserved_without_evidence(self, ctx):
        """'Paris' Person vs 'Paris' Location with no descriptions — homographs
        are preserved (fail-safe when the LLM cannot verify they're the same
        real-world entity)."""
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
        labels = {n.label for n in result.nodes}
        assert labels == {"Person", "Location"}

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
            llm=llm,
            embedder=embedder,
            hard_threshold=0.95,
            soft_threshold=0.80,
        )
        result = await resolver.resolve(gd, ctx)
        assert result.merged_count == 1
        assert len(result.nodes) == 1
        # LLM should NOT have been called for the hard-merge pair
        assert llm._call_index == 0

    async def test_hard_merge_property_inheritance(self, ctx):
        """Properties from duplicate are merged onto survivor."""
        vec = _unit([1.0, 0.0, 0.0, 0.0])
        embedder = ControlledEmbedder(
            {
                "Alice": vec,
                "Alice B": vec,
            }
        )
        gd = GraphData(
            nodes=[
                GraphNode(id="a1", label="Person", properties={"name": "Alice", "color": "red"}),
                GraphNode(id="a2", label="Person", properties={"name": "Alice B", "size": "large"}),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(
            embedder=embedder, hard_threshold=0.95, soft_threshold=0.80
        )
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
            llm=llm,
            embedder=embedder,
            hard_threshold=0.95,
            soft_threshold=0.80,
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
            llm=llm,
            embedder=embedder,
            hard_threshold=0.95,
            soft_threshold=0.80,
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
            llm=llm,
            embedder=embedder,
            hard_threshold=0.95,
            soft_threshold=0.80,
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
            llm=llm,
            embedder=embedder,
            hard_threshold=0.95,
            soft_threshold=0.80,
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
            llm=llm,
            embedder=embedder,
            hard_threshold=0.95,
            soft_threshold=0.80,
            max_llm_pairs=3,  # only top 3 pairs sent to LLM
        )
        result = await resolver.resolve(gd, ctx)
        # LLM called at most 3 times
        assert llm._call_index <= 3


# ── Skip zone (similarity < soft_threshold) ───────────────────────────────────


class TestSkipZone:
    async def test_low_similarity_no_merge_no_llm(self, ctx):
        """Orthogonal vectors → similarity = 0 < 0.80 → skip, no LLM."""
        embedder = ControlledEmbedder(
            {
                "Tolkien": _unit([1.0, 0.0, 0.0, 0.0]),
                "Paris": _unit([0.0, 1.0, 0.0, 0.0]),
            }
        )
        llm = MockLLM(responses=["YES"])

        gd = GraphData(
            nodes=[
                GraphNode(id="t", label="Person", properties={"name": "Tolkien"}),
                GraphNode(id="p", label="Person", properties={"name": "Paris"}),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(
            llm=llm,
            embedder=embedder,
            hard_threshold=0.95,
            soft_threshold=0.80,
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
            llm=None,
            embedder=embedder,
            hard_threshold=0.99,
            soft_threshold=0.80,
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
        embedder = ControlledEmbedder(
            {"Alice": vec, "Alice B": vec, "Bob": _unit([0.0, 1.0, 0.0, 0.0])}
        )

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
        resolver = LLMVerifiedResolution(
            embedder=embedder, hard_threshold=0.95, soft_threshold=0.80
        )
        result = await resolver.resolve(gd, ctx)

        survivor_id = next(
            n.id
            for n in result.nodes
            if n.label == "Person" and "Alice" in n.properties.get("name", "")
        )
        rel = next(r for r in result.relationships if r.type == "KNOWS")
        assert rel.end_node_id == survivor_id

    async def test_duplicate_relationships_deduped(self, ctx):
        """After merge, duplicate rels pointing to same (start, type, end) collapse."""
        vec = _unit([1.0, 0.0, 0.0, 0.0])
        embedder = ControlledEmbedder(
            {"Alice": vec, "Alice B": vec, "Acme": _unit([0.0, 1.0, 0.0, 0.0])}
        )

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
        resolver = LLMVerifiedResolution(
            embedder=embedder, hard_threshold=0.95, soft_threshold=0.80
        )
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

    async def test_all_different_labels_preserved_without_llm_confirmation(self, ctx):
        """Same-name cross-label nodes are preserved when the LLM can't
        verify them as the same real-world entity (here: no descriptions,
        so Phase 1 never asks the LLM). Homographs such as Paris/Person vs
        Paris/Location must not be silently fused."""
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
            hard_threshold=0.50,
            soft_threshold=0.10,
        )
        result = await resolver.resolve(gd, ctx)
        assert result.merged_count == 0
        assert len(result.nodes) == 3
        labels = {n.label for n in result.nodes}
        assert labels == {"Person", "Location", "Organization"}


# ── Batched verification ─────────────────────────────────────────────────────


class TestParseBatchVerdicts:
    """Reading verdicts out of a batched reply."""

    def test_numbered_lines(self):
        text = "1. A vs B | rule: 1a | NO\n2. C vs D | rule: 3 | YES"
        assert _parse_batch_verdicts(text, 2) == {1: False, 2: True}

    def test_takes_last_verdict_on_line(self):
        """The rule text may mention a verdict; only the final one counts."""
        text = "1. A vs B | rule: 1b says YES when it is only a fuller label | YES"
        assert _parse_batch_verdicts(text, 1) == {1: True}

    def test_ignores_out_of_range_numbers(self):
        text = "1. A vs B | YES\n7. stray line | YES"
        assert _parse_batch_verdicts(text, 2) == {1: True}

    def test_missing_pair_is_absent_not_false(self):
        """An unanswered pair must be absent, so callers can tell it apart."""
        assert _parse_batch_verdicts("2. C vs D | YES", 2) == {2: True}

    def test_garbage_yields_nothing(self):
        assert _parse_batch_verdicts("I cannot help with that.", 3) == {}
        assert _parse_batch_verdicts("", 3) == {}

    def test_bare_verdict_accepted_for_single_pair(self):
        assert _parse_batch_verdicts("YES\nSame person.", 1) == {1: True}
        assert _parse_batch_verdicts("NO\nDifferent.", 1) == {1: False}

    def test_bare_verdict_rejected_for_many_pairs(self):
        """With several pairs a bare verdict is ambiguous and must not merge."""
        assert _parse_batch_verdicts("YES", 4) == {}


class TestPacking:
    """Splitting pairs across calls by token budget."""

    def _resolver(self, budget: int) -> LLMVerifiedResolution:
        return LLMVerifiedResolution(verification_token_budget=budget)

    def test_small_input_is_one_call(self):
        blocks = ["short pair block"] * 5
        assert self._resolver(20000)._pack(blocks) == [(0, 5)]

    def test_budget_splits_into_several_calls(self):
        blocks = ["word " * 200] * 10
        windows = self._resolver(1200)._pack(blocks)
        assert len(windows) > 1
        assert windows[0][0] == 0
        assert windows[-1][1] == len(blocks)

    def test_windows_are_contiguous_and_cover_everything(self):
        blocks = ["word " * 50] * 23
        windows = self._resolver(900)._pack(blocks)
        assert windows[0][0] == 0
        assert windows[-1][1] == 23
        for (_, end), (nxt, _) in zip(windows, windows[1:]):
            assert end == nxt

    def test_oversized_pair_still_gets_a_call(self):
        """A pair larger than the whole budget must not be silently dropped."""
        blocks = ["word " * 5000, "word " * 5000]
        windows = self._resolver(10)._pack(blocks)
        assert windows == [(0, 1), (1, 2)]

    def test_tiny_budget_gives_one_pair_per_call(self):
        """The documented way to trade cost for the highest-precision mode."""
        blocks = ["a pair"] * 6
        assert self._resolver(0)._pack(blocks) == [(i, i + 1) for i in range(6)]


class TestBatchedVerification:
    """End-to-end behaviour of the batched path."""

    def _embedder(self) -> ControlledEmbedder:
        a = _unit([1.0, 0.0, 0.0, 0.0])
        b = _unit([0.85, 0.527, 0.0, 0.0])
        return ControlledEmbedder({"Mirabel Soto": a, "Señora Vega": b})

    def _graph(self) -> GraphData:
        return GraphData(
            nodes=[
                GraphNode(id="m1", label="Person", properties={"name": "Mirabel Soto"}),
                GraphNode(id="m2", label="Person", properties={"name": "Señora Vega"}),
            ],
            relationships=[],
        )

    async def test_batched_yes_merges(self, ctx):
        llm = MockLLM(responses=["1. Mirabel Soto vs Señora Vega | rule: 3 | YES"])
        resolver = LLMVerifiedResolution(llm=llm, embedder=self._embedder())
        result = await resolver.resolve(self._graph(), ctx)
        assert result.merged_count == 1
        assert len(result.nodes) == 1

    async def test_batched_no_keeps_both(self, ctx):
        llm = MockLLM(responses=["1. Mirabel Soto vs Señora Vega | rule: 1a | NO"])
        resolver = LLMVerifiedResolution(llm=llm, embedder=self._embedder())
        result = await resolver.resolve(self._graph(), ctx)
        assert len(result.nodes) == 2

    async def test_unreadable_reply_does_not_merge(self, ctx):
        """A malformed batch reply must lose merges, never invent one."""
        llm = MockLLM(responses=["Sorry, I can't determine that."])
        resolver = LLMVerifiedResolution(llm=llm, embedder=self._embedder())
        result = await resolver.resolve(self._graph(), ctx)
        assert len(result.nodes) == 2

    async def test_one_call_for_many_pairs(self, ctx):
        """Many pairs must cost dramatically fewer calls than asking per pair.

        This asserted a fixed ``<= 5`` when one call carried up to 30 pairs. The
        default is now 5 pairs per call (RESULTS.md P3.39: 30 per call loses
        ~10 merges of 73, 5 per call loses ~2 and is also the fastest arm), so a
        hard-coded ceiling would just have to be rewritten again on the next
        tuning change.

        Both modes are therefore measured on the SAME graph and compared to each
        other. That is the property batching actually claims, and it holds
        whatever the cap is set to.
        """
        vectors = {}
        for i in range(10):
            vectors[f"Name A{i}"] = _unit([1.0, 0.0, 0.0, float(i) * 1e-6])
            vectors[f"Name B{i}"] = _unit([0.85, 0.527, 0.0, float(i) * 1e-6])

        def graph():
            return GraphData(
                nodes=[
                    GraphNode(
                        id=f"{side}{i}",
                        label="Person",
                        properties={"name": f"Name {side}{i}"},
                    )
                    for i in range(10)
                    for side in ("A", "B")
                ],
                relationships=[],
            )

        batched_llm = MockLLM(
            responses=["\n".join(f"{n}. x vs y | rule: 4 | NO" for n in range(1, 200))]
        )
        await LLMVerifiedResolution(
            llm=batched_llm, embedder=ControlledEmbedder(vectors)
        ).resolve(graph(), ctx)

        per_pair_llm = MockLLM(responses=["NO"] * 500)
        await LLMVerifiedResolution(
            llm=per_pair_llm,
            embedder=ControlledEmbedder(vectors),
            batch_verification=False,
        ).resolve(graph(), ctx)

        assert batched_llm._call_index * 3 <= per_pair_llm._call_index, (
            f"batched used {batched_llm._call_index} calls against "
            f"{per_pair_llm._call_index} per-pair; batching must collapse many "
            "pairs into far fewer calls"
        )

    async def test_per_pair_mode_costs_far_more_calls(self, ctx):
        """Control: the bound above is only meaningful next to the alternative."""
        vectors = {}
        for i in range(10):
            vectors[f"Name A{i}"] = _unit([1.0, 0.0, 0.0, float(i) * 1e-6])
            vectors[f"Name B{i}"] = _unit([0.85, 0.527, 0.0, float(i) * 1e-6])
        gd = GraphData(
            nodes=[
                GraphNode(
                    id=f"{side}{i}",
                    label="Person",
                    properties={"name": f"Name {side}{i}"},
                )
                for i in range(10)
                for side in ("A", "B")
            ],
            relationships=[],
        )
        llm = MockLLM(responses=["NO"] * 500)
        resolver = LLMVerifiedResolution(
            llm=llm, embedder=ControlledEmbedder(vectors), batch_verification=False
        )
        await resolver.resolve(gd, ctx)
        assert llm._call_index > 5

    async def test_per_pair_mode_still_works(self, ctx):
        """batch_verification=False keeps the original one-call-per-pair path."""
        llm = MockLLM(responses=["YES\nSame person, married name."])
        resolver = LLMVerifiedResolution(
            llm=llm, embedder=self._embedder(), batch_verification=False
        )
        result = await resolver.resolve(self._graph(), ctx)
        assert result.merged_count == 1
        assert llm._call_index == 1


# ── PASS 2: cross-label merge ────────────────────────────────────────────────


class TestCrossLabelDoors:
    """The three candidate doors are pure functions — test them directly."""

    def test_token_door_ignores_case_and_punctuation(self):
        assert _name_tokens("The Boeing Co. 737") == {"boeing", "737"}
        assert _name_tokens("Naseem Ali") & _name_tokens("naseem")

    def test_token_door_drops_stopwords(self):
        """'Bank of America' and 'Museum of Art' must not pair on 'of'."""
        assert not (_name_tokens("Bank of America") & _name_tokens("Museum of Art"))

    def test_acronym_door_both_directions(self):
        assert _is_acronym("MIT", "Massachusetts Institute of Technology")
        assert _is_acronym("Massachusetts Institute of Technology", "MIT")

    def test_acronym_door_rejects_single_letter(self):
        """A one-letter 'acronym' matches far too much to be evidence."""
        assert not _is_acronym("A", "Acme")

    def test_acronym_door_rejects_non_matching_initials(self):
        assert not _is_acronym("IBM", "Massachusetts Institute of Technology")


class TestFlattenRemap:
    """Each pass merges the previous pass's survivors, so chains form."""

    def test_two_hop_chain_collapses(self):
        assert flatten_remap({"dup": "A", "A": "B"}) == {"dup": "B", "A": "B"}

    def test_deep_chain_collapses(self):
        assert flatten_remap({"a": "b", "b": "c", "c": "d"}) == {"a": "d", "b": "d", "c": "d"}

    def test_no_chain_is_unchanged(self):
        assert flatten_remap({"a": "z", "b": "z"}) == {"a": "z", "b": "z"}

    def test_cycle_terminates(self):
        """A malformed cyclic mapping must not hang."""
        assert flatten_remap({"a": "b", "b": "a"}) == {"a": "a", "b": "b"}

    def test_relationship_into_chained_node_reaches_final_survivor(self):
        """The bug this exists to prevent: an edge left pointing at a
        removed intermediate node."""
        rels = [GraphRelationship(start_node_id="dup", end_node_id="x", type="R", properties={})]
        out = remap_relationships(rels, flatten_remap({"dup": "A", "A": "B"}))
        assert out[0].start_node_id == "B"


class TestCrossLabelPass:
    def _embedder(self, sim_texts: dict[str, list[float]]) -> ControlledEmbedder:
        return ControlledEmbedder(sim_texts)

    def _far_apart(self) -> ControlledEmbedder:
        """Every text gets its own near-orthogonal vector."""
        return ControlledEmbedder({})

    async def test_cross_label_pair_merged_on_yes(self, ctx):
        """PASS 1 groups by label so this pair can only come from PASS 2."""
        a = _unit([1.0, 0.0, 0.0, 0.0])
        b = _unit([0.95, 0.31, 0.0, 0.0])
        embedder = ControlledEmbedder(
            {
                "Ada Lovelace: Mathematician.": a,
                "Ada Lovelace": a,
                "A. Lovelace: Mathematician.": b,
                "A. Lovelace": b,
            }
        )
        llm = MockLLM(responses=["YES\nSame person."])
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="p1",
                    label="Person",
                    properties={"name": "Ada Lovelace", "description": "Mathematician."},
                ),
                GraphNode(
                    id="p2",
                    label="Engineer",
                    properties={"name": "A. Lovelace", "description": "Mathematician."},
                ),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder)
        result = await resolver.resolve(gd, ctx)
        assert result.merged_count == 1
        assert len(result.nodes) == 1

    async def test_cross_label_pair_kept_on_no(self, ctx):
        a = _unit([1.0, 0.0, 0.0, 0.0])
        b = _unit([0.95, 0.31, 0.0, 0.0])
        embedder = ControlledEmbedder(
            {
                "Paris: Capital of France.": a,
                "Paris": a,
                "Paris: Prince of Troy.": b,
                "Paris ": b,
            }
        )
        llm = MockLLM(responses=["NO\nDifferent entities."])
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="c1",
                    label="City",
                    properties={"name": "Paris", "description": "Capital of France."},
                ),
                GraphNode(
                    id="c2",
                    label="Person",
                    properties={"name": "Paris ", "description": "Prince of Troy."},
                ),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder)
        result = await resolver.resolve(gd, ctx)
        assert result.merged_count == 0
        assert len(result.nodes) == 2

    async def test_rank_floor_rejects_without_an_llm_call(self, ctx):
        """Same name, unrelated meaning: the floor must reject it for free.

        Measured: Paris/Paris scores 0.387 and Apple/Apple 0.437 on real
        embeddings, both under the 0.55 floor.
        """
        a = _unit([1.0, 0.0, 0.0, 0.0])
        b = _unit([0.2, 1.0, 0.0, 0.0])
        embedder = ControlledEmbedder(
            {
                "Apple: Fruit.": a,
                "Apple": a,
                "Apple : Technology company.": b,
                "Apple ": b,
            }
        )
        llm = MockLLM(responses=["YES\nWould be wrong."])
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="f1", label="Fruit", properties={"name": "Apple", "description": "Fruit."}
                ),
                GraphNode(
                    id="o1",
                    label="Organization",
                    properties={"name": "Apple ", "description": "Technology company."},
                ),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder)
        result = await resolver.resolve(gd, ctx)
        assert result.merged_count == 0
        assert llm._call_index == 0, "floor should reject before any LLM call"

    async def test_same_label_pairs_are_not_reconsidered(self, ctx):
        """PASS 1 owns same-label pairs; PASS 2 must not re-ask them."""
        v = _unit([1.0, 0.0, 0.0, 0.0])
        embedder = ControlledEmbedder(
            {
                "Ada: One.": v,
                "Ada": v,
                "Ada Lovelace: Two.": v,
                "Ada Lovelace": v,
            }
        )
        llm = MockLLM(responses=["NO\nno."])
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="s1", label="Person", properties={"name": "Ada", "description": "One."}
                ),
                GraphNode(
                    id="s2",
                    label="Person",
                    properties={"name": "Ada Lovelace", "description": "Two."},
                ),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder)
        await resolver.resolve(gd, ctx)
        # identical vectors -> phase 2 hard-merges without the LLM; PASS 2 then
        # has a single surviving node and forms no pairs at all.
        assert llm._call_index == 0

    async def test_disabled_flag_skips_the_pass(self, ctx):
        a = _unit([1.0, 0.0, 0.0, 0.0])
        b = _unit([0.95, 0.31, 0.0, 0.0])
        embedder = ControlledEmbedder(
            {
                "Ada Lovelace: Mathematician.": a,
                "Ada Lovelace": a,
                "A. Lovelace: Mathematician.": b,
                "A. Lovelace": b,
            }
        )
        llm = MockLLM(responses=["YES\nSame person."])
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="p1",
                    label="Person",
                    properties={"name": "Ada Lovelace", "description": "Mathematician."},
                ),
                GraphNode(
                    id="p2",
                    label="Engineer",
                    properties={"name": "A. Lovelace", "description": "Mathematician."},
                ),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(
            llm=llm, embedder=embedder, cross_label_merge=False, unified_stage=False
        )
        result = await resolver.resolve(gd, ctx)
        assert result.merged_count == 0
        assert llm._call_index == 0

    async def test_relationships_follow_the_cross_label_survivor(self, ctx):
        a = _unit([1.0, 0.0, 0.0, 0.0])
        b = _unit([0.95, 0.31, 0.0, 0.0])
        # ControlledEmbedder falls back to [1,0,0,0] for unlisted text, which
        # would collide exactly with `a`. Every node needs an explicit vector.
        far = _unit([0.0, 0.0, 1.0, 0.0])
        embedder = ControlledEmbedder(
            {
                "Ada Lovelace: Mathematician.": a,
                "Ada Lovelace": a,
                "A. Lovelace: Mathematician.": b,
                "A. Lovelace": b,
                "Analytical Engine Ltd: A company.": far,
                "Analytical Engine Ltd": far,
            }
        )
        llm = MockLLM(responses=["YES\nSame person."])
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="p1",
                    label="Person",
                    properties={"name": "Ada Lovelace", "description": "Mathematician."},
                ),
                GraphNode(
                    id="p2",
                    label="Engineer",
                    properties={"name": "A. Lovelace", "description": "Mathematician."},
                ),
                GraphNode(
                    id="o1",
                    label="Organization",
                    properties={"name": "Analytical Engine Ltd", "description": "A company."},
                ),
            ],
            relationships=[
                GraphRelationship(
                    start_node_id="p2", end_node_id="o1", type="WORKS_AT", properties={}
                ),
            ],
        )
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder)
        result = await resolver.resolve(gd, ctx)
        survivors = {n.id for n in result.nodes}
        assert survivors == {"p1", "o1"}, "p2 must be absorbed into p1"
        assert len(result.relationships) == 1
        assert result.relationships[0].start_node_id == "p1"

    async def test_transitive_closure_repairs_a_rejected_pair(self, ctx):
        """A=C and B=C must group all three even when the LLM rejects A/B.

        Measured on real providers: the model answered NO to 'Na'/'Naseem'
        while confirming both against 'Naseem Ali'. Because every pair is asked
        against the ORIGINAL entities and nothing merges until all answers are
        back, the union step recovers the correct cluster.

        The vectors are placed at fixed angles so the pair order is known:
        A/C = cos(0.20) = 0.980, B/C = cos(0.40) = 0.921, A/B = cos(0.60) =
        0.825. All three clear the 0.60 vector door and the 0.55 floor. Merging
        greedily in that order instead of using union-find loses the cluster,
        which is what makes this test able to fail.
        """
        a, b, c = _angle(0.0), _angle(0.60), _angle(0.20)
        embedder = ControlledEmbedder(
            {
                "Ada Lovelace: Mathematician.": a,
                "Ada Lovelace": a,
                "A. Lovelace: Mathematician.": b,
                "A. Lovelace": b,
                "Lovelace: Mathematician.": c,
                "Lovelace": c,
            }
        )
        llm = PairScriptedLLM(
            {
                frozenset({"Ada Lovelace", "Lovelace"}): True,
                frozenset({"A. Lovelace", "Lovelace"}): True,
                frozenset({"Ada Lovelace", "A. Lovelace"}): False,
            }
        )
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="x1",
                    label="Person",
                    properties={"name": "Ada Lovelace", "description": "Mathematician."},
                ),
                GraphNode(
                    id="x2",
                    label="Engineer",
                    properties={"name": "A. Lovelace", "description": "Mathematician."},
                ),
                GraphNode(
                    id="x3",
                    label="Author",
                    properties={"name": "Lovelace", "description": "Mathematician."},
                ),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder, unified_stage=False)
        result = await resolver.resolve(gd, ctx)
        assert llm._call_index == 3, "all three pairs must be asked"
        assert len(result.nodes) == 1, "two YES answers must chain all three"

    async def test_multi_pass_remap_chain_is_flattened_end_to_end(self, ctx):
        """Phase 1 records dup -> A; PASS 2 then absorbs A into B.

        Without flattening, an edge into `dup` is rewritten to `A` — a node
        that resolve() already removed — leaving a dangling reference. Node
        order matters: PASS 2 keeps the earliest surviving node, so the
        Engineer is listed first to make it the final survivor.
        """
        far, near = _angle(0.0), _angle(0.60)
        embedder = ControlledEmbedder(
            {
                "A. Lovelace: M.": far,
                "A. Lovelace": far,
                "Ada Lovelace: M.": near,
                "Ada Lovelace": near,
                "ADA LOVELACE: M.": near,
                "ADA LOVELACE": near,
            }
        )
        llm = PairScriptedLLM(
            {
                frozenset({"A. Lovelace", "Ada Lovelace"}): True,
            }
        )
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="e1",
                    label="Engineer",
                    properties={"name": "A. Lovelace", "description": "M."},
                ),
                GraphNode(
                    id="p1",
                    label="Person",
                    properties={"name": "Ada Lovelace", "description": "M."},
                ),
                GraphNode(
                    id="p2",
                    label="Person",
                    properties={"name": "ADA LOVELACE", "description": "M."},
                ),
            ],
            relationships=[
                GraphRelationship(start_node_id="p2", end_node_id="other", type="R", properties={}),
            ],
        )
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder)
        result = await resolver.resolve(gd, ctx)

        assert [n.id for n in result.nodes] == ["e1"]
        # p2 -> p1 (phase 1) -> e1 (PASS 2). Both hops must collapse.
        assert result.remap["p2"] == "e1"
        assert result.relationships[0].start_node_id == "e1"
        survivors = {n.id for n in result.nodes}
        for rel in result.relationships:
            assert rel.start_node_id in survivors, "dangling edge after remap"

    async def test_vector_door_admits_a_pair_with_no_shared_token(self, ctx):
        """The door that only the embedding can open.

        "Munich"/"Munchen" share no token and neither spells the other's
        initials, yet they are the same city (measured cosine 0.609 on real
        embeddings). Dropping this door was measured to cost recall
        1.000 -> 0.800 on the cross-label corpus.
        """
        a, b = _angle(0.0), _angle(0.90)  # cos(0.90) = 0.622: over the 0.60 door
        assert 0.60 <= _cosine(a, b) < 0.95
        embedder = ControlledEmbedder(
            {
                "Munich: A city in Bavaria.": a,
                "Munich": a,
                "Munchen: A city in Bavaria.": b,
                "Munchen": b,
            }
        )
        assert not (_name_tokens("Munich") & _name_tokens("Munchen"))
        assert not _is_acronym("Munich", "Munchen")
        llm = PairScriptedLLM({frozenset({"Munich", "Munchen"}): True})
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="g1",
                    label="City",
                    properties={"name": "Munich", "description": "A city in Bavaria."},
                ),
                GraphNode(
                    id="g2",
                    label="Location",
                    properties={"name": "Munchen", "description": "A city in Bavaria."},
                ),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder, unified_stage=False)
        result = await resolver.resolve(gd, ctx)
        assert llm._call_index == 1, "only the vector door can form this pair"
        assert len(result.nodes) == 1

    async def test_nickname_clause_present_in_rules(self):
        """Measured: without it the model rejects Bob/Robert on name alone."""
        assert "diminutive" in _RULES
        assert "Bob for Robert" in _RULES


class TestCrossLabelMergedProperties:
    """A cross-label merge must not throw away the evidence that justified it.

    The naive "copy only keys the survivor lacks" rule keeps the survivor's
    description and silently drops the duplicate's — the opposite of what the
    same-label path in base.py does, and a loss of exactly the text the LLM
    used to decide the two were the same entity.
    """

    @pytest.mark.asyncio
    async def test_both_descriptions_are_kept(self, ctx):
        a = GraphNode(
            id="a",
            label="Person",
            properties={"name": "Ada Lovelace", "description": "DESC-PERSON"},
        )
        b = GraphNode(
            id="b",
            label="Engineer",
            properties={"name": "A. Lovelace", "description": "DESC-ENGINEER"},
        )
        emb = ControlledEmbedder(
            {
                "Ada Lovelace: DESC-PERSON": [1.0, 0.0],
                "A. Lovelace: DESC-ENGINEER": [0.99, 0.14],
                "Ada Lovelace": [1.0, 0.0],
                "A. Lovelace": [0.0, 1.0],
            }
        )
        llm = PairScriptedLLM({frozenset({"Ada Lovelace", "A. Lovelace"}): True})
        r = LLMVerifiedResolution(llm=llm, embedder=emb)
        out = await r.resolve(GraphData(nodes=[a, b], relationships=[]), ctx)

        assert len(out.nodes) == 1
        desc = out.nodes[0].properties["description"]
        assert "DESC-PERSON" in desc
        assert "DESC-ENGINEER" in desc

    @pytest.mark.asyncio
    async def test_identical_descriptions_are_not_duplicated(self, ctx):
        """Two chunks often emit the same sentence; joining it twice is noise."""
        a = GraphNode(
            id="a",
            label="Person",
            properties={"name": "Ada Lovelace", "description": "SAME"},
        )
        b = GraphNode(
            id="b",
            label="Engineer",
            properties={"name": "A. Lovelace", "description": "SAME"},
        )
        emb = ControlledEmbedder(
            {
                "Ada Lovelace: SAME": [1.0, 0.0],
                "A. Lovelace: SAME": [0.99, 0.14],
                "Ada Lovelace": [1.0, 0.0],
                "A. Lovelace": [0.0, 1.0],
            }
        )
        llm = PairScriptedLLM({frozenset({"Ada Lovelace", "A. Lovelace"}): True})
        r = LLMVerifiedResolution(llm=llm, embedder=emb)
        out = await r.resolve(GraphData(nodes=[a, b], relationships=[]), ctx)

        assert out.nodes[0].properties["description"] == "SAME"


@pytest.mark.asyncio
class TestSameLabelMergedDescriptions:
    """Phase 2-5 (same-label) had the same description-loss defect PASS 2 had.

    The property copy only fills keys the survivor lacks, and a survivor
    always has a description, so the duplicate's was dropped. Measured
    before the fix: two Person nodes at cosine 1.0 merged and the survivor
    kept only "DESC-ONE".
    """

    async def test_hard_merge_keeps_both_descriptions(self, ctx):
        """Above hard_threshold there is no LLM, but both descriptions stay."""
        v = _unit([1.0, 0.0, 0.0, 0.0])
        emb = ControlledEmbedder({"Naseem Ali": v, "Naseem  Ali": v})
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="p1",
                    label="Person",
                    properties={"name": "Naseem Ali", "description": "DESC-ONE"},
                ),
                GraphNode(
                    id="p2",
                    label="Person",
                    properties={"name": "Naseem  Ali", "description": "DESC-TWO"},
                ),
            ],
            relationships=[],
        )
        out = await LLMVerifiedResolution(llm=None, embedder=emb, cross_label_merge=False).resolve(
            gd, ctx
        )

        assert len(out.nodes) == 1
        desc = out.nodes[0].properties["description"]
        assert "DESC-ONE" in desc
        assert "DESC-TWO" in desc

    async def test_llm_confirmed_merge_keeps_both_descriptions(self, ctx):
        """The ambiguous zone routes through the LLM and must behave the same."""
        emb = ControlledEmbedder(
            {
                "Ada Lovelace": _unit([1.0, 0.0, 0.0, 0.0]),
                "A. Lovelace": _unit([1.0, 0.55, 0.0, 0.0]),
            }
        )
        llm = PairScriptedLLM({frozenset({"Ada Lovelace", "A. Lovelace"}): True})
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="a1",
                    label="Person",
                    properties={"name": "Ada Lovelace", "description": "DESC-ONE"},
                ),
                GraphNode(
                    id="a2",
                    label="Person",
                    properties={"name": "A. Lovelace", "description": "DESC-TWO"},
                ),
            ],
            relationships=[],
        )
        out = await LLMVerifiedResolution(llm=llm, embedder=emb, cross_label_merge=False).resolve(
            gd, ctx
        )

        assert len(out.nodes) == 1
        desc = out.nodes[0].properties["description"]
        assert "DESC-ONE" in desc
        assert "DESC-TWO" in desc

    async def test_identical_descriptions_are_not_repeated(self, ctx):
        """Two nodes carrying the same text must not yield 'SAME | SAME'."""
        v = _unit([1.0, 0.0, 0.0, 0.0])
        emb = ControlledEmbedder({"Naseem Ali": v, "Naseem  Ali": v})
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="p1",
                    label="Person",
                    properties={"name": "Naseem Ali", "description": "SAME"},
                ),
                GraphNode(
                    id="p2",
                    label="Person",
                    properties={"name": "Naseem  Ali", "description": "SAME"},
                ),
            ],
            relationships=[],
        )
        out = await LLMVerifiedResolution(llm=None, embedder=emb, cross_label_merge=False).resolve(
            gd, ctx
        )

        assert out.nodes[0].properties["description"] == "SAME"


class TestDefaultSoftThresholdBand:
    """P3.25: the 0.65-0.80 band must reach the LLM under the shipped default.

    Measured on the v2 benchmark gold, 31% of real duplicate pairs were
    same-label and scored in 0.55-0.80. Stage 2 floored at 0.80 and the
    cross-label pass never looks at same-label pairs, so they were put to no
    stage at all. The default moved to 0.65 (the knee of the recall/cost
    sweep). These tests fail if it silently moves back up.
    """

    def _pair_at(self, cos: float):
        """Two same-label nodes whose embeddings sit at exactly ``cos``."""
        a = _unit([1.0, 0.0, 0.0, 0.0])
        b = _unit([cos, math.sqrt(1.0 - cos * cos), 0.0, 0.0])
        assert abs(_cosine(a, b) - cos) < 1e-6
        embedder = ControlledEmbedder({"Adolphe Turenne": a, "Turenne": b})
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="n1",
                    label="Person",
                    properties={"name": "Adolphe Turenne", "description": "French officer."},
                ),
                GraphNode(
                    id="n2",
                    label="Person",
                    properties={"name": "Turenne", "description": "The officer, by surname."},
                ),
            ],
            relationships=[],
        )
        return gd, embedder

    async def test_pair_at_070_is_asked_and_merged_by_default(self, ctx):
        """0.70 is inside the old dead band; the default must now ask about it."""
        gd, embedder = self._pair_at(0.70)
        llm = PairScriptedLLM({frozenset({"Adolphe Turenne", "Turenne"}): True})
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder, unified_stage=False)
        result = await resolver.resolve(gd, ctx)
        assert llm.asked == [frozenset({"Adolphe Turenne", "Turenne"})]
        assert len(result.nodes) == 1

    async def test_same_pair_is_invisible_at_the_old_080_floor(self, ctx):
        """The control: at the previous default nobody asks, so it never merges."""
        gd, embedder = self._pair_at(0.70)
        llm = PairScriptedLLM({frozenset({"Adolphe Turenne", "Turenne"}): True})
        resolver = LLMVerifiedResolution(
            llm=llm, embedder=embedder, soft_threshold=0.80, unified_stage=False
        )
        result = await resolver.resolve(gd, ctx)
        assert llm.asked == []
        assert len(result.nodes) == 2

    async def test_below_the_new_floor_still_costs_no_llm_call(self, ctx):
        """0.65 is a floor, not an invitation: 0.50 must still be dropped free."""
        gd, embedder = self._pair_at(0.50)
        llm = PairScriptedLLM({frozenset({"Adolphe Turenne", "Turenne"}): True})
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder, unified_stage=False)
        result = await resolver.resolve(gd, ctx)
        assert llm.asked == []
        assert len(result.nodes) == 2

    async def test_llm_no_in_the_new_band_does_not_merge(self, ctx):
        """Widening the band must not merge on its own -- the LLM still decides.

        The sweep broke 0/20 must-not-merge pairs precisely because a NO is
        still honoured in the newly admitted range.
        """
        gd, embedder = self._pair_at(0.70)
        llm = PairScriptedLLM({frozenset({"Adolphe Turenne", "Turenne"}): False})
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder, unified_stage=False)
        result = await resolver.resolve(gd, ctx)
        assert llm.asked == [frozenset({"Adolphe Turenne", "Turenne"})]
        assert len(result.nodes) == 2


class TestUnifiedStage:
    """The unified stage replaces label bucketing + the cross-label pass.

    Measured on the v2 benchmark gold (239 real duplicate pairs, 39
    must-not-merge, real LLM-written descriptions for all 755 surface forms,
    both shipped caps applied):

        arm                          LLM calls  embeddings  reached  exposed
        label buckets + cross-label      548       1,510     67/239   23/39
        unified @ 0.75                   181         755     77/239    7/39

    Better on every axis at once. These tests pin the four properties that
    result makes load-bearing: one embedding pass over "name: description",
    no label bucketing, and no cross-label merge without an explicit LLM YES.
    See RESULTS.md P3.27.

    The 0.75 row records the default in force when that comparison was run.
    The default is now 0.65, which recovers 29% more true merges at no
    precision cost; see RESULTS.md P3.42 and the ``unified_threshold``
    docstring.
    """

    def _nodes(self):
        return [
            GraphNode(
                id="u1",
                label="Person",
                properties={"name": "Ada Lovelace", "description": "Mathematician."},
            ),
            GraphNode(
                id="u2",
                label="Engineer",
                properties={"name": "A. Lovelace", "description": "Mathematician."},
            ),
        ]

    async def test_default_unified_threshold_is_the_measured_knee(self, ctx):
        """The candidate gate is what caps how much this strategy can merge.

        The LLM only ever sees pairs at or above it, so this default decides
        the ceiling on recall. Measured end to end on 216 trustworthy gold
        pairs, 3 runs each: 0.75 merged 61.7 on average, 0.65 merged 79.7 --
        29% more recall -- with `wrong` pinned at exactly 2 in all six runs.
        0.55 and 0.50 landed inside 0.65's noise band. See RESULTS.md P3.42.
        """
        assert LLMVerifiedResolution(llm=MockLLM([]), embedder=ControlledEmbedder(
            {})).unified_threshold == 0.65

    async def test_embeds_name_and_description_not_the_name_alone(self, ctx):
        """The signal change. Names alone cannot separate real duplicates:
        must-not-merge pairs score HIGHER than true duplicates on names
        (0.687 vs 0.489); only "name: description" has a positive gap."""
        seen: list[str] = []

        class RecordingEmbedder(ControlledEmbedder):
            def embed_query(self, text: str, **kwargs):
                seen.append(text)
                return super().embed_query(text, **kwargs)

        embedder = RecordingEmbedder({})
        resolver = LLMVerifiedResolution(
            llm=MockLLM(responses=["NO\nDifferent."]), embedder=embedder
        )
        await resolver.resolve(GraphData(nodes=self._nodes(), relationships=[]), ctx)
        assert "Ada Lovelace: Mathematician." in seen
        assert "Ada Lovelace" not in seen

    async def test_legacy_path_still_embeds_the_name_alone(self, ctx):
        """The control for the test above: unified_stage=False is unchanged.

        Uses a same-label pair, because the legacy path skips any label bucket
        holding fewer than two nodes -- which is precisely the defect the
        unified stage removes.
        """
        seen: list[str] = []

        class RecordingEmbedder(ControlledEmbedder):
            def embed_query(self, text: str, **kwargs):
                seen.append(text)
                return super().embed_query(text, **kwargs)

        embedder = RecordingEmbedder({})
        nodes = self._nodes()
        nodes[1].label = "Person"
        resolver = LLMVerifiedResolution(
            llm=MockLLM(responses=["NO\nDifferent."]),
            embedder=embedder,
            unified_stage=False,
            cross_label_merge=False,
        )
        await resolver.resolve(GraphData(nodes=nodes, relationships=[]), ctx)
        assert "Ada Lovelace" in seen
        assert "Ada Lovelace: Mathematician." not in seen

    async def test_cross_label_pair_is_reached_without_the_cross_label_pass(self, ctx):
        """The point of the redesign: no label bucket, so one stage sees it."""
        a = _unit([1.0, 0.0, 0.0, 0.0])
        b = _unit([0.95, 0.31, 0.0, 0.0])
        embedder = ControlledEmbedder(
            {"Ada Lovelace: Mathematician.": a, "A. Lovelace: Mathematician.": b}
        )
        llm = PairScriptedLLM({frozenset({"Ada Lovelace", "A. Lovelace"}): True})
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder, cross_label_merge=False)
        result = await resolver.resolve(GraphData(nodes=self._nodes(), relationships=[]), ctx)
        assert llm.asked == [frozenset({"Ada Lovelace", "A. Lovelace"})]
        assert len(result.nodes) == 1

    async def test_identical_vectors_across_labels_never_merge_without_an_llm(self, ctx):
        """The guarantee carried over from the cross-label pass, which asked
        about every candidate it admitted. Cosine alone must not fuse two
        different types however high it scores -- "Paris"/Person and
        "Paris"/Location are not the same entity."""
        vec = _unit([1.0, 0.0, 0.0, 0.0])
        embedder = ControlledEmbedder({"Paris: The capital.": vec, "Paris: The heiress.": vec})
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="c1",
                    label="Location",
                    properties={"name": "Paris", "description": "The capital."},
                ),
                GraphNode(
                    id="c2",
                    label="Person",
                    properties={"name": "Paris", "description": "The heiress."},
                ),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(llm=None, embedder=embedder)
        result = await resolver.resolve(gd, ctx)
        assert len(result.nodes) == 2

    async def test_same_label_hard_merge_shortcut_is_still_free(self, ctx):
        """The control for the test above: the 0.95 shortcut is only withheld
        across labels, not disabled."""
        vec = _unit([1.0, 0.0, 0.0, 0.0])
        embedder = ControlledEmbedder({"Nasa: Agency one.": vec, "N.A.S.A.: Agency two.": vec})
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="s1", label="Org", properties={"name": "Nasa", "description": "Agency one."}
                ),
                GraphNode(
                    id="s2",
                    label="Org",
                    properties={"name": "N.A.S.A.", "description": "Agency two."},
                ),
            ],
            relationships=[],
        )
        llm = PairScriptedLLM({})
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder)
        result = await resolver.resolve(gd, ctx)
        assert llm.asked == [], "same-label pairs above 0.95 must not cost an LLM call"
        assert len(result.nodes) == 1

    async def test_descriptions_are_joined_not_overwritten(self, ctx):
        """P3.21/P3.23: the merge loop only copied keys the survivor lacked, so
        `description` -- always present -- was silently dropped."""
        vec = _unit([1.0, 0.0, 0.0, 0.0])
        embedder = ControlledEmbedder({"Nasa: DESC-ONE": vec, "N.A.S.A.: DESC-TWO": vec})
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="d1", label="Org", properties={"name": "Nasa", "description": "DESC-ONE"}
                ),
                GraphNode(
                    id="d2", label="Org", properties={"name": "N.A.S.A.", "description": "DESC-TWO"}
                ),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(llm=None, embedder=embedder)
        result = await resolver.resolve(gd, ctx)
        assert len(result.nodes) == 1
        desc = result.nodes[0].properties["description"]
        assert "DESC-ONE" in desc and "DESC-TWO" in desc

    async def test_below_the_unified_floor_costs_no_llm_call(self, ctx):
        """0.75 is a floor, not an invitation."""
        a, b = _angle(0.0), _angle(1.0)  # cos(1.0) = 0.540
        assert _cosine(a, b) < 0.75
        embedder = ControlledEmbedder(
            {"Ada Lovelace: Mathematician.": a, "A. Lovelace: Mathematician.": b}
        )
        llm = PairScriptedLLM({frozenset({"Ada Lovelace", "A. Lovelace"}): True})
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder)
        result = await resolver.resolve(GraphData(nodes=self._nodes(), relationships=[]), ctx)
        assert llm.asked == []
        assert len(result.nodes) == 2

    async def test_cross_label_pass_does_not_run_under_the_unified_stage(self, ctx):
        """Running it afterwards would re-ask questions already answered."""
        a = _unit([1.0, 0.0, 0.0, 0.0])
        b = _unit([0.95, 0.31, 0.0, 0.0])
        embedder = ControlledEmbedder(
            {"Ada Lovelace: Mathematician.": a, "A. Lovelace: Mathematician.": b}
        )
        llm = PairScriptedLLM({frozenset({"Ada Lovelace", "A. Lovelace"}): False})
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder, cross_label_merge=True)
        await resolver.resolve(GraphData(nodes=self._nodes(), relationships=[]), ctx)
        assert llm.asked == [frozenset({"Ada Lovelace", "A. Lovelace"})], (
            "the pair must be asked exactly once, not once per stage"
        )
        assert "cross_label_embedding_cache" not in ctx.metadata

    async def test_a_pair_only_descriptions_can_link_is_merged_by_default(self, ctx):
        """The load-bearing default test.

        Two surface forms of one place whose NAMES share nothing. The legacy
        path embeds names only, so it scores them far apart and never asks --
        and because they share a label, the cross-label pass never sees them
        either. This is the structural hole that capped the shipped design at
        28% reach on the v2 gold. The unified stage embeds
        "name: description", scores them together, and asks.
        """
        # cos(0.50) = 0.878: inside the LLM band (>= 0.75, < 0.95) so the pair
        # must be ASKED, not free-merged.
        near_a, near_b = _angle(0.0), _angle(0.50)
        far_a, far_b = _angle(0.0), _angle(1.30)  # cos = 0.267, below any floor
        embedder = ControlledEmbedder(
            {
                "the Ashford light: The lighthouse at Cape Morrow.": near_a,
                "Cape Morrow Lighthouse: The lighthouse at Cape Morrow.": near_b,
                "the Ashford light": far_a,
                "Cape Morrow Lighthouse": far_b,
            }
        )
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="l1",
                    label="Place",
                    properties={
                        "name": "the Ashford light",
                        "description": "The lighthouse at Cape Morrow.",
                    },
                ),
                GraphNode(
                    id="l2",
                    label="Place",
                    properties={
                        "name": "Cape Morrow Lighthouse",
                        "description": "The lighthouse at Cape Morrow.",
                    },
                ),
            ],
            relationships=[],
        )
        llm = PairScriptedLLM({frozenset({"the Ashford light", "Cape Morrow Lighthouse"}): True})
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder)
        result = await resolver.resolve(gd, ctx)
        assert llm.asked == [frozenset({"the Ashford light", "Cape Morrow Lighthouse"})]
        assert len(result.nodes) == 1

    async def test_the_legacy_path_cannot_reach_that_pair(self, ctx):
        """The control: same corpus, label bucketing on, nobody asks."""
        near_a, near_b = _angle(0.0), _angle(0.50)
        far_a, far_b = _angle(0.0), _angle(1.30)
        embedder = ControlledEmbedder(
            {
                "the Ashford light: The lighthouse at Cape Morrow.": near_a,
                "Cape Morrow Lighthouse: The lighthouse at Cape Morrow.": near_b,
                "the Ashford light": far_a,
                "Cape Morrow Lighthouse": far_b,
            }
        )
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="l1",
                    label="Place",
                    properties={
                        "name": "the Ashford light",
                        "description": "The lighthouse at Cape Morrow.",
                    },
                ),
                GraphNode(
                    id="l2",
                    label="Place",
                    properties={
                        "name": "Cape Morrow Lighthouse",
                        "description": "The lighthouse at Cape Morrow.",
                    },
                ),
            ],
            relationships=[],
        )
        llm = PairScriptedLLM({frozenset({"the Ashford light", "Cape Morrow Lighthouse"}): True})
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder, unified_stage=False)
        result = await resolver.resolve(gd, ctx)
        assert llm.asked == []
        assert len(result.nodes) == 2


class TestAbsorbedLabelIsPreserved:
    """A cross-label merge must not destroy the loser's label silently.

    Same defect shape as the description loss fixed in P3.21/P3.23: the merge
    loop copies a duplicate's properties only when the survivor LACKS the key,
    and the survivor always has a label, so the loser's type was dropped with
    no record. Measured before the fix on a Person/Engineer/Author cluster:
    both the legacy and unified paths lost two labels, with nothing
    recoverable from the surviving node.

    ``GraphNode.label`` is a single ``str``, so widening it would ripple
    through storage and query building. Recording it as a property is additive.
    """

    def _cluster(self):
        desc = "Mathematician who wrote the first algorithm."
        vectors = {
            f"Ada Lovelace: {desc}": _angle(0.0),
            f"A. Lovelace: {desc}": _angle(0.5),
            "Ada Lovelace": _angle(0.0),
            "A. Lovelace": _angle(1.3),
        }
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="n1",
                    label="Person",
                    properties={"name": "Ada Lovelace", "description": desc},
                ),
                GraphNode(
                    id="n2",
                    label="Engineer",
                    properties={"name": "A. Lovelace", "description": desc},
                ),
            ],
            relationships=[],
        )
        return gd, ControlledEmbedder(vectors)

    async def test_absorbed_label_is_recorded_on_the_survivor(self, ctx):
        gd, embedder = self._cluster()
        llm = PairScriptedLLM({frozenset({"Ada Lovelace", "A. Lovelace"}): True})
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder)
        result = await resolver.resolve(gd, ctx)
        assert len(result.nodes) == 1
        survivor = result.nodes[0]
        assert survivor.label == "Person"
        assert survivor.properties["merged_labels"] == "Engineer"

    async def test_labels_a_dup_already_absorbed_travel_with_it(self, ctx):
        """Two-hop case: the loser earned ``merged_labels`` in an earlier merge.

        Copying properties only when the survivor lacks the key dropped them
        whenever the survivor had absorbed a label of its own.
        """
        gd, embedder = self._cluster()
        gd.nodes[0].properties["merged_labels"] = "Mathematician"
        gd.nodes[1].properties["merged_labels"] = "Author"
        llm = PairScriptedLLM({frozenset({"Ada Lovelace", "A. Lovelace"}): True})
        result = await LLMVerifiedResolution(llm=llm, embedder=embedder).resolve(gd, ctx)
        assert len(result.nodes) == 1
        kept = set(result.nodes[0].properties["merged_labels"].split(" | "))
        assert kept == {"Mathematician", "Engineer", "Author"}

    async def test_same_label_merge_records_nothing(self, ctx):
        """The control: no noise on the overwhelmingly common case."""
        gd, embedder = self._cluster()
        gd.nodes[1].label = "Person"
        llm = PairScriptedLLM({frozenset({"Ada Lovelace", "A. Lovelace"}): True})
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder)
        result = await resolver.resolve(gd, ctx)
        assert len(result.nodes) == 1
        assert "merged_labels" not in result.nodes[0].properties

    async def test_the_label_reaches_the_database_write(self, ctx):
        """The claim this fix turns on.

        A comment in PASS 2 asserted that recording the label as a property had
        been tried and rejected because graph_store persists only
        name/type/description/source_chunk_ids. That is false of the current
        write path: ``_clean_properties`` whitelists no keys and
        ``upsert_nodes`` issues ``SET n += item.properties``. This drives the
        real GraphStore with a recording connection and asserts the property is
        in the parameters actually sent to the database.
        """
        from unittest.mock import AsyncMock

        from graphrag_sdk.storage.graph_store import GraphStore

        gd, embedder = self._cluster()
        llm = PairScriptedLLM({frozenset({"Ada Lovelace", "A. Lovelace"}): True})
        result = await LLMVerifiedResolution(llm=llm, embedder=embedder).resolve(gd, ctx)

        conn = AsyncMock()
        conn.query = AsyncMock(return_value=None)
        await GraphStore(conn).upsert_nodes(result.nodes)

        # upsert_nodes now issues a second query to promote merged labels, so
        # the property write is located explicitly rather than by position.
        upserts = [
            c[0][1] for c in conn.query.call_args_list if "MERGE" in c[0][0]
        ]
        assert upserts, "no upsert query was issued"
        params = upserts[0]
        props = params["batch"][0]["properties"] if "batch" in params else params["properties"]
        assert props.get("merged_labels") == "Engineer"

    async def test_legacy_cross_label_pass_preserves_it_too(self, ctx):
        """PASS 2 has its own merge loop and needed the same fix."""
        gd, embedder = self._cluster()
        llm = PairScriptedLLM({frozenset({"Ada Lovelace", "A. Lovelace"}): True})
        resolver = LLMVerifiedResolution(llm=llm, embedder=embedder, unified_stage=False)
        result = await resolver.resolve(gd, ctx)
        assert len(result.nodes) == 1
        assert result.nodes[0].properties["merged_labels"] == "Engineer"


class TestDefaultBatchSizeIsBounded:
    """The default budget must keep a call small enough to stay reliable.

    Measured on 174 real pairs with real descriptions: at the previous budget
    of 20000 a single call carried 155 pairs and the outcome was bimodal --
    25, 26, 27, 28 correct merges on four trials and 51, 53, 54, 58 on four
    others, with identical input. The reply was always complete and correctly
    numbered, so nothing detects the bad half. At 4000 (<= 29 pairs per call)
    four trials gave 51, 51, 52, 54. These tests stop the budget drifting back
    up without someone re-running that measurement.
    """

    @staticmethod
    def _realistic_block(i: int) -> str:
        # ~126 tokens, the measured average for a real name+description pair.
        return (
            f"Name: Entity Number {i}\n"
            f"Description: {'a moderately long factual sentence about it ' * 12}\n"
            f"Name: Entity Number {i} Variant\n"
            f"Description: {'a moderately long factual sentence about it ' * 12}\n"
            f"Similarity: 0.83\n"
        )

    def test_default_budget_keeps_calls_under_forty_pairs(self):
        blocks = [self._realistic_block(i) for i in range(400)]
        windows = LLMVerifiedResolution()._pack(blocks)
        largest = max(stop - start for start, stop in windows)
        assert largest <= 40, (
            f"{largest} pairs in one call; measured unreliable above ~60, "
            "bimodal at 155"
        )

    def test_the_old_budget_would_have_failed_this(self):
        """Control: the guard is not vacuous -- 20000 really does exceed it.

        The pair-count cap added later would mask this on its own, so it is
        lifted here to isolate the token budget's effect.
        """
        blocks = [self._realistic_block(i) for i in range(400)]
        windows = LLMVerifiedResolution(
            verification_token_budget=20000, verification_max_pairs_per_call=10**9
        )._pack(blocks)
        assert max(stop - start for start, stop in windows) > 40

    def test_every_pair_is_still_asked_about(self):
        """Shrinking the budget must not drop pairs."""
        blocks = [self._realistic_block(i) for i in range(400)]
        windows = LLMVerifiedResolution()._pack(blocks)
        covered = sum(stop - start for start, stop in windows)
        assert covered == 400
        assert windows[0][0] == 0 and windows[-1][1] == 400


class TestPairCountCapIndependentOfTokens:
    """The pair count must be bounded even when the blocks are tiny.

    The bimodal measurement in ``TestDefaultBatchSizeIsBounded`` rode on
    ~200-token name+description blocks, where a 4000-token budget yields ~17
    pairs. A corpus with no descriptions produces ~30-token blocks, and the
    same budget then packs **119 pairs into one call** -- back inside the range
    that went bimodal. What degrades is the number of questions in one reply,
    not the tokens, so the count needs its own cap.
    """

    SHORT = "Name: IBM\nName: I.B.M.\nSimilarity: 0.91\n"

    def test_tiny_blocks_are_still_capped(self):
        windows = LLMVerifiedResolution()._pack([self.SHORT] * 4000)
        largest = max(stop - start for start, stop in windows)
        assert largest <= 30, f"{largest} tiny pairs packed into one call"

    def test_without_the_count_cap_tiny_blocks_overflow(self):
        """Control: the token budget alone genuinely fails to bound this."""
        resolver = LLMVerifiedResolution(verification_max_pairs_per_call=10**9)
        windows = resolver._pack([self.SHORT] * 4000)
        assert max(stop - start for start, stop in windows) > 100

    def test_the_tighter_of_the_two_limits_wins(self):
        """Long blocks stay token-bound, below the count cap."""
        long_block = (
            "Name: Ada Lovelace\nDescription: " + ("a long factual sentence " * 30)
            + "\nName: A. Lovelace\nDescription: " + ("a long factual sentence " * 30)
            + "\nSimilarity: 0.83\n"
        )
        windows = LLMVerifiedResolution()._pack([long_block] * 500)
        largest = max(stop - start for start, stop in windows)
        assert largest < 30, "token budget should bind first for large blocks"

    def test_no_pairs_are_dropped_by_the_cap(self):
        windows = LLMVerifiedResolution()._pack([self.SHORT] * 4000)
        assert sum(stop - start for start, stop in windows) == 4000
        assert windows[0][0] == 0 and windows[-1][1] == 4000
        for (_, end), (nxt, _) in zip(windows, windows[1:]):
            assert end == nxt

    def test_a_cap_of_one_is_effectively_per_pair(self):
        resolver = LLMVerifiedResolution(verification_max_pairs_per_call=1)
        windows = resolver._pack([self.SHORT] * 12)
        assert len(windows) == 12


class TestAmbiguousClusteringDistances:
    """A cross-label pair scoring at or above 1.0 must not crash the stage.

    hnswlib returns float32 inner-product distances. For two identical unit
    vectors that distance can come back very slightly negative, so
    ``1 - distance`` exceeds 1.0 and a raw ``1 - similarity`` is negative.
    ``scipy.cluster.hierarchy.fcluster`` rejects a linkage matrix containing
    negative distances with a ``ValueError``, aborting the whole resolution
    stage — observed on the 755-node benchmark corpus.

    It is only reachable when a near-perfect pair lands in the ambiguous zone,
    which the cross-label guard makes possible: identical names under different
    labels are withheld from the hard-merge shortcut and sent to the verifier.
    """

    def test_a_similarity_above_one_stays_a_valid_distance(self):
        """The observed failing value: 1 - (-1.19e-07) from float32 round-off."""
        assert _pair_distance(1.0000001192092896) == 0.0

    def test_ordinary_similarities_are_untouched(self):
        for sim in (0.0, 0.25, 0.5, 0.65, 0.95, 1.0):
            assert _pair_distance(sim) == pytest.approx(1.0 - sim)

    def test_a_negative_similarity_stays_within_range(self):
        """Opposed vectors are legal in inner-product space."""
        assert _pair_distance(-0.4) == 1.0

    def test_scipy_accepts_a_matrix_built_from_it(self):
        """The end the clamp exists for: scipy validates the linkage input."""
        import numpy as np
        import scipy.cluster.hierarchy as sch
        import scipy.spatial.distance as ssd

        sims = [(0, 1, 1.0000001192092896), (0, 2, 0.97), (1, 2, 0.96)]
        dist = np.ones((3, 3), dtype=np.float32)
        np.fill_diagonal(dist, 0.0)
        for i, j, s in sims:
            dist[i, j] = dist[j, i] = _pair_distance(s)

        assert dist.min() >= 0.0
        linkage = sch.linkage(ssd.squareform(dist), method="average")
        assert len(sch.fcluster(linkage, t=0.05, criterion="distance")) == 3

    async def test_identical_cross_label_vectors_still_need_the_verifier(self, ctx):
        """The behaviour around the clamp is unchanged: a rejected pair stays
        split, and the pair does reach the LLM rather than merging on cosine."""
        vec = _unit([0.505994, 0.40823, 0.039448, 0.75879])
        embedder = ControlledEmbedder(
            {"Paris: The capital.": vec, "Paris: The heiress.": vec}
        )
        gd = GraphData(
            nodes=[
                GraphNode(
                    id="c1",
                    label="Location",
                    properties={"name": "Paris", "description": "The capital."},
                ),
                GraphNode(
                    id="c2",
                    label="Person",
                    properties={"name": "Paris", "description": "The heiress."},
                ),
            ],
            relationships=[],
        )
        resolver = LLMVerifiedResolution(llm=MockLLM(responses=["1. NO"]), embedder=embedder)

        result = await resolver.resolve(gd, ctx)

        assert len(result.nodes) == 2, "a rejected pair must stay separate"


class TestMaxLlmPairsTruncation:
    """``max_llm_pairs`` discards unverified candidates, so it must say so.

    The cap sorts boundary pairs by descending similarity and keeps the first
    ``max_llm_pairs``. Everything below the cut is dropped without ever being
    verified, so a duplicate there survives into the graph. That is a recall
    ceiling, and it used to be invisible: the progress line reported the
    pre-cap count as the number going to the LLM, so a run that verified 500 of
    15,877 candidates logged "15877 boundary pairs -> LLM".

    It matters at scale rather than on small graphs: candidate count grows as
    roughly n^2, so the cap is inert at a few hundred nodes and binding at a few
    thousand (RESULTS.md P3.52).
    """

    @staticmethod
    def _spread_nodes(count: int) -> list[GraphNode]:
        """Nodes whose pairwise cosines all land inside the ambiguous zone."""
        return [
            GraphNode(
                id=f"n{i}",
                label="Person",
                properties={"name": f"Person {i}", "description": f"Bio {i}."},
            )
            for i in range(count)
        ]

    @staticmethod
    def _spread_embedder(nodes: list[GraphNode], sim: float = 0.80) -> ControlledEmbedder:
        """Put every pair at the same cosine, inside the ambiguous band.

        A 2-D arc cannot do this: to keep adjacent nodes below ``hard_threshold``
        the steps must be wide, which pushes the outermost pair below
        ``unified_threshold``. Giving each node its own axis on top of a shared
        component makes all pairwise cosines exactly equal instead —
        ``v_i = sqrt(s) * e_0 + sqrt(1 - s) * e_(i+1)`` gives ``v_i . v_j = s``
        for every ``i != j``. At 0.80 every pair is above the 0.65 floor and
        below the 0.95 shortcut, so all of them are boundary pairs and the cap
        is the only thing deciding which get verified.
        """
        dim = len(nodes) + 1
        shared, own = math.sqrt(sim), math.sqrt(1.0 - sim)
        vectors = {}
        for i, n in enumerate(nodes):
            v = [0.0] * dim
            v[0] = shared
            v[i + 1] = own
            vectors[f"{n.properties['name']}: {n.properties['description']}"] = v
        return ControlledEmbedder(vectors, default_dim=dim)

    async def test_the_cap_truncates_and_warns(self, ctx, caplog):
        nodes = self._spread_nodes(8)
        resolver = LLMVerifiedResolution(
            llm=MockLLM(responses=["1. NO\n2. NO\n3. NO\n4. NO\n5. NO"]),
            embedder=self._spread_embedder(nodes),
            max_llm_pairs=3,
        )

        with caplog.at_level(logging.WARNING):
            await resolver.resolve(GraphData(nodes=nodes, relationships=[]), ctx)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("max_llm_pairs=3" in r.message for r in warnings), (
            "truncating verification must warn, or recall silently ceilings"
        )

    async def test_the_warning_reports_how_many_went_unverified(self, ctx, caplog):
        nodes = self._spread_nodes(8)
        resolver = LLMVerifiedResolution(
            llm=MockLLM(responses=["1. NO\n2. NO\n3. NO\n4. NO\n5. NO"]),
            embedder=self._spread_embedder(nodes),
            max_llm_pairs=3,
        )

        with caplog.at_level(logging.WARNING):
            await resolver.resolve(GraphData(nodes=nodes, relationships=[]), ctx)

        text = " ".join(r.message for r in caplog.records if r.levelno >= logging.WARNING)
        assert "left unverified" in text
        # 8 nodes on one arc => 28 candidate pairs, 3 verified, 25 dropped.
        assert "3 highest-similarity" in text
        assert "25 pair(s)" in text

    async def test_an_uncapped_run_is_silent(self, ctx, caplog):
        """The control: no warning when nothing is dropped, so the signal keeps
        meaning something."""
        nodes = self._spread_nodes(8)
        resolver = LLMVerifiedResolution(
            llm=MockLLM(responses=["1. NO\n2. NO\n3. NO\n4. NO\n5. NO"]),
            embedder=self._spread_embedder(nodes),
            max_llm_pairs=500,
        )

        with caplog.at_level(logging.WARNING):
            await resolver.resolve(GraphData(nodes=nodes, relationships=[]), ctx)

        assert not [
            r for r in caplog.records if "max_llm_pairs" in r.message
        ], "an inert cap must not warn"

    async def test_the_progress_line_counts_pairs_actually_sent(self, ctx, caplog):
        """The original defect: the pre-cap count was reported as the number
        reaching the LLM."""
        nodes = self._spread_nodes(8)
        resolver = LLMVerifiedResolution(
            llm=MockLLM(responses=["1. NO\n2. NO\n3. NO\n4. NO\n5. NO"]),
            embedder=self._spread_embedder(nodes),
            max_llm_pairs=3,
        )

        with caplog.at_level(logging.INFO):
            await resolver.resolve(GraphData(nodes=nodes, relationships=[]), ctx)

        progress = [r.message for r in caplog.records if "boundary pairs" in r.message]
        assert progress, "the progress line should still be emitted"
        assert "3 boundary pairs → LLM" in progress[0]
        assert "28 boundary pairs" not in progress[0]


class TestUnansweredPairsAreRetried:
    """A short batched reply must not silently lose merges.

    The model is asked N numbered questions in one call and sometimes answers
    fewer. Treating "no verdict" as "not a duplicate" drops real merges with
    only a log line to show for it, so every unanswered pair is re-asked one
    pair per call, where a short reply is not possible.
    """

    def _resolver(self, responses: list[str]) -> LLMVerifiedResolution:
        return LLMVerifiedResolution(llm=MockLLM(responses))

    @pytest.mark.asyncio
    async def test_missing_verdicts_are_re_asked_individually(self):
        res = self._resolver(["1. a vs b | YES\n3. e vs f | NO", "YES"])
        out = await res._verify_batched(["b1", "b2", "b3", "b4"])

        assert out == {0: True, 1: True, 2: False, 3: True}
        # one batched call, then one call for each of the two unanswered pairs
        assert res.llm._call_index == 3

    @pytest.mark.asyncio
    async def test_complete_reply_costs_no_extra_calls(self):
        res = self._resolver(["1. a | YES\n2. b | NO"])
        out = await res._verify_batched(["b1", "b2"])

        assert out == {0: True, 1: False}
        assert res.llm._call_index == 1

    @pytest.mark.asyncio
    async def test_retry_answer_can_be_no(self):
        """The retry must be able to reject, not just rescue."""
        res = self._resolver(["1. a | YES", "NO"])
        out = await res._verify_batched(["b1", "b2"])

        assert out == {0: True, 1: False}

    @pytest.mark.asyncio
    async def test_pair_still_unanswered_after_retry_stays_unmerged(self):
        """A pair with no verdict is absent, never a silent False."""
        res = self._resolver(["1. a | YES", "I cannot help with that."])
        out = await res._verify_batched(["b1", "b2"])

        assert out == {0: True}
