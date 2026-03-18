"""Tests for retrieval/reranking_strategies/rrf.py — Reciprocal Rank Fusion."""

from __future__ import annotations

from graphrag_sdk.retrieval.reranking_strategies.rrf import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_single_list(self):
        """Single ranked list should produce RRF scores."""
        results = reciprocal_rank_fusion([["A", "B", "C"]])
        ids = [item_id for item_id, _ in results]
        assert ids == ["A", "B", "C"]

    def test_two_lists_same_order(self):
        """Two lists with same order should reinforce ranking."""
        results = reciprocal_rank_fusion(
            [
                ["A", "B", "C"],
                ["A", "B", "C"],
            ]
        )
        ids = [item_id for item_id, _ in results]
        assert ids == ["A", "B", "C"]

    def test_two_lists_different_order(self):
        """RRF should combine signals from different orderings."""
        results = reciprocal_rank_fusion(
            [
                ["A", "B", "C"],
                ["C", "B", "A"],
            ]
        )
        ids = [item_id for item_id, _ in results]
        # All three have nearly identical scores due to symmetry
        # (1/61+1/63 ≈ 1/62+1/62 ≈ 1/63+1/61)
        assert set(ids) == {"A", "B", "C"}

    def test_consensus_item_wins(self):
        """Item ranked consistently high across all lists should win."""
        results = reciprocal_rank_fusion(
            [
                ["A", "B", "C", "D"],
                ["A", "C", "B", "D"],
                ["A", "B", "D", "C"],
            ]
        )
        ids = [item_id for item_id, _ in results]
        # A is rank 1 in all three lists → clear winner
        assert ids[0] == "A"

    def test_empty_lists(self):
        """Empty input should return empty output."""
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []

    def test_k_parameter(self):
        """Different k values should change relative scores."""
        results_k60 = reciprocal_rank_fusion([["A", "B"]], k=60)
        results_k1 = reciprocal_rank_fusion([["A", "B"]], k=1)
        # With k=1, rank difference matters more
        _, score_a_k1 = results_k1[0]
        _, score_b_k1 = results_k1[1]
        _, score_a_k60 = results_k60[0]
        _, score_b_k60 = results_k60[1]
        # Ratio of scores should be more extreme with smaller k
        ratio_k1 = score_a_k1 / score_b_k1
        ratio_k60 = score_a_k60 / score_b_k60
        assert ratio_k1 > ratio_k60

    def test_three_signals(self):
        """Three-signal fusion should work correctly."""
        results = reciprocal_rank_fusion(
            [
                ["A", "B", "C", "D"],  # cosine
                ["B", "C", "A", "D"],  # coverage
                ["C", "A", "B", "D"],  # PPR
            ]
        )
        ids = [item_id for item_id, _ in results]
        # D is last in all three → should be last
        assert ids[-1] == "D"

    def test_unique_items_across_lists(self):
        """Items appearing in only one list should still get scores."""
        results = reciprocal_rank_fusion(
            [
                ["A", "B"],
                ["C", "D"],
            ]
        )
        ids = [item_id for item_id, _ in results]
        assert len(ids) == 4
        # Items at rank 1 in their respective lists should tie
        assert set(ids[:2]) == {"A", "C"}

    def test_score_formula(self):
        """Verify RRF score formula: 1/(k + rank)."""
        results = reciprocal_rank_fusion([["A", "B"]], k=60)
        score_a = results[0][1]
        score_b = results[1][1]
        assert abs(score_a - 1.0 / 61) < 1e-10
        assert abs(score_b - 1.0 / 62) < 1e-10
