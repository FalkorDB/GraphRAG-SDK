"""Tests for retrieval/scoring/ppr.py — Personalized PageRank."""

from __future__ import annotations

from graphrag_sdk.retrieval.scoring.ppr import personalized_pagerank


class TestPersonalizedPageRank:
    def test_empty_graph(self):
        """Empty adjacency list should return empty scores."""
        assert personalized_pagerank({}, {}) == {}

    def test_single_node(self):
        """Single dangling node should get all probability mass."""
        scores = personalized_pagerank({"A": []}, {"A": 1.0})
        assert len(scores) == 1
        assert abs(scores["A"] - 1.0) < 1e-6

    def test_two_connected_nodes(self):
        """Seed node should have higher score than neighbor."""
        adj = {"A": ["B"], "B": ["A"]}
        scores = personalized_pagerank(adj, {"A": 1.0}, damping=0.5)
        assert scores["A"] > scores["B"]

    def test_seed_node_highest(self):
        """Seed node should always have highest PPR score."""
        adj = {
            "seed": ["A", "B"],
            "A": ["seed", "C"],
            "B": ["seed", "C"],
            "C": ["A", "B"],
        }
        scores = personalized_pagerank(adj, {"seed": 1.0}, damping=0.5)
        assert scores["seed"] == max(scores.values())

    def test_multiple_seeds(self):
        """Both seed nodes should have high scores."""
        adj = {
            "S1": ["A", "B"],
            "S2": ["B", "C"],
            "A": ["S1"],
            "B": ["S1", "S2"],
            "C": ["S2"],
        }
        scores = personalized_pagerank(adj, {"S1": 1.0, "S2": 1.0}, damping=0.5)
        # Both seeds should be in top 2
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_2 = {ranked[0][0], ranked[1][0]}
        assert "S1" in top_2
        assert "S2" in top_2

    def test_scores_sum_to_one(self):
        """PPR scores should sum to approximately 1."""
        adj = {
            "A": ["B", "C"],
            "B": ["A", "D"],
            "C": ["A"],
            "D": ["B"],
        }
        scores = personalized_pagerank(adj, {"A": 1.0})
        assert abs(sum(scores.values()) - 1.0) < 1e-4

    def test_damping_effect(self):
        """Higher damping should spread more mass to non-seed nodes."""
        adj = {
            "seed": ["A", "B"],
            "A": ["seed", "B"],
            "B": ["seed", "A"],
        }
        low_damping = personalized_pagerank(adj, {"seed": 1.0}, damping=0.3)
        high_damping = personalized_pagerank(adj, {"seed": 1.0}, damping=0.8)
        # Higher damping → seed gets less (more exploration)
        assert low_damping["seed"] > high_damping["seed"]

    def test_convergence(self):
        """Should converge within max_iterations for small graphs."""
        adj = {str(i): [str(j) for j in range(10) if j != i] for i in range(10)}
        scores = personalized_pagerank(adj, {"0": 1.0}, max_iterations=100)
        assert len(scores) == 10
        assert abs(sum(scores.values()) - 1.0) < 1e-4

    def test_disconnected_nodes(self):
        """Disconnected nodes should get score from personalization only."""
        adj = {"A": ["B"], "B": ["A"], "C": []}
        scores = personalized_pagerank(adj, {"A": 1.0}, damping=0.5)
        # C is disconnected and not a seed → gets only teleportation mass
        assert scores["C"] < scores["A"]

    def test_star_topology(self):
        """Hub node connected to many leaves should get high score."""
        leaves = [f"L{i}" for i in range(5)]
        adj = {"hub": leaves}
        for leaf in leaves:
            adj[leaf] = ["hub"]
        scores = personalized_pagerank(adj, {"L0": 1.0}, damping=0.5)
        # Hub should be 2nd highest (after seed L0)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        assert ranked[0][0] == "L0"
        assert ranked[1][0] == "hub"

    def test_seed_not_in_graph(self):
        """Seed nodes not in graph should be ignored gracefully."""
        adj = {"A": ["B"], "B": ["A"]}
        # Seed "X" is not in the graph
        scores = personalized_pagerank(adj, {"X": 1.0})
        # Should fallback to uniform personalization
        assert len(scores) == 2
        assert abs(sum(scores.values()) - 1.0) < 1e-4


class TestQCPPR:
    """Tests for Query-Conditioned PPR (edge_weights parameter)."""

    def test_edge_weights_favor_high_weight_neighbor(self):
        """QC-PPR should route more mass along high-weight edges."""
        # seed → A (high weight), seed → B (low weight)
        adj = {"seed": ["A", "B"], "A": ["seed"], "B": ["seed"]}
        weights = {
            ("seed", "A"): 1.0,
            ("seed", "B"): -1.0,
            ("A", "seed"): 1.0,
            ("B", "seed"): -1.0,
        }
        scores = personalized_pagerank(
            adj, {"seed": 1.0}, damping=0.5, edge_weights=weights
        )
        assert scores["A"] > scores["B"]

    def test_edge_weights_none_equals_uniform(self):
        """edge_weights=None should produce same results as no weights."""
        adj = {"A": ["B", "C"], "B": ["A"], "C": ["A"]}
        seeds = {"A": 1.0}
        scores_none = personalized_pagerank(adj, seeds, edge_weights=None)
        scores_default = personalized_pagerank(adj, seeds)
        for node in adj:
            assert abs(scores_none[node] - scores_default[node]) < 1e-10

    def test_edge_weights_sum_to_one(self):
        """PPR with edge weights should still sum to 1."""
        adj = {
            "S": ["A", "B", "C"],
            "A": ["S", "B"],
            "B": ["S", "A", "C"],
            "C": ["S", "B"],
        }
        weights = {
            ("S", "A"): 0.9, ("S", "B"): 0.1, ("S", "C"): 0.5,
            ("A", "S"): 0.3, ("A", "B"): 0.8,
            ("B", "S"): 0.2, ("B", "A"): 0.7, ("B", "C"): 0.4,
            ("C", "S"): 0.6, ("C", "B"): 0.3,
        }
        scores = personalized_pagerank(
            adj, {"S": 1.0}, damping=0.5, edge_weights=weights
        )
        assert abs(sum(scores.values()) - 1.0) < 1e-4

    def test_uniform_weights_match_no_weights(self):
        """Equal edge weights should produce same result as uniform."""
        adj = {"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}
        # All edges get same weight → softmax gives uniform
        weights = {(s, t): 0.5 for s in adj for t in adj[s]}
        scores_weighted = personalized_pagerank(
            adj, {"A": 1.0}, damping=0.5, edge_weights=weights
        )
        scores_uniform = personalized_pagerank(
            adj, {"A": 1.0}, damping=0.5
        )
        for node in adj:
            assert abs(scores_weighted[node] - scores_uniform[node]) < 1e-6
