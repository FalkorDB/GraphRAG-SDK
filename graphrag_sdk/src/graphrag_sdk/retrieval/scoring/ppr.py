# GraphRAG SDK 2.0 — Scoring: Client-Side Personalized PageRank
#
# Novelty 2: Bounded Subgraph PPR — database-extracted BFS subgraph +
# pure Python/numpy power iteration.  Zero external dependencies beyond numpy.
#
# Inspired by HippoRAG (damping=0.5) and PropRAG (subgraph PPR > full-graph PPR).
#
# QC-PPR extension: when edge_weights are provided, transition probabilities
# are proportional to semantic similarity (query × edge) instead of uniform
# 1/degree.  This makes the random walker follow query-relevant edges.

from __future__ import annotations

import numpy as np


def personalized_pagerank(
    adj: dict[str, list[str]],
    seed_weights: dict[str, float],
    *,
    damping: float = 0.5,
    max_iterations: int = 30,
    tolerance: float = 1e-6,
    edge_weights: dict[tuple[str, str], float] | None = None,
) -> dict[str, float]:
    """Compute Personalized PageRank via power iteration.

    Pure Python/numpy implementation — no igraph, no networkx.
    Completes in <10ms for typical subgraphs (200-500 nodes).

    Args:
        adj: Adjacency list ``{node_id: [neighbor_ids, ...]}``.
             Graph is treated as directed; callers should include both
             directions for undirected behavior.
        seed_weights: ``{entity_id: weight}`` for the personalization vector.
             Nodes not in this dict receive zero teleportation probability.
        damping: Damping factor.  0.5 = exploratory (HippoRAG default),
                 0.85 = standard web PageRank.  Lower values stay closer
                 to seed nodes; higher values explore further.
        max_iterations: Maximum power iteration steps.
        tolerance: L1 convergence threshold.
        edge_weights: Optional ``{(src, tgt): weight}`` for QC-PPR.
                 When provided, transition probabilities are proportional
                 to these weights (softmax per source node) instead of
                 uniform 1/degree.

    Returns:
        ``{node_id: ppr_score}`` for every node in the subgraph,
        normalized so scores sum to 1.
    """
    nodes = sorted(adj.keys())
    n = len(nodes)
    if n == 0:
        return {}

    idx = {node: i for i, node in enumerate(nodes)}

    # ── Personalization vector ──
    p = np.zeros(n, dtype=np.float64)
    for node, w in seed_weights.items():
        if node in idx:
            p[idx[node]] = w
    p_sum = p.sum()
    if p_sum > 0:
        p /= p_sum
    else:
        p[:] = 1.0 / n

    # ── Column-stochastic transition matrix ──
    # M[j, i] = transition_prob(i→j)
    M = np.zeros((n, n), dtype=np.float64)
    dangling = np.zeros(n, dtype=np.float64)

    for src, neighbors in adj.items():
        i = idx[src]
        valid = [nb for nb in neighbors if nb in idx]
        d = len(valid)
        if d == 0:
            dangling[i] = 1.0  # dangling node
            continue

        if edge_weights is not None:
            # QC-PPR: softmax of edge weights as transition probabilities
            raw = np.array(
                [edge_weights.get((src, nb), edge_weights.get((nb, src), 0.0)) for nb in valid],
                dtype=np.float64,
            )
            # Shift for numerical stability, then softmax
            raw -= raw.max()
            exp_w = np.exp(raw)
            total = exp_w.sum()
            if total > 0:
                probs = exp_w / total
            else:
                probs = np.full(d, 1.0 / d)
            for nb, prob in zip(valid, probs):
                M[idx[nb], i] += prob
        else:
            # Standard uniform: 1/degree
            weight = 1.0 / d
            for nb in valid:
                M[idx[nb], i] += weight

    # ── Power iteration ──
    # r(t+1) = damping * M @ r(t) + (1 - damping) * p + dangling_redistribution
    r = p.copy()
    for _ in range(max_iterations):
        r_new = damping * (M @ r) + (1 - damping) * p
        # Dangling nodes redistribute their mass via personalization
        dangling_mass = damping * np.dot(dangling, r)
        r_new += dangling_mass * p

        if np.abs(r_new - r).sum() < tolerance:
            r = r_new
            break
        r = r_new

    return {nodes[i]: float(r[i]) for i in range(n)}
