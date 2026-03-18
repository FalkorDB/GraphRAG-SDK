# GraphRAG SDK 2.0 — Reranking: Reciprocal Rank Fusion
#
# Novelty 4: Multi-signal RRF combining cosine similarity, entity coverage,
# and PPR structural score.  Inspired by HybridRAG (k=60 constant).

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    ``Score(item) = sum(1 / (k + rank_i))`` for each list in which the
    item appears.  Uses only rank positions, not raw scores — making it
    robust to incomparable score scales across retrieval methods.

    Args:
        ranked_lists: List of ranked lists, each containing item IDs
                      ordered from most relevant (rank 1) to least.
        k: Smoothing constant (default 60, per HybridRAG/original RRF paper).

    Returns:
        List of ``(item_id, rrf_score)`` sorted descending by score.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list, 1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
