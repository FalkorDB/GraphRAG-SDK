# GraphRAG SDK — Retrieval: Dynamic Graph Walk (Phase 3.3)
# PageRank-weighted traversal with beam search, bidirectional search,
# and path scoring. The core algorithms are decoupled from FalkorDB:
# they operate on an injected async ``neighbor_fn`` and an optional
# node-weight map, so they can be unit-tested in memory and reused by
# the agentic retriever (Phase 3.1) as a "traverse" action.

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    RawSearchResult,
    RetrieverResult,
    RetrieverResultItem,
    ScoredPath,
)
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy

logger = logging.getLogger(__name__)

# An async function: node_id -> list of (neighbor_id, edge_weight, edge_label).
NeighborFn = Callable[[str], Awaitable[list[tuple[str, float, str]]]]


def score_path(
    nodes: list[str],
    weights: dict[str, float] | None = None,
    edge_weights: list[float] | None = None,
    *,
    length_penalty: float = 0.1,
) -> float:
    """Score a path by node importance and edge strength, minus a length penalty.

    The score rewards paths that pass through important nodes (PageRank
    weights) over strong edges, while penalising longer paths so short,
    high-signal explanations rank first.
    """
    if not nodes:
        return 0.0
    weights = weights or {}
    node_score = sum(weights.get(n, 0.0) for n in nodes) / len(nodes)
    edge_score = (sum(edge_weights) / len(edge_weights)) if edge_weights else 0.0
    hops = max(0, len(nodes) - 1)
    return node_score + edge_score - length_penalty * hops


class DynamicGraphWalk:
    """Dynamic, weight-aware graph traversal.

    Args:
        neighbor_fn: Async callable returning weighted neighbors of a node.
        node_weights: Optional PageRank-style importance per node id.
        beam_width: Max number of partial paths kept per expansion round.
        max_depth: Max traversal depth (hops).
        length_penalty: Per-hop penalty applied during path scoring.
    """

    def __init__(
        self,
        neighbor_fn: NeighborFn,
        *,
        node_weights: dict[str, float] | None = None,
        beam_width: int = 5,
        max_depth: int = 4,
        length_penalty: float = 0.1,
    ) -> None:
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self._neighbor_fn = neighbor_fn
        self._weights = node_weights or {}
        self._beam_width = beam_width
        self._max_depth = max_depth
        self._length_penalty = length_penalty

    async def beam_search(
        self,
        start: str,
        *,
        goal: str | None = None,
        ctx: Context | None = None,
    ) -> list[ScoredPath]:
        """Beam-search outward from ``start``, keeping the top-k paths each round.

        If ``goal`` is given, paths reaching it are returned as soon as
        found; otherwise the highest-scoring frontier paths are returned.
        """
        # Each beam entry: (path_nodes, path_edge_labels, path_edge_weights).
        beam: list[tuple[list[str], list[str], list[float]]] = [([start], [], [])]
        completed: list[ScoredPath] = []
        visited: set[str] = {start}

        for depth in range(self._max_depth):
            if ctx is not None:
                ctx.ensure_budget(f"graph walk depth {depth}")
            candidates: list[tuple[list[str], list[str], list[float]]] = []
            for nodes, edges, e_weights in beam:
                tail = nodes[-1]
                if goal is not None and tail == goal:
                    completed.append(
                        ScoredPath(
                            nodes=list(nodes),
                            edges=list(edges),
                            score=score_path(
                                nodes,
                                self._weights,
                                e_weights,
                                length_penalty=self._length_penalty,
                            ),
                        )
                    )
                    continue
                for nbr, weight, label in await self._neighbor_fn(tail):
                    if nbr in nodes:
                        continue  # no cycles within a single path
                    candidates.append((nodes + [nbr], edges + [label], e_weights + [weight]))
                    visited.add(nbr)

            if not candidates:
                break

            # Keep the top-k candidates by current path score.
            candidates.sort(
                key=lambda c: score_path(
                    c[0], self._weights, c[2], length_penalty=self._length_penalty
                ),
                reverse=True,
            )
            beam = candidates[: self._beam_width]

            if goal is not None and any(c[0][-1] == goal for c in beam):
                # Promote goal-reaching paths immediately.
                for nodes, edges, e_weights in beam:
                    if nodes[-1] == goal:
                        completed.append(
                            ScoredPath(
                                nodes=list(nodes),
                                edges=list(edges),
                                score=score_path(
                                    nodes,
                                    self._weights,
                                    e_weights,
                                    length_penalty=self._length_penalty,
                                ),
                            )
                        )
                if completed:
                    break

        results = completed or [
            ScoredPath(
                nodes=nodes,
                edges=edges,
                score=score_path(
                    nodes, self._weights, e_weights, length_penalty=self._length_penalty
                ),
            )
            for nodes, edges, e_weights in beam
        ]
        results.sort(key=lambda p: p.score, reverse=True)
        return results[: self._beam_width]

    async def bidirectional_search(
        self,
        start: str,
        goal: str,
        *,
        ctx: Context | None = None,
    ) -> ScoredPath | None:
        """Search from both ``start`` and ``goal``, meeting in the middle.

        Returns the connecting path (oriented start→goal) or ``None`` if
        the two frontiers never meet within ``max_depth`` hops on each side.
        """
        if start == goal:
            return ScoredPath(nodes=[start], edges=[], score=score_path([start], self._weights))

        # parent maps: node -> (predecessor, edge_label, edge_weight)
        fwd: dict[str, tuple[str | None, str, float]] = {start: (None, "", 0.0)}
        bwd: dict[str, tuple[str | None, str, float]] = {goal: (None, "", 0.0)}
        fwd_frontier = [start]
        bwd_frontier = [goal]

        for depth in range(self._max_depth):
            if ctx is not None:
                ctx.ensure_budget(f"bidirectional walk depth {depth}")
            meet = await self._expand_frontier(fwd_frontier, fwd, bwd)
            if meet is not None:
                return self._stitch(meet, fwd, bwd)

            meet = await self._expand_frontier(bwd_frontier, bwd, fwd)
            if meet is not None:
                return self._stitch(meet, fwd, bwd)

        return None

    async def _expand_frontier(
        self,
        frontier: list[str],
        side: dict[str, tuple[str | None, str, float]],
        other: dict[str, tuple[str | None, str, float]],
    ) -> str | None:
        next_layer: list[str] = []
        for node in frontier:
            for nbr, weight, label in await self._neighbor_fn(node):
                if nbr not in side:
                    side[nbr] = (node, label, weight)
                    next_layer.append(nbr)
                if nbr in other:
                    return nbr
        frontier[:] = next_layer
        return None

    def _stitch(
        self,
        meet: str,
        fwd: dict[str, tuple[str | None, str, float]],
        bwd: dict[str, tuple[str | None, str, float]],
    ) -> ScoredPath:
        # Walk back from meet to start via fwd parents.
        left_nodes: list[str] = []
        left_edges: list[str] = []
        left_weights: list[float] = []
        node: str | None = meet
        while node is not None:
            left_nodes.append(node)
            pred, label, weight = fwd.get(node, (None, "", 0.0))
            if pred is not None:
                left_edges.append(label)
                left_weights.append(weight)
            node = pred
        left_nodes.reverse()
        left_edges.reverse()
        left_weights.reverse()

        # Walk forward from meet to goal via bwd parents.
        right_nodes: list[str] = []
        right_edges: list[str] = []
        right_weights: list[float] = []
        node = bwd.get(meet, (None, "", 0.0))[0]
        cur = meet
        while node is not None:
            _, label, weight = bwd.get(cur, (None, "", 0.0))
            right_nodes.append(node)
            right_edges.append(label)
            right_weights.append(weight)
            cur = node
            node = bwd.get(cur, (None, "", 0.0))[0]

        nodes = left_nodes + right_nodes
        edges = left_edges + right_edges
        weights = left_weights + right_weights
        return ScoredPath(
            nodes=nodes,
            edges=edges,
            score=score_path(nodes, self._weights, weights, length_penalty=self._length_penalty),
        )


class GraphWalkRetrieval(RetrievalStrategy):
    """RetrievalStrategy wrapper around :class:`DynamicGraphWalk`.

    Resolves seed entities for the query (via the injected ``seed_fn``),
    walks outward with PageRank-weighted beam search, and returns the
    highest-scoring paths as retrieval items.

    Args:
        graph_store: GraphStore for neighbor lookups and PageRank.
        seed_fn: Async callable(query, ctx) -> list[str] of start node ids.
        beam_width / max_depth / length_penalty: Walk parameters.
        use_pagerank: When True, fetch PageRank weights from the store.
    """

    def __init__(
        self,
        graph_store: Any,
        seed_fn: Callable[[str, Context], Awaitable[list[str]]],
        *,
        beam_width: int = 5,
        max_depth: int = 4,
        length_penalty: float = 0.1,
        use_pagerank: bool = True,
        max_paths: int = 10,
    ) -> None:
        super().__init__(graph_store=graph_store)
        self._seed_fn = seed_fn
        self._beam_width = beam_width
        self._max_depth = max_depth
        self._length_penalty = length_penalty
        self._use_pagerank = use_pagerank
        self._max_paths = max_paths

    async def _execute(self, query: str, ctx: Context, **kwargs: Any) -> RawSearchResult:
        seeds = await self._seed_fn(query, ctx)
        if not seeds:
            return RawSearchResult(records=[], metadata={"reason": "no seed entities"})

        weights: dict[str, float] = {}
        if self._use_pagerank:
            try:
                weights = await self._graph.pagerank()
            except Exception as exc:  # pragma: no cover - defensive
                ctx.log(f"PageRank unavailable, walking unweighted: {exc}")

        async def neighbor_fn(node_id: str) -> list[tuple[str, float, str]]:
            return await self._graph.weighted_neighbors(node_id)

        walk = DynamicGraphWalk(
            neighbor_fn,
            node_weights=weights,
            beam_width=self._beam_width,
            max_depth=self._max_depth,
            length_penalty=self._length_penalty,
        )

        all_paths: list[ScoredPath] = []
        for seed in seeds:
            all_paths.extend(await walk.beam_search(seed, ctx=ctx))
        all_paths.sort(key=lambda p: p.score, reverse=True)
        top = all_paths[: self._max_paths]

        records = [
            {
                "path": " -> ".join(p.nodes),
                "edges": p.edges,
                "score": p.score,
            }
            for p in top
        ]
        return RawSearchResult(records=records, metadata={"num_paths": len(top)})

    def _format(self, raw: RawSearchResult) -> RetrieverResult:
        items = [
            RetrieverResultItem(
                content=str(rec["path"]),
                score=float(rec.get("score", 0.0)),
                metadata={"edges": rec.get("edges", [])},
            )
            for rec in raw.records
        ]
        return RetrieverResult(items=items, metadata=raw.metadata)
