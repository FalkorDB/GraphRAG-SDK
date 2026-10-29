# GraphRAG SDK — Retrieval: Dynamic Graph Walk
# PageRank-weighted traversal with beam search, bidirectional search,
# and path scoring. The core algorithms are decoupled from FalkorDB:
# they operate on an injected async ``neighbor_fn`` and an optional
# node-weight map, so they can be unit-tested in memory and reused by
# the toolkit's ``graph_walk`` tool.

from __future__ import annotations

import asyncio
import logging
import time
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

# An async function: node_id -> list of (neighbor_id, edge_weight, edge_label),
# optionally with a fourth element ``outgoing``: True when the edge is stored
# from node_id to the neighbour, False when it points at node_id.
NeighborFn = Callable[[str], Awaitable[list[tuple[Any, ...]]]]

#: Most neighbour lookups one bidirectional search runs.
DEFAULT_MAX_EXPANSIONS = 200
#: Wall-clock seconds one bidirectional search may take.
DEFAULT_TIME_BUDGET_SECONDS = 20.0

# A side of a bidirectional search: node -> (parent, edge label, edge weight,
# edge stored from parent to node (None if unknown), hops from the root).
_Side = dict[str, tuple[str | None, str, float, "bool | None", int]]
# A beam-search partial path: (nodes, edge labels, edge weights, directions).
_Entry = tuple[list[str], list[str], list[float], list["bool | None"]]


def _edge(item: tuple[Any, ...]) -> tuple[str, float, str, bool | None]:
    """Unpack a neighbour tuple; the direction is ``None`` when not reported."""
    outgoing = item[3] if len(item) > 3 and item[3] is not None else None
    return item[0], float(item[1]), item[2], None if outgoing is None else bool(outgoing)


class _LookupTimeout(Exception):
    """A neighbour lookup's own timeout, kept apart from the walk's budget."""


def _flip(forward: bool | None) -> bool | None:
    return None if forward is None else not forward


#: Neighbour lookups a walk runs at once. Below the connection pool's default
#: size (16), so a wide beam waits for a free connection instead of failing.
DEFAULT_MAX_CONCURRENT_LOOKUPS = 8


async def gather_limited(
    calls: list[Callable[[], Awaitable[Any]]], limit: int = DEFAULT_MAX_CONCURRENT_LOOKUPS
) -> list[Any]:
    """Run ``calls`` concurrently, at most ``limit`` at a time, results in order."""
    semaphore = asyncio.Semaphore(max(1, limit))

    async def run(call: Callable[[], Awaitable[Any]]) -> Any:
        async with semaphore:
            return await call()

    return list(await asyncio.gather(*(run(c) for c in calls)))


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
        max_concurrency: Neighbour lookups run at once within a round.
        max_expansions: Most neighbour lookups one bidirectional search runs.
        time_budget_seconds: Wall-clock budget of one bidirectional search.

    After :meth:`bidirectional_search`, :attr:`stopped` says why it ended:
    ``"complete"``, ``"expansion_limit"`` or ``"time_budget"``.
    """

    def __init__(
        self,
        neighbor_fn: NeighborFn,
        *,
        node_weights: dict[str, float] | None = None,
        beam_width: int = 5,
        max_depth: int = 4,
        length_penalty: float = 0.1,
        max_concurrency: int = DEFAULT_MAX_CONCURRENT_LOOKUPS,
        max_expansions: int = DEFAULT_MAX_EXPANSIONS,
        time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
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
        self._max_concurrency = max(1, int(max_concurrency))
        self._max_expansions = max(1, int(max_expansions))
        self._time_budget = float(time_budget_seconds)
        self.stopped = "complete"

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
        # Each beam entry: (path_nodes, path_edge_labels, path_edge_weights,
        # path_edge_directions).
        beam: list[_Entry] = [([start], [], [], [])]
        completed: list[ScoredPath] = []
        # Paths whose tail cannot be extended (leaf / all neighbors already on
        # the path). Kept so short terminal branches — e.g. a direct leaf
        # neighbor of ``start`` — still appear in goal-less results instead of
        # silently vanishing while deeper branches survive.
        terminal: list[ScoredPath] = []

        for depth in range(self._max_depth):
            if ctx is not None:
                ctx.ensure_budget(f"graph walk depth {depth}")
            candidates: list[_Entry] = []
            dead_ends: list[_Entry] = []
            # One round's neighbour lookups run concurrently (they are
            # independent reads), at most ``max_concurrency`` at a time;
            # results are consumed in beam order, so the outcome is the same
            # as fetching them one by one.
            to_expand = [entry for entry in beam if goal is None or entry[0][-1] != goal]
            fetched = await gather_limited(
                [(lambda tail=entry[0][-1]: self._neighbor_fn(tail)) for entry in to_expand],
                self._max_concurrency,
            )
            neighbours = {id(entry): nbrs for entry, nbrs in zip(to_expand, fetched)}
            for entry in beam:
                nodes, edges, e_weights, dirs = entry
                tail = nodes[-1]
                if goal is not None and tail == goal:
                    completed.append(self._scored(entry))
                    continue
                extended = False
                for item in neighbours[id(entry)]:
                    nbr, weight, label, forward = _edge(item)
                    if nbr in nodes:
                        continue  # no cycles within a single path
                    candidates.append(
                        (nodes + [nbr], edges + [label], e_weights + [weight], dirs + [forward])
                    )
                    extended = True
                if not extended and len(nodes) > 1:
                    dead_ends.append(entry)

            if not candidates:
                # Entire beam is unextendable; it already represents these
                # paths in the final results, so don't double-count dead ends.
                break

            terminal.extend(self._scored(entry) for entry in dead_ends)

            # Keep the top-k candidates by current path score.
            candidates.sort(
                key=lambda c: score_path(
                    c[0], self._weights, c[2], length_penalty=self._length_penalty
                ),
                reverse=True,
            )
            # Look for the goal among all candidates before the beam is cut: a
            # path reaching the goal must not be pruned for a low score (with
            # beam_width=1 a goal that is a direct neighbour was dropped).
            goal_hits = [c for c in candidates if c[0][-1] == goal] if goal is not None else []
            beam = candidates[: self._beam_width]

            if goal_hits:
                # Promote goal-reaching paths immediately.
                completed.extend(self._scored(entry) for entry in goal_hits)
                break

        results = completed or [self._scored(entry) for entry in beam] + terminal
        results.sort(key=lambda p: p.score, reverse=True)
        return results[: self._beam_width]

    def _scored(self, entry: _Entry) -> ScoredPath:
        nodes, edges, e_weights, dirs = entry
        return ScoredPath(
            nodes=list(nodes),
            edges=list(edges),
            edge_forward=list(dirs),
            score=score_path(nodes, self._weights, e_weights, length_penalty=self._length_penalty),
        )

    async def bidirectional_search(
        self,
        start: str,
        goal: str,
        *,
        ctx: Context | None = None,
    ) -> ScoredPath | None:
        """Search from both ``start`` and ``goal`` until the two sides meet.

        ``max_depth`` bounds the whole path: the start side walks
        ``ceil(max_depth / 2)`` hops and the goal side ``floor(max_depth / 2)``.
        Each side expands only its ``beam_width`` best-scoring nodes per hop.
        When the sides meet, the best-scoring path found at that hop is
        returned (oriented start→goal): a strong short path, not a proven
        optimum. At most ``max_expansions`` neighbour lookups run, within
        ``time_budget_seconds``; when either runs out the search returns what
        it found so far (usually ``None``) and :attr:`stopped` says which.
        """
        self.stopped = "complete"
        if start == goal:
            return ScoredPath(nodes=[start], edges=[], score=score_path([start], self._weights))

        sides: dict[bool, _Side] = {
            True: {start: (None, "", 0.0, None, 0)},
            False: {goal: (None, "", 0.0, None, 0)},
        }
        frontiers: dict[bool, list[str]] = {True: [start], False: [goal]}
        hops = {True: (self._max_depth + 1) // 2, False: self._max_depth // 2}
        deadline = time.monotonic() + self._time_budget
        expansions = 0

        for level in range(hops[True]):
            for forward in (True, False):
                frontier = frontiers[forward]
                if level >= hops[forward] or not frontier:
                    continue
                if ctx is not None:
                    ctx.ensure_budget(f"bidirectional walk depth {level}")
                room = self._max_expansions - expansions
                if room < len(frontier):
                    self.stopped = "expansion_limit"
                    frontier = frontier[: max(room, 0)]
                    if not frontier:
                        return None
                left = deadline - time.monotonic()
                try:
                    if left <= 0:
                        raise asyncio.TimeoutError
                    fetched = await asyncio.wait_for(
                        gather_limited(
                            [(lambda n=n: self._lookup(n)) for n in frontier],
                            self._max_concurrency,
                        ),
                        timeout=left,
                    )
                except _LookupTimeout as exc:
                    # The lookup itself timed out, not the walk's budget.
                    raise exc.__cause__ from None
                except asyncio.TimeoutError:
                    self.stopped = "time_budget"
                    return None
                expansions += len(frontier)

                side, other = sides[forward], sides[not forward]
                meets: list[ScoredPath] = []
                next_layer: list[str] = []
                for node, items in zip(frontier, fetched):
                    for item in items:
                        nbr, weight, label, out = _edge(item)
                        if nbr in other:
                            joined = self._join(forward, node, (nbr, weight, label, out), sides)
                            if joined is not None:
                                meets.append(joined)
                        if nbr not in side:
                            side[nbr] = (node, label, weight, out, level + 1)
                            next_layer.append(nbr)
                if meets:
                    # Highest score wins; ties go to the first found, so the
                    # result is deterministic for a deterministic neighbour source.
                    return max(meets, key=lambda p: p.score)
                if self.stopped != "complete":
                    return None
                next_layer.sort(key=lambda n: self._chain_score(side, n), reverse=True)
                frontiers[forward] = next_layer[: self._beam_width]
        return None

    async def _lookup(self, node: str) -> list[tuple[Any, ...]]:
        try:
            return await self._neighbor_fn(node)
        except asyncio.TimeoutError as exc:
            raise _LookupTimeout from exc

    @staticmethod
    def _chain(
        side: _Side, node: str
    ) -> tuple[list[str], list[str], list[float], list[bool | None]]:
        """The root→``node`` path of one search side, with each edge's
        direction relative to that order."""
        nodes, labels, weights, dirs = [node], [], [], []
        cur = node
        while True:
            parent, label, weight, forward, _ = side[cur]
            if parent is None:
                break
            nodes.append(parent)
            labels.append(label)
            weights.append(weight)
            dirs.append(forward)
            cur = parent
        nodes.reverse()
        labels.reverse()
        weights.reverse()
        dirs.reverse()
        return nodes, labels, weights, dirs

    def _chain_score(self, side: _Side, node: str) -> float:
        nodes, _, weights, _ = self._chain(side, node)
        return score_path(nodes, self._weights, weights, length_penalty=self._length_penalty)

    def _join(
        self,
        forward: bool,
        node: str,
        edge: tuple[str, float, str, bool | None],
        sides: dict[bool, _Side],
    ) -> ScoredPath | None:
        """Stitch the start→goal path through the edge ``node``—``nbr`` the
        ``forward`` side just found; ``None`` if it would revisit a node."""
        nbr, weight, label, out = edge
        if forward:
            near, far, mid_forward = node, nbr, out
        else:
            near, far, mid_forward = nbr, node, _flip(out)
        l_nodes, l_labels, l_weights, l_dirs = self._chain(sides[True], near)
        r_nodes, r_labels, r_weights, r_dirs = self._chain(sides[False], far)
        nodes = l_nodes + r_nodes[::-1]
        if len(set(nodes)) != len(nodes):
            return None
        labels = l_labels + [label] + r_labels[::-1]
        weights = l_weights + [weight] + r_weights[::-1]
        dirs = l_dirs + [mid_forward] + [_flip(d) for d in r_dirs[::-1]]
        return ScoredPath(
            nodes=nodes,
            edges=labels,
            edge_forward=dirs,
            score=score_path(nodes, self._weights, weights, length_penalty=self._length_penalty),
        )


def _path_text(p: ScoredPath) -> str:
    """``A -[R]-> B <-[S]- C``: each edge drawn in its stored direction
    (undirected ``-[R]-`` when the direction is unknown)."""
    parts = [p.nodes[0]] if p.nodes else []
    for i, node in enumerate(p.nodes[1:]):
        label = p.edges[i] if i < len(p.edges) else "?"
        forward = p.edge_forward[i] if i < len(p.edge_forward) else None
        arrow = {True: f"-[{label}]->", False: f"<-[{label}]-", None: f"-[{label}]-"}[forward]
        parts.append(f"{arrow} {node}")
    return " ".join(parts)


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
        if max_paths < 0:
            raise ValueError("max_paths must be >= 0")
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

        async def neighbor_fn(node_id: str) -> list[tuple[Any, ...]]:
            return await self._graph.weighted_neighbors(node_id, with_direction=True)

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
                "path": _path_text(p),
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
