# GraphRAG SDK — Skills: Impact Analysis

from __future__ import annotations

from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import SkillResult
from graphrag_sdk.retrieval.graph_walk import gather_limited, score_path
from graphrag_sdk.skills.base import ENTITY_PARAM, Skill, skill_parameters

MAX_DEPTH = 5
MAX_BEAM_WIDTH = 20
#: Direct neighbours fetched for the entity itself.
MAX_DIRECT_NEIGHBOURS = 1000
#: Neighbours fetched per entity at later hops (the strongest first).
HOP_NEIGHBOURS = 50


class ImpactAnalysisSkill(Skill):
    """Estimate what a change to an entity ripples into.

    Walks the entity's neighbourhood breadth-first (both edge directions,
    expanding the best-scored entities of each hop) and ranks reachable
    entities by proximity and path score — the closer and more strongly
    connected, the higher the estimated impact.
    """

    name = "impact_analysis"
    description = (
        "Estimate the impact of changing an entity by walking its connected "
        "neighbourhood (both edge directions); closest entities first."
    )
    parameters = skill_parameters(
        {
            "entity": ENTITY_PARAM,
            "max_depth": {"type": "integer", "description": f"Hops, 1-{MAX_DEPTH}, default 3."},
            "beam_width": {
                "type": "integer",
                "description": (
                    f"Entities expanded to the next hop, 1-{MAX_BEAM_WIDTH}, default 8 "
                    "(every entity reached is reported)."
                ),
            },
        },
        ["entity"],
    )

    async def _neighbours(self, node: str, limit: int) -> tuple[list[tuple[Any, ...]], bool]:
        """The ``limit`` strongest neighbours, and whether there were more."""
        try:
            found = await self._graph.weighted_neighbors(node, limit=limit + 1)
        except TypeError:  # a graph store whose lookup takes no limit
            found = await self._graph.weighted_neighbors(node)
        return found[:limit], len(found) > limit

    async def run(self, ctx: Context | None = None, **params: Any) -> SkillResult:
        if not params.get("entity"):
            raise ValueError("impact_analysis requires 'entity'")
        entity = await self._resolve_entity(params["entity"])
        max_depth = self._int_param(params, "max_depth", 3, 1, MAX_DEPTH)
        beam_width = self._int_param(params, "beam_width", 8, 1, MAX_BEAM_WIDTH)

        try:
            weights = await self._graph.pagerank()
        except Exception:
            weights = {}

        # Breadth-first by hop: every neighbour reached at a hop is reported
        # at that distance (a beam would keep only ``beam_width`` paths and
        # drop most direct neighbours). ``beam_width`` bounds how many of a
        # hop's entities, best path score first, are expanded to the next
        # hop. Each hop's neighbour lookups run concurrently.
        impacted: dict[str, dict[str, Any]] = {}
        capped = False
        seen = {entity}
        frontier: list[tuple[list[str], list[float]]] = [([entity], [])]
        for hop in range(1, max_depth + 1):
            if not frontier:
                break
            if ctx is not None:
                ctx.ensure_budget(f"impact analysis hop {hop}")
            # The entity's own neighbours are fetched in full (up to
            # MAX_DIRECT_NEIGHBOURS): they are the most directly impacted.
            # Lookups run at most a few at a time (the connection pool).
            limit = MAX_DIRECT_NEIGHBOURS if hop == 1 else HOP_NEIGHBOURS
            fetched = await gather_limited(
                [(lambda node=nodes[-1]: self._neighbours(node, limit)) for nodes, _ in frontier]
            )
            capped = capped or any(more for _, more in fetched)
            best: dict[str, tuple[float, list[str], list[float]]] = {}
            for (nodes, edge_weights), (neighbours, _) in zip(frontier, fetched):
                for nbr, weight, *_rest in neighbours:
                    if nbr in seen:
                        continue
                    path = nodes + [nbr]
                    path_weights = edge_weights + [weight]
                    score = score_path(path, weights, path_weights)
                    if nbr not in best or score > best[nbr][0]:
                        best[nbr] = (score, path, path_weights)
            for nbr, (score, path, _) in best.items():
                impacted[nbr] = {"distance": hop, "score": score, "via": path}
            seen.update(best)
            expand = sorted(best.values(), key=lambda b: (-b[0], b[1][-1]))[:beam_width]
            frontier = [(path, path_weights) for _, path, path_weights in expand]
        ranked = sorted(
            ({"entity": k, **v} for k, v in impacted.items()),
            key=lambda d: (d["distance"], -d["score"], d["entity"]),
        )

        # ``truncated``: some entity had more neighbours than were fetched, so
        # entities past those caps were not reached.
        data = {
            "entity": entity,
            "impacted": ranked,
            "num_impacted": len(ranked),
            "truncated": capped,
        }
        summary = await self._summarize(
            ctx,
            f"Changing entity '{entity}' may affect these connected entities "
            f"(closest first): {[r['entity'] for r in ranked[:15]]}. "
            "Summarize the likely impact.",
        )
        return SkillResult(skill=self.name, summary=summary, data=data, sources=[entity])
