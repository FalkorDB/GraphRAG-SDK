# GraphRAG SDK — Skills: Impact Analysis (Phase 3.4)

from __future__ import annotations

from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import SkillResult
from graphrag_sdk.retrieval.graph_walk import DynamicGraphWalk
from graphrag_sdk.skills.base import ENTITY_PARAM, Skill, skill_parameters

MAX_DEPTH = 5
MAX_BEAM_WIDTH = 20


class ImpactAnalysisSkill(Skill):
    """Estimate what a change to an entity ripples into.

    Walks the entity's neighbourhood (weighted beam search, in both edge
    directions) and ranks reachable entities by proximity and path score —
    the closer and more strongly connected, the higher the estimated impact.
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
                "description": f"Paths kept per hop, 1-{MAX_BEAM_WIDTH}, default 8.",
            },
        },
        ["entity"],
    )

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

        async def neighbor_fn(node_id: str) -> list[tuple[str, float, str]]:
            return await self._graph.weighted_neighbors(node_id)

        walk = DynamicGraphWalk(
            neighbor_fn,
            node_weights=weights,
            beam_width=beam_width,
            max_depth=max_depth,
        )
        paths = await walk.beam_search(entity, ctx=ctx)

        impacted: dict[str, dict[str, Any]] = {}
        for path in paths:
            for hop, node in enumerate(path.nodes[1:], start=1):
                existing = impacted.get(node)
                # Keep distance/score/via as one coherent record from the best
                # path: shortest hop first, then highest score at equal hops.
                if (
                    existing is None
                    or hop < existing["distance"]
                    or (hop == existing["distance"] and path.score > existing["score"])
                ):
                    impacted[node] = {
                        "distance": hop,
                        "score": path.score,
                        "via": path.nodes,
                    }
        ranked = sorted(
            ({"entity": k, **v} for k, v in impacted.items()),
            key=lambda d: (d["distance"], -d["score"]),
        )

        data = {"entity": entity, "impacted": ranked, "num_impacted": len(ranked)}
        summary = await self._summarize(
            ctx,
            f"Changing entity '{entity}' may affect these connected entities "
            f"(closest first): {[r['entity'] for r in ranked[:15]]}. "
            "Summarize the likely impact.",
        )
        return SkillResult(skill=self.name, summary=summary, data=data, sources=[entity])
