# GraphRAG SDK — Skills: Impact Analysis (Phase 3.4)

from __future__ import annotations

from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import SkillResult
from graphrag_sdk.retrieval.graph_walk import DynamicGraphWalk
from graphrag_sdk.skills.base import Skill


class ImpactAnalysisSkill(Skill):
    """Estimate what a change to an entity ripples into.

    Walks outward from the target entity (weighted beam search) and ranks
    reachable entities by proximity and path score — the closer and more
    strongly connected, the higher the estimated impact.
    """

    name = "impact_analysis"
    description = (
        "Estimate the downstream impact of changing an entity by walking "
        "its outgoing graph neighborhood."
    )

    async def run(self, ctx: Context | None = None, **params: Any) -> SkillResult:
        entity = params.get("entity")
        if not entity:
            raise ValueError("impact_analysis requires 'entity'")
        max_depth = int(params.get("max_depth", 3))
        beam_width = int(params.get("beam_width", 8))

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
                if existing is None:
                    impacted[node] = {
                        "distance": hop,
                        "score": path.score,
                        "via": path.nodes,
                    }
                    continue
                if hop < existing["distance"]:
                    existing["distance"] = hop
                    existing["via"] = path.nodes
                if path.score > existing["score"]:
                    existing["score"] = path.score
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
