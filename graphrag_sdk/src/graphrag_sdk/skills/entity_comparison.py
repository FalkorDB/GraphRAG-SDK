# GraphRAG SDK — Skills: Entity Comparison (Phase 3.4)

from __future__ import annotations

from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import SkillResult
from graphrag_sdk.skills.base import Skill


class EntityComparisonSkill(Skill):
    """Compare two entities by their properties and graph neighborhoods."""

    name = "entity_comparison"
    description = (
        "Compare two entities: their attributes, shared neighbors, and what makes each distinct."
    )

    async def run(self, ctx: Context | None = None, **params: Any) -> SkillResult:
        entity_a = params.get("entity_a")
        entity_b = params.get("entity_b")
        if not entity_a or not entity_b:
            raise ValueError("entity_comparison requires 'entity_a' and 'entity_b'")

        props_a = await self._properties(entity_a)
        props_b = await self._properties(entity_b)
        nbrs_a = await self._neighbor_ids(entity_a)
        nbrs_b = await self._neighbor_ids(entity_b)

        shared = sorted(nbrs_a & nbrs_b)
        only_a = sorted(nbrs_a - nbrs_b)
        only_b = sorted(nbrs_b - nbrs_a)
        shared_keys = sorted(set(props_a) & set(props_b))
        differing = {
            k: {"a": props_a.get(k), "b": props_b.get(k)}
            for k in shared_keys
            if props_a.get(k) != props_b.get(k)
        }

        data = {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "shared_neighbors": shared,
            "neighbors_only_a": only_a,
            "neighbors_only_b": only_b,
            "differing_attributes": differing,
        }
        summary = await self._summarize(
            ctx,
            f"Compare entity '{entity_a}' and '{entity_b}'. "
            f"Shared connections: {shared}. Unique to {entity_a}: {only_a}. "
            f"Unique to {entity_b}: {only_b}. Differing attributes: {differing}. "
            "Write a concise comparison.",
        )
        return SkillResult(
            skill=self.name,
            summary=summary,
            data=data,
            sources=[entity_a, entity_b],
        )

    async def _properties(self, entity_id: str) -> dict[str, Any]:
        rows = await self._rows(
            "MATCH (e:__Entity__ {id: $id}) RETURN properties(e) AS props",
            {"id": entity_id},
        )
        if rows and rows[0]:
            return dict(rows[0][0] or {})
        return {}

    async def _neighbor_ids(self, entity_id: str) -> set[str]:
        try:
            neighbors = await self._graph.weighted_neighbors(entity_id)
            return {n[0] for n in neighbors}
        except Exception:
            rows = await self._rows(
                "MATCH (e:__Entity__ {id: $id})-[]-(m:__Entity__) RETURN DISTINCT m.id",
                {"id": entity_id},
            )
            return {r[0] for r in rows if r and r[0] is not None}
