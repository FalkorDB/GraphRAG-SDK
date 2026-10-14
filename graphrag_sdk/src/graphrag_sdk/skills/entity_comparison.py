# GraphRAG SDK — Skills: Entity Comparison (Phase 3.4)

from __future__ import annotations

from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import SkillResult
from graphrag_sdk.skills.base import ENTITY_PARAM, Skill, skill_parameters

#: Properties that say nothing about what an entity is: SDK bookkeeping and
#: vectors. Comparing them only adds noise (and embeddings are huge).
_IGNORED_PROPERTY_MARKERS = ("embedding",)
_IGNORED_PROPERTIES = frozenset({"id", "source_chunk_ids", "chunk_ids", "mentions"})


def _comparable(props: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in props.items()
        if k not in _IGNORED_PROPERTIES
        and not any(marker in k.lower() for marker in _IGNORED_PROPERTY_MARKERS)
    }


class EntityComparisonSkill(Skill):
    """Compare two entities by their properties and graph neighborhoods."""

    name = "entity_comparison"
    description = (
        "Compare two entities: their attributes, shared neighbors, and what makes each distinct."
    )
    parameters = skill_parameters(
        {"entity_a": ENTITY_PARAM, "entity_b": ENTITY_PARAM}, ["entity_a", "entity_b"]
    )

    async def run(self, ctx: Context | None = None, **params: Any) -> SkillResult:
        if not params.get("entity_a") or not params.get("entity_b"):
            raise ValueError("entity_comparison requires 'entity_a' and 'entity_b'")
        entity_a = await self._resolve_entity(params["entity_a"], "entity_a")
        entity_b = await self._resolve_entity(params["entity_b"], "entity_b")

        props_a = _comparable(await self._properties(entity_a))
        props_b = _comparable(await self._properties(entity_b))
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
            f"Shared connections: {shared[:30]}. Unique to {entity_a}: {only_a[:30]}. "
            f"Unique to {entity_b}: {only_b[:30]}. Differing attributes: {differing}. "
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
        neighbors = await self._graph.weighted_neighbors(entity_id)
        return {n[0] for n in neighbors}
