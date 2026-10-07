# GraphRAG SDK — Skills: Contradiction Detection (Phase 3.4)

from __future__ import annotations

import json
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import SkillResult
from graphrag_sdk.skills.base import ENTITY_PARAM, Skill, skill_parameters

MAX_FACTS = 200


class ContradictionDetectionSkill(Skill):
    """Detect conflicting facts about an entity in the knowledge graph.

    Gathers the relationships and descriptions attached to an entity and
    asks the LLM to flag mutually inconsistent statements. Without an LLM
    it returns the collected facts for external inspection.
    """

    name = "contradiction_detection"
    description = "Find contradictory or mutually inconsistent facts about an entity in the graph."
    parameters = skill_parameters(
        {
            "entity": ENTITY_PARAM,
            "limit": {"type": "integer", "description": f"Facts examined, 1-{MAX_FACTS}."},
        },
        ["entity"],
    )

    async def run(self, ctx: Context | None = None, **params: Any) -> SkillResult:
        if not params.get("entity"):
            raise ValueError("contradiction_detection requires 'entity'")
        limit = self._int_param(params, "limit", 50, -(10**9), 10**9)
        if limit < 1:
            raise ValueError("contradiction_detection 'limit' must be >= 1")
        limit = min(limit, MAX_FACTS)
        entity = await self._resolve_entity(params["entity"])

        rows = await self._rows(
            "MATCH (e:__Entity__ {id: $id})-[r:RELATES]-(m:__Entity__) "
            "RETURN coalesce(r.rel_type, type(r)) AS rel, m.id AS other, "
            "coalesce(r.fact, r.description, '') AS desc "
            "ORDER BY rel, other LIMIT $limit",
            {"id": entity, "limit": limit},
        )
        facts = [
            {"relation": r[0], "other": r[1], "description": r[2] if len(r) > 2 else ""}
            for r in rows
            if r
        ]

        contradictions: list[dict[str, Any]] = []
        summary = ""
        if self._llm is not None and facts:
            prompt = (
                f"Here are facts about entity '{entity}':\n"
                + "\n".join(f"- {f['relation']} {f['other']}: {f['description']}" for f in facts)
                + "\n\nIdentify any pairs of facts that contradict each other. "
                'Respond as JSON: {"contradictions": [{"a": "...", "b": "...", '
                '"reason": "..."}], "summary": "..."}'
            )
            raw = await self._summarize(ctx, prompt)
            parsed = _safe_json(raw)
            if isinstance(parsed, dict):
                found = parsed.get("contradictions", []) or []
                contradictions = [c for c in found if isinstance(c, dict)]
                summary = str(parsed.get("summary", "") or "")

        data = {
            "entity": entity,
            "facts_examined": len(facts),
            "contradictions": contradictions,
        }
        return SkillResult(skill=self.name, summary=summary, data=data, sources=[entity])


def _safe_json(text: str) -> Any:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return None
