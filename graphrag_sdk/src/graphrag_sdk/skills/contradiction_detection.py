# GraphRAG SDK — Skills: Contradiction Detection

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
    asks the LLM to flag mutually inconsistent statements. The examined
    facts are always returned in ``data["facts"]``, so a caller without an
    LLM (e.g. an MCP client's own agent) can judge them itself.
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
            "coalesce(r.fact, r.description, '') AS desc, "
            "coalesce(m.name, m.id) AS other_name, coalesce(e.name, e.id) AS entity_name "
            "ORDER BY rel, other LIMIT $limit",
            {"id": entity, "limit": limit + 1},
        )
        truncated = len(rows) > limit
        rows = rows[:limit]
        facts = [
            {
                "relation": r[0],
                "other": r[1],
                "other_name": r[3] if len(r) > 3 and r[3] is not None else r[1],
                "description": r[2] if len(r) > 2 else "",
            }
            for r in rows
            if r
        ]
        entity_name = next((r[4] for r in rows if r and len(r) > 4 and r[4]), entity)

        contradictions: list[dict[str, Any]] = []
        summary = ""
        llm_error: str | None = None
        checked = False
        if self._llm is not None and facts:
            prompt = (
                f"Here are facts about entity '{entity_name}':\n"
                + "\n".join(
                    f"- {f['relation']} {f['other_name']}: {f['description']}" for f in facts
                )
                + "\n\nIdentify any pairs of facts that contradict each other. "
                'Respond as JSON: {"contradictions": [{"a": "...", "b": "...", '
                '"reason": "..."}], "summary": "..."}'
            )
            raw, llm_error = await self._ask(ctx, prompt)
            parsed = _safe_json(raw) if llm_error is None else None
            found = parsed.get("contradictions") if isinstance(parsed, dict) else None
            if isinstance(found, list):
                contradictions = [c for c in found if isinstance(c, dict)]
                summary = str(parsed.get("summary", "") or "")
                checked = True
            elif llm_error is None:
                llm_error = "LLM reply was not the requested JSON"

        data: dict[str, Any] = {
            "entity": entity,
            "entity_name": entity_name,
            "facts_examined": len(facts),
            "facts": facts,
            "contradictions": contradictions,
            "checked_by_llm": checked,
            "truncated": truncated,
        }
        if llm_error is not None:
            data["llm_error"] = f"{llm_error}; judge the facts yourself."
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
