# GraphRAG SDK — Skills: Contradiction Detection (Phase 3.4)

from __future__ import annotations

import json
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import SkillResult
from graphrag_sdk.skills.base import Skill


class ContradictionDetectionSkill(Skill):
    """Detect conflicting facts about an entity in the knowledge graph.

    Gathers the relationships and descriptions attached to an entity and
    asks the LLM to flag mutually inconsistent statements. Without an LLM
    it returns the collected facts for external inspection.
    """

    name = "contradiction_detection"
    description = "Find contradictory or mutually inconsistent facts about an entity in the graph."

    async def run(self, ctx: Context | None = None, **params: Any) -> SkillResult:
        entity = params.get("entity")
        if not entity:
            raise ValueError("contradiction_detection requires 'entity'")
        limit = int(params.get("limit", 50))
        if limit < 1:
            raise ValueError("contradiction_detection 'limit' must be >= 1")

        rows = await self._rows(
            "MATCH (e:__Entity__ {id: $id})-[r]-(m:__Entity__) "
            "RETURN type(r) AS rel, m.id AS other, "
            "coalesce(r.description, '') AS desc "
            f"LIMIT {limit}",
            {"id": entity},
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
                contradictions = parsed.get("contradictions", []) or []
                summary = parsed.get("summary", "") or ""

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
