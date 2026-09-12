# GraphRAG SDK — Skills: Timeline Reconstruction (Phase 3.4)

from __future__ import annotations

import re
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import SkillResult
from graphrag_sdk.skills.base import Skill
from graphrag_sdk.utils.cypher import sanitize_cypher_label

_DATE_KEYS = ("date", "year", "time", "timestamp", "when", "created", "start", "end")
_YEAR_RE = re.compile(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?")


class TimelineReconstructionSkill(Skill):
    """Reconstruct a chronological timeline of entities/events.

    Collects entities that carry a temporal attribute, sorts them by the
    parsed date, and optionally narrates the sequence with the LLM.
    """

    name = "timeline_reconstruction"
    description = (
        "Reconstruct a chronological timeline from entities that carry "
        "temporal attributes (dates, years)."
    )

    async def run(self, ctx: Context | None = None, **params: Any) -> SkillResult:
        label = params.get("label")
        limit = int(params.get("limit", 100))

        if label:
            safe_label = sanitize_cypher_label(label)
            cypher = (
                f"MATCH (e:`{safe_label}`) RETURN e.id AS id, properties(e) AS props LIMIT {limit}"
            )
        else:
            cypher = f"MATCH (e:__Entity__) RETURN e.id AS id, properties(e) AS props LIMIT {limit}"
        rows = await self._rows(cypher)

        events: list[dict[str, Any]] = []
        for row in rows:
            if not row:
                continue
            entity_id = row[0]
            props = dict(row[1] or {}) if len(row) > 1 else {}
            sort_key = _extract_date(props)
            if sort_key is not None:
                events.append({"entity": entity_id, "date": sort_key[1], "sort": sort_key[0]})

        events.sort(key=lambda e: e["sort"])
        timeline = [{"entity": e["entity"], "date": e["date"]} for e in events]

        data = {"timeline": timeline, "num_events": len(timeline)}
        summary = await self._summarize(
            ctx,
            "Reconstruct the timeline from these dated events: "
            f"{timeline[:30]}. Narrate the chronological sequence.",
        )
        return SkillResult(
            skill=self.name,
            summary=summary,
            data=data,
            sources=[e["entity"] for e in timeline[:10]],
        )


def _extract_date(props: dict[str, Any]) -> tuple[tuple[int, int, int], str] | None:
    """Find a temporal value in an entity's properties and parse it.

    Returns ``((year, month, day), raw_string)`` or ``None``.
    """
    for key, value in props.items():
        if value is None:
            continue
        if any(token in key.lower() for token in _DATE_KEYS):
            match = _YEAR_RE.search(str(value))
            if match:
                year = int(match.group(1))
                month = int(match.group(2)) if match.group(2) else 0
                day = int(match.group(3)) if match.group(3) else 0
                return (year, month, day), str(value)
    return None
