# GraphRAG SDK — Skills: Timeline Reconstruction (Phase 3.4)

from __future__ import annotations

import re
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import SkillResult
from graphrag_sdk.skills.base import Skill, skill_parameters
from graphrag_sdk.utils.cypher import sanitize_cypher_label

MAX_EVENTS = 500
#: Entities scanned per requested event: properties are filtered in Python,
#: so the query reads more rows than it returns.
_SCAN_FACTOR = 5
_MAX_SCAN = 5000

#: Words in a property name that mark it as a date. Matched against whole
#: words of the name ("start_date", "foundedYear"), never substrings, so
#: "friend" or "legend" do not count as "end".
_DATE_WORDS = frozenset(
    {"date", "year", "time", "timestamp", "when", "start", "end", "founded", "born", "died", "on"}
)
#: SDK bookkeeping written at ingestion, which says nothing about the event.
_IGNORED_KEYS = frozenset({"created_at", "updated_at", "ingested_at", "created", "updated"})
_WORD_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_DATE_RE = re.compile(r"\b(1\d{3}|2\d{3})(?:-(\d{1,2}))?(?:-(\d{1,2}))?\b")


def _is_date_key(key: str) -> bool:
    if key.lower() in _IGNORED_KEYS:
        return False
    words = {w.lower() for w in _WORD_RE.findall(key)}
    return bool(words & _DATE_WORDS) and not (words == {"on"})


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
    parameters = skill_parameters(
        {
            "label": {"type": "string", "description": "Only entities of this type."},
            "limit": {
                "type": "integer",
                "description": f"Events returned, 1-{MAX_EVENTS}, default 100.",
            },
        }
    )

    async def run(self, ctx: Context | None = None, **params: Any) -> SkillResult:
        label = params.get("label")
        limit = self._int_param(params, "limit", 100, 1, MAX_EVENTS)
        scan = min(limit * _SCAN_FACTOR, _MAX_SCAN)

        match = f"(e:`{sanitize_cypher_label(label)}`)" if label else "(e:__Entity__)"
        rows = await self._rows(
            f"MATCH {match} RETURN e.id AS id, properties(e) AS props LIMIT $scan",
            {"scan": scan},
        )

        events: list[dict[str, Any]] = []
        for row in rows:
            if not row:
                continue
            entity_id = row[0]
            props = dict(row[1] or {}) if len(row) > 1 else {}
            found = _extract_date(props)
            if found is not None:
                sort_key, raw, key = found
                events.append(
                    {
                        "entity": entity_id,
                        "name": props.get("name", entity_id),
                        "date": raw,
                        "attribute": key,
                        "sort": sort_key,
                    }
                )

        events.sort(key=lambda e: (e["sort"], str(e["entity"])))
        timeline = [
            {
                "entity": e["entity"],
                "name": e["name"],
                "date": e["date"],
                "attribute": e["attribute"],
            }
            for e in events[:limit]
        ]

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


def _extract_date(props: dict[str, Any]) -> tuple[tuple[int, int, int], str, str] | None:
    """Find a temporal value in an entity's properties and parse it.

    Returns ``((year, month, day), raw_string, property_name)`` or ``None``.
    """
    for key in sorted(props):
        value = props[key]
        if value is None or not _is_date_key(key):
            continue
        match = _DATE_RE.search(str(value))
        if match:
            year = int(match.group(1))
            month = int(match.group(2)) if match.group(2) else 0
            day = int(match.group(3)) if match.group(3) else 0
            if month > 12 or day > 31:
                continue
            return (year, month, day), str(value), key
    return None
