# GraphRAG SDK — Skills: Timeline Reconstruction

from __future__ import annotations

import re
import string
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import SkillResult
from graphrag_sdk.skills.base import Skill, skill_parameters
from graphrag_sdk.utils.cypher import sanitize_cypher_label

MAX_EVENTS = 500
#: Candidate entities read per requested event. The query keeps only entities
#: with a date-like property name; the exact word check and date parsing run
#: in Python, so a few candidates per event can still drop out.
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
        # Filter to entities with a date-like property *before* the LIMIT, so
        # undated entities cannot use up the scan and hide dated ones, and
        # return only those properties (never the embedding vector). The key
        # filter is a superset of ``_is_date_key`` (which then decides): any
        # date word as a substring, or "on" as a word of its own — not
        # inside a word like "description", but "startedOn", "published_on_day"
        # or "ON_HOLD" (preceded and followed by no lowercase letter, or a
        # capital "On").
        rows = await self._rows(
            f"MATCH {match} "
            "WITH e, [k IN keys(e) WHERE NOT toLower(k) CONTAINS 'embedding' "
            "AND (any(w IN $words WHERE toLower(k) CONTAINS w) "
            "OR any(i IN range(0, size(k) - 2) WHERE toLower(substring(k, i, 2)) = 'on' "
            "AND (i = 0 OR substring(k, i, 1) = 'O' OR NOT substring(k, i - 1, 1) IN $lower) "
            "AND (i + 2 = size(k) OR NOT substring(k, i + 2, 1) IN $lower))) "
            "| [k, e[k]]] AS dated "
            "WHERE size(dated) > 0 "
            "RETURN e.id AS id, e.name AS name, dated LIMIT $scan",
            {
                "scan": scan,
                "words": sorted(_DATE_WORDS - {"on"}),
                "lower": list(string.ascii_lowercase),
            },
        )

        events: list[dict[str, Any]] = []
        for row in rows:
            if not row:
                continue
            entity_id = row[0]
            name = row[1] if len(row) > 1 and row[1] is not None else entity_id
            props = {str(k): v for k, v in (row[2] or [])} if len(row) > 2 else {}
            found = _extract_date(props)
            if found is not None:
                sort_key, raw, key = found
                events.append(
                    {
                        "entity": entity_id,
                        "name": name,
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
