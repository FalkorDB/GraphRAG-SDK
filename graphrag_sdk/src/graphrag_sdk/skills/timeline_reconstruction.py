# GraphRAG SDK — Skills: Timeline Reconstruction

from __future__ import annotations

import calendar
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
_NUMBER_RE = re.compile(r"\s*[-+]?\d+(?:\.\d+)?\s*")
_DATE_RE = re.compile(r"\b(1\d{3}|2\d{3})(?:-(\d{1,2}))?(?:-(\d{1,2}))?\b")


def _is_date_key(key: str) -> bool:
    if key.lower() in _IGNORED_KEYS:
        return False
    words = {w.lower() for w in _WORD_RE.findall(key)}
    return bool(words & _DATE_WORDS) and not (words == {"on"})


def _is_year_key(key: str) -> bool:
    """Whether a plain number under ``key`` can be read as a year.

    The name must say so: it contains "year" or "date", or every word of it
    is a date word ("founded", "born", "start"). "start_headcount: 1500" is
    a head count, not the year 1500.
    """
    words = {w.lower() for w in _WORD_RE.findall(key)}
    return bool(words & {"year", "date"}) or (bool(words) and words <= _DATE_WORDS)


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return isinstance(value, str) and _NUMBER_RE.fullmatch(value) is not None


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

        match = f"(e:__Entity__:`{sanitize_cypher_label(label)}`)" if label else "(e:__Entity__)"
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

        # The scan reads dated entities in storage order and dates are only
        # parsed here, so a full scan may have missed earlier events.
        scan_full = len(rows) >= scan
        data: dict[str, Any] = {
            "timeline": timeline,
            "num_events": len(timeline),
            "truncated": scan_full or len(events) > limit,
        }
        if scan_full:
            data["note"] = (
                f"Only the first {scan} dated entities (in storage order) were read, "
                "so earlier events may be missing; narrow with label."
            )
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
        if _is_number(value) and not _is_year_key(key):
            continue
        match = _DATE_RE.search(str(value))
        if match:
            year = int(match.group(1))
            month = int(match.group(2)) if match.group(2) else 0
            day = int(match.group(3)) if match.group(3) else 0
            # 0 means "not given"; a given month or day must exist (no 2001-02-31).
            if match.group(2) and not 1 <= month <= 12:
                continue
            if match.group(3) and not 1 <= day <= calendar.monthrange(year, month)[1]:
                continue
            return (year, month, day), str(value), key
    return None
