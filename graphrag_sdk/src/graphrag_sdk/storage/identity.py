# GraphRAG SDK — Storage: near-miss identity
#
# Finding entities that are probably the same thing but did not merge, because
# the merge is exact string equality on the display name and real sources do not
# agree on spelling. An HR export says "Maya Ellison"; the board review says
# "M. Ellison"; the graph keeps two people, one holding her age and the other
# holding what she did, and every question that needs both comes back wrong.
#
# This module only *finds* pairs. It never merges. See the note on recall below.

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

# Suffixes that name a company's legal form rather than the company. Dropped
# from the tail only: "Group" is noise in "Kestrel Grid Group" and is the whole
# name in "Group".
_LEGAL_SUFFIXES: frozenset[str] = frozenset(
    {
        "ab", "ag", "as", "asa", "bv", "co", "company", "corp", "corporation",
        "gmbh", "group", "holding", "holdings", "inc", "incorporated", "kk",
        "limited", "llc", "llp", "ltd", "nv", "oy", "plc", "pte", "pty", "sa",
        "sarl", "spa", "srl",
    }
)

# Written short in one source and long in another. Expanded before comparison so
# "Acme Corp" and "Acme Corporation" normalise to the same tokens.
_ABBREVIATIONS: dict[str, str] = {
    "co": "company",
    "corp": "corporation",
    "inc": "incorporated",
    "intl": "international",
    "ltd": "limited",
    "mfg": "manufacturing",
    "natl": "national",
    "svcs": "services",
    "tech": "technology",
    "univ": "university",
}

_WORD = re.compile(r"[^A-Za-z0-9]+")


def _tokens(name: str) -> list[str]:
    return [token for token in _WORD.split(name.lower()) if token]


def _core(name: str) -> list[str]:
    """Comparable tokens: abbreviations expanded, trailing legal forms dropped."""
    expanded = [_ABBREVIATIONS.get(token, token) for token in _tokens(name)]
    while len(expanded) > 1 and expanded[-1] in _LEGAL_SUFFIXES:
        expanded.pop()
    return expanded


def _initialled(a: list[str], b: list[str]) -> bool:
    """One given name written as an initial, same surname: Maya / M. Ellison."""
    if len(a) < 2 or len(b) < 2 or a[-1] != b[-1]:
        return False
    first_a, first_b = a[0], b[0]
    if first_a == first_b:
        return False
    if len(first_a) == 1 and first_b.startswith(first_a):
        return True
    return len(first_b) == 1 and first_a.startswith(first_b)


def why_same(name_a: str, name_b: str) -> str | None:
    """Why these two names might be one thing, or ``None``.

    Tuned for **recall**, not precision, and deliberately so: a pair this misses
    is a silent wrong answer, while a pair it wrongly surfaces costs the reader
    one line to dismiss. Nothing here merges anything, so a false positive is
    cheap and a false negative is not.

    Embeddings were measured on the same question and cannot do it: over pairs
    like these, "same" scored 0.601–0.980 and "different" scored 0.706–0.930, so
    the ranges overlap and no threshold exists. At the usual 0.90 cut that is two
    of seven real matches found and one wrong merge made — and a merge is not
    reversible. ``Maya Ellison`` / ``M. Ellison`` scores 0.601, *below*
    ``Acme Corporation`` / ``Acme Industries`` at 0.930.
    """
    core_a, core_b = _core(name_a), _core(name_b)
    if not core_a or not core_b or core_a == core_b == []:
        return None
    if name_a.strip().lower() == name_b.strip().lower():
        return None  # already merged by exact matching; not a near miss

    if core_a == core_b:
        return "the same name once legal suffixes and abbreviations are normalised"
    if _initialled(core_a, core_b):
        return "the same surname, with one given name written as an initial"
    if len(core_a) > 1 and sorted(core_a) == sorted(core_b):
        return "the same words in a different order"
    return None


@dataclass(frozen=True)
class NearMiss:
    """Two entities of one label that are probably one thing and did not merge."""

    label: str
    name_a: str
    name_b: str
    id_a: str
    id_b: str
    reason: str
    #: True when exactly one side came from a declared mapping. That is the case
    #: this feature exists for — a table row and a prose mention of one thing —
    #: and the one worth showing a user first.
    bridges_a_declared_source: bool

    def __str__(self) -> str:
        return f"{self.label} {self.name_a!r} ~ {self.name_b!r} — {self.reason}"


def find_near_misses(
    entities: Iterable[dict],
    *,
    limit: int = 50,
) -> list[NearMiss]:
    """Pairs of same-label entities whose names probably denote one thing.

    ``entities`` are dicts with ``id``, ``name``, ``label`` and ``is_stub`` — the
    shape the deduplicator already fetches. Comparison is within a label, because
    across labels the pair is a different problem with its own report.

    Cost is quadratic inside a label, so this is bounded: labels holding more
    than ``_MAX_PER_LABEL`` entities are skipped rather than allowed to dominate
    a finalize, and the skip is reported by the caller.
    """
    by_label: dict[str, list[dict]] = {}
    for entity in entities:
        label = (entity.get("label") or "").strip()
        name = (entity.get("name") or "").strip()
        if label and name:
            by_label.setdefault(label, []).append(entity)

    found: list[NearMiss] = []
    for label, group in sorted(by_label.items()):
        if len(group) > _MAX_PER_LABEL:
            continue
        for left, right in _pairs(group):
            reason = why_same(left["name"], right["name"])
            if not reason:
                continue
            declared = (left.get("is_stub") is not None) != (
                right.get("is_stub") is not None
            )
            found.append(
                NearMiss(
                    label=label,
                    name_a=left["name"],
                    name_b=right["name"],
                    id_a=left["id"],
                    id_b=right["id"],
                    reason=reason,
                    bridges_a_declared_source=declared,
                )
            )
            if len(found) >= limit:
                # Sorted below anyway; stopping early keeps a pathological graph
                # from turning finalize into an O(n^2) report generator.
                return _ranked(found)
    return _ranked(found)


#: Above this many entities under one label, the pairwise scan is skipped.
_MAX_PER_LABEL = 5_000


def _pairs(group: list[dict]) -> Iterator[tuple[dict, dict]]:
    for index, left in enumerate(group):
        for right in group[index + 1 :]:
            yield left, right


def _ranked(found: list[NearMiss]) -> list[NearMiss]:
    """A row-meets-prose pair first: that is the one this feature is about."""
    return sorted(found, key=lambda m: (not m.bridges_a_declared_source, m.label, m.name_a))
