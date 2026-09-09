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

import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Suffixes that name a company's legal form rather than the company. Dropped
# from the tail only: "Group" is noise in "Kestrel Grid Group" and is the whole
# name in "Group".
_LEGAL_SUFFIXES: frozenset[str] = frozenset(
    {
        "ab",
        "ag",
        "as",
        "asa",
        "bv",
        "co",
        "company",
        "corp",
        "corporation",
        "gmbh",
        "group",
        "holding",
        "holdings",
        "inc",
        "incorporated",
        "kk",
        "limited",
        "llc",
        "llp",
        "ltd",
        "nv",
        "oy",
        "plc",
        "pte",
        "pty",
        "sa",
        "sarl",
        "spa",
        "srl",
    }
)

# The subset of the above that is safe to strip when the answer DELETES A NODE.
#
# _LEGAL_SUFFIXES serves why_same, which only reports: over-stripping there costs
# a reader one line. canonical_key drives an irreversible merge, so it gets this
# shorter list instead. Every word left out — group, holding, holdings, spa, co,
# company — is an ordinary English word that routinely carries meaning rather
# than marking a legal form. Stripping them merged a parent company into its
# subsidiary: "Sony Group Corporation" into "Sony Corporation", "Roche Holding
# AG" into "Roche AG", "Blue Man Group" into "Blue Man", and the real surname
# "Lucio Co" into "Lucio". Those are separate legal entities and separate people.
_MERGE_SAFE_SUFFIXES: frozenset[str] = frozenset(
    {
        "ab",
        "ag",
        "as",
        "asa",
        "bv",
        "corp",
        "corporation",
        "gmbh",
        "inc",
        "incorporated",
        "kk",
        "limited",
        "llc",
        "llp",
        "ltd",
        "nv",
        "oy",
        "plc",
        "pte",
        "pty",
        "sa",
        "sarl",
        "srl",
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

# Unicode-aware on purpose. An ASCII-only class ([^A-Za-z0-9]) treats every
# accented letter as a separator, which collapses "Müller" and "Möller" — two
# real and different surnames — onto one key, and reduces any name with no Latin
# characters at all ("日本電力", "Газпром") to the empty string, putting every
# such name in a single group. Both are wrong merges, and a merge deletes a node.
_WORD = re.compile(r"[\W_]+")

# Characters that join a word rather than separate it. Splitting on an apostrophe
# turns "O'Brien" into two tokens and "OBrien" into one, so the two spellings of
# one name stop matching; dropping it first makes them agree.
_JOINERS = re.compile(r"['’ʼ]")

# Read aloud the same way, written either way.
_SYMBOL_WORDS = {"&": " and ", "+": " and ", "@": " at "}

# A symbol glued to the end of a word is part of the name, not punctuation
# between names: C# is not C, A+ is not A, C++ is neither. Splitting on it gave
# all of them the key of the bare letter and merged three languages into one
# node. Read aloud as the word, so "C sharp" and "C#" still agree.
_GLUED_SYMBOLS = (
    (re.compile(r"(?<=\w)\+\+(?!\w)"), " plus plus"),
    (re.compile(r"(?<=\w)\+(?!\w)"), " plus"),
    (re.compile(r"(?<=\w)#(?!\w)"), " sharp"),
)


def _uninvert(text: str) -> str:
    """Turn a single ``Surname, Given`` into ``Given Surname``.

    One comma is explicit evidence that a name was written back-to-front, which
    a bare reordering is not — see the note in :func:`canonical_key` on why word
    order is otherwise preserved. Skipped when the tail is a legal form, so
    ``Acme, Inc.`` keeps its suffix in the position the suffix rule looks for.
    """
    head, comma, tail = text.partition(",")
    if not comma or "," in tail:
        return text
    head, tail = head.strip(), tail.strip()
    if not head or not tail:
        return text
    tail_tokens = [t for t in _WORD.split(tail) if t]
    if any(_ABBREVIATIONS.get(t, t) in _LEGAL_SUFFIXES for t in tail_tokens):
        return text
    return f"{tail} {head}"


def _tokens(name: str) -> list[str]:
    text = _JOINERS.sub("", name.lower())
    for glued, word in _GLUED_SYMBOLS:
        text = glued.sub(word, text)
    for symbol, word in _SYMBOL_WORDS.items():
        text = text.replace(symbol, word)
    return [token for token in _WORD.split(_uninvert(text)) if token]


def _core(name: str) -> list[str]:
    """Comparable tokens: abbreviations expanded, trailing legal forms dropped."""
    expanded = [_ABBREVIATIONS.get(token, token) for token in _tokens(name)]
    while len(expanded) > 1 and expanded[-1] in _LEGAL_SUFFIXES:
        expanded.pop()
    return expanded


def _merge_core(name: str) -> list[str]:
    """Tokens for the merge path: at most ONE trailing legal form removed.

    The reporting variant pops a whole run of suffixes, which is how "Sony Group
    Corporation" lost both words and landed on "sony". Removing one is enough for
    every real spelling difference ("Kestrel Grid Ltd." / "Kestrel Grid") and
    leaves a second, meaning-bearing word where it is.
    """
    expanded = [_ABBREVIATIONS.get(token, token) for token in _tokens(name)]
    if len(expanded) > 1 and expanded[-1] in _MERGE_SAFE_SUFFIXES:
        expanded.pop()
    return expanded


def canonical_key(name: str) -> str:
    """The form two spellings of one name must share to be merged automatically.

    Lower-cased, punctuation-split, apostrophes and ``&`` normalised,
    ``Surname, Given`` un-inverted, abbreviations expanded, trailing legal forms
    dropped. Because both sides are reduced independently, the join is an
    equality rather than a pairwise comparison, so the order two sources arrive
    in does not change what merges.

    **Word order is deliberately preserved.** Sorting the tokens would join one
    more realistic pair (``Priya Raman`` / ``Raman, Priya``, which the comma rule
    now handles anyway) and would wrongly merge three: ``James Morgan`` with
    ``Morgan James``, ``Grace Newman`` with ``Newman Grace``, and ``Stanley
    Morgan`` with ``Morgan Stanley`` — a person and a bank. A merge is not
    reversible, so a pure reordering stays a *report* (:func:`why_same`) rather
    than becoming a merge.

    **Only one trailing legal form is removed, from a shorter list than the
    reporter uses** (:data:`_MERGE_SAFE_SUFFIXES`). Stripping a run from the full
    list merged nine realistic pairs that are separate entities, a parent company
    into its subsidiary among them. See that constant for why each word is out.

    Measured over seventeen realistic same-thing pairs and twenty-five that must
    stay apart: this merges fourteen with **zero** wrong merges. The three it does
    not reach are initial-against-full-given-name — ``M. Ellison`` and ``Maya
    Ellison`` — which no canonical form can unify without also merging every
    other M-surname person. Those are reported by :func:`why_same`, never merged,
    and so is every pair this deliberately declines to join.
    """
    return " ".join(_merge_core(name))


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
    a finalize. The skip is logged, so an empty report for such a label reads
    as "not checked" rather than "clean".
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
            logger.warning(
                "%s: %d entities exceed the %d-entity bound for the near-miss check; "
                "no probable duplicates were computed for this label",
                label,
                len(group),
                _MAX_PER_LABEL,
            )
            continue
        for left, right in _pairs(group):
            reason = why_same(left["name"], right["name"])
            if not reason:
                continue
            declared = (left.get("is_stub") is not None) != (right.get("is_stub") is not None)
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
