# GraphRAG SDK — Ingestion: where a chunk names an entity, and the text around it
#
# Each mention of an entity records its spans (every place the chunk names it)
# on its MENTIONED_IN edge, with what the chunk says the entity is. That is the
# extraction model's description; when the model gave none — a chunk whose
# verification step failed keeps the NER entities, which carry none — it is the
# sentence around the spans, cut from the chunk here. Nothing here calls a
# model, so the same chunk and spans always give the same text.

from __future__ import annotations

import bisect
import re
import unicodedata
from collections.abc import Iterable

#: Longest text cut for one mention, in characters.
MAX_EVIDENCE_CHARS = 400
#: How far a window reaches on each side of a span when no sentence boundary is
#: closer — a table row or a PDF line without punctuation is not a sentence.
_REACH = 180
#: Joins two windows of one chunk that do not touch.
WINDOW_JOIN = " … "

# A sentence ends at ., ! or ? followed by whitespace; at a blank line; or at a
# line break that starts a new block (a list item, a table row, a heading). A
# single line break inside a paragraph does not end it: PDFs and e-mails wrap
# lines mid-sentence.
_END = re.compile(r"[.!?](?=\s)|\n[ \t]*\n|\n(?=[ \t]*(?:[-*•|#>]|\d+[.)]\s))")


def _sentence_window(text: str, start: int, end: int) -> tuple[int, int]:
    """The sentence holding ``text[start:end]``, at most ``_REACH`` characters
    on either side of it."""
    lo_limit = max(0, start - _REACH)
    hi_limit = min(len(text), end + _REACH)
    lo = lo_limit
    for m in _END.finditer(text, lo_limit, start):
        lo = m.end()
    if lo == lo_limit and lo > 0 and not text[lo - 1].isspace():
        # No boundary in reach: start at a word, not inside one.
        while lo < start and not text[lo].isspace():
            lo += 1
    while lo < start and text[lo].isspace():
        lo += 1
    hi = hi_limit
    m = _END.search(text, end, hi_limit)
    if m:
        hi = m.end() if text[m.start()] != "\n" else m.start()
    elif hi < len(text):
        while hi > end and not text[hi - 1].isspace():
            hi -= 1
    return lo, max(hi, end)


_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _line_at(text: str, pos: int) -> tuple[int, int]:
    lo = text.rfind("\n", 0, pos) + 1
    hi = text.find("\n", pos)
    return lo, len(text) if hi < 0 else hi


def _table_header(text: str, pos: int) -> tuple[int, int] | None:
    """The header line of the markdown table whose row holds ``pos``, when
    ``pos`` is on a table row below the header. A row alone ("| AOI | 5 of 24 |
    pass |") does not say what its cells are; its header does."""
    lo, hi = _line_at(text, pos)
    if not text[lo:hi].lstrip().startswith("|"):
        return None
    lines: list[tuple[int, int]] = []
    while lo > 0:
        prev_lo, prev_hi = _line_at(text, lo - 1)
        if not text[prev_lo:prev_hi].lstrip().startswith("|"):
            break
        lines.append((prev_lo, prev_hi))
        lo = prev_lo
    # The table's first line is its header when the line under it is the
    # separator ("|---|---|"); the row itself is not its own header.
    if len(lines) >= 2 and _TABLE_SEPARATOR.match(text[lines[-2][0] : lines[-2][1]]):
        return lines[-1]
    return None


def _with_table_headers(text: str, windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    headers = [h for lo, _hi in windows if (h := _table_header(text, lo)) is not None]
    return windows + headers


def _merge(windows: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for lo, hi in sorted(windows):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _render(text: str, windows: list[tuple[int, int]], max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    parts = [" ".join(text[lo:hi].split()) for lo, hi in windows]
    joined = WINDOW_JOIN.join(p for p in parts if p)
    if len(joined) > max_chars:
        joined = joined[: max_chars - 1].rstrip() + "…"
    return joined


def evidence_text(
    text: str, spans: Iterable[tuple[int, int]], *, max_chars: int = MAX_EVIDENCE_CHARS
) -> str:
    """The sentence around each span of ``text``, overlapping sentences merged,
    whitespace collapsed, capped at ``max_chars``. Empty when no span lies
    inside ``text``."""
    windows = [_sentence_window(text, s, e) for s, e in spans if 0 <= s < e <= len(text)]
    if not windows:
        return ""
    return _render(text, _merge(_with_table_headers(text, windows)), max_chars)


# Scripts written without spaces between words: a name inside a longer run of
# them is still the name (杭州 in 在杭州成立), so no word boundary is asked for.
_UNSPACED_SCRIPTS = (
    "CJK",
    "HIRAGANA",
    "KATAKANA",
    "HALFWIDTH KATAKANA",
    "THAI",
    "LAO",
    "KHMER",
    "MYANMAR",
    "TIBETAN",
)


def _word_char(c: str) -> bool:
    """A letter, digit or combining mark (an accent written apart, a Devanagari
    vowel sign), or ``_``."""
    return c == "_" or unicodedata.category(c)[0] in "LNM"


def _script(c: str, *prefixes: str) -> bool:
    return unicodedata.name(c, "").startswith(prefixes)


def _one_word(outside: str, inside: str, *, after: bool) -> bool:
    """Whether a match's edge character ``inside`` and its neighbour
    ``outside`` read as one word, so the match is cut out of a longer one
    ("us" in "business", ".NET" in "ASP.NET", "C++" in "C++11", "Cafe" before
    a separate accent). Never where either is of a script without spaces;
    nor after a Korean name, which takes its particles at the end (서울에서)."""
    if not _word_char(outside):
        return False
    if _script(outside, *_UNSPACED_SCRIPTS) or _script(inside, *_UNSPACED_SCRIPTS):
        return False
    return not (after and _script(outside, "HANGUL") and _script(inside, "HANGUL"))


def is_whole_word(text: str, start: int, end: int) -> bool:
    """Whether ``text[start:end]`` is not part of a longer word ("US" is in
    "the US." but not in "business", nor "Al" in "Algorithm")."""
    if not 0 <= start < end <= len(text):
        return False
    return not (
        (start > 0 and _one_word(text[start - 1], text[start], after=False))
        or (end < len(text) and _one_word(text[end], text[end - 1], after=True))
    )


def find_spans(text: str, name: str) -> list[tuple[int, int]]:
    """Every occurrence of ``name`` in ``text`` as a whole word
    (:func:`is_whole_word`), whatever its case, none overlapping another.

    A name written with capitals is not found all-lowercase when it is found
    otherwise ("Apple" and "APPLE" the company, not "apple" the fruit); a name
    in capitals never is ("US" is not the pronoun "us", while "NASA" is found
    as "Nasa"). Where two occurrences overlap, the one written as the name is
    kept. Case is compared by Unicode case folding, so "I" is not the Turkish
    dotless "ı"."""
    name = name.strip()
    if not name:
        return []
    folded = name.casefold()
    found = [
        m.span(1)
        for m in re.finditer(f"(?=({re.escape(name)}))", text, re.IGNORECASE)
        if text[m.start(1) : m.end(1)].casefold() == folded and is_whole_word(text, *m.span(1))
    ]
    if any(c.isupper() for c in name):
        cased = [(s, e) for s, e in found if not text[s:e].islower()]
        found = cased if cased or name.isupper() else found
    exact = [(s, e) for s, e in found if text[s:e] == name]
    kept: list[tuple[int, int]] = []
    for s, e in exact + [span for span in found if text[span[0] : span[1]] != name]:
        i = bisect.bisect_left(kept, (s, e))
        if (i == 0 or kept[i - 1][1] <= s) and (i == len(kept) or kept[i][0] >= e):
            kept.insert(i, (s, e))
    return kept


def flat_spans(spans: Iterable[tuple[int, int]]) -> list[int]:
    """Spans as one flat list of ints, ``[start, end, start, end, ...]`` — the
    form a graph property can hold (a list of primitives)."""
    out: list[int] = []
    seen: set[tuple[int, int]] = set()
    for s, e in spans:
        pair = (int(s), int(e))
        if pair not in seen:
            seen.add(pair)
            out += [pair[0], pair[1]]
    return out
