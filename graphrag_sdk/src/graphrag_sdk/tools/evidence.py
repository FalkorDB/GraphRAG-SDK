# GraphRAG SDK — Tools: numbered evidence and citation checking
# Every result a tool returns that an answer may rely on gets a number [n]
# in the session's evidence ledger. The caller's agent cites those numbers;
# check_citations verifies them against the ledger.

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

#: Evidence kept per session. Older items are dropped first; a citation of a
#: dropped number reports it as invalid rather than matching something else.
DEFAULT_LEDGER_CAPACITY = 2000

#: A citation marker: ``[3]``, ``[1, 4]``, also glued to a word or a figure
#: (``plant[1]``, ``1998[2]``) or chained (``[1][2]``). Not when a number has
#: four or more digits (``[2024]`` is a year), or for ``[0]`` (evidence is
#: numbered from 1, so ``items[0]`` is code). Markers inside inline code or a
#: fenced code block are code and never touched.
_MARKER_RE = re.compile(r"\[(\s*\d{1,3}(?:\s*[,;]\s*\d{1,3})*\s*)\]")
#: Fenced code blocks (an unclosed fence runs to the end) and inline code.
_CODE_RE = re.compile(r"```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)|`[^`\n]*`")
#: Stands in for a removed marker until the whitespace around it is tidied.
_REMOVED = "\x00"


@dataclass
class Evidence:
    """One numbered piece of evidence a tool returned.

    ``n`` is the number an answer cites as ``[n]``. Numbers run across all
    tools of one session, so a citation is unambiguous whichever tool
    produced it.
    """

    n: int
    tool: str
    kind: str
    content: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n": self.n,
            "tool": self.tool,
            "kind": self.kind,
            "content": self.content,
        }
        if self.source:
            out["source"] = self.source
        if self.metadata:
            out["metadata"] = self.metadata
        return out


class EvidenceLedger:
    """A session's numbered evidence, shared by every tool call.

    Numbers only grow; once ``capacity`` items are held the oldest are
    dropped, so a long session keeps bounded memory.
    """

    def __init__(self, capacity: int = DEFAULT_LEDGER_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._items: OrderedDict[int, Evidence] = OrderedDict()
        self._next = 1

    def add(
        self,
        *,
        tool: str,
        kind: str,
        content: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Evidence:
        ev = Evidence(
            n=self._next,
            tool=tool,
            kind=kind,
            content=content,
            source=source,
            metadata=dict(metadata or {}),
        )
        self._next += 1
        self._items[ev.n] = ev
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)
        return ev

    def get(self, n: int) -> Evidence | None:
        return self._items.get(n)

    @property
    def items(self) -> list[Evidence]:
        return list(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, n: object) -> bool:
        return n in self._items


@dataclass
class GroundedAnswer:
    """An answer after its citations were checked.

    Attributes:
        answer: The answer with markers that match no evidence removed.
        grounded: Whether the answer cites at least one real piece of evidence.
        reason: ``"cited"``, ``"no_citations"``, ``"invalid_citations"`` or
            ``"empty"``.
        cited: Evidence numbers the answer cites that exist, in first-use order.
        invalid: Cited numbers that matched no evidence (removed from ``answer``).
    """

    answer: str
    grounded: bool
    reason: str
    cited: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)

    def citations(self, ledger: EvidenceLedger) -> list[Evidence]:
        return [ev for n in self.cited if (ev := ledger.get(n)) is not None]


def _clean_markers(text: str, valid: set[int] | EvidenceLedger) -> tuple[str, list[int], list[int]]:
    cited: list[int] = []
    invalid: list[int] = []

    def replace(match: re.Match[str]) -> str:
        numbers = [int(part.strip()) for part in re.split(r"[,;]", match.group(1))]
        if numbers == [0]:
            return match.group(0)
        keep: list[int] = []
        for n in numbers:
            if n in valid:
                keep.append(n)
                if n not in cited:
                    cited.append(n)
            elif n not in invalid:
                invalid.append(n)
        return f"[{', '.join(str(n) for n in keep)}]" if keep else _REMOVED

    # Code (fenced blocks, inline spans) is kept verbatim: odd pieces of the
    # split are code, even pieces are prose.
    pieces = _CODE_RE.split(text)
    codes = _CODE_RE.findall(text)
    out: list[str] = []
    for i, prose in enumerate(pieces):
        out.append(_MARKER_RE.sub(replace, prose.replace(_REMOVED, "")))
        if i < len(codes):
            out.append(codes[i])
    cleaned = "".join(out)

    # Tidy only where a marker was removed: "word [9]." -> "word.",
    # "a [9] b" -> "a b". Whitespace anywhere else is left as written.
    def tidy(match: re.Match[str]) -> str:
        after = cleaned[match.end() : match.end() + 1]
        if not after or after in ".,;:!?\n":
            return ""
        return " " if (match.group(1) or match.group(2)) else ""

    cleaned = re.sub(rf"([ \t]*){_REMOVED}(?:[ \t]*{_REMOVED})*([ \t]*)", tidy, cleaned)
    return cleaned.strip(), cited, invalid


def ground_answer(answer: str, ledger: EvidenceLedger) -> GroundedAnswer:
    """Check ``answer``'s ``[n]`` citations against ``ledger``.

    Markers that match no evidence are removed from the returned text. The
    answer is grounded when it cites at least one real piece of evidence.
    The check is that the answer rests on evidence, not that every sentence
    carries a citation.
    """
    raw = (answer or "").strip()
    if not raw:
        return GroundedAnswer("", False, "empty")
    cleaned, cited, invalid = _clean_markers(raw, ledger)
    if cited:
        return GroundedAnswer(cleaned, True, "cited", cited, invalid)
    reason = "invalid_citations" if invalid else "no_citations"
    return GroundedAnswer(cleaned, False, reason, cited, invalid)
