# GraphRAG SDK — Agentic Retrieval: citations and the grounding gate
# Every tool result the model may rely on is numbered in the run's evidence
# ledger. The answer cites those numbers as [N]. This module checks the
# citations against the ledger, removes the ones that point at nothing, and
# decides whether the answer is grounded.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from graphrag_sdk.retrieval.agentic.tools import Evidence, EvidenceLedger

GroundingPolicy = Literal["strict", "annotate", "off"]

#: Replaces an ungrounded answer under the ``strict`` policy.
DEFAULT_UNGROUNDED_ANSWER = "I couldn't find information about that in the knowledge graph."

#: Shown when the latency budget ran out before the model answered: the
#: question went unanswered for lack of time, not for lack of information.
DEFAULT_TIMEOUT_ANSWER = (
    "I ran out of time before finding an answer in the knowledge graph. Please try again."
)

#: An uncited answer this short, given without any tool call and stating no
#: figures, is treated as conversation ("Hello!", "You're welcome") rather
#: than an unsupported claim.
DEFAULT_MAX_UNCITED_CHARS = 280

#: A citation marker: ``[3]``, ``[1, 4]``, also glued to a word or a figure
#: (``plant[1]``, ``1998[2]``, ``(CEO)[3]``) or chained (``[1][2]``). Not when
#: a number has four or more digits (``[2024]`` is a year), or for ``[0]``
#: (evidence is numbered from 1, so ``items[0]`` is code). Markers inside
#: inline code or a fenced code block are code and never touched.
_MARKER_RE = re.compile(r"\[(\s*\d{1,3}(?:\s*[,;]\s*\d{1,3})*\s*)\]")
_DIGIT_RE = re.compile(r"\d")
#: Fenced code blocks (an unclosed fence runs to the end) and inline code.
_CODE_RE = re.compile(r"```[\s\S]*?(?:```|\Z)|`[^`\n]*`")
#: Stands in for a removed marker until the whitespace around it is tidied.
_REMOVED = "\x00"
#: Words of a message that is only small talk (a greeting, thanks, an
#: acknowledgement). Only such a message may get an uncited reply under the
#: grounding gate; any other question needs evidence.
_SMALL_TALK_WORDS = frozenset(
    {
        "hi", "hello", "hey", "hiya", "yo", "thanks", "thank", "thx", "ty", "you", "very",
        "much", "so", "a", "lot", "ok", "okay", "k", "cool", "great", "nice", "good",
        "morning", "afternoon", "evening", "night", "bye", "goodbye", "cheers", "yes",
        "no", "sure", "please", "awesome", "perfect", "got", "it", "alright", "fine",
        "welcome", "there", "again", "all", "right", "sounds", "that", "is",
    }
)  # fmt: skip
_MAX_SMALL_TALK_WORDS = 6


def is_small_talk(message: str) -> bool:
    """Whether ``message`` is only a greeting, thanks or acknowledgement."""
    words = re.findall(r"[a-z]+", (message or "").lower())
    return 0 < len(words) <= _MAX_SMALL_TALK_WORDS and all(w in _SMALL_TALK_WORDS for w in words)


@dataclass
class GroundedAnswer:
    """The answer after citation checking.

    Attributes:
        answer: The answer to show (invalid markers removed; replaced by the
            fallback when the policy is ``strict`` and it is not grounded).
        raw_answer: The model's answer as written.
        grounded: Whether the answer is supported by cited evidence (or is
            plain conversation that needed none).
        reason: ``"cited"``, ``"conversation"``, ``"no_citations"``,
            ``"invalid_citations"``, ``"empty"`` or ``"not_checked"``.
        cited: Evidence numbers the answer cites that exist, in first-use order.
        invalid: Cited numbers that matched no evidence (removed).
    """

    answer: str
    raw_answer: str
    grounded: bool
    reason: str
    cited: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)

    def citations(self, ledger: EvidenceLedger) -> list[Evidence]:
        return [ev for n in self.cited if (ev := ledger.get(n)) is not None]


def _clean_markers(text: str, valid: set[int]) -> tuple[str, list[int], list[int]]:
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
            return ""  # the marker ended a clause or line: drop its spacing too
        return " " if (match.group(1) or match.group(2)) else ""

    cleaned = re.sub(rf"([ \t]*){_REMOVED}(?:[ \t]*{_REMOVED})*([ \t]*)", tidy, cleaned)
    return cleaned.strip(), cited, invalid


def ground_answer(
    answer: str,
    ledger: EvidenceLedger,
    *,
    tools_called: bool,
    policy: GroundingPolicy = "strict",
    ungrounded_answer: str = DEFAULT_UNGROUNDED_ANSWER,
    max_uncited_chars: int = DEFAULT_MAX_UNCITED_CHARS,
    question: str | None = None,
) -> GroundedAnswer:
    """Check ``answer``'s ``[N]`` citations against ``ledger``.

    Markers that match no evidence are removed. The answer is grounded when
    it cites at least one real piece of evidence, or when it is short plain
    conversation: no tool call, no citation and no figures. With
    ``question`` given, that exemption also needs the question itself to be
    small talk (a greeting, thanks), so a factual question answered from
    memory without a tool call is not passed as conversation. An empty answer
    is never grounded. Otherwise it is ungrounded: ``strict`` replaces it
    with ``ungrounded_answer``, ``annotate`` keeps it and only reports the
    verdict, ``off`` skips the check entirely.

    The gate checks that the answer rests on evidence, not that every
    sentence carries a citation: one real citation makes it grounded.
    """
    raw = (answer or "").strip()
    if policy == "off":
        return GroundedAnswer(answer=raw, raw_answer=raw, grounded=True, reason="not_checked")
    if not raw:
        shown = ungrounded_answer if policy == "strict" else ""
        return GroundedAnswer(shown, raw, False, "empty")

    valid = {ev.n for ev in ledger.items}
    cleaned, cited, invalid = _clean_markers(raw, valid)
    if cited:
        return GroundedAnswer(cleaned, raw, True, "cited", cited, invalid)
    if (
        not invalid
        and not tools_called
        and len(cleaned) <= max_uncited_chars
        # ``[0]`` is kept as code (``items[0]``), not a figure.
        and not _DIGIT_RE.search(cleaned.replace("[0]", ""))
        and (question is None or is_small_talk(question))
    ):
        return GroundedAnswer(cleaned, raw, True, "conversation", cited, invalid)

    reason = "invalid_citations" if invalid else "no_citations"
    shown = ungrounded_answer if policy == "strict" else cleaned
    return GroundedAnswer(shown, raw, False, reason, cited, invalid)
