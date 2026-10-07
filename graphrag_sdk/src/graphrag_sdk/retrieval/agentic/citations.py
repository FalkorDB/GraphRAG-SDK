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

#: An uncited answer this short, given without any tool call, is treated as
#: conversation ("Hello!", "You're welcome") rather than an unsupported claim.
DEFAULT_MAX_UNCITED_CHARS = 280

_MARKER_RE = re.compile(r"\[(\s*\d+(?:\s*[,;]\s*\d+)*\s*)\]")


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
            ``"invalid_citations"`` or ``"not_checked"``.
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
        keep: list[int] = []
        for part in re.split(r"[,;]", match.group(1)):
            n = int(part.strip())
            if n in valid:
                keep.append(n)
                if n not in cited:
                    cited.append(n)
            elif n not in invalid:
                invalid.append(n)
        return f"[{', '.join(str(n) for n in keep)}]" if keep else ""

    cleaned = _MARKER_RE.sub(replace, text)
    # Removing a marker can leave "word ." or doubled spaces behind.
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, cited, invalid


def ground_answer(
    answer: str,
    ledger: EvidenceLedger,
    *,
    tools_called: bool,
    policy: GroundingPolicy = "strict",
    ungrounded_answer: str = DEFAULT_UNGROUNDED_ANSWER,
    max_uncited_chars: int = DEFAULT_MAX_UNCITED_CHARS,
) -> GroundedAnswer:
    """Check ``answer``'s ``[N]`` citations against ``ledger``.

    Markers that match no evidence are removed. The answer is grounded when
    it cites at least one real piece of evidence, or when it is short plain
    conversation given without any tool call or citation. Otherwise it is
    ungrounded: ``strict`` replaces it with ``ungrounded_answer``,
    ``annotate`` keeps it and only reports the verdict, ``off`` skips the
    check entirely.
    """
    raw = (answer or "").strip()
    if policy == "off":
        return GroundedAnswer(answer=raw, raw_answer=raw, grounded=True, reason="not_checked")

    valid = {ev.n for ev in ledger.items}
    cleaned, cited, invalid = _clean_markers(raw, valid)
    if cited:
        return GroundedAnswer(cleaned, raw, True, "cited", cited, invalid)
    if not invalid and not tools_called and len(cleaned) <= max_uncited_chars:
        return GroundedAnswer(cleaned, raw, True, "conversation", cited, invalid)

    reason = "invalid_citations" if invalid else "no_citations"
    shown = ungrounded_answer if policy == "strict" else cleaned
    return GroundedAnswer(shown, raw, False, reason, cited, invalid)
