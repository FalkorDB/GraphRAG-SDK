# GraphRAG SDK — Tools: numbered evidence and citation checking
# Every result a tool returns that an answer may rely on gets a number [n]
# in the session's evidence ledger. The caller's agent cites those numbers;
# check_citations verifies them against the ledger.

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

#: Evidence kept per session. Older items are dropped first; a citation of a
#: dropped number reports it as invalid rather than matching something else.
DEFAULT_LEDGER_CAPACITY = 2000

#: One cited number or range (``3``, ``1-3``, ``1–3``).
_CITE = r"\d{1,6}(?:\s*[-–]\s*\d{1,6})?"
#: A citation marker: ``[3]``, ``[1, 4]``, ``[1-3]``, also glued to a word or a
#: figure (``plant[1]``, ``1998[2]``) or chained (``[1][2]``). Not for ``[0]``
#: (evidence is numbered from 1, so ``items[0]`` is code), nor when a number of
#: four or more digits is not in the ledger (``[2024]`` is a year). Markers
#: inside inline code or a fenced code block are code and never touched.
_MARKER_RE = re.compile(rf"\[(\s*{_CITE}(?:\s*[,;]\s*{_CITE})*\s*)\]")
#: Longest range read as citations; ``[1-200]`` is a page span, not 200 cites.
_MAX_RANGE = 50
#: Fenced code blocks (an unclosed fence runs to the end) and inline code.
_CODE_RE = re.compile(r"```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)|`[^`\n]*`")
#: Stands in for a removed marker until the whitespace around it is tidied.
_REMOVED = "\x00"

#: Sentence ends: whitespace after ``.``, ``!`` or ``?`` (``2.8`` stays whole).
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
#: A full stop that ends an abbreviation (``Inc.``, ``Mr.``, ``e.g.``,
#: ``U.S.``), not a sentence.
_ABBREVIATION_END_RE = re.compile(
    r"(?:\b(?:Inc|Corp|Co|Ltd|LLC|Dr|Mr|Mrs|Ms|Prof|St|Jr|Sr|Mt)"
    r"|(?i:\be\.g|\bi\.e|\bvs)|(?:\b[A-Z]\.)+[A-Z])\.$"
)
#: Markers that open a split piece belong to the sentence before (``x. [1]``).
_LEADING_MARKERS_RE = re.compile(rf"^(?:\s*\[\s*{_CITE}(?:\s*[,;]\s*{_CITE})*\s*\])+")
#: A full stop with markers glued on (``it.[2] Next``): split after the markers.
_GLUED_END_RE = re.compile(rf"([.!?])((?:\[\s*{_CITE}(?:\s*[,;]\s*{_CITE})*\s*\])+)\s+")
#: Sentence openers that are capitalised without being part of a name.
_OPENERS = frozenset(
    """The A An This That These Those It Its Our Their His Her Both Each All Some
    After Before When While In On At For From With By As If So But And Then
    I We You He She They My Your Me Us Him Them There Here Who What Which Where
    Why How Whose Whom Is Are Was Were Be Been Do Does Did Has Have Had Can Could
    Will Would Should Shall May Might Must Not No Yes None Neither Nor Either Or
    Any Every Such Same Other Another Most Many Much More Less Few Several One
    Only Also Just Even Still Yet Already Now Today Currently Since Because
    Although Though However Unlike Like During Under Over Between Among Without
    Within Through Per About Around Above Below Into Onto Upon Across Along
    Against Toward Towards Beyond Despite Except Including Regarding Following
    Given Due According Based Overall Additionally Furthermore Moreover
    Meanwhile Therefore Thus Hence Instead Otherwise Finally First Second Third
    Last Later Earlier Recently Initially Originally Previously Subsequently
    Eventually Ultimately Specifically Notably Importantly Similarly Likewise
    Conversely Alternatively Unfortunately Fortunately Interestingly Apparently
    Clearly Generally Typically Usually Often Sometimes Never Always Approximately
    Nearly Almost Roughly Please Note See Answer Summary Sources Source Evidence
    Graph Total Unknown Whether Once Until Unless Whereas Where Nothing Everything
    Something Anything Nobody Everyone Someone Anyone To Of Up Out Off Very
    Two Three Four Five Six Seven Eight Nine Ten""".split()
)
#: A list bullet, numbered item, quote or heading prefix on a line.
_LINE_PREFIX_RE = re.compile(r"^\s*(?:[-*+>]|\d+[.)]|#{1,6})\s+")
#: A figure in a claim: ``20``, ``2.8``, ``1,200``, the digits of ``INC-0314``.
_FIGURE_RE = re.compile(r"\d+(?:[.,]\d+)*")
#: A word; a capitalised one after the first word of a sentence is a name.
_WORD_RE = re.compile(r"[^\W\d_][\w'’-]*")
#: Sentences a check reports, at most, per kind.
_MAX_FLAGGED = 10
#: What the evidence is called when a sentence says it lacks something.
_SOURCE = (
    r"(?:the|these|this|those|my|our|any)\s+(?-i:[a-z]+\s+)?"
    r"(?:knowledge graph|graph|evidence|sources?|context|data(?:set)?|documents?"
    r"|records?|passages?|results?|facts?|search(?:es)?|lookups?|retrieval)"
)
_SAY = (
    r"(?:say|mention|state|include|contain|cover|specify|give|list|show|answer|tell"
    r"|indicate|describe|provide|name|identify|explain|confirm|reveal|have|hold"
    r"|find|locate|determine|verify)"
)
#: A sentence that says the evidence lacks something ("The graph does not say
#: who founded Globex."). Saying so is honest, so it needs no citation. Only
#: sentences about the evidence or a failed lookup count: a negative claim
#: about the world ("Globex does not provide pensions.") still needs one.
_GAP_RE = re.compile(
    # The graph / the sources ... do not say, have no ..., none of them say.
    rf"\b{_SOURCE}\s+(?-i:[a-z]+\s+)?(?:(?:does|do|did|is|are|was|were|has|have|had|can"
    rf"|ca|could)(?: not|n['’]t|not)\s+(?:be\s+)?{_SAY}"
    r"|(?:has|have|had|holds?|contains?|gives?|provides?|returned|returns?|shows?)\s+no\b)"
    rf"|\b(?:none|neither) of {_SOURCE}\s+{_SAY}"
    # Not found / listed / mentioned in the graph.
    rf"|\bnot\s+(?:\w+\s+){{0,2}}?(?:in|from|within|part of|among|by)\s+{_SOURCE}\b"
    rf"|\bnothing\s+(?:in|from)\s+{_SOURCE}\b"
    # A failed lookup: I could not find, could not be determined.
    r"|\b(?:I|we)\s+(?:could|can|ca|did|do)(?: not|n['’]t|not)\s+"
    r"(?:find|locate|confirm|determine|verify|tell|see)\b"
    r"|\b(?:I|we)\s+(?:found|find) (?:no|nothing)\b"
    r"|\b(?:could|can|ca)(?: not|n['’]t|not) be (?:found|determined|confirmed|verified)\b"
    # No information about ... was found; not enough evidence.
    r"|(?:^|[.!?]\s+)(?:there (?:is|are|was|were) )?no (?:information|evidence|mention"
    r"|data|details?|records?|sources?|passages?|facts?|results?)\b"
    r"|\bnot enough (?:information|evidence)"
    # Who founded it is not stated.
    r"|\bnot\s+(?:explicitly\s+)?(?:mentioned|stated|specified)\s*[.!]?\s*$",
    re.I | re.M,
)
#: Trailing zeros of a decimal figure (``2.80``, ``40.0``), dropped before
#: figures are compared.
_TRAILING_ZEROS_RE = re.compile(r"(?<![\d.])(\d+)\.(\d*?)0+(?![\d.]?\d)")


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
        self._common: tuple[int, frozenset[str]] = (0, frozenset())

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
    def next_number(self) -> int:
        """The number the next item will get."""
        return self._next

    def discard(self, numbers: Iterable[int]) -> None:
        """Forget items, e.g. ones cut from a result the agent never saw."""
        for n in numbers:
            if self._items.pop(n, None) is not None:
                self._common = (0, frozenset())

    def common_words(self) -> frozenset[str]:
        """Words the evidence writes in lower case: ordinary words, not names."""
        if self._common[0] != self._next:  # numbers only grow: a new one means new items
            words = frozenset(
                w for ev in self._items.values() for w in re.findall(r"(?<!\w)[a-z]+", ev.content)
            )
            self._common = (self._next, words)
        return self._common[1]

    @property
    def items(self) -> list[Evidence]:
        return list(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, n: object) -> bool:
        return n in self._items


@dataclass
class UnsupportedClaim:
    """A cited sentence naming figures or names its cited evidence lacks.

    Attributes:
        sentence: The sentence as written, with its markers.
        missing: The figures and names found in no evidence it cites.
    """

    sentence: str
    missing: list[str] = field(default_factory=list)


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
        uncited: Sentences stating a figure or a name with no ``[n]`` of their
            own. Checked only on a grounded answer; does not change ``grounded``.
        unsupported: Cited sentences whose figures or names appear in none of
            the evidence they cite. Does not change ``grounded`` either.
    """

    answer: str
    grounded: bool
    reason: str
    cited: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)
    uncited: list[str] = field(default_factory=list)
    unsupported: list[UnsupportedClaim] = field(default_factory=list)

    def citations(self, ledger: EvidenceLedger) -> list[Evidence]:
        return [ev for n in self.cited if (ev := ledger.get(n)) is not None]


def _marker_numbers(body: str, valid: set[int] | EvidenceLedger | None = None) -> list[int] | None:
    """The numbers a marker cites, ranges expanded; ``None`` when it is not a
    citation (``[0]``, a backwards range, one longer than ``_MAX_RANGE``, or a
    number of four or more digits ``valid`` never numbered, such as a year)."""
    numbers: list[int] = []
    for part in re.split(r"[,;]", body):
        ends = [int(x) for x in re.split(r"[-–]", part)]
        if len(ends) == 2 and not 0 < ends[1] - ends[0] < _MAX_RANGE:
            return None
        numbers.extend(range(ends[0], ends[-1] + 1))
    if _names_a_year(body, valid):
        return None
    return None if numbers == [0] else numbers


def _names_a_year(body: str, valid: set[int] | EvidenceLedger | None) -> bool:
    """Whether a marker holds a number of four or more digits that ``valid``
    never numbered (``[2024]``). A number the ledger gave out and has since
    dropped is still a citation, of evidence no longer held."""
    for x in re.findall(r"\d+", body):
        n = int(x)
        if n < 1000:
            continue
        if isinstance(valid, EvidenceLedger):
            if n >= valid._next:
                return True
        elif valid is None or n not in valid:
            return True
    return False


def _clean_markers(text: str, valid: set[int] | EvidenceLedger) -> tuple[str, list[int], list[int]]:
    cited: dict[int, None] = {}
    invalid: dict[int, None] = {}

    def replace(match: re.Match[str]) -> str:
        numbers = _marker_numbers(match.group(1), valid)
        if numbers is None:
            return match.group(0)
        keep: list[int] = []
        for n in numbers:
            if n in valid:
                keep.append(n)
                cited[n] = None
            else:
                invalid[n] = None
        if keep == numbers and re.search(r"[-–]", match.group(1)):
            return match.group(0)  # a range that is all valid stays as written
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

    if _REMOVED in cleaned:
        # Anchored at the start of a whitespace run, so a long run of spaces
        # is scanned once rather than once per space.
        cleaned = re.sub(
            rf"(?<![ \t])([ \t]*){_REMOVED}(?:[ \t]*{_REMOVED})*([ \t]*)", tidy, cleaned
        )
    return cleaned.strip(), list(cited), list(invalid)


def states_a_gap(text: str) -> bool:
    """Whether ``text`` says the evidence lacks something ("not found")."""
    return _GAP_RE.search(text) is not None


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
        uncited, unsupported = _check_claims(cleaned, ledger)
        return GroundedAnswer(cleaned, True, "cited", cited, invalid, uncited, unsupported)
    reason = "invalid_citations" if invalid else "no_citations"
    return GroundedAnswer(cleaned, False, reason, cited, invalid)


def _sentences(text: str) -> list[str]:
    """Split prose into sentences, one line (bullet, paragraph) at a time.

    A marker written after the full stop (``x. [1]``) stays with its sentence.
    Code is blanked out first: it is neither a claim nor a citation.
    """
    out: list[str] = []
    for line in _CODE_RE.sub(" ", text).splitlines():
        line = _LINE_PREFIX_RE.sub("", line).replace("**", "").replace("__", "").strip()
        line = _GLUED_END_RE.sub(r"\1 \2 ", line)
        pieces: list[str] = []
        for piece in _merge_abbreviations(_SENTENCE_END_RE.split(line)):
            lead = _LEADING_MARKERS_RE.match(piece)
            if lead and pieces:
                pieces[-1] = f"{pieces[-1]} {lead.group(0).strip()}"
                piece = piece[lead.end() :]
            if piece.strip():
                pieces.append(piece.strip())
        out.extend(pieces)
    return out


def _merge_abbreviations(pieces: list[str]) -> list[str]:
    """Join a piece that ends with an abbreviation to the piece after it."""
    out: list[str] = []
    for piece in pieces:
        # ``Inc. [1] Next`` ends a sentence: the marker closes the piece before.
        if out and _ABBREVIATION_END_RE.search(out[-1]) and not _LEADING_MARKERS_RE.match(piece):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def _claim_terms(sentence: str, common: frozenset[str] = frozenset()) -> list[str]:
    """The figures and names a sentence states, in order, without repeats.

    A name is a run of capitalised words ("Corvane Systems"). A sentence's
    capitalised first word counts too when the sentence names something
    else ("Globex acquired Initech") or starts a run ("Corvane Systems
    ships"), unless it is an opener such as "The" or "However"; a word
    ``common`` holds (written in lower case elsewhere) only counts as the
    start of a run. Alone in its sentence ("Penalty: 2% a week", "Version
    2.8 ships") it is taken as an ordinary word.
    """
    terms: list[str] = []
    for figure in _FIGURE_RE.findall(sentence):
        if figure not in terms:
            terms.append(figure)
    words = [re.sub(r"['’]s$", "", w).strip("'’-") for w in _WORD_RE.findall(sentence)]
    run: list[str] = []
    for i, word in enumerate(words):
        capital = word[:1].isupper()
        if i == 0 and capital:
            follows = len(words) > 1 and words[1][:1].isupper()
            names_more = any(w[:1].isupper() and len(w) >= 3 for w in words[2:])
            ordinary = word.lower() in common
            capital = word not in _OPENERS and (follows or (names_more and not ordinary))
        if capital:
            run.append(word)
            continue
        if run and " ".join(run) not in terms and any(len(w) >= 3 for w in run):
            terms.append(" ".join(run))
        run = []
    if run and " ".join(run) not in terms and any(len(w) >= 3 for w in run):
        terms.append(" ".join(run))
    return terms


def _drop_trailing_zeros(text: str) -> str:
    """``2.80`` -> ``2.8`` and ``40.0`` -> ``40``, so equal figures compare equal."""
    return _TRAILING_ZEROS_RE.sub(lambda m: m[1] + (f".{m[2]}" if m[2] else ""), text)


def _in_evidence(term: str, text: str) -> bool:
    """A figure must appear as a number (``2.8`` matches ``2.80``). A name
    must have every word of three or more letters in the text as a whole
    word ("Corvane Systems" needs both; "Smith" alone still needs "smith").
    Case is ignored; ``text`` is already lower-case, with figures
    normalised by :func:`_drop_trailing_zeros`."""
    if term[0].isdigit():
        figure = re.escape(_drop_trailing_zeros(term.replace(",", "")))
        return re.search(rf"(?<![\d.]){figure}(?!\d|[.,]\d)", text) is not None
    words = [w.lower() for w in term.split() if len(w) >= 3]
    return all(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", text) is not None for w in words)


def _check_claims(answer: str, ledger: EvidenceLedger) -> tuple[list[str], list[UnsupportedClaim]]:
    """Find claims an answer does not back with what it cites.

    A claim is a sentence of three or more words stating a figure or a name;
    questions and lead-ins ending in ``:`` are not claims. A claim with no
    marker of its own is uncited. A cited claim whose figures or names are in
    none of the evidence it cites is unsupported. The check is literal, with
    no LLM: a figure written as a word ("two") counts as missing. A sentence
    that says the evidence lacks something ("The graph does not say who
    founded Globex.") needs no marker. Figures and names are checked against
    the cited evidence's content, not its source path.
    """
    uncited: list[str] = []
    unsupported: list[UnsupportedClaim] = []
    # Words written in lower case in the evidence or the answer are ordinary
    # words, not names, when they open a sentence.
    common = ledger.common_words() | frozenset(re.findall(r"(?<!\w)[a-z]+", answer))
    for sentence in _sentences(answer):
        if sentence.endswith(("?", ":")):
            continue
        cites: dict[int, None] = {}
        for marker in dict.fromkeys(_MARKER_RE.findall(sentence)):
            cites.update(dict.fromkeys(_marker_numbers(marker, ledger) or []))
        numbers = list(cites)
        # A bracketed year stays in the claim as a figure; every other marker
        # (a citation, ``[0]``, a page span ``[1-200]``) is blanked.
        bare = _MARKER_RE.sub(
            lambda m: m.group(0) if _names_a_year(m.group(1), ledger) else " ", sentence
        )
        bare = " ".join(bare.split())
        terms = _claim_terms(bare, common)
        if not terms or len(bare.split()) < 3:
            continue
        if not numbers:
            if len(uncited) < _MAX_FLAGGED and not _GAP_RE.search(bare):
                uncited.append(sentence)
            continue
        cited_text = " ".join(
            ev.content for n in numbers if (ev := ledger.get(n)) is not None
        ).lower()
        cited_text = _drop_trailing_zeros(re.sub(r"(?<=\d),(?=\d{3}\b)", "", cited_text))
        missing = [t for t in terms if not _in_evidence(t, cited_text)]
        if missing and len(unsupported) < _MAX_FLAGGED:
            unsupported.append(UnsupportedClaim(sentence, missing))
    return uncited, unsupported
