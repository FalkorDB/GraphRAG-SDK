# GraphRAG SDK — Retrieval: Cypher-First Aggregation Strategy
# Routes aggregation/quantitative questions through a deterministic graph-
# query path that treats Cypher as the answer source (not just another
# retrieval signal). Free-text properties (roles, projects) that aren't
# captured as typed entities are recovered by parsing Person chunk text at
# retrieval time.
#
# Non-aggregation questions delegate to a fallback strategy (default:
# MultiPathRetrieval). The strategy is therefore safe as the top-level
# strategy on GraphRAG — RAG questions still get the existing pipeline.
#
# Background: traditional RAG retrieves prose evidence and lets the LLM
# synthesize. For "how many X" / "which X has the most Y" / "BOTH A and B"
# questions, prose evidence is the wrong shape — the answer wants exact
# counts or set operations. Cypher gives us those deterministically, but
# only when the underlying graph has the right structure. This strategy
# wraps Cypher generation in three mechanisms that make it reliable on
# noisy, real-world graphs:
#
#   1. Multi-candidate Cypher with row-count selection (M2): K parallel
#      samples, execute all, pick the one with the most rows. Beats LLM
#      stochasticity without serial retries.
#   2. Column-named markdown table formatting (M3): uses FalkorDB's
#      ``result.header`` so the synthesizer sees ``acme_count=10`` not the
#      lossy ``10 | 7 | True``.
#   3. Description + chunk-text fuzzy hybrid (M5): for "shared X" / "BOTH
#      A and B" questions where X is a free-text property (role, project)
#      not extracted as a typed node, parse Person chunks via regex and
#      compute the set operation in Python with fuzzy token matching.
#
# Plus a deterministic numeric-math path (M6) — RETURN raw values, then
# average / sum / median in Python — and a negation-existential branch
# (M7) that treats an empty Cypher result as the definitive "No" when the
# question shape demands it.

from __future__ import annotations

import asyncio
import logging
import re
import statistics
from abc import ABC, abstractmethod
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    RawSearchResult,
    RetrieverResult,
    RetrieverResultItem,
)
from graphrag_sdk.core.providers import Embedder, LLMInterface
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy
from graphrag_sdk.retrieval.strategies.cypher_generation import (
    SCHEMA_PROMPT,
    _sanitize_cypher,
    extract_cypher,
    validate_cypher,
)
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Intent classification (M1)
# ─────────────────────────────────────────────────────────────────

_AGG_INTENT_PATTERNS = [
    r"\bhow\s+many\b", r"\bhow\s+much\b",
    r"\bwhich\s+\w+\b",
    r"\bwhat\s+(?:is\s+the\s+)?(?:average|mean|median|total|sum|count|number)\b",
    r"\baverage\b", r"\bmedian\b", r"\btotal\b",
    r"\bcount\s+of\b", r"\bnumber\s+of\b",
    r"\blist\s+(?:all|the|every)\b",
    r"\blist\s+\w+\s+(?:that|with|where|who)\b",
    r"\bare\s+there\s+any\b", r"\bis\s+there\s+any\b",
    # `more X than` / `fewer X than` — allow up to 4 words between
    r"\bmore(?:\s+\S+){1,4}\s+than\b",
    r"\bfewer(?:\s+\S+){0,4}\s+than\b",
    r"\bless(?:\s+\S+){0,4}\s+than\b",
    r"\bboth\s+\w+(?:\s+\S+)*?\s+and\s+\w+\b",
    r"\bbetween\s+\w+\s+and\s+\w+\b",
    r"\bexactly\s+\d+\b", r"\bat\s+least\s+\d+\b",
    r"\b(?:does|do|has|have|is|are)\s+(?:[A-Z]\w*\s+){1,4}(?:have|has|contain|own|run|host)\b",
    r"\bwho\s+(?:works|live|is)\s+(?:at|in|on)\b",
]
_AGG_INTENT_RE = re.compile("|".join(_AGG_INTENT_PATTERNS), re.IGNORECASE)

_NUMERIC_AGG_RE = re.compile(
    r"\b(average|mean|median|total|sum)\b\s+(?:of\s+)?(?:the\s+)?(?:\w+\s+)?"
    r"\b(year|years|age|amount|price|cost|revenue|number|count|"
    r"founding|founded|salary|salaries)\b",
    re.IGNORECASE,
)

_YES_NO_RE = re.compile(
    r"^\s*(?:is|are|was|were|do|does|did|has|have|had|can|could|will|"
    r"would|should|may|might|are\s+there|is\s+there)\b",
    re.IGNORECASE,
)

_WHICH_LIST_RE = re.compile(
    r"^\s*(?:which|what|list|name|identify|enumerate)\b",
    re.IGNORECASE,
)


def detect_aggregation_intent(question: str) -> str:
    """Return ``"numeric_math"``, ``"aggregation"``, or ``"rag"``.

    ``"numeric_math"`` triggers the Python-arithmetic path; ``"aggregation"``
    triggers the Cypher-first table path; ``"rag"`` falls back to the
    standard retrieval strategy.
    """
    if _NUMERIC_AGG_RE.search(question):
        return "numeric_math"
    if _AGG_INTENT_RE.search(question):
        return "aggregation"
    return "rag"


def is_yes_no(question: str) -> bool:
    return bool(_YES_NO_RE.match(question))


def is_which_list(question: str) -> bool:
    return bool(_WHICH_LIST_RE.match(question))


def _is_negation_existential(question: str) -> bool:
    """True for "are there any X without/no Y?" — empty Cypher = definitive No."""
    has_negation = bool(
        re.search(r"\b(?:no|not|without|never|none)\b", question, re.IGNORECASE)
    )
    has_existential = bool(
        re.search(r"\b(?:any|are\s+there|is\s+there)\b", question, re.IGNORECASE)
    )
    return has_negation and has_existential


# ─────────────────────────────────────────────────────────────────
# Free-text phrase extraction for description hybrid (M5)
# ─────────────────────────────────────────────────────────────────

# Roles end in a closed set of professional suffixes — anchoring on those
# keeps noisy phrases ("the cross-region replication initiative") from
# getting captured.
_ROLE_RE = re.compile(
    r"\b(?:as|is)\s+(?:an?\s+)?"
    r"((?:[a-z][\w\s]*?)?\b(?:engineer|scientist|manager|architect|researcher|"
    r"developer|analyst|specialist|designer|consultant)s?)\b",
    re.IGNORECASE,
)

_PROJECT_RE = re.compile(
    r"\bcontribut(?:es|ing)\s+to\s+"
    r"(?:the\s+|a\s+|an\s+)?"
    r"(.+?)"
    r"(?=\s+and\s+(?:is|active)|\s+based\s+in|,|\.\s|\.$|$)",
    re.IGNORECASE,
)

_BOTH_AB_RE = re.compile(
    r"\bboth\s+([\w\s]+?)\s+and\s+([\w\s'-]+?)(?:\?|\.|$|\bof\b|\bat\b|\bwith\b)",
    re.IGNORECASE,
)

_SAME_AS_RE = re.compile(
    r"\b(?:same\s+\w+\s+as|share\s+(?:an?\s+)?\w+\s+with|"
    r"share\s+(?:any\s+of\s+)?\w+\s+with)"
    r"\s+(?:someone\s+(?:at|in|from)\s+)?([\w\s]+?)(?:\?|\.|$)",
    re.IGNORECASE,
)


def _extract_roles(text: str) -> set[str]:
    out: set[str] = set()
    for m in _ROLE_RE.finditer(text or ""):
        role = re.sub(r"\s+", " ", m.group(1).strip().lower())
        if 3 < len(role) < 60:
            out.add(role)
    return out


def _extract_projects(text: str) -> set[str]:
    out: set[str] = set()
    for m in _PROJECT_RE.finditer(text or ""):
        proj = re.sub(r"\s+", " ", m.group(1).strip().lower())
        proj = re.sub(r"\s+(?:and|who|while)$", "", proj)
        if 5 < len(proj) < 100:
            out.add(proj)
    return out


def _extract_phrases(text: str, kind: str) -> set[str]:
    if kind == "role":
        return _extract_roles(text)
    if kind == "project":
        return _extract_projects(text)
    return set()


# ─────────────────────────────────────────────────────────────────
# Pluggable phrase extractor (R8)
# ─────────────────────────────────────────────────────────────────


class PhraseExtractor(ABC):
    """Pluggable phrase extractor for the shared-property hybrid path.

    The default implementation targets the prose patterns the SDK's
    ``GraphExtraction`` pipeline produces (``"works at X as a <role>"``,
    ``"contributes to <project>"``). Domain-specific subclasses can
    override ``extract`` to recognize medical / legal / e-commerce /
    non-English vocabularies without forking the strategy.

    Pass an instance via ``CypherFirstAggregationStrategy(..., phrase_extractor=MyExtractor())``.
    """

    @abstractmethod
    def extract(self, text: str, kind: str) -> set[str]:
        """Return phrases of ``kind`` found in ``text``.

        ``kind`` is typically ``"role"`` or ``"project"`` but extractors
        may define additional kinds. Unknown kinds should return an empty
        set rather than raise.
        """


class DefaultPhraseExtractor(PhraseExtractor):
    """The default role/project extractor used by
    ``CypherFirstAggregationStrategy``. Targets the prose patterns
    produced by the SDK's default ``GraphExtraction`` pipeline.

    See module-level ``_ROLE_RE`` and ``_PROJECT_RE`` for the exact
    regexes; the role suffix vocabulary is enumerated in the
    ``CypherFirstAggregationStrategy`` docstring.
    """

    def extract(self, text: str, kind: str) -> set[str]:
        return _extract_phrases(text, kind)


def _detect_property_kind(question: str) -> str | None:
    q = question.lower()
    if re.search(r"\b(?:role|roles|job|jobs|title|titles|position|positions)\b", q):
        return "role"
    if re.search(
        r"\b(?:project|projects|initiative|initiatives|work\s+on|works\s+on|"
        r"working\s+on|contribute|contributes|same\s+thing)\b",
        q,
    ):
        return "project"
    return None


def _fuzzy_intersect(a: set[str], b: set[str]) -> set[str]:
    """Return phrases from ``a`` that fuzzy-match any phrase in ``b``.

    Fuzzy = case-insensitive substring in either direction OR ≥2 shared
    content tokens. Catches cases where extraction paraphrased a project
    name, e.g. "next-generation pipeline rewrite" vs "pipeline rewrite".
    """
    stop = {"the", "a", "an", "to", "of", "for", "in", "on", "and", "or"}

    def _tokens(s: str) -> set[str]:
        return {t.lower() for t in re.findall(r"\w+", s)
                if t.lower() not in stop and len(t) > 2}

    out: set[str] = set()
    for x in a:
        xt = _tokens(x)
        if not xt:
            continue
        for y in b:
            if x == y or x in y or y in x:
                out.add(x)
                break
            yt = _tokens(y)
            if len(xt & yt) >= 2:
                out.add(x)
                break
    return out


# ─────────────────────────────────────────────────────────────────
# Cypher result → markdown table (M3)
# ─────────────────────────────────────────────────────────────────

def format_result_as_markdown_table(
    result: Any,
    *,
    cap: int = 100,
) -> tuple[str, list[dict[str, Any]], bool]:
    """Render a FalkorDB result_set as a markdown table with column headers.

    Returns ``(table_md, parsed_rows, truncated)``. ``parsed_rows`` is a list
    of dicts keyed by column name so callers can post-process structured
    data without re-parsing the markdown.
    """
    if not getattr(result, "result_set", None):
        return "(empty result)", [], False

    headers: list[str] = []
    if getattr(result, "header", None):
        for h in result.header:
            if isinstance(h, (list, tuple)) and len(h) >= 2:
                headers.append(str(h[1]))
            else:
                headers.append(str(h))
    if not headers:
        n_cols = len(result.result_set[0]) if result.result_set else 0
        headers = [f"col_{i}" for i in range(n_cols)]

    rows = result.result_set[:cap]
    truncated = len(result.result_set) > cap

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        d: dict[str, Any] = {}
        for i, val in enumerate(row):
            d[headers[i] if i < len(headers) else f"col_{i}"] = val
        parsed_rows.append(d)

    sep = " | "
    lines = [sep.join(headers), sep.join(["---"] * len(headers))]
    for row in rows:
        cells: list[str] = []
        for v in row:
            if v is None:
                cells.append("(null)")
            elif isinstance(v, list):
                cells.append(", ".join(str(x) for x in v))
            else:
                cells.append(str(v))
        lines.append(sep.join(cells))
    if truncated:
        lines.append(
            f"... (showing {len(rows)} of {len(result.result_set)} rows; "
            "result truncated)"
        )
    return "\n".join(lines), parsed_rows, truncated


# ─────────────────────────────────────────────────────────────────
# Schema prompt enrichment for aggregation generation
# ─────────────────────────────────────────────────────────────────

_AGG_SCHEMA_SUFFIX = """

## Additional rules for AGGREGATION questions

- For "BOTH X AND Y" / set-intersection over a shared property (e.g. "roles
  held at both A and B"), use TWO separate matches against DIFFERENT
  people, joined on the shared entity:
    MATCH (p1:Person)-[:RELATES]-(o1:Organization),
          (p1)-[:RELATES]-(prop:__Entity__),
          (p2:Person)-[:RELATES]-(o2:Organization),
          (p2)-[:RELATES]-(prop)
    WHERE o1.name CONTAINS '...' AND o2.name CONTAINS '...'
    RETURN DISTINCT prop.name AS shared_value
  The shared `prop` variable IS the intersection.
- For "more X than Y" comparison questions, RETURN both counts AS named
  columns (e.g. RETURN count(...) AS acme_count, count(...) AS initech_count)
  so the answer is unambiguous.
- For "average / total of YEARS or NUMBERS", just RETURN the raw values
  (e.g. d.name for Date entities). Arithmetic happens outside cypher.
- For "which X have/work/share Y" list questions, RETURN DISTINCT only the
  X you're asked about — do not add extra columns.
- Always alias every RETURN column with a descriptive name.
- Prefer RETURNing one row per group in group-by patterns rather than
  packing multiple counts into a single row.
"""

_DESC_HINT_SUFFIX = (
    "\n\nIMPORTANT: Person entities have rich free-text in their "
    "`description` property — phrases like 'works at X as a senior "
    "engineer' or 'contributes to internal tooling for observability'. "
    "If the question asks about ROLES, JOBS, PROJECTS, or other free-text "
    "properties that are NOT first-class entities in the schema, prefer "
    "filtering on `p.description CONTAINS '...'` over matching typed "
    "structural edges.\n"
)


# ─────────────────────────────────────────────────────────────────
# Aggregation-mode answer-side directive
# (synthesized into the retrieved item content; the existing system
# prompt rule 8 in api/main.py already tells the LLM to trust the
# "Authoritative Graph Query Results" section.)
# ─────────────────────────────────────────────────────────────────

_AUTH_HEADING = (
    "## Authoritative Graph Query Results "
    "(deterministic; trust over passages on counts and aggregates)"
)


def _wrap_authoritative(body: str, *, source_note: str = "") -> str:
    note = (
        f"\nSource: {source_note}." if source_note else
        "\nSource: text-to-Cypher run against the knowledge graph."
    )
    return f"{_AUTH_HEADING}{note}\n\n{body}"


# Canonical labels for the sub-path metadata key ``cypher_first_path``.
# Operators / metrics dashboards can group on these to see which path
# fires for each query.
PATH_NUMERIC_MATH = "numeric_math"
PATH_SHARED_PROPERTY_HYBRID = "shared_property_hybrid"
PATH_CYPHER_TABLE = "cypher_table"
PATH_NEGATION_EMPTY_NO = "negation_empty_no"
PATH_RAG_FALLBACK = "rag_fallback"
PATH_RAG_FALLBACK_NUMERIC_FAIL = "rag_fallback_numeric_fail"
PATH_RAG_FALLBACK_CYPHER_EMPTY = "rag_fallback_cypher_empty"


def _tag_path(result: RawSearchResult, path: str) -> RawSearchResult:
    """Attach the ``cypher_first_path`` label to a strategy result.

    Used both for results we construct ourselves and for results returned
    from the delegated RAG fallback strategy — operators get a uniform
    signal regardless of which branch handled the query.
    """
    meta = dict(result.metadata or {})
    meta.setdefault("strategy", "cypher_first")
    meta["cypher_first_path"] = path
    return RawSearchResult(records=result.records, metadata=meta)


# ─────────────────────────────────────────────────────────────────
# Sub-paths
#
# Each path is a small, focused class with a single ``maybe_handle()``
# method that either produces a final ``RawSearchResult`` or returns
# ``None`` to defer to the next path. The strategy's ``_execute()``
# dispatches by intent and iterates the relevant paths in order.
# Splitting this way makes the routing trivial to follow, each path
# trivially unit-testable in isolation, and adding new shapes (medical /
# legal / e-commerce) a matter of dropping in a new path class.
# ─────────────────────────────────────────────────────────────────


class _AggregationPath(ABC):
    """Base class for CypherFirstAggregationStrategy sub-paths.

    Holds a reference to the parent strategy so subclasses can reach the
    shared LLM / graph / fallback / k_candidates state without dragging
    around a long argument list.
    """

    def __init__(self, strategy: CypherFirstAggregationStrategy) -> None:
        self._s = strategy

    @abstractmethod
    async def maybe_handle(
        self,
        query: str,
        ctx: Context,
    ) -> RawSearchResult | None:
        """Return a final retrieval result, or ``None`` to defer."""


class _RagDelegationPath(_AggregationPath):
    """Hands the query to the RAG fallback verbatim. Used for intent="rag"."""

    async def maybe_handle(
        self,
        query: str,
        ctx: Context,
    ) -> RawSearchResult | None:
        return _tag_path(
            await self._s._fallback._execute(query, ctx),
            PATH_RAG_FALLBACK,
        )


class _NumericMathPath(_AggregationPath):
    """For "average / total / sum of YEARS / NUMBERS" — extract values via
    Cypher, do the arithmetic in Python. Avoids LLM-arithmetic errors."""

    async def maybe_handle(
        self,
        query: str,
        ctx: Context,
    ) -> RawSearchResult | None:
        extraction_prompt = SCHEMA_PROMPT.format(
            question=(
                f"Generate a cypher that returns the RAW NUMERIC VALUES "
                f"needed to answer this question (one value per row). "
                f"Do NOT compute averages or sums in cypher; just return "
                f"the raw numbers. Use Date entities if the question is "
                f"about years.\n\n"
                f"Question: {query}"
            )
        )
        cypher: str | None = None
        values: list[float] = []
        try:
            resp = await self._s._llm.ainvoke(extraction_prompt)
            cypher = extract_cypher(resp.content)
            errors = validate_cypher(cypher) if cypher else ["empty"]
            if errors:
                logger.debug("Numeric-math cypher validation failed: %s", errors)
                cypher = None
            else:
                cypher = _sanitize_cypher(cypher)
                result = await self._s._graph.query_raw(cypher)
                for row in (result.result_set or []):
                    for cell in row:
                        v = _coerce_number(cell)
                        if v is not None:
                            values.append(v)
        except Exception as exc:
            logger.debug("Numeric-math extraction failed: %s", exc)

        if not values:
            # Fall back to standard retrieval — the LLM may still be able
            # to extract the numbers from chunks.
            ctx.log("CypherFirst numeric_math: no values extracted, "
                    "falling back to RAG")
            return _tag_path(
                await self._s._fallback._execute(query, ctx),
                PATH_RAG_FALLBACK_NUMERIC_FAIL,
            )

        q_lower = query.lower()
        if "median" in q_lower:
            ans = statistics.median(values)
            op = "median"
        elif "total" in q_lower or "sum" in q_lower:
            ans = float(sum(values))
            op = "sum"
        else:  # average / mean / default for "average-like" questions
            ans = sum(values) / len(values)
            op = "average"

        ans_str = f"{int(ans)}" if ans == int(ans) else f"{ans:.1f}"
        body = (
            f"computed {op} = {ans_str}\n"
            f"source_values ({len(values)} rows): "
            f"{', '.join(str(int(v)) if v == int(v) else f'{v:.2f}' for v in values)}\n"
            f"cypher: {cypher}"
        )
        return RawSearchResult(
            records=[{
                "section": "cypher_results",
                "content": _wrap_authoritative(
                    body,
                    source_note="numeric extraction + Python arithmetic",
                ),
            }],
            metadata={
                "strategy": "cypher_first",
                "cypher_first_path": PATH_NUMERIC_MATH,
                "op": op,
                "value": ans,
                "n_values": len(values),
                "cypher": cypher,
            },
        )


class _SharedPropertyHybridPath(_AggregationPath):
    """For "BOTH A and B" / "same X as Z" questions over free-text
    properties (role, project) that aren't first-class entities. Parses
    Person chunks via regex and computes the set operation in Python
    with fuzzy token matching."""

    async def maybe_handle(
        self,
        query: str,
        ctx: Context,
    ) -> RawSearchResult | None:
        kind = _detect_property_kind(query)
        if kind is None:
            return None
        shape1 = _BOTH_AB_RE.search(query)
        shape2 = _SAME_AS_RE.search(query)
        if not (shape1 or shape2):
            return None

        batch_cypher = (
            "MATCH (o:Organization)<-[:RELATES]-(p:Person) "
            "OPTIONAL MATCH (p)-[:MENTIONED_IN]->(c:Chunk) "
            "RETURN o.name AS org, p.name AS person, "
            "  p.description AS desc, collect(DISTINCT c.text) AS chunks"
        )
        try:
            batch_res = await self._s._graph.query_raw(batch_cypher)
        except Exception as exc:
            logger.debug("Shared-property hybrid batch query failed: %s", exc)
            return None

        # Topology check: if the batched ``(Org)<-[:RELATES]-(Person)`` query
        # returns zero tuples, the graph doesn't match the assumptions M5
        # was tuned on (Person ↔ Organization edges + MENTIONED_IN chunks).
        # Surface this loudly once per call so operators using custom
        # extractors get a fast signal rather than silent wrong answers.
        if not (batch_res.result_set or []):
            logger.warning(
                "CypherFirst shared-property hybrid found zero "
                "(Organization)<-[:RELATES]-(Person) tuples; falling through. "
                "If your graph uses different edge shapes or doesn't extract "
                "Person/Organization labels, this hybrid will never fire — "
                "see the strategy docstring's 'Assumptions and known limits' "
                "section."
            )
            return None

        org_phrase_map: dict[str, set[str]] = {}
        for row in (batch_res.result_set or []):
            org = row[0] or ""
            person = row[1] or ""
            desc = row[2] or ""
            chunks = row[3] or []
            text_blob = desc + "\n" + "\n".join(chunks)
            phrases = org_phrase_map.setdefault(org, set())
            for sent in re.split(r"(?<=[.\n])\s+", text_blob):
                # Sentence-restrict to this person to avoid cross-paragraph
                # contamination from chunks that contain multiple people.
                if person and person.split()[0] not in sent:
                    continue
                phrases |= self._s._phrase_extractor.extract(sent, kind)

        def _gather(org_name: str) -> set[str]:
            out: set[str] = set()
            for name, phrases in org_phrase_map.items():
                if org_name.lower() in (name or "").lower():
                    out |= phrases
            return out

        if shape1:
            org_a, org_b = (shape1.group(1).strip(), shape1.group(2).strip())
            if not org_a or not org_b:
                return None
            a_phrases = _gather(org_a)
            b_phrases = _gather(org_b)
            common = sorted(_fuzzy_intersect(a_phrases, b_phrases))
            if not common:
                return None
            ctx.log(f"CypherFirst hybrid shape1: {len(common)} shared {kind}s")
            body = (
                f"The {kind}s held by employees at both {org_a} and "
                f"{org_b}: " + ", ".join(common)
            )
            return RawSearchResult(
                records=[{
                    "section": "cypher_results",
                    "content": _wrap_authoritative(
                        body,
                        source_note=(
                            "Person chunks + description regex; "
                            "fuzzy-intersected by content tokens"
                        ),
                    ),
                }],
                metadata={
                    "strategy": "cypher_first",
                    "cypher_first_path": PATH_SHARED_PROPERTY_HYBRID,
                    "shape": "both_a_and_b",
                    "kind": kind,
                    "common": common,
                },
            )

        # shape2
        target = shape2.group(1).strip().rstrip(",")
        target_phrases = _gather(target)
        if not target_phrases:
            return None
        sharing: list[str] = []
        for org_name, org_phrases in org_phrase_map.items():
            if not org_name:
                continue
            if (
                target.lower() in org_name.lower()
                or org_name.lower() in target.lower()
            ):
                continue
            if _fuzzy_intersect(org_phrases, target_phrases):
                sharing.append(org_name)
        sharing.sort()
        if not sharing:
            return None
        ctx.log(f"CypherFirst hybrid shape2: {len(sharing)} sharing orgs")
        body = (
            "The organizations that have at least one employee working on "
            f"the same {kind} as someone at {target}: " + ", ".join(sharing)
        )
        return RawSearchResult(
            records=[{
                "section": "cypher_results",
                "content": _wrap_authoritative(
                    body,
                    source_note=(
                        "Person chunks + description regex; fuzzy-matched "
                        "across orgs"
                    ),
                ),
            }],
            metadata={
                "strategy": "cypher_first",
                "cypher_first_path": PATH_SHARED_PROPERTY_HYBRID,
                "shape": "same_as",
                "kind": kind,
                "sharing": sharing,
            },
        )


class _MultiCandidateCypherPath(_AggregationPath):
    """Generates K parallel cypher candidates, executes them all, renders
    the highest-row-count result as a markdown table with column headers.

    Also handles the empty-result branches:
        - negation-existential ("are there any X without Y?") → return No
        - everything else → delegate to the RAG fallback
    Always returns a result (never ``None``) — this path is the last
    line of defense for aggregation intent.
    """

    async def maybe_handle(
        self,
        query: str,
        ctx: Context,
    ) -> RawSearchResult | None:
        # Pass 1: structural.
        candidates = await self._generate_k_candidates(query)
        cypher, table_md, parsed, truncated = await self._execute_and_pick(
            candidates,
        )

        # Pass 2 (description hint): if pass 1 was sparse for a "which X"
        # or "shared X" question, try again with the description hint
        # enabled. Cheap because cypher-gen runs in parallel.
        expects_many = is_which_list(query) or re.search(
            r"\bboth\b|\bshared\b|\bsame\b|\bcommon\b|\bin\s+common\b",
            query, re.IGNORECASE,
        )
        if expects_many and (parsed is None or len(parsed) < 3):
            more = await self._generate_k_candidates(query, with_desc_hint=True)
            combined = list({*(candidates or []), *(more or [])})
            cypher2, table_md2, parsed2, truncated2 = await self._execute_and_pick(combined)
            if parsed2 and len(parsed2) > len(parsed or []):
                cypher, table_md, parsed, truncated = (
                    cypher2, table_md2, parsed2, truncated2,
                )

        rows = len(parsed) if parsed else 0
        ctx.log(f"CypherFirst cypher_table: {rows} rows from "
                f"{len(candidates)} candidates")

        if cypher and parsed:
            directive = ""
            if is_which_list(query):
                directive = (
                    "\nNOTE: This is a 'which / list' question. Enumerate "
                    "EVERY DISTINCT VALUE from the first column in your "
                    "answer — do not summarize, truncate, or pick a "
                    "subset unless the question explicitly asked for the "
                    "top/most/fewest one."
                )
            body = table_md + directive
            return RawSearchResult(
                records=[{
                    "section": "cypher_results",
                    "content": _wrap_authoritative(body),
                }],
                metadata={
                    "strategy": "cypher_first",
                    "cypher_first_path": PATH_CYPHER_TABLE,
                    "cypher": cypher,
                    "cypher_rows": rows,
                    "cypher_truncated": truncated,
                },
            )

        # Cypher returned 0 / no candidate succeeded.
        if is_yes_no(query) and _is_negation_existential(query):
            ctx.log("CypherFirst empty-result branch: negation-existential = No")
            return RawSearchResult(
                records=[{
                    "section": "cypher_results",
                    "content": _wrap_authoritative(
                        "No matching items: the cypher query returned 0 "
                        "rows. For a negation-existential question of "
                        "this shape, that means no such items exist.",
                        source_note="Cypher returned 0 rows (definitive)",
                    ),
                }],
                metadata={
                    "strategy": "cypher_first",
                    "cypher_first_path": PATH_NEGATION_EMPTY_NO,
                    "cypher": cypher,
                },
            )

        # Vector fallback for everything else.
        ctx.log("CypherFirst cypher empty — falling back to RAG")
        return _tag_path(
            await self._s._fallback._execute(query, ctx),
            PATH_RAG_FALLBACK_CYPHER_EMPTY,
        )

    async def _generate_k_candidates(
        self,
        query: str,
        *,
        with_desc_hint: bool = False,
    ) -> list[str]:
        prompt = SCHEMA_PROMPT.format(question=query) + _AGG_SCHEMA_SUFFIX
        if with_desc_hint:
            prompt += _DESC_HINT_SUFFIX

        async def _one() -> str | None:
            try:
                resp = await self._s._llm.ainvoke(prompt)
                cypher = extract_cypher(resp.content)
                if not cypher:
                    return None
                errors = validate_cypher(cypher)
                if errors:
                    return None
                return _sanitize_cypher(cypher)
            except Exception as exc:
                logger.debug("Candidate generation failed: %s", exc)
                return None

        results = await asyncio.gather(*[_one() for _ in range(self._s._k)])
        # Dedupe while preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for c in results:
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    async def _execute_and_pick(
        self,
        candidates: list[str],
    ) -> tuple[str | None, str, list[dict[str, Any]], bool]:
        """Run all candidates in parallel; pick the one with most rows.

        Returns ``(cypher, table_md, parsed_rows, truncated)``.
        """
        if not candidates:
            return None, "(no candidate cypher)", [], False
        results = await asyncio.gather(
            *[self._s._graph.query_raw(c) for c in candidates],
            return_exceptions=True,
        )
        scored: list[tuple[int, int, str, Any]] = []
        for cypher, res in zip(candidates, results):
            if isinstance(res, BaseException):
                continue
            rows = len(res.result_set) if res.result_set else 0
            cols = len(res.result_set[0]) if (res.result_set and res.result_set[0]) else 0
            scored.append((rows, cols, cypher, res))
        if not scored:
            return None, "(no candidate executed successfully)", [], False
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        _, _, best_cypher, best_result = scored[0]
        table_md, parsed, truncated = format_result_as_markdown_table(best_result)
        return best_cypher, table_md, parsed, truncated


# ─────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────

class CypherFirstAggregationStrategy(RetrievalStrategy):
    """Aggregation-aware retrieval strategy.

    Routes each question by detected intent:

    - ``"numeric_math"`` — RETURN raw values via Cypher, then compute the
      ``average``/``sum``/``median`` in Python. Avoids LLM-arithmetic errors.
    - ``"aggregation"`` — multi-candidate Cypher, pick the highest-row-count
      result, render as a markdown table with column headers. Optionally
      run a description+chunk-text fuzzy hybrid for "shared X" / "BOTH A
      and B" shapes before falling back to vector retrieval on empty.
    - ``"rag"`` — delegate to ``rag_fallback`` (default: ``MultiPathRetrieval``).

    Safe as the top-level strategy on ``GraphRAG``: non-aggregation
    questions get the existing pipeline unchanged.

    Every returned :class:`RawSearchResult` carries a ``cypher_first_path``
    metadata key whose value is one of ``PATH_*`` module constants — useful
    for operator dashboards that want to see which sub-path fired.

    Assumptions and known limits
    ----------------------------
    The shared-property hybrid (M5) was tuned on graphs produced by the
    SDK's default ``GraphExtraction`` pipeline. It makes the following
    assumptions; when they're violated, the hybrid silently returns
    ``None`` and the strategy falls back to the multi-candidate Cypher
    path (which still works, just without free-text recovery):

    - **Graph topology.** Organizations are connected to Persons via
      ``[:RELATES]``, and Persons are connected to ``Chunk`` nodes via
      ``[:MENTIONED_IN]``. This is the canonical shape the SDK builds.
      Custom extractors that use different edge types or skip chunk
      provenance will not benefit from M5 — the strategy logs a warning
      and continues.
    - **Prose shape.** Role and project values are extracted by regex
      from Person descriptions and chunk text. The regexes target the
      phrasing patterns ``"works at X as a <role>"`` and ``"contributes
      to <project>"``. Domains whose prose departs from these patterns
      (medical / legal / e-commerce / non-English) will not match — the
      result is empty role/project sets, not wrong answers, but the
      hybrid won't help.
    - **Role vocabulary.** The role extractor anchors on the suffixes
      ``engineer | scientist | manager | architect | researcher |
      developer | analyst | specialist | designer | consultant``. Other
      job titles ("director", "VP", "lead") won't match.

    Accuracy ceiling
    ----------------
    The strategy faithfully returns what is in the graph. Duplicate
    entities ("Wayne En" vs "Wayne Enterprises"), chunk-boundary-truncated
    names, or non-deduplicated short-form references ("Carla" vs "Carla
    Okafor") all flow through into the answer. Cypher counts will be
    inflated; "which X" lists will contain duplicates. These are
    extraction-quality issues — not strategy bugs — and should be
    addressed in the ingestion pipeline (resolver, coref, dedup).

    Args:
        graph_store: Required for Cypher execution.
        vector_store: Required for the RAG fallback.
        embedder: Required for the RAG fallback.
        llm: Required for Cypher generation + synthesis.
        k_candidates: Number of parallel Cypher samples per aggregation
            question. Default 3 — enough to surface alternate structural
            interpretations without burning latency.
        rag_fallback: Strategy used for non-aggregation intent. If
            ``None``, a fresh ``MultiPathRetrieval`` is constructed
            internally.

    Example::

        strategy = CypherFirstAggregationStrategy(
            graph_store=rag._graph_store,
            vector_store=rag._vector_store,
            embedder=embedder,
            llm=llm,
        )
        async with GraphRAG(
            connection=conn, llm=llm, embedder=embedder,
            embedding_dimension=256,
            retrieval_strategy=strategy,
        ) as rag:
            ...
    """

    def __init__(
        self,
        graph_store: Any,
        vector_store: Any,
        embedder: Embedder,
        llm: LLMInterface,
        *,
        k_candidates: int = 3,
        rag_fallback: RetrievalStrategy | None = None,
        phrase_extractor: PhraseExtractor | None = None,
    ) -> None:
        super().__init__(graph_store=graph_store, vector_store=vector_store)
        self._embedder = embedder
        self._llm = llm
        self._k = max(1, k_candidates)
        self._fallback = rag_fallback or MultiPathRetrieval(
            graph_store=graph_store,
            vector_store=vector_store,
            embedder=embedder,
            llm=llm,
        )
        # Pluggable phrase extractor for the shared-property hybrid path.
        # Override with a domain-specific subclass to recognize roles /
        # projects beyond the default English-prose vocabulary.
        self._phrase_extractor = phrase_extractor or DefaultPhraseExtractor()
        # Sub-paths — each handles one shape of question. The order in
        # which they're consulted is encoded in ``_execute`` below; the
        # paths themselves don't know about each other.
        self._rag_path = _RagDelegationPath(self)
        self._numeric_path = _NumericMathPath(self)
        self._hybrid_path = _SharedPropertyHybridPath(self)
        self._cypher_table_path = _MultiCandidateCypherPath(self)

    # -- Template Method hook -------------------------------------

    async def _execute(
        self,
        query: str,
        ctx: Context,
        **kwargs: Any,
    ) -> RawSearchResult:
        intent = detect_aggregation_intent(query)
        ctx.log(f"CypherFirst intent={intent}")

        if intent == "rag":
            # Non-aggregation questions don't benefit from any of the
            # cypher-first mechanics — hand straight to the fallback.
            return await self._rag_path.maybe_handle(query, ctx)

        if intent == "numeric_math":
            return await self._numeric_path.maybe_handle(query, ctx)

        # intent == "aggregation":
        # Try the description+chunk hybrid for "shared X" shapes first —
        # when it fires it generally produces better answers than the
        # multi-candidate cypher because chunks preserve original corpus
        # phrasing that extraction may have summarized away.
        hybrid_result = await self._hybrid_path.maybe_handle(query, ctx)
        if hybrid_result is not None:
            return hybrid_result

        # The multi-candidate cypher path always returns a result; it
        # internally handles its own empty / negation / fallback branches.
        return await self._cypher_table_path.maybe_handle(query, ctx)

    # NOTE: the per-path logic (numeric math, shared-property hybrid,
    # multi-candidate cypher, etc.) lives in the ``_AggregationPath``
    # subclasses above this strategy. Keeping each path in its own class
    # keeps the dispatch above readable and makes it trivial to swap one
    # implementation out (e.g., a medical-prose phrase extractor) without
    # touching the strategy itself.

    # -- Custom _format ------------------------------------------

    def _format(self, raw: RawSearchResult) -> RetrieverResult:
        """Render section records as markdown content items, preserving the
        cypher metadata so callers / metrics can see which mode fired."""
        items: list[RetrieverResultItem] = []
        for record in raw.records:
            content = record.get("content", "") if isinstance(record, dict) else str(record)
            section = record.get("section", "") if isinstance(record, dict) else ""
            if content:
                items.append(
                    RetrieverResultItem(
                        content=content,
                        metadata={"section": section},
                    )
                )
        return RetrieverResult(items=items, metadata=raw.metadata)



def _coerce_number(cell: Any) -> float | None:
    """Extract a single numeric value from a cypher result cell.

    Accepts ints/floats directly; for strings, pulls the first integer or
    float substring (catches "1995" inside "1995 is the year ...").
    """
    if cell is None:
        return None
    if isinstance(cell, (int, float)):
        return float(cell)
    m = re.search(r"-?\d+(?:\.\d+)?", str(cell))
    return float(m.group()) if m else None
