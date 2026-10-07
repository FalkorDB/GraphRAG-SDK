# GraphRAG SDK — Ingestion: Strategy Planner (agentic strategy selection)
#
# A lightweight planner that decides *which* ingestion strategies to use for a
# given document — the chunker, the entity-extraction backend, and the entity
# resolver — instead of always using the fixed defaults. One cheap LLM call (or
# a heuristic) inspects a sample of the document and returns a plan; the
# pipeline builds the chosen strategies. Falls back to the defaults whenever the
# plan is empty/invalid or the planner errors, so ingestion behavior is never
# silently broken.
#
# This is analogous to retrieval-side routing: a small planner (heuristic or
# LLM) chooses ingestion strategies per document, with safe defaults on
# failure — applied here to the ingestion side of the pipeline.

from __future__ import annotations

import dataclasses
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LatencyBudgetExceededError
from graphrag_sdk.ingestion.chunking_strategies.contextual_chunking import (
    ContextualChunking,
)
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
from graphrag_sdk.ingestion.chunking_strategies.sentence_token_cap import (
    SentenceTokenCapChunking,
)
from graphrag_sdk.ingestion.chunking_strategies.structural_chunking import (
    StructuralChunking,
)
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
    GLiNERExtractor,
    LLMExtractor,
)
from graphrag_sdk.ingestion.extraction_strategies.graph_extraction import (
    GraphExtraction,
)
from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy
from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
    ExactMatchResolution,
)
from graphrag_sdk.ingestion.resolution_strategies.llm_verified_resolution import (
    LLMVerifiedResolution,
)

logger = logging.getLogger(__name__)

# ── The agent-selectable options ───────────────────────────────────────────
# "callable" chunking is intentionally excluded: it wraps a user-supplied
# function, which an LLM cannot synthesize. Everything else maps 1:1 to a
# concrete strategy the builder can instantiate from just an llm/embedder.
CHUNKERS: tuple[str, ...] = ("sentence", "fixed", "structural", "contextual")
EXTRACTORS: tuple[str, ...] = ("gliner", "llm")
# Matches the resolvers on main: SemanticResolution and DescriptionMergeResolution
# were removed there (see CHANGELOG), so the planner offers only these two.
RESOLVERS: tuple[str, ...] = ("exact", "llm_verified")

_CHUNKER_SET = frozenset(CHUNKERS)
_EXTRACTOR_SET = frozenset(EXTRACTORS)
_RESOLVER_SET = frozenset(RESOLVERS)

# Defaults — kept in lock-step with GraphRAG.ingest()'s defaults so that a plan
# that omits a field (or the whole planner falling back) reproduces today's
# behavior exactly.
DEFAULT_CHUNKER = "sentence"
DEFAULT_EXTRACTOR = "gliner"
DEFAULT_RESOLVER = "exact"

# ── Tunable parameters per strategy ─────────────────────────────────────────
# The agent may also pick the *values* inside the chosen strategy, not just the
# strategy id. Each entry is (kind, lo, hi, default); values are coerced and
# clamped to [lo, hi] before use, so the model can never produce an unsafe
# config. A param the model omits is simply left to the constructor default.
# The fourth value is the constructor's own default; the cross-field guards in
# clamp_params() compare a lone key against it, so it must match the class.
#
# GLiNER's threshold is deliberately absent: GLiNER thresholds are
# model-specific (see GLiNERExtractor — 0.75 suits gliner_medium-v2.1 but a
# bi-encoder returns almost nothing at it), and the planner does not know
# which model is loaded, so it must not pick one.
PARAM_SPECS: dict[str, dict[str, dict[str, tuple[str, float, float, float]]]] = {
    "chunker": {
        "sentence": {
            "max_tokens": ("int", 64, 2048, 384),
            "overlap_sentences": ("int", 0, 10, 2),
        },
        "structural": {
            "max_tokens": ("int", 64, 2048, 384),
            "overlap_sentences": ("int", 0, 10, 2),
        },
        "contextual": {
            "max_tokens": ("int", 64, 2048, 384),
            "overlap_sentences": ("int", 0, 10, 2),
        },
        "fixed": {
            "chunk_size": ("int", 100, 8000, 1000),
            "chunk_overlap": ("int", 0, 2000, 100),
        },
    },
    "extractor": {
        "gliner": {},
        "llm": {"threshold": ("float", 0.1, 0.95, 0.75)},
    },
    "resolver": {
        "exact": {},
        "llm_verified": {
            "hard_threshold": ("float", 0.5, 0.999, 0.95),
            "soft_threshold": ("float", 0.3, 0.98, 0.65),
            "ann_top_k": ("int", 5, 200, 50),
            "max_llm_pairs": ("int", 10, 5000, 500),
        },
    },
}


def clamp_params(component: str, strategy: str, raw: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce + clamp a raw param dict against ``PARAM_SPECS``.

    Only keys defined for ``(component, strategy)`` survive; each is coerced to
    its declared kind and clamped to ``[lo, hi]``. Unparseable values are
    dropped (the constructor default then applies). Cross-field constraints are
    enforced so the resulting kwargs can never make a constructor raise:

    - fixed chunker: ``chunk_overlap`` is forced below ``chunk_size``; a lone
      small ``chunk_size`` also gets an overlap that fits (the constructor's
      default overlap of 100 would otherwise make it raise).
    - llm_verified resolver: ``hard_threshold`` must stay above
      ``soft_threshold``; if the pair inverts (a lone key is checked against
      the other's default), both are dropped to defaults.
    """
    spec = PARAM_SPECS.get(component, {}).get(strategy, {})
    if not spec or not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, (kind, lo, hi, _default) in spec.items():
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, bool):
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(num):  # inf / nan: unusable, keep the default
            continue
        num = max(lo, min(hi, num))
        out[key] = int(num) if kind == "int" else num

    # Cross-field guards.
    if strategy == "fixed" and ("chunk_overlap" in out or "chunk_size" in out):
        size = int(out.get("chunk_size", spec["chunk_size"][3]))
        if "chunk_overlap" in out:
            if out["chunk_overlap"] >= size:
                out["chunk_overlap"] = max(0, size - 1)
        elif spec["chunk_overlap"][3] >= size:
            out["chunk_overlap"] = size // 10
    if strategy == "llm_verified" and ("hard_threshold" in out or "soft_threshold" in out):
        # A lone key is compared against the other's constructor default, so a
        # planner-picked hard_threshold=0.6 (vs default soft 0.65) can't make
        # the constructor raise and discard the whole plan.
        hard = out.get("hard_threshold", spec["hard_threshold"][3])
        soft = out.get("soft_threshold", spec["soft_threshold"][3])
        if hard <= soft:
            out.pop("hard_threshold", None)
            out.pop("soft_threshold", None)
    return out


# One description per option; the prompt lists only the options a planner is
# allowed to pick (see LLMIngestionPlanner's ``chunkers=`` / ``extractors=`` /
# ``resolvers=``), so the model is never offered something it may not choose.
_OPTION_TEXT: dict[str, dict[str, str]] = {
    "chunker": {
        "sentence": "sentence-aware, token-capped. Safe default for prose.",
        "fixed": "fixed-size character windows. Use for uniform/unstructured text.",
        "structural": "respect document structure (headings/lists/sections).\n"
        "    Use for Markdown / HTML / clearly structured documents.",
        "contextual": "sentence chunks + an LLM-written context prefix per chunk.\n"
        "    Best recall on dense/technical docs, but costs extra LLM calls.",
    },
    "extractor": {
        "gliner": "fast local NER model. Cheap default.",
        "llm": "LLM-based NER. Better on niche/ambiguous entities, costs more.",
    },
    "resolver": {
        "exact": "merge by exact normalized name. Cheap default.",
        "llm_verified": "exact match first, then embedding candidates that an\n"
        "    LLM confirms (catches paraphrased names). Costs extra LLM calls.",
    },
}
_COMPONENT_TEXT = {
    "chunker": "chunker — how the document is split:",
    "extractor": "extractor — how entities are found inside each chunk:",
    "resolver": "resolver — how duplicate entities are merged:",
}
_PARAM_TEXT: dict[str, str] = {
    "sentence": "max_tokens (64-2048, def 384), overlap_sentences (0-10, def 2).\n"
    "    Smaller chunks for dense facts; larger for narrative.",
    "fixed": "chunk_size (100-8000, def 1000), chunk_overlap (0-2000, def 100).",
    "llm": "threshold (0.1-0.95, def 0.75). Lower = more recall/noise;\n"
    "    higher = more precision.",
    "llm_verified": "hard_threshold (0.5-0.999, def 0.95), soft_threshold\n"
    "    (0.3-0.98, def 0.65, must stay below hard), ann_top_k (5-200, def 50),\n"
    "    max_llm_pairs (10-{max_llm_pairs}, def {default_pairs}).",
}


def _guide(
    chunkers: tuple[str, ...] = CHUNKERS,
    extractors: tuple[str, ...] = EXTRACTORS,
    resolvers: tuple[str, ...] = RESOLVERS,
    *,
    max_llm_pairs: int = 500,
) -> str:
    """Render the option guide for the allowed options only."""
    lines: list[str] = []
    for component, allowed in (
        ("chunker", chunkers),
        ("extractor", extractors),
        ("resolver", resolvers),
    ):
        lines.append(_COMPONENT_TEXT[component])
        for name in allowed:
            lines.append(f"  - {name}: {_OPTION_TEXT[component][name]}")
    tunable: list[str] = []
    chunker_params = [c for c in chunkers if c in ("sentence", "structural", "contextual")]
    if chunker_params:
        tunable.append(f"  - {'/'.join(chunker_params)}: {_PARAM_TEXT['sentence']}")
    if "fixed" in chunkers:
        tunable.append(f"  - fixed: {_PARAM_TEXT['fixed']}")
    if "llm" in extractors:
        tunable.append(f"  - llm extractor: {_PARAM_TEXT['llm']}")
    if "llm_verified" in resolvers:
        default_pairs = min(500, max_llm_pairs)
        tunable.append(
            "  - llm_verified: "
            + _PARAM_TEXT["llm_verified"].format(
                max_llm_pairs=max_llm_pairs, default_pairs=default_pairs
            )
        )
    if tunable:
        lines.append("")
        lines.append(
            "You MAY also tune parameters inside each chosen strategy (omit to keep the\n"
            "safe default; out-of-range values are clamped):"
        )
        lines.extend(tunable)
    return "\n".join(lines) + "\n"


_GUIDE = _guide()


@dataclass(frozen=True)
class IngestionPlan:
    """A validated decision about which ingestion strategies to use.

    Each field is one of the option ids above. ``reason`` is a short,
    free-text rationale (from the LLM, or a heuristic tag) kept only for
    logging/observability.
    """

    chunker: str = DEFAULT_CHUNKER
    extractor: str = DEFAULT_EXTRACTOR
    resolver: str = DEFAULT_RESOLVER
    # Free text for observability only: two plans that build the same
    # strategies are equal whatever their reasons say.
    reason: str = field(default="", compare=False)
    chunker_params: dict[str, Any] = field(default_factory=dict)
    extractor_params: dict[str, Any] = field(default_factory=dict)
    resolver_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.chunker not in _CHUNKER_SET:
            raise ValueError(f"unknown chunker {self.chunker!r}")
        if self.extractor not in _EXTRACTOR_SET:
            raise ValueError(f"unknown extractor {self.extractor!r}")
        if self.resolver not in _RESOLVER_SET:
            raise ValueError(f"unknown resolver {self.resolver!r}")
        # Clamp any supplied params to safe ranges so a hand- or LLM-built plan
        # can never carry an out-of-range value into a constructor.
        object.__setattr__(
            self, "chunker_params", clamp_params("chunker", self.chunker, self.chunker_params)
        )
        object.__setattr__(
            self,
            "extractor_params",
            clamp_params("extractor", self.extractor, self.extractor_params),
        )
        object.__setattr__(
            self, "resolver_params", clamp_params("resolver", self.resolver, self.resolver_params)
        )


def default_plan() -> IngestionPlan:
    """The plan that reproduces GraphRAG.ingest()'s default strategies."""
    return IngestionPlan()


def _check_allowed(component: str, allowed: Any, known: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(allowed, str):
        allowed = (allowed,)
    out = tuple(dict.fromkeys(allowed))
    if not out:
        raise ValueError(f"{component}s: allow at least one option")
    unknown = [o for o in out if o not in known]
    if unknown:
        raise ValueError(f"unknown {component} option(s) {unknown}; choose from {list(known)}")
    return out


def restrict_plan(
    plan: IngestionPlan,
    *,
    chunkers: tuple[str, ...] = CHUNKERS,
    extractors: tuple[str, ...] = EXTRACTORS,
    resolvers: tuple[str, ...] = RESOLVERS,
    max_llm_pairs: int | None = None,
) -> IngestionPlan:
    """Keep ``plan`` inside the caller's cost ceiling.

    A choice outside the allowed options is replaced with the default (or, when
    the default itself is not allowed, the first allowed option) and its params
    are dropped. ``max_llm_pairs`` caps the llm_verified resolver's pair budget.
    The document sample is part of the planner's prompt, so this is what keeps
    document text from steering ingestion into the expensive strategies.
    """

    def pick(value: str, allowed: tuple[str, ...], default: str) -> str:
        if value in allowed:
            return value
        return default if default in allowed else allowed[0]

    chunker = pick(plan.chunker, chunkers, DEFAULT_CHUNKER)
    extractor = pick(plan.extractor, extractors, DEFAULT_EXTRACTOR)
    resolver = pick(plan.resolver, resolvers, DEFAULT_RESOLVER)
    resolver_params = dict(plan.resolver_params) if resolver == plan.resolver else {}
    if resolver == "llm_verified" and max_llm_pairs is not None:
        default_pairs = int(PARAM_SPECS["resolver"]["llm_verified"]["max_llm_pairs"][3])
        pairs = int(resolver_params.get("max_llm_pairs", default_pairs))
        if pairs > max_llm_pairs:
            resolver_params["max_llm_pairs"] = int(max_llm_pairs)
    changed = (chunker, extractor, resolver) != (plan.chunker, plan.extractor, plan.resolver)
    reason = plan.reason
    if changed:
        note = "restricted to the allowed options"
        reason = f"{reason} ({note})" if reason else note
    return IngestionPlan(
        chunker=chunker,
        extractor=extractor,
        resolver=resolver,
        reason=reason,
        chunker_params=plan.chunker_params if chunker == plan.chunker else {},
        extractor_params=plan.extractor_params if extractor == plan.extractor else {},
        resolver_params=resolver_params,
    )


def parse_plan(text: str) -> IngestionPlan | None:
    """Parse a model response into a validated :class:`IngestionPlan`.

    Accepts either a JSON object (``{"chunker": ..., "extractor": ...}``) or
    loose ``key: value`` / ``key=value`` lines. Unknown or missing fields fall
    back to the corresponding default, so a partial response still yields a
    usable plan. Returns ``None`` only when nothing recognizable is found, in
    which case the caller should use :func:`default_plan`.
    """
    raw = (text or "").strip()
    if not raw:
        return None

    fields: dict[str, str] = {}
    params: dict[str, dict[str, Any]] = {}

    # First try strict JSON (possibly wrapped in ```json fences).
    fenced = re.search(r"\{.*\}", raw, re.DOTALL)
    if fenced:
        try:
            obj = json.loads(fenced.group(0))
            if isinstance(obj, dict):
                for k in ("chunker", "extractor", "resolver", "reason"):
                    v = obj.get(k)
                    if isinstance(v, str):
                        fields[k] = v.strip().lower() if k != "reason" else v.strip()
                for k in ("chunker_params", "extractor_params", "resolver_params"):
                    v = obj.get(k)
                    if isinstance(v, dict):
                        params[k] = v
        except (ValueError, TypeError):
            logger.debug("Planner output was not valid JSON; falling back to key:value parsing")

    # Fall back to / augment with key:value scraping.
    if not {"chunker", "extractor", "resolver"} & fields.keys():
        for key in ("chunker", "extractor", "resolver"):
            m = re.search(rf"\b{key}\b[\"']?\s*[:=]\s*[\"']?([a-z_]+)", raw, re.IGNORECASE)
            if m:
                fields[key] = m.group(1).strip().lower()

    chunker = fields.get("chunker", "")
    extractor = fields.get("extractor", "")
    resolver = fields.get("resolver", "")

    if not (chunker in _CHUNKER_SET or extractor in _EXTRACTOR_SET or resolver in _RESOLVER_SET):
        return None

    return IngestionPlan(
        chunker=chunker if chunker in _CHUNKER_SET else DEFAULT_CHUNKER,
        extractor=extractor if extractor in _EXTRACTOR_SET else DEFAULT_EXTRACTOR,
        resolver=resolver if resolver in _RESOLVER_SET else DEFAULT_RESOLVER,
        reason=fields.get("reason", ""),
        chunker_params=params.get("chunker_params", {}),
        extractor_params=params.get("extractor_params", {}),
        resolver_params=params.get("resolver_params", {}),
    )


def _sample(text: str | None, source: str | None, limit: int = 1500) -> str:
    """Build the document signal the planner reasons over.

    Prefers a leading slice of the actual content; in file mode (no text yet)
    falls back to the file name/extension so structural cues (``.md`` etc.)
    are still available.
    """
    if text:
        head = text[:limit]
        suffix = " …[truncated]" if len(text) > limit else ""
        return f"(source: {source or 'text'})\n{head}{suffix}"
    return f"(file: {source or 'unknown'} — content not yet loaded)"


@runtime_checkable
class IngestionPlanner(Protocol):
    """The shape a custom ``planner=`` argument must implement.

    ``HeuristicIngestionPlanner`` and ``LLMIngestionPlanner`` both satisfy
    this protocol; documents the contract for anyone plugging in their own
    planner without forcing a common base class.
    """

    async def plan(
        self,
        text: str | None,
        *,
        source: str | None = None,
        ctx: Context | None = None,
    ) -> IngestionPlan | None: ...


class HeuristicIngestionPlanner:
    """Zero-cost planner: picks strategies from cheap document features.

    Useful when you want adaptive ingestion without an extra LLM call. Keeps to
    the cheap, local options (never selects ``contextual``/``llm``/``llm_verified``
    that would add cost) — it only upgrades the chunker when the document looks
    structured.
    """

    async def plan(
        self,
        text: str | None,
        *,
        source: str | None = None,
        ctx: Context | None = None,
    ) -> IngestionPlan:
        chunker = DEFAULT_CHUNKER
        src = (source or "").lower()
        looks_structured = src.endswith((".md", ".markdown", ".html", ".htm"))
        if not looks_structured and text:
            # Markdown-ish headings or list bullets in the body.
            if re.search(r"(?m)^\s{0,3}#{1,6}\s|\n\s*[-*]\s+\S", text):
                looks_structured = True
        if looks_structured:
            chunker = "structural"
        return IngestionPlan(
            chunker=chunker,
            extractor=DEFAULT_EXTRACTOR,
            resolver=DEFAULT_RESOLVER,
            reason="heuristic",
        )


class LLMIngestionPlanner:
    """LLM-backed planner: one small call selects the ingestion strategies.

    The prompt carries a sample of the document, so document text can try to
    steer the choice. Bound what it can pick with the allow-lists below — for a
    planner that never adds LLM cost, allow only ``chunkers=("sentence",
    "fixed", "structural")``, ``extractors=("gliner",)`` and
    ``resolvers=("exact",)``.

    Args:
        llm: provider exposing ``ainvoke(prompt, timeout=...)`` and returning
            an object with a ``.content`` string (the common LLM interface
            used across the SDK).
        chunkers / extractors / resolvers: the options this planner may pick
            (default: all). A choice outside them falls back to the default
            (or the first allowed option) and is never built.
        max_llm_pairs: upper bound on the llm_verified resolver's LLM pair
            budget the plan may set (default 500, the resolver's own default).
    """

    def __init__(
        self,
        llm: Any,
        *,
        chunkers: tuple[str, ...] | list[str] = CHUNKERS,
        extractors: tuple[str, ...] | list[str] = EXTRACTORS,
        resolvers: tuple[str, ...] | list[str] = RESOLVERS,
        max_llm_pairs: int = 500,
    ) -> None:
        self._llm = llm
        self.chunkers = _check_allowed("chunker", chunkers, CHUNKERS)
        self.extractors = _check_allowed("extractor", extractors, EXTRACTORS)
        self.resolvers = _check_allowed("resolver", resolvers, RESOLVERS)
        lo, hi = PARAM_SPECS["resolver"]["llm_verified"]["max_llm_pairs"][1:3]
        if not (lo <= max_llm_pairs <= hi):
            raise ValueError(f"max_llm_pairs must be in [{int(lo)}, {int(hi)}]")
        self.max_llm_pairs = int(max_llm_pairs)

    def _restrict(self, plan: IngestionPlan) -> IngestionPlan:
        return restrict_plan(
            plan,
            chunkers=self.chunkers,
            extractors=self.extractors,
            resolvers=self.resolvers,
            max_llm_pairs=self.max_llm_pairs,
        )

    async def plan(
        self,
        text: str | None,
        *,
        source: str | None = None,
        ctx: Context | None = None,
    ) -> IngestionPlan:
        ctx = ctx or Context()
        guide = _guide(
            self.chunkers, self.extractors, self.resolvers, max_llm_pairs=self.max_llm_pairs
        )
        prompt = (
            "You are an ingestion planner for a knowledge-graph RAG system. "
            "Given a sample of a document, choose the best ingestion strategies "
            "for building a graph from it. Prefer the cheap default unless the "
            "document clearly benefits from a richer option.\n\n"
            f"Options:\n{guide}\n"
            "Respond with ONLY a JSON object of the form "
            '{"chunker": "...", "extractor": "...", "resolver": "...", '
            '"reason": "...", "chunker_params": {}, "extractor_params": {}, '
            '"resolver_params": {}} and nothing else. Include a *_params object '
            "only for parameters you want to change from the default.\n\n"
            f"Document sample:\n{_sample(text, source)}\n\nJSON:"
        )
        try:
            ctx.ensure_budget("ingestion planner LLM call")
            response = await self._llm.ainvoke(
                prompt,
                timeout=ctx.provider_timeout_seconds("ingestion planner LLM call"),
            )
            plan = parse_plan(getattr(response, "content", "") or "")
            if plan is not None:
                plan = self._restrict(plan)
                ctx.log(
                    f"IngestionPlanner: chunker={plan.chunker} "
                    f"extractor={plan.extractor} resolver={plan.resolver}"
                )
                return plan
            logger.warning("IngestionPlanner: reply was not a plan; using the default strategies")
            reason = "default: the planner's reply was not a plan"
        except LatencyBudgetExceededError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "IngestionPlanner LLM call failed (%s); using the default strategies", exc
            )
            reason = f"default: the planner call failed ({type(exc).__name__})"
        return self._restrict(dataclasses.replace(default_plan(), reason=reason))


def build_chunker(
    name: str,
    *,
    llm: Any | None = None,
    params: dict[str, Any] | None = None,
) -> SentenceTokenCapChunking | FixedSizeChunking | StructuralChunking | ContextualChunking:
    """Instantiate the chunker named by an :class:`IngestionPlan`."""
    # Re-clamping here is intentional/idempotent, not redundant: `params` may
    # come straight from a caller-built IngestionPlan (already clamped in
    # __post_init__) or from a hand-assembled dict passed directly to this
    # builder function, which has not gone through that path.
    kw = clamp_params("chunker", name, params)
    if name == "fixed":
        return FixedSizeChunking(**kw)
    if name == "structural":
        return StructuralChunking(**kw)
    if name == "contextual":
        if llm is None:
            raise ValueError("contextual chunker requires an llm")
        return ContextualChunking(llm=llm, **kw)
    return SentenceTokenCapChunking(**kw)


def build_extractor(
    name: str,
    *,
    llm: Any,
    entity_types: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> ExtractionStrategy:
    """Instantiate a GraphExtraction with the named entity-extraction backend."""
    kw = clamp_params("extractor", name, params)
    entity_extractor = LLMExtractor(llm=llm, **kw) if name == "llm" else GLiNERExtractor(**kw)
    return GraphExtraction(
        llm=llm,
        entity_extractor=entity_extractor,
        entity_types=entity_types,
    )


def build_resolver(
    name: str,
    *,
    llm: Any | None = None,
    embedder: Any | None = None,
    params: dict[str, Any] | None = None,
) -> ResolutionStrategy:
    """Instantiate the resolver named by an :class:`IngestionPlan`."""
    kw = clamp_params("resolver", name, params)
    if name == "llm_verified":
        return LLMVerifiedResolution(llm=llm, embedder=embedder, **kw)
    return ExactMatchResolution()


def build_ingestion_strategies(
    plan: IngestionPlan,
    *,
    llm: Any,
    embedder: Any | None = None,
    entity_types: list[str] | None = None,
) -> tuple[Any, ExtractionStrategy, ResolutionStrategy]:
    """Build concrete ``(chunker, extractor, resolver)`` from a plan.

    The planner only ever decides *which* strategy; this factory turns those
    ids into instances wired with the caller's ``llm``/``embedder``. Kept
    separate from the planner so the decision stays pure and testable.
    """
    return (
        build_chunker(plan.chunker, llm=llm, params=plan.chunker_params),
        build_extractor(
            plan.extractor, llm=llm, entity_types=entity_types, params=plan.extractor_params
        ),
        build_resolver(plan.resolver, llm=llm, embedder=embedder, params=plan.resolver_params),
    )
