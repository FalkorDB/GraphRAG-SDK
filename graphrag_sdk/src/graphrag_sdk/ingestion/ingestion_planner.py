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
# This mirrors retrieval's PathRouter (graphrag_sdk/retrieval/strategies/
# path_router.py): same "heuristic or one small LLM call, always safe-fallback"
# shape, applied to the ingestion side of the pipeline.

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

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
from graphrag_sdk.ingestion.resolution_strategies.description_merge import (
    DescriptionMergeResolution,
)
from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
    ExactMatchResolution,
)
from graphrag_sdk.ingestion.resolution_strategies.llm_verified_resolution import (
    LLMVerifiedResolution,
)
from graphrag_sdk.ingestion.resolution_strategies.semantic_resolution import (
    SemanticResolution,
)

logger = logging.getLogger(__name__)

# ── The agent-selectable options ───────────────────────────────────────────
# "callable" chunking is intentionally excluded: it wraps a user-supplied
# function, which an LLM cannot synthesize. Everything else maps 1:1 to a
# concrete strategy the builder can instantiate from just an llm/embedder.
CHUNKERS: tuple[str, ...] = ("sentence", "fixed", "structural", "contextual")
EXTRACTORS: tuple[str, ...] = ("gliner", "llm")
RESOLVERS: tuple[str, ...] = ("exact", "description_merge", "semantic", "llm_verified")

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
PARAM_SPECS: dict[str, dict[str, dict[str, tuple[str, float, float, float]]]] = {
    "chunker": {
        "sentence": {
            "max_tokens": ("int", 64, 2048, 512),
            "overlap_sentences": ("int", 0, 10, 2),
        },
        "structural": {
            "max_tokens": ("int", 64, 2048, 512),
            "overlap_sentences": ("int", 0, 10, 2),
        },
        "contextual": {
            "max_tokens": ("int", 64, 2048, 512),
            "overlap_sentences": ("int", 0, 10, 2),
        },
        "fixed": {
            "chunk_size": ("int", 100, 8000, 1000),
            "chunk_overlap": ("int", 0, 2000, 100),
        },
    },
    "extractor": {
        "gliner": {"threshold": ("float", 0.1, 0.95, 0.75)},
        "llm": {"threshold": ("float", 0.1, 0.95, 0.75)},
    },
    "resolver": {
        "exact": {},
        "description_merge": {
            "force_summary_threshold": ("int", 1, 50, 3),
            "max_summary_tokens": ("int", 50, 2000, 500),
        },
        "semantic": {
            "similarity_threshold": ("float", 0.5, 0.999, 0.95),
            "ann_top_k": ("int", 5, 200, 50),
            "force_summary_threshold": ("int", 1, 50, 3),
            "max_summary_tokens": ("int", 50, 2000, 500),
        },
        "llm_verified": {
            "hard_threshold": ("float", 0.5, 0.999, 0.95),
            "soft_threshold": ("float", 0.3, 0.98, 0.80),
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

    - fixed chunker: ``chunk_overlap`` is forced below ``chunk_size``.
    - llm_verified resolver: ``hard_threshold`` must stay above
      ``soft_threshold``; if the pair inverts, both are dropped to defaults.
    """
    spec = PARAM_SPECS.get(component, {}).get(strategy, {})
    if not spec or not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, (kind, lo, hi, _default) in spec.items():
        if key not in raw:
            continue
        try:
            val: float = int(raw[key]) if kind == "int" else float(raw[key])
        except (TypeError, ValueError):
            continue
        out[key] = max(lo, min(hi, val))
        if kind == "int":
            out[key] = int(out[key])

    # Cross-field guards.
    if strategy == "fixed" and "chunk_overlap" in out:
        size = out.get("chunk_size", 1000)
        if out["chunk_overlap"] >= size:
            out["chunk_overlap"] = max(0, int(size) - 1)
    if strategy == "llm_verified" and "hard_threshold" in out and "soft_threshold" in out:
        if out["hard_threshold"] <= out["soft_threshold"]:
            out.pop("hard_threshold", None)
            out.pop("soft_threshold", None)
    return out


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
    reason: str = ""
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
            pass

    # Fall back to / augment with key:value scraping.
    if not {"chunker", "extractor", "resolver"} & fields.keys():
        for key in ("chunker", "extractor", "resolver"):
            m = re.search(rf"\b{key}\b\s*[:=]\s*([a-z_]+)", raw, re.IGNORECASE)
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


def build_chunker(
    name: str,
    *,
    llm: Any | None = None,
    params: dict[str, Any] | None = None,
) -> SentenceTokenCapChunking | FixedSizeChunking | StructuralChunking | ContextualChunking:
    """Instantiate the chunker named by an :class:`IngestionPlan`."""
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
    if name == "description_merge":
        return DescriptionMergeResolution(llm=llm, **kw)
    if name == "semantic":
        return SemanticResolution(llm=llm, embedder=embedder, **kw)
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
