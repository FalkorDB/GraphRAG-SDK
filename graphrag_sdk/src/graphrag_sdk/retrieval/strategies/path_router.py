# GraphRAG SDK — Retrieval: Path Router (agentic path selection)
#
# A lightweight planner that decides *which* of MultiPathRetrieval's
# retrieval paths to run for a given question, instead of always running
# all of them. One cheap LLM call (or a heuristic) returns a subset; the
# strategy skips the rest. Falls back to "all paths" whenever the plan is
# empty or the router errors, so recall is never silently lost.

from __future__ import annotations

import logging
import re
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LatencyBudgetExceededError

logger = logging.getLogger(__name__)

# The five gateable retrieval paths of MultiPathRetrieval.
RETRIEVAL_PATHS: tuple[str, ...] = (
    "relates",  # RELATES edge vector search (facts + seed entities)
    "entity_cypher",  # entity discovery via Cypher exact/CONTAINS name match
    "entity_fulltext",  # entity discovery via fulltext index
    "expansion",  # 1-hop/2-hop relationship expansion
    "chunks",  # chunk retrieval (fulltext + vector + MENTIONED_IN + 2-hop)
)
_PATH_SET: frozenset[str] = frozenset(RETRIEVAL_PATHS)

_PATH_GUIDE = (
    "- relates: semantic/fact questions, paraphrases, 'what is', 'tell me about'.\n"
    "- entity_cypher: the question names a specific entity / proper noun exactly.\n"
    "- entity_fulltext: fuzzy or partial entity names, typos, multi-word names.\n"
    "- expansion: 'how are X and Y connected', relationships, neighbors, paths.\n"
    "- chunks: needs supporting passages / quotes / detailed source text."
)


def all_paths() -> set[str]:
    """Return the full path set (the default, unrouted behavior)."""
    return set(RETRIEVAL_PATHS)


def parse_paths(text: str) -> set[str]:
    """Parse a model/heuristic response into a validated set of path ids.

    Accepts comma/space/newline separated tokens; ignores anything that is
    not a known path. Returns an empty set when nothing valid is found (the
    caller is expected to fall back to ``all_paths()``).
    """
    tokens = re.split(r"[\s,]+", (text or "").strip().lower())
    return {t for t in tokens if t in _PATH_SET}


class HeuristicPathRouter:
    """Zero-cost router: picks paths from cheap question features.

    Useful when you want path pruning without an extra LLM call. Always
    keeps at least ``relates`` + ``chunks`` so open-ended questions still
    get semantic recall.
    """

    async def plan(self, query: str, ctx: Context | None = None) -> set[str]:
        q = (query or "").lower()
        plan: set[str] = {"relates", "chunks"}
        # Relationship / connection questions → traversal.
        rel_pattern = (
            r"\b(connect|related|relationship|between|link|neighbor"
            r"|path|how (?:are|is|do))\b"
        )
        if re.search(rel_pattern, q):
            plan |= {"expansion", "entity_cypher"}
        # Quoted or capitalized proper nouns in the original query → name match.
        if '"' in query or "'" in query or re.search(r"\b[A-Z][a-z]+\b", query or ""):
            plan |= {"entity_cypher", "entity_fulltext"}
        return plan & _PATH_SET or all_paths()


class LLMPathRouter:
    """LLM-backed router: one small call selects the paths to run.

    Args:
        llm: provider exposing ``ainvoke(prompt, timeout=...)``.
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def plan(self, query: str, ctx: Context | None = None) -> set[str]:
        ctx = ctx or Context()
        prompt = (
            "You are a retrieval planner for a knowledge-graph RAG system. "
            "Given a question, choose ONLY the retrieval paths needed to answer "
            "it well — usually 1 to 3, not all. Return a comma-separated list of "
            "path ids from this set, nothing else.\n\n"
            f"Paths:\n{_PATH_GUIDE}\n\n"
            f"Question: {query}\n\nPaths:"
        )
        try:
            ctx.ensure_budget("path router LLM call")
            response = await self._llm.ainvoke(
                prompt,
                timeout=ctx.provider_timeout_seconds("path router LLM call"),
            )
            plan = parse_paths(getattr(response, "content", "") or "")
            if plan:
                ctx.log(f"PathRouter: selected {sorted(plan)}")
                return plan
            logger.debug("PathRouter: empty/unparseable plan, using all paths")
        except LatencyBudgetExceededError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("PathRouter LLM call failed (%s); using all paths", exc)
        return all_paths()
