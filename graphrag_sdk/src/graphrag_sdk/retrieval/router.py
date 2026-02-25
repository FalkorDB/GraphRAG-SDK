# GraphRAG SDK 2.0 — Retrieval: Semantic Router
# Classifies query intent and selects the appropriate retrieval strategy.
# Origin: User design — absent in Neo4j (which forces a single retriever at init).
#
# v1: Optional — users can also pass strategy= explicitly to GraphRAG.query().
# v2+: Full implementation with trained classifier or LLM-based routing.

from __future__ import annotations

import logging
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy

logger = logging.getLogger(__name__)


class SemanticRouter:
    """Route queries to the best retrieval strategy based on intent.

    In v1, this is a simple rule-based router. Users register strategies
    with keywords or conditions, and the router picks the best match.

    Args:
        default_strategy: Fallback strategy when no rules match.
        strategies: Dict of strategy_name → (strategy, condition_fn) pairs.

    Example::

        router = SemanticRouter(default_strategy=local_retrieval)
        router.register("multi_hop", multi_hop_strategy, lambda q: "how" in q.lower())

        # Router picks strategy based on query
        result = await router.route("How does X relate to Y?", ctx)
    """

    def __init__(
        self,
        default_strategy: RetrievalStrategy,
        strategies: dict[str, tuple[RetrievalStrategy, Any]] | None = None,
    ) -> None:
        self._default = default_strategy
        self._strategies: dict[str, tuple[RetrievalStrategy, Any]] = strategies or {}

    def register(
        self,
        name: str,
        strategy: RetrievalStrategy,
        condition: Any,
    ) -> None:
        """Register a strategy with a routing condition.

        Args:
            name: Strategy identifier.
            strategy: The retrieval strategy instance.
            condition: A callable(query: str) -> bool that determines
                       when this strategy should be selected.
        """
        self._strategies[name] = (strategy, condition)

    async def route(
        self,
        query: str,
        ctx: Context | None = None,
    ) -> Any:
        """Route a query to the appropriate strategy and execute it.

        Args:
            query: User's search query.
            ctx: Execution context.

        Returns:
            RetrieverResult from the selected strategy.
        """
        if ctx is None:
            ctx = Context()

        selected_name, selected_strategy = self._select(query)
        ctx.log(f"Router selected strategy: {selected_name}")

        return await selected_strategy.search(query, ctx)

    def _select(self, query: str) -> tuple[str, RetrievalStrategy]:
        """Select the best strategy for a query.

        Iterates through registered strategies and returns the first
        whose condition matches. Falls back to default.
        """
        for name, (strategy, condition) in self._strategies.items():
            try:
                if callable(condition) and condition(query):
                    return name, strategy
            except Exception:
                continue

        return "default", self._default
