# GraphRAG SDK 2.0 — Retrieval: Strategy ABC (Template Method)
# Pattern: Template Method — base handles telemetry/validation/formatting,
#          subclasses implement only ``_execute()``.
# Origin: Neo4j Retriever.search() → get_search_results() pattern, upgraded
#         with Context + telemetry + explicit validation.

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import RetrieverError
from graphrag_sdk.core.models import RawSearchResult, RetrieverResult, RetrieverResultItem

logger = logging.getLogger(__name__)


class RetrievalStrategy(ABC):
    """Abstract base class for all retrieval strategies.

    Uses the **Template Method** pattern:
    - ``search()`` is the public API — validates, traces, delegates, formats
    - ``_execute()`` is the abstract hook subclasses implement

    This ensures every retrieval strategy gets:
    - Input validation
    - Latency tracking
    - Consistent error handling
    - Result formatting

    ...without each strategy reimplementing it.

    Args:
        graph_store: Graph data access object (from ``storage/``).
        vector_store: Vector data access object (from ``storage/``).

    Example::

        class MultiHopRetrieval(RetrievalStrategy):
            async def _execute(self, query, ctx):
                # Your retrieval logic here
                return RawSearchResult(records=[...])
    """

    def __init__(
        self,
        graph_store: Any | None = None,
        vector_store: Any | None = None,
    ) -> None:
        self._graph = graph_store
        self._vector = vector_store

    async def search(
        self,
        query: str,
        ctx: Context | None = None,
        **kwargs: Any,
    ) -> RetrieverResult:
        """Execute a retrieval search.

        This is the public entry point. Do NOT override — implement
        ``_execute()`` instead.

        Args:
            query: User's search query.
            ctx: Execution context.
            **kwargs: Strategy-specific parameters.

        Returns:
            RetrieverResult with formatted items.
        """
        if ctx is None:
            ctx = Context()

        self._validate(query)

        start = time.monotonic()
        ctx.log(f"Retrieval [{self.__class__.__name__}] starting")

        try:
            raw = await self._execute(query, ctx, **kwargs)
            formatted = self._format(raw)

            elapsed = (time.monotonic() - start) * 1000
            ctx.log(
                f"Retrieval [{self.__class__.__name__}] complete: "
                f"{len(formatted.items)} items in {elapsed:.1f}ms"
            )
            return formatted

        except RetrieverError:
            raise
        except Exception as exc:
            raise RetrieverError(
                f"Retrieval [{self.__class__.__name__}] failed: {exc}"
            ) from exc

    @abstractmethod
    async def _execute(
        self,
        query: str,
        ctx: Context,
        **kwargs: Any,
    ) -> RawSearchResult:
        """Subclasses implement actual retrieval logic here.

        Args:
            query: User's search query.
            ctx: Execution context.
            **kwargs: Strategy-specific parameters.

        Returns:
            RawSearchResult with raw database records.
        """
        ...

    def _validate(self, query: str) -> None:
        """Validate query input. Override to add strategy-specific validation."""
        if not query or not query.strip():
            raise RetrieverError("Empty query")

    def _format(self, raw: RawSearchResult) -> RetrieverResult:
        """Format raw results into RetrieverResult.

        Override for custom formatting — default stringifies each record.
        """
        items = [
            RetrieverResultItem(
                content=str(record),
                metadata=raw.metadata,
            )
            for record in raw.records
        ]
        return RetrieverResult(items=items, metadata=raw.metadata)
