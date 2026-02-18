# GraphRAG SDK 2.0 — Ingestion: Resolution Strategy ABC
# Pattern: Strategy — different deduplication approaches implement this interface.

from __future__ import annotations

from abc import ABC, abstractmethod

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, ResolutionResult


class ResolutionStrategy(ABC):
    """Abstract base class for entity resolution (deduplication) strategies.

    A resolution strategy receives extracted graph data and merges
    duplicate entities, producing a deduplicated result.

    Example::

        class VectorFuzzyResolution(ResolutionStrategy):
            async def resolve(self, graph_data, ctx):
                # Use embeddings to find near-duplicate entities
                ...
    """

    @abstractmethod
    async def resolve(
        self,
        graph_data: GraphData,
        ctx: Context,
    ) -> ResolutionResult:
        """Resolve duplicate entities in the extracted graph data.

        Args:
            graph_data: Extracted nodes and relationships.
            ctx: Execution context.

        Returns:
            ResolutionResult with deduplicated data and merge statistics.
        """
        ...
