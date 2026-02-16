# GraphRAG SDK 2.0 — Ingestion: Extraction Strategy ABC
# Pattern: Strategy — different extraction approaches implement this interface.

from __future__ import annotations

from abc import ABC, abstractmethod

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, GraphSchema, TextChunks


class ExtractionStrategy(ABC):
    """Abstract base class for entity/relationship extraction strategies.

    An extraction strategy takes text chunks and a schema, then returns
    ``GraphData`` containing extracted nodes and relationships.

    The schema constrains the extraction — implementations should respect
    the defined entity types, relationship types, and patterns.

    Example::

        class OpenIEExtraction(ExtractionStrategy):
            async def extract(self, chunks, schema, ctx):
                # Open information extraction without schema constraints
                ...
    """

    @abstractmethod
    async def extract(
        self,
        chunks: TextChunks,
        schema: GraphSchema,
        ctx: Context,
    ) -> GraphData:
        """Extract entities and relationships from text chunks.

        Args:
            chunks: Text chunks to extract from.
            schema: Graph schema constraining extraction.
            ctx: Execution context.

        Returns:
            GraphData with extracted nodes and relationships.
        """
        ...
