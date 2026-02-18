# GraphRAG SDK 2.0 — Ingestion: Loader Strategy ABC
# Pattern: Strategy — every data source adapter implements this interface.

from __future__ import annotations

from abc import ABC, abstractmethod

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import DocumentOutput


class LoaderStrategy(ABC):
    """Abstract base class for data source loaders.

    A loader reads from a source (file path, URL, S3 key, etc.) and
    returns a ``DocumentOutput`` containing the extracted text and
    document metadata.

    Example::

        class MyLoader(LoaderStrategy):
            async def load(self, source: str, ctx: Context) -> DocumentOutput:
                text = read_my_source(source)
                return DocumentOutput(text=text)
    """

    @abstractmethod
    async def load(self, source: str, ctx: Context) -> DocumentOutput:
        """Load raw text from a data source.

        Args:
            source: Path, URL, or identifier for the data source.
            ctx: Execution context.

        Returns:
            DocumentOutput with extracted text and metadata.
        """
        ...
