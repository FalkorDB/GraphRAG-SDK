# GraphRAG SDK 2.0 — Ingestion: Text File Loader

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LoaderError
from graphrag_sdk.core.models import DocumentInfo, DocumentOutput
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy

logger = logging.getLogger(__name__)


class TextLoader(LoaderStrategy):
    """Load text from a plain text or markdown file.

    Supports any UTF-8 encoded text file.
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    async def load(self, source: str, ctx: Context) -> DocumentOutput:
        ctx.log(f"Loading text file: {source}")
        return await asyncio.to_thread(self._load_sync, source)

    def _load_sync(self, source: str) -> DocumentOutput:
        path = Path(source)
        if not path.exists():
            raise LoaderError(f"File not found: {source}")

        try:
            text = path.read_text(encoding=self.encoding)
            return DocumentOutput(
                text=text,
                document_info=DocumentInfo(
                    path=str(path),
                    metadata={
                        "size_bytes": path.stat().st_size,
                        "loader": "text",
                        "suffix": path.suffix,
                    },
                ),
            )
        except Exception as exc:
            raise LoaderError(f"Failed to read {source}: {exc}") from exc
