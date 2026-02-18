# GraphRAG SDK 2.0 — Ingestion: PDF Loader

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LoaderError
from graphrag_sdk.core.models import DocumentInfo, DocumentOutput
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy

logger = logging.getLogger(__name__)


class PdfLoader(LoaderStrategy):
    """Load text from a PDF file using ``pypdf``.

    Requires the ``pdf`` extra: ``pip install graphrag-sdk[pdf]``
    """

    async def load(self, source: str, ctx: Context) -> DocumentOutput:
        ctx.log(f"Loading PDF: {source}")
        return await asyncio.to_thread(self._load_sync, source)

    def _load_sync(self, source: str) -> DocumentOutput:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF loading. "
                "Install with: pip install graphrag-sdk[pdf]"
            )

        path = Path(source)
        if not path.exists():
            raise LoaderError(f"PDF file not found: {source}")

        try:
            reader = PdfReader(str(path))
            pages: list[str] = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)

            full_text = "\n\n".join(pages)
            return DocumentOutput(
                text=full_text,
                document_info=DocumentInfo(
                    path=str(path),
                    metadata={"page_count": len(reader.pages), "loader": "pdf"},
                ),
            )
        except Exception as exc:
            raise LoaderError(f"Failed to read PDF {source}: {exc}") from exc
