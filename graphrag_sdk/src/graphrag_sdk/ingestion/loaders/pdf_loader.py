# GraphRAG SDK — Ingestion: PDF Loader

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
    """Load text from a PDF file.

    Prefers ``PyMuPDF`` (fitz) with ``sort=True`` for layout-preserving extraction
    that keeps table column alignment intact — this materially improves entity
    extraction recall on technical PDFs. Falls back to ``pypdf`` when PyMuPDF
    is not installed.

    Install one of:
      - ``pip install graphrag-sdk[pdf]``       — pypdf (default, Apache-2.0)
      - ``pip install graphrag-sdk[pdf-fast]``  — PyMuPDF (AGPL-3.0, faster + table-aware)
    """

    async def load(self, source: str, ctx: Context) -> DocumentOutput:
        ctx.log(f"Loading PDF: {source}")
        return await asyncio.to_thread(self._load_sync, source)

    def _load_sync(self, source: str) -> DocumentOutput:
        path = Path(source)
        if not path.exists():
            raise LoaderError(f"PDF file not found: {source}")

        # Try PyMuPDF first (better table/layout support)
        try:
            return self._load_with_pymupdf(path)
        except ImportError:
            logger.info("PyMuPDF not available, falling back to pypdf")

        # Fallback to pypdf
        try:
            return self._load_with_pypdf(path)
        except ImportError:
            raise ImportError(
                "A PDF library is required for PDF loading. Install one of:\n"
                "  pip install graphrag-sdk[pdf]       # pypdf (default)\n"
                "  pip install graphrag-sdk[pdf-fast]  # PyMuPDF, AGPL-3.0"
            )

    def _load_with_pymupdf(self, path: Path) -> DocumentOutput:
        """Extract text using PyMuPDF with sort=True for table-aware layout."""
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        try:
            pages: list[str] = []
            for page in doc:
                text = page.get_text(sort=True)
                if text and text.strip():
                    pages.append(text)
            page_count = len(doc)
        except Exception as exc:
            raise LoaderError(f"Failed to read PDF {path}: {exc}") from exc
        finally:
            doc.close()

        full_text = "\n\n".join(pages)
        logger.info(
            "PDF loaded with PyMuPDF (sort=True): %d pages, %d chars",
            page_count,
            len(full_text),
        )
        return DocumentOutput(
            text=full_text,
            document_info=DocumentInfo(
                path=str(path),
                metadata={
                    "page_count": page_count,
                    "loader": "pdf",
                    "pdf_backend": "pymupdf",
                },
            ),
        )

    def _load_with_pypdf(self, path: Path) -> DocumentOutput:
        """Extract text using pypdf (fallback)."""
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(path))
            pages: list[str] = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)

            full_text = "\n\n".join(pages)
            logger.info(
                "PDF loaded with pypdf: %d pages, %d chars",
                len(reader.pages),
                len(full_text),
            )
            return DocumentOutput(
                text=full_text,
                document_info=DocumentInfo(
                    path=str(path),
                    metadata={
                        "page_count": len(reader.pages),
                        "loader": "pdf",
                        "pdf_backend": "pypdf",
                    },
                ),
            )
        except Exception as exc:
            raise LoaderError(f"Failed to read PDF {path}: {exc}") from exc
