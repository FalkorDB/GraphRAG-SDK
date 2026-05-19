# GraphRAG SDK — Ingestion: Docling Base Loader
# Pattern: Strategy Base Class

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LoaderError
from graphrag_sdk.core.models import DocumentElement, DocumentInfo, DocumentOutput
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy

logger = logging.getLogger(__name__)


class DoclingBaseLoader(LoaderStrategy):
    """Base loader using docling for advanced document parsing.

    Subclasses should define the `extension_name` property.
    """

    def __init__(self, **docling_kwargs: Any) -> None:
        """Initialize the loader.

        Args:
            **docling_kwargs: Arbitrary keyword arguments passed to
                `docling.document_converter.DocumentConverter` (e.g.,
                pipeline_options).
        """
        self.docling_kwargs = docling_kwargs

    @property
    def extension_name(self) -> str:
        return "unknown"

    async def load(self, source: str, ctx: Context) -> DocumentOutput:
        ctx.log(f"Loading {self.extension_name.upper()} file via docling: {source}")
        # Run synchronous docling extraction in a non-blocking thread
        return await asyncio.to_thread(self._load_sync, source)

    def _load_sync(self, source: str) -> DocumentOutput:
        path = Path(source)
        if not path.exists():
            raise LoaderError(f"File not found: {source}")

        try:
            from docling.datamodel.document import DocItemLabel
            from docling.document_converter import DocumentConverter
        except ImportError:
            raise LoaderError(
                f"{self.extension_name.upper()} parsing requires 'docling'. Install with:\n"
                "  pip install graphrag-sdk[docling]"
            )

        try:
            converter = DocumentConverter(**self.docling_kwargs)
            result = converter.convert(source)
            doc = result.document
        except Exception as exc:
            raise LoaderError(f"Docling failed to process {source}: {exc}") from exc

        elements: list[DocumentElement] = []
        current_breadcrumbs: list[tuple[int, str]] = []
        full_text_blocks = []

        # Map docling hierarchy to GraphRAG DocumentElements
        for item, level in doc.iterate_items():
            content = getattr(item, "text", "")
            if not content and hasattr(item, "export_to_markdown"):
                try:
                    content = item.export_to_markdown()
                except Exception:
                    pass

            if not content:
                continue

            full_text_blocks.append(content)
            label = getattr(item, "label", None)

            if label in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER):
                # Update breadcrumbs
                while current_breadcrumbs and current_breadcrumbs[-1][0] >= level:
                    current_breadcrumbs.pop()
                current_breadcrumbs.append((level, content))

                elements.append(
                    DocumentElement(
                        type="header",
                        level=level,
                        content=content,
                        breadcrumbs=[b[1] for b in current_breadcrumbs],
                    )
                )
            elif label in (DocItemLabel.PARAGRAPH, DocItemLabel.TEXT):
                elements.append(
                    DocumentElement(
                        type="paragraph",
                        content=content,
                        breadcrumbs=[b[1] for b in current_breadcrumbs],
                    )
                )
            elif label == DocItemLabel.LIST_ITEM:
                elements.append(
                    DocumentElement(
                        type="list",
                        content=content,
                        breadcrumbs=[b[1] for b in current_breadcrumbs],
                    )
                )
            elif label == DocItemLabel.TABLE:
                elements.append(
                    DocumentElement(
                        type="table",
                        content=content,
                        breadcrumbs=[b[1] for b in current_breadcrumbs],
                    )
                )
            elif label == DocItemLabel.CODE:
                elements.append(
                    DocumentElement(
                        type="code",
                        content=content,
                        breadcrumbs=[b[1] for b in current_breadcrumbs],
                    )
                )
            else:
                # Default for CAPTION, FOOTNOTE, etc.
                elements.append(
                    DocumentElement(
                        type="paragraph",
                        content=content,
                        breadcrumbs=[b[1] for b in current_breadcrumbs],
                        metadata={"label": label.value if hasattr(label, "value") else label},
                    )
                )

        full_text = "\n\n".join(full_text_blocks)

        return DocumentOutput(
            text=full_text,
            document_info=DocumentInfo(
                path=str(path),
                metadata={
                    "size_bytes": path.stat().st_size,
                    "loader": self.extension_name,
                    "suffix": path.suffix,
                },
            ),
            elements=elements,
        )
