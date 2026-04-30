# GraphRAG SDK — Ingestion: Markdown Loader

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LoaderError
from graphrag_sdk.core.models import DocumentElement, DocumentInfo, DocumentOutput
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy

logger = logging.getLogger(__name__)


class MarkdownLoader(LoaderStrategy):
    """Load text and structural elements from a markdown file.

    Parses the markdown structure (headers and paragraphs) to build
    a flat list of ``DocumentElement``s. Each element contains ``breadcrumbs``
    representing its header hierarchy, enabling precise structural chunking.
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    async def load(self, source: str, ctx: Context) -> DocumentOutput:
        ctx.log(f"Loading markdown file: {source}")
        return await asyncio.to_thread(self._load_sync, source)

    def _load_sync(self, source: str) -> DocumentOutput:
        path = Path(source)
        if not path.exists():
            raise LoaderError(f"File not found: {source}")

        try:
            text = path.read_text(encoding=self.encoding)
            elements = self._parse_markdown(text)

            return DocumentOutput(
                text=text,
                document_info=DocumentInfo(
                    path=str(path),
                    metadata={
                        "size_bytes": path.stat().st_size,
                        "loader": "markdown",
                        "suffix": path.suffix,
                    },
                ),
                elements=elements,
            )
        except Exception as exc:
            raise LoaderError(f"Failed to read {source}: {exc}") from exc

    def _parse_markdown(self, text: str) -> list[DocumentElement]:
        """Parse raw markdown text into structural DocumentElements."""
        from markdown_it import MarkdownIt

        md = MarkdownIt("commonmark").enable("table")
        tokens = md.parse(text)
        lines = text.split("\n")

        elements: list[DocumentElement] = []
        current_breadcrumbs: list[tuple[int, str]] = []

        def get_content(t) -> str:
            if not t.map:
                return ""
            start, end = t.map
            return "\n".join(lines[start:end]).strip()

        def skip_to_close(start_idx: int, open_type: str, close_type: str) -> int:
            nesting = 1
            idx = start_idx + 1
            while idx < len(tokens):
                if tokens[idx].type == open_type:
                    nesting += 1
                elif tokens[idx].type == close_type:
                    nesting -= 1
                    if nesting == 0:
                        return idx
                idx += 1
            return idx

        def _emit(el_type: str, content: str) -> None:
            if content:
                elements.append(
                    DocumentElement(
                        type=el_type,
                        content=content,
                        breadcrumbs=[b[1] for b in current_breadcrumbs],
                    )
                )

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.type == "heading_open":
                content = get_content(token)
                level = int(token.tag[1:])

                title = ""
                for j in range(i + 1, len(tokens)):
                    if tokens[j].type == "inline":
                        title = tokens[j].content
                        break
                    if tokens[j].type == "heading_close":
                        break

                while current_breadcrumbs and current_breadcrumbs[-1][0] >= level:
                    current_breadcrumbs.pop()

                current_breadcrumbs.append((level, title))

                elements.append(
                    DocumentElement(
                        type="header",
                        level=level,
                        content=content,
                        breadcrumbs=[b[1] for b in current_breadcrumbs],
                    )
                )
                i = skip_to_close(i, "heading_open", "heading_close")

            elif token.type == "paragraph_open":
                _emit("paragraph", get_content(token))
                i = skip_to_close(i, "paragraph_open", "paragraph_close")

            elif token.type in ("bullet_list_open", "ordered_list_open"):
                _emit("list", get_content(token))
                close_type = token.type.replace("_open", "_close")
                i = skip_to_close(i, token.type, close_type)

            elif token.type == "table_open":
                _emit("table", get_content(token))
                i = skip_to_close(i, "table_open", "table_close")

            elif token.type == "blockquote_open":
                _emit("blockquote", get_content(token))
                i = skip_to_close(i, "blockquote_open", "blockquote_close")

            elif token.type in ("fence", "code_block"):
                _emit("code", get_content(token))

            i += 1

        return elements
