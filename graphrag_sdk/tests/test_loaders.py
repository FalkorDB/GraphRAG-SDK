"""Tests for ingestion/loaders/ — TextLoader and PdfLoader."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LoaderError
from graphrag_sdk.ingestion.loaders.text_loader import TextLoader


class TestTextLoader:
    async def test_load_text_file(self, ctx, tmp_path):
        file = tmp_path / "test.txt"
        file.write_text("Hello, GraphRAG!")
        loader = TextLoader()
        result = await loader.load(str(file), ctx)
        assert result.text == "Hello, GraphRAG!"
        assert result.document_info.path == str(file)
        assert result.document_info.metadata["loader"] == "text"

    async def test_load_markdown_file(self, ctx, tmp_path):
        file = tmp_path / "doc.md"
        file.write_text("# Title\n\nSome content here.")
        loader = TextLoader()
        result = await loader.load(str(file), ctx)
        assert "# Title" in result.text
        assert result.document_info.metadata["suffix"] == ".md"

    async def test_file_not_found(self, ctx):
        loader = TextLoader()
        with pytest.raises(LoaderError, match="File not found"):
            await loader.load("/nonexistent/path.txt", ctx)

    async def test_metadata_has_size(self, ctx, tmp_path):
        file = tmp_path / "sized.txt"
        content = "X" * 42
        file.write_text(content)
        loader = TextLoader()
        result = await loader.load(str(file), ctx)
        assert result.document_info.metadata["size_bytes"] == 42

    async def test_empty_file(self, ctx, tmp_path):
        file = tmp_path / "empty.txt"
        file.write_text("")
        loader = TextLoader()
        result = await loader.load(str(file), ctx)
        assert result.text == ""

    async def test_custom_encoding(self, ctx, tmp_path):
        file = tmp_path / "latin.txt"
        file.write_bytes("café".encode("utf-8"))
        loader = TextLoader(encoding="utf-8")
        result = await loader.load(str(file), ctx)
        assert "café" in result.text

    async def test_document_info_has_uid(self, ctx, tmp_path):
        file = tmp_path / "uid.txt"
        file.write_text("content")
        loader = TextLoader()
        result = await loader.load(str(file), ctx)
        assert result.document_info.uid  # non-empty UUID


class TestPdfLoader:
    """PDF loader tests — skipped if pypdf not installed."""

    @pytest.fixture
    def pdf_available(self):
        try:
            import pypdf
            return True
        except ImportError:
            pytest.skip("pypdf not installed")

    async def test_file_not_found(self, ctx, pdf_available):
        from graphrag_sdk.ingestion.loaders.pdf_loader import PdfLoader

        loader = PdfLoader()
        with pytest.raises(LoaderError, match="PDF file not found"):
            await loader.load("/nonexistent.pdf", ctx)
