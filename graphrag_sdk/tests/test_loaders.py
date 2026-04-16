"""Tests for ingestion/loaders/ — TextLoader and PdfLoader."""
from __future__ import annotations

import pytest

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
        file.write_bytes("café".encode())
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
    """PDF loader tests — skipped if no PDF backend is installed."""

    @pytest.fixture
    def pdf_available(self):
        try:
            import pypdf  # noqa: F401
        except ImportError:
            try:
                import fitz  # noqa: F401
            except ImportError:
                pytest.skip("no PDF backend installed (pypdf or pymupdf)")

    async def test_file_not_found(self, ctx, pdf_available):
        from graphrag_sdk.ingestion.loaders.pdf_loader import PdfLoader

        loader = PdfLoader()
        with pytest.raises(LoaderError, match="PDF file not found"):
            await loader.load("/nonexistent.pdf", ctx)

    async def test_prefers_pymupdf_when_available(self, ctx, tmp_path, monkeypatch):
        """When both backends are wired, PyMuPDF path runs and metadata reflects it."""
        from graphrag_sdk.core.models import DocumentInfo, DocumentOutput
        from graphrag_sdk.ingestion.loaders import pdf_loader as pdf_mod

        file = tmp_path / "stub.pdf"
        file.write_bytes(b"%PDF-stub")
        calls: dict[str, int] = {"pymupdf": 0, "pypdf": 0}

        def fake_pymupdf(self, path):
            calls["pymupdf"] += 1
            return DocumentOutput(
                text="pymupdf-text",
                document_info=DocumentInfo(
                    path=str(path),
                    metadata={"page_count": 1, "loader": "pdf", "pdf_backend": "pymupdf"},
                ),
            )

        def fake_pypdf(self, path):
            calls["pypdf"] += 1
            raise AssertionError("pypdf should not be used when pymupdf succeeds")

        monkeypatch.setattr(pdf_mod.PdfLoader, "_load_with_pymupdf", fake_pymupdf)
        monkeypatch.setattr(pdf_mod.PdfLoader, "_load_with_pypdf", fake_pypdf)

        result = await pdf_mod.PdfLoader().load(str(file), ctx)
        assert result.document_info.metadata["pdf_backend"] == "pymupdf"
        assert calls == {"pymupdf": 1, "pypdf": 0}

    async def test_falls_back_to_pypdf_when_pymupdf_missing(self, ctx, tmp_path, monkeypatch):
        """If PyMuPDF raises ImportError, loader falls back to pypdf and marks backend."""
        from graphrag_sdk.core.models import DocumentInfo, DocumentOutput
        from graphrag_sdk.ingestion.loaders import pdf_loader as pdf_mod

        file = tmp_path / "stub.pdf"
        file.write_bytes(b"%PDF-stub")

        def fake_pymupdf(self, path):
            raise ImportError("fitz not installed")

        def fake_pypdf(self, path):
            return DocumentOutput(
                text="pypdf-text",
                document_info=DocumentInfo(
                    path=str(path),
                    metadata={"page_count": 1, "loader": "pdf", "pdf_backend": "pypdf"},
                ),
            )

        monkeypatch.setattr(pdf_mod.PdfLoader, "_load_with_pymupdf", fake_pymupdf)
        monkeypatch.setattr(pdf_mod.PdfLoader, "_load_with_pypdf", fake_pypdf)

        result = await pdf_mod.PdfLoader().load(str(file), ctx)
        assert result.document_info.metadata["pdf_backend"] == "pypdf"

    async def test_raises_when_no_backend_installed(self, ctx, tmp_path, monkeypatch):
        """With neither backend available, a clear install message is raised."""
        from graphrag_sdk.ingestion.loaders import pdf_loader as pdf_mod

        file = tmp_path / "stub.pdf"
        file.write_bytes(b"%PDF-stub")

        def missing(self, path):
            raise ImportError("not installed")

        monkeypatch.setattr(pdf_mod.PdfLoader, "_load_with_pymupdf", missing)
        monkeypatch.setattr(pdf_mod.PdfLoader, "_load_with_pypdf", missing)

        with pytest.raises(ImportError, match=r"graphrag-sdk\[pdf"):
            await pdf_mod.PdfLoader().load(str(file), ctx)
