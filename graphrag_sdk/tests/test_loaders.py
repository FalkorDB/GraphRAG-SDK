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


class TestMarkdownLoader:
    async def test_load_markdown_structure(self, ctx, tmp_path):
        from graphrag_sdk.ingestion.loaders.markdown_loader import MarkdownLoader

        file = tmp_path / "struct.md"
        content = (
            "# Main Title\n"
            "Intro paragraph.\n\n"
            "## Section 1\n"
            "Section 1 details.\n\n"
            "### Subsection 1.1\n"
            "Deep details."
        )
        file.write_text(content)
        loader = MarkdownLoader()
        result = await loader.load(str(file), ctx)
        
        elements = result.elements
        assert elements is not None
        assert len(elements) == 6  # 3 headers, 3 paragraphs
        
        # Verify first header
        assert elements[0].type == "header"
        assert elements[0].level == 1
        assert elements[0].content == "Main Title"
        assert elements[0].breadcrumbs == ["Main Title"]
        
        # Verify first paragraph
        assert elements[1].type == "paragraph"
        assert elements[1].content == "Intro paragraph."
        assert elements[1].breadcrumbs == ["Main Title"]
        
        # Verify H2
        assert elements[2].type == "header"
        assert elements[2].level == 2
        assert elements[2].content == "Section 1"
        assert elements[2].breadcrumbs == ["Main Title", "Section 1"]
        
        # Verify deeply nested paragraph
        assert elements[5].type == "paragraph"
        assert elements[5].content == "Deep details."
        assert elements[5].breadcrumbs == ["Main Title", "Section 1", "Subsection 1.1"]

    async def test_file_not_found(self, ctx):
        from graphrag_sdk.ingestion.loaders.markdown_loader import MarkdownLoader
        loader = MarkdownLoader()
        with pytest.raises(LoaderError, match="File not found"):
            await loader.load("/nonexistent.md", ctx)

    async def test_markdown_complex_structures(self, ctx, tmp_path):
        from graphrag_sdk.ingestion.loaders.markdown_loader import MarkdownLoader

        file = tmp_path / "complex.md"
        content = (
            "# Data\n\n"
            "| ID | Name |\n"
            "|---|---|\n"
            "| 1 | Alice |\n\n"
            "Some list:\n"
            "- Item 1\n"
            "  - Subitem\n"
            "\n"
            "- Item 2\n\n"
            "Code:\n"
            "```python\n"
            "def foo():\n"
            "    return 42\n"
            "```\n"
        )
        file.write_text(content)
        loader = MarkdownLoader()
        result = await loader.load(str(file), ctx)
        
        elements = result.elements
        assert elements is not None
        
        # Verify table
        assert elements[1].type == "table"
        assert "| 1 | Alice |" in elements[1].content
        
        # Verify paragraph
        assert elements[2].type == "paragraph"
        assert elements[2].content == "Some list:"
        
        # Verify list
        assert elements[3].type == "list"
        assert "- Item 1" in elements[3].content
        assert "- Item 2" in elements[3].content
        
        # Verify code block
        assert elements[5].type == "code"
        assert "def foo():" in elements[5].content
