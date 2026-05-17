import pytest
from unittest.mock import MagicMock, patch
from graphrag_sdk.core.exceptions import LoaderError
from graphrag_sdk.core.models import DocumentOutput, DocumentInfo, DocumentElement
from graphrag_sdk.ingestion.loaders.docling_base import DoclingBaseLoader

class TestDoclingBaseLoader:
    """Tests for DoclingBaseLoader and its derived loaders."""

    def test_import_error_wrapped_in_loader_error(self, ctx, tmp_path, monkeypatch):
        """Verify that ImportError when docling is missing is wrapped in LoaderError."""
        file = tmp_path / "test.docx"
        file.write_text("dummy content")
        
        loader = DoclingBaseLoader()
        loader.extension_name = "docx"

        # Mocking the import to raise ImportError
        with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: 
                   (exec('raise ImportError("module not found")') if name == "docling.document_converter" else None)):
            with pytest.raises(LoaderError, match=r"DOCX parsing requires 'docling'"):
                import asyncio
                asyncio.run(loader.load(str(file), ctx))

    async def test_label_mapping_and_metadata_preservation(self, ctx, tmp_path, monkeypatch):
        """Verify mapping of DocItemLabel and preservation of labels in metadata for fallback cases."""
        from docling.datamodel.document import DocItemLabel

        file = tmp_path / "test.docx"
        file.write_text("dummy content")

        # Mock Docling's result structure
        mock_item_header = MagicMock()
        mock_item_header.label = DocItemLabel.SECTION_HEADER
        mock_item_header.text = "Header 1"
        
        mock_item_para = MagicMock()
        mock_item_para.label = DocItemLabel.PARAGRAPH
        mock_item_para.text = "Paragraph 1"
        
        mock_item_footnote = MagicMock()
        mock_item_footnote.label = DocItemLabel.FOOTNOTE
        mock_item_footnote.text = "Footnote content"
        
        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = [
            (mock_item_header, 1),
            (mock_item_para, 2),
            (mock_item_footnote, 2),
        ]

        mock_converter = MagicMock()
        mock_converter.convert.return_value.document = mock_doc

        monkeypatch.setattr("docling.document_converter.DocumentConverter", lambda **kwargs: mock_converter)

        loader = DoclingBaseLoader()
        loader.extension_name = "docx"
        result = await loader.load(str(file), ctx)

        elements = result.elements
        assert len(elements) == 3
        assert elements[0].type == "header"
        assert elements[0].content == "Header 1"
        assert elements[1].type == "paragraph"
        assert elements[1].content == "Paragraph 1"
        # Check fallback and metadata preservation
        assert elements[2].type == "paragraph"
        assert elements[2].content == "Footnote content"
        assert elements[2].metadata["label"] == DocItemLabel.FOOTNOTE.value

    async def test_breadcrumbs_construction(self, ctx, tmp_path, monkeypatch):
        """Verify the breadcrumbs are built correctly following the header hierarchy."""
        file = tmp_path / "test.docx"
        file.write_text("dummy content")

        from docling.datamodel.document import DocItemLabel

        # Hierarchy: H1 -> H2 -> P -> H2 (new) -> P
        items = [
            (MagicMock(label=DocItemLabel.SECTION_HEADER, text="Root"), 1),
            (MagicMock(label=DocItemLabel.SECTION_HEADER, text="Child"), 2),
            (MagicMock(label=DocItemLabel.PARAGRAPH, text="Text 1"), 3),
            (MagicMock(label=DocItemLabel.SECTION_HEADER, text="Sibling"), 2),
            (MagicMock(label=DocItemLabel.PARAGRAPH, text="Text 2"), 3),
        ]
        # Fix texts since we use MagicMocks
        for item, _ in items:
            item.text = item.text if hasattr(item, 'text') else "" # not needed due to initialization above

        # Redefining properly with actual text
        mock_items = []
        mock_items.append((MagicMock(label=DocItemLabel.SECTION_HEADER, text="Root"), 1))
        mock_items.append((MagicMock(label=DocItemLabel.SECTION_HEADER, text="Child"), 2))
        mock_items.append((MagicMock(label=DocItemLabel.PARAGRAPH, text="Text 1"), 3))
        mock_items.append((MagicMock(label=DocItemLabel.SECTION_HEADER, text="Sibling"), 2))
        mock_items.append((MagicMock(label=DocItemLabel.PARAGRAPH, text="Text 2"), 3))

        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = mock_items
        mock_converter = MagicMock()
        mock_converter.convert.return_value.document = mock_doc
        monkeypatch.setattr("docling.document_converter.DocumentConverter", lambda **kwargs: mock_converter)

        loader = DoclingBaseLoader()
        loader.extension_name = "docx"
        result = await loader.load(str(file), ctx)

        elements = result.elements
        assert elements[0].breadcrumbs == ["Root"]
        assert elements[1].breadcrumbs == ["Root", "Child"]
        assert elements[2].breadcrumbs == ["Root", "Child"]
        assert elements[3].breadcrumbs == ["Root", "Sibling"]
        assert elements[4].breadcrumbs == ["Root", "Sibling"]

    @pytest.mark.skip(reason="Requires actual docling installation and sample files")
    async def test_real_file_loading(self, ctx, tmp_path):
        """Carga de arquivos reais ( Integration test )."""
        from graphrag_sdk.ingestion.loaders.docx_loader import DocxLoader
        from graphrag_sdk.ingestion.loaders.xlsx_loader import XlsxLoader
        
        # This would require actual bytes of docx/xlsx etc.
        pass
