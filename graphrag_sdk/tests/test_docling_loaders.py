import pytest
import sys
from unittest.mock import MagicMock, patch
from graphrag_sdk.core.exceptions import LoaderError
from graphrag_sdk.core.models import DocumentOutput, DocumentInfo, DocumentElement
from graphrag_sdk.ingestion.loaders.docling_base import DoclingBaseLoader

class MockDocxLoader(DoclingBaseLoader):
    @property
    def extension_name(self) -> str:
        return "docx"

class MockLabel:
    def __init__(self, val): self.value = val
    def __eq__(self, other):
        if isinstance(other, type) and hasattr(other, "SECTION_HEADER"):
            return False # Not a direct match
        return isinstance(other, MockLabel) and self.value == other.value

class LabelEnum:
    SECTION_HEADER = MockLabel("section_header")
    PARAGRAPH = MockLabel("paragraph")
    FOOTNOTE = MockLabel("footnote")
    TITLE = MockLabel("title")
    TEXT = MockLabel("text")
    LIST_ITEM = MockLabel("list_item")
    TABLE = MockLabel("table")
    CODE = MockLabel("code")

class TestDoclingBaseLoader:
    """Tests for DoclingBaseLoader and its derived loaders."""

    @pytest.fixture(autouse=True)
    def mock_docling_modules(self):
        """Mock the docling module namespace in sys.modules."""
        mock_datamodel = MagicMock()
        mock_datamodel.DocItemLabel = LabelEnum
        
        mock_converter_mod = MagicMock()
        mock_converter_mod.DocumentConverter = MagicMock()

        mock_docling = MagicMock()
        mock_docling.__path__ = []
        
        mock_datamodel_pkg = MagicMock()
        mock_datamodel_pkg.__path__ = []
        
        modules = {
            "docling": mock_docling,
            "docling.datamodel": mock_datamodel_pkg,
            "docling.datamodel.document": mock_datamodel,
            "docling.document_converter": mock_converter_mod,
        }
        
        with patch.dict("sys.modules", modules):
            yield

    async def test_import_error_wrapped_in_loader_error(self, ctx, tmp_path):
        """Verify that ImportError when docling is missing is wrapped in LoaderError."""
        file = tmp_path / "test.docx"
        file.write_text("dummy content")
        
        loader = MockDocxLoader()

        # Mocking the import to raise ImportError
        real_import = __import__
        def _import(name, *args, **kwargs):
            if name == "docling.document_converter":
                raise ImportError("module not found")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import):
            with pytest.raises(LoaderError, match=r"DOCX parsing requires 'docling'"):
                await loader.load(str(file), ctx)

    async def test_label_mapping_and_metadata_preservation(self, ctx, tmp_path):
        """Verify mapping of DocItemLabel and preservation of labels in metadata for fallback cases."""
        file = tmp_path / "test.docx"
        file.write_text("dummy content")

        # Mock Docling's result structure
        mock_item_header = MagicMock()
        mock_item_header.label = LabelEnum.SECTION_HEADER
        mock_item_header.text = "Header 1"
        
        mock_item_para = MagicMock()
        mock_item_para.label = LabelEnum.PARAGRAPH
        mock_item_para.text = "Paragraph 1"
        
        mock_item_footnote = MagicMock()
        mock_item_footnote.label = LabelEnum.FOOTNOTE
        mock_item_footnote.text = "Footnote content"
        
        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = [
            (mock_item_header, 1),
            (mock_item_para, 2),
            (mock_item_footnote, 2),
        ]

        mock_converter = MagicMock()
        mock_converter.convert.return_value.document = mock_doc
        sys.modules["docling.document_converter"].DocumentConverter = lambda **kwargs: mock_converter

        loader = MockDocxLoader()
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
        assert elements[2].metadata["label"] == LabelEnum.FOOTNOTE.value

    async def test_breadcrumbs_construction(self, ctx, tmp_path):
        """Verify the breadcrumbs are built correctly following the header hierarchy."""
        file = tmp_path / "test.docx"
        file.write_text("dummy content")

        # Hierarchy: H1 -> H2 -> P -> H2 (new) -> P
        mock_items = []
        mock_items.append((MagicMock(label=LabelEnum.SECTION_HEADER, text="Root"), 1))
        mock_items.append((MagicMock(label=LabelEnum.SECTION_HEADER, text="Child"), 2))
        mock_items.append((MagicMock(label=LabelEnum.PARAGRAPH, text="Text 1"), 3))
        mock_items.append((MagicMock(label=LabelEnum.SECTION_HEADER, text="Sibling"), 2))
        mock_items.append((MagicMock(label=LabelEnum.PARAGRAPH, text="Text 2"), 3))

        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = mock_items
        mock_converter = MagicMock()
        mock_converter.convert.return_value.document = mock_doc
        sys.modules["docling.document_converter"].DocumentConverter = lambda **kwargs: mock_converter

        loader = MockDocxLoader()
        result = await loader.load(str(file), ctx)

        elements = result.elements
        assert elements[0].breadcrumbs == ["Root"]
        assert elements[1].breadcrumbs == ["Root", "Child"]
        assert elements[2].breadcrumbs == ["Root", "Child"]
        assert elements[3].breadcrumbs == ["Root", "Sibling"]
        assert elements[4].breadcrumbs == ["Root", "Sibling"]

    async def test_file_not_found(self, ctx):
        """Verify that LoaderError is raised when the file does not exist."""
        loader = MockDocxLoader()
        with pytest.raises(LoaderError, match="File not found"):
            await loader.load("/non/existent/path.docx", ctx)

    async def test_docling_conversion_failure(self, ctx, tmp_path):
        """Verify that exceptions during docling conversion are wrapped in LoaderError."""
        file = tmp_path / "test.docx"
        file.write_text("dummy content")

        mock_converter = MagicMock()
        mock_converter.convert.side_effect = Exception("Conversion failed")
        sys.modules["docling.document_converter"].DocumentConverter = lambda **kwargs: mock_converter

        loader = MockDocxLoader()
        with pytest.raises(LoaderError, match="Docling failed to process"):
            await loader.load(str(file), ctx)

    async def test_export_to_markdown_fallback(self, ctx, tmp_path):
        """Verify fallback to export_to_markdown when text attribute is empty."""
        file = tmp_path / "test.docx"
        file.write_text("dummy content")

        mock_item = MagicMock()
        mock_item.label = LabelEnum.PARAGRAPH
        mock_item.text = ""
        mock_item.export_to_markdown.return_value = "Fallback Markdown Content"
        
        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = [(mock_item, 1)]
        
        mock_converter = MagicMock()
        mock_converter.convert.return_value.document = mock_doc
        sys.modules["docling.document_converter"].DocumentConverter = lambda **kwargs: mock_converter

        loader = MockDocxLoader()
        result = await loader.load(str(file), ctx)

        assert len(result.elements) == 1
        assert result.elements[0].content == "Fallback Markdown Content"

    async def test_specialized_element_types(self, ctx, tmp_path):
        """Verify mapping of list, table, and code elements."""
        file = tmp_path / "test.docx"
        file.write_text("dummy content")

        mock_items = [
            (MagicMock(label=LabelEnum.LIST_ITEM, text="List item 1"), 1),
            (MagicMock(label=LabelEnum.TABLE, text="Table content"), 1),
            (MagicMock(label=LabelEnum.CODE, text="print('hello')"), 1),
        ]
        
        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = mock_items
        
        mock_converter = MagicMock()
        mock_converter.convert.return_value.document = mock_doc
        sys.modules["docling.document_converter"].DocumentConverter = lambda **kwargs: mock_converter

        loader = MockDocxLoader()
        result = await loader.load(str(file), ctx)

        elements = result.elements
        assert elements[0].type == "list"
        assert elements[1].type == "table"
        assert elements[2].type == "code"
