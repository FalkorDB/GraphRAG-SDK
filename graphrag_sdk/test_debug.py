import asyncio
from unittest.mock import MagicMock
from graphrag_sdk.core.context import Context
from tests.test_docling_loaders import MockDocxLoader, LabelEnum

loader = MockDocxLoader()
mock_items = [
    (MagicMock(label=LabelEnum.LIST_ITEM, text="List item 1"), 1),
    (MagicMock(label=LabelEnum.TABLE, text="Table content"), 1),
    (MagicMock(label=LabelEnum.CODE, text="print('hello')"), 1),
]

mock_doc = MagicMock()
mock_doc.iterate_items.return_value = mock_items

mock_converter = MagicMock()
mock_converter.convert.return_value.document = mock_doc

# Force the monkeypatch manually
import sys
sys.modules["docling"] = MagicMock()
sys.modules["docling.datamodel"] = MagicMock()
sys.modules["docling.datamodel.document"] = MagicMock(DocItemLabel=LabelEnum)
sys.modules["docling.document_converter"] = MagicMock(DocumentConverter=MagicMock(return_value=mock_converter))

ctx = Context()
result = loader._load_sync("dummy_path")
print(result.elements)
