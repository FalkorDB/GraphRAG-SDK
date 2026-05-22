import sys
from unittest.mock import MagicMock
class LabelEnum:
    LIST_ITEM = "list_item"

mock_datamodel = MagicMock()
mock_datamodel.DocItemLabel = LabelEnum

sys.modules["docling"] = MagicMock()
sys.modules["docling.datamodel"] = MagicMock()
sys.modules["docling.datamodel.document"] = mock_datamodel

from docling.datamodel.document import DocItemLabel
print("DocItemLabel is:", DocItemLabel)
print("DocItemLabel.LIST_ITEM is:", getattr(DocItemLabel, "LIST_ITEM", None))
