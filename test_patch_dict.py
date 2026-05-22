import sys
from unittest.mock import patch, MagicMock
with patch.dict("sys.modules", {"docling": MagicMock(), "docling.document_converter": MagicMock()}):
    import docling.document_converter
    print("Success patch.dict!")
