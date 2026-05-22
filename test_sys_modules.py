import sys
from unittest.mock import MagicMock
sys.modules['docling'] = MagicMock()
sys.modules['docling.document_converter'] = MagicMock()
import docling.document_converter
print("Success!")
