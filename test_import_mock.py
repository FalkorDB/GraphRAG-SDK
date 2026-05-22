import sys
from unittest.mock import patch, MagicMock

modules = {"fake": MagicMock()}
with patch.dict("sys.modules", modules):
    def _import(name, *args, **kwargs):
        if name == "fake":
            raise ImportError("module not found")
        return __import__(name, *args, **kwargs)
    with patch("builtins.__import__", side_effect=_import):
        try:
            import fake
            print("Import succeeded")
        except ImportError:
            print("ImportError raised")
