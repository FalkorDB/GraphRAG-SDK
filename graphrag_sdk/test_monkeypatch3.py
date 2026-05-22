import sys, asyncio
from unittest.mock import patch, MagicMock

def load_sync():
    from docling.datamodel.document import DocItemLabel
    return "success"

async def test_first():
    real_import = __import__
    def _import(name, *args, **kwargs):
        if name == "docling.datamodel.document":
            raise ImportError("module not found")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_import):
        try:
            await asyncio.to_thread(load_sync)
        except Exception as e:
            pass # catch the mocked exception

async def test_second():
    res = await asyncio.to_thread(load_sync)
    print("Second test:", res)

async def main():
    modules = {
        'docling': MagicMock(),
        'docling.datamodel': MagicMock(),
        'docling.datamodel.document': MagicMock()
    }
    with patch.dict('sys.modules', modules):
        await test_first()
        await test_second()

asyncio.run(main())
