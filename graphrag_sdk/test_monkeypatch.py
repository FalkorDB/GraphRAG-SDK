import sys, asyncio
from unittest.mock import patch, MagicMock

async def main():
    mock_docling = MagicMock()
    # mock_docling.__path__ = [] # Let's see without this
    modules = {
        'docling': mock_docling,
        'docling.datamodel': MagicMock(),
        'docling.datamodel.document': MagicMock()
    }
    with patch.dict('sys.modules', modules):
        def worker():
            from docling.datamodel.document import DocItemLabel
            return 'success'
        res = await asyncio.to_thread(worker)
        print(res)

asyncio.run(main())
