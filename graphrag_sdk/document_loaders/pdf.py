from typing import Iterator
from graphrag_sdk.document import Document

class PDFLoader():
    """
    Load PDF
    """

    def __init__(self, path: str) -> None:
        """
        Initialize loader

        Parameters:
            path (str): path to PDF.
        """

        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf package not found, please install it with " "`pip install pypdf`"
        )

        self.path = path

    def load(self) -> Iterator[Document]:
        """
        Load PDF

        Returns:
            Iterator[Document]: document iterator
        """
        
        from pypdf import PdfReader # pylint: disable=import-outside-toplevel

        reader = PdfReader(self.path)
        yield from [
            Document(page.extract_text())
            for page in reader.pages
        ]
