from typing import Iterator
from graphrag_sdk.document import Document

class PDFLoader():
    """
    Load PDF
    """

    def __init__(self, source) -> None:
        """
        Initialize loader

        Parameters:
            source: source
        """

        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf package not found, please install it with " "`pip install pypdf`"
        )

        self.source = source
        self.path = source.data_source

    def load(self, chunking_processor=None) -> Iterator[Document]:
        """
        Load PDF
        
        Parameters:
            chunking_processor (function): function to process chunks

        Returns:
            Iterator[Document]: document iterator
        """
        
        from pypdf import PdfReader # pylint: disable=import-outside-toplevel

        reader = PdfReader(self.path)
        
        full_pdf_text = "\n".join(page.extract_text() or '' for page in reader.pages)
        
        chunks = self.source.get_chunks(full_pdf_text, chunking_processor)
        
        yield from [
            Document(chunk)
            for chunk in chunks
        ]
