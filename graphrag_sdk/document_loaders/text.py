from typing import Iterator
from graphrag_sdk.document import Document

class TextLoader():
    """
    Load Text
    """

    def __init__(self, source) -> None:
        """
        Initialize loader

        Parameters:
            source: source
        """

        self.source = source
        self.path = source.data_source

    def load(self, chunking_processor=None) -> Iterator[Document]:
        """
        Load Text
        
        Parameters:
            chunking_processor (function): function to process chunks

        Returns:
            Iterator[Document]: document iterator
        """

        with open(self.path, 'r') as f:
            f.read()

        chunks = self.source.get_chunks(f, chunking_processor)
        
        yield from [
            Document(chunk)
            for chunk in chunks
        ]

