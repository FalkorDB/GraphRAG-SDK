from typing import Iterator
from graphrag_sdk.document import Document

class StringLoader():
    """
    Load String 
    """

    def __init__(self, source) -> None:
        """
        Initialize loader

        Parameters:
            source: source
        """
        self.source = source
        self.string = source.data_source

    def load(self, chunking_processor=None) -> Iterator[Document]:
        """
        Load string from memory
        
        Parameters:
            chunking_processor (function): function to process chunks

        Returns:
            Iterator[Document]: document iterator
        """
        
        
        chunks = self.source.get_chunks(self.string, chunking_processor)
        
        yield from [
            Document(chunk)
            for chunk in chunks
        ]