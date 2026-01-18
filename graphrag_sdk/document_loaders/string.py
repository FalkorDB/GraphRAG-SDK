from typing import Iterator
from graphrag_sdk.document import Document

class StringLoader():
    """
    Load String 
    """

    def __init__(self, string: str) -> None:
        """
        Initialize loader

        Args:
            string (str): string from memory.
        """
        self.string = string

    def load(self) -> Iterator[Document]:
        """
        Load string from memory

        Returns:
            Iterator[Document]: document iterator
        """
        yield Document(self.string)