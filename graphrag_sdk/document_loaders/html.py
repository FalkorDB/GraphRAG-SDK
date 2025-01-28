import re
import requests
from typing import Iterator
from bs4 import BeautifulSoup
from graphrag_sdk.document import Document


class HTMLLoader:
    """
    Load HTML
    """

    def __init__(self, source) -> None:
        """
        Initialize loader

        Parameters:
            source: source
        """

        self.source = source
        self.path = source.data_source

    def _get_file(self) -> str:
        try:
            with open(self.path, "r") as f:
                return f.read()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    def load(self, chunking_processor=None) -> Iterator[Document]:
        """
        Load HTML
        
        Parameters:
            chunking_processor (function): function to process chunks

        Returns:
            Iterator[Document]: document iterator
        """

        # Download URL
        content = self._get_file()

        # extract text from HTML, populate content
        soup = BeautifulSoup(content, "html.parser")

        # Extract text from the HTML
        content = soup.get_text()

        # Remove extra newlines
        content = re.sub(r"\n{2,}", "\n", content)

        chunks = self.source.get_chunks(content, chunking_processor)
        
        yield from [
            Document(chunk)
            for chunk in chunks
        ]

