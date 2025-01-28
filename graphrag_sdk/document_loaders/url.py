import re
import requests
from typing import Iterator
from bs4 import BeautifulSoup
from graphrag_sdk.document import Document

class URLLoader():
    """
    Load URL
    """

    def __init__(self, source) -> None:
        """
        Initialize loader

        Parameters:
            source: source
        """

        self.source = source
        self.url = source.data_source

    def _download(self) -> str:
        try:
            response = requests.get(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    def load(self, chunking_processor=None) -> Iterator[Document]:
        """
        Load URL
        
        Parameters:
            chunking_processor (function): function to process chunks

        Returns:
            Iterator[Document]: document iterator
        """

        # Download URL
        content = self._download()

        # extract text from HTML, populate content
        soup = BeautifulSoup(content, 'html.parser')

        # Extract text from the HTML
        content = soup.get_text()

        # Remove extra newlines
        content = re.sub(r'\n{2,}', '\n', content)
        
        chunks = self.source.get_chunks(content, chunking_processor)
        
        yield from [
            Document(chunk)
            for chunk in chunks
        ]
