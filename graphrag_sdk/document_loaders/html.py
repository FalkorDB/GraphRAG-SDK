import re
import requests
from typing import Iterator
from bs4 import BeautifulSoup
from graphrag_sdk.Document import Document

class HTMLLoader():
    """
    Load HTML
    """

    def __init__(self, path: str) -> None:
        """
        Initialize loader

        Parameters:
            path (str): path to HTML.
        """

        self.path = path

    def _download(self) -> str:
        try:
            response = requests.get(self.path)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    def load(self) -> Iterator[Document]:
        """
        Load HTML

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

        yield Document(content)
        #return f"{self.source}\n{self.content}"
