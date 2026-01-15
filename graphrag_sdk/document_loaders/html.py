import re
import requests
from typing import Iterator
from bs4 import BeautifulSoup
from graphrag_sdk.document import Document


class HTMLLoader:
    """
    Load HTML
    """

    def __init__(self, path: str) -> None:
        """
        Initialize loader

        Args:
            path (str): path to HTML.
        """

        self.path = path

    def _get_file(self) -> str:
        try:
            with open(self.path, "r") as f:
                return f.read()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    def load(self) -> Iterator[Document]:
        """
        Load HTML

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

        yield Document(content, self.path)