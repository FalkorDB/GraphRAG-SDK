import re
import requests
from typing import Iterator
from bs4 import BeautifulSoup
from graphrag_sdk.document import Document

class URLLoader():
    """
    Load URL
    """

    def __init__(self, url: str) -> None:
        """
        Initialize loader

        Args:
            url (str): url.
        """

        self.url = url

    def _download(self) -> str:
        try:
            response = requests.get(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    def load(self) -> Iterator[Document]:
        """
        Load URL

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

        yield Document(content, self.url)