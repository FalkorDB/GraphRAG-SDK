import re
import requests
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod

def download_resource(url):
    print(f"Downloading: {url}")

    # Send an HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    content = None
    if response.status_code == 200:
        # Retrieve content
        content = response.text
    else:
        print(f"Failed to retrieve content. Status code: {response.status_code}")

    return content

def Source(source:str):
    if not isinstance(source, str) or source == "":
        raise Exception("Invalid argument, source should be a none empty string.")

    if ".pdf" in source.lower():
        return PDF(source)
    elif ".html" in source.lower():
        return HTML(source)
    else:
        return TEXT(source)

class AbstractSource(ABC):
    def __init__(self, source:str):
        self.source = source
        self.content = None

    @abstractmethod
    def load(self) -> str:
        pass

    def __eq__(self, other) -> bool:
        if not isinstance(other, AbstractSource):
            return False

        return self.source == other.source

    def __hash__(self):
        return hash(self.source)

class PDF(AbstractSource):
    def __init__(self, source):
        super().__init__(source)

    def load(self):
        # extract text from PDF, populate self.content
        self.content = None

class TEXT(AbstractSource):
    def __init__(self, source):
        super().__init__(source)

    def load(self):
        if self.content is None:
            # populate self.content
            with open(self.source, 'r') as f:
                self.content = f.read()

        return self.content

class HTML(AbstractSource):
    def __init__(self, source):
        super().__init__(source)

    def load(self):
        with open(self.source, 'r') as f:
            self.content = f.read()

        # extract text from HTML, populate self.content
        soup = BeautifulSoup(self.content, 'html.parser')

        # Extract text from the HTML
        self.content = soup.get_text()

        # Remove extra newlines
        self.content = re.sub(r'\n{2,}', '\n', self.content)
