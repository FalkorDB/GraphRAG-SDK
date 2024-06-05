import re
import requests
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod

def Source(source:str) ->'AbstractSource':
    """
    Creates a source object

    Parameters:
        source (str): path to source

    Returns:
        AbstractSource: source
    """

    if not isinstance(source, str) or source == "":
        raise Exception("Invalid argument, source should be a none empty string.")

    if ".pdf" in source.lower():
        return PDF(source)
    elif ".html" in source.lower():
        return HTML(source)
    else:
        return TEXT(source)

class AbstractSource(ABC):
    """
    Abstract class representing a source file
    """

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
    """
    PDF resource
    """

    def __init__(self, source):
        super().__init__(source)

    def load(self):
        raise Exception("Not implemented")
        # extract text from PDF, populate self.content
        self.content = None

class TEXT(AbstractSource):
    """
    TEXT resource
    """

    def __init__(self, source):
        super().__init__(source)

    def load(self):
        if self.content is None:
            # populate self.content
            with open(self.source, 'r') as f:
                self.content = f.read()

        return self.content

class HTML(AbstractSource):
    """
    HTML resource
    """

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
