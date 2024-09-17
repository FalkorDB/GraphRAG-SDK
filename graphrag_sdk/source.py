from typing import Iterator
from abc import ABC, abstractmethod
from graphrag_sdk.document import Document
from graphrag_sdk.document_loaders import (
    PDFLoader,
    TextLoader,
    URLLoader,
    HTMLLoader,
    CSVLoader,
    JSONLLoader,
)


def Source(path: str, instruction: str | None = None) -> "AbstractSource":
    """
    Creates a source object

    Parameters:
        path (str): path to source
        instruction (str): source specific instruction for the LLM

    Returns:
        AbstractSource: source
    """

    if not isinstance(path, str) or path == "":
        raise Exception("Invalid argument, path should be a none empty string.")

    s = None

    if ".pdf" in path.lower():
        s = PDF(path)
    elif ".html" in path.lower():
        s = HTML(path)
    elif "http" in path.lower():
        s = URL(path)
    elif ".csv" in path.lower():
        s = CSV(path)
    elif ".jsonl" in path.lower():
        s = JSONL(path)
    else:
        s = TEXT(path)

    # Set source instructions
    s.instruction = instruction

    return s


class AbstractSource(ABC):
    """
    Abstract class representing a source file
    """

    def __init__(self, path: str):
        """
        Initializes a new instance of the Source class.

        Args:
            path (str): The path to the source file.

        Attributes:
            path (str): The path to the source file.
            loader: The loader object associated with the source file.
            instruction (str): The instruction for the source file.
        """
        self.path = path
        self.loader = None
        self.instruction = ""

    def load(self) -> Iterator[Document]:
        """
        Loads documents from the source.

        Returns:
            An iterator of Document objects.
        """
        return self.loader.load()

    def __eq__(self, other) -> bool:
        """
        Check if this source object is equal to another source object.

        Args:
            other: The other source object to compare with.

        Returns:
            bool: True if the source objects are equal, False otherwise.
        """
        if not isinstance(other, AbstractSource):
            return False

        return self.path == other.path

    def __hash__(self):
        """
        Calculates the hash value of the Source object based on its path.

        Returns:
            int: The hash value of the Source object.
        """
        return hash(self.path)


class PDF(AbstractSource):
    """
    PDF resource
    """

    def __init__(self, path):
        super().__init__(path)
        self.loader = PDFLoader(self.path)


class TEXT(AbstractSource):
    """
    TEXT resource
    """

    def __init__(self, path):
        super().__init__(path)
        self.loader = TextLoader(self.path)


class URL(AbstractSource):
    """
    URL resource
    """

    def __init__(self, path):
        super().__init__(path)
        self.loader = URLLoader(self.path)


class HTML(AbstractSource):
    """
    HTML resource
    """

    def __init__(self, path):
        super().__init__(path)
        self.loader = HTMLLoader(self.path)


class CSV(AbstractSource):
    """
    CSV resource
    """

    def __init__(self, path, rows_per_document: int = 50):
        super().__init__(path)
        self.loader = CSVLoader(self.path, rows_per_document)


class JSONL(AbstractSource):
    """
    JSONL resource
    """

    def __init__(self, path, rows_per_document: int = 50):
        super().__init__(path)
        self.loader = JSONLLoader(self.path, rows_per_document)
