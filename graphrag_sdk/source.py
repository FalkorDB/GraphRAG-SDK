from abc import ABC
from typing import Optional, Iterator
from graphrag_sdk.document import Document
from graphrag_sdk.document_loaders import (
    PDFLoader,
    TextLoader,
    URLLoader,
    HTMLLoader,
    CSVLoader,
    JSONLLoader,
    StringLoader,
)


def Source(data_source: str, instruction: Optional[str] = None) -> "AbstractSource":
    """
    Creates a source object from a given data_source.

    Parameters:
        data_source (str): path to a source or data string.
        instruction (str, optional): source specific instruction for the LLM. Defaults to None.

    Returns:
        AbstractSource: A source object corresponding to the input data source.
    """

    if not isinstance(data_source, str) or data_source == "":
        raise Exception("Invalid argument: data source should be a non-empty string.")

    if ".pdf" in data_source.lower():
        s = PDF(data_source)
    elif ".html" in data_source.lower():
        s = HTML(data_source)
    elif "http" in data_source.lower():
        s = URL(data_source)
    elif ".csv" in data_source.lower():
        s = CSV(data_source)
    elif ".jsonl" in data_source.lower():
        s = JSONL(data_source)
    elif ".txt" in data_source.lower():
        s = TEXT(data_source)
    else:
        s = STRING(data_source)

    # Set source instructions
    s.instruction = instruction

    return s

def Source_FromRawText(text: str, instruction: Optional[str] = None) -> "AbstractSource":
    """
    Creates a source object from raw text

    Args:
        text (str): raw text
        instruction (str): source specific instruction for the LLM

    Returns:
        AbstractSource: A string source object.
    """
    if not isinstance(text, str) or text == "":
        raise Exception("Invalid argument, text should be a none empty string.")

    s = STRING(text)
    s.instruction = instruction

    return s


class AbstractSource(ABC):
    """
    Abstract class representing a source file
    """

    def __init__(self, data_source: str):
        """
        Initializes a new instance of the Source class.

        Args:
            data_source (str): Either a file path or a string.

        Attributes:
            data_source (str): The source path for the data or the data as a string.
            loader: The loader object associated with the source.
            instruction (str): The instruction for the source file.
        """
        self.data_source = data_source
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

        return self.data_source == other.data_source

    def __hash__(self):
        """
        Calculates the hash value of the Source object based on its data_source.

        Returns:
            int: The hash value of the Source object.
        """
        return hash(self.data_source)


class PDF(AbstractSource):
    """
    PDF resource
    """

    def __init__(self, path):
        super().__init__(path)
        self.loader = PDFLoader(self.data_source)


class TEXT(AbstractSource):
    """
    TEXT resource
    """

    def __init__(self, path):
        super().__init__(path)
        self.loader = TextLoader(self.data_source)


class URL(AbstractSource):
    """
    URL resource
    """

    def __init__(self, path):
        super().__init__(path)
        self.loader = URLLoader(self.data_source)


class HTML(AbstractSource):
    """
    HTML resource
    """

    def __init__(self, path):
        super().__init__(path)
        self.loader = HTMLLoader(self.data_source)


class CSV(AbstractSource):
    """
    CSV resource
    """

    def __init__(self, path, rows_per_document: int = 50):
        super().__init__(path)
        self.loader = CSVLoader(self.data_source, rows_per_document)


class JSONL(AbstractSource):
    """
    JSONL resource
    """

    def __init__(self, path, rows_per_document: int = 50):
        super().__init__(path)
        self.loader = JSONLLoader(self.data_source, rows_per_document)
        
class STRING(AbstractSource):
    """
    String resource
    """

    def __init__(self, string: str):
        super().__init__(string)
        self.loader = StringLoader(self.data_source)
