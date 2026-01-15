from typing import Optional


class Document():
    """
    Common class containing text extracted from a source
    """

    def __init__(self, content: str, id: Optional[str] = None) -> None:
        """
        Initializes a new Document object.

        Args:
            content (str): The content of the document.
            id (Optional[str], optional): The identifier of the document. Defaults to None.

        Returns:
            None
        """
        self.content = content
        self.id = id
        
    def not_empty(self):
        return self.content is not None and len(self.content) > 0
