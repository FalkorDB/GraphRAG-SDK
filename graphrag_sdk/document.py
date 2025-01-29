class Document():
    """
    Common class containing text extracted from a source
    """

    def __init__(self, content: str) -> None:
        """
        Initializes a new Document object.

        Args:
            content (str): The content of the document.

        Returns:
            None
        """
        self.content = content

    def not_empty(self):
        return self.content is not None and len(self.content) > 0

