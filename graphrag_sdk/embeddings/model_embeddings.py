from abc import ABC, abstractmethod

class EmbeddingsGenerativeModel(ABC):
    """
    A generative model that can be used to convert text to embeddings vector.
    """

    @abstractmethod
    def embed(self, text: str) -> list:
        """
        Get the embeddings of the text
        
        Args:
            text (str): The text to embed
            
        Returns:
            list: The embeddings of the text
        
        """
        pass