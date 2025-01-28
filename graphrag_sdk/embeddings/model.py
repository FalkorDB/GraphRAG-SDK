from abc import ABC, abstractmethod

class EmbeddingModel(ABC):
    """
    A generative model that can be used to convert text to embeddings vector.
    """

    @abstractmethod
    def get_embedding(self, text: str) -> list:
        """
        Get the embeddings of the text
        
        Args:
            text (str): The text to embed
            
        Returns:
            list: The embeddings of the text
        
        """
        pass