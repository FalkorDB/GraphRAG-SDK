from openai import OpenAI
from .model_embeddings import (
    EmbeddingsGenerativeModel,
)


class OpenAiModelEmbeddings(EmbeddingsGenerativeModel):
    
    client: OpenAI = None

    def __init__(
        self,
        model_name: str,
    ):
        self.model_name = model_name
        self.client = OpenAI()
    
    def embed(self, text: str) -> list:
        """
        Get the embeddings of the text
        
        Args:
            text (str): The text to embed
            
        Returns:
            list: The embeddings of the text
        
        """
        text = text.replace("\n", " ")
        embeddings = self.client.embeddings.create(input = [text], model=self.model_name).data[0].embedding
        return embeddings