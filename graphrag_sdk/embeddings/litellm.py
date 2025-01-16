from litellm import embedding, validate_environment
from .model import (
    EmbeddingModel,
)


class LiteModelEmbeddings(EmbeddingModel):
    

    def __init__(
        self,
        model_name: str,
    ):
        self.model_name = model_name
        env_val = validate_environment(model_name)
        if not env_val['keys_in_environment']:
            raise ValueError(f"Missing {env_val['missing_keys']} in the environment.")
    
    def get_embedding(self, text: str) -> list:
        """
        Get the embeddings of the text
        
        Args:
            text (str): The text to embed
            
        Returns:
            list: The embeddings of the text
        
        """
        text = text.replace("\n", " ")
        embeddings = embedding(input = [text], model=self.model_name)
        return embeddings