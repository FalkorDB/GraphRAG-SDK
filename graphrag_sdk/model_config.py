from graphrag_sdk.models import GenerativeModel
from graphrag_sdk.embeddings import EmbeddingsGenerativeModel


class KnowledgeGraphModelConfig:
    """
    Represents the configuration for a knowledge graph model.

    Args:
        extract_data (GenerativeModel): The generative model for extracting data.
        cypher_generation (GenerativeModel): The generative model for cypher generation.
        qa (GenerativeModel): The generative model for question answering.

    """

    def __init__(
        self,
        extract_data: GenerativeModel,
        cypher_generation: GenerativeModel,
        qa: GenerativeModel,
        embeddings: EmbeddingsGenerativeModel = None,
    ):
        """
        Initialize a KnowledgeGraphModelConfig object.

        Args:
            extract_data (GenerativeModel): The generative model for data extraction.
            cypher_generation (GenerativeModel): The generative model for Cypher query generation.
            qa (GenerativeModel): The generative model for question answering.
        """
        self.extract_data = extract_data
        self.cypher_generation = cypher_generation
        self.qa = qa
        self.embeddings = embeddings

    @staticmethod
    def with_model(model: GenerativeModel, model_embedding: EmbeddingsGenerativeModel = None) -> "KnowledgeGraphModelConfig":
        """
        Creates a new KnowledgeGraphModelConfig instance with the given generative model.

        Args:
            model (GenerativeModel): The generative model to use.

        Returns:
            KnowledgeGraphModelConfig: The new KnowledgeGraphModelConfig instance.

        """
        return KnowledgeGraphModelConfig(
            extract_data=model,
            cypher_generation=model,
            qa=model,
            embeddings=model_embedding,
            
        )
    
    @staticmethod
    def from_json(json: dict) -> "KnowledgeGraphModelConfig":
        """
        Creates a new KnowledgeGraphModelConfig instance from a JSON dictionary.

        Args:
            json (dict): The JSON dictionary representing the model configuration.

        Returns:
            KnowledgeGraphModelConfig: The new KnowledgeGraphModelConfig instance.

        """
        return KnowledgeGraphModelConfig(
            GenerativeModel.from_json(json["extract_data"]),
            GenerativeModel.from_json(json["cypher_generation"]),
            GenerativeModel.from_json(json["qa"]),
        )
    
    def to_json(self) -> dict:
        """
        Converts the model configuration to a JSON dictionary.

        Returns:
            dict: The JSON dictionary representing the model configuration.

        """
        return {
            "extract_data": self.extract_data.to_json(),
            "cypher_generation": self.cypher_generation.to_json(),
            "qa": self.qa.to_json(),
        }
