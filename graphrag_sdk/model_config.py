from graphrag_sdk.models import (GenerativeModel,
                                 GenerativeModelConfig)


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

    @staticmethod
    def with_model(model: GenerativeModel) -> "KnowledgeGraphModelConfig":
        """
        Creates a new KnowledgeGraphModelConfig instance with the given generative model.
        
        The extract_data model will be configured with temperature=0 and JSON response format
        for structured data extraction.

        Args:
            model (GenerativeModel): The generative model to use.

        Returns:
            KnowledgeGraphModelConfig: The new KnowledgeGraphModelConfig instance.

        """        
        # Take user's existing config and modify for extract_data
        config_dict = model.generation_config.to_json()
        
        # Check if temperature was explicitly set by the user
        if model.generation_config._temperature_was_set:
            # User explicitly set temperature (even if to None), only override if not None
            if model.generation_config.temperature is not None:
                config_dict['temperature'] = 0
            # If user explicitly set temperature=None, respect that and don't override
        else:
            # User didn't set temperature (using default None), override to 0
            config_dict['temperature'] = 0
        # Add JSON output format
        config_dict['response_format'] = {"type": "json_object"}
        extract_config = GenerativeModelConfig.from_json(config_dict)
        
        # Create extract_data model with specialized configuration
        try:
            extract_data_model = model.__class__(
                model_name=model.model_name,
                generation_config=extract_config,
                system_instruction=model.system_instruction
            )
        except Exception:
            # Fallback: use original model if creation fails
            extract_data_model = model
        
        return KnowledgeGraphModelConfig(
            extract_data=extract_data_model,
            cypher_generation=model,
            qa=model,
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
