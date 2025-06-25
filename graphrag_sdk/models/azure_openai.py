from typing import Any
from typing import Optional
from .litellm import LiteModel, LiteModelChatSession
from .model import (
    GenerativeModel,
    GenerativeModelConfig,
    GenerativeModelChatSession,
)


class AzureOpenAiGenerativeModel(LiteModel):
    """
    A generative model that interfaces with Azure's OpenAI API for chat completions.
    
    Inherits from LiteModel and automatically converts Azure OpenAI model names to 
    LiteLLM format internally (e.g., "gpt-4o" -> "azure/gpt-4o") while 
    exposing the original model name through the public API.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        generation_config: Optional[GenerativeModelConfig] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the AzureOpenAiGenerativeModel with required parameters.

        Args:
            model_name (str): Name of the Azure OpenAI model.
            generation_config (Optional[GenerativeModelConfig]): Configuration settings for generation.
            system_instruction (Optional[str]): System-level instruction for the model.
            kwargs (Any): Additional arguments required by Azure OpenAI API.
        """
        # Convert to LiteLLM format and call parent constructor
        lite_model_name = f"azure/{model_name}"
        super().__init__(
            model_name=lite_model_name,
            generation_config=generation_config,
            system_instruction=system_instruction,
            additional_params=kwargs
        )
        
        # Store original model name for compatibility (without the azure/ prefix)
        self._original_model_name = model_name

    @property
    def model_name(self) -> str:
        """Get the original model name (without azure/ prefix)."""
        return self._original_model_name

    def start_chat(self, system_instruction: Optional[str] = None) -> GenerativeModelChatSession:
        """
        Start a new chat session.
        
        Args:
            system_instruction (Optional[str]): Optional system instruction to guide the chat session.
            
        Returns:
            GenerativeModelChatSession: A new instance of the chat session.
        """
        return AzureOpenAiChatSession(self, system_instruction)

    def to_json(self) -> dict:
        """
        Serialize the model's configuration and state to JSON format.

        Returns:
            dict: The serialized JSON data with clean Azure OpenAI model names.
        """
        return {
            "model_name": self.model_name,  # Return original model name without azure/ prefix
            "generation_config": self.generation_config.to_json(),
            "system_instruction": self.system_instruction,
        }

    @staticmethod
    def from_json(json: dict) -> "GenerativeModel":
        """
        Deserialize a JSON object to create an instance of AzureOpenAiGenerativeModel.

        Args:
            json (dict): The serialized JSON data.

        Returns:
            GenerativeModel: A new instance of AzureOpenAiGenerativeModel.
        """
        return AzureOpenAiGenerativeModel(
            json["model_name"],
            generation_config=GenerativeModelConfig.from_json(
                json["generation_config"]
            ),
            system_instruction=json["system_instruction"],
        )


class AzureOpenAiChatSession(LiteModelChatSession):
    """
    A chat session for interacting with the Azure OpenAI model.
    
    This implementation inherits from LiteModelChatSession to leverage all chat functionality
    without any code duplication. All methods (send_message, send_message_stream, 
    get_chat_history, delete_last_message) are inherited from the parent class.
    """

    def __init__(self, model: AzureOpenAiGenerativeModel, system_instruction: Optional[str] = None) -> None:
        """
        Initialize the chat session and set up the conversation history.
        
        Args:
            model (AzureOpenAiGenerativeModel): The model instance for the session.
            system_instruction (Optional[str]): Optional system instruction.
        """
        super().__init__(model, system_instruction)
