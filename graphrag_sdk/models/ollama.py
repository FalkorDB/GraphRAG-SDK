import logging
from typing import Optional
from .litellm import LiteModel, LiteModelChatSession
from .model import (
    GenerativeModel,
    GenerativeModelConfig,
    GenerativeModelChatSession,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 

class OllamaGenerativeModel(LiteModel):
    """
    A generative model that interfaces with the Ollama Client for chat completions.
    
    Inherits from LiteModel and automatically converts Ollama model names to 
    LiteLLM format internally (e.g., "llama3:8b" -> "ollama/llama3:8b") while 
    exposing the original model name through the public API.
    """

    def __init__(
        self,
        model_name: str,
        generation_config: Optional[GenerativeModelConfig] = None,
        system_instruction: Optional[str] = None,
        host: Optional[str] = None,
    ):
        """
        Initialize the OllamaGenerativeModel with the required parameters.

        Args:
            model_name (str): The name of the Ollama model.
            generation_config (Optional[GenerativeModelConfig]): Configuration settings for generation.
            system_instruction (Optional[str]): System-level instruction for the model.
            host (Optional[str]): Host for connecting to the Ollama API.
        """
        # Convert to LiteLLM format
        lite_model_name = f"ollama/{model_name}"
        
        # Handle host parameter for Ollama
        additional_params = {}
        if host is not None:
            additional_params['api_base'] = host  # LiteLLM uses api_base for custom endpoints
        
        # Call parent constructor
        super().__init__(
            model_name=lite_model_name,
            generation_config=generation_config,
            system_instruction=system_instruction,
            additional_params=additional_params
        )
        
        # Store original model name for compatibility (without the ollama/ prefix)
        self._original_model_name = model_name

    @property
    def model_name(self) -> str:
        """Get the original model name (without ollama/ prefix)."""
        return self._original_model_name

    def start_chat(self, system_instruction: Optional[str] = None) -> GenerativeModelChatSession:
        """
        Start a new chat session.
        
        Args:
            system_instruction (Optional[str]): Optional system instruction to guide the chat session.
            
        Returns:
            GenerativeModelChatSession: A new instance of the chat session.
        """
        return OllamaChatSession(self, system_instruction)

    def to_json(self) -> dict:
        """
        Serialize the model's configuration and state to JSON format.

        Returns:
            dict: The serialized JSON data with clean Ollama model names.
        """
        return {
            "model_name": self.model_name,  # Return original model name without ollama/ prefix
            "generation_config": self.generation_config.to_json(),
            "system_instruction": self.system_instruction,
        }

    @staticmethod
    def from_json(json: dict) -> "GenerativeModel":
        """
        Deserialize a JSON object to create an instance of OllamaGenerativeModel.

        Args:
            json (dict): The serialized JSON data.

        Returns:
            GenerativeModel: A new instance of OllamaGenerativeModel.
        """
        return OllamaGenerativeModel(
            json["model_name"],
            generation_config=GenerativeModelConfig.from_json(
                json["generation_config"]
            ),
            system_instruction=json["system_instruction"],
        )


class OllamaChatSession(LiteModelChatSession):
    """
    A chat session for interacting with the Ollama model.
    
    This implementation inherits from LiteModelChatSession to leverage all chat functionality
    without any code duplication. All methods (send_message, send_message_stream, 
    get_chat_history, delete_last_message) are inherited from the parent class.
    """

    def __init__(self, model: OllamaGenerativeModel, system_instruction: Optional[str] = None) -> None:
        """
        Initialize the chat session and set up the conversation history.

        Args:
            model (OllamaGenerativeModel): The model instance for the session.
            system_instruction (Optional[str]): Optional system instruction.
        """
        super().__init__(model, system_instruction)
