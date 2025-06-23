import logging
from typing import Optional
from .litellm import LiteModel
from .model import (
    GenerativeModel,
    GenerativeModelConfig,
    GenerationResponse,
    GenerativeModelChatSession,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 

class OllamaGenerativeModel(GenerativeModel):
    """
    A generative model that interfaces with the Ollama Client for chat completions.
    This implementation uses LiteLLM as the backend while maintaining the original API.
    """

    def __init__(
        self,
        model_name: str,
        generation_config: Optional[GenerativeModelConfig] = GenerativeModelConfig(),
        system_instruction: Optional[str] = None,
        host: Optional[str] = None,
    ):
        """
        Initialize the OllamaGenerativeModel with the required parameters.

        Args:
            model_name (str): The name of the Ollama model.
            generation_config (Optional[GenerativeModelConfig]): Configuration settings for generation.
            system_instruction (Optional[str]): Instruction to guide the model.
            host (Optional[str]): Host for connecting to the Ollama API (ignored in LiteLLM implementation).
        """
        # Convert to LiteLLM format
        lite_model_name = f"ollama/{model_name}"
        
        # Handle host parameter for Ollama
        additional_params = {}
        if host is not None:
            additional_params['api_base'] = host  # LiteLLM uses api_base for custom endpoints
        
        # Create internal LiteLLM model
        self._lite_model = LiteModel(
            model_name=lite_model_name,
            generation_config=generation_config,
            system_instruction=system_instruction,
            additional_params=additional_params
        )
        
        # Store original model name and host for compatibility
        self._original_model_name = model_name
        self._host = host

    @property
    def model_name(self) -> str:
        """Get the original model name (without ollama/ prefix)."""
        return self._original_model_name
    
    @property
    def system_instruction(self) -> Optional[str]:
        """Get the system instruction from the internal LiteLLM model."""
        return self._lite_model.system_instruction
    
    @property 
    def generation_config(self) -> GenerativeModelConfig:
        """Get the generation config from the internal LiteLLM model."""
        return self._lite_model.generation_config

    def start_chat(self, system_instruction: Optional[str] = None) -> GenerativeModelChatSession:
        """
        Start a new chat session.
        
        Args:
            system_instruction (Optional[str]): Optional system instruction to guide the chat session.
            
        Returns:
            GenerativeModelChatSession: A new instance of the chat session.
        """
        return OllamaChatSession(self, system_instruction)

    def parse_generate_content_response(self, response: any) -> GenerationResponse:
        """
        Parse the model's response using the internal LiteLLM model.
        """
        return self._lite_model.parse_generate_content_response(response)

    def to_json(self) -> dict:
        """
        Serialize the model's configuration and state to JSON format.

        Returns:
            dict: The serialized JSON data.
        """
        return {
            "model_name": self.model_name,
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
            GenerativeModel: A new instance of the model.
        """
        return OllamaGenerativeModel(
            json["model_name"],
            generation_config=GenerativeModelConfig.from_json(
                json["generation_config"]
            ),
            system_instruction=json["system_instruction"],
        )


class OllamaChatSession(GenerativeModelChatSession):
    """
    A chat session for interacting with the Ollama model.
    This implementation delegates to LiteLLM for actual API calls.
    """

    def __init__(self, model: OllamaGenerativeModel, system_instruction: Optional[str] = None):
        """
        Initialize the chat session and set up the conversation history.

        Args:
            model (OllamaGenerativeModel): The model instance for the session.
            system_instruction (Optional[str]): Optional system instruction.
        """
        self._model = model
        # Create internal LiteLLM chat session
        self._lite_chat = model._lite_model.start_chat(system_instruction)

    def send_message(self, message: str) -> GenerationResponse:
        """
        Send a message in the chat session and receive the model's response.

        Args:
            message (str): The message to send.

        Returns:
            GenerationResponse: The generated response.
        """
        # Delegate to LiteLLM chat session
        return self._lite_chat.send_message(message)
    
    def send_message_stream(self, message: str):
        """
        Send a message and receive the response in a streaming fashion.
        Args:
            message (str): The message to send.
        Yields:
            str: Streamed chunks of the model's response.
        """
        # Delegate to LiteLLM chat session
        return self._lite_chat.send_message_stream(message)
    
    def get_chat_history(self) -> list[dict]:
        """
        Retrieve the conversation history for the current chat session.

        Returns:
            list[dict]: The chat session's conversation history.
        """
        return self._lite_chat.get_chat_history()
    
    def delete_last_message(self):
        """
        Deletes the last message exchange (user message and assistant response) from the chat history.
        """
        self._lite_chat.delete_last_message()
