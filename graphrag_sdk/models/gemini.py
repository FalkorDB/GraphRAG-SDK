from typing import Optional
from .litellm import LiteModel, LiteModelChatSession
from .model import (
    GenerativeModel,
    GenerativeModelConfig,
    GenerativeModelChatSession,
)


class GeminiGenerativeModel(LiteModel):
    """
    A generative model that interfaces with GoogleAI API for chat completions.
    
    Inherits from LiteModel and automatically converts Gemini model names to 
    LiteLLM format internally (e.g., "gemini-1.5-pro" -> "gemini/gemini-1.5-pro") while 
    exposing the original model name through the public API.
    """

    def __init__(
        self,
        model_name: str,
        generation_config: Optional[GenerativeModelConfig] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize the GeminiGenerativeModel with required parameters.
        
        Args:
            model_name (str): Name of the Gemini model.
            generation_config (Optional[GenerativeModelConfig]): Configuration settings for generation.
            system_instruction (Optional[str]): System-level instruction for the model.
        """
        # Convert to LiteLLM format and call parent constructor
        lite_model_name = f"gemini/{model_name}"
        super().__init__(
            model_name=lite_model_name,
            generation_config=generation_config,
            system_instruction=system_instruction
        )
        
        # Store original model name for compatibility (without the gemini/ prefix)
        self._original_model_name = model_name

    @property
    def model_name(self) -> str:
        """Get the original model name (without gemini/ prefix)."""
        return self._original_model_name

    def start_chat(self, system_instruction: Optional[str] = None) -> GenerativeModelChatSession:
        """
        Start a new chat session.
        
        Args:
            system_instruction (Optional[str]): Optional system instruction to guide the chat session.
            
        Returns:
            GenerativeModelChatSession: A new instance of the chat session.
        """
        return GeminiChatSession(self, system_instruction)

    def to_json(self) -> dict:
        """
        Serialize the model's configuration and state to JSON format.
        
        Returns:
            dict: The serialized JSON data with clean Gemini model names.
        """
        return {
            "model_name": self.model_name,  # Return original model name without gemini/ prefix
            "generation_config": self.generation_config.to_json(),
            "system_instruction": self.system_instruction,
        }

    @staticmethod
    def from_json(json: dict) -> "GenerativeModel":
        """
        Deserialize a JSON object to create an instance of GeminiGenerativeModel.
        
        Args:
            json (dict): The serialized JSON data.
            
        Returns:
            GenerativeModel: A new instance of GeminiGenerativeModel.
        """
        return GeminiGenerativeModel(
            json["model_name"],
            generation_config=GenerativeModelConfig.from_json(
                json["generation_config"]
            ),
            system_instruction=json["system_instruction"],
        )


class GeminiChatSession(LiteModelChatSession):
    """
    A chat session for interacting with the Gemini model.
    
    This implementation inherits from LiteModelChatSession to leverage all chat functionality
    without any code duplication. All methods (send_message, send_message_stream, 
    get_chat_history, delete_last_message) are inherited from the parent class.
    """
    
    def __init__(self, model: GeminiGenerativeModel, system_instruction: Optional[str] = None) -> None:
        """
        Initialize the chat session and set up the conversation history.
        
        Args:
            model (GeminiGenerativeModel): The model instance for the session.
            system_instruction (Optional[str]): Optional system instruction.
        """
        super().__init__(model, system_instruction)
