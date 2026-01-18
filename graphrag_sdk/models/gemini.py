<<<<<<< HEAD
import os
from typing import Union

=======
from typing import Optional
from .litellm import LiteModel, LiteModelChatSession
>>>>>>> b2aa07fc70e298ca25ae07c67c1e8af35dd2953b
from .model import (
    GenerativeModel,
    GenerativeModelConfig,
    GenerativeModelChatSession,
)


class GeminiGenerativeModel(LiteModel):
    """
    A generative model that interfaces with GoogleAI API for chat completions.
    
    Inherits from LiteModel and automatically converts Gemini model names to 
    LiteLLM format internally (e.g., "gemini-2.0-flash" -> "gemini/gemini-2.0-flash") while 
    exposing the original model name through the public API.
    """

    def __init__(
        self,
        model_name: str,
<<<<<<< HEAD
        generation_config: Union[GoogleGenerationConfig, None] = None,
        system_instruction: Union[str, None] = None,
    ):
        self._model_name = model_name
        self._generation_config = generation_config
        self._system_instruction = system_instruction
        configure(api_key=os.environ["GOOGLE_API_KEY"])


    def _get_model(self) -> GoogleGenerativeModel:
        if self._model is None:
            self._model = GoogleGenerativeModel(
                self._model_name,
                generation_config=(
                    GoogleGenerationConfig(
                        temperature=self._generation_config.temperature,
                        top_p=self._generation_config.top_p,
                        top_k=self._generation_config.top_k,
                        max_output_tokens=self._generation_config.max_output_tokens,
                        stop_sequences=self._generation_config.stop_sequences,
                    )
                    if self._generation_config is not None
                    else None
                ),
                system_instruction=self._system_instruction,
            )

        return self._model

    def with_system_instruction(self, system_instruction: str) -> "GenerativeModel":
        self._system_instruction = system_instruction
        self._model = None
        self._get_model()

        return self

    def start_chat(self, args: Union[dict, None] = None) -> GenerativeModelChatSession:
        return GeminiChatSession(self, args)

    def parse_generate_content_response(
        self, response: types.generation_types.GenerateContentResponse
    ) -> GenerationResponse:
        return GenerationResponse(
            text=response.text,
            finish_reason=(
                FinishReason.MAX_TOKENS
                if response.candidates[0].finish_reason
                == protos.Candidate.FinishReason.MAX_TOKENS
                else (
                    FinishReason.STOP
                    if response.candidates[0].finish_reason == protos.Candidate.FinishReason.STOP
                    else FinishReason.OTHER
                )
            ),
=======
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
>>>>>>> b2aa07fc70e298ca25ae07c67c1e8af35dd2953b
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


<<<<<<< HEAD
class GeminiChatSession(GenerativeModelChatSession):

    def __init__(self, model: GeminiGenerativeModel, args: Union[dict, None] = None):
        self._model = model
        self._chat_session = self._model._model.start_chat(
            history=args.get("history", []) if args is not None else [],
        )

    def send_message(self, message: str, output_method: OutputMethod = OutputMethod.DEFAULT) -> GenerationResponse:
        generation_config = self._get_generation_config(output_method)
        response = self._chat_session.send_message(message, generation_config=generation_config)
        return self._model.parse_generate_content_response(response)
=======
class GeminiChatSession(LiteModelChatSession):
    """
    A chat session for interacting with the Gemini model.
>>>>>>> b2aa07fc70e298ca25ae07c67c1e8af35dd2953b
    
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
