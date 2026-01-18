from typing import Optional
from .litellm import LiteModel, LiteModelChatSession
from .model import (
    GenerativeModel,
    GenerativeModelConfig,
    GenerativeModelChatSession,
)
<<<<<<< HEAD
from typing import Union
from openai import OpenAI
=======
>>>>>>> b2aa07fc70e298ca25ae07c67c1e8af35dd2953b


class OpenAiGenerativeModel(LiteModel):
    """
    A generative model that interfaces with OpenAI's API for chat completions.
    
    Inherits from LiteModel and automatically converts OpenAI model names to 
    LiteLLM format internally (e.g., "gpt-4o" -> "openai/gpt-4o") while 
    exposing the original model name through the public API.
    """

    def __init__(
        self,
<<<<<<< HEAD
        model_name: str,
        generation_config: Union[GenerativeModelConfig, None] = None,
        system_instruction: Union[str, None] = None,
    ):
        self.model_name = model_name
        self.generation_config = generation_config or GenerativeModelConfig()
        self.system_instruction = system_instruction

    def _get_model(self) -> OpenAI:
        if self.client is None:
            self.client = OpenAI()

        return self.client

    def with_system_instruction(self, system_instruction: str) -> "GenerativeModel":
        self.system_instruction = system_instruction
        self.client = None
        self._get_model()

        return self

    def start_chat(self, args: Union[dict, None] = None) -> GenerativeModelChatSession:
        return OpenAiChatSession(self, args)

    def parse_generate_content_response(self, response: any) -> GenerationResponse:
        return GenerationResponse(
            text=response.choices[0].message.content,
            finish_reason=(
                FinishReason.STOP
                if response.choices[0].finish_reason == "stop"
                else (
                    FinishReason.MAX_TOKENS
                    if response.choices[0].finish_reason == "length"
                    else FinishReason.OTHER
                )
            ),
=======
        model_name: str = "gpt-4.1",
        generation_config: Optional[GenerativeModelConfig] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize the OpenAiGenerativeModel with required parameters.
        
        Args:
            model_name (str): Name of the OpenAI model.
            generation_config (Optional[GenerativeModelConfig]): Configuration settings for generation.
            system_instruction (Optional[str]): System-level instruction for the model.
        """
        # Convert to LiteLLM format and call parent constructor
        lite_model_name = f"openai/{model_name}"
        super().__init__(
            model_name=lite_model_name,
            generation_config=generation_config,
            system_instruction=system_instruction
>>>>>>> b2aa07fc70e298ca25ae07c67c1e8af35dd2953b
        )
        
        # Store original model name for compatibility (without the openai/ prefix)
        self._original_model_name = model_name

    @property
    def model_name(self) -> str:
        """Get the original model name (without openai/ prefix)."""
        return self._original_model_name

    def start_chat(self, system_instruction: Optional[str] = None) -> GenerativeModelChatSession:
        """
        Start a new chat session.
            
        Args:
            system_instruction (Optional[str]): Optional system instruction to guide the chat session.
        Returns:
            GenerativeModelChatSession: A new instance of the chat session.
        """
        return OpenAiChatSession(self, system_instruction)

    def to_json(self) -> dict:
        """
        Serialize the model's configuration and state to JSON format.
        
        Returns:
            dict: The serialized JSON data with clean OpenAI model names.
        """
        return {
            "model_name": self.model_name,  # Return original model name without openai/ prefix
            "generation_config": self.generation_config.to_json(),
            "system_instruction": self.system_instruction,
        }

    @staticmethod
    def from_json(json: dict) -> "GenerativeModel":
        """
        Deserialize a JSON object to create an instance of OpenAiGenerativeModel.
        
        Args:
            json (dict): The serialized JSON data.
            
        Returns:
            GenerativeModel: A new instance of OpenAiGenerativeModel.
        """
        return OpenAiGenerativeModel(
            json["model_name"],
            generation_config=GenerativeModelConfig.from_json(
                json["generation_config"]
            ),
            system_instruction=json["system_instruction"],
        )


<<<<<<< HEAD
class OpenAiChatSession(GenerativeModelChatSession):

    _history = []

    def __init__(self, model: OpenAiGenerativeModel, args: Union[dict, None] = None):
        self._model = model
        self._args = args
        self._history = (
            [{"role": "system", "content": self._model.system_instruction}]
            if self._model.system_instruction is not None
            else []
        )

    def send_message(self, message: str, output_method: OutputMethod = OutputMethod.DEFAULT) -> GenerationResponse:
        generation_config = self._get_generation_config(output_method)
        prompt = []
        prompt.extend(self._history)
        prompt.append({"role": "user", "content": message[:14385]})
        response = self._model.client.chat.completions.create(
            model=self._model.model_name,
            messages=prompt,
            **generation_config
        )
        content = self._model.parse_generate_content_response(response)
        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": content.text})
        return content
=======
class OpenAiChatSession(LiteModelChatSession):
    """
    A chat session for interacting with the OpenAI model.
>>>>>>> b2aa07fc70e298ca25ae07c67c1e8af35dd2953b
    
    This implementation inherits from LiteModelChatSession to leverage all chat functionality
    without any code duplication. All methods (send_message, send_message_stream, 
    get_chat_history, delete_last_message) are inherited from the parent class.
    """
    
    def __init__(self, model: OpenAiGenerativeModel, system_instruction: Optional[str] = None) -> None:
        """
        Initialize the chat session and set up the conversation history.
        
        Args:
            model (OpenAiGenerativeModel): The model instance for the session.
            system_instruction (Optional[str]): Optional system instruction.
        """
        super().__init__(model, system_instruction)
