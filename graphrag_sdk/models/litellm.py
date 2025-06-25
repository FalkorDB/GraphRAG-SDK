import logging
from typing import Optional, Iterator
from litellm import completion, validate_environment, utils as litellm_utils

# Optional import for Ollama
try:
    from ollama import Client as OllamaClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OllamaClient = None
    OLLAMA_AVAILABLE = False

from .model import (
    GenerativeModel,
    GenerativeModelConfig,
    GenerationResponse,
    FinishReason,
    GenerativeModelChatSession,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 

class LiteModel(GenerativeModel):
    """
    A generative model that interfaces with the LiteLLM for chat completions.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-4.1",
        generation_config: Optional[GenerativeModelConfig] = None,
        system_instruction: Optional[str] = None,
        additional_params: Optional[dict] = None,
    ):
        """
        Initialize the LiteModel with the required parameters.
        
        LiteLLM model_name format: <provider>/<model_name>
         Examples:
         - openai/gpt-4.1
         - azure/gpt-4.1
         - gemini/gemini-1.5-pro
         - ollama/llama3:8b

        Args:
            model_name (str): The name and the provider for the LiteLLM client. Defaults to "openai/gpt-4.1".
            generation_config (Optional[GenerativeModelConfig]): Configuration settings for generation.
            system_instruction (Optional[str]): Instruction to guide the model.
            additional_params (Optional[dict]): Additional provider-specific parameters.
        """

        env_val = validate_environment(model_name)
        if not env_val['keys_in_environment']:
            raise ValueError(f"Missing {env_val['missing_keys']} in the environment.")
        self._internal_model_name, provider, _, _ = litellm_utils.get_llm_provider(model_name)
        self.model = model_name
        
        if provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ValueError("Ollama client not available. Install with: pip install ollama")
            self.ollama_client = OllamaClient()
            self.check_and_pull_model()
        if not self.check_valid_key(model_name):
            raise ValueError(f"Invalid keys for model {model_name}.")
        
        if self._internal_model_name == "gpt-4.1":
            # Set default temperature to 0 for gpt-4.1
            if generation_config is None:
                generation_config = GenerativeModelConfig(temperature=0)
        self.generation_config = generation_config or GenerativeModelConfig()

        self.system_instruction = system_instruction
        self.additional_params = additional_params or {}
        
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._internal_model_name
        
    def check_valid_key(self, model: str):
        """
        Checks if the environment key is valid for a specific model by making a litellm.completion call with max_tokens=10

        Args:
            model (str): The name of the model to check the key against.

        Returns:
            bool: True if the key is valid for the model, False otherwise.
        """
        messages = [{"role": "user", "content": "Hey, how's it going?"}]
        try:
            completion(
                model=model, messages=messages, max_completion_tokens=10
            )
            return True
        except Exception as e:
            return False
            
    def check_and_pull_model(self) -> None:
        """
        Checks if the specified model is available locally, and pulls it if not.

        Logs:
            - Info: If the model is already available or after successfully pulling the model.
            - Error: If there is a failure in pulling the model.

        Raises:
            Exception: If there is an error during the model pull process.
        """
        # Get the list of available models
        response = self.ollama_client.list()  # This returns a dictionary
        available_models = [model['name'] for model in response['models']]  # Extract model names

        # Check if the model is already pulled
        if self._internal_model_name in available_models:
            logger.info(f"The model '{self._internal_model_name}' is already available.")
        else:
            logger.info(f"Pulling the model '{self._internal_model_name}'...")
            try:
                self.ollama_client.pull(self._internal_model_name)  # Pull the model
                logger.info(f"Model '{self._internal_model_name}' pulled successfully.")
            except Exception as e:
                logger.error(f"Failed to pull the model '{self._internal_model_name}': {e}")
                raise ValueError(f"Failed to pull the model '{self._internal_model_name}': {e}")

    def start_chat(self, system_instruction: Optional[str] = None) -> GenerativeModelChatSession:
        """
        Start a new chat session.
        
        Args:
            system_instruction (Optional[str]): Optional system instruction to guide the chat session.
            
        Returns:
            GenerativeModelChatSession: A new instance of the chat session.
        """
        return LiteModelChatSession(self, system_instruction)

    def parse_generate_content_response(self, response: any) -> GenerationResponse:
        """
        Parse the model's response and extract content for the user.

        Args:
            response (any): The raw response from the model.

        Returns:
            GenerationResponse: Parsed response containing the generated text.
        """
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
        )

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
        Deserialize a JSON object to create an instance of LiteLLMGenerativeModel.

        Args:
            json (dict): The serialized JSON data.

        Returns:
            GenerativeModel: A new instance of the model.
        """
        return LiteModel(
            json["model_name"],
            generation_config=GenerativeModelConfig.from_json(
                json["generation_config"]
            ),
            system_instruction=json["system_instruction"],
        )


class LiteModelChatSession(GenerativeModelChatSession):
    """
    A chat session for interacting with the LiteLLM model, maintaining conversation history.
    """

    def __init__(self, model: LiteModel, system_instruction: Optional[str] = None):
        """
        Initialize the chat session and set up the conversation history.

        Args:
            model (OllamaGenerativeModel): The model instance for the session.
            system_instruction (Optional[str]): Optional system instruction
        """
        self._model = model
        self._chat_history = (
            [{"role": "system", "content": system_instruction}]
            if system_instruction is not None
            else []
        )

    def send_message(self, message: str) -> GenerationResponse:
        """
        Send a message in the chat session and receive the model's response.

        Args:
            message (str): The message to send.

        Returns:
            GenerationResponse: The generated response.
        """
        self._chat_history.append({"role": "user", "content": message})
        try:
            response = completion(
                model=self._model.model,
                messages=self._chat_history,
                **self._model.generation_config.to_json(),
                **self._model.additional_params
            )
        except Exception as e:
            raise ValueError(f"Error during completion request, please check the credentials - {e}")
        content = self._model.parse_generate_content_response(response)
        self._chat_history.append({"role": "assistant", "content": content.text})
        return content
    
    def send_message_stream(self, message: str) -> Iterator[str]:
        """
        Send a message and receive the response in a streaming fashion.

        Args:
            message (str): The message to send.

        Yields:
            str: Streamed chunks of the model's response.
        """
        self._chat_history.append({"role": "user", "content": message})

        try:
            response_stream = completion(
                model=self._model.model,
                messages=self._chat_history,
                stream=True,  # Enable streaming mode
                **self._model.generation_config.to_json(),
                **self._model.additional_params
            )
            
            chunks = []
            for chunk in response_stream:
                if not chunk or "choices" not in chunk or not chunk["choices"]:
                    continue  # Skip empty or malformed chunks

                content = chunk["choices"][0].get("delta", {}).get("content", "")
                if content:
                    chunks.append(content)
                    yield content  # Yield streamed response chunks
        
            # Save the final response to chat history
            full_response = "".join(chunks)  # Collect full response
            self._chat_history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            raise ValueError(f"Error during streaming request, check credentials - {e}") from e
    

    def get_chat_history(self) -> list[dict]:
        """
        Retrieve the conversation history for the current chat session.

        Returns:
            list[dict]: The chat session's conversation history.
        """
        return self._chat_history.copy()

    def delete_last_message(self):
        """
        Deletes the last message exchange (user message and assistant response) from the chat history.
        Preserves the system message if present.
        
        Example:
            Before:
            [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "Assistant response"},
            ]
            After:
            [
                {"role": "system", "content": "System message"},
            ]

        Note: Does nothing if the chat history is empty or contains only a system message.
        """
        # Keep at least the system message if present
        min_length = 1 if self._model.system_instruction else 0
        if len(self._chat_history) - 2 >= min_length:
            self._chat_history.pop()
            self._chat_history.pop()
        else:
            # Reset to initial state with just system message if present
            self._chat_history = (
            [{"role": "system", "content": self._model.system_instruction}]
            if self._model.system_instruction is not None
            else []
        )
