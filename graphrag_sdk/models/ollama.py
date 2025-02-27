import logging
from typing import Optional
from ollama import Client, Options
from .model import (
    OutputMethod,
    GenerativeModel,
    GenerativeModelConfig,
    GenerationResponse,
    FinishReason,
    GenerativeModelChatSession,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 

class OllamaGenerativeModel(GenerativeModel):
    """
    A generative model that interfaces with the Ollama Client for chat completions.
    """

    client: Client = None

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
            system_instruction (Optional[str]): Instruction to guide the model.
            host (Optional[str]): Host for connecting to the Ollama API.
        """
        self.model_name = model_name
        self.generation_config = generation_config or GenerativeModelConfig()
        self.system_instruction = system_instruction
        self._host = host
        try:
            self.client = Client(host)
        except Exception as e:
            logger.error(f"Failed to initialize the Ollama client: {e}")
            raise e
        self.check_and_pull_model()

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
        response = self.client.list()  # This returns a dictionary
        available_models = [model['name'] for model in response['models']]  # Extract model names

        # Check if the model is already pulled
        if self.model_name in available_models:
            logger.info(f"The model '{self.model_name}' is already available.")
        else:
            logger.info(f"Pulling the model '{self.model_name}'...")
            try:
                self.client.pull(self.model_name)  # Pull the model
                logger.info(f"Model '{self.model_name}' pulled successfully.")
            except Exception as e:
                logger.error(f"Failed to pull the model '{self.model_name}': {e}")
    
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
        Parse the model's response and extract content for the user.

        Args:
            response (any): The raw response from the model.

        Returns:
            GenerationResponse: Parsed response containing the generated text.
        """
        return GenerationResponse(
            text=response["message"]["content"],
            finish_reason=FinishReason.STOP
            )

    def to_json(self) -> dict:
        """
        Serialize the model's configuration and state to JSON format.

        Returns:
            dict: The serialized JSON data.
        """
        return {
            "model_name": self._model_name,
            "generation_config": self._generation_config.to_json(),
            "system_instruction": self._system_instruction,
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
            model_name=json["model_name"],
            generation_config=GenerativeModelConfig.from_json(
                json["generation_config"]
            ),
            system_instruction=json["system_instruction"],
        )

class OllamaChatSession(GenerativeModelChatSession):
    """
    A chat session for interacting with the Ollama model, maintaining conversation history.
    """
    
    def __init__(self, model: OllamaGenerativeModel, system_instruction: Optional[str] = None):
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
    
    def get_chat_history(self) -> list[dict]:
        """
        Retrieve the conversation history for the current chat session.

        Returns:
            list[dict]: The chat session's conversation history.
        """
        return self._chat_history.copy()

    def send_message(self, message: str, output_method: OutputMethod = OutputMethod.DEFAULT) -> GenerationResponse:
        """
        Send a message in the chat session and receive the model's response.

        Args:
            message (str): The message to send.
            output_method (OutputMethod): Format for the model's output.

        Returns:
            GenerationResponse: The generated response.
        """
        generation_config = self._adjust_generation_config(output_method)
        self._chat_history.append({"role": "user", "content": message[:14385]})
        response = self._model.client.chat(
            model=self._model.model_name,
            messages=self._chat_history,
            options=Options(**generation_config)
        )
        content = self._model.parse_generate_content_response(response)
        self._chat_history.append({"role": "assistant", "content": content.text})
        return content
    
    def _adjust_generation_config(self, output_method: OutputMethod) -> dict:
        """
        Adjust the generation configuration based on the specified output method.

        Args:
            output_method (OutputMethod): The desired output method (e.g., default or JSON).

        Returns:
            dict: The adjusted configuration settings for generation.
        """
        config = self._model.generation_config.to_json()
        if output_method == OutputMethod.JSON:
            config['temperature'] = 0
            config['format'] = 'json'
        
        return config
    
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