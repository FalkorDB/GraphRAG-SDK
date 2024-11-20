import os
from typing import Optional
from openai import AzureOpenAI
from .model import (
    OutputMethod,
    GenerativeModel,
    GenerativeModelConfig,
    GenerationResponse,
    FinishReason,
    GenerativeModelChatSession,
)

class AzureOpenAiGenerativeModel(GenerativeModel):
    """
    A generative model that interfaces with Azure's OpenAI API for chat completions.
    """

    client: AzureOpenAI = None

    def __init__(
        self,
        model_name: str,
        generation_config: Optional[GenerativeModelConfig] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize the AzureOpenAiGenerativeModel with required parameters.

        Args:
            model_name (str): Name of the Azure OpenAI model.
            generation_config (Optional[GenerativeModelConfig]): Configuration settings for generation.
            system_instruction (Optional[str]): System-level instruction for the model.
        """
        self.model_name = model_name
        self.generation_config = generation_config or GenerativeModelConfig()
        self.system_instruction = system_instruction
        
        # Credentials
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.api_version = os.getenv("AZURE_API_VERSION")
        
        if not self.api_key or not self.azure_endpoint or not self.api_version:
            raise ValueError(
                "Missing credentials in the environment: AZURE_OPENAI_API_KEY, AZURE_ENDPOINT, or AZURE_API_VERSION."
            )


    def _connect_to_model(self) -> None:
        """
        Establish a connection to the Azure OpenAI model by initializing the AzureOpenAI client.
        """
        self.client = AzureOpenAI(azure_endpoint=self.azure_endpoint,
                                api_version=self.api_version,
                                api_key=self.api_key,
                                )

    def with_system_instruction(self, system_instruction: str) -> "GenerativeModel":
        """
        Set or update the system instruction and connect to the Azure OpenAI model.

        Args:
            system_instruction (str): System instructions for the model.
        
        Returns:
            GenerativeModel: The updated model instance.
        """
        self.system_instruction = system_instruction
        self._connect_to_model()

        return self

    def start_chat(self, args: Optional[dict] = None) -> GenerativeModelChatSession:
        """
        Start a new chat session.

        Args:
            args (Optional[dict]): Additional arguments for the chat session.

        Returns:
            GenerativeModelChatSession: A new instance of the chat session.
        """
        return AzureOpenAiChatSession(self, args)

    def ask(self, message: str) -> GenerationResponse:
        """
        Send a message to the model and receive a response.

        Args:
            message (str): The user's message input.

        Returns:
            GenerationResponse: The model's generated response.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_instruction},
                {"role": "user", "content": message[:14385]},
            ],
            max_tokens=self.generation_config.max_output_tokens,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            top_k=self.generation_config.top_k,
            stop=self.generation_config.stop_sequences,
        )
        return self._parse_generate_content_response(response)

    def _parse_generate_content_response(self, response: any) -> GenerationResponse:
        """
        Parse the model's response and extract content for the user.

        Args:
            response (any): The raw response from the model.

        Returns:
            GenerationResponse: Parsed response containing the generated text and finish reason.
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
        Deserialize a JSON object to create an instance of AzureOpenAiGenerativeModel.

        Args:
            json (dict): The serialized JSON data.

        Returns:
            GenerativeModel: A new instance of the model.
        """
        return AzureOpenAiGenerativeModel(
            json["model_name"],
            generation_config=GenerativeModelConfig.from_json(
                json["generation_config"]
            ),
            system_instruction=json["system_instruction"],
        )


class AzureOpenAiChatSession(GenerativeModelChatSession):
    """
    A chat session for interacting with the Azure OpenAI model, maintaining conversation history.
    """


    _history = []

    def __init__(self, model: AzureOpenAiGenerativeModel, args: Optional[dict] = None):
        """
        Initialize the chat session and set up the conversation history.

        Args:
            model (AzureOpenAiGenerativeModel): The model instance for the session.
            args (Optional[dict]): Additional arguments for customization.
        """
        self._model = model
        self._args = args
        self._history = (
            [{"role": "system", "content": self._model.system_instruction}]
            if self._model.system_instruction is not None
            else []
        )

    def send_message(self, message: str, output_method: OutputMethod = OutputMethod.DEFAULT) -> GenerationResponse:
        """
        Send a message in the chat session and receive the model's response.

        Args:
            message (str): The message to send.
            output_method (OutputMethod): Format for the model's output.

        Returns:
            GenerationResponse: The generated response.
        """
        generation_config = self._get_generation_config(output_method)
        prompt = []
        prompt.extend(self._history)
        prompt.append({"role": "user", "content": message[:14385]})
        response = self._model.client.chat.completions.create(
            model=self._model.model_name,
            messages=prompt,
            **generation_config
        )
        content = self._model._parse_generate_content_response(response)
        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": content.text})
        return content
    
    def _get_generation_config(self, output_method: OutputMethod):
        """
        Adjust the generation configuration based on the output method.

        Args:
            output_method (OutputMethod): The desired output method (e.g., default or JSON).

        Returns:
            dict: The configuration settings for generation.
        """
        config = self._model.generation_config.to_json()
        if output_method == OutputMethod.JSON:
            config['temperature'] = 0
            config['response_format'] = { "type": "json_object" }
        
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
        if len(self._history) - 2 >= min_length:
            self._history.pop()
            self._history.pop()
        else:
            # Reset to initial state with just system message if present
            self._history = (
            [{"role": "system", "content": self._model.system_instruction}]
            if self._model.system_instruction is not None
            else []
        )
