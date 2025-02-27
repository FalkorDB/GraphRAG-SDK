import os
from openai import OpenAI
from typing import Optional
from .model import (
    OutputMethod,
    GenerativeModel,
    GenerativeModelConfig,
    GenerationResponse,
    FinishReason,
    GenerativeModelChatSession,
)


class OpenAiGenerativeModel(GenerativeModel):
    """
    A generative model that interfaces with OpenAI's API for chat completions.
    """

    client: OpenAI = None

    def __init__(
        self,
        model_name: str,
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
        self.model_name = model_name
        self.generation_config = generation_config or GenerativeModelConfig()
        self.system_instruction = system_instruction
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Missing OpenAI API key in the environment: OPENAI_API_KEY")
        self.client = OpenAI()

    def start_chat(self, system_instruction: Optional[str] = None) -> GenerativeModelChatSession:
        """
        Start a new chat session.
            
        Args:
            system_instruction (Optional[str]): Optional system instruction to guide the chat session.
        Returns:
            GenerativeModelChatSession: A new instance of the chat session.
        """
        return OpenAiChatSession(self, system_instruction)

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
        Deserialize a JSON object to create an instance of OpenAiGenerativeModel.
        
        Args:
            json (dict): The serialized JSON data.
            
        Returns:
            GenerativeModel: A new instance of the model.
        """
        return OpenAiGenerativeModel(
            json["model_name"],
            generation_config=GenerativeModelConfig.from_json(
                json["generation_config"]
            ),
            system_instruction=json["system_instruction"],
        )


class OpenAiChatSession(GenerativeModelChatSession):
    """
    A chat session for interacting with the OpenAI model, maintaining conversation history.
    """
    
    def __init__(self, model: OpenAiGenerativeModel, system_instruction: Optional[str] = None) -> None:
        """
        Initialize the chat session and set up the conversation history.
        Args:
            model (OpenAiGenerativeModel): The model instance for the session.
            system_instruction (Optional[str]): Optional system instruction.
        """
        self._model = model
        self._chat_history = (
            [{"role": "system", "content": system_instruction}]
            if system_instruction is not None
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
        generation_config = self._adjust_generation_config(output_method)
        self._chat_history.append({"role": "user", "content": message[:14385]})
        response = self._model.client.chat.completions.create(
            model=self._model.model_name,
            messages=self._chat_history,
            **generation_config
        )
        content = self._model.parse_generate_content_response(response)
        self._chat_history.append({"role": "assistant", "content": content.text})
        return content
    
    def get_chat_history(self) -> list[dict]:
        """
        Retrieve the conversation history for the current chat session.
        
        Returns:
            list[dict]: The chat session's conversation history.
        """
        return self._chat_history.copy()
    
    def _adjust_generation_config(self, output_method: OutputMethod):
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
