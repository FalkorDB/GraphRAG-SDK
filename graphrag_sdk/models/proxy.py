import logging
import requests
from typing import Optional
from graphrag_sdk import GenerativeModelConfig
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

class ProxyLLMModel(GenerativeModel):
    """
    A generative model that interfaces with a proxy LLM for chat completions.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        internal_subscription_key: Optional[str] = None,
        generation_config: Optional[GenerativeModelConfig] = None,
        system_instruction: Optional[str] = None,
        auth_method: str = "bearer",
    ):
        """
        Initialize the ProxyLLMModel with the required parameters.
        
        Args:
            base_url (str): The base URL for the proxy LLM.
            model_name (str): The model name to use.
            api_key (Optional[str]): API key for authentication.
            internal_subscription_key (Optional[str]): Internal subscription key for authentication.
            generation_config (Optional[GenerativeModelConfig]): Configuration settings for generation.
            system_instruction (Optional[str]): Instruction to guide the model.
            auth_method (str): Authentication method - "bearer", "api-key", "subscription-key", or "both".
        """
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.internal_subscription_key = internal_subscription_key
        self.auth_method = auth_method
        self.generation_config = generation_config or GenerativeModelConfig()
        self.system_instruction = system_instruction

        # Set headers based on authentication method
        self.headers = {"Content-Type": "application/json"}
        self._setup_authentication()

    def _setup_authentication(self) -> None:
        """
        Setup authentication headers based on the specified authentication method.
        
        Raises:
            ValueError: If authentication method requires keys that are not provided.
        """
        if self.auth_method == "bearer":
            if not self.api_key:
                raise ValueError("API key is required for bearer authentication")
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.auth_method == "api-key":
            if not self.api_key:
                raise ValueError("API key is required for api-key authentication")
            self.headers["api-key"] = self.api_key
        elif self.auth_method == "subscription-key":
            if not self.internal_subscription_key:
                raise ValueError("Internal subscription key is required for subscription-key authentication")
            self.headers["ocp-apim-subscription-key"] = self.internal_subscription_key
        elif self.auth_method == "both":
            if not self.api_key or not self.internal_subscription_key:
                raise ValueError("Both API key and subscription key are required for 'both' authentication")
            self.headers["api-key"] = self.api_key
            self.headers["ocp-apim-subscription-key"] = self.internal_subscription_key
        else:
            raise ValueError(f"Unsupported authentication method: {self.auth_method}")

    def send_request(self, payload: dict) -> dict:
        """
        Send a request to the proxy LLM.

        Args:
            payload (dict): The request payload.

        Returns:
            dict: The response from the proxy.
            
        Raises:
            ValueError: If the request fails or returns an error response.
        """
        try:
            logger.debug("Sending request to proxy LLM at %s", self.base_url)
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error("Request to proxy LLM failed: %s", e)
            raise ValueError(f"Error during request to proxy LLM: {e}") from e

    def start_chat(self, system_instruction: Optional[str] = None) -> GenerativeModelChatSession:
        """
        Start a new chat session.
        
        Args:
            system_instruction (Optional[str]): Optional system instruction to guide the chat session.
            
        Returns:
            GenerativeModelChatSession: A new instance of the chat session.
        """
        return ProxyLLMChatSession(self, system_instruction)

    def parse_generate_content_response(self, response: dict) -> GenerationResponse:
        """
        Parse the model's response and extract content for the user.

        Args:
            response (dict): The raw response from the model.

        Returns:
            GenerationResponse: Parsed response containing the generated text.
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("No choices found in response")
            
            message = choices[0].get("message", {})
            content = message.get("content", "")
            finish_reason = choices[0].get("finish_reason", "stop")
            
            return GenerationResponse(
                text=content,
                finish_reason=(
                    FinishReason.STOP
                    if finish_reason == "stop"
                    else (
                        FinishReason.MAX_TOKENS
                        if finish_reason in ["length", "max_tokens"]
                        else FinishReason.OTHER
                    )
                ),
            )
        except (KeyError, IndexError) as e:
            logger.error("Failed to parse response: %s", e)
            raise ValueError(f"Invalid response format: {e}") from e

    def with_system_instruction(self, system_instruction: str) -> "ProxyLLMModel":
        """
        Set a system instruction for the model.
        
        Args:
            system_instruction (str): The system instruction to set.
            
        Returns:
            ProxyLLMModel: The model instance with updated system instruction.
        """
        self.system_instruction = system_instruction
        return self

    def to_json(self) -> dict:
        """
        Serialize the model's configuration and state to JSON format.

        Returns:
            dict: The serialized JSON data.
        """
        return {
            "base_url": self.base_url,
            "model_name": self.model_name,
            "api_key": self.api_key,
            "internal_subscription_key": self.internal_subscription_key,
            "auth_method": self.auth_method,
            "generation_config": self.generation_config.to_json(),
            "system_instruction": self.system_instruction,
        }

    @staticmethod
    def from_json(json_data: dict) -> "ProxyLLMModel":
        """
        Deserialize a JSON object to create an instance of ProxyLLMModel.

        Args:
            json_data (dict): The serialized JSON data.

        Returns:
            ProxyLLMModel: A new instance of the model.
        """
        return ProxyLLMModel(
            base_url=json_data["base_url"],
            model_name=json_data["model_name"],
            api_key=json_data.get("api_key"),
            internal_subscription_key=json_data.get("internal_subscription_key"),
            auth_method=json_data.get("auth_method", "bearer"),
            generation_config=GenerativeModelConfig.from_json(
                json_data.get("generation_config", {})
            ),
            system_instruction=json_data.get("system_instruction"),
        )


class ProxyLLMChatSession(GenerativeModelChatSession):
    """
    A chat session for interacting with the Proxy LLM model, maintaining conversation history.
    """

    def __init__(self, model: ProxyLLMModel, system_instruction: Optional[str] = None):
        """
        Initialize the chat session and set up the conversation history.

        Args:
            model (ProxyLLMModel): The model instance for the session.
            system_instruction (Optional[str]): Optional system instruction to override model's default.
        """
        self._model = model
        # Use provided system instruction or fall back to model's system instruction
        effective_instruction = system_instruction or self._model.system_instruction
        self._chat_history = (
            [{"role": "system", "content": effective_instruction}]
            if effective_instruction is not None
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
        self._chat_history.append({"role": "user", "content": message})
        
        payload = {
            "model": self._model.model_name,
            "messages": self._chat_history,
            **generation_config,
        }
        
        # Add max_completion_tokens if not already specified in generation_config
        if "max_completion_tokens" not in payload and "max_tokens" not in payload:
            payload["max_completion_tokens"] = 100000
            
        try:
            response = self._model.send_request(payload)
            content = self._model.parse_generate_content_response(response)
            self._chat_history.append({"role": "assistant", "content": content.text})
            return content
        except Exception as e:
            logger.error("Error during message completion: %s", e)
            raise ValueError(f"Error during completion request, please check the credentials - {e}") from e
    
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
            config['response_format'] = {"type": "json_object"}
        
        return config

    def get_chat_history(self) -> list[dict]:
        """
        Retrieve the conversation history for the current chat session.

        Returns:
            list[dict]: The chat session's conversation history.
        """
        return self._chat_history.copy()

    def delete_last_message(self) -> None:
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
        # Determine if we have a system message
        has_system_msg = (
            self._chat_history and 
            self._chat_history[0].get("role") == "system"
        )
        min_length = 1 if has_system_msg else 0
        
        if len(self._chat_history) >= min_length + 2:
            # Remove the last assistant and user messages
            self._chat_history.pop()  # Remove assistant message
            self._chat_history.pop()  # Remove user message
        else:
            # Reset to initial state with just system message if present
            effective_instruction = (
                self._model.system_instruction 
                if hasattr(self._model, 'system_instruction') 
                else None
            )
            self._chat_history = (
                [{"role": "system", "content": effective_instruction}]
                if effective_instruction is not None
                else []
            )