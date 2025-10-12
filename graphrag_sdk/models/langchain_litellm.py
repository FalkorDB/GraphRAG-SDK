import logging
from typing import Optional, Iterator

from .model import (
    GenerativeModel,
    GenerativeModelConfig,
    GenerationResponse,
    FinishReason,
    GenerativeModelChatSession,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LangChainLiteModel(GenerativeModel):
    """
    A generative model that interfaces with LiteLLM through LangChain's ChatLiteLLM wrapper.
    
    This provides an alternative to the direct LiteLLM integration, leveraging LangChain's
    ecosystem and capabilities while still using LiteLLM's multi-provider support.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        generation_config: Optional[GenerativeModelConfig] = None,
        system_instruction: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        additional_params: Optional[dict] = None,
    ):
        """
        Initialize the LangChainLiteModel with the required parameters.
        
        Args:
            model_name (str): The model name for LiteLLM (can include provider prefix like 'openai/gpt-4' 
                or use direct model names). Defaults to "gpt-4o-mini".
            generation_config (Optional[GenerativeModelConfig]): Configuration settings for generation.
            system_instruction (Optional[str]): Instruction to guide the model.
            api_key (Optional[str]): API key for the LiteLLM service/provider.
            api_base (Optional[str]): Base URL for the LiteLLM API endpoint.
            additional_params (Optional[dict]): Additional provider-specific parameters.
        """
        try:
            from langchain_litellm import ChatLiteLLM
        except ImportError:
            raise ImportError(
                "langchain-litellm package is required for LangChainLiteModel. "
                "Install it with: pip install langchain-litellm"
            )

        self.model_name = model_name
        self.generation_config = generation_config or GenerativeModelConfig()
        self.system_instruction = system_instruction
        self.api_key = api_key
        self.api_base = api_base
        self.additional_params = additional_params or {}

        # Initialize the ChatLiteLLM model
        chat_params = {
            "model": model_name,
        }
        
        if api_key:
            chat_params["api_key"] = api_key
        if api_base:
            chat_params["api_base"] = api_base
            
        # Merge additional params
        chat_params.update(self.additional_params)
        
        # Add generation config parameters
        gen_config = self.generation_config.to_json()
        if gen_config:
            chat_params.update(gen_config)

        try:
            self._chat_model = ChatLiteLLM(**chat_params)
        except Exception as e:
            raise ValueError(
                f"Failed to initialize ChatLiteLLM with model '{model_name}': {e}"
            ) from e

    def start_chat(
        self, system_instruction: Optional[str] = None
    ) -> GenerativeModelChatSession:
        """
        Start a new chat session.
        
        Args:
            system_instruction (Optional[str]): Optional system instruction to guide the chat session.
            
        Returns:
            GenerativeModelChatSession: A new instance of the chat session.
        """
        return LangChainLiteModelChatSession(self, system_instruction)

    def parse_generate_content_response(self, response: any) -> GenerationResponse:
        """
        Parse the LangChain model's response and extract content for the user.

        Args:
            response (any): The raw response from the LangChain model.

        Returns:
            GenerationResponse: Parsed response containing the generated text.
        """
        # LangChain returns AIMessage objects
        finish_reason = FinishReason.STOP
        
        # Check if response has finish_reason in response_metadata
        if hasattr(response, "response_metadata"):
            metadata_finish_reason = response.response_metadata.get("finish_reason", "stop")
            if metadata_finish_reason == "length":
                finish_reason = FinishReason.MAX_TOKENS
            elif metadata_finish_reason != "stop":
                finish_reason = FinishReason.OTHER

        return GenerationResponse(
            text=response.content,
            finish_reason=finish_reason,
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
            "api_key": self.api_key,
            "api_base": self.api_base,
            "additional_params": self.additional_params,
        }

    @staticmethod
    def from_json(json: dict) -> "GenerativeModel":
        """
        Deserialize a JSON object to create an instance of LangChainLiteModel.

        Args:
            json (dict): The serialized JSON data.

        Returns:
            GenerativeModel: A new instance of the model.
        """
        return LangChainLiteModel(
            model_name=json["model_name"],
            generation_config=GenerativeModelConfig.from_json(
                json["generation_config"]
            ),
            system_instruction=json.get("system_instruction"),
            api_key=json.get("api_key"),
            api_base=json.get("api_base"),
            additional_params=json.get("additional_params"),
        )


class LangChainLiteModelChatSession(GenerativeModelChatSession):
    """
    A chat session for interacting with the LangChain LiteLLM model, maintaining conversation history.
    """

    def __init__(
        self, model: LangChainLiteModel, system_instruction: Optional[str] = None
    ):
        """
        Initialize the chat session and set up the conversation history.

        Args:
            model (LangChainLiteModel): The model instance for the session.
            system_instruction (Optional[str]): Optional system instruction to guide the chat session.
        """
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        self._model = model
        self._chat_history = []
        self._SystemMessage = SystemMessage
        self._HumanMessage = HumanMessage
        self._AIMessage = AIMessage

        # Add system instruction if provided
        instruction = system_instruction or model.system_instruction
        if instruction:
            self._chat_history.append(self._SystemMessage(content=instruction))

    def send_message(self, message: str) -> GenerationResponse:
        """
        Send a message in the chat session and receive the model's response.

        Args:
            message (str): The message to send.

        Returns:
            GenerationResponse: The generated response.
        """
        self._chat_history.append(self._HumanMessage(content=message))

        try:
            response = self._model._chat_model.invoke(self._chat_history)
        except Exception as e:
            # Remove the user message to keep history consistent
            self._chat_history.pop()
            raise ValueError(
                f"Error during LangChain LiteLLM completion request: {e}"
            ) from e

        content = self._model.parse_generate_content_response(response)
        self._chat_history.append(self._AIMessage(content=content.text))
        return content

    def send_message_stream(self, message: str) -> Iterator[str]:
        """
        Send a message and receive the response in a streaming fashion.

        Args:
            message (str): The message to send.

        Yields:
            str: Streamed chunks of the model's response.
        """
        self._chat_history.append(self._HumanMessage(content=message))

        try:
            chunks = []
            for chunk in self._model._chat_model.stream(self._chat_history):
                content = chunk.content
                if content:
                    chunks.append(content)
                    yield content

            # Save the full response to chat history
            full_response = "".join(chunks)
            self._chat_history.append(self._AIMessage(content=full_response))

        except Exception as e:
            # Remove the user message to keep history consistent
            self._chat_history.pop()
            raise ValueError(
                f"Error during LangChain LiteLLM streaming request: {e}"
            ) from e

    def get_chat_history(self) -> list[dict]:
        """
        Retrieve the conversation history for the current chat session.

        Returns:
            list[dict]: The chat session's conversation history in dictionary format.
        """
        history = []
        for msg in self._chat_history:
            if isinstance(msg, self._SystemMessage):
                history.append({"role": "system", "content": msg.content})
            elif isinstance(msg, self._HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, self._AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history

    def delete_last_message(self):
        """
        Deletes the last message exchange (user message and assistant response) from the chat history.
        Preserves the system message if present.

        Note: Does nothing if the chat history is empty or contains only a system message.
        """
        # Determine if the first message is a system message
        has_system = (
            len(self._chat_history) > 0
            and isinstance(self._chat_history[0], self._SystemMessage)
        )
        # The index where user/assistant messages start
        start_idx = 1 if has_system else 0
        # Number of user/assistant messages in history
        num_user_assistant_msgs = len(self._chat_history) - start_idx

        if num_user_assistant_msgs >= 2:
            # Remove last assistant message and user message
            self._chat_history.pop()
            self._chat_history.pop()
        else:
            # Reset to initial state with just system message if present
            if has_system:
                system_msg = self._chat_history[0]
                self._chat_history = [system_msg]
            else:
                self._chat_history = []
