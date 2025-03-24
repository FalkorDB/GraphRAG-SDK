from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional, Iterator


class FinishReason:
    MAX_TOKENS = "MAX_TOKENS"
    STOP = "STOP"
    OTHER = "OTHER"

class OutputMethod(Enum):
    JSON = 'json'
    DEFAULT = 'default'

class GenerativeModelConfig:
    """
    Configuration for a generative model.
    
    This configuration follows OpenAI-style parameter naming but is designed to be compatible with other generative models.
    
    Args:
        temperature (Optional[float]): Controls the randomness of the output. Higher values (e.g., 1.0) make responses more random, 
            while lower values (e.g., 0.1) make them more deterministic.
        top_p (Optional[float]): Nucleus sampling parameter. A value of 0.9 considers only the top 90% of probability mass.
        top_k (Optional[int]): Limits sampling to the top-k most probable tokens.
        max_tokens (Optional[int]): The maximum number of tokens the model is allowed to generate in a response.
        stop (Optional[list[str]]): A list of stop sequences that signal the model to stop generating further tokens.
        response_format (Optional[dict]): Specifies the desired format of the response, if supported by the model.
    Example:
        >>> config = GenerativeModelConfig(
        ...     temperature=0.5, 
        ...     top_p=0.9, 
        ...     top_k=50, 
        ...     max_tokens=100, 
        ...     stop=[".", "?", "!"]
        ... )
    """

    def __init__(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        response_format: Optional[dict] = None,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.stop = stop
        self.response_format = response_format

    def __str__(self) -> str:
        return f"GenerativeModelConfig(temperature={self.temperature}, top_p={self.top_p}, top_k={self.top_k}, max_tokens={self.max_tokens}, stop={self.stop})"

    def to_json(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "response_format": self.response_format,
        }

    @staticmethod
    def from_json(json: dict) -> "GenerativeModelConfig":
        return GenerativeModelConfig(
            temperature=json.get("temperature"),
            top_p=json.get("top_p"),
            top_k=json.get("top_k"),
            max_tokens=json.get("max_tokens"),
            stop=json.get("stop"),
        )


class GenerationResponse:

    def __init__(self, text: str, finish_reason: FinishReason):
        self.text = text
        self.finish_reason = finish_reason

    def __str__(self) -> str:
        return (
            f"GenerationResponse(text={self.text}, finish_reason={self.finish_reason})"
        )


class GenerativeModelChatSession(ABC):
    """
    A chat session with a generative model.
    """

    @abstractmethod
    def __init__(self, model: "GenerativeModel"):
        self.model = model

    @abstractmethod
    def send_message(self, message: str, output_method: OutputMethod = OutputMethod.DEFAULT) -> GenerationResponse:
        pass

    def send_message_stream(self, message: str) -> Iterator[str]:
        raise NotImplementedError("Streaming not supported by this API implementation.")


class GenerativeModel(ABC):
    """
    A generative model that can be used to generate text.
    """

    @abstractmethod
    def start_chat(self, system_instruction: Optional[str] = None) -> GenerativeModelChatSession:
        pass
    
    @staticmethod
    @abstractmethod
    def from_json(json: dict) -> "GenerativeModel":
        pass

    @abstractmethod
    def to_json(self) -> dict:
        pass
