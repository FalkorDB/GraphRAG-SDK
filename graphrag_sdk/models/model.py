from abc import ABC, abstractmethod
from enum import Enum


class FinishReason:
    MAX_TOKENS = "MAX_TOKENS"
    STOP = "STOP"
    OTHER = "OTHER"

class OutputMethod(Enum):
    JSON = 'json'
    DEFAULT = 'default'

class GenerativeModelConfig:
    """
    Configuration for a generative model

    Args:
        temperature (float): The temperature to use for sampling.
        top_p (float): The top-p value to use for sampling.
        top_k (int): The top-k value to use for sampling.
        max_output_tokens (int): The maximum number of tokens to generate.
        stop_sequences (list[str]): The stop sequences to use for sampling.
        response_format (dict): The format of the response.

    Examples:

        >>> config = GenerativeModelConfig(temperature=0.5, top_p=0.9, top_k=50, max_output_tokens=100, stop_sequences=[".", "?", "!"])
    """

    def __init__(
        self,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_output_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        response_format: dict | None = None,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.stop_sequences = stop_sequences
        self.response_format = response_format

    def __str__(self) -> str:
        return f"GenerativeModelConfig(temperature={self.temperature}, top_p={self.top_p}, top_k={self.top_k}, max_output_tokens={self.max_output_tokens}, stop_sequences={self.stop_sequences})"

    def to_json(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_output_tokens,
            "stop": self.stop_sequences,
            "response_format": self.response_format,
        }

    @staticmethod
    def from_json(json: dict) -> "GenerativeModelConfig":
        return GenerativeModelConfig(
            temperature=json.get("temperature"),
            top_p=json.get("top_p"),
            top_k=json.get("top_k"),
            max_output_tokens=json.get("max_tokens"),
            stop_sequences=json.get("stop"),
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


class GenerativeModel(ABC):
    """
    A generative model that can be used to generate text.
    """

    @abstractmethod
    def with_system_instruction(self, system_instruction: str) -> "GenerativeModel":
        pass

    @abstractmethod
    def start_chat(self, args: dict | None) -> GenerativeModelChatSession:
        pass

    @abstractmethod
    def ask(self, message: str) -> GenerationResponse:
        pass

    @staticmethod
    @abstractmethod
    def from_json(json: dict) -> "GenerativeModel":
        pass

    @abstractmethod
    def to_json(self) -> dict:
        pass
