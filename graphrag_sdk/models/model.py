from abc import ABC, abstractmethod
from typing import Optional, Iterator


class FinishReason:
    MAX_TOKENS = "MAX_TOKENS"
    STOP = "STOP"
    OTHER = "OTHER"

class GenerativeModelConfig:
    """
    Configuration for a generative model.

    This configuration follows OpenAI-style parameter naming but is designed to be 
    compatible with other generative models. Supports both predefined and arbitrary parameters.

    Args:
        temperature (Optional[float]): Controls randomness of the output.
        top_p (Optional[float]): Nucleus sampling parameter.
        top_k (Optional[int]): Limits sampling to the top-k most probable tokens.
        max_tokens (Optional[int]): Maximum number of tokens to generate.
        stop (Optional[list[str]]): Stop sequences.
        response_format (Optional[dict]): Desired response format.
        **kwargs: Any additional parameters not explicitly defined.

    Example:
        >>> config = GenerativeModelConfig(
        ...     temperature=0.5, 
        ...     max_tokens=100,
        ... )
        >>> config.to_json()
        {'temperature': 0.5, 'max_tokens': 100}
    """

    def __init__(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        response_format: Optional[dict] = None,
        **kwargs,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.stop = stop
        self.response_format = response_format

        # Store extra parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        return f"GenerativeModelConfig({', '.join(f'{k}={v}' for k, v in self.to_json().items())})"
    
    def to_json(self) -> dict:
        """
        Serialize the configuration to a dictionary, excluding any fields with None values.

        Returns:
            dict: A dictionary containing only the parameters that are explicitly set 
                (i.e., not None).
        
        Example:
            >>> config = GenerativeModelConfig(temperature=0.7, max_tokens=100)
            >>> config.to_json()
            {'temperature': 0.7, 'max_tokens': 100}
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @staticmethod
    def from_json(json: dict) -> "GenerativeModelConfig":
        # Extract known parameters
        known_params = {
            'temperature': json.get("temperature"),
            'top_p': json.get("top_p"),
            'top_k': json.get("top_k"),
            'max_tokens': json.get("max_tokens"),
            'stop': json.get("stop"),
            'response_format': json.get("response_format"),
        }
        
        # Extract any additional parameters not in the known set
        known_keys = set(known_params.keys())
        extra_params = {k: v for k, v in json.items() if k not in known_keys}
        
        return GenerativeModelConfig(**known_params, **extra_params)


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
    def send_message(self, message: str) -> GenerationResponse:
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
