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
    
    # Sentinel value to detect when temperature was not explicitly set
    _TEMP_NOT_SET = object()

    def __init__(
        self,
        temperature = _TEMP_NOT_SET,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list] = None,
        response_format: Optional[dict] = None,
        **kwargs,
    ):
        # Set temperature and track if it was explicitly set
        if temperature is self._TEMP_NOT_SET:
            self.temperature = None
            self._temperature_was_set = False
        else:
            self.temperature = temperature
            self._temperature_was_set = True
            
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
        return {k: v for k, v in self.__dict__.items() if v is not None and k != '_temperature_was_set'}

    @staticmethod
    def from_json(json: dict) -> "GenerativeModelConfig":
        # Extract known parameters, but only pass them if they exist in JSON
        params = {}
        if "temperature" in json:
            params['temperature'] = json["temperature"]
        # For other parameters, only pass if they exist in JSON to maintain default behavior
        if "top_p" in json:
            params['top_p'] = json["top_p"]
        if "top_k" in json:
            params['top_k'] = json["top_k"]
        if "max_tokens" in json:
            params['max_tokens'] = json["max_tokens"]
        if "stop" in json:
            params['stop'] = json["stop"]
        if "response_format" in json:
            params['response_format'] = json["response_format"]
        
        # Extract any additional parameters not in the known set
        known_keys = {'temperature', 'top_p', 'top_k', 'max_tokens', 'stop', 'response_format'}
        extra_params = {k: v for k, v in json.items() if k not in known_keys}
        
        return GenerativeModelConfig(**params, **extra_params)


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
