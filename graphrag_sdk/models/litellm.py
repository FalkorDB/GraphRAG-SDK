from typing import Optional
from litellm import completion
from .model import (
    OutputMethod,
    GenerativeModel,
    GenerativeModelConfig,
    GenerationResponse,
    FinishReason,
    GenerativeModelChatSession,
)


class LiteLLMGenerativeModel(GenerativeModel):
    """
    A generative model that interfaces with the LiteLLM for chat completions.
    """

    def __init__(
        self,
        model_name: str,
        generation_config: Optional[GenerativeModelConfig] = None,
        system_instruction: Optional[str] = None,
    ):
        self.model_name = model_name
        self.generation_config = generation_config or GenerativeModelConfig()
        self.system_instruction = system_instruction


    def with_system_instruction(self, system_instruction: str) -> "GenerativeModel":
        self.system_instruction = system_instruction
        return self

    def start_chat(self, args: Optional[dict] = None) -> GenerativeModelChatSession:
        return LiteLLMChatSession(self, args)

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
        return {
            "model_name": self.model_name,
            "generation_config": self.generation_config.to_json(),
            "system_instruction": self.system_instruction,
        }

    @staticmethod
    def from_json(json: dict) -> "GenerativeModel":
        return LiteLLMGenerativeModel(
            json["model_name"],
            generation_config=GenerativeModelConfig.from_json(
                json["generation_config"]
            ),
            system_instruction=json["system_instruction"],
        )


class LiteLLMChatSession(GenerativeModelChatSession):

    _history = []

    def __init__(self, model: LiteLLMGenerativeModel, args: Optional[dict] = None):
        self._model = model
        self._args = args
        self._history = (
            [{"role": "system", "content": self._model.system_instruction}]
            if self._model.system_instruction is not None
            else []
        )

    def send_message(self, message: str, output_method: OutputMethod = OutputMethod.DEFAULT) -> GenerationResponse:
        generation_config = self._get_generation_config(output_method)
        prompt = []
        prompt.extend(self._history)
        prompt.append({"role": "user", "content": message[:14385]})
        response = completion(
            model=self._model.model_name,
            messages=prompt,
            **generation_config
            )
        content = self._model.parse_generate_content_response(response)
        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": content.text})
        return content
    
    def _get_generation_config(self, output_method: OutputMethod):
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
