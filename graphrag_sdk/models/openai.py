from .model import (
    GenerativeModel,
    GenerativeModelConfig,
    GenerationResponse,
    FinishReason,
    GenerativeModelChatSession,
)
from openai import OpenAI


class OpenAiGenerativeModel(GenerativeModel):

    client: OpenAI = None

    def __init__(
        self,
        model_name: str,
        generation_config: GenerativeModelConfig | None = None,
        system_instruction: str | None = None,
    ):
        self.model_name = model_name
        self.generation_config = generation_config
        self.system_instruction = system_instruction

    def _get_model(self) -> OpenAI:
        if self.client is None:
            self.client = OpenAI()

        return self.client

    def with_system_instruction(self, system_instruction: str) -> "GenerativeModel":
        self.system_instruction = system_instruction
        self.client = None
        self._get_model()

        return self

    def start_chat(self, args: dict | None = None) -> GenerativeModelChatSession:
        return OpenAiChatSession(self, args)

    def ask(self, message: str) -> GenerationResponse:
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
        return self.parse_generate_content_response(response)

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
        return OpenAiGenerativeModel(
            json["model_name"],
            generation_config=GenerativeModelConfig.from_json(
                json["generation_config"]
            ),
            system_instruction=json["system_instruction"],
        )


class OpenAiChatSession(GenerativeModelChatSession):

    _history = []

    def __init__(self, model: OpenAiGenerativeModel, args: dict | None = None):
        self._model = model
        self._args = args
        self._history = (
            [{"role": "system", "content": self._model.system_instruction}]
            if self._model.system_instruction is not None
            else []
        )

    def send_message(self, message: str) -> GenerationResponse:
        prompt = []
        prompt.extend(self._history)
        prompt.append({"role": "user", "content": message[:14385]})
        response = self._model.client.chat.completions.create(
            model=self._model.model_name,
            messages=prompt,
            max_tokens=(
                self._model.generation_config.max_output_tokens
                if self._model.generation_config is not None
                else None
            ),
            temperature=(
                self._model.generation_config.temperature
                if self._model.generation_config is not None
                else None
            ),
            top_p=(
                self._model.generation_config.top_p
                if self._model.generation_config is not None
                else None
            ),
            stop=(
                self._model.generation_config.stop_sequences
                if self._model.generation_config is not None
                else None
            ),
        )
        content = self._model.parse_generate_content_response(response)
        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": content.text})
        return content
