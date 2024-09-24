import os
from .model import (
    OutputMethod,
    GenerativeModel,
    GenerativeModelConfig,
    GenerationResponse,
    FinishReason,
    GenerativeModelChatSession,
)

from google.generativeai import (
    GenerativeModel as GoogleGenerativeModel,
    GenerationConfig as GoogleGenerationConfig,
    configure,
    protos,
    types,)


class GeminiGenerativeModel(GenerativeModel):

    _model: GoogleGenerativeModel = None

    def __init__(
        self,
        model_name: str,
        generation_config: GoogleGenerationConfig | None = None,
        system_instruction: str | None = None,
    ):
        self._model_name = model_name
        self._generation_config = generation_config
        self._system_instruction = system_instruction
        configure(api_key=os.environ["GOOGLE_API_KEY"])


    def _get_model(self) -> GoogleGenerativeModel:
        if self._model is None:
            self._model = GoogleGenerativeModel(
                self._model_name,
                generation_config=(
                    GoogleGenerationConfig(
                        temperature=self._generation_config.temperature,
                        top_p=self._generation_config.top_p,
                        top_k=self._generation_config.top_k,
                        max_output_tokens=self._generation_config.max_output_tokens,
                        stop_sequences=self._generation_config.stop_sequences,
                    )
                    if self._generation_config is not None
                    else None
                ),
                system_instruction=self._system_instruction,
            )

        return self._model

    def with_system_instruction(self, system_instruction: str) -> "GenerativeModel":
        self._system_instruction = system_instruction
        self._model = None
        self._get_model()

        return self

    def start_chat(self, args: dict | None = None) -> GenerativeModelChatSession:
        return GeminiChatSession(self, args)

    def ask(self, message: str) -> GenerationResponse:
        response = self._model.generate_content(message)
        return self.parse_generate_content_response(response)

    def parse_generate_content_response(
        self, response: types.generation_types.GenerateContentResponse
    ) -> GenerationResponse:
        return GenerationResponse(
            text=response.text,
            finish_reason=(
                FinishReason.MAX_TOKENS
                if response.candidates[0].finish_reason
                == protos.Candidate.FinishReason.MAX_TOKENS
                else (
                    FinishReason.STOP
                    if response.candidates[0].finish_reason == protos.Candidate.FinishReason.STOP
                    else FinishReason.OTHER
                )
            ),
        )

    def to_json(self) -> dict:
        return {
            "model_name": self._model_name,
            "generation_config": self._generation_config.to_json(),
            "system_instruction": self._system_instruction,
        }

    @staticmethod
    def from_json(json: dict) -> "GenerativeModel":
        return GeminiGenerativeModel(
            model_name=json["model_name"],
            generation_config=GenerativeModelConfig.from_json(
                json["generation_config"]
            ),
            system_instruction=json["system_instruction"],
        )


class GeminiChatSession(GenerativeModelChatSession):

    def __init__(self, model: GeminiGenerativeModel, args: dict | None = None):
        self._model = model
        self._chat_session = self._model._model.start_chat(
            history=args.get("history", []) if args is not None else [],
        )

    def send_message(self, message: str, output_method: OutputMethod = OutputMethod.DEFAULT) -> GenerationResponse:
        generation_config = self._get_generation_config(output_method)
        response = self._chat_session.send_message(message, generation_config=generation_config)
        return self._model.parse_generate_content_response(response)
    
    def _get_generation_config(self, output_method: OutputMethod):
        if output_method == OutputMethod.JSON:
            return {
                "response_mime_type": "application/json",
                "temperature": 0
            }
        return self._model._generation_config
