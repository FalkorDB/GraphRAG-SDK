import os
from openai import AzureOpenAI
from .model import (
    OutputMethod,
    GenerativeModel,
    GenerativeModelConfig,
    GenerationResponse,
    FinishReason,
    GenerativeModelChatSession,
)

class AzureOpenAiGenerativeModel(GenerativeModel):

    client: AzureOpenAI = None

    def __init__(
        self,
        model_name: str,
        generation_config: GenerativeModelConfig | None = None,
        system_instruction: str | None = None,
    ):
        self.model_name = model_name
        self.generation_config = generation_config or GenerativeModelConfig()
        self.system_instruction = system_instruction
        
        # Credentials
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.api_version = os.getenv("AZURE_API_VERSION")


    def _get_model(self) -> AzureOpenAI:
        if self.client is None:
            self.client = AzureOpenAI(azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
            api_key=self.api_key,
        )
        return self.client

    def with_system_instruction(self, system_instruction: str) -> "GenerativeModel":
        self.system_instruction = system_instruction
        self.client = None
        self._get_model()

        return self

    def start_chat(self, args: dict | None = None) -> GenerativeModelChatSession:
        return AzureOpenAiChatSession(self, args)

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
        return AzureOpenAiGenerativeModel(
            json["model_name"],
            generation_config=GenerativeModelConfig.from_json(
                json["generation_config"]
            ),
            system_instruction=json["system_instruction"],
        )


class AzureOpenAiChatSession(GenerativeModelChatSession):

    _history = []

    def __init__(self, model: AzureOpenAiGenerativeModel, args: dict | None = None):
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
        response = self._model.client.chat.completions.create(
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
