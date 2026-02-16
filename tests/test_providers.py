"""Tests for core/providers.py — Embedder and LLMInterface ABCs."""
from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from graphrag_sdk.core.models import LLMResponse
from graphrag_sdk.core.providers import Embedder, LLMInterface


# ── Concrete test implementations ──────────────────────────────


class SimpleEmbedder(Embedder):
    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        return [float(len(text)), 0.5, 0.1]


class SimpleLLM(LLMInterface):
    def __init__(self) -> None:
        super().__init__(model_name="simple-test")

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return LLMResponse(content=f"Echo: {prompt[:20]}")


class TestEmbedder:
    def test_embed_query(self):
        emb = SimpleEmbedder()
        result = emb.embed_query("hello")
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == 5.0  # len("hello")

    async def test_aembed_query_default(self):
        """Default async falls back to sync via asyncio.to_thread."""
        emb = SimpleEmbedder()
        result = await emb.aembed_query("world")
        assert isinstance(result, list)
        assert result[0] == 5.0  # len("world")

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Embedder()  # type: ignore[abstract]


class TestLLMInterface:
    def test_invoke(self):
        llm = SimpleLLM()
        response = llm.invoke("What is 2+2?")
        assert response.content.startswith("Echo:")
        assert "What is 2+2?" in response.content

    def test_model_name(self):
        llm = SimpleLLM()
        assert llm.model_name == "simple-test"

    def test_model_params_default(self):
        llm = SimpleLLM()
        assert llm.model_params == {}

    async def test_ainvoke_default(self):
        llm = SimpleLLM()
        response = await llm.ainvoke("Async test")
        assert response.content.startswith("Echo:")

    async def test_invoke_with_model(self):
        class Result(BaseModel):
            answer: int

        class StructuredLLM(LLMInterface):
            def __init__(self) -> None:
                super().__init__(model_name="struct-test")

            def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
                return LLMResponse(content='{"answer": 42}')

        llm = StructuredLLM()
        result = llm.invoke_with_model("What?", response_model=Result)
        assert isinstance(result, Result)
        assert result.answer == 42

    async def test_ainvoke_with_model(self):
        class Result(BaseModel):
            value: str

        class StructuredLLM(LLMInterface):
            def __init__(self) -> None:
                super().__init__(model_name="struct-test")

            def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
                return LLMResponse(content='{"value": "hello"}')

        llm = StructuredLLM()
        result = await llm.ainvoke_with_model("prompt", response_model=Result)
        assert result.value == "hello"

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            LLMInterface(model_name="abstract")  # type: ignore[abstract]
