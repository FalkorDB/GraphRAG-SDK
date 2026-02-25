"""Tests for core/providers.py — Embedder and LLMInterface ABCs + built-in providers."""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from graphrag_sdk.core.models import LLMResponse
from graphrag_sdk.core.providers import (
    Embedder,
    LLMBatchItem,
    LLMInterface,
    LiteLLM,
    LiteLLMEmbedder,
    OpenRouterEmbedder,
    OpenRouterLLM,
)


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

    def test_max_concurrency_default(self):
        llm = SimpleLLM()
        assert llm.max_concurrency == 12

    def test_max_concurrency_custom(self):
        class CustomLLM(LLMInterface):
            def __init__(self) -> None:
                super().__init__(model_name="custom", max_concurrency=5)

            def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
                return LLMResponse(content="ok")

        llm = CustomLLM()
        assert llm.max_concurrency == 5


class TestLLMBatchItem:
    def test_ok_with_response(self):
        item = LLMBatchItem(index=0, response=LLMResponse(content="hi"))
        assert item.ok is True
        assert item.error is None

    def test_not_ok_with_error(self):
        item = LLMBatchItem(index=1, error=RuntimeError("fail"))
        assert item.ok is False
        assert item.response is None

    def test_index_preserved(self):
        item = LLMBatchItem(index=42, response=LLMResponse(content=""))
        assert item.index == 42


class TestAbatchInvoke:
    @pytest.mark.asyncio
    async def test_returns_all_results(self):
        llm = SimpleLLM()
        prompts = ["a", "b", "c"]
        results = await llm.abatch_invoke(prompts)
        assert len(results) == 3
        for item in results:
            assert item.ok

    @pytest.mark.asyncio
    async def test_preserves_order(self):
        llm = SimpleLLM()
        prompts = ["first", "second", "third"]
        results = await llm.abatch_invoke(prompts)
        for i, item in enumerate(results):
            assert item.index == i

    @pytest.mark.asyncio
    async def test_captures_errors(self):
        class FailingLLM(LLMInterface):
            def __init__(self) -> None:
                super().__init__(model_name="fail-test")

            def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
                if "fail" in prompt:
                    raise RuntimeError("intentional failure")
                return LLMResponse(content=f"ok: {prompt}")

        llm = FailingLLM()
        results = await llm.abatch_invoke(
            ["good", "fail_me", "also_good"], max_retries=1
        )
        assert len(results) == 3

        ok_items = [r for r in results if r.ok]
        err_items = [r for r in results if not r.ok]
        assert len(ok_items) == 2
        assert len(err_items) == 1
        assert isinstance(err_items[0].error, RuntimeError)

    @pytest.mark.asyncio
    async def test_empty_list(self):
        llm = SimpleLLM()
        results = await llm.abatch_invoke([])
        assert results == []

    @pytest.mark.asyncio
    async def test_respects_max_concurrency(self):
        """Verify concurrency limit is respected via tracking."""
        peak_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        class TrackingLLM(LLMInterface):
            def __init__(self) -> None:
                super().__init__(model_name="tracking", max_concurrency=2)

            def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
                return LLMResponse(content="ok")

            async def ainvoke(self, prompt, *, max_retries=3, **kwargs):
                nonlocal peak_concurrent, current_concurrent
                async with lock:
                    current_concurrent += 1
                    if current_concurrent > peak_concurrent:
                        peak_concurrent = current_concurrent
                await asyncio.sleep(0.01)  # simulate work
                async with lock:
                    current_concurrent -= 1
                return LLMResponse(content="ok")

        llm = TrackingLLM()
        results = await llm.abatch_invoke(
            [f"prompt-{i}" for i in range(6)], max_concurrency=2
        )
        assert len(results) == 6
        assert all(r.ok for r in results)
        assert peak_concurrent <= 2


# ═══════════════════════════════════════════════════════════════════
# Helpers for built-in provider tests
# ═══════════════════════════════════════════════════════════════════


def _mock_litellm_completion_response(content: str = "Hello!"):
    """Build a mock response matching litellm.completion() shape."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _mock_litellm_embedding_response(vectors: list[list[float]]):
    """Build a mock response matching litellm.embedding() shape (dict-like data)."""
    data = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    resp = MagicMock()
    resp.data = data
    return resp


def _mock_openai_completion_response(content: str = "Hello!"):
    """Build a mock response matching openai.chat.completions.create() shape."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _mock_openai_embedding_response(vectors: list[list[float]]):
    """Build a mock response matching openai.embeddings.create() shape (attribute access)."""
    data = []
    for i, v in enumerate(vectors):
        item = MagicMock()
        item.index = i
        item.embedding = v
        data.append(item)
    resp = MagicMock()
    resp.data = data
    return resp


# ═══════════════════════════════════════════════════════════════════
# TestLiteLLM
# ═══════════════════════════════════════════════════════════════════


class TestLiteLLM:
    def test_invoke(self):
        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = _mock_litellm_completion_response("Hi there")
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            llm = LiteLLM(model="azure/gpt-4", api_key="test-key", api_base="https://test.openai.azure.com")
            result = llm.invoke("Hello")

        assert isinstance(result, LLMResponse)
        assert result.content == "Hi there"
        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["model"] == "azure/gpt-4"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["api_base"] == "https://test.openai.azure.com"

    def test_invoke_passes_extra_kwargs(self):
        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = _mock_litellm_completion_response("ok")
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            llm = LiteLLM(model="gpt-4", temperature=0.5, top_p=0.9)
            llm.invoke("test")

        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_ainvoke(self):
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_litellm_completion_response("async hi")
        )
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            llm = LiteLLM(model="azure/gpt-4")
            result = await llm.ainvoke("Hello")

        assert result.content == "async hi"
        mock_litellm.acompletion.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ainvoke_retries_on_failure(self):
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                RuntimeError("rate limit"),
                _mock_litellm_completion_response("recovered"),
            ]
        )
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                llm = LiteLLM(model="gpt-4")
                result = await llm.ainvoke("test", max_retries=3)

        assert result.content == "recovered"
        assert mock_litellm.acompletion.await_count == 2

    def test_import_error(self):
        with patch.dict("sys.modules", {"litellm": None}):
            llm = LiteLLM(model="gpt-4")
            with pytest.raises(ImportError, match="pip install graphrag-sdk\\[litellm\\]"):
                llm.invoke("test")


# ═══════════════════════════════════════════════════════════════════
# TestLiteLLMEmbedder
# ═══════════════════════════════════════════════════════════════════


class TestLiteLLMEmbedder:
    def test_embed_query(self):
        vec = [0.1, 0.2, 0.3]
        mock_litellm = MagicMock()
        mock_litellm.embedding.return_value = _mock_litellm_embedding_response([vec])
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            embedder = LiteLLMEmbedder(model="azure/text-embedding-ada-002", api_key="k")
            result = embedder.embed_query("hello")

        assert result == vec
        mock_litellm.embedding.assert_called_once()

    def test_embed_documents_batching(self):
        mock_litellm = MagicMock()
        batch1_vecs = [[0.1, 0.2], [0.3, 0.4]]
        batch2_vecs = [[0.5, 0.6]]
        mock_litellm.embedding.side_effect = [
            _mock_litellm_embedding_response(batch1_vecs),
            _mock_litellm_embedding_response(batch2_vecs),
        ]
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            embedder = LiteLLMEmbedder(model="text-embedding-ada-002", batch_size=2)
            result = embedder.embed_documents(["a", "b", "c"])

        assert len(result) == 3
        assert result == batch1_vecs + batch2_vecs
        assert mock_litellm.embedding.call_count == 2

    def test_embed_documents_empty(self):
        mock_litellm = MagicMock()
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            embedder = LiteLLMEmbedder(model="text-embedding-ada-002")
            result = embedder.embed_documents([])

        assert result == []
        mock_litellm.embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_aembed_query(self):
        vec = [0.7, 0.8, 0.9]
        mock_litellm = MagicMock()
        mock_litellm.aembedding = AsyncMock(
            return_value=_mock_litellm_embedding_response([vec])
        )
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            embedder = LiteLLMEmbedder(model="text-embedding-ada-002")
            result = await embedder.aembed_query("hello")

        assert result == vec

    @pytest.mark.asyncio
    async def test_aembed_documents(self):
        mock_litellm = MagicMock()
        vecs = [[0.1, 0.2], [0.3, 0.4]]
        mock_litellm.aembedding = AsyncMock(
            return_value=_mock_litellm_embedding_response(vecs)
        )
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            embedder = LiteLLMEmbedder(model="text-embedding-ada-002")
            result = await embedder.aembed_documents(["a", "b"])

        assert result == vecs

    def test_import_error(self):
        with patch.dict("sys.modules", {"litellm": None}):
            embedder = LiteLLMEmbedder(model="text-embedding-ada-002")
            with pytest.raises(ImportError, match="pip install graphrag-sdk\\[litellm\\]"):
                embedder.embed_query("test")


# ═══════════════════════════════════════════════════════════════════
# TestOpenRouterLLM
# ═══════════════════════════════════════════════════════════════════


class TestOpenRouterLLM:
    def test_invoke(self):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_completion_response("router hi")
        mock_openai.OpenAI.return_value = mock_client
        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenRouterLLM(model="anthropic/claude-sonnet-4-20250514", api_key="or-key")
            result = llm.invoke("Hello")

        assert result.content == "router hi"
        mock_openai.OpenAI.assert_called_once()
        call_kwargs = mock_openai.OpenAI.call_args[1]
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert call_kwargs["api_key"] == "or-key"

    @pytest.mark.asyncio
    async def test_ainvoke(self):
        mock_openai = MagicMock()
        mock_async_client = MagicMock()
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion_response("async router")
        )
        mock_openai.AsyncOpenAI.return_value = mock_async_client
        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenRouterLLM(model="openai/gpt-4", api_key="or-key")
            result = await llm.ainvoke("Hello")

        assert result.content == "async router"
        mock_openai.AsyncOpenAI.assert_called_once()

    def test_env_var_fallback(self):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_completion_response("ok")
        mock_openai.OpenAI.return_value = mock_client
        with patch.dict("sys.modules", {"openai": mock_openai}):
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-key"}):
                llm = OpenRouterLLM(model="gpt-4")
                assert llm._api_key == "env-key"

    def test_import_error(self):
        with patch.dict("sys.modules", {"openai": None}):
            llm = OpenRouterLLM(model="gpt-4", api_key="k")
            with pytest.raises(ImportError, match="pip install graphrag-sdk\\[openrouter\\]"):
                llm.invoke("test")


# ═══════════════════════════════════════════════════════════════════
# TestOpenRouterEmbedder
# ═══════════════════════════════════════════════════════════════════


class TestOpenRouterEmbedder:
    def test_embed_query(self):
        vec = [0.1, 0.2, 0.3]
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _mock_openai_embedding_response([vec])
        mock_openai.OpenAI.return_value = mock_client
        with patch.dict("sys.modules", {"openai": mock_openai}):
            embedder = OpenRouterEmbedder(model="openai/text-embedding-ada-002", api_key="k")
            result = embedder.embed_query("hello")

        assert result == vec

    def test_embed_documents_batching(self):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        batch1_vecs = [[0.1, 0.2], [0.3, 0.4]]
        batch2_vecs = [[0.5, 0.6]]
        mock_client.embeddings.create.side_effect = [
            _mock_openai_embedding_response(batch1_vecs),
            _mock_openai_embedding_response(batch2_vecs),
        ]
        mock_openai.OpenAI.return_value = mock_client
        with patch.dict("sys.modules", {"openai": mock_openai}):
            embedder = OpenRouterEmbedder(model="openai/text-embedding-ada-002", api_key="k", batch_size=2)
            result = embedder.embed_documents(["a", "b", "c"])

        assert len(result) == 3
        assert result == batch1_vecs + batch2_vecs
        assert mock_client.embeddings.create.call_count == 2

    @pytest.mark.asyncio
    async def test_aembed_query(self):
        vec = [0.4, 0.5, 0.6]
        mock_openai = MagicMock()
        mock_async_client = MagicMock()
        mock_async_client.embeddings.create = AsyncMock(
            return_value=_mock_openai_embedding_response([vec])
        )
        mock_openai.AsyncOpenAI.return_value = mock_async_client
        with patch.dict("sys.modules", {"openai": mock_openai}):
            embedder = OpenRouterEmbedder(model="openai/text-embedding-ada-002", api_key="k")
            result = await embedder.aembed_query("hello")

        assert result == vec

    def test_import_error(self):
        with patch.dict("sys.modules", {"openai": None}):
            embedder = OpenRouterEmbedder(model="text-embedding-ada-002", api_key="k")
            with pytest.raises(ImportError, match="pip install graphrag-sdk\\[openrouter\\]"):
                embedder.embed_query("test")
