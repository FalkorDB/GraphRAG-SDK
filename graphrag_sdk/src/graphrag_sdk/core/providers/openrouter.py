# GraphRAG SDK 2.0 — OpenRouter Provider
# LLM and Embedder backed by OpenRouter (uses the OpenAI SDK).
# Requires: pip install graphrag-sdk[openrouter]

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from graphrag_sdk.core.models import LLMResponse
from graphrag_sdk.core.providers.base import Embedder, LLMInterface
from graphrag_sdk.core.providers._retry import (
    binary_split_retry_async,
    binary_split_retry_sync,
)

logger = logging.getLogger(__name__)


class OpenRouterLLM(LLMInterface):
    """LLM provider backed by OpenRouter (uses the OpenAI SDK).

    Requires ``pip install graphrag-sdk[openrouter]``.

    Example::

        llm = OpenRouterLLM(model="anthropic/claude-sonnet-4-20250514", api_key="...")
        response = llm.invoke("Hello!")
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(model_name=model)
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra_headers = extra_headers or {}
        self._client = None
        self._async_client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "The openai package is required for the OpenRouter provider. "
                    "Install with: pip install graphrag-sdk[openrouter]"
                )
            self._client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self._api_key,
                default_headers=self._extra_headers or None,
            )
        return self._client

    def _get_async_client(self):
        if self._async_client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "The openai package is required for the OpenRouter provider. "
                    "Install with: pip install graphrag-sdk[openrouter]"
                )
            self._async_client = openai.AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self._api_key,
                default_headers=self._extra_headers or None,
            )
        return self._async_client

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        create_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            **kwargs,
        }
        if self._max_tokens is not None:
            create_kwargs.setdefault("max_tokens", self._max_tokens)
        response = client.chat.completions.create(**create_kwargs)
        content = response.choices[0].message.content or ""
        return LLMResponse(content=content)

    async def ainvoke(
        self,
        prompt: str,
        *,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_async_client()
        create_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            **kwargs,
        }
        if self._max_tokens is not None:
            create_kwargs.setdefault("max_tokens", self._max_tokens)
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(**create_kwargs)
                content = response.choices[0].message.content or ""
                return LLMResponse(content=content)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    delay = 2**attempt
                    logger.warning(
                        f"OpenRouter call failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s: {exc}"
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]


class OpenRouterEmbedder(Embedder):
    """Embedding provider backed by OpenRouter (uses the OpenAI SDK).

    Requires ``pip install graphrag-sdk[openrouter]``.

    Example::

        embedder = OpenRouterEmbedder(model="openai/text-embedding-ada-002", api_key="...")
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        batch_size: int = 2048,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.batch_size = batch_size
        self._extra_headers = extra_headers or {}
        self._client = None
        self._async_client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "The openai package is required for the OpenRouterEmbedder provider. "
                    "Install with: pip install graphrag-sdk[openrouter]"
                )
            self._client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self._api_key,
                default_headers=self._extra_headers or None,
            )
        return self._client

    def _get_async_client(self):
        if self._async_client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "The openai package is required for the OpenRouterEmbedder provider. "
                    "Install with: pip install graphrag-sdk[openrouter]"
                )
            self._async_client = openai.AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self._api_key,
                default_headers=self._extra_headers or None,
            )
        return self._async_client

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        client = self._get_client()
        response = client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def _raw_embed_sync(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Raw sync embed without retry — called by binary_split_retry_sync."""
        client = self._get_client()
        response = client.embeddings.create(model=self.model, input=texts)
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]

    async def _raw_embed_async(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Raw async embed without retry — called by binary_split_retry_async."""
        client = self._get_async_client()
        response = await client.embeddings.create(model=self.model, input=texts)
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            results.extend(binary_split_retry_sync(self._raw_embed_sync, batch, **kwargs))
        return results

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        client = self._get_async_client()
        response = await client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    async def aembed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            results.extend(await binary_split_retry_async(self._raw_embed_async, batch, **kwargs))
        return results
