# GraphRAG SDK 2.0 — Core: Provider ABCs + Built-in Providers
# Thin abstract interfaces for Embedders and LLMs, plus concrete
# LiteLLM and OpenRouter implementations (optional dependencies).

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Type

from pydantic import BaseModel

from graphrag_sdk.core.models import LLMResponse


@dataclass
class LLMBatchItem:
    """Result of one item in a batch invocation."""

    index: int
    response: LLMResponse | None = None
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.response is not None

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Abstract base class for all embedding providers.

    Subclasses must implement ``embed_query``.  The async variant
    defaults to running the sync method in a thread pool — override
    ``aembed_query`` for true async providers.

    Example::

        class OpenAIEmbedder(Embedder):
            def embed_query(self, text: str, **kw) -> list[float]:
                return openai.embed(text)
    """

    @abstractmethod
    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Embed a single text string into a float vector."""
        ...

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Async variant — defaults to sync-in-thread."""
        return await asyncio.to_thread(self.embed_query, text, **kwargs)

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Batch embed multiple texts. Default: sequential fallback."""
        return [self.embed_query(t, **kwargs) for t in texts]

    async def aembed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Async batch embed. Default: sync-in-thread."""
        return await asyncio.to_thread(self.embed_documents, texts, **kwargs)


class LLMInterface(ABC):
    """Abstract base class for LLM providers.

    Supports both simple text-in/text-out and structured output
    (via ``response_model``).  Async variant defaults to sync-in-thread.

    Attributes:
        model_name: Identifier of the model (e.g. ``"gpt-4o"``).
        model_params: Provider-specific parameters (temperature, etc.).
    """

    def __init__(
        self,
        model_name: str,
        model_params: dict[str, Any] | None = None,
        max_concurrency: int = 12,
    ) -> None:
        self.model_name = model_name
        self.model_params = model_params or {}
        self.max_concurrency = max_concurrency

    @abstractmethod
    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Synchronous text-in / text-out invocation."""
        ...

    async def ainvoke(
        self,
        prompt: str,
        *,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async variant with retry + exponential backoff.

        Retries on any exception up to ``max_retries`` times with
        1s / 2s / 4s delays between attempts.
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return await asyncio.to_thread(self.invoke, prompt, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    delay = 2**attempt  # 1, 2, 4
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s: {exc}"
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def invoke_with_model(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        **kwargs: Any,
    ) -> BaseModel:
        """Invoke LLM requesting structured output validated against a Pydantic model.

        Default: call ``invoke()`` and parse JSON into ``response_model``.
        Override for providers with native structured-output support.
        """
        response = self.invoke(prompt, **kwargs)
        return response_model.model_validate_json(response.content)

    async def ainvoke_with_model(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        *,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> BaseModel:
        """Async structured output invocation with retry."""
        response = await self.ainvoke(prompt, max_retries=max_retries, **kwargs)
        return response_model.model_validate_json(response.content)

    async def abatch_invoke(
        self,
        prompts: list[str],
        *,
        max_concurrency: int | None = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> list[LLMBatchItem]:
        """Invoke LLM on multiple prompts concurrently with per-item error handling.

        Args:
            prompts: List of prompt strings to process.
            max_concurrency: Override the instance default concurrency limit.
            max_retries: Retry count passed to each ``ainvoke`` call.
            **kwargs: Extra arguments forwarded to ``ainvoke``.

        Returns:
            List of ``LLMBatchItem`` in the same order as *prompts*.
        """
        if not prompts:
            return []

        sem = asyncio.Semaphore(max_concurrency or self.max_concurrency)

        async def _call(i: int, prompt: str) -> LLMBatchItem:
            async with sem:
                try:
                    resp = await self.ainvoke(prompt, max_retries=max_retries, **kwargs)
                    return LLMBatchItem(index=i, response=resp)
                except Exception as exc:
                    return LLMBatchItem(index=i, error=exc)

        return list(
            await asyncio.gather(*[_call(i, p) for i, p in enumerate(prompts)])
        )


# ═══════════════════════════════════════════════════════════════════
# Built-in Providers (optional dependencies)
# ═══════════════════════════════════════════════════════════════════


class LiteLLM(LLMInterface):
    """LLM provider backed by LiteLLM.

    Supports 100+ providers via a unified interface.  Requires
    ``pip install graphrag-sdk[litellm]``.

    Example::

        llm = LiteLLM(model="azure/gpt-4", api_key="...", api_base="...", api_version="...")
        response = llm.invoke("Hello!")
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model)
        self._api_key = api_key
        self._api_base = api_base
        self._api_version = api_version
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra = kwargs

    def _completion_kwargs(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            **self._extra,
            **kwargs,
        }
        if self._api_key is not None:
            kw["api_key"] = self._api_key
        if self._api_base is not None:
            kw["api_base"] = self._api_base
        if self._api_version is not None:
            kw["api_version"] = self._api_version
        if self._max_tokens is not None:
            kw["max_tokens"] = self._max_tokens
        return kw

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "LiteLLM is required for the LiteLLM provider. "
                "Install with: pip install graphrag-sdk[litellm]"
            )
        response = litellm.completion(**self._completion_kwargs(prompt, **kwargs))
        content = response.choices[0].message.content or ""
        return LLMResponse(content=content)

    async def ainvoke(
        self,
        prompt: str,
        *,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> LLMResponse:
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "LiteLLM is required for the LiteLLM provider. "
                "Install with: pip install graphrag-sdk[litellm]"
            )
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = await litellm.acompletion(
                    **self._completion_kwargs(prompt, **kwargs)
                )
                content = response.choices[0].message.content or ""
                return LLMResponse(content=content)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    delay = 2**attempt
                    logger.warning(
                        f"LiteLLM call failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s: {exc}"
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]


class LiteLLMEmbedder(Embedder):
    """Embedding provider backed by LiteLLM.

    Supports OpenAI, Azure, Cohere, and other embedding models.
    Requires ``pip install graphrag-sdk[litellm]``.

    Example::

        embedder = LiteLLMEmbedder(model="azure/text-embedding-ada-002",
                                    api_key="...", api_base="...", api_version="...")
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        batch_size: int = 2048,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._api_base = api_base
        self._api_version = api_version
        self.batch_size = batch_size
        self._extra = kwargs

    def _embedding_kwargs(self, input_: str | list[str], **kwargs: Any) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "model": self.model,
            "input": input_,
            **self._extra,
            **kwargs,
        }
        if self._api_key is not None:
            kw["api_key"] = self._api_key
        if self._api_base is not None:
            kw["api_base"] = self._api_base
        if self._api_version is not None:
            kw["api_version"] = self._api_version
        return kw

    def _import_litellm(self):
        try:
            import litellm
            return litellm
        except ImportError:
            raise ImportError(
                "LiteLLM is required for the LiteLLMEmbedder provider. "
                "Install with: pip install graphrag-sdk[litellm]"
            )

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        litellm = self._import_litellm()
        response = litellm.embedding(**self._embedding_kwargs(text, **kwargs))
        return response.data[0]["embedding"]

    def _embed_batch_sync(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float] | None]:
        """Embed a batch with binary-split recovery on failure."""
        litellm = self._import_litellm()
        try:
            response = litellm.embedding(**self._embedding_kwargs(texts, **kwargs))
            sorted_data = sorted(response.data, key=lambda x: x["index"])
            return [d["embedding"] for d in sorted_data]
        except Exception:
            if len(texts) == 1:
                logger.warning("Embedding failed for text (len=%d): skipped", len(texts[0]))
                return [None]
            mid = len(texts) // 2
            left = self._embed_batch_sync(texts[:mid], **kwargs)
            right = self._embed_batch_sync(texts[mid:], **kwargs)
            return left + right

    async def _embed_batch_async(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float] | None]:
        """Async embed a batch with binary-split recovery on failure."""
        litellm = self._import_litellm()
        try:
            response = await litellm.aembedding(**self._embedding_kwargs(texts, **kwargs))
            sorted_data = sorted(response.data, key=lambda x: x["index"])
            return [d["embedding"] for d in sorted_data]
        except Exception:
            if len(texts) == 1:
                logger.warning("Embedding failed for text (len=%d): skipped", len(texts[0]))
                return [None]
            mid = len(texts) // 2
            left = await self._embed_batch_async(texts[:mid], **kwargs)
            right = await self._embed_batch_async(texts[mid:], **kwargs)
            return left + right

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float] | None]:
        if not texts:
            return []
        results: list[list[float] | None] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            results.extend(self._embed_batch_sync(batch, **kwargs))
        return results

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        litellm = self._import_litellm()
        response = await litellm.aembedding(**self._embedding_kwargs(text, **kwargs))
        return response.data[0]["embedding"]

    async def aembed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float] | None]:
        if not texts:
            return []
        results: list[list[float] | None] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            results.extend(await self._embed_batch_async(batch, **kwargs))
        return results


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
        }
        if self._max_tokens is not None:
            create_kwargs["max_tokens"] = self._max_tokens
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
        }
        if self._max_tokens is not None:
            create_kwargs["max_tokens"] = self._max_tokens
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

    def _embed_batch_sync(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float] | None]:
        """Embed a batch with binary-split recovery on failure."""
        client = self._get_client()
        try:
            response = client.embeddings.create(model=self.model, input=texts)
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [d.embedding for d in sorted_data]
        except Exception:
            if len(texts) == 1:
                logger.warning("Embedding failed for text (len=%d): skipped", len(texts[0]))
                return [None]
            mid = len(texts) // 2
            left = self._embed_batch_sync(texts[:mid], **kwargs)
            right = self._embed_batch_sync(texts[mid:], **kwargs)
            return left + right

    async def _embed_batch_async(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float] | None]:
        """Async embed a batch with binary-split recovery on failure."""
        client = self._get_async_client()
        try:
            response = await client.embeddings.create(model=self.model, input=texts)
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [d.embedding for d in sorted_data]
        except Exception:
            if len(texts) == 1:
                logger.warning("Embedding failed for text (len=%d): skipped", len(texts[0]))
                return [None]
            mid = len(texts) // 2
            left = await self._embed_batch_async(texts[:mid], **kwargs)
            right = await self._embed_batch_async(texts[mid:], **kwargs)
            return left + right

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float] | None]:
        if not texts:
            return []
        results: list[list[float] | None] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            results.extend(self._embed_batch_sync(batch, **kwargs))
        return results

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        client = self._get_async_client()
        response = await client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    async def aembed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float] | None]:
        if not texts:
            return []
        results: list[list[float] | None] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            results.extend(await self._embed_batch_async(batch, **kwargs))
        return results
