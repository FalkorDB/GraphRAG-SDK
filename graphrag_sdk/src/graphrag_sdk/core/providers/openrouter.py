# GraphRAG SDK — OpenRouter Provider
# LLM and Embedder backed by OpenRouter (uses the OpenAI SDK).
# Requires: pip install graphrag-sdk[openrouter]

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from typing import Any

from graphrag_sdk.core.exceptions import EmbeddingTimeoutError, LLMTimeoutError
from graphrag_sdk.core.models import ChatMessage, LLMResponse
from graphrag_sdk.core.providers._retry import (
    binary_split_retry_async,
    binary_split_retry_sync,
    final_failure_log_level,
    summarize_exception,
)
from graphrag_sdk.core.providers._sampling import SamplingParams, name_says_reasoning
from graphrag_sdk.core.providers._timeout import validate_timeout, wait_for_provider_call
from graphrag_sdk.core.providers.base import Embedder, LLMInterface

logger = logging.getLogger(__name__)


class OpenRouterLLM(LLMInterface):
    """LLM provider backed by OpenRouter (uses the OpenAI SDK).

    Requires ``pip install graphrag-sdk[openrouter]``.

    Example::

        llm = OpenRouterLLM(model="anthropic/claude-sonnet-4-20250514", api_key="...")
        response = llm.invoke("Hello!")

    Reasoning models need no setting: when a model rejects ``temperature``
    or ``max_tokens``, the request goes again without the temperature, or
    with ``max_completion_tokens``, and so do the instance's later requests.
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
        self._sampling = SamplingParams(model)

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Whether the name marks a reasoning model (o-series, gpt-5, gpt-6).

        Only the first guess of how to send ``temperature`` and the token
        cap; a rejected request corrects it (see ``SamplingParams``).
        """
        return name_says_reasoning(model)

    def _build_create_kwargs(
        self,
        messages: list[dict[str, str]],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble ``client.chat.completions.create`` kwargs.

        Caller values (via ``extra``) win over the instance defaults; a
        temperature the model rejects is dropped whoever set it, and
        ``max_tokens`` is renamed for a model that wants
        ``max_completion_tokens`` (see ``SamplingParams``).
        """
        kw: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            **extra,
        }
        self._sampling.apply(kw, self._temperature, self._max_tokens)
        return kw

    def _create(self, client: Any, build: Callable[[], dict[str, Any]]) -> Any:
        """``chat.completions.create`` on ``build()``, sent again when a parameter is rejected."""
        while True:
            kw = build()
            try:
                return client.chat.completions.create(**kw)
            except Exception as exc:
                if not self._sampling.adapt(exc, kw):
                    raise

    async def _acreate(
        self,
        client: Any,
        build: Callable[[], dict[str, Any]],
        *,
        timeout: float | None,
        operation: str,
    ) -> Any:
        """Async ``chat.completions.create`` on ``build()``, sent again on a rejected parameter.

        The request sent again is a new request, with a fresh *timeout*; it
        is not one of the caller's ``max_retries``.
        """
        while True:
            kw = build()
            try:
                return await wait_for_provider_call(
                    client.chat.completions.create(**kw),
                    timeout=timeout,
                    timeout_error=LLMTimeoutError,
                    operation=operation,
                )
            except Exception as exc:
                if not self._sampling.adapt(exc, kw):
                    raise

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
        messages = [{"role": "user", "content": prompt}]
        response = self._create(client, lambda: self._build_create_kwargs(messages, kwargs))
        content = response.choices[0].message.content or ""
        return LLMResponse(content=content)

    async def ainvoke(
        self,
        prompt: str,
        *,
        max_retries: int = 3,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        validate_timeout(timeout)
        client = self._get_async_client()
        prompt_messages = [{"role": "user", "content": prompt}]
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = await self._acreate(
                    client,
                    lambda: self._build_create_kwargs(prompt_messages, kwargs),
                    timeout=timeout,
                    operation=f"OpenRouter call to {self.model_name}",
                )
                content = response.choices[0].message.content or ""
                return LLMResponse(content=content)
            except LLMTimeoutError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    delay = 2**attempt
                    logger.warning(
                        "OpenRouter call failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        max_retries,
                        delay,
                        summarize_exception(exc),
                    )
                    logger.debug("OpenRouter call failure details", exc_info=True)
                    await asyncio.sleep(delay)
        # ERROR, unless the caller retries this temporary failure itself
        # (ontology discovery): see final_failure_log_level.
        logger.log(
            final_failure_log_level(last_exc),
            "OpenRouter call failed after %d attempts: %s",
            max_retries,
            summarize_exception(last_exc) if last_exc is not None else "UnknownError",
        )
        if last_exc is not None:
            logger.debug(
                "OpenRouter call final failure details",
                exc_info=(type(last_exc), last_exc, last_exc.__traceback__),
            )
        raise last_exc  # type: ignore[misc]

    async def ainvoke_messages(
        self,
        messages: list[ChatMessage],
        *,
        max_retries: int = 3,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Native multi-turn completion via OpenRouter."""
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        validate_timeout(timeout)
        client = self._get_async_client()
        message_dicts = [m.to_dict() for m in messages]
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = await self._acreate(
                    client,
                    lambda: self._build_create_kwargs(message_dicts, kwargs),
                    timeout=timeout,
                    operation=f"OpenRouter messages call to {self.model_name}",
                )
                content = response.choices[0].message.content or ""
                return LLMResponse(content=content)
            except LLMTimeoutError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    delay = 2**attempt
                    logger.warning(
                        "OpenRouter call failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        max_retries,
                        delay,
                        summarize_exception(exc),
                    )
                    logger.debug("OpenRouter call failure details", exc_info=True)
                    await asyncio.sleep(delay)
        # ERROR, unless the caller retries this temporary failure itself
        # (ontology discovery): see final_failure_log_level.
        logger.log(
            final_failure_log_level(last_exc),
            "OpenRouter messages call failed after %d attempts: %s",
            max_retries,
            summarize_exception(last_exc) if last_exc is not None else "UnknownError",
        )
        if last_exc is not None:
            logger.debug(
                "OpenRouter messages call final failure details",
                exc_info=(type(last_exc), last_exc, last_exc.__traceback__),
            )
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

    @property
    def model_name(self) -> str:
        """Identifier of the embedding model."""
        return self.model

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
        response = client.embeddings.create(model=self.model, input=text, **kwargs)
        return response.data[0].embedding

    def _raw_embed_sync(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Raw sync embed without retry — called by binary_split_retry_sync."""
        client = self._get_client()
        response = client.embeddings.create(model=self.model, input=texts, **kwargs)
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]

    async def _raw_embed_async(
        self,
        texts: list[str],
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Raw async embed without retry — called by binary_split_retry_async."""
        validate_timeout(timeout)
        client = self._get_async_client()
        response = await wait_for_provider_call(
            client.embeddings.create(model=self.model, input=texts, **kwargs),
            timeout=timeout,
            timeout_error=EmbeddingTimeoutError,
            operation=f"OpenRouter embedding batch with {self.model}",
        )
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

    async def aembed_query(
        self,
        text: str,
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> list[float]:
        validate_timeout(timeout)
        client = self._get_async_client()
        response = await wait_for_provider_call(
            client.embeddings.create(model=self.model, input=text, **kwargs),
            timeout=timeout,
            timeout_error=EmbeddingTimeoutError,
            operation=f"OpenRouter embedding query with {self.model}",
        )
        return response.data[0].embedding

    async def aembed_documents(
        self,
        texts: list[str],
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        validate_timeout(timeout)
        if not texts:
            return []
        deadline = time.monotonic() + timeout if timeout is not None else None

        def remaining_timeout() -> float | None:
            if deadline is None:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EmbeddingTimeoutError(
                    f"OpenRouter embedding documents with {self.model} timed out after "
                    f"{timeout:.3g}s"
                )
            return remaining

        async def embed_batch_with_deadline(
            batch: list[str], **inner_kwargs: Any
        ) -> list[list[float]]:
            return await self._raw_embed_async(
                batch,
                timeout=remaining_timeout(),
                **inner_kwargs,
            )

        results: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            results.extend(
                await binary_split_retry_async(
                    embed_batch_with_deadline,
                    batch,
                    **kwargs,
                )
            )
        return results
