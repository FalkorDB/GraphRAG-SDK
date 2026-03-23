# GraphRAG SDK 2.0 — Core: Provider ABCs
# Thin abstract interfaces for Embedders and LLMs.

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Type

from pydantic import BaseModel

from graphrag_sdk.core.models import LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class LLMBatchItem:
    """Result of one item in a batch invocation."""

    index: int
    response: LLMResponse | None = None
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.response is not None


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
        """Async variant with retry + jittered exponential backoff.

        Retries on any exception up to ``max_retries`` times with
        jittered delays between attempts.
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return await asyncio.to_thread(self.invoke, prompt, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    delay = (2**attempt) * (0.5 + random.random())
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay:.1f}s: {exc}"
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Async streaming — default yields the full response as one chunk."""
        resp = await self.ainvoke(prompt, **kwargs)
        yield resp.content

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
