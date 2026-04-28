# GraphRAG SDK — LiteLLM Provider
# LLM and Embedder backed by LiteLLM (100+ providers via unified interface).
# Requires: pip install graphrag-sdk[litellm]

from __future__ import annotations

import asyncio
import logging
from typing import Any

from graphrag_sdk.core.models import ChatMessage, LLMResponse
from graphrag_sdk.core.providers._retry import (
    binary_split_retry_async,
    binary_split_retry_sync,
    summarize_exception,
)
from graphrag_sdk.core.providers.base import Embedder, LLMInterface

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Return True for OpenAI reasoning models (o1 / o3 / gpt-5 family).

        These models impose two restrictions vs the chat-completion default:
        ``temperature`` only accepts the default value (1.0), and the token
        cap parameter is ``max_completion_tokens`` rather than ``max_tokens``.
        """
        # Strip provider prefix (``openai/``, ``azure/``…) before matching.
        name = model.split("/")[-1].lower()
        return name.startswith(("o1", "o3", "gpt-5"))

    def _completion_kwargs(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **self._extra,
            **kwargs,
        }
        is_reasoning = self._is_reasoning_model(self.model_name)
        if is_reasoning:
            # Reasoning models (o1/o3/gpt-5) reject explicit temperature
            # and use ``max_completion_tokens``. Strip any caller-supplied
            # values too — instance-default guards alone are not enough
            # because ``**kwargs``/``self._extra`` may have already
            # injected them above.
            kw.pop("temperature", None)
            caller_max_tokens = kw.pop("max_tokens", None)
        else:
            kw["temperature"] = self._temperature
            caller_max_tokens = None
        if self._api_key is not None:
            kw["api_key"] = self._api_key
        if self._api_base is not None:
            kw["api_base"] = self._api_base
        if self._api_version is not None:
            kw["api_version"] = self._api_version
        # Resolve max-tokens cap — caller wins over instance default.
        effective_max_tokens = caller_max_tokens if caller_max_tokens is not None else self._max_tokens
        if effective_max_tokens is not None:
            if is_reasoning:
                kw["max_completion_tokens"] = effective_max_tokens
            else:
                kw["max_tokens"] = effective_max_tokens
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
                response = await litellm.acompletion(**self._completion_kwargs(prompt, **kwargs))
                content = response.choices[0].message.content or ""
                return LLMResponse(content=content)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    delay = 2**attempt
                    logger.warning(
                        "LiteLLM call failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        max_retries,
                        delay,
                        summarize_exception(exc),
                    )
                    logger.debug("LiteLLM call failure details", exc_info=True)
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def _messages_completion_kwargs(
        self, messages: list[ChatMessage], **kwargs: Any
    ) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "model": self.model_name,
            "messages": [m.to_dict() for m in messages],
            **self._extra,
            **kwargs,
        }
        # See ``_completion_kwargs`` — same translation rules; we strip
        # caller-supplied ``temperature`` / ``max_tokens`` too, not just
        # the instance defaults.
        is_reasoning = self._is_reasoning_model(self.model_name)
        if is_reasoning:
            kw.pop("temperature", None)
            caller_max_tokens = kw.pop("max_tokens", None)
        else:
            kw["temperature"] = self._temperature
            caller_max_tokens = None
        if self._api_key is not None:
            kw["api_key"] = self._api_key
        if self._api_base is not None:
            kw["api_base"] = self._api_base
        if self._api_version is not None:
            kw["api_version"] = self._api_version
        effective_max_tokens = caller_max_tokens if caller_max_tokens is not None else self._max_tokens
        if effective_max_tokens is not None:
            if is_reasoning:
                kw["max_completion_tokens"] = effective_max_tokens
            else:
                kw["max_tokens"] = effective_max_tokens
        return kw

    async def ainvoke_messages(
        self,
        messages: list[ChatMessage],
        *,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> LLMResponse:
        """Native multi-turn completion via LiteLLM."""
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
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
                    **self._messages_completion_kwargs(messages, **kwargs)
                )
                content = response.choices[0].message.content or ""
                return LLMResponse(content=content)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    delay = 2**attempt
                    logger.warning(
                        "LiteLLM call failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        max_retries,
                        delay,
                        summarize_exception(exc),
                    )
                    logger.debug("LiteLLM call failure details", exc_info=True)
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

    @property
    def model_name(self) -> str:
        """Identifier of the embedding model."""
        return self.model

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

    def _raw_embed_sync(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Raw sync embed without retry — called by binary_split_retry_sync."""
        litellm = self._import_litellm()
        response = litellm.embedding(**self._embedding_kwargs(texts, **kwargs))
        sorted_data = sorted(response.data, key=lambda x: x["index"])
        return [d["embedding"] for d in sorted_data]

    async def _raw_embed_async(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Raw async embed without retry — called by binary_split_retry_async."""
        litellm = self._import_litellm()
        response = await litellm.aembedding(**self._embedding_kwargs(texts, **kwargs))
        sorted_data = sorted(response.data, key=lambda x: x["index"])
        return [d["embedding"] for d in sorted_data]

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            results.extend(binary_split_retry_sync(self._raw_embed_sync, batch, **kwargs))
        return results

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        litellm = self._import_litellm()
        response = await litellm.aembedding(**self._embedding_kwargs(text, **kwargs))
        return response.data[0]["embedding"]

    async def aembed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            results.extend(await binary_split_retry_async(self._raw_embed_async, batch, **kwargs))
        return results
