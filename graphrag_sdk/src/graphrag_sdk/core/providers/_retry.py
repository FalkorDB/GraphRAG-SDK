# GraphRAG SDK — Embedding retry utilities
# Binary-split recovery for transient batch embedding failures.

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from graphrag_sdk.core.exceptions import EmbeddingTimeoutError, LLMTimeoutError

logger = logging.getLogger(__name__)

_MAX_EXC_SUMMARY_LEN = 200


def summarize_exception(exc: BaseException) -> str:
    """Render an exception for a sanitized one-line WARNING log.

    Returns ``ExcType: <first-line, truncated>``. Provider exception strings
    can include request payloads, response bodies, or proxy URLs; those
    belong at DEBUG via ``exc_info=``, not in a single-line WARNING that may
    be aggregated into shared log streams.
    """
    text = str(exc)
    msg = text.splitlines()[0] if text else ""
    if len(msg) > _MAX_EXC_SUMMARY_LEN:
        msg = msg[:_MAX_EXC_SUMMARY_LEN] + "..."
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


#: HTTP statuses a provider returns for a condition that passes on its own:
#: request timeout, conflict, rate limit. Every 5xx counts as well.
_TRANSIENT_STATUSES = frozenset({408, 409, 429})


def _status_code(exc: BaseException) -> int | None:
    """The HTTP status a provider exception carries, on itself or its response."""
    for owner in (exc, getattr(exc, "response", None)):
        status = getattr(owner, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool):
            return status
    return None


def is_transient_llm_error(exc: BaseException) -> bool:
    """Return True if an LLM call failed for a reason worth retrying shortly.

    Provider exceptions (LiteLLM, OpenAI, Azure) carry an HTTP status: a
    5xx, 408, 409 or 429 is temporary, anything else (400, 401, 403, 404,
    422, a content filter) will fail the same way again. Without a status,
    only a dropped connection counts: the built-in ``ConnectionError``, or
    the OpenAI client's ``APIConnectionError`` (raised by the OpenRouter
    provider, with no status and outside the built-in hierarchy). An
    ``LLMTimeoutError`` is not retried: the timeout is the caller's
    deadline, and retrying would stretch it.
    """
    if isinstance(exc, LLMTimeoutError):
        return False
    status = _status_code(exc)
    if status is not None:
        return 500 <= status < 600 or status in _TRANSIENT_STATUSES
    return isinstance(exc, (ConnectionError, *_provider_connection_errors()))


def _provider_connection_errors() -> tuple[type[BaseException], ...]:
    """Status-less connection errors of optional provider clients that are installed."""
    try:
        from openai import APIConnectionError
    except ImportError:
        return ()
    return (APIConnectionError,)


def is_transient_embedding_error(exc: Exception) -> bool:
    """Return True if the embedding error is transient (worth retrying via split).

    Non-transient errors (auth failures, invalid key, permission denied)
    should be raised immediately instead of binary-splitting down to
    single texts.
    """
    msg = str(exc).lower()
    non_transient = (
        "401",
        "403",
        "invalid api key",
        "authentication",
        "unauthorized",
        "permission denied",
        "invalid_api_key",
    )
    return not any(marker in msg for marker in non_transient)


def binary_split_retry_sync(
    embed_fn: Callable[..., list[list[float]]],
    texts: list[str],
    **kwargs: Any,
) -> list[list[float]]:
    """Try to embed *texts* as one batch; on transient failure, split and retry.

    Args:
        embed_fn: Callable that takes a list of texts and returns embeddings.
        texts: Texts to embed.
        **kwargs: Extra keyword arguments forwarded to *embed_fn*.

    Returns:
        List of embedding vectors (one per text). Failed single-text
        embeddings return ``[[]]`` (empty vector) instead of raising.
    """
    try:
        return embed_fn(texts, **kwargs)
    except EmbeddingTimeoutError:
        raise
    except Exception as exc:
        if not is_transient_embedding_error(exc):
            logger.error("Non-transient embedding failure: %s", summarize_exception(exc))
            logger.debug("Non-transient embedding failure details", exc_info=True)
            raise
        if len(texts) == 1:
            logger.warning("Embedding failed for text (len=%d): skipped", len(texts[0]))
            return [[]]
        mid = len(texts) // 2
        left = binary_split_retry_sync(embed_fn, texts[:mid], **kwargs)
        right = binary_split_retry_sync(embed_fn, texts[mid:], **kwargs)
        return left + right


async def binary_split_retry_async(
    embed_fn: Callable[..., Awaitable[list[list[float]]]],
    texts: list[str],
    **kwargs: Any,
) -> list[list[float]]:
    """Async variant of :func:`binary_split_retry_sync`."""
    try:
        return await embed_fn(texts, **kwargs)
    except EmbeddingTimeoutError:
        raise
    except Exception as exc:
        if not is_transient_embedding_error(exc):
            logger.error("Non-transient embedding failure: %s", summarize_exception(exc))
            logger.debug("Non-transient embedding failure details", exc_info=True)
            raise
        if len(texts) == 1:
            logger.warning("Embedding failed for text (len=%d): skipped", len(texts[0]))
            return [[]]
        mid = len(texts) // 2
        left = await binary_split_retry_async(embed_fn, texts[:mid], **kwargs)
        right = await binary_split_retry_async(embed_fn, texts[mid:], **kwargs)
        return left + right
