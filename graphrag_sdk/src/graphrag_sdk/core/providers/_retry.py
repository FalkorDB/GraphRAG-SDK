# GraphRAG SDK — Embedding retry utilities
# Binary-split recovery for transient batch embedding failures.

from __future__ import annotations

import email.utils
import logging
import math
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
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


def _reply_header(exc: BaseException, name: str) -> str | None:
    """A header of the HTTP reply a provider exception carries, if any.

    openai and LiteLLM exceptions keep the reply on ``exc.response``
    (an ``httpx.Response``); LiteLLM also copies its headers to
    ``exc.litellm_response_headers``. Lookups are case-insensitive on
    ``httpx.Headers``; plain dicts are expected to use lowercase names.
    """
    for headers in (
        getattr(getattr(exc, "response", None), "headers", None),
        getattr(exc, "litellm_response_headers", None),
    ):
        getter = getattr(headers, "get", None)
        if getter is None:
            continue
        try:
            value = getter(name)
        except Exception:  # noqa: BLE001 - reading a header must never mask the error
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def retry_after_seconds(exc: BaseException) -> float | None:
    """How long the provider asked to wait before a retry, or ``None``.

    Reads ``retry-after-ms`` (milliseconds, sent by OpenAI and Azure), then
    ``retry-after`` (seconds or an HTTP date) from the reply the exception
    carries. A value that does not parse is ignored.
    """
    raw_ms = _reply_header(exc, "retry-after-ms")
    if raw_ms is not None:
        try:
            millis = float(raw_ms)
        except ValueError:
            millis = math.nan
        if math.isfinite(millis) and millis >= 0:
            return millis / 1000
    raw = _reply_header(exc, "retry-after")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(raw)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(seconds, 0.0)


def provider_refuses_retry(exc: BaseException) -> bool:
    """True when the provider's reply says a retry will not help.

    OpenAI sends ``x-should-retry: false`` with errors such as an exhausted
    quota, where the status alone (a 429) looks temporary.
    """
    return (_reply_header(exc, "x-should-retry") or "").lower() == "false"


def describe_llm_error(exc: BaseException) -> str:
    """``TypeName (HTTP 503)``: an LLM error for a shared log, without its text.

    The exception text can carry request content, so a WARNING that may
    reach shared log streams names only the type and the HTTP status.
    """
    status = _status_code(exc)
    name = type(exc).__name__
    return f"{name} (HTTP {status})" if status is not None else name


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
