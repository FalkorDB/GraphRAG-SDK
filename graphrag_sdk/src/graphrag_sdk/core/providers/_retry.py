# GraphRAG SDK — Embedding retry utilities
# Binary-split recovery for transient batch embedding failures.

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

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
    except Exception as exc:
        if not is_transient_embedding_error(exc):
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
    except Exception as exc:
        if not is_transient_embedding_error(exc):
            raise
        if len(texts) == 1:
            logger.warning("Embedding failed for text (len=%d): skipped", len(texts[0]))
            return [[]]
        mid = len(texts) // 2
        left = await binary_split_retry_async(embed_fn, texts[:mid], **kwargs)
        right = await binary_split_retry_async(embed_fn, texts[mid:], **kwargs)
        return left + right
