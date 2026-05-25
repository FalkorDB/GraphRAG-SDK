from __future__ import annotations

import asyncio
from typing import Any

from graphrag_sdk.core.exceptions import EmbeddingTimeoutError, LLMTimeoutError


async def wait_for_provider_call(
    awaitable: Any,
    *,
    timeout: float | None,
    timeout_error: type[LLMTimeoutError] | type[EmbeddingTimeoutError],
    operation: str,
) -> Any:
    if timeout is None:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError) as exc:
        raise timeout_error(f"{operation} timed out after {timeout:.3g}s") from exc


def validate_timeout(timeout: float | None) -> None:
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be > 0")
