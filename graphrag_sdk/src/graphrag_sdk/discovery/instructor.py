# GraphRAG SDK — Discovery: validation-retry LLM wrapper
#
# Asks the LLM for structured output, validates the response against a
# Pydantic model (plus an optional semantic validator), and on failure
# feeds the specific error back to the LLM and retries. The
# conversation is preserved across attempts so the model sees its own
# rejected response and the precise correction it needs to make.
#
# Used by every LLM call in the discovery pipeline (doc summary,
# per-chunk proposal, normalization). The wrapper is strict — on
# exhausted retries it raises ``OntologyDiscoveryError`` — and the
# pipeline above it decides how to react (skip a chunk, fall back to
# the un-normalized draft, etc.).

from __future__ import annotations

import asyncio
import logging
import random
import re
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from graphrag_sdk.core.models import ChatMessage, LLMResponse
from graphrag_sdk.core.providers._retry import (
    caller_retries_transient_errors,
    describe_llm_error,
    is_transient_llm_error,
    provider_refuses_retry,
    retry_after_seconds,
)
from graphrag_sdk.core.providers.base import LLMInterface
from graphrag_sdk.discovery.prompts import (
    format_parse_feedback,
    format_validation_feedback,
)
from graphrag_sdk.discovery.proposal import OntologyDiscoveryError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# LLMs occasionally wrap JSON in ```json … ``` fences despite being told
# not to. Cheap to handle defensively rather than spend a retry on it.
_FENCED = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _strip_fences(content: str) -> str:
    stripped = content.strip()
    match = _FENCED.match(stripped)
    return match.group(1).strip() if match else stripped


#: Waits before each retry of a call that failed for a temporary reason
#: (see ``is_transient_llm_error``). Its length is the retry count.
_TRANSIENT_BACKOFF_S = (1.0, 4.0)

#: Longest wait a provider's Retry-After is honoured for. Asking for longer
#: (a quota window, a long outage) ends the retries instead of stalling the run.
_MAX_RETRY_AFTER_S = 60.0


#: What a ``chunk_id`` prefix says about the call, for the retry warning. The
#: rest of the id is left out of the log: it can carry a document id, often a
#: file path.
_STAGES = {
    "summary": "discovery summary",
    "normalize": "discovery normalization",
    "trim": "grounded trim",
    "table": "table mapping proposal",
}


def _stage(chunk_id: str | None) -> str:
    if not chunk_id:
        return "model"
    return _STAGES.get(chunk_id.split(":", 1)[0], "discovery chunk proposal")


def _jittered(delay: float) -> float:
    """``delay`` spread over 0.5x-1.5x, as ``LLMInterface.ainvoke`` does.

    Discovery runs several calls at once; calls that hit the same rate limit
    or blip would otherwise all resend in the same instant.
    """
    return delay * (0.5 + random.random())


async def _call_once(
    llm: LLMInterface, history: list[ChatMessage], *, stage: str = "model"
) -> LLMResponse:
    """One logical model call, retried apart from the parse budget when it fails temporarily.

    The provider's own retry loop is disabled (``max_retries=1``) so that the
    validation loop's attempt count stays exact. That also stopped any retry
    of a call that never produced a response, so one 5xx from the provider
    failed the whole discovery run. Those are retried here; the conversation
    is resent unchanged, because nothing came back to correct.

    This is a stopgap at the discovery layer. The provider loops still retry
    every exception without classifying it (issue #386); once they retry only
    temporary failures, honour Retry-After and are the single retry layer,
    discovery can pass a normal ``max_retries`` and this loop can go.
    """
    for retry, delay in enumerate((*_TRANSIENT_BACKOFF_S, None)):
        try:
            if delay is None:  # the last try: its failure is final
                return await llm.ainvoke_messages(history, max_retries=1)
            with caller_retries_transient_errors():
                return await llm.ainvoke_messages(history, max_retries=1)
        except Exception as exc:
            if delay is None or not is_transient_llm_error(exc) or provider_refuses_retry(exc):
                raise
            # A 429 or 503 often says how long to wait; resending sooner is
            # refused again, and a wait past the cap is not worth stalling for.
            wait = _jittered(delay)
            asked = retry_after_seconds(exc)
            if asked is not None:
                if asked > _MAX_RETRY_AFTER_S:
                    raise
                wait = max(wait, asked)
            # Only the exception type and HTTP status: the chunk id carries the
            # document id (often a file path) and the exception text can carry
            # request content, and a WARNING may reach shared log streams.
            logger.warning(
                "%s call failed (retry %d/%d in %.1fs): %s",
                stage,
                retry + 1,
                len(_TRANSIENT_BACKOFF_S),
                wait,
                describe_llm_error(exc),
            )
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover


async def extract_with_retry(
    llm: LLMInterface,
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    *,
    extra_validate: Callable[[T], list[str]] | None = None,
    max_retries: int = 3,
    chunk_id: str | None = None,
) -> T:
    """Call the LLM for structured output, retrying on validation failure.

    On each attempt the wrapper:

    1. Sends the current conversation to the LLM.
    2. Strips markdown fences if present and parses the response into
       ``response_model``. A Pydantic ``ValidationError`` becomes
       targeted JSON-shape feedback for the next turn.
    3. Runs ``extra_validate`` (if supplied) for semantic checks
       beyond JSON shape (e.g. "this relation references an
       undeclared entity"). Errors become bullet-pointed feedback.

    Feedback is appended as a new ``user`` message and the
    model's own rejected response stays in the history, so the LLM
    sees what it sent and what to correct.

    Args:
        llm: Any ``LLMInterface`` — uses ``ainvoke_messages``.
        system_prompt: System message defining the role and rules.
        user_prompt: Initial user request.
        response_model: Pydantic class the response must validate against.
        extra_validate: Returns a list of error strings; empty list means
            valid. Runs only after a successful Pydantic parse.
        max_retries: Number of *retry* attempts after the first, each one a
            response that came back but failed to parse or validate.
            ``max_retries=3`` therefore allows up to 4 responses.
            Each attempt may also resend after a temporary provider failure
            (up to twice, see ``_call_once``), so up to 12 provider calls.
        chunk_id: Identifier surfaced on ``OntologyDiscoveryError`` for
            traceability.

    Returns:
        A validated instance of ``response_model``.

    Raises:
        OntologyDiscoveryError: All attempts exhausted without producing
            a response that parses and passes ``extra_validate``.
        Exception: The provider's own error, unchanged, when a call fails
            permanently (a 4xx) or a temporary failure outlasts the retries.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    history: list[ChatMessage] = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]
    last_error: Exception | None = None
    attempts = 0
    total_attempts = max_retries + 1

    for attempt in range(total_attempts):
        attempts += 1
        # A temporary provider failure is retried inside ``_call_once`` and
        # is not counted as an attempt; see its docstring for why the
        # provider's own retry loop is off.
        response = await _call_once(llm, history, stage=_stage(chunk_id))
        history.append(ChatMessage(role="assistant", content=response.content))

        text = _strip_fences(response.content)

        try:
            parsed = response_model.model_validate_json(text)
        except ValidationError as exc:
            last_error = exc
            logger.debug(
                "discovery parse failed (chunk=%s, attempt=%d/%d): %s",
                chunk_id,
                attempt + 1,
                total_attempts,
                exc,
            )
            if attempt < max_retries:
                history.append(
                    ChatMessage(
                        role="user",
                        content=format_parse_feedback(str(exc)),
                    )
                )
            continue

        if extra_validate is not None:
            errors = extra_validate(parsed)
            if errors:
                last_error = ValueError("; ".join(errors))
                logger.debug(
                    "discovery validate failed (chunk=%s, attempt=%d/%d): %s",
                    chunk_id,
                    attempt + 1,
                    total_attempts,
                    errors,
                )
                if attempt < max_retries:
                    history.append(
                        ChatMessage(
                            role="user",
                            content=format_validation_feedback(errors),
                        )
                    )
                continue

        return parsed

    raise OntologyDiscoveryError(
        f"Failed to obtain a valid {response_model.__name__} after "
        f"{attempts} attempt(s)" + (f" for chunk {chunk_id}" if chunk_id else ""),
        chunk_id=chunk_id,
        attempts=attempts,
        last_error=last_error,
    )
