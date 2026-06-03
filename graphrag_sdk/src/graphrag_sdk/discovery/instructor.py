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

import logging
import re
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from graphrag_sdk.core.models import ChatMessage
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
        max_retries: Number of *retry* attempts after the first call.
            ``max_retries=3`` therefore allows up to 4 LLM calls.
        chunk_id: Identifier surfaced on ``OntologyDiscoveryError`` for
            traceability.

    Returns:
        A validated instance of ``response_model``.

    Raises:
        OntologyDiscoveryError: All attempts exhausted without producing
            a response that parses and passes ``extra_validate``.
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
        # ``max_retries=1`` disables the provider-level retry loop —
        # ``extract_with_retry`` owns the retry policy. Without this the
        # default provider budget (3 attempts) would silently triple every
        # logical attempt here, blowing the documented budget and skewing
        # ``OntologyDiscoveryError.attempts``.
        response = await llm.ainvoke_messages(history, max_retries=1)
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
