# GraphRAG SDK — temperature and token cap per chat model
#
# Reasoning models (OpenAI o-series, gpt-5, gpt-6 …) reject a non-default
# ``temperature`` and take ``max_completion_tokens`` instead of ``max_tokens``.
# A provider starts from a guess by model name, then learns from the answers:
# when a request is rejected because of one of these two parameters, the
# parameter is dropped (``temperature``, at that request's reasoning effort) or
# renamed (``max_tokens``) for that LLM instance and the request is sent again
# at once.
#
# LiteLLM raises that rejection itself, before any network call, for every
# model it knows (it also knows that ``reasoning_effort="none"`` lets a
# gpt-5.x model take a temperature). An Azure deployment name, or a model newer
# than the installed LiteLLM, gets the same rejection as the provider's 400
# answer. Either way the model works without the user naming it a reasoning
# model, and a model whose name is not recognised costs at most one rejected
# request per parameter.

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Names treated as reasoning models before any request is sent. Only a first
# guess: a model missing here is learned from its first rejected request.
REASONING_NAME_PREFIXES = ("o1", "o3", "o4", "gpt-5", "gpt-6")

# Wording of a rejected parameter: LiteLLM's own check ("gpt-5.5 doesn't
# support temperature=0.0 …", "O-series models don't support temperature=0.0")
# and OpenAI / Azure 400 answers ("Unsupported value: 'temperature' does not
# support 0.0 …", "Unsupported parameter: 'max_tokens' is not supported …").
_REJECTION_RE = re.compile(
    r"unsupported|not supported|does not support|doesn't support|don't support",
    re.IGNORECASE,
)
# Whole names only: ``max_completion_tokens`` must not read as ``max_tokens``.
_PARAM_RES = {
    "temperature": re.compile(r"(?<!\w)temperature(?!\w)"),
    "max_tokens": re.compile(r"(?<!\w)max_tokens(?!\w)"),
}


def name_says_reasoning(model: str) -> bool:
    """Whether the model's name marks it as a reasoning model.

    The provider prefix (``openai/``, ``azure/`` …) is ignored.
    """
    name = model.split("/")[-1].lower()
    return name.startswith(REASONING_NAME_PREFIXES)


def rejected_params(exc: BaseException) -> set[str]:
    """``temperature`` / ``max_tokens``, whichever *exc* says the model rejected.

    Only a 400 answer counts (LiteLLM's ``UnsupportedParamsError`` and the
    ``BadRequestError`` of LiteLLM and the OpenAI SDK all carry
    ``status_code == 400``), and only when its message says a parameter is
    not supported.
    """
    if getattr(exc, "status_code", None) != 400:
        return set()
    text = str(exc)
    if not _REJECTION_RE.search(text):
        return set()
    return {name for name, pattern in _PARAM_RES.items() if pattern.search(text)}


def _effort_of(kw: dict[str, Any]) -> str | None:
    """The request's ``reasoning_effort``, as a string; None when it sets none."""
    effort = kw.get("reasoning_effort")
    return None if effort is None else str(effort)


class SamplingParams:
    """How one LLM instance sends ``temperature`` and its token cap.

    Starts from the model name (see :data:`REASONING_NAME_PREFIXES`) and
    changes when a request is rejected over one of the two (:meth:`adapt`).

    Whether a temperature goes depends on each request's
    ``reasoning_effort`` — set on the LLM or passed with the call: with
    ``"none"`` a gpt-5.x model takes one, so the name alone never drops it.
    A rejected temperature is remembered for the effort it was sent at only,
    so one call at ``"low"`` does not cost the later ``"none"`` calls their
    temperature. A rejected ``max_tokens`` is renamed for every request.
    Changes only ever go one way, so a request is sent again at most twice.
    """

    def __init__(self, model: str) -> None:
        self._model = model
        self._name_says_reasoning = name_says_reasoning(model)
        # Efforts (a request's ``reasoning_effort``; None when unset) at which
        # the model rejected a temperature.
        self._temperature_rejected_at: set[str | None] = set()
        self.use_max_completion_tokens = self._name_says_reasoning

    def omits_temperature(self, effort: str | None) -> bool:
        """Whether a request at ``reasoning_effort`` *effort* goes without a temperature."""
        if effort in self._temperature_rejected_at:
            return True
        return self._name_says_reasoning and effort != "none"

    def apply(self, kw: dict[str, Any], temperature: float, max_tokens: int | None) -> None:
        """Set the temperature and the token cap on request kwargs *kw*, in place.

        *kw* already holds the caller's arguments, which win over the
        instance defaults *temperature* and *max_tokens*: a caller's
        ``max_tokens`` (renamed when the model wants ``max_completion_tokens``)
        replaces the default, ``max_tokens=None`` from the caller means no
        cap, and the default never overwrites a caller's
        ``max_completion_tokens``. A dropped temperature is dropped whoever
        set it.
        """
        if self.omits_temperature(_effort_of(kw)):
            kw.pop("temperature", None)
        else:
            kw.setdefault("temperature", temperature)
        key = "max_completion_tokens" if self.use_max_completion_tokens else "max_tokens"
        if "max_tokens" in kw:
            cap = kw.pop("max_tokens")
            if cap is not None:
                kw[key] = cap
        elif max_tokens is not None:
            kw.setdefault(key, max_tokens)

    def adapt(self, exc: BaseException, sent: dict[str, Any]) -> bool:
        """Learn from *exc*, the answer to request kwargs *sent*.

        Returns True when *exc* rejected a parameter that *sent* carried and
        that the same request would no longer carry, i.e. when it should go
        again now. A request built before another thread made the same change
        also gets True, so it is not failed for what was already fixed.
        """
        rejected = rejected_params(exc)
        retry = False
        if "temperature" in rejected and "temperature" in sent:
            effort = _effort_of(sent)
            if effort not in self._temperature_rejected_at:
                self._temperature_rejected_at.add(effort)
                at = "" if effort is None else f" at reasoning_effort={effort!r}"
                logger.info(
                    "%s does not accept temperature=%s%s; such requests go without a temperature",
                    self._model,
                    sent["temperature"],
                    at,
                )
            retry = True
        if "max_tokens" in rejected and "max_tokens" in sent:
            if not self.use_max_completion_tokens:
                self.use_max_completion_tokens = True
                logger.info(
                    "%s does not accept max_tokens; its requests send max_completion_tokens",
                    self._model,
                )
            retry = True
        return retry
