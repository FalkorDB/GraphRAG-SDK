---
title: "Providers"
nav_order: 4
parent: "API Reference"
grand_parent: "GraphRAG-SDK"
description: "LLMInterface, Embedder ABCs and the bundled LiteLLM, OpenRouter implementations."
---

# Providers

Module: `graphrag_sdk`  ·  Submodule: `graphrag_sdk.core.providers`

LLM and embedder ABCs, plus the bundled `LiteLLM` / `OpenRouter` implementations.

---

## `LLMInterface` (ABC)

```python
class LLMInterface(ABC):
    def __init__(
        self,
        model_name: str,
        model_params: dict[str, Any] | None = None,
        max_concurrency: int = 12,
    ) -> None
```

### Subclass contract

Subclasses must implement `invoke(prompt) -> LLMResponse`. The SDK provides:

| Method | Purpose |
|---|---|
| `invoke(prompt, **kwargs) -> LLMResponse` | **abstract** — synchronous text-in/text-out. |
| `ainvoke(prompt, *, max_retries=3, timeout=None, **kwargs) -> LLMResponse` | Async variant with retry + jittered exponential backoff. Defaults to running `invoke` in a thread. |
| `ainvoke_messages(messages, *, max_retries=3, timeout=None, **kwargs) -> LLMResponse` | Multi-turn variant. Default implementation flattens `messages` into a single prompt; override for providers with native chat APIs. |
| `astream(prompt, *, timeout=None, **kwargs) -> AsyncIterator[str]` | Async streaming. Default yields the full response as one chunk. |
| `invoke_with_model(prompt, response_model, **kwargs) -> BaseModel` | Sync structured output — parses JSON into a Pydantic model. |
| `ainvoke_with_model(prompt, response_model, *, max_retries=3, timeout=None, **kwargs) -> BaseModel` | Async structured output with retry. |
| `abatch_invoke(prompts, *, max_concurrency=None, max_retries=3, timeout=None, **kwargs) -> list[LLMBatchItem]` | Concurrent batch invocation with per-item error capture. |

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | — required — | Model identifier (`"gpt-4o"`, etc.). |
| `model_params` | `dict[str, Any] \| None` | `None` | Provider-specific parameters (`temperature`, etc.). |
| `max_concurrency` | `int` | `12` | Default cap used by `abatch_invoke`. Must be ≥ 1. |

---

## `Embedder` (ABC)

```python
class Embedder(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...
    @abstractmethod
    def embed_query(self, text: str, **kwargs) -> list[float]: ...
```

| Method | Purpose |
|---|---|
| `embed_query(text, **kwargs) -> list[float]` | **abstract** — embed a single text. |
| `aembed_query(text, *, timeout=None, **kwargs) -> list[float]` | Async variant; default runs sync in a thread. |
| `embed_documents(texts, **kwargs) -> list[list[float]]` | Batch embed (default: sequential fallback). |
| `aembed_documents(texts, *, timeout=None, **kwargs) -> list[list[float]]` | Async batch embed. |

---

## `LiteLLM`

LLM provider backed by [LiteLLM](https://docs.litellm.ai/). 100+ providers behind one API.

```python
from graphrag_sdk import LiteLLM


class LiteLLM(LLMInterface):
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
    ) -> None
```

| Name | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | — required — | Provider-prefixed model id (`"openai/gpt-4o-mini"`, `"anthropic/claude-3-5-sonnet-20241022"`, `"gemini/gemini-2.0-flash"`, etc.). |
| `api_key` | `str \| None` | env-var fallback | Per-call API key. Falls back to provider-specific env var when not supplied. |
| `api_base` | `str \| None` | provider default | For self-hosted, Azure, or Ollama endpoints. |
| `api_version` | `str \| None` | provider default | Azure OpenAI API version. |
| `temperature` | `float` | `0.0` | Sampling temperature. Stripped for reasoning models. |
| `max_tokens` | `int \| None` | unset | Token cap. Translated to `max_completion_tokens` for reasoning models. |
| `**kwargs` | `Any` | — | Forwarded to `litellm.completion()`. |

### Reasoning-model handling

For OpenAI reasoning models (o1, o3, gpt-5 family), `temperature` is silently stripped and `max_tokens` becomes `max_completion_tokens` — no code change needed in caller code.

### Requires

`pip install graphrag-sdk[litellm]` (pulls the `litellm` package).

---

## `LiteLLMEmbedder`

Embedder backed by LiteLLM. OpenAI, Azure, Cohere, Voyage, Gemini, and others.

```python
from graphrag_sdk import LiteLLMEmbedder


class LiteLLMEmbedder(Embedder):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        batch_size: int = 2048,
        **kwargs: Any,
    ) -> None
```

| Name | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | — required — | Embedder model identifier. |
| `api_key` / `api_base` / `api_version` | per `LiteLLM`. |
| `batch_size` | `int` | `2048` | Max texts per `litellm.embedding(...)` call. Larger batches that exceed the provider's token budget are recursively split (binary-split-retry). |
| `**kwargs` | forwarded to `litellm.embedding()`. |

`embed_documents` and `aembed_documents` are batched and recover from over-large batches automatically via `binary_split_retry`.

---

## `OpenRouterLLM` and `OpenRouterEmbedder`

```python
from graphrag_sdk import OpenRouterLLM, OpenRouterEmbedder
```

Equivalent to `LiteLLM` / `LiteLLMEmbedder` with `api_base` pre-pointed at OpenRouter. Use these only if you've explicitly chosen OpenRouter as your gateway — `LiteLLM("openrouter/...")` works just as well via the LiteLLM provider prefix.

---

## `LLMBatchItem`

Result of one item in `abatch_invoke`.

```python
@dataclass
class LLMBatchItem:
    index: int
    response: LLMResponse | None = None
    error: Exception | None = None
```

| Field | Type | Description |
|---|---|---|
| `index` | `int` | Position of this item in the original `prompts` list. |
| `response` | `LLMResponse \| None` | Set on success. |
| `error` | `Exception \| None` | Set on failure. |
| `ok` (property) | `bool` | `response is not None`. |

---

## `LLMResponse`

The standard response type returned by every `LLMInterface` method.

```python
class LLMResponse(DataModel):
    content: str
    tool_calls: list[dict[str, Any]] | None = None
```

## See also

- [Guides → Swap LLM provider](../guides/swap-llm-provider) — Anthropic, Gemini, Azure, Ollama, custom.
- [LiteLLM providers list](https://docs.litellm.ai/docs/providers).
