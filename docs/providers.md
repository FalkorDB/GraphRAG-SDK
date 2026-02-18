# Providers: LLM & Embedder Configuration

GraphRAG SDK uses two provider types: **LLM** (for text generation and extraction) and **Embedder** (for vector embeddings). Both are abstract base classes with built-in implementations for LiteLLM and OpenRouter.

## Provider Overview

| Provider | LLM Class | Embedder Class | Install Extra | Models Supported |
|----------|-----------|---------------|---------------|-----------------|
| **LiteLLM** | `LiteLLM` | `LiteLLMEmbedder` | `pip install graphrag-sdk[litellm]` | Azure OpenAI, OpenAI, Anthropic, Cohere, 100+ |
| **OpenRouter** | `OpenRouterLLM` | `OpenRouterEmbedder` | `pip install graphrag-sdk[openrouter]` | All OpenRouter models |
| **Custom** | Subclass `LLMInterface` | Subclass `Embedder` | -- | Anything |

## LiteLLM (Recommended)

LiteLLM provides a unified interface to 100+ LLM providers. It is the recommended default.

### LLM

```python
from graphrag_sdk import LiteLLM

# Azure OpenAI
llm = LiteLLM(
    model="azure/gpt-4.1",
    api_key="your-azure-key",
    api_base="https://your-resource.openai.azure.com/",
    api_version="2024-12-01-preview",
    temperature=0.0,      # default: 0.0
    max_tokens=None,      # default: None (provider default)
)

# OpenAI direct
llm = LiteLLM(
    model="gpt-4o",
    api_key="your-openai-key",
)

# Anthropic
llm = LiteLLM(
    model="anthropic/claude-sonnet-4-20250514",
    api_key="your-anthropic-key",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model identifier (use `provider/model` format for non-OpenAI) |
| `api_key` | `str \| None` | `None` | API key (or set via environment variable) |
| `api_base` | `str \| None` | `None` | Base URL (required for Azure) |
| `api_version` | `str \| None` | `None` | API version (required for Azure) |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `max_tokens` | `int \| None` | `None` | Max output tokens |

### Embedder

```python
from graphrag_sdk import LiteLLMEmbedder

# Azure OpenAI
embedder = LiteLLMEmbedder(
    model="azure/text-embedding-ada-002",
    api_key="your-azure-key",
    api_base="https://your-resource.openai.azure.com/",
    api_version="2024-12-01-preview",
)

# OpenAI direct
embedder = LiteLLMEmbedder(
    model="text-embedding-3-small",
    api_key="your-openai-key",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Embedding model identifier |
| `api_key` | `str \| None` | `None` | API key |
| `api_base` | `str \| None` | `None` | Base URL |
| `api_version` | `str \| None` | `None` | API version |

## OpenRouter

OpenRouter aggregates models from multiple providers behind a single API.

### LLM

```python
from graphrag_sdk import OpenRouterLLM

llm = OpenRouterLLM(
    model="anthropic/claude-sonnet-4-20250514",
    api_key="your-openrouter-key",      # or set OPENROUTER_API_KEY env var
    temperature=0.0,
    max_tokens=None,
    extra_headers={},                    # optional custom headers
)
```

### Embedder

```python
from graphrag_sdk import OpenRouterEmbedder

embedder = OpenRouterEmbedder(
    model="openai/text-embedding-ada-002",
    api_key="your-openrouter-key",
    extra_headers={},
)
```

## LLMInterface ABC

To integrate a provider not covered by LiteLLM or OpenRouter, subclass `LLMInterface`.

### Required Method

```python
from graphrag_sdk import LLMInterface
from graphrag_sdk.core.models import LLMResponse

class MyLLM(LLMInterface):
    def __init__(self, model_name: str = "my-model", **kwargs):
        super().__init__(model_name=model_name)
        # Initialize your client

    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        """Synchronous text generation. REQUIRED."""
        response = my_client.generate(prompt)
        return LLMResponse(content=response.text)
```

### Optional Overrides

| Method | Default Behavior | Override When |
|--------|-----------------|--------------|
| `ainvoke(prompt, max_retries=3)` | Runs `invoke()` in a thread pool with retry | You have a native async client |
| `invoke_with_model(prompt, response_model)` | Calls `invoke()` and parses JSON into Pydantic model | Your provider has native structured output |
| `ainvoke_with_model(prompt, response_model)` | Calls `ainvoke()` and parses JSON | Same, async version |
| `abatch_invoke(prompts, max_concurrency)` | Concurrent `ainvoke()` with semaphore | You have a native batch API |

### Constructor Parameters

```python
LLMInterface.__init__(
    model_name: str,                    # Model identifier
    model_params: dict | None = None,   # Provider-specific params
    max_concurrency: int = 12,          # Parallel call limit for abatch_invoke
)
```

## Embedder ABC

### Required Method

```python
from graphrag_sdk import Embedder

class MyEmbedder(Embedder):
    def embed_query(self, text: str, **kwargs) -> list[float]:
        """Embed a single text. REQUIRED."""
        return my_model.encode(text).tolist()
```

### Optional Overrides

| Method | Default Behavior | Override When |
|--------|-----------------|--------------|
| `aembed_query(text)` | Runs `embed_query()` in thread pool | You have async embedding |
| `embed_documents(texts)` | Sequential `embed_query()` per text | You can batch embeddings |
| `aembed_documents(texts)` | Runs `embed_documents()` in thread pool | You have async batch |

### Batch Embedding

The `embed_documents()` and `aembed_documents()` methods are critical for performance. The ingestion pipeline calls them with hundreds or thousands of texts. If your provider supports batch embedding, always override these methods:

```python
class MyEmbedder(Embedder):
    def embed_query(self, text: str, **kwargs) -> list[float]:
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        # Batch embedding -- much faster than sequential
        return self.model.encode(texts).tolist()
```

## Environment Variables

For convenience, you can configure providers via environment variables instead of passing parameters directly. LiteLLM respects standard environment variables:

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` | Azure OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `COHERE_API_KEY` | Cohere |
| `OPENROUTER_API_KEY` | OpenRouter |

See the [LiteLLM documentation](https://docs.litellm.ai/) for the full list of supported providers and their environment variables.

## Choosing a Provider

| Use Case | Recommendation |
|----------|---------------|
| **Production (Azure)** | `LiteLLM` with `azure/` prefix |
| **Development (OpenAI)** | `LiteLLM` with OpenAI models |
| **Budget-conscious** | `OpenRouterLLM` for model price comparison |
| **Local models** | Custom `LLMInterface` wrapping Ollama, vLLM, etc. |
| **Local embeddings** | Custom `Embedder` wrapping sentence-transformers |
