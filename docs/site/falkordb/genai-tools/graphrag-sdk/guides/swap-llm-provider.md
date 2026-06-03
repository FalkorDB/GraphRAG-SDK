---
title: "Swap LLM provider"
nav_order: 5
parent: "Guides"
grand_parent: "GraphRAG-SDK"
description: "Use Anthropic, Gemini, Azure, Ollama, or any LiteLLM-supported provider in place of OpenAI. Or plug a custom LLM in behind the LLMInterface ABC."
---

# Swap LLM provider

The SDK is provider-agnostic. Anything reachable through [LiteLLM](https://docs.litellm.ai/) works out of the box — 100+ providers including Anthropic, Gemini, Azure OpenAI, Ollama, Groq, Cohere, OpenRouter, Mistral, AWS Bedrock, GCP Vertex.

For providers LiteLLM doesn't cover, subclass `LLMInterface` / `Embedder`.

## Anthropic Claude

```python
import os
from graphrag_sdk import LiteLLM, LiteLLMEmbedder

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

llm = LiteLLM(model="anthropic/claude-3-5-sonnet-20241022")
embedder = LiteLLMEmbedder(model="openai/text-embedding-3-large")  # Anthropic has no embeddings; mix providers freely
```

## Google Gemini

```python
os.environ["GEMINI_API_KEY"] = "..."

llm = LiteLLM(model="gemini/gemini-2.0-flash")
embedder = LiteLLMEmbedder(model="gemini/text-embedding-004")
```

## Azure OpenAI

```python
llm = LiteLLM(
    model="azure/my-gpt-4o-deployment",
    api_key="...",
    api_base="https://my-resource.openai.azure.com/",
    api_version="2024-08-01-preview",
)
embedder = LiteLLMEmbedder(
    model="azure/my-embedding-deployment",
    api_key="...",
    api_base="https://my-resource.openai.azure.com/",
    api_version="2024-08-01-preview",
)
```

## Ollama (local)

```python
# After `ollama pull llama3.1` and `ollama pull nomic-embed-text`:
llm = LiteLLM(model="ollama/llama3.1", api_base="http://localhost:11434")
embedder = LiteLLMEmbedder(model="ollama/nomic-embed-text", api_base="http://localhost:11434")
```

`embedding_dimension` on `GraphRAG` must match your embedder's output size — `nomic-embed-text` is 768, `text-embedding-3-large` is 256/1024/3072 (the SDK's default is 256). Mismatch is caught at first ingest with a `ConfigError`.

## OpenAI reasoning models (o1, o3, gpt-5)

LiteLLM in this SDK transparently rewrites kwargs for reasoning models: `temperature` is stripped (those models only accept the default) and `max_tokens` becomes `max_completion_tokens`. No code change needed:

```python
llm = LiteLLM(model="openai/o3-mini")
```

## Custom provider — subclass `LLMInterface`

When LiteLLM doesn't cover your provider, implement the two ABCs directly:

```python
from typing import Any
from graphrag_sdk import LLMInterface, Embedder, LLMResponse


class MyCustomLLM(LLMInterface):
    def __init__(self, endpoint: str, api_key: str, model: str) -> None:
        super().__init__(model_name=model)
        self._endpoint = endpoint
        self._api_key = api_key

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        import httpx
        resp = httpx.post(
            self._endpoint,
            json={"prompt": prompt, "model": self.model_name},
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=60,
        )
        resp.raise_for_status()
        return LLMResponse(content=resp.json()["text"])


class MyCustomEmbedder(Embedder):
    def __init__(self, endpoint: str, api_key: str, model: str) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        import httpx
        resp = httpx.post(
            self._endpoint + "/embed",
            json={"text": text, "model": self._model},
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
```

You only need to implement `invoke` (sync) — the SDK's async path (`ainvoke`) runs it in a thread with retry + jittered backoff. Override `ainvoke` directly if your client is async-native. Similarly, `aembed_query` falls through to `embed_query` in a thread; override for async-native embedders.

Once written, pass to `GraphRAG`:

```python
rag = GraphRAG(
    connection=ConnectionConfig(...),
    llm=MyCustomLLM(endpoint="...", api_key="...", model="acme-large"),
    embedder=MyCustomEmbedder(endpoint="...", api_key="...", model="acme-embed"),
)
```

## See also

- [API Reference → Providers](../api-reference/providers) — `LLMInterface`, `Embedder`, `LiteLLM`, `LiteLLMEmbedder`, OpenRouter variants.
- [LiteLLM providers list](https://docs.litellm.ai/docs/providers) — every model identifier LiteLLM supports.
