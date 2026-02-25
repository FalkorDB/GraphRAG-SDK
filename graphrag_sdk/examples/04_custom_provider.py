"""
GraphRAG SDK v2 -- Custom LLM & Embedder Providers
====================================================
Shows how to implement your own LLM and Embedder by subclassing the ABCs.
Use this pattern to integrate local models, custom APIs, or any provider
not covered by the built-in LiteLLM/OpenRouter classes.

This example uses stub implementations for demonstration. Replace the
invoke/embed methods with your actual model calls.

Prerequisites:
    pip install graphrag-sdk
    docker run -p 6379:6379 falkordb/falkordb
"""

import asyncio
from typing import Any

from graphrag_sdk import ConnectionConfig, Embedder, GraphRAG, LLMInterface
from graphrag_sdk.core.models import LLMResponse


# --- Custom LLM Provider ---


class MyCustomLLM(LLMInterface):
    """Example: wrap a local model or custom API.

    You must implement `invoke()`. The async `ainvoke()` will
    automatically run it in a thread pool, or you can override
    `ainvoke()` for true async support.
    """

    def __init__(self, model_name: str = "my-local-model", **kwargs: Any) -> None:
        super().__init__(model_name=model_name)
        # Initialize your client here, e.g.:
        # self.client = MyModelClient(**kwargs)

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        # Replace with your actual model call:
        # response = self.client.generate(prompt)
        # return LLMResponse(content=response.text)
        return LLMResponse(content=f"[Stub response for: {prompt[:50]}...]")

    # Optional: override for native async support
    # async def ainvoke(self, prompt, *, max_retries=3, **kwargs) -> LLMResponse:
    #     response = await self.client.agenerate(prompt)
    #     return LLMResponse(content=response.text)


# --- Custom Embedder Provider ---


class MyCustomEmbedder(Embedder):
    """Example: wrap a local embedding model or custom API.

    You must implement `embed_query()`. Optionally override
    `embed_documents()` for batch optimization and `aembed_query()`
    / `aembed_documents()` for async support.
    """

    def __init__(self, dimension: int = 1536) -> None:
        self.dimension = dimension
        # Initialize your embedding model here, e.g.:
        # self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        # Replace with your actual embedding call:
        # return self.model.encode(text).tolist()
        return [0.0] * self.dimension  # Stub: zero vector

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        # Optional batch optimization. Default calls embed_query per text.
        # return self.model.encode(texts).tolist()
        return [self.embed_query(t) for t in texts]

    # Optional: native async versions
    # async def aembed_query(self, text, **kwargs) -> list[float]: ...
    # async def aembed_documents(self, texts, **kwargs) -> list[list[float]]: ...


# --- Usage ---


async def main():
    # Use your custom providers with GraphRAG
    rag = GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="custom_provider_demo"),
        llm=MyCustomLLM(model_name="my-local-llama"),
        embedder=MyCustomEmbedder(dimension=1536),
    )

    # Ingest with custom providers
    result = await rag.ingest(
        "demo_source",
        text="The quick brown fox jumps over the lazy dog.",
    )
    print(f"Ingested: {result.nodes_created} nodes")

    # Query
    answer = await rag.query("What did the fox do?")
    print(f"Answer: {answer.answer}")


if __name__ == "__main__":
    asyncio.run(main())
