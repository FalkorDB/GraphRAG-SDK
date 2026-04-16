# GraphRAG SDK

**The most accurate Graph RAG framework. Built on [FalkorDB](https://www.falkordb.com/).**

GraphRAG SDK builds knowledge graphs from documents and answers questions over them using graph-based retrieval-augmented generation. Every pipeline step is a swappable strategy behind an abstract interface.

## Key Highlights

- **#1 on GraphRAG-Bench Novel** — 63.73 overall ACC on 2,010 questions ([benchmark](benchmark.md))
- **Simple API** -- `ingest()` + `completion()` with sensible defaults
- **100+ LLM providers** via LiteLLM (OpenAI, Azure, Anthropic, Cohere, Ollama, and more)
- **Fully modular** -- swap chunking, extraction, resolution, retrieval, and reranking strategies
- **Production-ready** -- async-first, connection pooling, circuit breaker, batched writes
- **Full provenance** -- every answer traces back to its source document and chunk

## Quick Start

```bash
pip install graphrag-sdk[litellm]
docker run -p 6379:6379 falkordb/falkordb
```

```python
import asyncio
from graphrag_sdk import GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder

async def main():
    async with GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="my_graph"),
        llm=LiteLLM(model="openai/gpt-4o"),
        embedder=LiteLLMEmbedder(model="openai/text-embedding-3-large", dimensions=256),
        embedding_dimension=256,
    ) as rag:
        await rag.ingest("my_document.pdf")
        await rag.finalize()
        answer = await rag.completion("What is the main topic?")
        print(answer.answer)

asyncio.run(main())
```

## Next Steps

- [Getting Started](getting-started.md) -- Full tutorial from install to first query
- [Architecture](architecture.md) -- How the 9-step pipeline works
- [Strategies](strategies.md) -- All swappable strategy ABCs and built-in options
- [Benchmark](benchmark.md) -- Methodology and reproduction instructions
- [API Reference](api-reference.md) -- Full API documentation
