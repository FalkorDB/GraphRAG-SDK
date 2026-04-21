# GraphRAG SDK

**The most accurate Graph RAG framework. Built on [FalkorDB](https://www.falkordb.com/).**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/LICENSE)
[![Version: 1.0.0](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/pyproject.toml)
[![Tests: 582 passing](https://img.shields.io/badge/tests-582%20passing-brightgreen.svg)](https://github.com/FalkorDB/GraphRAG-SDK/tree/main/graphrag_sdk/tests/)

GraphRAG SDK builds knowledge graphs from documents and answers questions over them using retrieval-augmented generation. Every algorithmic concern (chunking, extraction, resolution, retrieval, reranking) is a swappable strategy behind an abstract interface. The default pipeline scores **~85% accuracy** on a 100-question benchmark using GPT-4.1.

## Quick Start

```python
import asyncio
from graphrag_sdk import GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder

async def main():
    async with GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="my_graph"),
        llm=LiteLLM(model="openai/gpt-4o"),
        embedder=LiteLLMEmbedder(model="openai/text-embedding-3-small"),
    ) as rag:
        result = await rag.ingest("my_document.txt")
        print(f"Created {result.nodes_created} nodes, {result.relationships_created} edges")

        answer = await rag.completion("What is the main theme?")
        print(answer.answer)

asyncio.run(main())
```

## Installation

```bash
pip install graphrag-sdk[litellm]       # OpenAI, Azure, Anthropic, 100+ models
pip install graphrag-sdk[openrouter]    # OpenRouter models
pip install graphrag-sdk[pdf]           # PDF ingestion
pip install graphrag-sdk[all]           # Everything
```

### Prerequisites

- **Python** >= 3.10
- **FalkorDB**: `docker run -p 6379:6379 falkordb/falkordb`
- An **LLM API key** (OpenAI, Azure OpenAI, OpenRouter, etc.)

## Usage

### Ingest & Query

```python
import asyncio
from graphrag_sdk import GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder

async def main():
    async with GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="my_graph"),
        llm=LiteLLM(model="openai/gpt-4o"),
        embedder=LiteLLMEmbedder(model="openai/text-embedding-3-small"),
    ) as rag:
        await rag.ingest("report.pdf")                              # PDF
        await rag.ingest("source_id", text="Alice works at Acme.")  # Raw text
        await rag.finalize()                                         # Dedup + index

        # Retrieve context only
        context = await rag.retrieve("Where does Alice work?")

        # Full RAG: retrieve + generate answer
        result = await rag.completion("Where does Alice work?")
        print(result.answer)

asyncio.run(main())
```

### Multi-Turn Conversations

`completion()` supports multi-turn conversations. With the built-in providers (`LiteLLM`, `OpenRouterLLM`), messages are passed natively to the LLM's chat API. Custom providers that only implement `invoke()` get automatic fallback via message concatenation.

```python
from graphrag_sdk import ChatMessage

answer = await rag.completion(
    "What happened next?",
    history=[
        ChatMessage(role="user", content="Who is Alice?"),
        ChatMessage(role="assistant", content="Alice is an engineer at Acme Corp."),
    ],
)
```

Supported roles: `"system"`, `"user"`, `"assistant"`. Invalid roles raise `ValueError`.

### Schema Definition

```python
from graphrag_sdk import GraphSchema, EntityType, RelationType

schema = GraphSchema(
    entities=[
        EntityType(label="Person", description="A human being"),
        EntityType(label="Organization", description="A company or institution"),
    ],
    relations=[
        RelationType(
            label="WORKS_AT",
            description="Is employed by",
            patterns=[("Person", "Organization")],
        ),
    ],
)

rag = GraphRAG(connection=conn, llm=llm, embedder=embedder, schema=schema)  # conn, llm, embedder from above
```

### Strategy Customization

Override any pipeline step by passing a strategy:

```python
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
from graphrag_sdk import GraphExtraction, LLMExtractor
from graphrag_sdk.ingestion.resolution_strategies import SemanticResolution

# Custom chunking
await rag.ingest("doc.txt", chunker=FixedSizeChunking(chunk_size=1500, chunk_overlap=200))

# LLM-based entity extraction instead of GLiNER
await rag.ingest("doc.txt", extractor=GraphExtraction(llm=llm, entity_extractor=LLMExtractor(llm)))
```

## Strategy Reference

Every algorithmic concern is a swappable strategy behind an abstract base class:

| Concern | ABC | Built-in Options | Default |
|---------|-----|-----------------|---------|
| **Loading** | `LoaderStrategy` | `TextLoader`, `PdfLoader` | Auto-detect by extension |
| **Chunking** | `ChunkingStrategy` | `FixedSizeChunking`, `SentenceTokenCapChunking`, `ContextualChunking`, `CallableChunking` | `FixedSizeChunking` |
| **Extraction** | `ExtractionStrategy` | `GraphExtraction` (GLiNER2 + LLM) | `GraphExtraction` |
| **Resolution** | `ResolutionStrategy` | `ExactMatchResolution`, `DescriptionMergeResolution`, `SemanticResolution`, `LLMVerifiedResolution` | `ExactMatch` |
| **Retrieval** | `RetrievalStrategy` | `LocalRetrieval`, `MultiPathRetrieval` | `MultiPath` (5-path) |
| **Reranking** | `RerankingStrategy` | `CosineReranker` | Cosine |

### LLM & Embedding Providers

| Provider | LLM Class | Embedder Class | Models |
|----------|-----------|---------------|--------|
| **LiteLLM** | `LiteLLM` | `LiteLLMEmbedder` | OpenAI, Azure, Anthropic, Cohere, 100+ |
| **OpenRouter** | `OpenRouterLLM` | `OpenRouterEmbedder` | All OpenRouter models |
| **Custom** | Subclass `LLMInterface` | Subclass `Embedder` | Anything |

## Benchmark

**#1 on [GraphRAG-Bench](https://graphrag-bench.github.io) Novel** — 63.73 ACC, ahead of MS-GraphRAG (50.93) and LightRAG (45.09).

| Metric | Value |
|--------|-------|
| **Novel ACC** | 63.73 (#1) |
| **Fact retrieval** | 65.22 |
| **Complex reasoning** | 58.63 |
| **Contextual summarization** | 69.54 |
| **Creative generation** | 57.08 |
| **Questions** | 2,010 across 20 novels |

See [docs/benchmark.md](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/docs/benchmark.md) for methodology and reproduction.

## Examples

| # | Example | Description |
|---|---------|-------------|
| 1 | [`01_quickstart.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/examples/01_quickstart.py) | Minimal ingest & query |
| 2 | [`02_pdf_with_schema.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/examples/02_pdf_with_schema.py) | PDF with custom schema |
| 3 | [`03_custom_strategies.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/examples/03_custom_strategies.py) | Benchmark-winning pipeline |
| 4 | [`04_custom_provider.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/examples/04_custom_provider.py) | Custom LLM/Embedder |
| 5 | [`05_notebook_demo.ipynb`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/examples/05_notebook_demo.ipynb) | Interactive notebook walkthrough |

## Documentation

- [Getting Started](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/docs/getting-started.md) -- Install to first query
- [Architecture](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/docs/architecture.md) -- Pipeline design and graph schema
- [Configuration](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/docs/configuration.md) -- Connection and provider reference
- [Strategies](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/docs/strategies.md) -- All ABCs and built-in implementations
- [Providers](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/docs/providers.md) -- LLM & embedder configuration
- [Benchmark](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/docs/benchmark.md) -- Methodology and reproduction
- [API Reference](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/docs/api-reference.md) -- Full API documentation

## License

[Apache License 2.0](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/LICENSE)
