# GraphRAG SDK

GraphRAG SDK is a production-grade Python SDK for building and querying knowledge graphs using Graph-based Retrieval Augmented Generation (GraphRAG). It uses [FalkorDB](https://www.falkordb.com/) as the graph database and supports any LLM/embedder provider through a pluggable provider interface.

## Key Features

- **Single RELATES edge architecture**: All extracted relationships are unified under one edge type (`RELATES`) with a `rel_type` property, eliminating the explosion of per-type edge indexes.
- **9-step ingestion pipeline**: 7 sequential steps followed by 2 parallel steps (Load, Chunk, Lexical Graph, Extract, Prune, Resolve, Write, then Mentions and Chunk Indexing in parallel).
- **Multi-path retrieval**: Combines entity discovery (vector + fulltext), relationship expansion, chunk-level retrieval, and cosine reranking for comprehensive answer generation.
- **Entity deduplication**: Semantic resolution strategy merges duplicate entities based on embedding similarity and description overlap.
- **Full provenance chain**: Every answer traces back through entities, relationships, and source chunks to the original document.
- **Benchmark-validated**: 84.8% accuracy (8.48/10) on a 100-question literary benchmark across fact retrieval, complex reasoning, contextual summarization, and creative question types.

## Quick Start

### Prerequisites

1. Install the SDK:

```bash
pip install -e "graphrag_sdk[all]"
```

2. Start FalkorDB via Docker:

```bash
docker run -p 6379:6379 falkordb/falkordb
```

### Minimal Example

```python
import asyncio
from graphrag_sdk import GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder

async def main():
    rag = GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="my_graph"),
        llm=LiteLLM(model="azure/gpt-4.1"),
        embedder=LiteLLMEmbedder(model="azure/text-embedding-ada-002"),
    )

    # Ingest a document
    result = await rag.ingest("my_document.txt")
    print(f"Nodes: {result.nodes_created}, Edges: {result.relationships_created}")

    # Finalize (dedup + embeddings + indexes)
    await rag.finalize()

    # Query
    answer = await rag.completion("What is the main topic?")
    print(answer.answer)

asyncio.run(main())
```

## Configuration

### ConnectionConfig

Configure the connection to your FalkorDB instance:

```python
from graphrag_sdk import ConnectionConfig

config = ConnectionConfig(
    host="localhost",
    port=6379,
    graph_name="my_graph",
    query_timeout_ms=10000,
)
```

### LLM and Embedder Providers

The SDK supports any LLM and embedder provider via LiteLLM. For Azure OpenAI, set the following environment variables:

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
```

Then configure the providers:

```python
from graphrag_sdk import LiteLLM, LiteLLMEmbedder

llm = LiteLLM(model="azure/gpt-4.1")
embedder = LiteLLMEmbedder(model="azure/text-embedding-ada-002")
```

## Usage

### Ingesting Documents

```python
# Ingest a single file
result = await rag.ingest("path/to/document.txt")

# Check ingestion statistics
print(f"Nodes created: {result.nodes_created}")
print(f"Relationships created: {result.relationships_created}")
```

### Finalizing the Graph

After ingestion, call `finalize()` to run entity deduplication, backfill missing embeddings, and ensure all indexes are created:

```python
await rag.finalize()
```

### Querying the Graph

```python
# Retrieve context only (no LLM call)
context = await rag.retrieve("Who are the main characters?")

# Full RAG: retrieve + generate answer
answer = await rag.completion("Who are the main characters in the story?")
print(answer.answer)
```

### Multi-Turn Conversations

`completion()` supports multi-turn conversations. With the built-in providers (`LiteLLM`, `OpenRouterLLM`), history messages are passed natively to the LLM's chat API. Custom providers that only implement `invoke()` get automatic fallback via message concatenation.

```python
from graphrag_sdk import ChatMessage

answer = await rag.completion(
    "What happened to her after that?",
    history=[
        ChatMessage(role="user", content="Who is Alice?"),
        ChatMessage(role="assistant", content="Alice is a software engineer at Acme Corp."),
    ],
)
```

You can also pass history as plain dicts — roles are validated automatically:

```python
answer = await rag.completion(
    "Tell me more about that.",
    history=[
        {"role": "user", "content": "What is Acme Corp?"},
        {"role": "assistant", "content": "Acme Corp is a tech company."},
    ],
)
```

## Architecture Overview

The ingestion pipeline consists of 9 steps:

1. **Load** -- Read raw text from a source file or string.
2. **Chunk** -- Split the document into overlapping text chunks.
3. **Lexical Graph** -- Create Document and Chunk nodes with provenance edges.
4. **Extract** -- Use the LLM to extract entities and relationships from each chunk.
5. **Prune** -- Filter extracted data against the schema.
6. **Resolve** -- Deduplicate entities (exact match or description merge).
7. **Write** -- Batched MERGE of nodes and relationships to FalkorDB.
8. **Mentions** (parallel) -- Write MENTIONED_IN edges linking entities to source chunks.
9. **Index Chunks** (parallel) -- Embed and index chunk text in vector store.

Steps 8 and 9 run in parallel after step 7 completes.

For a detailed architecture description, see [docs/architecture.md](docs/architecture.md).

## Documentation

- [Getting Started](docs/getting-started.md) -- Step-by-step tutorial from installation to first query.
- [Architecture](docs/architecture.md) -- Pipeline design, graph schema, and retrieval strategy details.
- [Configuration](docs/configuration.md) -- Comprehensive configuration reference for connections, providers, and tuning.
- [Strategies](docs/strategies.md) -- Extraction, resolution, and retrieval strategy documentation.
- [API Reference](docs/api-reference.md) -- Full API documentation for all public classes and methods.
- [Benchmark](docs/benchmark.md) -- Benchmark methodology, results, and reproduction instructions.
- [Providers](docs/providers.md) -- Guide to configuring LLM and embedder providers.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
