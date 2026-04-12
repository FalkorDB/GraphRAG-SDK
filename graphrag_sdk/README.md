# GraphRAG SDK

**A modular, async-first Graph RAG framework for FalkorDB.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Version: 1.0.0](https://img.shields.io/badge/version-1.0.0-orange.svg)](pyproject.toml)
[![Tests: 556 passing](https://img.shields.io/badge/tests-556%20passing-brightgreen.svg)](tests/)

GraphRAG SDK builds knowledge graphs from documents and answers questions over them using retrieval-augmented generation. Every algorithmic concern (chunking, extraction, resolution, retrieval, reranking) is a swappable strategy behind an abstract interface. The default pipeline scores **84.8% accuracy** on a 20-document novel benchmark using GPT-4.1.

```
Document --> [Load] --> [Chunk] --> [Extract] --> [Resolve] --> [Write] --> Knowledge Graph
                                                                                |
Question --> [Retrieve (multi-path)] --> [Rerank] --> [Generate] -----> Answer
```

## Quick Start

```python
import asyncio
from graphrag_sdk import GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder

async def main():
    rag = GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="my_graph"),
        llm=LiteLLM(model="azure/gpt-4.1", api_key="..."),
        embedder=LiteLLMEmbedder(model="azure/text-embedding-ada-002", api_key="..."),
    )

    # Ingest a document
    result = await rag.ingest("my_document.txt")
    print(f"Created {result.nodes_created} nodes, {result.relationships_created} edges")

    # Retrieve context only
    context = await rag.retrieve("What is the main theme?")

    # Full RAG: retrieve + generate answer
    answer = await rag.completion("What is the main theme?")
    print(answer.answer)

asyncio.run(main())
```

## Installation

```bash
# Core + LiteLLM provider (Azure OpenAI, OpenAI, Anthropic, Cohere, 100+ models)
pip install graphrag-sdk[litellm]

# Core + OpenRouter provider
pip install graphrag-sdk[openrouter]

# PDF support
pip install graphrag-sdk[pdf]

# Everything
pip install graphrag-sdk[all]
```

### Prerequisites

- **Python** >= 3.10
- **FalkorDB** running locally or remotely:
  ```bash
  docker run -p 6379:6379 falkordb/falkordb
  ```
- An **LLM API key** (Azure OpenAI, OpenAI, OpenRouter, etc.)

## Usage

### Basic: Ingest & Query

```python
from graphrag_sdk import GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder

rag = GraphRAG(
    connection=ConnectionConfig(host="localhost", graph_name="my_graph"),
    llm=LiteLLM(model="azure/gpt-4.1", api_key="YOUR_KEY", api_base="YOUR_ENDPOINT"),
    embedder=LiteLLMEmbedder(model="azure/text-embedding-ada-002", api_key="YOUR_KEY", api_base="YOUR_ENDPOINT"),
)

# Ingest from file (auto-detects PDF vs text)
await rag.ingest("report.pdf")

# Ingest from raw text
await rag.ingest("source_id", text="Alice works at Acme Corp in London.")

# Retrieve context only (no LLM call)
context = await rag.retrieve("Where does Alice work?")

# Full RAG: retrieve + generate answer
result = await rag.completion("Where does Alice work?")
print(result.answer)  # "Alice works at Acme Corp in London."

# With context inspection
result = await rag.completion("Where does Alice work?", return_context=True)
print(result.retriever_result.items)  # See what was retrieved
```

### Multi-Turn Conversations

`completion()` supports multi-turn conversations. With the built-in providers (`LiteLLM`, `OpenRouterLLM`), messages are passed natively to the LLM's chat API. Custom providers that only implement `invoke()` get automatic fallback via message concatenation.

```python
from graphrag_sdk import ChatMessage

# Using ChatMessage objects (validated)
answer = await rag.completion(
    "What happened next?",
    history=[
        ChatMessage(role="system", content="Answer concisely."),
        ChatMessage(role="user", content="Who is Alice?"),
        ChatMessage(role="assistant", content="Alice is an engineer at Acme Corp."),
    ],
)

# Using plain dicts (also validated)
answer = await rag.completion(
    "Tell me more.",
    history=[
        {"role": "user", "content": "What is Acme Corp?"},
        {"role": "assistant", "content": "A tech company in London."},
    ],
)
```

Supported roles: `"system"`, `"user"`, `"assistant"`. Invalid roles raise `ValueError`.

### Schema Definition

Define a schema to constrain what entities and relationships the LLM extracts:

```python
from graphrag_sdk import GraphSchema, EntityType, RelationType, SchemaPattern

schema = GraphSchema(
    entities=[
        EntityType(label="Person", description="A human being"),
        EntityType(label="Organization", description="A company or institution"),
        EntityType(label="Place", description="A geographic location"),
    ],
    relations=[
        RelationType(label="WORKS_AT", description="Is employed by"),
        RelationType(label="LOCATED_IN", description="Is located in"),
    ],
    patterns=[
        SchemaPattern(source="Person", relationship="WORKS_AT", target="Organization"),
        SchemaPattern(source="Organization", relationship="LOCATED_IN", target="Place"),
    ],
)

rag = GraphRAG(connection=conn, llm=llm, embedder=embedder, schema=schema)
```

### Strategy Customization

Override any pipeline step by passing a strategy:

```python
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
from graphrag_sdk import GraphExtraction, GLiNERExtractor, LLMExtractor
from graphrag_sdk.ingestion.resolution_strategies import (
    ExactMatchResolution,
    DescriptionMergeResolution,
    SemanticResolution,
    LLMVerifiedResolution,
)
# Full 4-stage deduplication pipeline
async def ingest_with_full_resolution(rag, path):
    data = await rag.ingest(path, chunker=FixedSizeChunking(chunk_size=1500, chunk_overlap=200))
    ctx = data.context
    gd = data.graph_data
    for resolver in [
        ExactMatchResolution(resolve_property="name"),
        DescriptionMergeResolution(llm=llm),
        SemanticResolution(llm=llm, embedder=embedder, similarity_threshold=0.85),
        LLMVerifiedResolution(llm=llm, embedder=embedder, hard_threshold=0.95, soft_threshold=0.60),
    ]:
        result = await resolver.resolve(gd, ctx)
        gd.nodes = result.nodes
        gd.relationships = result.relationships

# Use LLM for entity extraction instead of GLiNER
await rag.ingest(
    "document.txt",
    extractor=GraphExtraction(llm=llm, entity_extractor=LLMExtractor(llm)),
)
```

### Post-Ingestion Operations

After ingesting all documents, run `finalize()` to deduplicate entities, backfill embeddings, and ensure indexes:

```python
# Run all post-ingestion steps
result = await rag.finalize()
print(f"Deduplicated: {result['entities_deduplicated']}")
print(f"Embedded: {result['entities_embedded']} entities, {result['relationships_embedded']} rels")

# Inspect graph statistics
stats = await rag.graph_store.get_statistics()
print(f"Nodes: {stats['node_count']}, Edges: {stats['edge_count']}")
```

## Strategy Reference

Every algorithmic concern is a swappable strategy behind an abstract base class:

| Concern | ABC | Built-in Options | Default |
|---------|-----|-----------------|---------|
| **Loading** | `LoaderStrategy` | `TextLoader`, `PdfLoader` | Auto-detect by file extension |
| **Chunking** | `ChunkingStrategy` | `FixedSizeChunking`, `SentenceTokenCapChunking`, `ContextualChunking`, `CallableChunking` | `FixedSizeChunking` (1000 chars, 100 overlap) |
| **Extraction** | `ExtractionStrategy` | `GraphExtraction` (GLiNER2 + LLM) | GraphExtraction |
| **Resolution** | `ResolutionStrategy` | `ExactMatchResolution`, `DescriptionMergeResolution`, `SemanticResolution`, `LLMVerifiedResolution` | ExactMatch |
| **Retrieval** | `RetrievalStrategy` | `LocalRetrieval`, `MultiPathRetrieval` | MultiPath (5-path) |
| **Reranking** | `RerankingStrategy` | `CosineReranker` | Cosine (built into MultiPath) |

### LLM & Embedding Providers

| Provider | LLM Class | Embedder Class | Supports |
|----------|-----------|---------------|----------|
| **LiteLLM** | `LiteLLM` | `LiteLLMEmbedder` | Azure OpenAI, OpenAI, Anthropic, Cohere, 100+ models |
| **OpenRouter** | `OpenRouterLLM` | `OpenRouterEmbedder` | All OpenRouter models |
| **Custom** | Subclass `LLMInterface` | Subclass `Embedder` | Anything |

## Benchmark

The default pipeline achieves **84.8% accuracy** (8.48/10) on a 100-question benchmark over 20 Project Gutenberg novels.

| Metric | Value |
|--------|-------|
| **Accuracy** | 8.48/10 (84.8%) |
| **Questions tested** | 100 (fact retrieval, complex reasoning, summarization) |
| **Documents** | 20 novels (Project Gutenberg) |
| **Indexing time** | ~47 min (20 docs) |
| **Query latency P50** | 5.4s |
| **Query latency P95** | 9.2s |

### Winning Pipeline Configuration

The benchmark-winning pipeline uses:
- **Extraction**: `GraphExtraction` -- GLiNER2 local NER (step 1) + LLM verify & relationship extraction (step 2)
- **Resolution**: `ExactMatchResolution` → `DescriptionMergeResolution` → `SemanticResolution` → `LLMVerifiedResolution` -- 4-stage pipeline: exact match, normalized-name merge, hnswlib HNSW semantic clustering, scipy agglomerative clustering + batched LLM verification
- **Retrieval**: `MultiPathRetrieval` -- 5-path entity discovery, 2-hop relationship expansion, 5-path chunk retrieval, cosine reranking, fact retrieval
- **Chunking**: 1500 chars / 200 overlap
- **LLM**: GPT-4.1 (Azure OpenAI), **Embeddings**: text-embedding-ada-002 (1536 dim)

See [docs/benchmark.md](docs/benchmark.md) for full reproduction instructions.

## Project Structure

```
graphrag_sdk/
├── api/main.py                     # GraphRAG facade (ingest, query, finalize)
├── core/
│   ├── connection.py               # FalkorDB connection & config
│   ├── context.py                  # Execution context (logging, budget)
│   ├── exceptions.py               # Exception hierarchy
│   ├── models.py                   # Pydantic data models (30+ classes)
│   └── providers.py                # LLM & Embedder ABCs + built-in providers
├── ingestion/
│   ├── pipeline.py                 # 9-step ingestion orchestrator
│   ├── loaders/                    # TextLoader, PdfLoader
│   ├── chunking_strategies/        # FixedSizeChunking, SentenceTokenCapChunking, ContextualChunking, CallableChunking
│   ├── extraction_strategies/      # GraphExtraction (GLiNER2 + LLM)
│   └── resolution_strategies/      # ExactMatch, DescriptionMerge, Semantic, LLMVerified
├── retrieval/
│   ├── strategies/                 # LocalRetrieval, MultiPathRetrieval
│   └── reranking_strategies/       # CosineReranker
├── storage/
│   ├── graph_store.py              # Batched Cypher operations
│   └── vector_store.py             # Vector index management & search
├── telemetry/                      # Tracing
└── utils/                          # Graph visualization
```

## Examples

- [`examples/01_quickstart.py`](examples/01_quickstart.py) -- Minimal ingest & query (30 lines)
- [`examples/02_pdf_with_schema.py`](examples/02_pdf_with_schema.py) -- PDF ingestion with custom schema
- [`examples/03_custom_strategies.py`](examples/03_custom_strategies.py) -- Benchmark-winning pipeline configuration
- [`examples/04_custom_provider.py`](examples/04_custom_provider.py) -- Write your own LLM/Embedder

## Documentation

- [Architecture](docs/architecture.md) -- How the pipeline works
- [Strategy Reference](docs/strategies.md) -- All ABCs and built-in implementations
- [Providers](docs/providers.md) -- LLM & Embedder configuration
- [Benchmark](docs/benchmark.md) -- Reproducing the 84.8% accuracy result
- [API Reference](docs/api-reference.md) -- Full API documentation

## Core Principles

- **Strategy Modularity** -- Swap any algorithmic concern via strategy ABCs
- **Zero-Loss Data** -- Full traceability from raw text to graph nodes
- **Production Latency** -- Async-first, pooled connections, batched writes
- **Simplicity** -- One entry point, flat structure, no meta-programming

## License

Apache 2.0
