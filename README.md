# GraphRAG SDK v2

[![Dockerhub](https://img.shields.io/docker/pulls/falkordb/falkordb?label=Docker)](https://hub.docker.com/r/falkordb/falkordb/)
[![pypi](https://badge.fury.io/py/graphrag_sdk.svg)](https://pypi.org/project/graphrag_sdk/)
[![Discord](https://img.shields.io/discord/1146782921294884966?style=flat-square)](https://discord.gg/6M4QwDXn2w)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A modular, async-first Graph RAG framework for [FalkorDB](https://www.falkordb.com/).

GraphRAG SDK v2 builds knowledge graphs from unstructured text and queries them using LLM-powered retrieval-augmented generation. It combines entity extraction, graph storage, vector search, and LLM generation into a single pipeline with pluggable strategies at every step.

## Key Features

- **Two-line usage** — `ingest()` to build, `query()` to ask
- **Async-first** — all I/O is non-blocking with sync convenience methods
- **Pluggable strategies** — swap loaders, chunkers, extractors, resolvers, retrievers, and rerankers
- **Zero-loss provenance** — every chunk traces back to its source document
- **Production-grade** — batched writes, connection pooling, retry with backoff, latency budgets
- **Schema-guided extraction** — constrain LLM output to your domain ontology

## Setup

### 1. Start FalkorDB

[![Try Free](https://img.shields.io/badge/Try%20Free-FalkorDB%20Cloud-FF8101?labelColor=FDE900&style=for-the-badge&link=https://app.falkordb.cloud)](https://app.falkordb.cloud)

Or run locally with Docker:

```sh
docker run -p 6379:6379 -p 3000:3000 -it --rm -v ./data:/data falkordb/falkordb:latest
```

### 2. Install the SDK

```sh
pip install graphrag-sdk
```

With optional providers:

```sh
pip install graphrag-sdk[openai]       # OpenAI LLM + embeddings
pip install graphrag-sdk[pdf]          # PDF document loading
pip install graphrag-sdk[all]          # Everything
```

### 3. Implement Your Providers

The SDK defines two abstract base classes that you need to implement for your LLM and embedding provider:

```python
from graphrag_sdk import LLMInterface, Embedder, LLMResponse


class MyLLM(LLMInterface):
    def __init__(self):
        super().__init__(model_name="gpt-4o")
        # Initialize your client here

    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        # Call your LLM provider
        response = your_llm_client.chat(prompt)
        return LLMResponse(content=response.text)


class MyEmbedder(Embedder):
    def embed_query(self, text: str, **kwargs) -> list[float]:
        # Return embedding vector
        return your_embedding_client.embed(text)

    # Optional: override for true batch embedding (much faster)
    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        return your_embedding_client.embed_batch(texts)
```

> The SDK provides `ainvoke()` and `aembed_query()` async methods automatically by wrapping your sync implementations in a thread. Override them directly if your provider has native async support.

## Quick Start

### Ingest a Document

```python
import asyncio
from graphrag_sdk import GraphRAG, ConnectionConfig

rag = GraphRAG(
    connection=ConnectionConfig(host="localhost", port=6379, graph_name="my_graph"),
    llm=MyLLM(),
    embedder=MyEmbedder(),
)

result = asyncio.run(rag.ingest("report.pdf"))
print(f"Created {result.nodes_created} nodes, {result.relationships_created} relationships")
```

### Query the Knowledge Graph

```python
answer = asyncio.run(rag.query("What were the key findings?"))
print(answer.answer)
```

### Ingest Raw Text

```python
result = asyncio.run(rag.ingest("source", text="Alice works at Acme Corp as a senior engineer."))
```

### Sync Convenience Methods

```python
# No need for asyncio.run() — sync wrappers are built in
result = rag.ingest_sync("document.pdf")
answer = rag.query_sync("What is this about?")
```

## Architecture

The SDK is organized into four layers:

```
API          GraphRAG (facade)
Ingestion    Load → Chunk → Extract → Quality Filter → Prune → Resolve → Write → Index
Retrieval    Search → Rerank → Generate
Storage      GraphStore (FalkorDB) + VectorStore (FalkorDB vector indices)
```

### Ingestion Pipeline

The pipeline processes documents through 8 sequential steps:

| Step | What it does |
|------|-------------|
| **Load** | Read text from file (PDF, text, or raw string) |
| **Chunk** | Split text into overlapping segments |
| **Lexical Graph** | Create Document and Chunk nodes with provenance edges |
| **Extract** | LLM extracts entities and relationships from each chunk |
| **Quality Filter** | Remove empty IDs, invalid nodes, dangling relationships |
| **Prune** | Filter against schema constraints |
| **Resolve** | Deduplicate entities by normalized name |
| **Write + Index** | Batch upsert to graph, embed and index chunks |

### Query Flow

1. **Retrieve** — vector search + graph traversal to find relevant chunks and entities
2. **Rerank** (optional) — reorder results by relevance
3. **Generate** — LLM produces an answer from the retrieved context

## Schema-Guided Extraction

Define a schema to constrain what the LLM extracts:

```python
from graphrag_sdk import GraphRAG, GraphSchema, EntityType, RelationType, SchemaPattern

schema = GraphSchema(
    entities=[
        EntityType(label="Person", description="A human being"),
        EntityType(label="Company", description="A business organization"),
        EntityType(label="Product", description="A commercial product"),
    ],
    relations=[
        RelationType(label="WORKS_AT", description="Employment relationship"),
        RelationType(label="PRODUCES", description="Company produces a product"),
    ],
    patterns=[
        SchemaPattern(source="Person", relationship="WORKS_AT", target="Company"),
        SchemaPattern(source="Company", relationship="PRODUCES", target="Product"),
    ],
)

rag = GraphRAG(
    connection=ConnectionConfig(host="localhost", graph_name="my_graph"),
    llm=MyLLM(),
    embedder=MyEmbedder(),
    schema=schema,
)
```

Without a schema, the extractor operates in open mode and extracts all entities and relationships it finds.

## Custom Strategies

Every algorithmic step is replaceable via strategy interfaces.

### Custom Loader

```python
from graphrag_sdk import LoaderStrategy
from graphrag_sdk.core.models import DocumentOutput, DocumentInfo
from graphrag_sdk.core.context import Context


class DatabaseLoader(LoaderStrategy):
    async def load(self, source: str, ctx: Context) -> DocumentOutput:
        text = await my_database.fetch_document(source)
        return DocumentOutput(
            text=text,
            document_info=DocumentInfo(path=source),
        )

result = await rag.ingest("doc-id-123", loader=DatabaseLoader())
```

### Custom Chunking

```python
from graphrag_sdk import ChunkingStrategy
from graphrag_sdk.core.models import TextChunks, TextChunk
from graphrag_sdk.core.context import Context


class SentenceChunking(ChunkingStrategy):
    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        sentences = text.split(". ")
        return TextChunks(
            chunks=[TextChunk(text=s, index=i) for i, s in enumerate(sentences)]
        )

result = await rag.ingest("doc.txt", chunker=SentenceChunking())
```

### Custom Extraction

```python
from graphrag_sdk import ExtractionStrategy
from graphrag_sdk.core.models import GraphData, GraphNode, GraphRelationship, TextChunks, GraphSchema
from graphrag_sdk.core.context import Context


class MyExtractor(ExtractionStrategy):
    async def extract(self, chunks: TextChunks, schema: GraphSchema, ctx: Context) -> GraphData:
        nodes, relationships = [], []
        for chunk in chunks.chunks:
            # Your extraction logic here — call an LLM, use NER, etc.
            ...
        return GraphData(nodes=nodes, relationships=relationships)

result = await rag.ingest("doc.txt", extractor=MyExtractor())
```

### Custom Resolution

```python
from graphrag_sdk import ResolutionStrategy
from graphrag_sdk.core.models import GraphData, ResolutionResult
from graphrag_sdk.core.context import Context


class FuzzyResolution(ResolutionStrategy):
    async def resolve(self, graph_data: GraphData, ctx: Context) -> ResolutionResult:
        # Your deduplication logic — fuzzy matching, embedding similarity, etc.
        return ResolutionResult(
            nodes=deduplicated_nodes,
            relationships=remapped_rels,
            merged_count=num_merged,
        )

result = await rag.ingest("doc.txt", resolver=FuzzyResolution())
```

### Custom Retrieval

```python
from graphrag_sdk import RetrievalStrategy
from graphrag_sdk.core.models import RawSearchResult
from graphrag_sdk.core.context import Context


class MultiHopRetrieval(RetrievalStrategy):
    def __init__(self, graph_store, vector_store, embedder, hops=2):
        super().__init__(graph_store, vector_store)
        self.embedder = embedder
        self.hops = hops

    async def _execute(self, query: str, ctx: Context, **kwargs) -> RawSearchResult:
        # 1. Vector search for seed chunks
        vec = await self.embedder.aembed_query(query)
        chunks = await self._vector.search(vec, top_k=10)

        # 2. Multi-hop graph traversal from seed chunks
        for chunk in chunks:
            entities = await self._graph.get_connected_entities(chunk["id"], max_hops=self.hops)
            # ... build expanded context

        return RawSearchResult(records=expanded_results)

answer = await rag.query("Complex question?", strategy=MultiHopRetrieval(...))
```

### Custom Reranking

```python
from graphrag_sdk import RerankingStrategy
from graphrag_sdk.core.models import RetrieverResult
from graphrag_sdk.core.context import Context


class CrossEncoderReranker(RerankingStrategy):
    async def rerank(self, query: str, result: RetrieverResult, ctx: Context) -> RetrieverResult:
        # Score and reorder items
        scored = [(item, cross_encoder.score(query, item.content)) for item in result.items]
        scored.sort(key=lambda x: x[1], reverse=True)
        result.items = [item for item, _ in scored]
        return result

answer = await rag.query("Question?", reranker=CrossEncoderReranker())
```

### Custom Prompt Template

```python
answer = await rag.query(
    "What happened in chapter 3?",
    prompt_template=(
        "You are an expert analyst. Use ONLY the provided context to answer.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Provide a detailed answer with citations:"
    ),
)
```

## Built-in Strategies

### Extraction Strategies

| Strategy | Description |
|----------|-------------|
| `SchemaGuidedExtraction` | LLM extracts entities/relationships as JSON, constrained by schema. Default. |
| `MergedExtraction` | Combines LightRAG-style typed extraction with HippoRAG-style fact triples and entity mentions. Supports optional gleaning (second LLM pass). |

```python
from graphrag_sdk.ingestion.extraction_strategies.merged_extraction import MergedExtraction

extractor = MergedExtraction(
    llm=my_llm,
    enable_gleaning=True,   # Second LLM pass to catch missed entities
    max_concurrency=12,     # Parallel chunk processing
)

result = await rag.ingest("novel.txt", extractor=extractor)
```

### Resolution Strategies

| Strategy | Description |
|----------|-------------|
| `ExactMatchResolution` | Groups nodes by `(label, id)`, merges properties. Default. |
| `DescriptionMergeResolution` | Groups by normalized name, merges descriptions. Uses LLM summarization when many descriptions accumulate. |

```python
from graphrag_sdk.ingestion.resolution_strategies.description_merge import DescriptionMergeResolution

resolver = DescriptionMergeResolution(
    llm=my_llm,
    force_summary_threshold=3,   # Use LLM summary when 3+ descriptions exist
)

result = await rag.ingest("doc.txt", resolver=resolver)
```

### Chunking Strategies

| Strategy | Description |
|----------|-------------|
| `FixedSizeChunking` | Sliding window with configurable size and overlap. Default: 1000 chars, 100 overlap. |

```python
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking

chunker = FixedSizeChunking(chunk_size=1500, chunk_overlap=200)
result = await rag.ingest("doc.txt", chunker=chunker)
```

### Loaders

| Strategy | Description |
|----------|-------------|
| `TextLoader` | Plain text and markdown files. Default for non-PDF. |
| `PdfLoader` | PDF files via `pypdf`. Requires `pip install graphrag-sdk[pdf]`. |

## Execution Context

The `Context` object threads through every strategy call for tracing, budgeting, and multi-tenancy:

```python
from graphrag_sdk import Context

ctx = Context(
    tenant_id="customer-123",          # Multi-tenant isolation
    latency_budget_ms=5000.0,          # Stop extraction if budget exceeded
)

result = await rag.ingest("doc.pdf", ctx=ctx)
answer = await rag.query("Question?", ctx=ctx)

print(f"Elapsed: {ctx.elapsed_ms:.0f}ms")
print(f"Budget remaining: {ctx.remaining_budget_ms:.0f}ms")
```

## Direct Pipeline Usage

For advanced control, use `IngestionPipeline` directly instead of the `GraphRAG` facade:

```python
from graphrag_sdk import (
    IngestionPipeline, GraphStore, VectorStore,
    ConnectionConfig, FalkorDBConnection, GraphSchema, Context,
)
from graphrag_sdk.ingestion.loaders.text_loader import TextLoader
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
from graphrag_sdk.ingestion.extraction_strategies.merged_extraction import MergedExtraction
from graphrag_sdk.ingestion.resolution_strategies.description_merge import DescriptionMergeResolution

conn = FalkorDBConnection(ConnectionConfig(host="localhost", graph_name="my_graph"))
graph_store = GraphStore(conn)
vector_store = VectorStore(conn, embedder=my_embedder)

pipeline = IngestionPipeline(
    loader=TextLoader(),
    chunker=FixedSizeChunking(chunk_size=1500, chunk_overlap=200),
    extractor=MergedExtraction(llm=my_llm, max_concurrency=12),
    resolver=DescriptionMergeResolution(llm=my_llm),
    graph_store=graph_store,
    vector_store=vector_store,
    schema=GraphSchema(entities=[...], relations=[...]),
)

result = await pipeline.run("document.txt", Context())
```

## Direct Storage Access

Query the graph and vector stores directly for custom retrieval logic:

```python
# Vector search
vector = await my_embedder.aembed_query("search query")
chunks = await vector_store.search(vector, top_k=10, label="Chunk")

# Entity vector search
entities = await vector_store.search_entities(vector, top_k=5)

# Graph traversal — get entities connected to a chunk
connected = await graph_store.get_connected_entities("chunk-uuid", max_hops=2)

# Raw Cypher
result = await graph_store.query_raw(
    "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name LIMIT 10"
)
```

## Graph Inspection

Use `GraphVisualizer` to inspect your knowledge graph:

```python
from graphrag_sdk.utils.graph_viz import GraphVisualizer

viz = GraphVisualizer(conn)
stats = await viz.get_stats()
print(f"Nodes: {stats['node_count']}, Edges: {stats['relationship_count']}")
print(f"Labels: {stats['labels']}")

# Human-readable summary
print(await viz.describe())

# Sample nodes of a specific type
people = await viz.sample_nodes(label="Person", limit=5)
```

## Telemetry

Built-in span-based tracing for performance profiling:

```python
from graphrag_sdk.telemetry.tracer import Tracer

tracer = Tracer(service_name="my-app")

with tracer.span("ingestion") as span:
    result = await rag.ingest("doc.pdf")
    span.metadata["nodes"] = result.nodes_created

with tracer.span("query") as span:
    answer = await rag.query("Question?")
    span.metadata["answer_length"] = len(answer.answer)

print(tracer.summary())
```

## Project Structure

```
graphrag_sdk/
├── api/
│   └── main.py                          # GraphRAG facade
├── core/
│   ├── connection.py                    # FalkorDB connection + pooling
│   ├── context.py                       # Execution context (tracing, budgets)
│   ├── exceptions.py                    # Exception hierarchy
│   ├── models.py                        # All Pydantic data models
│   └── providers.py                     # Embedder + LLMInterface ABCs
├── ingestion/
│   ├── pipeline.py                      # 8-step sequential orchestrator
│   ├── loaders/                         # TextLoader, PdfLoader
│   ├── chunking_strategies/             # FixedSizeChunking
│   ├── extraction_strategies/           # SchemaGuided, MergedExtraction
│   └── resolution_strategies/           # ExactMatch, DescriptionMerge
├── retrieval/
│   ├── strategies/                      # LocalRetrieval (vector + graph)
│   └── reranking_strategies/            # RerankingStrategy ABC
├── storage/
│   ├── graph_store.py                   # Batched Cypher upserts (UNWIND)
│   └── vector_store.py                  # Vector + fulltext indexing
├── telemetry/
│   └── tracer.py                        # Span-based performance tracing
└── utils/
    └── graph_viz.py                     # Graph inspection utilities
```

## Configuration Reference

### ConnectionConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | `"localhost"` | FalkorDB host |
| `port` | `6379` | FalkorDB port |
| `username` | `None` | Auth username (optional for local) |
| `password` | `None` | Auth password (optional for local) |
| `graph_name` | `"knowledge_graph"` | Name of the graph in FalkorDB |
| `max_connections` | `16` | Connection pool size |
| `retry_count` | `3` | Number of query retries |
| `retry_delay` | `1.0` | Base retry delay in seconds (exponential backoff) |

### GraphRAG Constructor

| Parameter | Required | Description |
|-----------|----------|-------------|
| `connection` | Yes | `FalkorDBConnection` or `ConnectionConfig` |
| `llm` | Yes | LLM provider implementing `LLMInterface` |
| `embedder` | Yes | Embedding provider implementing `Embedder` |
| `schema` | No | `GraphSchema` for extraction constraints |
| `retrieval_strategy` | No | Default retrieval strategy (uses `LocalRetrieval` if None) |

## FAQ

**Which databases are supported?**
GraphRAG SDK v2 is built for FalkorDB. The storage layer is abstracted behind `GraphStore` and `VectorStore`, but currently only FalkorDB is implemented.

**What LLM providers work?**
Any provider that implements the `LLMInterface` and `Embedder` ABCs. The SDK has optional dependencies for OpenAI, Anthropic, Cohere, and sentence-transformers.

**How does the SDK handle failures?**
LLM calls retry automatically with exponential backoff (1s, 2s, 4s). Batch graph writes fall back to per-item upserts on failure. Content filter errors (Azure) are retried and gracefully skipped if persistent.

**What about PDF files with special characters?**
The SDK strips null bytes from all property values and IDs before writing to FalkorDB, preventing parse errors from PDF-extracted text.

**Can I use this without a schema?**
Yes. Without a schema, the extractor operates in open mode and extracts all entities and relationships it finds. Schema is recommended for domain-specific use cases.

**How does deduplication work?**
The resolution step groups entities by normalized name and merges their properties and descriptions. With `DescriptionMergeResolution`, an LLM can summarize accumulated descriptions.

## Community

- [GitHub Issues](https://github.com/FalkorDB/GraphRAG-SDK/issues) — bug reports and feature requests
- [Discord](https://discord.com/invite/6M4QwDXn2w) — questions and discussion
- [FalkorDB Cloud](https://app.falkordb.cloud) — managed FalkorDB hosting

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
