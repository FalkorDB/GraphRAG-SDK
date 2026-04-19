# Getting Started with GraphRAG SDK

A step-by-step tutorial for building a knowledge graph from documents and querying it with natural language.

---

## 1. Prerequisites

- **Python 3.10+**
- **FalkorDB** (easiest via Docker -- see below)
- An **LLM API key** from one of the supported providers:
  - Azure OpenAI
  - OpenAI
  - Anthropic

---

## 2. Installation

Install the SDK with all optional dependencies:

```bash
pip install "graphrag-sdk[all]"
```

For a local editable install from a cloned repo:

```bash
pip install -e "./graphrag_sdk[all]"
```

---

## 3. Start FalkorDB

Run FalkorDB as a Docker container:

```bash
docker run -d --name falkordb -p 6379:6379 falkordb/falkordb
```

Verify it is running:

```bash
docker ps | grep falkordb
```

---

## 4. Configure Environment

Set the environment variables for your LLM provider. The example below uses Azure OpenAI:

```bash
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
```

If you use a `.env` file, load it yourself before importing the SDK (e.g., via `python-dotenv` or `export` commands). The SDK reads environment variables but does not auto-load `.env` files.

---

## 5. Define a Schema

A `GraphSchema` tells the extraction pipeline which entity and relationship types to look for in your documents.

```python
from graphrag_sdk import GraphSchema, EntityType, RelationType

schema = GraphSchema(
    entities=[
        EntityType(label="Person", description="A human being"),
        EntityType(label="Organization", description="A company or institution"),
        EntityType(label="Place", description="A geographic location"),
    ],
    relations=[
        RelationType(label="WORKS_AT", description="Employment relationship"),
        RelationType(label="LOCATED_IN", description="Geographic location"),
        RelationType(label="KNOWS", description="Personal acquaintance"),
    ],
)
```

You can add as many entity and relationship types as your domain requires. Descriptions help the LLM decide when to extract each type.

---

## 6. Initialize GraphRAG

Create a `GraphRAG` instance by providing a connection, LLM, embedder, and schema:

```python
from graphrag_sdk import GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder

rag = GraphRAG(
    connection=ConnectionConfig(host="localhost", graph_name="my_graph"),
    llm=LiteLLM(model="azure/gpt-4.1"),
    embedder=LiteLLMEmbedder(model="azure/text-embedding-ada-002"),
    schema=schema,
    embedding_dimension=1536,  # must match your embedding model's output dimension
)
```

If your embedding model produces vectors with a different dimensionality (e.g., `text-embedding-3-large` at 256 or 1024 dimensions), set `embedding_dimension` accordingly. The default is `1536`.

`ConnectionConfig` accepts additional parameters such as `port`, `username`, `password`, and `query_timeout_ms`. See [docs/configuration.md](configuration.md) for the full list.

> **Alternative providers:** The SDK also exports `OpenRouterLLM` and `OpenRouterEmbedder` for use with OpenRouter. See [docs/configuration.md](configuration.md) for details.

---

## 7. Ingest a Document

### From a file path

```python
result = await rag.ingest("path/to/document.txt")
print(f"Created {result.nodes_created} nodes, {result.relationships_created} relationships")
print(f"Indexed {result.chunks_indexed} chunks")
```

### From raw text

```python
result = await rag.ingest("acme_doc", text="Acme Corp was founded in 1985 by Jane Doe in Austin, Texas.")
print(f"Created {result.nodes_created} nodes, {result.relationships_created} relationships")
print(f"Indexed {result.chunks_indexed} chunks")
```

The ingestion pipeline runs a 9-step process: Load, Chunk, Lexical Graph, Extract (includes quality filtering), Prune, Resolve, Write, then Mentions and Chunk Indexing in parallel.

---

## 8. Query the Knowledge Graph

### Retrieve context only

Use `retrieve()` when you want to inspect the context or use your own LLM:

```python
context = await rag.retrieve("Who works at Acme Corp?")
for item in context.items:
    print(f"[{item.score:.2f}] {item.content[:100]}...")
```

### Generate an answer

Use `completion()` for the full RAG pipeline — retrieval + answer generation:

```python
result = await rag.completion("Who works at Acme Corp?")
print(result.answer)
```

### With context inspection

Pass `return_context=True` to see which chunks and entities the retriever used to build the answer:

```python
result = await rag.completion("Who works at Acme Corp?", return_context=True)
for item in result.retriever_result.items:
    print(f"[{item.metadata.get('section', '')}] {item.content[:100]}...")
```

### Multi-turn conversations

`completion()` supports native multi-turn conversations. Messages are passed directly to the LLM's chat API as structured messages:

```python
from graphrag_sdk import ChatMessage

result = await rag.completion(
    "What happened to her after that?",
    history=[
        ChatMessage(role="user", content="Who works at Acme Corp?"),
        ChatMessage(role="assistant", content="Jane Doe works at Acme Corp."),
    ],
)
print(result.answer)
```

You can also pass history as plain dicts:

```python
result = await rag.completion(
    "Tell me more.",
    history=[
        {"role": "user", "content": "Who founded Acme?"},
        {"role": "assistant", "content": "Jane Doe founded Acme in 1985."},
    ],
)
```

Supported roles: `"system"`, `"user"`, `"assistant"`. Invalid roles raise `ValueError`.

---

## 9. Inspect Graph Contents

Use `get_statistics()` to see a summary of what the graph contains:

```python
stats = await rag.graph_store.get_statistics()
print(f"Nodes: {stats['node_count']}, Edges: {stats['edge_count']}")
```

You can also run raw Cypher queries against the graph:

```python
results = await rag.graph_store.query_raw("MATCH (p:Person)-[:WORKS_AT]->(o:Organization) RETURN p.name, o.name LIMIT 10")
for row in results.result_set:
    print(row)
```

---

## 10. Finalize

After all documents have been ingested, run `finalize()` to deduplicate entities, backfill embeddings, and create indexes:

```python
results = await rag.finalize()
print(f"Deduplicated: {results['entities_deduplicated']}")
print(f"Embedded: {results['entities_embedded']} entities, {results['relationships_embedded']} relationships")
```

This step is important for query accuracy. It merges duplicate entities (e.g., "J. Doe" and "Jane Doe") and ensures all entities have vector embeddings for semantic search.

---

## 11. Next Steps

- [docs/configuration.md](configuration.md) -- Tuning connection settings, chunking parameters, and retrieval options.
- [docs/strategies.md](strategies.md) -- Custom extraction and resolution strategies.
- [docs/benchmark.md](benchmark.md) -- Reproducing benchmark results on the 100-question novel corpus.

---

## Synchronous API

If you are not in an async context, use the synchronous convenience methods:

```python
result = rag.ingest_sync("path/to/document.txt")
context = rag.retrieve_sync("Who works at Acme Corp?")
result = rag.completion_sync("Who works at Acme Corp?")
results = rag.finalize_sync()
```

These wrap the async methods in `asyncio.run()` and are useful for scripts and notebooks.
