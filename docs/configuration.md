# GraphRAG SDK v2 -- Configuration Reference

This document is the comprehensive configuration reference for GraphRAG SDK v2. Each section covers a configurable component, its parameters, defaults, and usage examples.

---

## 1. ConnectionConfig

`ConnectionConfig` is a dataclass that defines how the SDK connects to a FalkorDB instance. It is passed to `GraphRAG` or used to create a `FalkorDBConnection` directly.

### Fields

| Field              | Type              | Default             | Description                                                         |
|--------------------|-------------------|---------------------|---------------------------------------------------------------------|
| `host`             | `str`             | `"localhost"`       | FalkorDB server hostname or IP address.                             |
| `port`             | `int`             | `6379`              | FalkorDB server port.                                               |
| `username`         | `str \| None`     | `None`              | Authentication username (omit for unauthenticated connections).     |
| `password`         | `str \| None`     | `None`              | Authentication password.                                            |
| `graph_name`       | `str`             | `"knowledge_graph"` | Name of the FalkorDB graph to operate on.                           |
| `max_connections`  | `int`             | `16`                | Maximum number of connections in the Redis `BlockingConnectionPool`.|
| `retry_count`      | `int`             | `3`                 | Number of retry attempts for transient query failures.              |
| `retry_delay`      | `float`           | `1.0`               | Base delay (seconds) between retries (multiplied by attempt number).|
| `pool_timeout`     | `float`           | `30.0`              | Timeout (seconds) waiting to acquire a connection from the pool.    |
| `query_timeout_ms` | `int \| None`     | `10_000`            | Per-query timeout in milliseconds forwarded to FalkorDB. Set to `None` to disable. |

### Creating from a URL

`ConnectionConfig.from_url()` parses a `redis://` URL and returns a `ConnectionConfig`:

```python
config = ConnectionConfig.from_url(
    "redis://user:pass@my-falkordb.example.com:6380",
    graph_name="my_graph",
    query_timeout_ms=15_000,
)
```

The URL format is `redis://[user:pass@]host[:port][/db]`. Any keyword argument overrides the value parsed from the URL.

### Passing to GraphRAG

You can pass either a `ConnectionConfig` or a pre-built `FalkorDBConnection`:

```python
from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.api.main import GraphRAG

# Option A: pass a config (connection created internally)
rag = GraphRAG(
    connection=ConnectionConfig(host="localhost", port=6379, graph_name="novels"),
    llm=my_llm,
    embedder=my_embedder,
)

# Option B: pass a FalkorDBConnection directly (full control)
conn = FalkorDBConnection(ConnectionConfig(host="10.0.0.5", password="secret"))
rag = GraphRAG(connection=conn, llm=my_llm, embedder=my_embedder)
```

### Retry Behavior

Queries are retried up to `retry_count` times with linear backoff (`retry_delay * attempt_number`). Non-transient errors -- those containing `"already indexed"`, `"already exists"`, or `"unknown index"` -- are raised immediately without retrying.

---

## 2. LLM Providers

The SDK defines an abstract `LLMInterface` base class. All LLM providers must implement `invoke()` for synchronous calls. Async calls (`ainvoke`) default to running `invoke` in a thread pool but can be overridden for true async support.

### Common Parameters

The `LLMInterface` base class accepts:

| Parameter         | Type                    | Default | Description                                    |
|-------------------|-------------------------|---------|------------------------------------------------|
| `model_name`      | `str`                   | --      | Model identifier (e.g. `"gpt-4.1"`).          |
| `model_params`    | `dict[str, Any] \| None`| `{}`    | Provider-specific parameters.                  |
| `max_concurrency` | `int`                   | `12`    | Concurrency limit for `abatch_invoke()`.       |

### LiteLLM (Recommended)

LiteLLM supports 100+ LLM providers through a unified interface. Install with `pip install graphrag-sdk[litellm]`.

```python
from graphrag_sdk.core.providers import LiteLLM

# OpenAI
llm = LiteLLM(model="gpt-4.1", api_key="sk-...")

# Azure OpenAI
llm = LiteLLM(
    model="azure/gpt-4.1",
    api_key="your-azure-key",
    api_base="https://your-resource.openai.azure.com/",
    api_version="2024-06-01",
    temperature=0.0,
    max_tokens=4096,
)

# Anthropic
llm = LiteLLM(model="anthropic/claude-sonnet-4-20250514", api_key="sk-ant-...")
```

**Parameters:**

| Parameter     | Type            | Default | Description                              |
|---------------|-----------------|---------|------------------------------------------|
| `model`       | `str`           | --      | Model identifier in LiteLLM format.      |
| `api_key`     | `str \| None`   | `None`  | API key (or set via environment variable).|
| `api_base`    | `str \| None`   | `None`  | API base URL (required for Azure).       |
| `api_version` | `str \| None`   | `None`  | API version string (required for Azure). |
| `temperature` | `float`         | `0.0`   | Sampling temperature.                    |
| `max_tokens`  | `int \| None`   | `None`  | Maximum tokens in response.              |

### OpenRouter

OpenRouter provides access to many models through a single API. Install with `pip install graphrag-sdk[openrouter]`.

```python
from graphrag_sdk.core.providers import OpenRouterLLM

llm = OpenRouterLLM(
    model="anthropic/claude-sonnet-4-20250514",
    api_key="sk-or-...",
    temperature=0.0,
    max_tokens=4096,
)
```

**Parameters:**

| Parameter       | Type                     | Default                                       | Description                          |
|-----------------|--------------------------|-----------------------------------------------|--------------------------------------|
| `model`         | `str`                    | --                                            | Model identifier in OpenRouter format.|
| `api_key`       | `str \| None`            | `os.environ["OPENROUTER_API_KEY"]`            | OpenRouter API key.                  |
| `temperature`   | `float`                  | `0.0`                                         | Sampling temperature.                |
| `max_tokens`    | `int \| None`            | `None`                                        | Maximum tokens in response.          |
| `extra_headers` | `dict[str, str] \| None` | `None`                                        | Additional HTTP headers.             |

### Azure OpenAI via Environment Variables

When using LiteLLM with Azure, the following environment variables are recognized:

```bash
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-06-01"
```

Then configure the LLM:

```python
import os
from graphrag_sdk.core.providers import LiteLLM

llm = LiteLLM(
    model="azure/gpt-4.1",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_base=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
```

### Custom LLM Provider

Implement the `LLMInterface` abstract class:

```python
from graphrag_sdk.core.providers import LLMInterface
from graphrag_sdk.core.models import LLMResponse

class MyLLM(LLMInterface):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name)

    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        # Call your LLM here
        text = my_custom_api(prompt)
        return LLMResponse(content=text)
```

---

## 3. Embedder Providers

The SDK defines an abstract `Embedder` base class with `embed_query()` (single text) and `embed_documents()` (batch). Batch embedding is critical for performance.

### Performance Note: Batch Embedding

Individual embedding calls to Azure OpenAI take approximately 0.22 seconds each. A batch of 500 texts takes approximately 8 seconds. Always use batch embedding (`embed_documents` / `aembed_documents`) rather than looping over `embed_query`.

### LiteLLMEmbedder

Supports OpenAI, Azure, Cohere, and other embedding models via LiteLLM.

```python
from graphrag_sdk.core.providers import LiteLLMEmbedder

# Azure OpenAI
embedder = LiteLLMEmbedder(
    model="azure/text-embedding-ada-002",
    api_key="your-key",
    api_base="https://your-resource.openai.azure.com/",
    api_version="2024-06-01",
    batch_size=500,
)

# OpenAI
embedder = LiteLLMEmbedder(model="text-embedding-ada-002", api_key="sk-...")
```

**Parameters:**

| Parameter     | Type            | Default | Description                                                    |
|---------------|-----------------|---------|----------------------------------------------------------------|
| `model`       | `str`           | --      | Model identifier in LiteLLM format.                            |
| `api_key`     | `str \| None`   | `None`  | API key.                                                       |
| `api_base`    | `str \| None`   | `None`  | API base URL.                                                  |
| `api_version` | `str \| None`   | `None`  | API version string.                                            |
| `batch_size`  | `int`           | `2048`  | Maximum texts per batch call. Azure users should set to `500`. |

For Azure OpenAI, set `batch_size=500` to stay within the API rate limits. The default of 2048 works well for OpenAI's direct API.

### OpenRouterEmbedder

```python
from graphrag_sdk.core.providers import OpenRouterEmbedder

embedder = OpenRouterEmbedder(
    model="openai/text-embedding-ada-002",
    api_key="sk-or-...",
    batch_size=2048,
)
```

**Parameters:**

| Parameter       | Type                     | Default                                       | Description              |
|-----------------|--------------------------|-----------------------------------------------|--------------------------|
| `model`         | `str`                    | --                                            | Model identifier.        |
| `api_key`       | `str \| None`            | `os.environ["OPENROUTER_API_KEY"]`            | API key.                 |
| `batch_size`    | `int`                    | `2048`                                        | Maximum texts per batch. |
| `extra_headers` | `dict[str, str] \| None` | `None`                                        | Additional HTTP headers. |

### Custom Embedder

Implement the `Embedder` abstract class:

```python
from graphrag_sdk.core.providers import Embedder

class MyEmbedder(Embedder):
    def embed_query(self, text: str, **kwargs) -> list[float]:
        return my_embedding_api(text)

    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        # Implement batch embedding for performance
        return my_batch_embedding_api(texts)
```

Override `aembed_query` and `aembed_documents` if your provider supports true async. The defaults run the sync methods in a thread pool via `asyncio.to_thread`.

### Binary-Split Error Recovery

Both `LiteLLMEmbedder` and `OpenRouterEmbedder` implement binary-split error recovery for batch embedding. If a batch fails with a transient error, the batch is split in half and each half is retried recursively. Non-transient errors (401, 403, authentication failures) are raised immediately.

---

## 4. GraphSchema

`GraphSchema` defines the structure of your knowledge graph. It constrains LLM extraction and powers the pruning step that filters non-conforming data.

### Components

**EntityType** -- defines a node type:

| Field         | Type                | Default       | Description                            |
|---------------|---------------------|---------------|----------------------------------------|
| `label`       | `str`               | --            | The node label (e.g. `"Person"`).      |
| `description` | `str \| None`       | `None`        | Human-readable description.            |
| `properties`  | `list[PropertyType]` | `[]`          | Expected properties on this node type. |

**RelationType** -- defines a relationship type:

| Field         | Type                | Default       | Description                              |
|---------------|---------------------|---------------|------------------------------------------|
| `label`       | `str`               | --            | The relationship type (e.g. `"KNOWS"`).  |
| `description` | `str \| None`       | `None`        | Human-readable description.              |
| `properties`  | `list[PropertyType]` | `[]`          | Expected properties on this relationship.|

**PropertyType** -- defines a property on a node or relationship:

| Field         | Type            | Default     | Description                                                  |
|---------------|-----------------|-------------|--------------------------------------------------------------|
| `name`        | `str`           | --          | Property name.                                               |
| `type`        | `str`           | `"STRING"`  | Type hint: `STRING`, `INTEGER`, `FLOAT`, `BOOLEAN`, `DATE`, `LIST`. |
| `description` | `str \| None`   | `None`      | Human-readable description.                                  |
| `required`    | `bool`          | `False`     | Whether the property is required.                            |

**SchemaPattern** -- defines a valid source-relationship-target triple:

| Field          | Type  | Description                        |
|----------------|-------|------------------------------------|
| `source`       | `str` | Source entity type label.          |
| `relationship` | `str` | Relationship type label.           |
| `target`       | `str` | Target entity type label.          |

### Example Schema Definition

```python
from graphrag_sdk.core.models import (
    EntityType, RelationType, PropertyType, SchemaPattern, GraphSchema,
)

schema = GraphSchema(
    entities=[
        EntityType(
            label="Person",
            description="A character or real person",
            properties=[
                PropertyType(name="name", type="STRING", required=True),
                PropertyType(name="age", type="INTEGER"),
                PropertyType(name="occupation", type="STRING"),
            ],
        ),
        EntityType(
            label="Location",
            description="A geographical place or setting",
            properties=[
                PropertyType(name="name", type="STRING", required=True),
                PropertyType(name="country", type="STRING"),
            ],
        ),
        EntityType(
            label="Organization",
            description="A company, institution, or group",
        ),
    ],
    relations=[
        RelationType(label="LIVES_IN", description="Person resides at location"),
        RelationType(label="WORKS_FOR", description="Person is employed by organization"),
        RelationType(label="LOCATED_IN", description="Organization is located at a place"),
        RelationType(
            label="KNOWS",
            description="Two people know each other",
            properties=[
                PropertyType(name="since", type="DATE"),
            ],
        ),
    ],
    patterns=[
        SchemaPattern(source="Person", relationship="LIVES_IN", target="Location"),
        SchemaPattern(source="Person", relationship="WORKS_FOR", target="Organization"),
        SchemaPattern(source="Organization", relationship="LOCATED_IN", target="Location"),
        SchemaPattern(source="Person", relationship="KNOWS", target="Person"),
    ],
)
```

### Open Schema Mode

If no entity types or relation types are defined (empty `GraphSchema()`), the extraction operates in open-schema mode and the pruning step is skipped. This lets the LLM extract any entities and relationships it finds.

---

## 5. Pipeline Tuning

### Chunking Parameters

`FixedSizeChunking` splits text into fixed-size character windows with overlap.

| Parameter       | Type  | Default | Benchmark Value | Description                                           |
|-----------------|-------|---------|-----------------|-------------------------------------------------------|
| `chunk_size`    | `int` | `1000`  | `1500`          | Maximum characters per chunk.                         |
| `chunk_overlap` | `int` | `100`   | `200`           | Overlapping characters between consecutive chunks.    |

```python
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking

chunker = FixedSizeChunking(chunk_size=1500, chunk_overlap=200)

result = await rag.ingest("document.txt", chunker=chunker)
```

Larger chunks provide more context per extraction call but increase LLM token usage. The benchmark-optimized values (1500/200) balance extraction quality against cost.

### Extraction Strategy Parameters

**HybridExtraction** -- composable 2-step extraction (GLiNER2 NER + LLM relationship extraction):

| Parameter          | Type            | Default        | Description                                                |
|--------------------|-----------------|----------------|------------------------------------------------------------|
| `entity_extractor` | `EntityExtractor \| None` | `None` (GLiNER2) | Pluggable NER backend. Pass `EntityExtractor(llm=llm)` for LLM-based NER. |
| `coref_resolver`   | `CorefResolver \| None` | `None` | Optional coreference resolution (e.g. `FastCorefResolver()`). |
| `embedder`         | `Embedder \| None` | `None` | Embedder instance (reserved for future use). |
| `entity_types`     | `list[str] \| None` | `None` (11 default types) | Custom entity types. Overridden by `schema.entities` if set. |
| `max_concurrency`  | `int \| None`   | `None` (uses LLM default) | Maximum parallel LLM calls during extraction. |

```python
from graphrag_sdk import HybridExtraction, EntityExtractor

# Default: GLiNER2 for entity NER, LLM for relationship extraction
extractor = HybridExtraction(llm=my_llm)

# With LLM-based entity NER instead of GLiNER2
extractor = HybridExtraction(
    llm=my_llm,
    entity_extractor=EntityExtractor(llm=my_llm),
)

# With custom entity types
extractor = HybridExtraction(
    llm=my_llm,
    entity_types=["Gene", "Protein", "Disease"],
)

result = await rag.ingest("document.txt", extractor=extractor)
```

### LLM Concurrency

The `LLMInterface.max_concurrency` parameter (default: 12) controls how many LLM calls run in parallel during `abatch_invoke()`. Set it lower to avoid rate limits:

```python
llm = LiteLLM(model="azure/gpt-4.1", api_key="...")
llm.max_concurrency = 8  # limit to 8 parallel calls
```

For `HybridExtraction`, you can also pass `max_concurrency` directly:

```python
extractor = HybridExtraction(llm=my_llm, max_concurrency=6)
```

---

## 6. Retrieval Tuning

### MultiPathRetrieval

`MultiPathRetrieval` is the default retrieval strategy. It combines multiple search paths with cosine reranking.

| Parameter            | Type  | Default | Description                                           |
|----------------------|-------|---------|-------------------------------------------------------|
| `chunk_top_k`        | `int` | `15`    | Final chunks kept after cosine reranking.             |
| `max_entities`       | `int` | `30`    | Maximum entities to include in context.               |
| `max_relationships`  | `int` | `20`    | Maximum relationships in context (after 1-hop + 2-hop expansion). |
| `rel_top_k`          | `int` | `15`    | RELATES edge vector search results to retrieve.       |
| `keyword_limit`      | `int` | `10`    | Maximum keywords extracted from the question.         |

```python
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval

retriever = MultiPathRetrieval(
    graph_store=rag.graph_store,
    vector_store=rag.vector_store,
    embedder=rag.embedder,
    llm=rag.llm,
    chunk_top_k=20,           # more passages for complex questions
    max_entities=40,          # wider entity coverage
    max_relationships=30,     # more graph context
    rel_top_k=20,             # more RELATES edge hits
    keyword_limit=12,         # extract more keywords
)

result = await rag.query("What happened?", strategy=retriever)
```

### Retrieval Pipeline (9 Steps)

The retrieval pipeline proceeds as follows:

1. **Keyword extraction** -- stopword filtering + LLM proper-noun extraction.
2. **Embed question** -- single embedding API call for the query.
3. **RELATES edge vector search** -- finds fact strings and entity entry points via edge embeddings.
4. **Entity discovery** (2 paths) -- Cypher `CONTAINS` on entity names + fulltext search on the `__Entity__` index. Merged with entities from step 3.
5. **Relationship expansion** -- 1-hop (top 15 entities, limit 150) + 2-hop (top 5 entities, limit 25) traversal of RELATES edges.
6. **Chunk retrieval** (4 paths) -- fulltext search, vector search, MENTIONED_IN traversal, and 2-hop entity-to-neighbor-to-chunk traversal.
7. **Source document names** -- batch-fetch document paths via PART_OF edges.
8. **Cosine reranking** -- batch-embed candidate chunks and sort by cosine similarity to the query vector.
9. **Context assembly** -- structured sections: hint, entities, relationships, facts, passages.

### Overriding the Default Strategy

Pass a custom strategy to individual queries or set it as the default:

```python
# Per-query override
result = await rag.query("...", strategy=my_custom_retriever)

# Default at init time
rag = GraphRAG(
    connection=config,
    llm=my_llm,
    embedder=my_embedder,
    retrieval_strategy=my_custom_retriever,
)
```

---

## 7. Post-Ingestion

After all documents have been ingested, run post-ingestion steps to deduplicate entities, backfill embeddings, and ensure all indexes exist.

### `finalize()` -- All-In-One

The recommended approach is to call `finalize()` after all ingestion is complete. It bundles four steps in order:

1. `deduplicate_entities()` -- global exact-name deduplication.
2. `backfill_entity_embeddings()` -- embed entity names for vector search.
3. `embed_relationships()` -- embed fact text on RELATES edges.
4. `ensure_indices()` -- create all 5 standard indexes (idempotent).

```python
# After ingesting all documents:
stats = await rag.finalize()
print(stats)
# {
#     "entities_deduplicated": 142,
#     "entities_embedded": 3200,
#     "relationships_embedded": 8500,
#     "indexes": {
#         "vector_Chunk": True,
#         "vector___Entity__": True,
#         "vector_RELATES": True,
#         "fulltext_Chunk": True,
#         "fulltext___Entity__": True,
#     },
# }
```

A synchronous convenience method is also available:

```python
stats = rag.finalize_sync()
```

### `deduplicate_entities()` -- Entity Deduplication

Call this when you need fine-grained control over deduplication.

```python
merged_count = await rag.deduplicate_entities(
    fuzzy=False,                  # True to also run embedding-based dedup
    similarity_threshold=0.9,     # cosine threshold for fuzzy matching
    batch_size=500,               # entities per query batch
)
```

**Parameters:**

| Parameter              | Type    | Default | Description                                                         |
|------------------------|---------|---------|---------------------------------------------------------------------|
| `fuzzy`                | `bool`  | `False` | If `True`, runs a second fuzzy dedup phase using embedding similarity.|
| `similarity_threshold` | `float` | `0.9`   | Cosine similarity threshold for fuzzy matching.                     |
| `batch_size`           | `int`   | `500`   | Entities per query batch.                                           |

**Phase 1 (always runs): Exact name match.** Groups entities by normalized name (lowercase, stripped) and label to prevent cross-type merging. Keeps the entity with the longest description as the survivor. Remaps all RELATES and MENTIONED_IN edges from duplicates to the survivor, then deletes the duplicate nodes.

**Phase 2 (optional, `fuzzy=True`): Embedding-based match.** Re-fetches all surviving entities, batch-embeds their names, computes pairwise cosine similarity in memory-efficient blocks (1000 entities per block), and merges near-duplicates above the threshold.

### `backfill_entity_embeddings()` -- Entity Vector Backfill

Embeds `__Entity__` nodes that are missing embeddings. Queries entities where `embedding IS NULL`, batch-embeds the entity `name`, and stores vectors. Safe for incremental runs.

```python
count = await rag.vector_store.backfill_entity_embeddings(batch_size=500)
```

### `embed_relationships()` -- RELATES Edge Embeddings

Batch-embeds all RELATES edges that have a `fact` property but are missing embeddings. These edge embeddings power the RELATES vector search path in retrieval.

```python
count = await rag.vector_store.embed_relationships(batch_size=500)
```

### `ensure_indices()` -- Index Creation

Creates all standard indexes (idempotent -- safe to call repeatedly):

| Index Type   | Label/Type     | Property               |
|-------------|----------------|------------------------|
| Vector      | `Chunk`        | `embedding`            |
| Vector      | `__Entity__`   | `embedding`            |
| Vector      | `RELATES` (edge) | `embedding`          |
| Fulltext    | `Chunk`        | `text`                 |
| Fulltext    | `__Entity__`   | `name`, `description`  |

```python
results = await rag.vector_store.ensure_indices()
```

Note: `ensure_indices()` is called automatically after each `ingest()` call. The `finalize()` method resets the internal `_indices_ensured` flag and re-runs it to catch any newly needed indexes.

### When to Call Each

| Scenario                                    | What to Call                          |
|---------------------------------------------|---------------------------------------|
| After ingesting all documents               | `await rag.finalize()`                |
| After incremental ingestion (new documents) | `await rag.finalize()`                |
| Only need dedup (embeddings already exist)  | `await rag.deduplicate_entities()`    |
| Only need entity embeddings                 | `await rag.vector_store.backfill_entity_embeddings()` |
| Only need RELATES edge embeddings           | `await rag.vector_store.embed_relationships()`        |
| Only need indexes                           | `await rag.vector_store.ensure_indices()`             |

Do **not** call `backfill_entity_embeddings()` inside an ingestion loop (i.e., after each document). It re-scans all entities and is slow when called repeatedly. Instead, ingest all documents first, then call `finalize()` once.
