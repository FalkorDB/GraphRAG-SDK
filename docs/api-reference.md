# API Reference

Complete reference for all public classes and methods exported by `graphrag_sdk`.

## Table of Contents

- [GraphRAG (Facade)](#graphrag-facade)
- [Connection](#connection)
- [Providers](#providers)
- [Data Models](#data-models)
- [Schema](#schema)
- [Ingestion Strategies](#ingestion-strategies)
- [Ingestion Pipeline](#ingestion-pipeline)
- [Retrieval Strategies](#retrieval-strategies)
- [Reranking Strategies](#reranking-strategies)
- [Storage](#storage)
- [Context](#context)
- [Exceptions](#exceptions)

---

## GraphRAG (Facade)

The main entry point. Two primary operations: `ingest()` and `query()`.

```python
from graphrag_sdk import GraphRAG
```

### Constructor

```python
GraphRAG(
    connection: FalkorDBConnection | ConnectionConfig,
    llm: LLMInterface,
    embedder: Embedder,
    schema: GraphSchema | None = None,
    retrieval_strategy: RetrievalStrategy | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `connection` | `FalkorDBConnection \| ConnectionConfig` | required | Database connection or config to create one |
| `llm` | `LLMInterface` | required | LLM provider |
| `embedder` | `Embedder` | required | Embedding provider |
| `schema` | `GraphSchema \| None` | `None` | Schema constraints for extraction (empty = unconstrained) |
| `retrieval_strategy` | `RetrievalStrategy \| None` | `None` | Default retrieval strategy (uses `MultiPathRetrieval` if None) |

**Public attributes:** `llm`, `embedder`, `schema`, `graph_store`, `vector_store`

### ingest()

```python
async def ingest(
    source: str,
    *,
    text: str | None = None,
    loader: LoaderStrategy | None = None,
    chunker: ChunkingStrategy | None = None,
    extractor: ExtractionStrategy | None = None,
    resolver: ResolutionStrategy | None = None,
    ctx: Context | None = None,
) -> IngestionResult
```

Build a knowledge graph from a source. Auto-detects loader from file extension.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | required | File path, URL, or source identifier |
| `text` | `str \| None` | `None` | Raw text (skips loader if provided) |
| `loader` | `LoaderStrategy \| None` | `None` | Custom loader (auto-detect if None) |
| `chunker` | `ChunkingStrategy \| None` | `None` | Custom chunker (FixedSizeChunking(1000) if None) |
| `extractor` | `ExtractionStrategy \| None` | `None` | Custom extractor (SchemaGuidedExtraction if None) |
| `resolver` | `ResolutionStrategy \| None` | `None` | Custom resolver (ExactMatchResolution if None) |
| `ctx` | `Context \| None` | `None` | Execution context |

**Returns:** `IngestionResult`

### query()

```python
async def query(
    question: str,
    *,
    strategy: RetrievalStrategy | None = None,
    reranker: RerankingStrategy | None = None,
    prompt_template: str | None = None,
    return_context: bool = False,
    ctx: Context | None = None,
) -> RagResult
```

Query the knowledge graph and generate an answer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | `str` | required | The user's question |
| `strategy` | `RetrievalStrategy \| None` | `None` | Override retrieval strategy |
| `reranker` | `RerankingStrategy \| None` | `None` | Optional reranking after retrieval |
| `prompt_template` | `str \| None` | `None` | Custom prompt (must contain `{context}` and `{question}`) |
| `return_context` | `bool` | `False` | Include retriever results in output |
| `ctx` | `Context \| None` | `None` | Execution context |

**Returns:** `RagResult`

### detect_synonymy()

```python
async def detect_synonymy(
    *,
    similarity_threshold: float = 0.9,
    batch_size: int = 500,
) -> int
```

Post-ingestion synonym detection. Embeds all entity names, computes pairwise cosine similarity, creates SYNONYM edges. Call once after all documents are ingested.

**Returns:** Number of SYNONYM edges created.

### Sync Wrappers

```python
def query_sync(question: str, **kwargs) -> RagResult
def ingest_sync(source: str, **kwargs) -> IngestionResult
```

Convenience methods that run the async versions in `asyncio.run()`.

---

## Connection

```python
from graphrag_sdk import ConnectionConfig, FalkorDBConnection
```

### ConnectionConfig

```python
@dataclass
class ConnectionConfig:
    host: str = "localhost"
    port: int = 6379
    username: str | None = None
    password: str | None = None
    graph_name: str = "knowledge_graph"
    max_connections: int = 16
    retry_count: int = 3
    retry_delay: float = 1.0
```

### FalkorDBConnection

```python
FalkorDBConnection(config: ConnectionConfig | None = None)
```

| Method | Description |
|--------|-------------|
| `await conn.query(cypher, params=None, timeout=None)` | Execute a Cypher query |
| `await conn.close()` | Close the connection pool |
| `conn.graph` | Lazy property returning the AsyncGraph handle |

---

## Providers

```python
from graphrag_sdk import LLMInterface, Embedder, LLMBatchItem
from graphrag_sdk import LiteLLM, LiteLLMEmbedder, OpenRouterLLM, OpenRouterEmbedder
```

### LLMInterface (ABC)

```python
LLMInterface(model_name: str, model_params: dict | None = None, max_concurrency: int = 12)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `invoke` | `(prompt: str, **kwargs) -> LLMResponse` | Sync text generation (abstract) |
| `ainvoke` | `(prompt: str, *, max_retries=3, **kwargs) -> LLMResponse` | Async with retry + backoff |
| `invoke_with_model` | `(prompt: str, response_model: Type[BaseModel], **kwargs) -> BaseModel` | Structured output |
| `ainvoke_with_model` | `(prompt: str, response_model: Type[BaseModel], *, max_retries=3) -> BaseModel` | Async structured output |
| `abatch_invoke` | `(prompts: list[str], *, max_concurrency=None, max_retries=3) -> list[LLMBatchItem]` | Concurrent batch |

### Embedder (ABC)

| Method | Signature | Description |
|--------|-----------|-------------|
| `embed_query` | `(text: str, **kwargs) -> list[float]` | Single text embedding (abstract) |
| `aembed_query` | `(text: str, **kwargs) -> list[float]` | Async single (default: thread pool) |
| `embed_documents` | `(texts: list[str], **kwargs) -> list[list[float]]` | Batch (default: sequential) |
| `aembed_documents` | `(texts: list[str], **kwargs) -> list[list[float]]` | Async batch (default: thread pool) |

### LLMBatchItem

```python
@dataclass
class LLMBatchItem:
    index: int
    response: LLMResponse | None = None
    error: Exception | None = None

    @property
    def ok(self) -> bool  # True if response is not None
```

### LiteLLM

```python
LiteLLM(model: str, *, api_key=None, api_base=None, api_version=None, temperature=0.0, max_tokens=None, **kwargs)
```

### LiteLLMEmbedder

```python
LiteLLMEmbedder(model: str, *, api_key=None, api_base=None, api_version=None, **kwargs)
```

### OpenRouterLLM

```python
OpenRouterLLM(model: str, *, api_key=None, temperature=0.0, max_tokens=None, extra_headers=None)
```

### OpenRouterEmbedder

```python
OpenRouterEmbedder(model: str, *, api_key=None, extra_headers=None)
```

---

## Data Models

All models extend `DataModel` (Pydantic `BaseModel` with `extra="allow"`).

```python
from graphrag_sdk import GraphNode, GraphRelationship, GraphData
from graphrag_sdk import TextChunk, TextChunks
from graphrag_sdk import DocumentInfo, DocumentOutput
from graphrag_sdk import IngestionResult, RagResult
from graphrag_sdk import RetrieverResult, RetrieverResultItem
from graphrag_sdk import ResolutionResult, SearchType
```

### GraphNode

```python
class GraphNode(DataModel):
    id: str                                              # Unique identifier
    label: str                                           # Node label (Person, Place, etc.)
    properties: dict[str, Any] = {}                      # Key-value properties
    embedding_properties: dict[str, list[float]] | None = None
```

### GraphRelationship

```python
class GraphRelationship(DataModel):
    start_node_id: str
    end_node_id: str
    type: str                                            # Relationship type (WORKS_AT, etc.)
    properties: dict[str, Any] = {}
    embedding_properties: dict[str, list[float]] | None = None
```

### GraphData

```python
class GraphData(DataModel):
    nodes: list[GraphNode] = []
    relationships: list[GraphRelationship] = []
```

### TextChunk

```python
class TextChunk(DataModel):
    text: str
    index: int
    metadata: dict[str, Any] = {}
    uid: str                                             # Auto-generated UUID
```

### TextChunks

```python
class TextChunks(DataModel):
    chunks: list[TextChunk] = []
```

### DocumentInfo

```python
class DocumentInfo(DataModel):
    path: str | None = None
    uid: str                                             # Auto-generated UUID
    metadata: dict[str, Any] = {}
```

### DocumentOutput

```python
class DocumentOutput(DataModel):
    text: str
    document_info: DocumentInfo = DocumentInfo()
```

### IngestionResult

```python
class IngestionResult(DataModel):
    document_info: DocumentInfo = DocumentInfo()
    nodes_created: int = 0
    relationships_created: int = 0
    chunks_indexed: int = 0
    metadata: dict[str, Any] = {}
```

### RagResult

```python
class RagResult(DataModel):
    answer: str
    retriever_result: RetrieverResult | None = None      # Populated when return_context=True
    metadata: dict[str, Any] = {}                        # Contains model, num_context_items, strategy
```

### RetrieverResult

```python
class RetrieverResult(DataModel):
    items: list[RetrieverResultItem] = []
    metadata: dict[str, Any] = {}
```

### RetrieverResultItem

```python
class RetrieverResultItem(DataModel):
    content: str
    metadata: dict[str, Any] = {}
    score: float | None = None
```

### ResolutionResult

```python
class ResolutionResult(DataModel):
    nodes: list[GraphNode] = []
    relationships: list[GraphRelationship] = []
    merged_count: int = 0
```

### LLMResponse

```python
class LLMResponse(DataModel):
    content: str
    tool_calls: list[dict[str, Any]] | None = None
```

### SearchType

```python
class SearchType(str, Enum):
    VECTOR = "vector"
    FULLTEXT = "fulltext"
    HYBRID = "hybrid"
```

---

## Schema

```python
from graphrag_sdk import GraphSchema, EntityType, RelationType, SchemaPattern
```

### EntityType

```python
class EntityType(DataModel):
    label: str                            # e.g. "Person"
    description: str | None = None        # Helps LLM understand what to extract
    properties: list[PropertyType] = []   # Optional property definitions
```

### RelationType

```python
class RelationType(DataModel):
    label: str                            # e.g. "WORKS_AT"
    description: str | None = None
    properties: list[PropertyType] = []
```

### PropertyType

```python
class PropertyType(DataModel):
    name: str
    type: str = "STRING"                  # STRING, INTEGER, FLOAT, BOOLEAN, DATE, LIST
    description: str | None = None
    required: bool = False
```

### SchemaPattern

```python
class SchemaPattern(DataModel):
    source: str                           # Source entity label
    relationship: str                     # Relationship type
    target: str                           # Target entity label
```

### GraphSchema

```python
class GraphSchema(DataModel):
    entities: list[EntityType] = []
    relations: list[RelationType] = []
    patterns: list[SchemaPattern] = []    # Allowed source-relationship-target triples
```

---

## Ingestion Strategies

### LoaderStrategy (ABC)

```python
class LoaderStrategy(ABC):
    @abstractmethod
    async def load(self, source: str, ctx: Context) -> DocumentOutput: ...
```

**Built-in:** `TextLoader(encoding="utf-8")`, `PdfLoader()`

### ChunkingStrategy (ABC)

```python
class ChunkingStrategy(ABC):
    @abstractmethod
    async def chunk(self, text: str, ctx: Context) -> TextChunks: ...
```

**Built-in:** `FixedSizeChunking(chunk_size=1000, chunk_overlap=100)`

### ExtractionStrategy (ABC)

```python
class ExtractionStrategy(ABC):
    @abstractmethod
    async def extract(self, chunks: TextChunks, schema: GraphSchema, ctx: Context) -> GraphData: ...
```

**Built-in:**
- `SchemaGuidedExtraction(llm, chunk_batch_size=1)`
- `MergedExtraction(llm, embedder=None, enable_gleaning=False, max_concurrency=None)`

### ResolutionStrategy (ABC)

```python
class ResolutionStrategy(ABC):
    @abstractmethod
    async def resolve(self, graph_data: GraphData, ctx: Context) -> ResolutionResult: ...
```

**Built-in:**
- `ExactMatchResolution(resolve_property="id")`
- `DescriptionMergeResolution(llm=None, force_summary_threshold=3, max_summary_tokens=500)`

---

## Ingestion Pipeline

```python
from graphrag_sdk import IngestionPipeline
```

```python
IngestionPipeline(
    loader: LoaderStrategy,
    chunker: ChunkingStrategy,
    extractor: ExtractionStrategy,
    resolver: ResolutionStrategy,
    graph_store: GraphStore,
    vector_store: VectorStore,
    schema: GraphSchema | None = None,
    embedder: Embedder | None = None,
    skip_synonymy: bool = False,
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `(source, ctx=None, *, text=None, document_info=None) -> IngestionResult` | Execute the full pipeline |

---

## Retrieval Strategies

### RetrievalStrategy (ABC)

Uses the Template Method pattern.

```python
class RetrievalStrategy(ABC):
    def __init__(self, graph_store=None, vector_store=None): ...

    async def search(self, query, ctx=None, **kwargs) -> RetrieverResult: ...

    @abstractmethod
    async def _execute(self, query, ctx, **kwargs) -> RawSearchResult: ...
```

### LocalRetrieval

```python
LocalRetrieval(graph_store, vector_store, embedder, top_k=5, include_entities=True)
```

### MultiPathRetrieval

```python
MultiPathRetrieval(
    graph_store, vector_store, embedder, llm,
    *,
    entity_top_k=5,
    chunk_top_k=15,
    fact_top_k=15,
    max_entities=30,
    max_relationships=20,
    keyword_limit=10,
    llm_rerank_top_k=8,
    llm_rerank=True,
)
```

---

## Reranking Strategies

### RerankingStrategy (ABC)

```python
class RerankingStrategy(ABC):
    @abstractmethod
    async def rerank(self, query, result: RetrieverResult, ctx: Context) -> RetrieverResult: ...
```

### CosineReranker

```python
CosineReranker(embedder: Embedder, top_k: int = 15)
```

---

## Storage

```python
from graphrag_sdk import GraphStore, VectorStore
```

### GraphStore

```python
GraphStore(connection: FalkorDBConnection)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `upsert_nodes` | `(nodes: list[GraphNode]) -> int` | Batched MERGE, returns count |
| `upsert_relationships` | `(rels: list[GraphRelationship]) -> int` | Batched MERGE with label hints |
| `get_connected_entities` | `(chunk_id, max_hops=1) -> list[dict]` | N-hop entity traversal |
| `query_raw` | `(cypher, params=None) -> Any` | Raw Cypher execution |
| `get_statistics` | `() -> dict` | Node/edge counts, types, density |
| `delete_all` | `() -> None` | Delete all data |

### VectorStore

```python
VectorStore(connection, embedder=None, index_name="chunk_embeddings", embedding_dimension=1536, similarity_function="cosine")
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `create_vector_index` | `(label="Chunk", property="embedding") -> None` | Create vector index |
| `create_entity_vector_index` | `() -> None` | Create entity vector index |
| `create_fact_vector_index` | `() -> None` | Create fact vector index |
| `create_fulltext_index` | `(label="Chunk", *properties) -> None` | Create fulltext index |
| `ensure_indices` | `() -> None` | Create all standard indices |
| `index_chunks` | `(chunks: TextChunks) -> int` | Embed and store chunk vectors |
| `index_facts` | `(fact_strings, facts) -> int` | Embed and store fact vectors |
| `backfill_entity_embeddings` | `() -> int` | Embed all entities missing vectors |
| `search` | `(query_vector, top_k=5, label="Chunk") -> list[dict]` | Vector similarity search |
| `search_entities` | `(query_vector, top_k=5) -> list[dict]` | Entity vector search |
| `search_facts` | `(query_vector, top_k=5) -> list[dict]` | Fact vector search |
| `fulltext_search` | `(query, top_k=5, label="Chunk") -> list[dict]` | Fulltext keyword search |

---

## Context

```python
from graphrag_sdk import Context
```

Execution context for logging and budget tracking.

```python
Context(tenant_id: str = "default", latency_budget_ms: float = 60000.0)
```

| Method/Property | Description |
|----------------|-------------|
| `ctx.log(message, log_level=logging.INFO)` | Log a message |
| `ctx.budget_exceeded` | True if elapsed time > latency_budget_ms |

---

## Exceptions

```python
from graphrag_sdk import GraphRAGError
```

| Exception | When Raised |
|-----------|------------|
| `GraphRAGError` | Base exception for all SDK errors |
| `LoaderError` | File loading failures |
| `ChunkingError` | Text splitting failures |
| `ExtractionError` | LLM extraction or JSON parsing failures |
| `ResolutionError` | Entity deduplication failures |
| `RetrieverError` | Retrieval execution failures |
| `DatabaseError` | FalkorDB connection or query failures |
| `IngestionError` | Pipeline orchestration failures |
| `SchemaValidationError` | Schema constraint violations |
