# GraphRAG SDK v2.0 — Target Architecture

## Guiding Principles

| Principle | Definition | Enforcement |
|-----------|-----------|------------|
| **Strategy Modularity** | Every algorithmic concern is a swappable strategy behind an ABC | `*_strategies/` folders with one interface per domain |
| **Zero-Loss Data** | Every extracted triple traces back to its source chunk and document | Mandatory lexical graph in ingestion pipeline |
| **Production Latency** | Async-first, pooled connections, batched writes, latency budgets | `core/connection.py`, `core/context.py`, `storage/` batching |
| **Simplicity** | One entry point, flat structure, no meta-programming | `api/main.py` Facade, 1-2 level deep folders |
| **Credibility** | Graph faithfully represents source material | Schema-guided extraction + pruning + provenance chain |
| **Accuracy** | Multi-hop reasoning across the knowledge graph | Dedicated `multi_hop.py` retrieval strategy |
| **Adaptability** | Core is optimization-ready, strategies are swappable | Strategy + Repository patterns decouple logic from infrastructure |
| **Velocity** | Production-grade throughput | Async pipelines, connection pooling, batched DB operations |

---

## Code Structure

```
graphrag_sdk/
├── pyproject.toml
└── src/
    └── graphrag_sdk/
        ├── __init__.py                         # Public API exports + __version__
        │
        ├── core/                               # 🟢 FOUNDATION — Stable Contracts
        │   ├── __init__.py
        │   ├── models.py                       # Pydantic v2 data models (all SDK types)
        │   ├── providers.py                    # Embedder & LLM abstract interfaces
        │   ├── connection.py                   # Async FalkorDB client (pool + retries)
        │   ├── context.py                      # TenantID, TraceID, latency budgeting
        │   └── exceptions.py                   # Exception hierarchy
        │
        ├── ingestion/                          # 🟠 BUILDER — Knowledge Graph Construction
        │   ├── __init__.py
        │   ├── pipeline.py                     # Sequential orchestrator (Load→Chunk→Extract→Write)
        │   ├── loaders/                        # Data source adapters
        │   │   ├── __init__.py
        │   │   ├── base.py                     # LoaderStrategy ABC
        │   │   ├── pdf_loader.py               # PDF → text
        │   │   ├── text_loader.py              # Plain text / markdown
        │   │   └── ...                         # (S3, Slack, Notion — future)
        │   ├── chunking_strategies/            # Text splitting
        │   │   ├── __init__.py
        │   │   ├── base.py                     # ChunkingStrategy ABC
        │   │   ├── fixed_size.py               # Fixed window + overlap
        │   │   └── ...                         # (Semantic, Markdown, JSON — future)
        │   ├── extraction_strategies/          # Entity & relationship extraction
        │   │   ├── __init__.py
        │   │   ├── base.py                     # ExtractionStrategy ABC
        │   │   ├── schema_guided.py            # Schema-constrained LLM extraction
        │   │   └── ...                         # (Open-IE, HippoRAG-IE — future)
        │   └── resolution_strategies/          # Entity deduplication
        │       ├── __init__.py
        │       ├── base.py                     # ResolutionStrategy ABC
        │       ├── exact_match.py              # Property-based exact match
        │       └── ...                         # (Vector-Fuzzy, LLM-Oracle — future)
        │
        ├── retrieval/                          # 🟣 BRAIN — Intelligent Search
        │   ├── __init__.py
        │   ├── router.py                       # Semantic intent router (optional)
        │   ├── strategies/                     # Retrieval methods
        │   │   ├── __init__.py
        │   │   ├── base.py                     # RetrievalStrategy ABC (Template Method)
        │   │   ├── local.py                    # Vector + 1-hop traversal
        │   │   ├── global_.py                  # Community summaries (LightRAG-style)
        │   │   ├── multi_hop.py                # Recursive path traversal
        │   │   ├── cypher_gen.py               # Natural language → Cypher
        │   │   └── ...                         # (Custom user strategies — future)
        │   └── reranking_strategies/           # Result quality layer
        │       ├── __init__.py
        │       ├── base.py                     # RerankingStrategy ABC
        │       └── ...                         # (Cross-Encoder, RRF, MMR — future)
        │
        ├── storage/                            # 🔵 VAULT — Data Access Layer
        │   ├── __init__.py
        │   ├── graph_store.py                  # Cypher query builder + batched upserts
        │   └── vector_store.py                 # Vector index management + search
        │
        ├── api/                                # ⚪ INTERFACE — User Entry Point
        │   ├── __init__.py
        │   └── main.py                         # GraphRAG Facade class
        │
        ├── utils/                              # 🛠️ TOOLS — Internal Helpers
        │   ├── __init__.py
        │   └── graph_viz.py                    # Graph visualization & debugging
        │
        └── telemetry/                          # 🟡 VISIBILITY — Enterprise Observability
            ├── __init__.py
            └── tracer.py                       # OpenTelemetry spans & performance tracking
```

---

## Design Patterns

### Primary Patterns (Carry 90% of the SDK)

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Strategy** | Every `*_strategies/` folder | Swappable algorithms behind a single ABC |
| **Template Method** | `retrieval/strategies/base.py`, `ingestion/pipeline.py` | Skeleton with mandatory telemetry/validation; subclasses implement core logic only |
| **Pipeline (Sequential)** | `ingestion/pipeline.py` | Domain-specific linear orchestrator: Load → Chunk → Lexical Graph → Extract → Prune → Resolve → Write |

### Supporting Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Facade** | `api/main.py` | `GraphRAG` class hides all wiring — single entry point for users |
| **Repository** | `storage/graph_store.py`, `storage/vector_store.py` | Abstracts all DB operations; strategies never write raw Cypher |
| **Context Object** | `core/context.py` | Threaded through every call: tenant_id, trace_id, latency_budget, tracer |

### Explicitly Avoided

| Pattern | Why Not |
|---------|---------|
| Metaclass / `__init_subclass__` | Magic invisible in code review; explicit ABCs are clearer |
| Generic DAG / Orchestrator | Over-engineered for a linear pipeline; add only if users prove need |
| Observer / EventEmitter | OpenTelemetry replaces custom event systems |
| Factory from Config / `import_class()` | Hard to debug; users construct in Python code, not JSON |
| Decorator chains | Invisible call stacks; Template Method base handles cross-cutting concerns |

---

## Origin Map — What Comes From Where

### From Your Design (Domain Architecture)

| Element | Details |
|---------|---------|
| **Domain-oriented folder structure** | `ingestion/`, `retrieval/`, `storage/`, `api/`, `telemetry/` — flat, self-describing |
| **`core/context.py`** | TenantID, TraceID, latency budgeting — production multi-tenancy (not in Neo4j) |
| **`core/connection.py`** | Async FalkorDB client with pooling + retries (Neo4j uses raw driver) |
| **Semantic Router** | `retrieval/router.py` — classifies query intent, picks strategy dynamically (Neo4j forces one retriever at init) |
| **Reranking as a separate layer** | `reranking_strategies/` — composable result quality (absent in Neo4j) |
| **`telemetry/` as first-class module** | OpenTelemetry integration at the top level (Neo4j has custom EventNotifier) |
| **`storage/` separation** | Clean data access layer (Neo4j embeds queries in components) |
| **`utils/graph_viz.py`** | Graph visualization and debugging tool |
| **Strategy folders per domain** | `chunking_strategies/`, `extraction_strategies/`, `resolution_strategies/`, `reranking_strategies/` |

### From Neo4j (Proven Engineering)

| Element | Neo4j Origin | Where It Lives |
|---------|-------------|---------------|
| **`DataModel` base class** | `DataModel(BaseModel)` — all pipeline data extends Pydantic | `core/models.py` — every strategy input/output is a model |
| **Lexical graph (provenance chain)** | `LexicalGraphBuilder` — Document → Chunk → Entity traceability | Built-in mandatory step in `ingestion/pipeline.py` |
| **Schema-guided extraction** | `SchemaBuilder` + `GraphPruning` — constrain LLM output to defined types | `extraction_strategies/schema_guided.py` + post-extraction pruning step |
| **Context passing** | `RunContext(run_id, task_name, notifier)` threaded through components | `core/context.py` — simplified to always-present (no dual `run()`/`run_with_context()`) |
| **Batched upserts** | `FalkorDBWriter` — batch MERGE for nodes and relationships | `storage/graph_store.py` — `upsert_nodes(nodes)`, `upsert_relationships(rels)` |
| **Provider ABCs** | `Embedder(ABC)`, `LLMInterface(ABC)` — thin interfaces with async fallback | `core/providers.py` — same minimal surface, async default falls back to sync |
| **Template Method on retrieval** | `Retriever.search()` calls abstract `get_search_results()` | `retrieval/strategies/base.py` — base handles telemetry + validation, subclass implements `_execute()` |

### From My Suggestions (Gap Analysis)

| Element | Gap Identified | Resolution |
|---------|---------------|-----------|
| **`core/providers.py`** | LLM and Embedder are cross-cutting deps used by ingestion + retrieval; need a home | Provider ABCs live in `core/` as foundational contracts |
| **Provenance as non-optional** | Zero-Loss principle requires mandatory provenance, not a pluggable strategy | Lexical graph is a hardcoded step in `ingestion/pipeline.py`, never skippable |
| **Schema in `core/models.py`** | Schema definition needed by both extraction and retrieval layers | `EntityType`, `RelationType`, `SchemaPattern` defined in `core/models.py` |
| **Graph write as infrastructure** | After extraction, writing to FalkorDB isn't a "strategy" — there's one way to MERGE | Write logic in `storage/graph_store.py`, called by pipeline directly |
| **Router as optional** | Semantic router is ambitious for v1; users should be able to pick strategy explicitly | `router.py` exists but `GraphRAG.query()` also accepts explicit `strategy=` parameter |
| **Linear pipeline first** | DAG adds complexity without proportional value for KG building | Sequential pipeline in v1; branching (parallel embed + extract) deferred to v2 |
| **Template Method base for all strategies** | Ensures telemetry, validation, error handling happen once | Every `base.py` in strategy folders uses Template Method |

---

## Key Interfaces

### Provider Contracts (`core/providers.py`)

```python
class Embedder(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    async def aembed_query(self, text: str) -> list[float]:
        """Default: run sync in thread pool."""
        return await asyncio.to_thread(self.embed_query, text)

class LLMInterface(ABC):
    @abstractmethod
    def invoke(self, prompt: str) -> str: ...

    async def ainvoke(self, prompt: str) -> str:
        return await asyncio.to_thread(self.invoke, prompt)

    def invoke_with_model(
        self, prompt: str, response_model: type[T]
    ) -> T:
        """Structured output — LLM returns validated Pydantic model."""
        ...
```

### Strategy ABCs

```python
# ingestion/loaders/base.py
class LoaderStrategy(ABC):
    @abstractmethod
    async def load(self, source: str, ctx: Context) -> DocumentOutput: ...

# ingestion/chunking_strategies/base.py
class ChunkingStrategy(ABC):
    @abstractmethod
    async def chunk(self, text: str, ctx: Context) -> TextChunks: ...

# ingestion/extraction_strategies/base.py
class ExtractionStrategy(ABC):
    @abstractmethod
    async def extract(
        self, chunks: TextChunks, schema: GraphSchema, ctx: Context
    ) -> GraphData: ...

# ingestion/resolution_strategies/base.py
class ResolutionStrategy(ABC):
    @abstractmethod
    async def resolve(
        self, graph_data: GraphData, ctx: Context
    ) -> ResolutionResult: ...
```

### Retrieval Strategy (Template Method)

```python
# retrieval/strategies/base.py
class RetrievalStrategy(ABC):
    def __init__(self, graph_store: GraphStore, vector_store: VectorStore):
        self._graph = graph_store
        self._vector = vector_store

    async def search(self, query: str, ctx: Context) -> RetrieverResult:
        span = ctx.tracer.start_span(f"retrieval.{self.__class__.__name__}")
        try:
            self._validate(query)
            raw = await self._execute(query, ctx)
            formatted = self._format(raw)
            return formatted
        finally:
            span.end()

    @abstractmethod
    async def _execute(self, query: str, ctx: Context) -> RawSearchResult: ...

    def _validate(self, query: str) -> None:
        if not query or not query.strip():
            raise RetrieverError("Empty query")

    def _format(self, raw: RawSearchResult) -> RetrieverResult:
        """Override for custom formatting. Default passes through."""
        return RetrieverResult(items=raw.items, metadata=raw.metadata)
```

### Ingestion Pipeline (Sequential)

```python
# ingestion/pipeline.py
class IngestionPipeline:
    def __init__(
        self,
        loader: LoaderStrategy,
        chunker: ChunkingStrategy,
        extractor: ExtractionStrategy,
        resolver: ResolutionStrategy,
        graph_store: GraphStore,
        vector_store: VectorStore,
        schema: GraphSchema,
        ctx: Context,
    ): ...

    async def run(self, source: str) -> IngestionResult:
        # Step 1: Load
        document = await self.loader.load(source, self.ctx)

        # Step 2: Chunk
        chunks = await self.chunker.chunk(document.text, self.ctx)

        # Step 3: Build lexical graph (MANDATORY — not a strategy)
        await self._build_lexical_graph(document, chunks)

        # Step 4: Extract entities & relationships
        graph_data = await self.extractor.extract(chunks, self.schema, self.ctx)

        # Step 5: Prune against schema
        graph_data = self._prune(graph_data, self.schema)

        # Step 6: Resolve duplicate entities
        resolved = await self.resolver.resolve(graph_data, self.ctx)

        # Step 7: Write to graph (batched)
        await self.graph_store.upsert_nodes(resolved.nodes)
        await self.graph_store.upsert_relationships(resolved.relationships)

        # Step 8: Embed & index chunks
        await self.vector_store.index_chunks(chunks)

        return IngestionResult(...)

    async def _build_lexical_graph(self, doc, chunks):
        """Non-optional provenance chain: Document → PART_OF → Chunk → NEXT_CHUNK → Chunk"""
        ...
```

### Facade (`api/main.py`)

```python
# api/main.py
class GraphRAG:
    def __init__(
        self,
        driver: FalkorDBConnection,
        llm: LLMInterface,
        embedder: Embedder,
        schema: GraphSchema | None = None,
        retrieval_strategy: RetrievalStrategy | None = None,
    ): ...

    async def ingest(
        self,
        source: str,
        *,
        loader: LoaderStrategy | None = None,
        chunker: ChunkingStrategy | None = None,
        extractor: ExtractionStrategy | None = None,
        resolver: ResolutionStrategy | None = None,
    ) -> IngestionResult:
        """Build knowledge graph from source. Uses sensible defaults for any unspecified strategy."""
        ...

    async def query(
        self,
        question: str,
        *,
        strategy: RetrievalStrategy | None = None,
        reranker: RerankingStrategy | None = None,
    ) -> RagResult:
        """Query the knowledge graph. Uses configured default strategy or explicit override."""
        ...
```

---

## Data Flow

```
                        INGESTION
                        ─────────
Source (PDF/text/URL)
    │
    ▼
┌──────────┐     ┌──────────────┐     ┌─────────────────┐
│  Loader  │────▶│   Chunker    │────▶│  Lexical Graph   │ ◄── MANDATORY
│ Strategy │     │  Strategy    │     │  (provenance)    │     (not a strategy)
└──────────┘     └──────────────┘     └─────────────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │   Extractor      │
                                     │   Strategy       │
                                     │ (schema-guided)  │
                                     └─────────────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐     ┌──────────────┐
                                     │    Pruner        │────▶│   Resolver   │
                                     │ (schema filter)  │     │   Strategy   │
                                     └─────────────────┘     └──────────────┘
                                                                     │
                                              ┌──────────────────────┤
                                              ▼                      ▼
                                     ┌─────────────────┐     ┌──────────────┐
                                     │  Graph Store     │     │ Vector Store  │
                                     │ (batched MERGE)  │     │ (embed+index) │
                                     └─────────────────┘     └──────────────┘


                        RETRIEVAL
                        ─────────
User Query
    │
    ▼
┌──────────┐     ┌─────────────────┐     ┌──────────────┐
│  Router  │────▶│   Retrieval     │────▶│   Reranker   │
│(optional)│     │   Strategy      │     │   Strategy   │
└──────────┘     │ (Template Mthd) │     └──────────────┘
                 └─────────────────┘              │
                    │          │                   ▼
                    ▼          ▼              ┌──────────┐
              Graph Store  Vector Store      │   LLM    │
                                             │ Generate │
                                             └──────────┘
                                                  │
                                                  ▼
                                             RagResult
```

---

## v1 Scope vs Future

### v1 — Ship This

| Module | What's Implemented |
|--------|--------------------|
| `core/models.py` | All Pydantic data models, schema types |
| `core/providers.py` | `Embedder` and `LLMInterface` ABCs with async fallback |
| `core/connection.py` | FalkorDB async client with pooling |
| `core/context.py` | Context object (tenant, trace, latency budget) |
| `core/exceptions.py` | Full exception hierarchy |
| `ingestion/pipeline.py` | Sequential orchestrator with mandatory lexical graph |
| `ingestion/loaders/` | `PdfLoader`, `TextLoader` |
| `ingestion/chunking_strategies/` | `FixedSizeChunking` |
| `ingestion/extraction_strategies/` | `SchemaGuidedExtraction` |
| `ingestion/resolution_strategies/` | `ExactMatchResolution` |
| `retrieval/strategies/base.py` | Template Method base class |
| `retrieval/strategies/local.py` | Vector + 1-hop traversal |
| `storage/graph_store.py` | Batched upserts, Cypher builder |
| `storage/vector_store.py` | Vector index CRUD + search |
| `api/main.py` | `GraphRAG` facade |
| `telemetry/tracer.py` | OpenTelemetry span integration |

### v2+ — Future Strategies (Drop-In)

| Module | What's Added |
|--------|-------------|
| `ingestion/loaders/` | S3, Slack, Notion, Unstructured |
| `ingestion/chunking_strategies/` | Semantic, Markdown-aware, JSON |
| `ingestion/extraction_strategies/` | Open-IE, HippoRAG-IE |
| `ingestion/resolution_strategies/` | Vector-Fuzzy, LLM-Oracle |
| `retrieval/strategies/` | `global_.py`, `multi_hop.py`, `cypher_gen.py` |
| `retrieval/reranking_strategies/` | Cross-Encoder, RRF, MMR, LLM-Rank |
| `retrieval/router.py` | Semantic intent classification |
| Pipeline branching | Parallel embed + extract (if users need it) |
| Config-from-file | JSON/YAML pipeline construction (if users demand it) |
