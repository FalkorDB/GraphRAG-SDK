# GraphRAG SDK v2.0 — Architecture Review & End-to-End Suggestions

> **Reviewer**: Claude (AI Architecture Review)
> **Date**: February 2026
> **Scope**: Full review of Target Architecture, Infrastructure Blueprint, and current codebase

---

## 1. Executive Summary

The GraphRAG SDK v2.0 architecture is well-designed and makes strong trade-offs for a v1 release. The decision to simplify away from the Neo4j blueprint's complexity (DAG engine, metaclasses, config system) is correct and results in a cleaner, more maintainable SDK. This document covers what's working well, what needs attention before shipping, and what to consider for v2+.

---

## 2. What's Working Well

### 2.1 Deliberate Simplification Over Neo4j

The "Explicitly Avoided" patterns section is the most valuable architectural decision in the project. Specifically:

| Rejected Pattern | Why It's Right |
|---|---|
| Metaclass introspection (`ComponentMeta`) | Errors surface as metaclass failures during import, not clear runtime messages. Explicit ABCs with `@abstractmethod` give the same enforcement with standard Python errors. |
| Generic DAG pipeline engine | KG construction is always linear (Load → Chunk → Extract → Write). A DAG runner with topological sorting and cycle detection is solving a problem that doesn't exist. |
| Factory-from-config (`ObjectConfig`, `import_class()`) | Dynamic class loading from YAML is hostile to debugging and IDE support. Python construction is clearer. |
| Observer / EventEmitter | OpenTelemetry is the industry standard. A custom event system adds maintenance burden with no interoperability benefit. |
| Decorator chains | Template Method base classes achieve the same cross-cutting concerns (telemetry, validation) with visible call stacks. |

### 2.2 Mandatory Lexical Graph

In Neo4j's SDK, `LexicalGraphBuilder` is a pluggable `Component` — it can be accidentally omitted or mis-wired in the DAG. By hardcoding provenance as Step 3 in the sequential pipeline, the Zero-Loss Data principle becomes **structurally enforced**, not just documented. This is the right call.

### 2.3 Production-Oriented Context

Neo4j's `RunContext` is pipeline-scoped: `run_id` + `task_name` + `notifier`. The SDK's `Context` carries `tenant_id`, `trace_id`, and `latency_budget` — multi-tenant SaaS infrastructure baked into the foundation rather than bolted on later.

### 2.4 Clean Ingestion/Retrieval Separation

Ingestion and retrieval share `core/` contracts and `storage/` but are otherwise independent. New retrieval strategies (multi-hop, cypher gen, global) can be added without touching ingestion code.

### 2.5 Facade Pattern Delivers

`GraphRAG` achieves the "two-line usage" goal:

```python
rag = GraphRAG(connection=..., llm=..., embedder=...)
result = await rag.query("What is X?")
```

While still exposing full strategy customization via keyword arguments. The auto-detection of loader from file extension is a nice UX touch.

---

## 3. Issues to Address Before Shipping

### 3.1 `VectorStore` Should Be an ABC

**Problem**: The Target Architecture uses the Strategy pattern consistently for every algorithmic concern — except `VectorStore`. It's a concrete class tightly coupled to FalkorDB's native vector indexing. This is inconsistent with the rest of the design and blocks users who need external vector databases (Qdrant, Pinecone, Weaviate) alongside FalkorDB's graph.

**The Infrastructure Blueprint explicitly identified this** — it has `ExternalRetriever` for external vector stores. The current architecture has no clean extension point for this.

**Suggestion**: Make `VectorStore` an ABC with `FalkorDBVectorStore` as the default implementation, mirroring the strategy pattern used everywhere else.

```python
# storage/base.py (new file)
from abc import ABC, abstractmethod

class VectorStoreBase(ABC):
    @abstractmethod
    async def index_chunks(self, chunks: TextChunks) -> None: ...

    @abstractmethod
    async def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]: ...

class GraphStoreBase(ABC):
    @abstractmethod
    async def upsert_nodes(self, nodes: list[GraphNode]) -> None: ...

    @abstractmethod
    async def upsert_relationships(self, rels: list[GraphRelationship]) -> None: ...


# storage/graph_store.py
class GraphStore(GraphStoreBase):
    """FalkorDB implementation."""
    ...

# storage/vector_store.py
class VectorStore(VectorStoreBase):
    """FalkorDB native vector implementation."""
    ...
```

**Impact**: Low effort, high value. Enables external vector store integrations in v2+ without refactoring.

---

### 3.2 Fix `Any` Type Hints in Pipeline

**Problem**: In `ingestion/pipeline.py` lines 79-80:

```python
graph_store: Any,  # storage.GraphStore — import avoided for layering
vector_store: Any,  # storage.VectorStore
```

This defeats `mypy --strict` (which is enabled in `pyproject.toml`). The comment acknowledges the issue but doesn't solve it.

**Suggestion**: Define `Protocol` classes or use the ABCs from suggestion 3.1 in `core/` so the pipeline can type-check without importing concrete storage classes.

```python
# core/protocols.py (new file) — OR use the ABCs from 3.1
from typing import Protocol

class GraphStoreProtocol(Protocol):
    async def upsert_nodes(self, nodes: list[GraphNode]) -> None: ...
    async def upsert_relationships(self, rels: list[GraphRelationship]) -> None: ...

class VectorStoreProtocol(Protocol):
    async def index_chunks(self, chunks: TextChunks) -> None: ...
    async def search(self, query_vector: list[float], top_k: int) -> list[dict]: ...
```

Then in `pipeline.py`:

```python
def __init__(
    self,
    ...
    graph_store: GraphStoreProtocol,
    vector_store: VectorStoreProtocol,
    ...
)
```

**Impact**: Restores full mypy coverage. No runtime behavior change.

---

### 3.3 Define Router vs Explicit Strategy Precedence

**Problem**: `retrieval/router.py` exists for semantic intent routing, and `GraphRAG.query()` accepts `strategy=` for explicit strategy override. The Target Architecture calls the router "optional" but never defines:

- When does the router activate vs when does the explicit strategy win?
- What happens if someone passes `strategy=MyStrategy` **and** there's a router configured?
- Is the router set at `__init__` time or per-query?

**Suggestion**: Define a clear precedence rule and document it:

```
1. Explicit `strategy=` parameter on query() → always wins (user override)
2. Router (if configured) → classifies intent, selects strategy
3. Default strategy (set at __init__) → fallback
```

Implementation in `GraphRAG.query()`:

```python
async def query(self, question: str, *, strategy=None, ...):
    if strategy is not None:
        # Explicit override — highest priority
        retrieval = strategy
    elif self._router is not None:
        # Router selects strategy based on query intent
        retrieval = await self._router.route(question, ctx)
    else:
        # Default strategy from __init__
        retrieval = self._retrieval_strategy
    ...
```

**Impact**: Prevents ambiguous behavior and user confusion. Important to nail down before the router is implemented.

---

### 3.4 Promote Pruning to a Strategy

**Problem**: Every algorithmic concern is a swappable strategy — except pruning. `_prune()` is a private method on `IngestionPipeline`. The argument that "there's only one way to prune" doesn't hold long-term:

- Prune by confidence scores from extraction
- Prune by graph connectivity (orphan removal)
- Prune by LLM re-evaluation (ask the LLM "does this triple make sense?")
- Prune by frequency threshold (only keep entities mentioned N+ times)

When a second pruning approach is needed, this will require refactoring the pipeline.

**Suggestion**: Create a `PruningStrategy` ABC, make `SchemaPruning` the default:

```python
# ingestion/pruning_strategies/base.py
class PruningStrategy(ABC):
    @abstractmethod
    async def prune(self, graph_data: GraphData, schema: GraphSchema, ctx: Context) -> GraphData: ...

# ingestion/pruning_strategies/schema_pruning.py
class SchemaPruning(PruningStrategy):
    """Filter graph data to only include schema-conforming nodes and relationships."""
    async def prune(self, graph_data, schema, ctx):
        # Current _prune() logic moved here
        ...
```

Pipeline becomes:

```python
pipeline = IngestionPipeline(
    ...
    pruner=pruner or SchemaPruning(),
    ...
)
```

**Impact**: Maintains consistency with the Strategy Modularity principle. Low effort since the logic already exists.

---

### 3.5 Expose Batch Size Controls

**Problem**: The architecture lists "batched writes" and "Production-grade throughput" as core principles, but batch size management is entirely hidden inside the storage layer. The pipeline calls:

```python
await self.graph_store.upsert_nodes(resolved.nodes)  # full list, no batching control
```

For large documents producing thousands of entities, users have no way to tune batch sizes, and the pipeline has no way to report progress within a batch.

**Suggestion**: Allow batch_size configuration at the storage layer and optionally at the pipeline level:

```python
# Storage layer — batch_size in constructor
class GraphStore(GraphStoreBase):
    def __init__(self, connection, batch_size: int = 500):
        self._batch_size = batch_size

    async def upsert_nodes(self, nodes: list[GraphNode]) -> None:
        for batch in _chunked(nodes, self._batch_size):
            await self._execute_batch_merge(batch)

# Pipeline level — pass-through option
pipeline = IngestionPipeline(..., batch_size=1000)

# Or via GraphRAG facade
rag = GraphRAG(..., graph_store_batch_size=500)
```

**Impact**: Needed for production workloads. Without this, large ingestions will either OOM or hit query size limits on FalkorDB.

---

### 3.6 Fix `asyncio.run()` in Sync Wrappers

**Problem**: `query_sync()` and `ingest_sync()` use `asyncio.run()`:

```python
def query_sync(self, question: str, **kwargs) -> RagResult:
    import asyncio
    return asyncio.run(self.query(question, **kwargs))
```

This will raise `RuntimeError: This event loop is already running` when called from:
- Jupyter notebooks (which run their own event loop)
- FastAPI/Starlette endpoints
- Any async framework context

**Suggestion**: Use a thread-pool approach:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

_SYNC_EXECUTOR = ThreadPoolExecutor(max_workers=1)

def _run_sync(coro):
    """Run an async coroutine synchronously, safe from nested event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running — safe to use asyncio.run()
        return asyncio.run(coro)
    else:
        # Already in an event loop — run in a separate thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()

def query_sync(self, question: str, **kwargs) -> RagResult:
    return _run_sync(self.query(question, **kwargs))
```

Or alternatively, document that `_sync` methods should not be used inside async contexts and recommend `nest_asyncio` for Jupyter.

**Impact**: Critical for developer experience. Jupyter is the #1 environment for RAG prototyping.

---

## 4. Suggestions for Robustness

### 4.1 Add Storage Layer Tests

**Problem**: There are tests for every module except `storage/`. The test suite covers models, context, providers, chunking, loaders, extraction, resolution, pipeline, retrieval, router, reranking, and tracer — but no `test_graph_store.py` or `test_vector_store.py`.

**Suggestion**: Add unit tests with mocked FalkorDB connections:

```
tests/
├── test_graph_store.py     # Upsert nodes/rels, batch behavior, error handling
├── test_vector_store.py    # Index chunks, search, embedding integration
```

At minimum, test:
- Batch splitting behavior (when implemented)
- Cypher query generation correctness
- Error handling for connection failures
- Empty input handling

---

### 4.2 Add Pipeline Error Recovery / Partial Results

**Problem**: If the pipeline fails at Step 6 (resolution), Steps 1-5 have already completed — chunks are in the graph, but entities aren't resolved. The current error handling wraps everything in a single try/except and raises `IngestionError`. There's no way to:
- Know which step failed
- Resume from a checkpoint
- Get partial results

**Suggestion**: Return partial results on failure:

```python
class IngestionResult(DataModel):
    ...
    completed_steps: list[str] = Field(default_factory=list)
    failed_step: Optional[str] = None
    error: Optional[str] = None
    partial: bool = False
```

Pipeline catches per-step and records progress:

```python
try:
    # Step 4: Extract
    ctx.log("Step 4/8: Extracting")
    graph_data = await self.extractor.extract(chunks, self.schema, ctx)
    completed_steps.append("extract")
except Exception as exc:
    return IngestionResult(
        ...,
        completed_steps=completed_steps,
        failed_step="extract",
        error=str(exc),
        partial=True,
    )
```

**Impact**: Essential for debugging production pipelines. Users need to know *where* things broke.

---

### 4.3 Add Concurrency Limits for LLM Calls in Extraction

**Problem**: `SchemaGuidedExtraction` presumably calls the LLM for each chunk. For a document with 200 chunks, this means 200 concurrent LLM API calls if done with `asyncio.gather()`. This will hit rate limits on every LLM provider.

**Suggestion**: Add a semaphore-based concurrency limiter:

```python
class SchemaGuidedExtraction(ExtractionStrategy):
    def __init__(self, llm: LLMInterface, max_concurrency: int = 5):
        self.llm = llm
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def _extract_chunk(self, chunk, schema, ctx):
        async with self._semaphore:
            return await self.llm.ainvoke(...)

    async def extract(self, chunks, schema, ctx):
        tasks = [self._extract_chunk(c, schema, ctx) for c in chunks.chunks]
        results = await asyncio.gather(*tasks)
        ...
```

**Impact**: Without this, any real-world ingestion will fail with rate limit errors from OpenAI/Anthropic/Cohere.

---

### 4.4 Schema Validation at Pipeline Init, Not Runtime

**Problem**: If a user passes an invalid schema (e.g., a `SchemaPattern` referencing a non-existent `EntityType`), the error only surfaces during extraction or pruning — deep in the pipeline run.

**Suggestion**: Validate schema consistency in `IngestionPipeline.__init__()`:

```python
def __init__(self, ..., schema: GraphSchema | None = None):
    ...
    self.schema = schema or GraphSchema()
    self._validate_schema(self.schema)

def _validate_schema(self, schema: GraphSchema) -> None:
    entity_labels = {e.label for e in schema.entities}
    relation_labels = {r.label for r in schema.relations}
    for pattern in schema.patterns:
        if pattern.source not in entity_labels:
            raise SchemaValidationError(
                f"Pattern source '{pattern.source}' not found in entity types"
            )
        if pattern.relationship not in relation_labels:
            raise SchemaValidationError(
                f"Pattern relationship '{pattern.relationship}' not found in relation types"
            )
        if pattern.target not in entity_labels:
            raise SchemaValidationError(
                f"Pattern target '{pattern.target}' not found in entity types"
            )
```

**Impact**: Fail-fast at construction time, not deep in a pipeline run.

---

## 5. Infrastructure Blueprint — What to Keep, What to Archive

### 5.1 Keep as Reference

| Section | Value |
|---|---|
| Component Catalog (5.1) | Checklist of capabilities a Graph RAG SDK needs |
| Data Models (Section 9) | Informed `core/models.py` — useful for comparing type coverage |
| Retriever Type Matrix (7.1) | Roadmap for v2+ retrieval strategies |
| Key Differences table (end) | Useful for positioning against Neo4j |

### 5.2 Archive / Do Not Implement

| Section | Why Not |
|---|---|
| Pipeline Engine (Section 4) | DAG orchestrator, metaclass introspection, result stores — over-engineered for linear pipeline |
| Config System (Section 6) | `AbstractConfig` → `AbstractPipelineConfig` → `TemplatePipelineConfig` hierarchy, `ObjectConfig` with dynamic class loading, `ParamResolver` — framework-within-a-framework |
| Event Notification (4.7) | Replaced by OpenTelemetry in Target Architecture |
| `experimental/` directory structure | The flat domain-oriented structure (`ingestion/`, `retrieval/`, `storage/`) is superior |

### 5.3 Consider for v2+

| Feature | When to Add |
|---|---|
| External vector store adapters | When users request Qdrant/Pinecone/Weaviate alongside FalkorDB graph |
| Config-from-file pipeline construction | Only if non-developer users (ML Ops) need to configure pipelines via YAML |
| Pipeline streaming / progress events | When users need real-time progress for long-running ingestions (OpenTelemetry spans may be sufficient) |
| `LLMInterfaceV2` (message-based) | When structured output / tool use becomes a core extraction pattern |

---

## 6. v2+ Retrieval Strategy Roadmap

The Target Architecture lists these as future strategies. Here's a suggested priority order:

| Priority | Strategy | Value | Complexity |
|---|---|---|---|
| 1 | `multi_hop.py` — Recursive path traversal | High — differentiator for complex queries | Medium |
| 2 | `cypher_gen.py` — NL → Cypher | High — enables precise graph queries | Medium-High |
| 3 | `global_.py` — Community summaries | Medium — good for "summarize everything about X" | High (needs community detection) |
| 4 | Cross-Encoder reranking | Medium — improves result quality | Low |
| 5 | RRF / MMR reranking | Medium — enables hybrid strategy fusion | Low |
| 6 | Semantic router implementation | Low-Medium — convenience, not capability | Medium |

**Rationale**: Multi-hop is the core value proposition of a *graph* RAG system over vanilla RAG. Cypher generation lets power users leverage the graph directly. These two together justify the "Graph" in GraphRAG. Community summaries (LightRAG-style) are valuable but require upstream community detection which is a significant addition.

---

## 7. Summary of Recommendations

### Must-Do (Before v1 Ship)

| # | Suggestion | Effort | Impact |
|---|---|---|---|
| 3.1 | Make `VectorStore` an ABC | Low | Enables extensibility |
| 3.2 | Fix `Any` type hints with Protocols | Low | Restores mypy coverage |
| 3.3 | Define router vs strategy precedence | Low | Prevents ambiguous behavior |
| 3.6 | Fix `asyncio.run()` in sync wrappers | Low | Critical for Jupyter users |
| 4.1 | Add storage layer tests | Medium | Test coverage gap |

### Should-Do (Before Production Use)

| # | Suggestion | Effort | Impact |
|---|---|---|---|
| 3.4 | Promote pruning to a strategy | Low | Consistency |
| 3.5 | Expose batch size controls | Medium | Production throughput |
| 4.2 | Pipeline partial results on failure | Medium | Debuggability |
| 4.3 | LLM concurrency limits in extraction | Low | Prevents rate limiting |
| 4.4 | Schema validation at init time | Low | Fail-fast |

### Consider (v2+)

| # | Suggestion | Effort | Impact |
|---|---|---|---|
| 5.3 | External vector store adapters | Medium | User flexibility |
| 6 | Multi-hop + Cypher gen retrieval | High | Core differentiator |
