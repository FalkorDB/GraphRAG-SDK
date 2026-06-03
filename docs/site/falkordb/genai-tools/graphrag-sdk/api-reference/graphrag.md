---
title: "GraphRAG"
nav_order: 1
parent: "API Reference"
grand_parent: "GraphRAG-SDK"
description: "The GraphRAG facade — every public method with signature, parameters, returns, and raises."
---

# `GraphRAG`

Module: `graphrag_sdk`  ·  Import: `from graphrag_sdk import GraphRAG`

The single user-facing class. One instance per FalkorDB connection. Async-only; sync wrappers are at the bottom of this page.

```python
class GraphRAG:
    def __init__(
        self,
        connection: FalkorDBConnection | ConnectionConfig,
        llm: LLMInterface,
        embedder: Embedder,
        ontology: Ontology | None = None,
        retrieval_strategy: RetrievalStrategy | None = None,
        embedding_dimension: int = 256,
        *,
        schema: Ontology | None = None,
    ) -> None
```

## Constructor

#### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `connection` | `FalkorDBConnection \| ConnectionConfig` | — required — | A live connection, or a `ConnectionConfig` the SDK uses to create one. |
| `llm` | `LLMInterface` | — required — | LLM provider used by extraction and generation. Typically `LiteLLM(model="...")`. |
| `embedder` | `Embedder` | — required — | Embedding provider used for vector indexes and retrieval. Typically `LiteLLMEmbedder(model="...")`. |
| `ontology` | `Ontology \| None` | `None` | Optional schema. When omitted, the persisted ontology is reused; if none exists, a built-in default is seeded on first ingest. |
| `retrieval_strategy` | `RetrievalStrategy \| None` | `None` (→ `MultiPathRetrieval`) | Default retrieval strategy. Per-call overrides are accepted on `retrieve` / `completion`. |
| `embedding_dimension` | `int` | `256` | Output dimension of the embedder. Must match the model's actual output — mismatch is caught at first ingest with `ConfigError`. |
| `schema` | `Ontology \| None` | `None` | **Deprecated.** Old name for `ontology=`. Emits `DeprecationWarning`. Passing both raises `TypeError`. |

#### Raises

- `TypeError` — both `ontology=` and `schema=` passed.

#### Lifecycle

`GraphRAG` is an async context manager but is **not reentrant** — one `async with` block per instance. Use `await rag.close()` for manual lifecycle.

```python
async with GraphRAG(connection=..., llm=..., embedder=...) as rag:
    ...
```

---

## Ingestion

### `ingest`

Build the knowledge graph from one or more sources.

```python
@overload
async def ingest(
    self,
    source: str | None = None,
    *,
    text: str | None = None,
    document_id: str | None = None,
    loader: LoaderStrategy | None = None,
    chunker: ChunkingStrategy | None = None,
    extractor: ExtractionStrategy | None = None,
    resolver: ResolutionStrategy | None = None,
    ctx: Context | None = None,
) -> IngestionResult: ...

@overload
async def ingest(
    self,
    source: list[str],
    *,
    loader: LoaderStrategy | None = None,
    chunker: ChunkingStrategy | None = None,
    extractor: ExtractionStrategy | None = None,
    resolver: ResolutionStrategy | None = None,
    max_concurrency: int = 3,
    ctx: Context | None = None,
) -> list[IngestionResult | Exception]: ...
```

Two input modes:

- **File mode** — pass `source` (single path or list). Loader reads from disk; `document_id` defaults to `os.path.normpath(source)`.
- **Text mode** — pass `text` directly. Provide `document_id` if you want a stable id (otherwise a `text-<hex>` id is generated).

#### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `source` | `str \| list[str] \| None` | `None` | File path or list of paths. Mutually exclusive with `text`. |
| `text` | `str \| None` | `None` | Raw text. Skips the loader. Mutually exclusive with `source`. |
| `document_id` | `str \| None` | derived from `source` | Stable id for incremental updates. Required-derived in batch file mode (per-file). |
| `loader` | `LoaderStrategy \| None` | per-extension auto | Override the loader. File mode only. |
| `chunker` | `ChunkingStrategy \| None` | `SentenceTokenCapChunking(max_tokens=512, overlap_sentences=2)` | Override the chunker. |
| `extractor` | `ExtractionStrategy \| None` | `GraphExtraction(llm=..., entity_types=ontology.labels)` | Override the extractor. |
| `resolver` | `ResolutionStrategy \| None` | `ExactMatchResolution()` | Override per-document resolution. |
| `max_concurrency` | `int` | `3` | Parallel ingestions across a list source. Single-source mode ignores this. |
| `ctx` | `Context \| None` | `Context()` | Execution context for logging / latency budget. |

#### Returns

- Single source → `IngestionResult`.
- List source → `list[IngestionResult | Exception]` aligned by index. One bad source does not abort the batch; failures are logged at WARNING.

#### Raises

- `ValueError` — neither `source` nor `text` provided, both provided, empty `document_id`, `document_id` on batch ingest.
- `ConfigError` — embedder dimension mismatch with the existing graph config.

---

### `update`

Re-sync a previously-ingested document.

```python
async def update(
    self,
    source: str | None = None,
    *,
    text: str | None = None,
    document_id: str | None = None,
    loader: LoaderStrategy | None = None,
    chunker: ChunkingStrategy | None = None,
    extractor: ExtractionStrategy | None = None,
    resolver: ResolutionStrategy | None = None,
    if_missing: Literal["error", "ingest"] = "error",
    ctx: Context | None = None,
) -> UpdateResult
```

A SHA-256 content hash short-circuits no-op updates — touch-only edits cost one Cypher query.

#### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `source` | `str \| None` | `None` | File path. Mutually exclusive with `text`. |
| `text` | `str \| None` | `None` | Raw text. `document_id` is **required** in text mode. |
| `document_id` | `str \| None` | from `source` | Stable Document id to update. |
| `loader` / `chunker` / `extractor` / `resolver` | per `ingest()` | per-call strategy overrides. |
| `if_missing` | `Literal["error", "ingest"]` | `"error"` | When the id is unknown: `"error"` raises `DocumentNotFoundError`; `"ingest"` falls through to a fresh ingest. |
| `ctx` | `Context \| None` | `Context()` | Execution context. |

#### Returns

`UpdateResult` — extends `IngestionResult` with `chunks_deleted`, `entities_deleted`, `no_op`, `replaced_existing`.

#### Raises

- `ValueError` — bad argument combination; empty `document_id`; text mode without `document_id`.
- `DocumentNotFoundError` — id unknown and `if_missing="error"`.

---

### `delete_document`

Remove a document and its orphaned entities.

```python
async def delete_document(
    self,
    document_id: str,
    *,
    if_missing: Literal["error", "ignore"] = "error",
) -> DeleteDocumentResult
```

#### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `document_id` | `str` | — required — | The Document node id. |
| `if_missing` | `Literal["error", "ignore"]` | `"error"` | `"ignore"` returns an empty result (zero counts), making the call idempotent. |

#### Returns

`DeleteDocumentResult` — `document_uid`, `chunks_deleted`, `entities_deleted`.

#### Raises

- `ValueError` — empty `document_id`.
- `DocumentNotFoundError` — `if_missing="error"` and id unknown.

---

### `apply_changes`

Heterogeneous batch — adds, modifies, deletes in one call. The canonical CI / PR-merge primitive.

```python
async def apply_changes(
    self,
    *,
    added: list[str] | None = None,
    modified: list[str] | None = None,
    deleted: list[str] | None = None,
    loader: LoaderStrategy | None = None,
    chunker: ChunkingStrategy | None = None,
    extractor: ExtractionStrategy | None = None,
    resolver: ResolutionStrategy | None = None,
    max_concurrency: int = 3,
    update_concurrency: int = 1,
    ctx: Context | None = None,
) -> ApplyChangesResult
```

Dispatches: `added` → `ingest()`, `modified` → `update(if_missing="ingest")`, `deleted` → `delete_document()`. Execution order is deletes → updates → adds (public contract).

#### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `added` | `list[str] \| None` | `None` | New file paths to ingest. |
| `modified` | `list[str] \| None` | `None` | File paths whose content changed. |
| `deleted` | `list[str] \| None` | `None` | Document ids to remove. |
| `loader` / `chunker` / `extractor` / `resolver` | strategy overrides, forwarded to `ingest`/`update`. |
| `max_concurrency` | `int` | `3` | Parallelism for the `added` list. |
| `update_concurrency` | `int` | `1` | Parallelism for the `modified` list. **Don't raise without reading the source comment** — v1.1's orphan-cleanup correctness depends on this default. |
| `ctx` | `Context \| None` | `Context()` | Execution context. |

#### Returns

`ApplyChangesResult` — three lists of `BatchEntry`. Per-file errors are wrapped, not raised. Inspect `entry.is_success` / `entry.error_type`.

#### Notes

- **Does NOT call `finalize()`.** Caller drives that cadence — typically once per CI run.

---

### `finalize`

Cross-document deduplication, entity/edge embedding backfill, index creation. **O(graph size)**, call once per ingestion batch.

```python
async def finalize(self) -> FinalizeResult
```

#### Returns

`FinalizeResult` — typed counts: `null_stubs_removed`, `entities_deduplicated`, `entities_embedded`, `relationships_embedded`, `indexes`.

---

### `deduplicate_entities`

Just the dedup step from `finalize()`. Use when you want fuzzy dedup and the rest of `finalize()` has already run.

```python
async def deduplicate_entities(
    self,
    *,
    fuzzy: bool = False,
    similarity_threshold: float = 0.9,
    batch_size: int = 500,
) -> int
```

#### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `fuzzy` | `bool` | `False` | Phase 2 fuzzy dedup via name embeddings. |
| `similarity_threshold` | `float` | `0.9` | Cosine cutoff for fuzzy matches. |
| `batch_size` | `int` | `500` | Entities per scan batch. |

#### Returns

Total duplicates merged (`int`).

---

## Retrieval

### `retrieve`

Retrieve context only — no LLM generation.

```python
async def retrieve(
    self,
    question: str,
    *,
    strategy: RetrievalStrategy | None = None,
    reranker: RerankingStrategy | None = None,
    ctx: Context | None = None,
) -> RetrieverResult
```

#### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `question` | `str` | — required — | The user's question. |
| `strategy` | `RetrievalStrategy \| None` | constructor default (`MultiPathRetrieval`) | Override the retrieval strategy. |
| `reranker` | `RerankingStrategy \| None` | `None` | Apply a reranker to the retrieved items. |
| `ctx` | `Context \| None` | `Context()` | Execution context. |

#### Returns

`RetrieverResult` — items with content, metadata, score.

---

### `completion`

Full RAG: retrieve context, generate an answer, return both.

```python
async def completion(
    self,
    question: str,
    *,
    history: list[ChatMessage | dict[str, str]] | None = None,
    strategy: RetrievalStrategy | None = None,
    reranker: RerankingStrategy | None = None,
    prompt_template: str | None = None,
    rewrite_question_with_history: bool = False,
    return_context: bool = False,
    ctx: Context | None = None,
) -> RagResult
```

#### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `question` | `str` | — required — | The user's question. |
| `history` | `list[ChatMessage \| dict] \| None` | `None` | Conversation history. First message may be a `system` role — passed as-is. |
| `strategy` | `RetrievalStrategy \| None` | constructor default | Per-call retrieval strategy. |
| `reranker` | `RerankingStrategy \| None` | `None` | Per-call reranker. |
| `prompt_template` | `str \| None` | built-in | Template with `{context}` and `{question}` placeholders. |
| `rewrite_question_with_history` | `bool` | `False` | When True and history is provided, run a cheap LLM call to rewrite the question into standalone form before retrieval. |
| `return_context` | `bool` | `False` | Include `RetrieverResult` on the returned `RagResult`. |
| `ctx` | `Context \| None` | `Context()` | Execution context. |

#### Returns

`RagResult` — `answer`, optional `retriever_result`, `metadata` (model, num context items, strategy name, has_history, retrieval_query).

---

## Schema discovery (new in v1.2)

### `suggest_schema_extensions`

Propose additions to the committed ontology from new sources. Nothing is applied to the graph.

```python
async def suggest_schema_extensions(
    self,
    sources: str | list[str],
    *,
    boundaries: str | None = None,
    sample_chunks_per_doc: int = 3,
    max_retries: int = 3,
    concurrency: int = 4,
    chunker: ChunkingStrategy | None = None,
    loader: LoaderStrategy | None = None,
    seed: int | None = None,
) -> SchemaExtensionProposal
```

#### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `sources` | `str \| list[str]` | — required — | New documents that motivate additions. |
| `boundaries` | `str \| None` | `None` | Free-text scope hint passed into every prompt. |
| `sample_chunks_per_doc` | `int` | `3` | Chunks per doc fed to the per-chunk proposal step. |
| `max_retries` | `int` | `3` | Retry budget per LLM call inside the validation-retry wrapper. |
| `concurrency` | `int` | `4` | Max in-flight LLM calls. |
| `chunker` / `loader` | strategy overrides. |
| `seed` | `int \| None` | `None` | RNG seed for deterministic chunk sampling. |

#### Returns

`SchemaExtensionProposal` — `new_entities`, `new_relations`, `new_patterns`, `new_attributes`, `sources_scanned`. May be empty (`proposal.is_empty == True`).

---

## Ontology accessors

### `get_ontology`

Reload the persisted ontology and return it.

```python
async def get_ontology(self) -> Ontology
```

### `refresh_ontology`

Alias for `get_ontology()` — reload and propagate to the retrieval strategy. Use when another process may have evolved the ontology and you want the next retrieval to see it.

```python
async def refresh_ontology(self) -> Ontology
```

### `set_ontology`

Replace the working ontology and re-register it.

```python
async def set_ontology(self, ontology: Ontology) -> Ontology
```

Adding new labels is always accepted; redeclaring an existing property's type raises `OntologyContradictionError`.

### `save_ontology`

Write the current ontology to a JSON file.

```python
async def save_ontology(self, path: str, *, indent: int = 2) -> None
```

---

## Ontology evolution

All evolution methods return the refreshed `Ontology` (unless noted) and persist the change. They follow a data-first ordering — data graph mutations happen before ontology mutations so a crash mid-call leaves a state the next retry can recover from.

### Descriptions (no data, no LLM)

```python
async def set_entity_description(self, label: str, description: str | None) -> Ontology
async def set_relation_description(self, label: str, description: str | None) -> Ontology
async def set_attribute_description(self, owner_label: str, attribute_name: str, description: str | None) -> Ontology
```

Set or clear the description on an existing entity, relation, or attribute.

### Additions (data-aware)

```python
async def add_entity(self, entity: Entity) -> Ontology
async def add_relation_pattern(self, rel_label: str, source: str, target: str) -> Ontology

async def add_attribute(
    self,
    owner_label: str,
    attribute: Attribute,
    *,
    concurrency: int = 4,
    dry_run: bool = False,
) -> EvolutionResult
```

`add_entity` and `add_relation_pattern` declare types without re-scanning chunks — pair with `backfill_*` to populate from existing data.

**`add_attribute` is atomic and LLM-backfilled.** It re-scans every chunk that mentions an entity of `owner_label`, asks the LLM to extract the new attribute's value, and writes it. The ontology graph is committed *last* — a crash leaves the ontology at its pre-call state and a retry is idempotent. `dry_run=True` previews the chunk count without invoking the LLM. Relation owners raise `NotImplementedError` in v1.2 (see Concepts → Ontology discovery for the workaround).

### Renames (data migration first, then ontology)

```python
async def rename_entity(self, old: str, new: str) -> Ontology
async def rename_relation(self, old: str, new: str) -> Ontology
async def rename_attribute(self, owner_label: str, old_name: str, new_name: str) -> Ontology
```

Each runs the Cypher rename across the data graph first, then updates the ontology graph. Crash between the two: data is migrated, ontology lags one rename — re-run to recover.

### Drops

```python
async def drop_entity(self, label: str) -> Ontology
async def drop_relation(self, label: str) -> Ontology
async def drop_attribute(self, owner_label: str, name: str) -> Ontology
async def drop_relation_pattern(self, rel_label: str, source: str, target: str) -> Ontology
```

`drop_entity` `DETACH DELETE`s every instance. `drop_relation` and `drop_relation_pattern` delete edges from the data graph and the matching ontology entries.

### Backfills

```python
async def backfill_entity(
    self,
    label: str,
    *,
    scope: str | list[str] = "all",
    concurrency: int = 4,
    dry_run: bool = False,
) -> BackfillResult

async def backfill_relation_pattern(
    self,
    rel_label: str,
    source: str,
    target: str,
    *,
    scope: str = "candidate-pairs",
    concurrency: int = 4,
    dry_run: bool = False,
) -> BackfillResult
```

After declaring a new entity or relation pattern with `add_entity` / `add_relation_pattern`, use the matching `backfill_*` to find instances in already-ingested chunks. Cost is proportional to chunks in scope; both methods are idempotent via per-chunk operation markers.

`backfill_entity(scope="all")` re-scans every chunk in the graph; pass a list of chunk ids to constrain. `backfill_relation_pattern(scope="candidate-pairs")` only considers chunks where both source-type and target-type entities co-occur — the universe of plausible new edges.

---

## Graph administration

### `get_statistics`

```python
async def get_statistics(self) -> dict[str, Any]
```

Node and edge counts, entity/relation type lists, density, `MENTIONED_IN` edge count.

### `delete_all`

```python
async def delete_all(self) -> None
```

Drop the entire knowledge graph (data + ontology). Irreversible. Invalidates cached config so the next `ingest()` re-validates and re-registers.

### `close`

```python
async def close(self) -> None
```

Close the underlying connection pool. Prefer `async with` for automatic cleanup.

---

## Sync convenience wrappers

```python
def retrieve_sync(self, question: str, *, strategy=None, reranker=None, ctx=None) -> RetrieverResult
def completion_sync(self, question: str, *, history=None, strategy=None, reranker=None, prompt_template=None, rewrite_question_with_history=False, return_context=False, ctx=None) -> RagResult
def ingest_sync(self, source=None, *, text=None, document_id=None, loader=None, chunker=None, extractor=None, resolver=None, max_concurrency=3, ctx=None) -> IngestionResult | list[IngestionResult | Exception]
```

Each wraps the matching async method with `asyncio.run()`. **Do not call from inside an existing event loop** — `asyncio.run()` will raise `RuntimeError`. Use the async methods directly in that case.

---

## Deprecated

| Symbol | Replacement | Removed in |
|---|---|---|
| `GraphRAG(..., schema=...)` | `ontology=` | future major |
| `rag.schema` (property) | `rag.ontology` | future major |

Each still works but emits `DeprecationWarning` on every access.

## See also

- [Concepts → Ingestion pipeline](../concepts/ingestion-pipeline)
- [Concepts → Retrieval pipeline](../concepts/retrieval-pipeline)
- [Concepts → Incremental updates](../concepts/incremental-updates)
- [Concepts → Ontology discovery](../concepts/ontology-discovery)
