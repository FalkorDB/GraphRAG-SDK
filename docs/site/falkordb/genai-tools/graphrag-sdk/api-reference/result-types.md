---
title: "Result types"
nav_order: 8
parent: "API Reference"
grand_parent: "GraphRAG-SDK"
description: "Typed return values from ingest, update, delete_document, apply_changes, finalize, retrieve, completion, and the evolution methods."
---

# Result types

Module: `graphrag_sdk`  ·  All importable from the top level.

Every public method returns a typed Pydantic v2 model — call sites get IDE autocomplete and mypy enforcement on field access.

---

## Ingestion / Update / Delete

### `IngestionResult`

Returned by `GraphRAG.ingest(single_source)`.

```python
class IngestionResult(DataModel):
    document_info: DocumentInfo
    nodes_created: int = 0
    relationships_created: int = 0
    chunks_indexed: int = 0
    metadata: dict[str, Any]
```

### `UpdateResult`

Returned by `GraphRAG.update`. Subclass of `IngestionResult`.

```python
class UpdateResult(IngestionResult):
    chunks_deleted: int = 0
    entities_deleted: int = 0
    no_op: bool = False
    replaced_existing: bool = False
```

| Field | Description |
|---|---|
| `no_op` | `True` when the new content's SHA-256 matched the stored hash — nothing was written. |
| `replaced_existing` | `True` when an existing document was replaced. `False` when `if_missing="ingest"` fell through to a fresh ingest. |

### `DeleteDocumentResult`

Returned by `GraphRAG.delete_document`.

```python
class DeleteDocumentResult(DataModel):
    document_uid: str
    chunks_deleted: int = 0
    entities_deleted: int = 0
```

### `BatchEntry[T]`

Wraps the per-file outcome in `apply_changes`. Generic over the typed result.

```python
class BatchEntry(DataModel, Generic[T_BatchResult]):
    result: T_BatchResult | None = None
    error: str | None = None
    error_type: str | None = None
```

| Field | Description |
|---|---|
| `result` | Typed payload on success. |
| `error` | Formatted exception message on failure. |
| `error_type` | Exception class name (`"DocumentNotFoundError"`, etc.) for programmatic branching. |
| `is_success` (property) | `error is None`. |

### `ApplyChangesResult`

Returned by `GraphRAG.apply_changes`.

```python
class ApplyChangesResult(DataModel):
    added: list[BatchEntry[IngestionResult]] = []
    modified: list[BatchEntry[UpdateResult]] = []
    deleted: list[BatchEntry[DeleteDocumentResult]] = []
```

Each list aligns with the corresponding input list by index.

### `FinalizeResult`

Returned by `GraphRAG.finalize`.

```python
class FinalizeResult(DataModel):
    null_stubs_removed: int = 0
    entities_deduplicated: int = 0
    entities_embedded: int = 0
    relationships_embedded: int = 0
    indexes: dict[str, bool]
```

---

## Retrieval / Completion

### `RetrieverResultItem`

```python
class RetrieverResultItem(DataModel):
    content: str
    metadata: dict[str, Any]
    score: float | None = None
```

### `RetrieverResult`

```python
class RetrieverResult(DataModel):
    items: list[RetrieverResultItem] = []
    metadata: dict[str, Any]
```

### `RagResult`

Returned by `GraphRAG.completion`.

```python
class RagResult(DataModel):
    answer: str
    retriever_result: RetrieverResult | None = None
    metadata: dict[str, Any]
```

`metadata` carries `model`, `num_context_items`, `strategy`, `has_history`, `retrieval_query`.

---

## Extraction internals

### `GraphData`

Output of the extraction stage. Internal — surfaced for advanced users writing custom extractors.

```python
class GraphData(DataModel):
    nodes: list[GraphNode] = []
    relationships: list[GraphRelationship] = []
    mentions: list[EntityMention] = []
    extracted_entities: list[ExtractedEntity] = []
    extracted_relations: list[ExtractedRelation] = []
```

### `ResolutionResult`

Output of the resolution stage.

```python
class ResolutionResult(DataModel):
    nodes: list[GraphNode] = []
    relationships: list[GraphRelationship] = []
    merged_count: int = 0
    remap: dict[str, str] = {}
```

`remap` records every `loser_id → survivor_id` decision the resolver made. The pipeline rewrites `MENTIONS` edges through this so they point at survivors.

---

## Evolution / Backfill

### `EvolutionResult`

Returned by `GraphRAG.add_attribute`.

```python
class EvolutionResult:
    ontology: Ontology
    chunks_in_scope: int = 0
    chunks_scanned: int = 0
    chunks_skipped: int = 0
    llm_calls: int = 0
    values_filled: int = 0
    values_skipped: int = 0
    elapsed_s: float = 0.0
```

### `BackfillResult`

Returned by `GraphRAG.backfill_entity` and `backfill_relation_pattern`.

```python
class BackfillResult:
    operation_id: str = ""
    chunks_in_scope: int = 0
    chunks_scanned: int = 0
    chunks_skipped: int = 0
    target_nodes: int = 0
    failed_chunks: list[str] = []
    llm_calls: int = 0
    values_filled: int = 0
    values_skipped: int = 0
    elapsed_s: float = 0.0
```

`failed_chunks` lists per-chunk hard failures (LLM/parse errors that exhausted retries). Re-running the same backfill is idempotent — chunk markers skip completed chunks.

---

## Support types

### `DocumentInfo`

```python
class DocumentInfo(DataModel):
    path: str | None = None
    uid: str        # auto-generated UUID4 if not supplied
    metadata: dict[str, Any] = {}
```

### `TextChunk` / `TextChunks`

```python
class TextChunk(DataModel):
    text: str
    index: int
    metadata: dict[str, Any] = {}
    uid: str        # auto-generated UUID4

class TextChunks(DataModel):
    chunks: list[TextChunk] = []
```

### `GraphNode` / `GraphRelationship`

```python
class GraphNode(DataModel):
    id: str
    label: str
    properties: dict[str, Any] = {}
    embedding_properties: dict[str, list[float]] | None = None

class GraphRelationship(DataModel):
    start_node_id: str
    end_node_id: str
    type: str
    properties: dict[str, Any] = {}
    embedding_properties: dict[str, list[float]] | None = None
```

### `ChatMessage`

```python
class ChatMessage(DataModel):
    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]
```

### `SearchType`

```python
class SearchType(str, Enum):
    VECTOR = "vector"
    FULLTEXT = "fulltext"
    HYBRID = "hybrid"
```

## See also

- [API Reference → GraphRAG](./graphrag) — methods that return these types.
- [Concepts → Incremental updates](../concepts/incremental-updates) — `BatchEntry` failure semantics.
