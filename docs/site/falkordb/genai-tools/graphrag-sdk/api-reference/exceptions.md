---
title: "Exceptions"
nav_order: 10
parent: "API Reference"
grand_parent: "GraphRAG-SDK"
description: "Every exception class the SDK can raise, organised by subsystem."
---

# Exceptions

Module: `graphrag_sdk`  ·  Many subclasses live under `graphrag_sdk.core.exceptions` and `graphrag_sdk.discovery`.

All SDK exceptions inherit from `GraphRAGError`, so a single `except graphrag_sdk.GraphRAGError` clause catches everything the SDK can throw.

```
GraphRAGError                              (base)
├── LatencyBudgetExceededError
├── LLMError
│   └── LLMTimeoutError
├── EmbeddingError
│   └── EmbeddingTimeoutError
├── IngestionError
│   ├── LoaderError
│   ├── ChunkingError
│   ├── ExtractionError
│   └── ResolutionError
├── RetrieverError
├── DatabaseError
├── IndexError_
├── DocumentNotFoundError
├── SchemaValidationError
└── ConfigError

# Free-standing (not subclasses of GraphRAGError)
OntologyDiscoveryError(RuntimeError)
OntologyEvolutionError
OntologyContradictionError(ValueError)
OntologyModificationNotAllowedError(RuntimeError)
DBpediaFetchError(RuntimeError)
```

---

## Top-level

### `GraphRAGError`

Base for everything below. Import from `graphrag_sdk`.

### `LatencyBudgetExceededError`

Raised when an operation cannot start within the remaining `Context.latency_budget`. Use `Context(latency_budget=...)` to enforce a soft deadline across a request.

---

## Provider errors

### `LLMError` / `LLMTimeoutError`

`LLMError` wraps a failed LLM call. `LLMTimeoutError` is its timeout subclass — raised when `wait_for_provider_call` hits the configured `timeout`.

### `EmbeddingError` / `EmbeddingTimeoutError`

Same shape for the embedding provider.

---

## Ingestion errors

`IngestionError` is the base for stage-specific failures. Each ingestion stage has its own:

- `LoaderError` — a `LoaderStrategy.load()` failed.
- `ChunkingError` — a `ChunkingStrategy.chunk()` failed.
- `ExtractionError` — extraction (LLM + NER) failed.
- `ResolutionError` — resolution failed.

In batch ingestion (`ingest(list_of_sources)` and `apply_changes`), these become entries in the per-file result list — they don't propagate out of the batch.

---

## Retrieval errors

### `RetrieverError`

Raised during retrieval. `MultiPathRetrieval` swallows per-mode failures internally and logs them; this exception surfaces only when every mode fails or a top-level call cannot proceed.

---

## Storage errors

### `DatabaseError`

FalkorDB driver-level failures and invariant violations (e.g., a `mark_pending_committed` that affected zero or two nodes when exactly one was expected).

### `IndexError_`

Index creation / management failures. Named with a trailing underscore to avoid shadowing the built-in `IndexError`.

---

## Document lifecycle

### `DocumentNotFoundError`

Raised by `update` and `delete_document` when the supplied id is unknown and `if_missing="error"`.

---

## Schema / configuration

### `SchemaValidationError`

Raised when graph schema validation fails — typically construction-time issues uncovered by `OntologyStore.register`.

### `ConfigError`

Raised when SDK configuration is invalid — most commonly an embedder dimension that doesn't match the persisted `__GraphRAGConfig__` node.

---

## Ontology lifecycle

These are **not** subclasses of `GraphRAGError` — they predate the unified hierarchy or are intentionally non-domain-specific.

### `OntologyContradictionError`

Raised by `OntologyStore.register` (and `GraphRAG.set_ontology`) when the user redeclares an existing property with a conflicting type. Subclass of `ValueError`.

### `OntologyModificationNotAllowedError`

Strict-mode invariant violation. Subclass of `RuntimeError`. Deprecated alias `SchemaModificationNotAllowedError` is still importable but emits `DeprecationWarning`.

### `OntologyEvolutionError`

Raised by `GraphRAG.add_attribute` when one or more chunks hard-fail during the LLM-backfill phase. The ontology graph is *not* committed in this case — re-running the same `add_attribute(...)` call is safe and idempotent. The exception's `failed_chunks` and `chunks_scanned` attributes carry the diagnostic detail.

### `OntologyDiscoveryError`

Raised by the validation-retry wrapper inside the discovery pipeline when an individual LLM call exhausts its retry budget. Subclass of `RuntimeError`. The pipeline catches these as soft-fail, so end-users normally don't see this exception unless they call `extract_with_retry` directly.

| Attribute | Type | Description |
|---|---|---|
| `chunk_id` | `str \| None` | Unit being processed — chunk uid, `"summary:<src>"`, `"normalize"`, or `None`. |
| `attempts` | `int` | LLM calls made before giving up. |
| `last_error` | `Exception \| None` | Last validation / parse error. |

### `DBpediaFetchError`

Raised by `DBpediaCatalog` when DBpedia or Schema.org is unreachable and the on-disk cache is missing or stale. Subclass of `RuntimeError`. Pre-warm the cache, or pass a `cache_path` pointing at a known-good copy.

## See also

- [API Reference → GraphRAG](./graphrag) — per-method `Raises` sections call out which exceptions can surface.
- [Concepts → Incremental updates](../concepts/incremental-updates) — how `apply_changes` wraps these into `BatchEntry.error_type`.
