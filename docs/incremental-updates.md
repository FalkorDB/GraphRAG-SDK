# Incremental Updates — Keeping the Graph in Sync

Once a graph has been ingested, real corpora keep changing: documents get edited, replaced, or removed. Re-running a full ingest on every change is expensive and discards the existing graph. The v1.1.0 incremental update primitives let you mutate an already-built graph in place, with crash-safe semantics and scoped orphan cleanup.

This page is the API reference and usage guide for those primitives:

- [`update()`](#update) — re-sync a single document
- [`delete_document()`](#delete_document) — remove a single document and its orphans
- [`apply_changes()`](#apply_changes) — heterogeneous batch (added / modified / deleted)
- [`finalize()`](#finalize) — run once at the end of a batch

The canonical use case is **CI-driven graph updates on PR merge**: a typical diff has all three change types in one go, and `apply_changes()` routes each list to the right primitive.

---

## Mental Model — What Lives in the Graph

To understand the effect of each primitive, it helps to know the node shapes the lexical graph keeps:

| Label              | Purpose                                                                 |
| ------------------ | ----------------------------------------------------------------------- |
| `Document`         | One per ingested source. Carries `content_hash`, `path`, and metadata.  |
| `Chunk`            | Linked to its `Document` via `PART_OF`. Holds embeddings.               |
| `__Entity__`       | Extracted entities. Linked to chunks via `MENTIONED_IN`.                |
| `RELATES` (edge)   | Fact between two entities. Carries `source_chunk_ids` for provenance.   |

Incremental updates are scoped via these relationships — orphan cleanup never goes global, it only inspects entities the touched document actually referenced.

---

## `update()`

Re-sync a previously-ingested document. The new content replaces the old chunks, entities are re-extracted, and entities that become orphaned (no remaining `MENTIONED_IN` edges) are cleaned up along with their incident `RELATES` edges.

```python
result = await rag.update(
    source=None,                    # OR text=...
    text=None,
    document_id=None,
    loader=None,
    chunker=None,
    extractor=None,
    resolver=None,
    if_missing="error",             # "error" | "ingest"
    ctx=None,
)
```

### Parameters

| Argument                                | Meaning                                                                                                                                                           |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `source`                                | File path. Mutually exclusive with `text`. In file mode, `document_id` defaults to `os.path.normpath(source)`.                                                    |
| `text`                                  | Raw text. Skips the loader. Requires an explicit `document_id`.                                                                                                   |
| `document_id`                           | Stable id of the `Document` node to update. Required for text mode.                                                                                               |
| `loader` / `chunker` / `extractor` / `resolver` | Per-call strategy overrides, identical to `ingest()`.                                                                                                     |
| `if_missing`                            | `"error"` (default) raises `DocumentNotFoundError` if the id is unknown. `"ingest"` falls through to a fresh `ingest()` — upsert semantics.                       |

### Returns — `UpdateResult`

```python
UpdateResult(
    document_info: DocumentInfo,
    nodes_created: int,
    relationships_created: int,
    chunks_indexed: int,            # new chunks written
    chunks_deleted: int,            # old chunks removed
    entities_deleted: int,          # orphan entities pruned
    no_op: bool,                    # True if content_hash matched
    replaced_existing: bool,        # False if if_missing="ingest" upserted
    metadata: dict[str, Any],
)
```

### Effect on the graph

| Scenario                                  | What changes                                                                                                                                                                                 |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Content hash matches (touch-only edit)    | **Nothing.** SHA-256 short-circuits the call to a single Cypher lookup. `no_op=True`. Use this for CRLF/formatter-only PRs.                                                                  |
| Real content change                       | New chunks written under a pending Document, then atomically cut over to the canonical id. Old chunks are deleted. Entities previously mentioned only by this document are removed.          |
| Entity still referenced by another doc    | **Preserved.** Orphan cleanup is scoped to candidates from this document — never global.                                                                                                     |
| `RELATES` edge sourced only from old chunks | **Removed** as a stale fact (cleanup keyed on `source_chunk_ids`).                                                                                                                          |
| `if_missing="ingest"` and id unknown      | Behaves like a fresh `ingest()`. `replaced_existing=False`.                                                                                                                                  |

### Crash safety

The call uses a six-phase state machine with one load-bearing commit point (`mark_pending_committed`). Crashes before the commit roll back (pending data is discarded); crashes after the commit roll forward (cleanup resumes on the next call to `update`/`delete_document` against this id).

### Example

```python
# Initial ingest with a stable id
await rag.ingest(text=V1_TEXT, document_id="alice_bio")

# Touch-only — short-circuits via content hash
result = await rag.update(text=V1_TEXT, document_id="alice_bio")
assert result.no_op is True

# Real edit — replaces chunks, prunes orphans
result = await rag.update(text=V2_TEXT, document_id="alice_bio")
print(f"replaced {result.chunks_deleted} chunks, "
      f"removed {result.entities_deleted} orphan entities, "
      f"wrote {result.chunks_indexed} new chunks")
```

---

## `delete_document()`

Remove a single document and everything it uniquely owns.

```python
result = await rag.delete_document(
    document_id: str,
    *,
    if_missing="error",             # "error" | "ignore"
)
```

### Parameters

| Argument      | Meaning                                                                                                                                                                                |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `document_id` | The Document node id (e.g. `os.path.normpath(path)` for file-mode ingests).                                                                                                            |
| `if_missing`  | `"error"` (default) raises `DocumentNotFoundError`. `"ignore"` returns an empty result with zero counts — useful for CI deletes when the caller doesn't track which files were ever ingested. |

### Returns — `DeleteDocumentResult`

```python
DeleteDocumentResult(
    document_uid: str,
    chunks_deleted: int,
    entities_deleted: int,
)
```

### Effect on the graph

| What is removed                                                                  | Notes                                                                |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| All `Chunk` nodes linked to the Document via `PART_OF`                           | Unconditional — the chunks belong only to this Document.             |
| The `Document` node itself                                                       | Removed last, after cleanup completes.                               |
| `__Entity__` nodes that the doc referenced and have **no** remaining `MENTIONED_IN` | Scoped to candidates from this document; entities still cited elsewhere are kept. |
| `RELATES` edges whose `source_chunk_ids` are entirely from this Document's chunks | Stale-fact cleanup. Edges with multi-document provenance survive.    |

### Crash safety

A single atomic write (`pending_delete=true` + cleanup-state arrays on the doc) is the commit marker. Before that write the live document is untouched. After it, every remaining step is idempotent and resumes on the next call against this id.

### Example

```python
await rag.delete_document("alice_bio")

# Idempotent cleanup of files removed in a PR — never errors if unknown
await rag.delete_document("some/path.md", if_missing="ignore")
```

---

## `apply_changes()`

Heterogeneous batch — the convenience wrapper for CI-driven incremental ingestion. Dispatches each list to the right primitive in a fixed order:

1. `deleted` → `delete_document()`
2. `modified` → `update(if_missing="ingest")` (so a "modified" file the graph has never seen is upserted, not errored)
3. `added` → `ingest()`

```python
result = await rag.apply_changes(
    *,
    added=None,
    modified=None,
    deleted=None,
    loader=None,
    chunker=None,
    extractor=None,
    resolver=None,
    max_concurrency=3,              # for `added`
    update_concurrency=1,           # for `modified` — see warning below
    ctx=None,
)
```

### Parameters

| Argument             | Meaning                                                                                                                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `added`              | New file paths to ingest.                                                                                                                                                                              |
| `modified`           | File paths whose content changed.                                                                                                                                                                      |
| `deleted`            | Document ids (typically file paths) to remove.                                                                                                                                                         |
| `loader` / `chunker` / `extractor` / `resolver` | Strategy overrides forwarded to the add/update path. Ignored for deletes.                                                                                                                  |
| `max_concurrency`    | Parallelism cap for `ingest()` of the `added` list. Safe to raise — adds have no orphan-cleanup race surface.                                                                                          |
| `update_concurrency` | **Default 1, and you almost certainly should not raise it.** The orphan-cleanup invariant only holds while updates serialize. See [Concurrency](#concurrency-and-the-update_concurrency-trap) below.   |

The order `deletes → updates → adds` is part of the public contract; do not assume reordering is safe.

Overlapping ids across input lists raise `ValueError` at the input boundary — this is almost always a broken git-diff parser.

### Returns — `ApplyChangesResult`

```python
ApplyChangesResult(
    added:    list[BatchEntry[IngestionResult]],
    modified: list[BatchEntry[UpdateResult]],
    deleted:  list[BatchEntry[DeleteDocumentResult]],
)
```

Each entry aligns by index with the corresponding input list. Per-file errors are wrapped as `BatchEntry` with `error` (string) and `error_type` (exception class name) set; the batch never raises. Branch on `entry.is_success`:

```python
for entry in result.modified:
    if entry.is_success:
        ...   # entry.result is an UpdateResult
    elif entry.error_type == "DocumentNotFoundError":
        ...
```

### Effect on the graph

`apply_changes()` does not itself touch the graph — it dispatches to the primitives documented above and aggregates their results. Two whole-batch consequences worth knowing:

- **No automatic `finalize()`.** Cross-document deduplication is O(graph size), so the batch deliberately leaves it to the caller. Call `finalize()` once after the batch, not once per file.
- **Peak entity cardinality is minimised** because deletes run first — orphan candidates are gone before adds can re-introduce overlapping ids.

### Canonical CI usage

```python
async with GraphRAG(connection=..., llm=..., embedder=...) as rag:
    await rag.apply_changes(**parse_git_diff(pr_sha, base_sha))
    await rag.finalize()
```

### Concurrency and the `update_concurrency` trap

Raising `update_concurrency` above 1 is unsafe in general because the orphan-cleanup correctness proof relies on `MENTIONED_IN` edges being persisted before any cutover begins. Two concurrent updates sharing an entity `e1` are only guaranteed to preserve it because:

- Pre-cutover, `e1` still has its old `MENTIONED_IN` edges.
- Post-`pipeline.run()` (but pre-cutover), the new `MENTIONED_IN` edges are already written.

So an update doing orphan cleanup will always see at least one incident edge. Raising the default to ≥2 is **only safe** if you can independently guarantee that no two parallel updates can share an entity in their candidate snapshots. The integration test `test_concurrent_updates_preserve_shared_entity` is the tripwire for this invariant — break it before bumping the value.

---

## `finalize()`

Run **once** after a batch of ingests/updates/deletes. Skipping it leaves cross-document duplicates in place and disables entity-/edge-level vector search.

```python
result = await rag.finalize()
```

### Returns — `FinalizeResult`

```python
FinalizeResult(
    null_stubs_removed:     int,
    entities_deduplicated:  int,
    entities_embedded:      int,
    relationships_embedded: int,
    indexes:                dict[str, bool],
)
```

### What it does (in order)

1. Removes NULL-name stub entities (legacy cleanup).
2. `deduplicate_entities()` — global exact-name dedup across all documents.
3. `backfill_entity_embeddings()` — embeds entity names that have no embedding yet (incremental-safe).
4. `embed_relationships()` — embeds fact text on `RELATES` edges.
5. `ensure_indices()` — rebuilds/verifies all indexes.

Steps 2 and 3 are why `apply_changes()` does not call it automatically: they scan the whole graph and would re-run on every file in a batch.

---

## Choosing the Right Primitive

| Situation                                                                | Use                                       |
| ------------------------------------------------------------------------ | ----------------------------------------- |
| Single doc, you know it exists and the text changed                      | `update(...)`                             |
| Single doc, may or may not exist (upsert)                                | `update(..., if_missing="ingest")`        |
| Single doc, remove it                                                    | `delete_document(...)`                    |
| Single doc, remove it but the caller is not sure it was ever ingested    | `delete_document(..., if_missing="ignore")` |
| PR diff or any heterogeneous batch                                       | `apply_changes(added=..., modified=..., deleted=...)` then `finalize()` |
| Just added/changed one doc and you're about to query                     | `ingest()` or `update()` then `finalize()` |

---

## End-to-End Example

A full runnable example lives in [`graphrag_sdk/examples/07_incremental_updates.py`](../graphrag_sdk/examples/07_incremental_updates.py). It exercises:

- Initial ingest with a stable `document_id`
- No-op update (content-hash short-circuit)
- Real update with orphan cleanup
- Adding a second document
- A heterogeneous `apply_changes()` batch
- A single trailing `finalize()`
- Verifying with `completion()`

---

## See Also

- [Ingestion](ingestion.md) — the underlying 9-step pipeline that `ingest()` and `update()` both drive.
- [Storage](storage.md) — node/edge shapes, including why MERGE makes re-ingestion idempotent.
- [Configuration](configuration.md) — `finalize()` reference and post-ingestion knobs.
- [Ontology Discovery](ontology-discovery.md) — if you're updating against a corpus and don't have a schema yet, or want to propose schema additions from new documents via `suggest_schema_extensions`.
- [Ontology Evolution](ontology-evolution.md) — if your schema needs to change between updates (`add_entity`, `add_attribute`, `rename_*`, etc.) — these are the mutation primitives that the discovery proposal hands off to.
