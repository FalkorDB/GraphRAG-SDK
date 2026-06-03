---
title: "Incremental updates"
nav_order: 6
parent: "Concepts"
grand_parent: "GraphRAG-SDK"
description: "Re-sync individual documents without rebuilding. The update / delete_document / apply_changes primitives and the cost model behind finalize()."
---

# Incremental updates

Most graphs don't sit still. Docs change, files get deleted, new files arrive. Three methods let you re-sync a knowledge graph without rebuilding it from scratch:

| Method | When to use |
|---|---|
| `update(source, document_id=...)` | One document's content changed. Re-ingests, replaces the document's chunks and entities, and updates the content hash. |
| `delete_document(document_id)` | Document removed. Cleans up the document's chunks and any entities orphaned by the deletion. Entities still referenced by other documents are preserved. |
| `apply_changes(added=[...], modified=[...], deleted=[...])` | Heterogeneous batch — multiple add / modify / delete operations in one call. The canonical CI pattern on PR merge. |

All three operate on a stable `document_id`. In file mode that defaults to `os.path.normpath(source)`; in text mode you pass an explicit id.

## The content-hash short-circuit

`update()` computes a SHA-256 hash of the new content and compares it to the hash stored on the document node. If they match, the call returns immediately with `no_op=True` — one Cypher query, no chunking, no LLM, no extraction. A "touch" PR that didn't actually change the file is effectively free.

```python
result = await rag.update("docs/api.md", document_id="docs/api.md")
if result.no_op:
    print("Unchanged — skipped")
```

## `apply_changes` — the canonical batch

The PR-merge use case is the model. You have three sets of paths: added files, modified files, deleted files. `apply_changes` runs them as one batch:

```python
async with GraphRAG(connection=ConnectionConfig(...), llm=..., embedder=...) as rag:
    result = await rag.apply_changes(
        added=["docs/new_feature.md"],
        modified=["docs/api.md"],
        deleted=["docs/removed_page.md"],
    )
    await rag.finalize()   # exactly once per batch

    for entry in result.added + result.modified + result.deleted:
        if not entry.is_success:
            print(f"failed: {entry.error_type}: {entry.error}")
```

Two properties of the batch matter:

- **Per-file errors are collected, not raised.** Each input file becomes a `BatchEntry` — either `entry.result` (the typed payload) or `entry.error` (formatted message) plus `entry.error_type` (the exception class name, e.g. `"DocumentNotFoundError"`). The batch never raises. Branch on `entry.is_success` or `entry.error_type` programmatically.
- **`apply_changes` does NOT call `finalize()`.** It runs the per-file write path and returns. You drive `finalize` cadence — typically once at the end of the run.

## Cost model — `finalize` is O(graph size)

`finalize()` runs cross-document deduplication. That step scans the full entity table, so its cost is **O(graph size)** — independent of how many documents changed in this batch.

The implication for CI: do not call `finalize()` per file. Batch every change in the run through `apply_changes` (or alternate `update` / `delete_document` calls) and call `finalize()` **once** at the end. A 100-file PR with per-file finalize would do ~100× more work than a single end-of-run finalize.

## Crash safety

`update()` uses a rollforward cutover pattern:

1. Mark the document as `pending_update` and write the new chunks under a fresh content-hash key.
2. Cut over: rewrite `PART_OF` edges to the new chunks, drop the old chunks.
3. Clear the `pending_update` marker.

If the process crashes between (1) and (3), the next call sees the marker and resumes from where it stopped. The same crash before (1) is a no-op — the old document is unchanged. Result: re-running the same `update()` call after a crash is safe and idempotent.

`delete_document()` similarly stages the delete before committing — a crash mid-delete leaves the document discoverable on retry.

## When to use `update` vs re-ingest

| Situation | API |
|---|---|
| Document content has changed | `update(source, document_id=...)` — replaces the chunks, preserves provenance through shared entities. |
| Document is brand new, no prior version | `ingest(source, document_id=...)` — same write path; `update` would fall back to ingest with `if_missing="ingest"`. |
| Schema changed, want to re-extract every doc | `delete_all()` + fresh ingest of everything. There is no in-place schema-driven re-extraction in v1. |
| Adding a new attribute to an existing schema | `add_attribute(...)` — atomic, LLM-backfilled, no re-ingest needed. |

## `if_missing` policy

`update(source, document_id=..., if_missing=...)` controls the fallback when `document_id` doesn't exist:

- `"ingest"` (default-ish) — fresh-ingest the document. `result.replaced_existing=False`.
- `"error"` — raise `DocumentNotFoundError`. Use when the caller expects the document to exist.

## See also

- [API Reference → GraphRAG](../api-reference/graphrag) — `update`, `delete_document`, `apply_changes`, `finalize`, all parameters and result types.
- [API Reference → Result types](../api-reference/result-types) — `UpdateResult`, `DeleteDocumentResult`, `ApplyChangesResult`, `BatchEntry`.
- [Guides → CI / PR-merge pattern](../guides/ci-pr-merge-pattern) — runnable example with `gh pr diff`.
