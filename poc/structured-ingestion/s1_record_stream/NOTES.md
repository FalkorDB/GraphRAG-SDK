# s1 — record stream shape · DECIDED

**Question.** Proposal #1 declares `records: Iterable[dict[str, Any]]` on a pydantic `DataModel`
and asserts it is "streamed, never fully materialised". Is that true, and is it safe?

**Run:** `python s1_record_stream/spike.py` (no DB, no keys). All checks pass.

## What actually happens

Pydantic v2.11 does **not** materialise an `Iterable[...]` field — it replaces whatever you pass
with a `pydantic_core.ValidatorIterator`. Streaming works, and works well:

| 200,000 rows | peak memory |
| --- | --- |
| streamed | ~0.0 MB |
| materialised via `list()` | 71.6 MB |

`repr()`, `model_dump()` and `model_copy()` all leave the stream intact, so incidental logging is
not a hazard. So far the design's claim holds.

## The two things the design got wrong

**1 — the field is one-shot, and proposal #6 iterates it twice.**
The 9-step pipeline consumes records in step 3 (build record `Chunk`s) and again in step 4
(map records → `GraphData`). Over a `ValidatorIterator` the second pass yields nothing:

```
step 3 saw 10 records, step 4 saw 0 — no error raised
```

No exception, no warning — ingestion would report success and write **zero nodes**. This is the
worst possible failure shape and it is latent in the design as written.

**2 — the annotation erases list-ness.**
Even when the caller passes a fully materialised `list`, the field comes back as a
`ValidatorIterator` and `len()` raises `TypeError`. Any downstream code wanting a cheap record
count (progress reporting, `IngestionResult.records_processed`, batch sizing) cannot have one.

## Decision

Use a **stream factory**, not a stream:

```python
class RecordBatch(DataModel):
    open_records: Callable[[], Iterator[dict[str, Any]]]   # re-iterable by construction
    document_info: DocumentInfo
    inferred_types: dict[str, str]
    record_count: int | None = None    # set when the loader knows it cheaply; None when streaming

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self.open_records()
```

Verified in the spike: this shape survives `model_dump()` **and** the two-pass pipeline, returning
10/10 records on both passes.

A loader then hands over a re-openable source (reopen the file handle / re-run the cursor) rather
than a live generator, which is also the honest contract — a CSV *can* be read twice, and where it
genuinely cannot (a network stream), the loader spools once and closes over the buffer, making the
cost explicit at the loader instead of silently corrupting the write.

**Feeds back into the design:** proposal #1's `RecordBatch` signature, and a note on #6 that
step 3 and step 4 are two independent passes.
