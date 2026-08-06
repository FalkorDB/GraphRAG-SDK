# s4 — record-as-chunk & the `update()` cutover · CONFIRMED

**Question.** Proposal #5 claims record-as-chunk makes `update()` / `delete_document()` work
unchanged. The design review *predicted*, from reading `api/main.py:2104` and
`rollforward_cutover()`, that deterministic chunk uids keyed on the canonical document id would
silently destroy data. A prediction from reading code is a hypothesis; this spike runs it against
a real FalkorDB through the real `GraphStore`.

**Run:** `python s4_record_as_chunk/spike.py` (needs FalkorDB; no keys). All checks pass.

## The trap is real

| chunk uid keyed on | chunks before `update()` | chunk nodes shared with pending | chunks after cutover |
| --- | --- | --- | --- |
| **canonical** document id | 3 | **3** | **0** |
| **effective** (pending) document id | 3 | 0 | **3** |

Mechanism, now observed rather than inferred:

1. `update()` writes the new version under `pending_id = f"{resolved_id}__pending__{uuid4().hex[:8]}"`.
2. With canonical-keyed uids, the pending run's `MERGE` lands on the **same `Chunk` nodes** as the
   live document — the spike measured all 3 chunks carrying `PART_OF` from *both* Documents.
3. `rollforward_cutover()` step 1 calls `delete_document_chunks_and_node(real_id)`, whose Cypher is
   `MATCH (:Document {id})-[:PART_OF]->(c:Chunk) DETACH DELETE c` — it deletes the shared nodes.
4. Step 3 renames the pending to the canonical id. It is promoted **with zero chunks**.

No exception. `rollforward_cutover`'s precondition guard doesn't help — the pending Document node
exists, it is only its chunks that have been destroyed. The user sees a successful `update()` and a
document whose every record has vanished, along with the provenance that orphan cleanup depends on.

Today's `uuid4()` chunk uids are accidentally immune, which is exactly why making them
deterministic is the dangerous part of proposal #5.

**Fix (verified):** key record chunk uids on the run's *effective* `DocumentInfo.uid` —
`sha256(effective_document_id :: record_key)`. The pending run's chunk set is then disjoint
(0 shared), the cutover deletes only v1's chunks, and all 3 new chunks are promoted. This
preserves today's disjointness property while keeping uids deterministic *within* a run, which is
all that re-ingest idempotency actually requires.

## The rest of proposal #5 holds

Second scenario: delete Carol (`E-3`) from `employees.csv` and re-ingest, then run the real
cleanup primitives.

| primitive | result |
| --- | --- |
| `get_document_entity_candidates()` | 5 candidates — record-chunk entities are found by the existing `(:__Entity__)-[:MENTIONED_IN]->(:Chunk)<-[:PART_OF]-(:Document)` walk |
| `delete_stale_relationships()` | 1 edge deleted — Carol's `WORKS_AT` fact GC'd via `source_chunk_ids` |
| `delete_orphan_entities()` | 2 deleted — Carol **and** `ORG-7`, which lost its only mention; Alice and Bob untouched |

So the headline claim survives: **record-as-chunk inherits `update()`/`delete_document()` for free,
conditional on effective-uid keying.** The three write-time properties the design flags as
mandatory (`name` on nodes, `fact` and `source_chunk_ids` on `RELATES`) are load-bearing here —
`delete_stale_relationships` is driven entirely by `source_chunk_ids`, and without it Carol's fact
would survive forever.

## Consequence for the design

`docs/design/structured-ingestion.md` §5's callout box is **confirmed** and should be promoted from
"design-review finding" to a hard rule with this measurement attached. Proposal #5's claim that
row-level incremental update follows for free stays **refuted** (§8.8) — the pending id is
randomised per run, so a v2 run can never MERGE onto v1's chunks by construction. Row-level diffing
needs its own mechanism, not a uid convention.
