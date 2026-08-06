# FINDINGS — what the spikes changed about the design

Five spikes, each answering one open question against real code and a real FalkorDB.
`python run_all.py` → **all 5 pass**.

Four of the five falsified something in `docs/design/structured-ingestion.md`. One of those
(s3) inverts a headline decision.

| # | Spike | Verdict on the design |
| --- | --- | --- |
| s1 | record stream | **amend** — `RecordBatch` must expose a stream *factory*; as written it silently writes zero rows |
| s2 | mapping DSL | **amend** — edges must address node *aliases*, not labels; `to_ontology()` needs two guards |
| s3 | identity | **invert** — `identity=["name"]` as the default is wrong; key + `alias_ids` is the only policy that works |
| s4 | record-as-chunk | **confirmed, incl. the predicted data-loss trap** |
| s5 | pipeline seam | **amend** — share a base class, don't subclass; one new kwarg for `NEXT_CHUNK` |

---

## 1. `RecordBatch` must be a stream factory  → proposal #1

Pydantic v2 keeps an `Iterable[dict]` field lazy (200k rows: ~0 MB vs 71.6 MB materialised) but
wraps it in a one-shot `ValidatorIterator`. Proposal #6 iterates records **twice** — step 3 builds
record chunks, step 4 maps records to `GraphData`. Measured result:

```
step 3 saw 10 records, step 4 saw 0 — no error raised
```

Silent zero-row ingest. Also, the annotation erases list-ness: `len()` raises even when the caller
passed a list, so no cheap record count is available for progress or `IngestionResult`.

**Change:** `open_records: Callable[[], Iterator[dict]]` + `record_count: int | None`. Verified
re-iterable and `model_dump()`-safe. *(details: `s1_record_stream/NOTES.md`)*

## 2. Edges must address aliases, not labels  → proposal #2

`EdgeMapping(source="Organization", target="Organization")` cannot express a transaction with a
buyer and a seller. Executed against `transactions.csv` it produced a **self-loop**
(`ORG-7 -> ORG-7` instead of `ORG-7 -> ORG-42`) with no error. Buyer/seller, manager/report,
parent/subsidiary is the standard shape of transactional data, not an edge case.

**Change:** `NodeMapping(alias=..., ...)`, `EdgeMapping(source=<alias>, target=<alias>)`, alias
defaulting to the label so the 80% case is untouched. Two further guards on `to_ontology()`: reject
`_RESERVED_ATTRIBUTE_NAMES` (a mapping can otherwise shadow SDK-written keys like `id` and
`description`), and emit stubs for reference-only labels (otherwise `Ontology`'s own validator
warns on every ingest — 2 warnings from one mapping in the spike).

Bonus: `Entity(label=..., identity=[...])` **already works** via `Config.extra = "allow"` and
survives `model_dump()`, so #3 is prototypable with zero `src` changes and old `ontology.json`
files stay loadable. *(details: `s2_mapping_dsl/NOTES.md`)*

## 3. Identity: the design's default is backwards  → proposals #3, #4

Measured on the #82 acceptance corpus, three policies × two ingest orders:

| Policy | Acme nodes | #82 traversal |
| --- | --- | --- |
| name-first (`identity=["name"]`, **the design's default**) | 2 | **0 people reachable** |
| key-only | 2 | **0 people reachable** |
| key + `alias_ids` + resolve pass | **1** | **2 people reachable** |

Cause: `employees.csv` references its org by `org_id=ORG-42` and has no `org_name` column. Under
name-first identity the mapping **cannot compute the identity of the entity it points at**, so
proposal #2's rule "each mapping must supply the type's identity attributes" is unsatisfiable for
any normalised foreign key. The result is a stub node holding all the `WORKS_AT` edges, sitting
next to the real Acme node that holds the prose — the exact failure #82 exists to prevent, silent.

**Change:** structured writes are **key-identified**, and `alias_ids` moves from optional bridge to
**critical path**. Caveat, also measured: with no source carrying key *and* name together, the
bridge degrades to 2 nodes / 0 reachable. The honest contract is **ingest order does not matter,
but presence does** — and unbridged stubs must be reported in `IngestionResult` rather than left
silent. *(details: `s3_identity/NOTES.md`)*

## 4. The predicted cutover trap is real  → proposal #5

The design review predicted, from reading code, that deterministic chunk uids keyed on the
canonical document id would destroy data during `update()`. Run against real FalkorDB through the
real `GraphStore`:

| chunk uid keyed on | chunks before | shared with pending | **after cutover** |
| --- | --- | --- | --- |
| canonical doc id | 3 | 3 | **0** |
| effective (pending) doc id | 3 | 0 | **3** |

The pending run `MERGE`s onto the live document's chunk nodes; `rollforward_cutover()` step 1
deletes them; step 3 promotes an empty document. No exception. Today's `uuid4()` uids are
accidentally immune — which is why making them deterministic is the dangerous part of #5.

Everything else in #5 **holds**: `get_document_entity_candidates()` found 5 record-chunk entities,
`delete_stale_relationships()` GC'd exactly the removed row's fact, `delete_orphan_entities()`
removed exactly Carol and the org that lost its last mention.

**Change:** promote the §5 callout from "review finding" to a hard rule with this measurement
attached. §8.8 (row-level incremental update is *not* free) stays refuted.
*(details: `s4_record_as_chunk/NOTES.md`)*

## 5. Share a base class, don't subclass  → proposal #6

`_build_lexical_graph` / `_prune` / `_write_mentions` are reusable **verbatim** — the first already
consumes `TextChunks`, which is what proposal #5 turns records into. Both factorings produced
identical graphs (9 nodes, 6 `MENTIONED_IN`).

But `IngestionPipeline.__init__` requires `loader, chunker, extractor, resolver, graph_store,
vector_store`. A structured pipeline has no chunker and no LLM extractor, so subclassing means
passing `None` and hoping — it works today only by accident of which methods are called.

**Change:** extract the three methods into a `LexicalGraphWriter` base depending only on
`graph_store`. Plus one new kwarg: `_build_lexical_graph(..., link_sequential: bool = True)`, since
reusing it as-is chains `NEXT_CHUNK` between unrelated CSV rows (N-1 edges asserting a sequence
that doesn't exist, while `cypher_generation.py` tells the LLM those edges mean "next sequential
Chunk"). *(details: `s5_pipeline_seam/NOTES.md`)*

---

## What did **not** change

The core architecture survived contact with the database. Record-as-chunk, `RELATES` + `rel_type`,
mapping-as-ontology-fragment, deterministic no-LLM mapping, and "no retrieval strategy is touched"
all held up. Every correction above is a change to a *signature or a default*, not to the shape of
the design — which is the outcome a spike round is supposed to produce.

## Disposal

Once these are folded into `docs/design/structured-ingestion.md` and the implementation lands,
delete `poc/`. Nothing imports it; nothing ships it.
