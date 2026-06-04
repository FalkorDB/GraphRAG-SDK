# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3.0] - 2026-06-04

Ontology discovery (#271): bootstrap an ontology straight from a
corpus instead of hand-writing one, and propose additive schema
extensions against a live graph as new sources arrive. Two discovery
modes ship — LLM-driven (the default) and **grounded**, which anchors
discovered types to an external vocabulary catalog (Schema.org types,
DBpedia entity linking).

No breaking changes. All existing ingest paths are unaffected; the new
API is purely additive, and the invariant from 1.2.0 still holds — a
discovered attribute is only declared when the corpus supports it.

### Added

#### Corpus-driven ontology discovery (#271)

- **`Ontology.from_sources(sources, llm=None, *, method="llm",
  boundaries=None, existing=None, catalog=None, sample_chunks_per_doc=3,
  max_retries=3, concurrency=4, seed=None, ...)`** — build an
  `Ontology` from a document corpus. `method="llm"` (default) derives
  entities, relations, and attributes from sampled chunks via the LLM;
  `method="grounded"` resolves entity mentions against a `Catalog` and
  pulls typed schema from the external vocabulary. `boundaries` scopes
  the domain, `existing` seeds discovery from a prior ontology, and
  `seed` makes sampling deterministic.

- **`GraphRAG.suggest_schema_extensions(sources, *, boundaries=None,
  sample_chunks_per_doc=3, max_retries=3, concurrency=4, seed=None,
  ...)`** — scan new sources against the currently committed ontology
  and return an additive `SchemaExtensionProposal` (new entities,
  relations, patterns, attributes). Read-only: proposes, never mutates
  — apply the pieces you want via the 1.2.0 evolution API.

- **`SchemaExtensionProposal`** — additions-only result carrying
  `new_entities`, `new_relations`, `new_patterns`,
  `new_attributes`, and `sources_scanned`, plus an `is_empty` property
  and a `summary()` helper.

**Grounded discovery — vocabulary catalogs:**

- **`Catalog`** — ABC for grounding discovery in an external
  vocabulary: `link_entity(name)`, `lookup(type_name)`, and
  `relations_among(type_names)`.

- **`DBpediaCatalog(sparql_endpoint=None, schema_org_url=None,
  cache_path=None, cache_ttl_days=30)`** — built-in catalog that
  resolves entity mentions to types via the DBpedia SPARQL endpoint and
  pulls type schema from Schema.org's JSON-LD vocabulary. Schema.org is
  disk-cached (default `~/.cache/graphrag-sdk/schema_org.json`, 30-day
  TTL); `DBpediaFetchError` is raised when a service is unreachable and
  no usable cache exists. **Note:** grounded discovery makes live
  network calls to DBpedia and Schema.org.

**Lower-level pipeline (`graphrag_sdk.discovery`):**

- **`discover_ontology(...)`**, **`discover_grounded(...)`**, and
  **`suggest_extensions(...)`** — the functions backing
  `from_sources(method="llm")`, `from_sources(method="grounded")`, and
  `suggest_schema_extensions` respectively, exposed for callers that
  want to drive the pipeline directly.
- **`OntologyDiscoveryError`** — raised when an LLM extraction call
  exhausts `max_retries`; carries `chunk_id`, `attempts`, and
  `last_error`.

- **`docs/ontology-discovery.md`** — end-to-end guide covering both
  discovery modes and the extension-proposal workflow.

Two large, related additions: a persistent ontology graph with
schema-driven typed attributes (#256), and a mutating evolution API
that lets you safely change the ontology after first ingest (#268).
Both ship under a strict alignment invariant: **a declared attribute
is queryable on every instance of its owner entity type** — the SDK
exposes no API that can declare schema your data doesn't match.

No breaking changes. The vocabulary rename (`GraphSchema` →
`Ontology`, `EntityType` → `Entity`, `RelationType` → `Relation`,
`PropertyType` → `Attribute`) ships with backwards-compatible aliases
that emit `DeprecationWarning` on each access.

### Added

#### Persistent ontology graph + typed attributes (#256)

- **Three-node schema graph** lives in a paired
  `<data_graph>__ontology` graph: `:Entity`, `:Relation`, and
  `:Property` nodes, connected by `HAS_PROPERTY` / `SOURCE` / `TARGET`
  edges. One `:Relation` node per declared `(label, src, tgt)` triple;
  open-mode relations are a single node without `SOURCE`/`TARGET`.
  Always-on per `GraphRAG` instance; created lazily on first use,
  dropped on `delete_all()`.

- **`Attribute(name, type, description)`** Pydantic model with typed
  values (`STRING`, `INTEGER`, `FLOAT`, `BOOLEAN`, `DATE`, `LIST`).
  Declared per `Entity` / `Relation` via `properties=[...]`; the
  extractor surfaces declared attributes in the LLM prompt, parses
  typed values out of the response, and stores them on the data
  graph with type coercion at write time.

- **`GraphRAG.get_ontology()` / `refresh_ontology()` /
  `set_ontology(ontology)` / `save_ontology(path)`** — facade for
  reading and exporting the persisted ontology. `set_ontology()` is
  the public seam for swapping the working ontology between ingest
  passes (additive only — modification rules below).

- **`OntologyStore.register(ontology)`** enforces strict ingest
  semantics: new labels are accepted, re-declarations of existing
  labels must be a strict subset of the persisted definition, and
  any attempt to add properties / patterns to an existing label
  raises `OntologyModificationNotAllowedError`. Type contradictions
  raise `OntologyContradictionError`. Both are raised before any
  partial state persists.

- **`examples/08_ontology_lifecycle.py`** — runnable walkthrough of
  declaring typed attributes, round-tripping the ontology through a
  JSON file (`Ontology.save_to_file` / `from_file`), and the additive
  evolution rules.

#### Mutating evolution API (#268)

15 new public methods on `GraphRAG`, grouped by what they touch.

**Group 1 — pure declarations (cheap, no LLM)** — return `Ontology`:
- `set_entity_description(label, description)`
- `set_relation_description(label, description)`
- `set_attribute_description(owner_label, attribute_name, description)`
- `add_entity(entity)` (declaration only)
- `add_relation_pattern(rel_label, source, target)` (declaration only)

**Group 2 — mechanical data migration (Cypher, no LLM)** — return
`Ontology`. Data migration runs first; ontology graph is updated
second. Idempotent on crash.
- `rename_entity(old, new)`
- `rename_attribute(owner_label, old_name, new_name)`
- `rename_relation(old, new)` (recreate-and-delete via UNWIND;
  warns at >10k edges)
- `drop_entity(label)` (cascades to relation patterns referencing it)
- `drop_relation(label)`
- `drop_relation_pattern(rel_label, source, target)`

**Group 3 — atomic attribute evolution (LLM, invariant-enforcing)** —
return `EvolutionResult`:
- `add_attribute(owner_label, attribute, *, concurrency=4, dry_run=False)`
  — declare + LLM backfill across chunks mentioning the owner +
  ontology commit, atomically. The ontology graph write is the
  commit point (last). Entity owners only.
- `drop_attribute(owner_label, name)` — REMOVE on every instance +
  ontology delete. Entity owners only.

Type changes deliberately go through `drop_attribute` + `add_attribute`
(the LLM re-derives values from chunks). There is no `retype_attribute`.

**Group 4 — opportunistic discovery (opt-in, not invariant-enforcing)**
— return `BackfillResult`:
- `backfill_entity(label, *, scope, concurrency=4, dry_run=False)` —
  re-scan chunks for any entities of this type you might have missed.
- `backfill_relation_pattern(rel_label, source, target, *, dry_run=False)`
  — re-scan candidate co-mention chunks for any edges of this pattern.

**Supporting types:**
- **`EvolutionResult`** — return of `add_attribute`. Carries the
  refreshed ontology plus observability counters (`chunks_in_scope`,
  `chunks_scanned`, `chunks_skipped`, `llm_calls`, `values_filled`,
  `values_skipped`, `elapsed_s`).
- **`OntologyEvolutionError`** — raised by `add_attribute` when one
  or more chunks hard-fail. The ontology graph is NOT updated when
  raised. Carries `failed_chunks: list[str]` and `chunks_scanned`.
- **`BackfillResult`** — return of Group 4. Carries `chunks_in_scope`,
  `chunks_scanned`, `chunks_skipped`, `llm_calls`, `values_filled`,
  `failed_chunks: list[str]`, etc. `failed_chunks` here is collected,
  not raised on — opportunistic discovery isn't invariant-enforcing.
- **`BackfillExecutor`** — worker-pool executor backing all Group 3/4
  methods. Live `asyncio.Task` count stays O(concurrency) regardless
  of corpus size (workers consume `ChunkContext` items from a bounded
  `asyncio.Queue`).
- **`ChunkContext`**, **`BackfillMergeStats`** — handed to the
  per-operation prompt/parse/merge callbacks.

All exported from the top-level package.

- **`examples/09_ontology_evolution.py`** — runnable walkthrough of
  the full evolution API.

- **`docs/ontology-evolution.md`** — design doc: the invariant, all
  four groups, atomicity model, failure & retry, type-change pattern,
  concurrency rule, cost preview (`dry_run`), end-to-end example,
  reference tables, and what's not in the API and why.

### Changed

- **Vocabulary rename** (non-breaking via aliases): `GraphSchema` →
  `Ontology`, `EntityType` → `Entity`, `RelationType` → `Relation`,
  `PropertyType` → `Attribute`,
  `SchemaModificationNotAllowedError` →
  `OntologyModificationNotAllowedError`, and the `schema=` keyword on
  `GraphRAG(...)` / `OntologyStore.register()` / `IngestionPipeline(...)`
  / retrieval strategy constructors → `ontology=`. The old names
  remain importable and the old keywords still accepted, each emitting
  a `DeprecationWarning` pointing at the new name. Will be removed
  in a future release.

- **`set_ontology()` and `get_ontology()` propagate to retrieval.**
  Updating the working ontology (via `set_ontology`, `get_ontology`,
  or any Group 1–3 evolution method) reassigns
  `_retrieval_strategy._ontology` in lockstep with `_global_ontology`
  and `self.ontology`. Concurrent retrieval calls always see the
  same ontology snapshot as the most recent evolution write.

### Deprecated

- **Legacy ontology vocabulary.** Importing `GraphSchema`,
  `EntityType`, `RelationType`, `PropertyType`,
  `SchemaModificationNotAllowedError` from `graphrag_sdk` or
  `graphrag_sdk.core.models` emits `DeprecationWarning`. Passing
  `schema=` to `GraphRAG(...)` or `OntologyStore.register()` also
  emits one. Old names continue to work; replace at your convenience.

### Notes

- **Alignment invariant.** Group 3 (`add_attribute`,
  `drop_attribute`) is the only invariant-enforcing tier:
  `add_attribute` runs the LLM backfill **before** committing the
  ontology graph, so the schema never claims a property that the
  data graph hasn't been asked to populate. Entities for which the
  LLM returned `null` (or which weren't mentioned in any chunk) read
  as `null` — Cypher treats absent and explicit-null identically
  under `WHERE n.attr IS NULL`, so callers get a consistent view
  regardless of which underlies a given entity.

- **Concurrency rule.** Evolution calls are not safe to run
  concurrently with `ingest()` or with each other on the same graph.
  The extractor reads the persisted ontology to decide what to
  extract; Group 2/4 calls internally use a count-then-mutate
  pattern across two Cypher statements. Gate evolution behind an
  application-level lock in multi-process deployments. Treat
  evolution as a maintenance operation: pause ingest, run the
  evolution, resume.

- **Cost preview.** `add_attribute`, `backfill_entity`, and
  `backfill_relation_pattern` accept `dry_run=True` to run the scope
  query and return `chunks_in_scope` (the count this run *would*
  scan) without invoking the LLM or committing anything. The count
  reflects work that remains for this run — chunks already marked
  from a prior partial run are filtered out — so the preview is
  honest about the actual remaining cost.

- **Idempotent retries.** Both atomic evolution and opportunistic
  discovery use per-chunk `extracted_ops` markers on `:Chunk` nodes.
  `op_id` is deterministic from the operation signature
  (`f"add_attribute:{owner_label}:{attribute.name}:{attribute.type}"`)
  including the type — so `drop_attribute` + `add_attribute` with a
  new type triggers a fresh rescan rather than being filtered out
  by stale markers from the previous type's run.

- **Relation-attribute evolution is not implemented in 1.2.0.**
  `add_attribute` / `drop_attribute` / `rename_attribute` on a
  relation owner raise `NotImplementedError` — the workaround is
  `delete_all()` + re-ingest with the updated ontology. Tracked
  in #269 along with other deferred follow-ups (richer
  `OntologyEvolutionError` payload, token-cost preview, provenance
  on backfilled non-RELATES edges).

## [1.1.1] - 2026-05-13

Closes a usability gap in v1.1.0's `apply_changes` convenience wrapper:
per-call strategy overrides (`loader` / `chunker` / `extractor` /
`resolver`) now reach the inner `ingest()` and `update()` dispatches.
Previously callers who passed strategies to `apply_changes` got SDK
defaults silently — the only escape was to bypass the wrapper and loop
over the primitives directly. Default behaviour is unchanged.

### Added

- **`GraphRAG.apply_changes(*, loader=, chunker=, extractor=, resolver=, ...)`**
  and `apply_changes_sync()` — strategy overrides are now forwarded to
  the inner `ingest()` (for `added`) and `update()` (for `modified`)
  calls. `delete_document` does not take strategies and is unaffected.
  All four kwargs default to `None`, preserving v1.1.0 behaviour for
  callers that don't pass them.

## [1.1.0] - 2026-05-05

Adds incremental ingestion primitives and a CI-friendly batch wrapper.
Built around a crash-safe state-machine cutover (idempotent rollforward
keyed off a `ready_to_commit` marker on the pending Document node) so
mid-flight failures recover correctly on retry. Documents the cost
model so consumers can pick a sensible `finalize()` cadence. Closes a
pre-existing silent-correctness bug in fuzzy resolvers (`MENTIONED_IN`
edges were dropped for any merged entity). No breaking changes; one
previously-rejected argument shape (`document_id` in file mode) is now
accepted.

### Added

- **`GraphRAG.update(source=..., text=..., document_id=..., if_missing=...)`**
  and `update_sync()` — re-sync a previously-ingested document
  without rebuilding the rest of the graph. SHA-256 content-hash
  short-circuits no-op updates: a touch-only PR (CRLF, formatter-only
  changes) costs roughly one Cypher lookup.

  Cutover is a documented state machine
  (`EMPTY → WRITING → WRITTEN → COMMITTED → FINAL`) with a single
  load-bearing transition: setting `ready_to_commit = true` on the
  pending Document node. That is the commit point — it is one Cypher
  statement (atomic at the per-statement level FalkorDB does
  guarantee), and it must complete before any destructive op touches
  the live document. Once committed, recovery on crash is rollforward
  (each post-commit op is idempotent) rather than rollback. The
  practical effect: an `update()` interrupted by a transient failure
  retries cleanly instead of raising `DocumentNotFoundError` against
  a half-deleted document.

  Cleans up entities orphaned by the change (zero remaining
  `MENTIONED_IN`, scoped to entities the prior version of the document
  referenced) along with their incident `RELATES` edges; entities
  still referenced by other documents are preserved.

- **`GraphRAG.delete_document(document_id)`** and
  `delete_document_sync()` — remove one document's chunks and the
  Document node, plus orphaned entities scoped to that document. Other
  documents' data is untouched.

- **`GraphRAG.apply_changes(added=..., modified=..., deleted=...,
  max_concurrency=3, update_concurrency=1)`** and
  `apply_changes_sync()` — convenience batch dispatcher for CI-driven
  incremental ingestion. Routes each list to the right primitive
  (`ingest` for added, `update(if_missing="ingest")` for modified,
  `delete_document` for deleted). Per-file failures are collected,
  not raised — matching `ingest()`'s batch contract. Does **not**
  call `finalize()`; that stays the caller's responsibility so a
  50-file batch pays one finalize cost, not 50.

  `update_concurrency` defaults to 1 — forced by the orphan-cleanup
  invariant (concurrent updates A and B sharing entity `e1` are
  race-free only because `pipeline.run()` writes the new
  `MENTIONED_IN` edges before returning, and therefore before any
  cutover begins). The integration test
  `test_concurrent_updates_preserve_shared_entity` is the tripwire
  that protects this default. `max_concurrency` continues to control
  the `added` list and matches `ingest()`'s existing knob.

- **`UpdateResult`** (extends `IngestionResult` with `chunks_deleted`,
  `entities_deleted`, `no_op`, `replaced_existing`),
  **`DeleteDocumentResult`** (`document_uid`, `chunks_deleted`,
  `entities_deleted`), **`BatchEntry[T]`** (typed `result | error |
  error_type` wrapper with `is_success`), and **`ApplyChangesResult`**
  (`added: list[BatchEntry[IngestionResult]]`, `modified:
  list[BatchEntry[UpdateResult]]`, `deleted:
  list[BatchEntry[DeleteDocumentResult]]`) Pydantic models. All
  exported from the top-level package. The document id is consistently
  exposed via `DocumentInfo.uid` / `DeleteDocumentResult.document_uid`
  across all result types.

- **`DocumentNotFoundError`** — raised by `update()` (when
  `if_missing="error"`, the default) and `delete_document()` when the
  given id is unknown. `update()` accepts `if_missing="ingest"` for
  upsert semantics.

- **`Document.content_hash` and `Document.ready_to_commit`** —
  additive properties written by the ingestion pipeline /
  state-machine. Pre-existing graphs are unaffected; documents
  ingested before this release simply lack `content_hash` and
  `update()` always runs the full path for them (fail-safe).

- **Parametrized real-FalkorDB integration suite** —
  `tests/test_integration.py::TestIncrementalUpdateInvariants` covers
  four cases (shared-entity preservation, orphan-by-delete,
  orphan-by-update, concurrent-updates tripwire) across both
  `ExactMatchResolution` and `SemanticResolution`. Env-gated
  (`RUN_INTEGRATION=1`) and exercised by a new `integration` job in
  CI that boots a real FalkorDB service container.

### Changed

- **`document_id` accepted in file-mode `ingest()`.** Previously
  rejected with `ValueError`; now optional in both modes. When
  supplied, anchors the Document node's stable id — the handle
  callers pass to `update()` or `delete_document()`. When omitted
  in file mode, defaults to `os.path.normpath(source)` so
  `update(path)` matches the corresponding `ingest(path)` call
  with no extra plumbing. In text mode, `ingest()` auto-generates
  a `text-<hex>` id when omitted; supplying one is recommended for
  stability across runs and is required by `update()` /
  `delete_document()` since there's no path to derive an id from.

- **Path-conflict guard** replaces the v1.0.2 blanket rejection:
  ingesting/updating with a `document_id` already bound to a
  *different* path raises `ValueError`. This catches accidental
  aliasing across files; legitimate rebinds should go through
  `update()`.

- **`IngestionPipeline.run(..., document_info=...)`** now honors the
  caller-supplied uid in file mode (loader output is overlaid, not
  the other way around). Required for stable-id ingest; existing
  callers that didn't supply `document_info` are unaffected.

### Fixed

- **`MENTIONED_IN` edges silently dropped for entities merged by fuzzy
  resolvers.** `pipeline._write_mentions` consumed
  `graph_data.mentions` (pre-resolution entity ids) without rewriting
  them through the resolver's merge decisions. With default
  `ExactMatchResolution` this was harmless because exact-match groups
  by `(label, id)` and never merges across distinct ids, but
  `SemanticResolution` and `LLMVerifiedResolution` merge by embedding
  similarity / LLM judgment, so the post-resolution
  `upsert_relationships` `MATCH (a)` against the merged-away id
  silently failed and the mention edge was lost. v1.1.0's orphan
  cleanup would have inherited that breakage. The pipeline now
  rewrites mentions through `ResolutionResult.remap` between resolve
  and write; all bundled resolvers populate `remap`. Default-resolver
  users see no change.

### Hardening (post-review)

These changes landed after the initial PR commit, addressing review
findings that were discovered while walking the diff file-by-file.
None changes the high-level behavior of the v1.1.0 primitives; all
either close a latent bug or harden a contract.

- **`pipeline._remap_mentions` follows transitive remap chains.** The
  initial fix only handled single-hop remaps. Two-stage resolvers
  (`SemanticResolution`, `LLMVerifiedResolution`) merge per-phase
  remap dicts without flattening, so the combined dict could carry
  chains like `{a: b, b: c}` where `b` was itself merged away. A
  single-hop lookup pointed mentions at the now-orphan intermediate;
  the MENTIONED_IN write then silently MATCH-failed on `b`. Now
  follows each chain to its terminal survivor with a visited-set
  guard against cyclic remaps.

- **`GraphStore.find_pending` queries COMMITTED state explicitly first.**
  The initial implementation used `ORDER BY p.id LIMIT 1` across the
  pending prefix. Lexicographic order on the random pending suffix
  meant that under compounded crashes — when the graph briefly held
  both a WRITTEN and a COMMITTED pending — a WRITTEN one could sort
  first, causing the caller to take the rollback path and the
  COMMITTED pending to silently replay over freshly-written live
  data on the next cycle. Now issues two queries: COMMITTED first
  (which MUST be rolled forward), WRITTEN as fallback.

- **`GraphStore.rollforward_cutover` precondition checks pending
  exists.** Previously, calling rollforward with a stale `pending_id`
  would delete the live document (steps 1-2) and then silently no-op
  the rename (step 3 MATCH finds nothing), leaving the graph empty.
  Now refuses to proceed unless the pending node is present.

- **`GraphRAG.update` asserts `mark_pending_committed` returned exactly
  1.** A return of 0 means the pending vanished mid-flow — the
  rollforward queries are idempotent so they would silently no-op,
  losing the new data while reporting success to the caller. Now
  raises `DatabaseError` before Phase 5.

- **`GraphRAG.update` Phase 0 raises on corrupt committed pending.**
  Previously, a COMMITTED pending without persisted path metadata
  was silently defaulted (`path = resolved_id`, `hash = ""`). The
  empty hash broke future no-op short-circuits forever for that doc.
  Now raises `DatabaseError` — corruption surface, not a default-and-
  continue case.

- **Reserved-substring guard on `document_id`.** A new
  `_check_no_pending_marker` helper rejects ids containing the
  literal `__pending__` separator at the public API boundary (in
  `update`, `delete_document`, and `ingest`). Without this guard, a
  Document with id `foo__pending__bar.txt` would be matched by
  `find_pending("foo")`'s prefix scan and incorrectly treated as a
  leftover pending of `foo`.

- **`apply_changes` rejects overlapping ids across input lists.**
  If the same id appears in two of `{added, modified, deleted}`
  (typically a broken git-diff parser), the dispatch order would
  silently apply both operations against the same doc. Now raises
  `ValueError` at the input boundary listing the offending ids.

- **`delete_document` gains `if_missing="ignore"`.** Closes the
  asymmetry with `update`'s `if_missing` parameter. Useful for CI
  cleanup of files removed in a PR when the caller doesn't track
  which were ever ingested. Returns an empty `DeleteDocumentResult`
  instead of raising `DocumentNotFoundError`.

- **`delete_document` runs orphan cleanup as best-effort.** A
  try/finally ensures the orphan sweep still runs even if the
  chunk-and-node deletion raises mid-flight (network blip, transient
  Cypher error). Process death between the two remains unrecoverable
  here (no persistent recovery handle for delete); a full
  state-machine for delete is left as a future enhancement.

- **Result types tightened.** `UpdateResult.previous_document_uid` was
  misleading (always equal to the current uid on both no-op and
  success paths) — replaced with `replaced_existing: bool`. The
  document id is uniformly available via `document_info.uid`
  (matching `IngestionResult`) or `DeleteDocumentResult.document_uid`
  (matching `DocumentInfo.uid` naming).

- **`ApplyChangesResult` uses typed `BatchEntry[T]` wrappers.** Was
  `list[Result | Exception]` with `arbitrary_types_allowed=True`
  (which disabled Pydantic validation for entries). Now
  `list[BatchEntry[T]]` where `BatchEntry` carries `result | error |
  error_type` with an `is_success` property. JSON-serialisable; full
  validation re-enabled. Callers branch on `entry.is_success` instead
  of `isinstance(entry, Exception)`.

- **`text-mode` document_id collision space bumped to 64 bits.**
  Was `f"text-{uuid4().hex[:8]}"` (32 bits, ~12% birthday-collision
  at 10K text-mode ingests). Now `[:16]` (64 bits, ~2 in 10^11 for
  the same volume).

- **Crash-recovery integration test added.** End-to-end test that
  manually creates a COMMITTED pending via direct Cypher, drops the
  GraphRAG (simulating process death), opens a fresh instance against
  the same graph_name, and verifies Phase 0 rolls forward. The only
  test that verifies the "FalkorDB persists `ready_to_commit` across
  a connection drop" claim against a real database.

### Notes

- **`finalize()` cost model.** Step 1 (NULL-name stub cleanup) and
  step 2 (`deduplicate_entities`) scan the full entity table; steps
  3–4 (`backfill_entity_embeddings`, `embed_relationships`) only
  touch nodes/edges missing embeddings. Net: O(graph size) for dedup,
  O(change size) for embedding backfill. For CI use cases, batch
  all PR changes via `apply_changes` and call `finalize` once at
  the end of the run, not once per file.

- **Pipeline step-8 ordering is load-bearing.** The `asyncio.gather`
  that writes `MENTIONED_IN` edges in `pipeline.run()` must complete
  before the function returns; deferring it to a background task or
  batching it across pipelines silently re-introduces a concurrent-
  update race even with `update_concurrency=1`. Marked with a
  load-bearing comment in `pipeline.py` plus the
  `test_concurrent_updates_preserve_shared_entity` integration
  tripwire.

- **Renames** (git's `R old.md -> new.md`) are out of scope for
  v1.1.0 — treat as delete + add. The `apply_changes` signature
  reserves room for a future `renamed=` keyword so consumers don't
  need to refactor when that lands.

- **Concurrent updates to the same `document_id`** are unsupported.
  Each call uses a unique pending-uid so the temp graphs don't
  collide, but the cutover races; callers must serialize updates
  to the same id.

## [1.0.2] - 2026-05-04

Patch release. One retrieval correctness fix and one default-value
change carried over from the post-1.0.1 README onboarding work.

### Fixed

- **Chunk citations preserve the full `Document.path`.** The chunk
  retrieval strategy was reducing the path returned from the graph
  to a basename via `path.rsplit("/", 1)[-1]` before handing it off
  to the citation pipeline. That dropped real information: files
  sharing a basename across directories — e.g. `operations/index.md`
  vs `commands/index.md` — collapsed to the same identifier
  downstream, and consumers building source links from the citation
  could no longer reconstruct the original location. `Document.path`
  already stored the full path passed to `rag.ingest()`, so this is
  a read-side fix only; existing graphs start emitting full paths in
  the next query with no migration required.

### Changed

- **Default `embedding_dimension` lowered from 1536 to 256.** Aligns
  the out-of-the-box default with the `text-embedding-3-large`
  Matryoshka 256-dim configuration used in the benchmark (overall
  ACC 69.73). Affects `GraphRAG(...)` and `VectorStore(...)` when
  `embedding_dimension` is left unset; existing graphs created
  with the prior default continue to work because the dimension
  is stored in the FalkorDB vector index. To preserve the old
  behavior on new graphs, pass `embedding_dimension=1536`
  explicitly. README, getting-started, api-reference, storage,
  graph-schema docs, and the custom-provider example updated to
  match.

## [1.0.1] - 2026-04-28

Security and API hygiene release addressing findings from a full
audit of v1.0.0. Includes one outright security fix (Cypher injection
surface), two correctness improvements (per-source error handling,
embedder/dim validation), and several public-API changes that move
the SDK to a more honest type contract.

This release contains breaking changes; see "Migration" below.

### Security

- **Cypher injection surface eliminated in `VectorStore`.** Replaced
  the parameterized `create_vector_index(label, property)` /
  `create_fulltext_index(label, *properties)` / `drop_vector_index(label)`
  with named methods (`create_chunk_vector_index`,
  `create_entity_vector_index`, `create_relates_vector_index`, etc.).
  Every Cypher query now uses literal identifiers — no user-supplied
  string is f-string-interpolated into a query. `embedding_dimension`
  is bound-checked (1..8192) at construction; `similarity_function`
  parameter dropped (was always `cosine`). Search/fulltext-search
  methods (`search`, `fulltext_search`) split into per-target
  variants (`search_chunks`, `fulltext_search_chunks`,
  `fulltext_search_entities`) for the same reason.
- **TLS support for FalkorDB connections.** Added `ssl`, `ssl_cert_reqs`,
  `ssl_ca_certs`, `ssl_certfile`, `ssl_keyfile`, `ssl_check_hostname` to
  `ConnectionConfig`. `from_url("rediss://...")` now auto-enables TLS;
  unknown URL schemes raise `ValueError` (closes a silent-downgrade
  footgun where `rediss://` previously got plaintext).
- **Prompt injection hardening on `completion()`.** When the default
  prompt template is in use, retrieved context is wrapped in
  `<context>...</context>` tags, the system prompt instructs the model
  to treat that block as untrusted reference data, and any `</context>`
  inside an item's content is neutralized so a malicious chunk cannot
  forge the closing tag and escape into instruction territory.
- **Provider exception logs sanitized.** The 5 retry sites in
  `LLMInterface.ainvoke`, `LiteLLM.ainvoke`/`ainvoke_messages`, and
  `OpenRouterLLM.ainvoke`/`ainvoke_messages` now log only a
  bounded one-line summary at WARNING (`type(exc).__name__: <first
  line, truncated to 200 chars>`); full exception with traceback
  goes to DEBUG via `exc_info=`. Prevents accidental leakage of
  request payloads, response bodies, or proxy URLs into shared logs.
- **Pagination loops capped.** Four `while True:` loops in
  `EntityDeduplicator` and `VectorStore` now use `for-else` over
  `_MAX_PAGINATION_ITERATIONS = 10_000`. Trips with a clear ERROR log
  if a server bug or driver issue ever causes a stall.
- **Tighter version pins.** `tiktoken<1.0`, `openai<2.0`,
  `anthropic<1.0`, `litellm<2.0`. Stable libs (`python-dotenv`,
  `hnswlib`) left unconstrained.

### Added

- **`GraphRAG.get_statistics()` / `GraphRAG.delete_all()`** — facade
  methods that replace the old pattern of reaching into
  `rag.graph_store` directly.
- **`FinalizeResult`** Pydantic model — typed return for `finalize()`
  / `finalize_sync()`. Exported from the top-level package.
- **`document_id` parameter on `ingest()`** — explicit identifier for
  text-mode ingestion. Auto-generated as `text-<8hex>` if omitted.
- **`RelationType.patterns` directionality diagnostics** — when
  schema pruning drops relationships because `(src, tgt)` doesn't
  match a declared pattern, a structured WARNING per relation type
  names the offending pairs (sampled to 3) and the declared patterns
  with a hint to check direction. `GraphSchema` also warns at
  construction when patterns reference undeclared entity labels.
- **Embedder dimension probe** — `_validate_graph_config()` invokes
  the embedder once and verifies the produced vector matches
  `embedding_dimension`, raising `ConfigError` on mismatch. Probe
  failures (network, auth) are logged at DEBUG and skipped.
- **`_validate_graph_config()` runs on `ingest()`** — cross-session
  embedder/dimension mismatches now surface before any extraction
  work, not just on first `retrieve()`.

### Changed

#### Breaking

- **Storage layer privatized.** `rag.graph_store` and `rag.vector_store`
  are now `_graph_store` / `_vector_store`. Replace
  `rag.graph_store.get_statistics()` with `rag.get_statistics()` and
  `rag.graph_store.delete_all()` with `rag.delete_all()`. The
  `GraphStore` and `VectorStore` classes remain publicly importable.
- **`ingest(max_concurrent=N)` renamed to `ingest(max_concurrency=N)`**
  for consistency with `LLMInterface.max_concurrency` (and the rest of
  the codebase). Old keyword raises `TypeError`.
- **`ingest()` source/text overload split.** `source` and `text` are now
  mutually exclusive. In text mode, pass an explicit `document_id` (or
  let it auto-generate). `text + loader` is rejected (loader was
  silently ignored before). `text` with a `list[str]` source is rejected.
- **Batch `ingest(list_of_sources)` returns `list[IngestionResult | Exception]`**
  instead of raising on first failure. Per-source errors are captured in
  the result list and logged at WARNING; the rest of the batch continues.
  Callers must inspect each entry; for fail-fast semantics, raise on the
  first `Exception` in the list.
- **`finalize()` / `finalize_sync()` return `FinalizeResult`** (typed
  Pydantic model) instead of `dict[str, Any]`. Replace
  `result["entities_deduplicated"]` with `result.entities_deduplicated`.

#### Non-breaking

- `_RAG_PROMPT` and the default system prompt updated for prompt
  injection hardening (see Security above).
- Sync wrappers (`retrieve_sync`, `completion_sync`, `ingest_sync`)
  re-declared with explicit kwargs mirroring their async counterparts;
  `**kwargs: Any` removed. IDE autocomplete and mypy strict mode now
  enforce kwarg names.
- `FixedSizeChunking` docstring surfaces the GraphRAG-Bench
  `chunk_size=1500, chunk_overlap=200` configuration as a documented
  trade-off; default remains 1000.
- `GraphRAG.__aenter__` / `__aexit__` documented as single-entry
  (non-reentrant). `__aexit__` close-error log message clarifies that
  the inner exception still propagates.
- `_rewrite_question_with_history` now logs a WARNING with full
  traceback when rewrite fails, so unexpected errors surface in
  operator logs while the function's "never raises" contract holds.

### Removed

- **`GraphRAG.query()` and `GraphRAG.query_sync()`** — deprecated in
  v1.0.0, removed entirely. Use `completion()` / `completion_sync()`
  for full RAG or `retrieve()` / `retrieve_sync()` for retrieval-only.

### Fixed

- `examples/02_pdf_with_schema.py` now calls `await rag.finalize()`
  after ingestion, matching the other examples.
- `examples/04_custom_provider.py` — the stub `MyCustomEmbedder.embed_query`
  body now raises `NotImplementedError` instead of returning a zero
  vector, with a prominent docstring warning. Users copying the
  example as a template now get a clear error instead of a silently
  broken graph.
- README quickstart updated for the `ingest()` API change and to use
  `openai/gpt-4o` instead of the non-existent `openai/gpt-5.4`.

### Migration

```python
# Storage access
rag.graph_store.get_statistics()  →  rag.get_statistics()
rag.graph_store.delete_all()      →  rag.delete_all()

# Batch ingest concurrency
await rag.ingest(sources, max_concurrent=5)
  →  await rag.ingest(sources, max_concurrency=5)

# Text-mode ingest
await rag.ingest("doc-id", text="...")
  →  await rag.ingest(text="...", document_id="doc-id")

# Batch result handling
results = await rag.ingest(["a.txt", "b.txt"])
for r in results:
    # r is now IngestionResult or Exception
    if isinstance(r, Exception):
        ...  # handle failure
    else:
        ...  # use r.nodes_created, etc.

# finalize() return access
result["entities_deduplicated"]  →  result.entities_deduplicated

# Removed methods
await rag.query(q)         →  await rag.completion(q)  # or rag.retrieve(q)
rag.query_sync(q)          →  rag.completion_sync(q)   # or rag.retrieve_sync(q)
```

For deployments using `rediss://` URLs: TLS now actually engages
(previously silently downgraded to plaintext). Verify your
FalkorDB/Redis endpoint accepts TLS before upgrading.

### Internal

- 33 new tests covering each new validation path, helper, and
  contract. Total unit test count: 615 (was 582 in v1.0.0).
- New helper `summarize_exception()` in
  `graphrag_sdk.core.providers._retry`.
- New `_neutralize_context_close_tag()` in `graphrag_sdk.api.main`.

## [1.0.0] - 2026-04-21

First stable release of the v1.0 rewrite. `pip install graphrag-sdk` now resolves
to this version by default. Legacy v0.x users can pin `graphrag-sdk==0.8.2`.

### Added

- **9-step ingestion pipeline**: Load, Chunk, Lexical Graph, Extract, Prune, Resolve, Write, Mentions (parallel), Index Chunks (parallel).
- **Multi-path retrieval**: Entity discovery via vector + fulltext search, relationship expansion, chunk retrieval, cosine reranking, and LLM-based answer generation.
- **Strategy pattern**: Swappable algorithms for every pipeline concern — chunking, extraction, resolution, retrieval, and reranking — each behind an abstract base class.
- **GraphExtraction strategy**: Two-step extraction using GLiNER2 for entity recognition (step 1) and LLM for relationship extraction and verification (step 2).
- **Resolution strategies**: ExactMatchResolution, DescriptionMergeResolution (LLM-assisted), SemanticResolution (embedding-based), and LLMVerifiedResolution.
- **LiteLLM provider**: Supports Azure OpenAI, OpenAI, Anthropic, Cohere, and 100+ LLM providers.
- **OpenRouter provider**: Alternative LLM/embedder provider via OpenRouter API.
- **PDF ingestion**: `PdfLoader` for processing PDF documents.
- **Entity deduplication**: `finalize()` post-ingestion step for dedup, embedding backfill, and index creation.
- **Circuit breaker**: Resilient FalkorDB connection with automatic failure detection and recovery.
- **Multi-tenant support**: `Context` with tenant isolation, distributed tracing, and latency budgeting.
- **Parallel multi-source ingestion**: `ingest()` accepts `str | list[str]` with `max_concurrent` parameter for bounded parallel ingestion.
- **Retrieve/completion split**: `retrieve()` for retrieval-only (no LLM call); `completion()` for full RAG pipeline with conversation history support.
- **Native multi-turn conversations**: `completion(history=[...])` passes messages natively to the LLM provider's chat API (not string-stuffed into a single prompt). History accepts `ChatMessage` objects or `{"role": ..., "content": ...}` dicts with validated roles (`system`, `user`, `assistant`).
- **`ChatMessage` model**: Pydantic-validated message type with `role: Literal["system", "user", "assistant"]` and `content: str`. Exported from the top-level package.
- **`LLMInterface.ainvoke_messages()`**: New method for multi-turn message-based LLM calls. Default implementation falls back to `ainvoke()` (string concatenation), so custom providers work without changes. `LiteLLM` and `OpenRouterLLM` override with native implementations.
- **Graph config node**: `__GraphRAGConfig__` singleton stores the embedding model and dimension used to build the graph; mismatches are caught on retrieval.
- **Embedder.model_name**: Abstract property on the `Embedder` ABC for identifying the embedding model.
- **556 tests**: Comprehensive unit and integration test suite with mock providers.
- **Full documentation**: Architecture, strategies, configuration, providers, benchmark, and API reference.
- **4 examples**: Quickstart, PDF with schema, custom strategies, and custom provider.

### Changed

- `query()` is deprecated — use `retrieve()` for retrieval-only or `completion()` for full RAG.
- `query_sync()` is deprecated — use `retrieve_sync()` or `completion_sync()`.
- `ConnectionConfig.password` field is hidden from `repr()` output.
- Dependency version bounds tightened: `numpy<3`, `scipy<2`, `falkordb<2`, `gliner<1`.
- `pypdf` minimum bumped to `>=6.9.2` (CVE fixes).
- Development status classifier changed from Alpha to Production/Stable.

### Fixed

- `hnswlib` import guard in SemanticResolution and LLMVerifiedResolution — raises clear `ImportError` instead of `AttributeError` when hnswlib is not installed.
- 14 ruff lint errors (import sorting, line length) resolved; CI no longer ignores lint rules.
