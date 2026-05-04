# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
