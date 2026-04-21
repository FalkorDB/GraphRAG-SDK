# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
