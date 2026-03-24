# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0a1] - 2026-03-24

### Added

- **9-step ingestion pipeline**: Load, Chunk, Lexical Graph, Extract, Prune, Resolve, Write, Mentions (parallel), Index Chunks (parallel).
- **Multi-path retrieval**: Entity discovery via vector + fulltext search, relationship expansion, chunk retrieval, cosine reranking, and LLM-based answer generation.
- **Strategy pattern**: Swappable algorithms for every pipeline concern — chunking, extraction, resolution, retrieval, and reranking — each behind an abstract base class.
- **GraphExtraction strategy**: Two-step extraction using GLiNER2 for entity recognition (step 1) and LLM for relationship extraction and verification (step 2).
- **Resolution strategies**: ExactMatchResolution, DescriptionMergeResolution (LLM-assisted), and SemanticResolution (embedding-based).
- **LiteLLM provider**: Supports Azure OpenAI, OpenAI, Anthropic, Cohere, and 100+ LLM providers.
- **OpenRouter provider**: Alternative LLM/embedder provider via OpenRouter API.
- **PDF ingestion**: `PdfLoader` for processing PDF documents.
- **Entity deduplication**: `finalize()` post-ingestion step for dedup, embedding backfill, and index creation.
- **Circuit breaker**: Resilient FalkorDB connection with automatic failure detection and recovery.
- **Multi-tenant support**: `Context` with tenant isolation, distributed tracing, and latency budgeting.
- **491 tests**: Comprehensive unit and integration test suite with mock providers.
- **Full documentation**: Architecture, strategies, configuration, providers, benchmark, and API reference.
- **4 examples**: Quickstart, PDF with schema, custom strategies, and custom provider.
