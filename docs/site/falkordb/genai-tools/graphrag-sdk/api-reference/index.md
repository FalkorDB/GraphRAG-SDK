---
title: "API Reference"
nav_order: 5
parent: "GraphRAG-SDK"
grand_parent: "GenAI Tools"
has_children: true
description: "Complete API reference for GraphRAG-SDK — every public class and method with signature, parameters, returns, and raises."
---

# API Reference

Per-symbol reference for everything importable from `graphrag_sdk` and its public submodules. Modeled after the `developers.openai.com/api/docs` shape: signature → parameters → returns → raises → example.

## By page

| Page | Symbols |
|---|---|
| [GraphRAG](./graphrag) | `GraphRAG` — the facade. Constructor, ingestion, retrieval, completion, incremental updates, ontology evolution, schema discovery, statistics, lifecycle. |
| [Ontology](./ontology) | `Ontology`, `Entity`, `Relation`, `Attribute`. Allowed attribute types, reserved names. |
| [Discovery](./discovery) | `Catalog` ABC, `DBpediaCatalog`, `SchemaExtensionProposal`, `OntologyDiscoveryError`, `DBpediaFetchError`, `discover_ontology`, `discover_grounded`, `suggest_extensions`, `extract_with_retry`. |
| [Providers](./providers) | `LLMInterface`, `Embedder`, `LiteLLM`, `LiteLLMEmbedder`, `OpenRouterLLM`, `OpenRouterEmbedder`, `LLMBatchItem`. |
| [Connection](./connection) | `ConnectionConfig`, `FalkorDBConnection`. |
| [Ingestion strategies](./ingestion-strategies) | `ChunkingStrategy` and built-ins, `LoaderStrategy`, `ExtractionStrategy` and built-ins, `ResolutionStrategy` and built-ins, `IngestionPipeline`, `BackfillExecutor`. |
| [Retrieval strategies](./retrieval-strategies) | `RetrievalStrategy`, `MultiPathRetrieval`, `RerankingStrategy`, `CosineReranker`. |
| [Result types](./result-types) | `IngestionResult`, `UpdateResult`, `DeleteDocumentResult`, `ApplyChangesResult`, `BatchEntry`, `FinalizeResult`, `RagResult`, `RetrieverResult`, `RetrieverResultItem`, `IngestionResult`, `ResolutionResult`, `GraphData`, `EvolutionResult`, `BackfillResult`. |
| [Storage](./storage) | `GraphStore`, `OntologyStore`, `VectorStore`, `OntologyContradictionError`, `OntologyModificationNotAllowedError`. |
| [Exceptions](./exceptions) | `GraphRAGError`, `LatencyBudgetExceededError`, `DocumentNotFoundError`, `OntologyDiscoveryError`, `OntologyEvolutionError`, plus the LLM / Embedding / Ingestion / Retrieval / Storage subclasses. |

## Conventions

- Every method whose signature starts with `async def` must be awaited from within an event loop. Sync convenience wrappers (`*_sync`) are listed under [GraphRAG](./graphrag#sync-convenience-wrappers).
- Defaults shown in tables match the source. When a parameter is `*` -keyword-only in the signature, the table flags it as **keyword-only**.
- Parameters described as "**deprecated**" still work but emit `DeprecationWarning` and will be removed in a future major release.
