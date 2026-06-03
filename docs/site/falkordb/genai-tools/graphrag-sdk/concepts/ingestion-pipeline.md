---
title: "Ingestion pipeline"
nav_order: 4
parent: "Concepts"
grand_parent: "GraphRAG-SDK"
description: "Load → chunk → extract → resolve → write. The four pluggable stages that turn documents into a knowledge graph, and where to override each."
---

# Ingestion pipeline

Every call to `rag.ingest(...)` runs a four-stage pipeline. Each stage is governed by a strategy ABC — pass a custom implementation to override the default, leave the argument off to get the sensible default.

```
   source (path | text)
        │
        ▼
   ┌────────────┐
   │   LOAD     │  LoaderStrategy
   │            │  default: PdfLoader / MarkdownLoader / TextLoader by extension
   └─────┬──────┘
         │ DocumentOutput (text + structural elements)
         ▼
   ┌────────────┐
   │   CHUNK    │  ChunkingStrategy
   │            │  default: SentenceTokenCapChunking(max_tokens=512, overlap_sentences=2)
   └─────┬──────┘
         │ TextChunks
         ▼
   ┌────────────┐
   │  EXTRACT   │  ExtractionStrategy
   │            │  default: GraphExtraction (LLM, schema-aware)
   └─────┬──────┘
         │ GraphData (nodes + relationships + mentions)
         ▼
   ┌────────────┐
   │  RESOLVE   │  ResolutionStrategy
   │            │  default: ExactMatchResolution(resolve_property="id")
   └─────┬──────┘
         │ ResolutionResult
         ▼
   write into FalkorDB
```

## Load

A `LoaderStrategy.load()` reads raw bytes and returns a `DocumentOutput` with text plus optional structural elements (headers, lists, tables). The structural elements are used by structure-aware chunkers to avoid splitting mid-section.

Default loaders are auto-selected by file extension:

| Extension | Loader | Notes |
|---|---|---|
| `.txt` | `TextLoader` | Plain text passthrough. |
| `.md`, `.markdown` | `MarkdownLoader` | Parses headings and lists into structural elements. |
| `.pdf` | `PdfLoader` | Requires `graphrag-sdk[pdf]` (pypdf) or `[pdf-fast]` (PyMuPDF, table-aware, AGPL). |

For unsupported formats, write your own `LoaderStrategy` subclass and pass it as `loader=...` to `ingest()`. Or supply the text directly with `rag.ingest(text="...")` and skip the loader entirely.

## Chunk

A `ChunkingStrategy.chunk()` slices a `DocumentOutput` into `TextChunks`. The default — `SentenceTokenCapChunking(max_tokens=512, overlap_sentences=2)` — is **sentence-aware**: it never splits inside a sentence, and adjacent chunks overlap by two sentences so entity mentions are never severed across chunk boundaries.

Built-in chunkers:

| Strategy | When to use |
|---|---|
| `SentenceTokenCapChunking` (default) | General prose. Optimised against entity-name boundary splits. |
| `FixedSizeChunking(chunk_size, chunk_overlap)` | Character-window chunking. Predictable size; ignores sentence boundaries. |
| `ContextualChunking(...)` | Anthropic's contextual-retrieval approach — each chunk gets a one-line LLM-written summary prepended. Higher cost, better retrieval on long documents. |
| `CallableChunking(fn)` | Adapter for a plain function `(DocumentOutput) -> TextChunks`. |

## Extract

An `ExtractionStrategy.extract()` reads a chunk and produces `GraphData` — entities, relationships, and the `EntityMention` pairs that link them back to the chunk. The default — `GraphExtraction` — is a two-step LLM-driven extractor:

1. **Entity step.** Either a GLiNER NER pass (fast, local) or an LLM call, depending on `entity_extractor=`. Produces typed entity spans constrained by the ontology's declared labels.
2. **Relation step.** LLM call: "given these entities and this chunk, what are the typed relationships among them?" Constrained by the ontology's declared relations and their patterns.

`GraphExtraction` accepts an `EntityExtractor` (`GLiNERExtractor` or `LLMExtractor`) and an optional `CorefResolver` (e.g. `FastCorefResolver`) that re-anchors pronominal references before extraction.

To skip LLM extraction entirely — for example, when you already have entities from a database lookup — write a custom `ExtractionStrategy` and pass it as `extractor=...`.

## Resolve

A `ResolutionStrategy.resolve()` deduplicates entities that surfaced under different surface forms ("Alice L." and "Alice Liddell" → one `:Person`). It returns a `ResolutionResult` with a `remap: dict[str, str]` of merged-id → survivor-id; the pipeline rewrites `MENTIONS` edges through that remap so chunks point at the survivor.

Built-in resolvers:

| Strategy | Cost | Accuracy |
|---|---|---|
| `ExactMatchResolution` (default) | Zero — pure groupby on `(label, name)`. | Conservative — only collapses identical names. |
| `SemanticResolution` | One embedding per entity pair candidate. | Catches near-duplicates (`Acme Corp.` and `Acme Corporation`). |
| `LLMVerifiedResolution` | One LLM call per candidate pair above an embedding threshold. | Highest. Use when entities have rich descriptions that disambiguate. |
| `DescriptionMergeResolution` | One LLM call per surviving entity to merge descriptions. | Improves the entity-card text without changing membership. |

Multi-document corpora typically chain a fast resolver per-document with a final cross-document pass in `finalize()`. `finalize()` runs `EntityDeduplicator` automatically — that's the engine that unions provenance lists (`source_chunk_ids`) across all chunks that introduced the same surviving entity.

## finalize() — the single mandatory post-step

After all your `ingest()` calls, **call `finalize()` exactly once**. It does three things:

1. **Cross-document deduplication.** Runs `EntityDeduplicator` over the full entity table, merges remaining duplicates, unions provenance.
2. **Embedding backfill.** Computes embeddings for entities and relations that don't have them yet (chunk embeddings are written during ingestion; entity / relation embeddings are deferred to one batch here for throughput).
3. **Index creation.** Builds the vector indexes on entity / relation / chunk embeddings, full-text indexes on entity names, and any range indexes the retrieval strategies need.

The cost is **O(graph size)**, not O(changes). For CI workflows that ingest many documents per run, batch all your `ingest()`/`update()` calls and call `finalize()` once at the end — never per file.

## Concurrency

`ingest(list_of_sources, max_concurrency=3)` parallelises across sources. Inside each ingestion, LLM calls go through `LLMInterface.abatch_invoke()` with the provider's own concurrency cap. The defaults are conservative — bump `max_concurrency` and the LLM's `max_concurrency` if your provider rate limit and your wallet permit.

## See also

- [API Reference → GraphRAG](../api-reference/graphrag) — `ingest`, `finalize`, all parameters.
- [API Reference → Ingestion strategies](../api-reference/ingestion-strategies) — every chunker, extractor, and resolver.
- [Guides → Ingest PDF and Markdown](../guides/ingest-pdf-and-markdown) — runnable multi-format example.
