---
title: "Retrieval pipeline"
nav_order: 5
parent: "Concepts"
grand_parent: "GraphRAG-SDK"
description: "How vector search, full-text search, LLM-generated Cypher, and multi-hop graph walks combine into one retrieval pass, and where to override each."
---

# Retrieval pipeline

`rag.retrieve(question)` and `rag.completion(question)` run the same retrieval pipeline. `completion` adds one extra step at the end — feeding the retrieved context to the LLM along with the question and a system prompt.

```
   question
      │
      ▼
   ┌─────────────────────────────────────────────────────┐
   │  RetrievalStrategy  (default: MultiPathRetrieval)   │
   │                                                      │
   │   ┌─ vector search ─────────────────────┐           │
   │   │  embed(question) → top-K chunks +    │           │
   │   │  top-K entities (cosine in HNSW)     │           │
   │   ├─ full-text search ──────────────────┤           │
   │   │  FalkorDB FT.SEARCH on entity names  │           │
   │   ├─ Cypher generation (experimental) ──┤           │
   │   │  LLM emits a Cypher query against    │           │
   │   │  the ontology; executed read-only    │           │
   │   ├─ relationship expansion ────────────┤           │
   │   │  for each seed entity, walk 1–2 hops │           │
   │   │  to gather connected context         │           │
   │   └──────────────────┬──────────────────┘           │
   │                       │                              │
   │                       ▼                              │
   │   reranker (default: CosineReranker)                 │
   │                       │                              │
   └───────────────────────┼──────────────────────────────┘
                           │
                           ▼
                   RetrieverResult  ─── completion() ──► LLM ──► RagResult
```

## Why combine multiple search modes

No single retrieval mode is best for every question shape:

- **Vector search** rewards semantic similarity but misses queries that hinge on rare proper nouns or numbers — those don't move much in embedding space.
- **Full-text search** nails proper nouns, dates, identifiers — but misses paraphrases and synonyms.
- **Cypher generation** is the only path for questions whose answer requires a structural operation: "How many people work at Acme?", "Who reports to the engineer who wrote module X?", "What are all the projects without an owner?"
- **Relationship expansion** turns a partial vector hit (one seeded entity) into a multi-hop neighbourhood — the kind of context vector RAG can't construct.

`MultiPathRetrieval` runs all four, deduplicates by chunk id, and reranks the union.

## What the reranker does

The default reranker — `CosineReranker` — re-scores every candidate passage by cosine similarity to the **question embedding** (not the chunk-vs-chunk similarity used during retrieval). This gives a single consistent score across passages that came from different retrieval modes, so the top-N has the best signal regardless of which path surfaced it.

Custom rerankers plug in behind the `RerankingStrategy` ABC — for example, an LLM-based reranker that pairwise-compares candidates.

## Cypher generation — what's covered

The Cypher-generation path is experimental and conservative:

- Only runs when the question shape suggests structural intent (aggregation, "how many", "all the X that …").
- The LLM sees the live ontology — labels, properties, relation patterns — so generated queries respect the schema.
- Queries are executed read-only against the data graph; the result rows are formatted as text and added to the retrieval set.
- Failures degrade gracefully — a malformed query is logged and skipped, never raised.

## Multi-turn conversations

`completion(history=[...])` accepts a list of `ChatMessage` objects. Before retrieval, a one-shot LLM call rewrites the latest question as a standalone question that resolves pronouns and references against the history. The rewritten question feeds the retrieval pipeline; the original conversation is included in the generation prompt.

This separation is important: retrieval is question-grounded, generation is conversation-grounded. Without rewriting, "where does he work?" retrieves nothing useful.

## Cited answers

Every entity carries `source_chunk_ids`. Every chunk carries `MENTIONS` edges to the entities it introduced. When you pass `return_context=True` to `completion()`, the result's `retriever_result.items` lists the chunks that supported the answer, with their scores. Combine with the entity provenance lists and you can build a per-fact citation trail.

## Sync convenience wrappers

`rag.retrieve_sync()`, `rag.completion_sync()`, `rag.ingest_sync()` exist for scripts that aren't async-native. They wrap the async methods with `asyncio.run()`. Don't use them inside an existing event loop — they'll raise `RuntimeError`.

## See also

- [API Reference → GraphRAG](../api-reference/graphrag) — `retrieve`, `completion`, `completion_sync`, parameters.
- [API Reference → Retrieval strategies](../api-reference/retrieval-strategies) — `MultiPathRetrieval`, `CosineReranker`, and the ABCs.
