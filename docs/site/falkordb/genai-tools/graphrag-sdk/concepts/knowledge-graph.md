---
title: "Knowledge graph"
nav_order: 1
parent: "Concepts"
grand_parent: "GraphRAG-SDK"
description: "The entity / relation / chunk model GraphRAG-SDK writes into FalkorDB, and the provenance edges that make every answer traceable."
---

# Knowledge graph

The SDK writes three layers into the same FalkorDB graph:

```
┌────────── Text layer ──────────┐
│  Document → Chunk → Chunk → …  │
│              │  NEXT_CHUNK     │
│              ▼                  │
│       (raw source text)         │
└──────────────┬──────────────────┘
               │  MENTIONS
┌──────────────▼──────────────────┐
│  Entity layer                   │
│   :Person  :Organization  :Loc  │
│   …connected by typed edges    │
│   (WORKS_AT, LOCATED_IN, …)     │
└─────────────────────────────────┘
```

## Three node kinds

| Kind | Label(s) | Holds |
|---|---|---|
| Document | `:Document` | One per ingested source. Stores `path`, `content_hash` (SHA-256), `metadata`. |
| Chunk | `:Chunk` | Text shard produced by the chunker. Stores `text`, `index`, `embedding`. |
| Entity | User-defined (`:Person`, `:Organization`, …) | One per resolved entity. Stores `name`, `description`, any declared attributes, optional `embedding`. |

## Edges the SDK writes

| Edge | Direction | Meaning |
|---|---|---|
| `PART_OF` | `Chunk → Document` | Chunk belongs to source document. |
| `NEXT_CHUNK` | `Chunk → Chunk` | Sequential chunk order inside a document (for context-window expansion at retrieval time). |
| `MENTIONS` | `Chunk → Entity` | This chunk introduced or referenced this entity. **The provenance backbone.** |
| `RELATES` | `Entity → Entity` | Generic typed relation. The original relation type lives in `properties.rel_type`. Carries `source_chunk_ids` (union over every chunk that introduced this fact), `description`, and `embedding`. |
| User-typed | `Entity → Entity` | When you declare a typed relation in your ontology (e.g. `:WORKS_AT`), the extractor writes that as a first-class edge type. |

## Why a graph, not just vectors

Vector RAG retrieves *chunks*. Graph RAG retrieves chunks **and** entities **and** the typed relationships between them, which means:

- **Multi-hop questions answerable.** "Who manages the engineer who wrote the auth library?" is a single Cypher traversal; with vector RAG you'd hope the right chunks ended up near the question by similarity.
- **Aggregation answerable.** "How many people work at Acme?" — counting nodes by label is one query; vector RAG can't reliably count.
- **Cited answers.** Every entity carries `source_chunk_ids` and `MENTIONS` edges back to the chunks that introduced it. `completion(return_context=True)` returns that trail.
- **Deduplication.** "Alice Liddell" and "Alice L." can be merged into a single `:Person` node — resolution runs as part of `finalize()`. Vector RAG keeps them as two unrelated chunks.

## Provenance is mandatory, not optional

Every entity and relationship the SDK writes carries `source_chunk_ids` — the union of every chunk that introduced or restated the fact. Combined with `MENTIONS` edges, this lets you answer "where did we learn that?" for any node in the graph. The `completion()` method exposes this trail via `return_context=True`.

This is why the SDK insists on `finalize()` being called at the end of an ingestion batch — it's the step that runs cross-document deduplication and unions provenance lists across all the chunks that introduced the same surviving entity.

## See also

- [Concepts → Ontology](./ontology) — how you constrain which entity and relation types appear.
- [Concepts → Ingestion pipeline](./ingestion-pipeline) — how documents become this graph.
- [Concepts → Retrieval pipeline](./retrieval-pipeline) — how the graph is queried.
