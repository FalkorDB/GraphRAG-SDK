# Architecture

GraphRAG SDK v2 follows a **strategy-based pipeline architecture**. Every algorithmic concern is an abstract base class (ABC) with swappable implementations. The system has two main flows: **ingestion** (document to knowledge graph) and **retrieval** (question to answer).

## Design Principles

- **Strategy Modularity** -- Every pipeline step is an ABC. Swap any implementation without touching other code.
- **Zero-Loss Data** -- Full provenance chain from raw text to graph nodes (Document -> Chunk -> Entity).
- **Production Latency** -- Async-first, connection pooling, batched writes, parallel pipeline steps.
- **Simplicity** -- Single entry point (`GraphRAG`), flat package structure, no meta-programming.

## Ingestion Pipeline

The ingestion pipeline transforms documents into a knowledge graph in 9 steps. Steps 1-7 run sequentially (each depends on the previous), steps 8-9 run in parallel.

```
                    Sequential                              Parallel
  ┌──────────────────────────────────────────┐    ┌──────────────────────┐
  │ 1.Load  2.Chunk  3.Lexical  4.Extract    │    │ 8. Mentions          │
  │ 5.Prune  6.Resolve  7.Write             │──>│ 9. Index Chunks      │
  └──────────────────────────────────────────┘    └──────────────────────┘
```

### Step-by-Step

| Step | Name | What it does | Strategy ABC |
|------|------|-------------|-------------|
| 1 | **Load** | Reads raw text from a source (file, URL, raw string) | `LoaderStrategy` |
| 2 | **Chunk** | Splits text into overlapping windows | `ChunkingStrategy` |
| 3 | **Lexical Graph** | Creates Document and Chunk nodes with PART_OF and NEXT_CHUNK edges (provenance chain) | Built-in |
| 4 | **Extract** | LLM extracts entities, relationships, and mentions from each chunk | `ExtractionStrategy` |
| 5 | **Prune** | Filters extracted data against the schema (removes off-schema entities/relationships) | Built-in |
| 6 | **Resolve** | Deduplicates entities (exact match or LLM-assisted description merge) | `ResolutionStrategy` |
| 7 | **Write** | Batched MERGE of nodes and relationships into FalkorDB | Built-in |
| 8 | **Mentions** | Writes MENTIONED_IN edges linking entities to their source chunks | Built-in (parallel) |
| 9 | **Index Chunks** | Embeds chunk text and stores vector embeddings on Chunk nodes | Built-in (parallel) |

### Data Flow

```
str (raw text)
  -> DocumentOutput (text + metadata)
    -> TextChunks (list of TextChunk with uid, text, index)
      -> GraphData (nodes + relationships extracted by LLM)
        -> ResolutionResult (deduplicated nodes + remapped relationships)
          -> FalkorDB (MERGE via GraphStore + vector index via VectorStore)
```

## Graph Schema

The knowledge graph contains these node types and edge types:

### Node Labels

| Label | Purpose | Has Embedding |
|-------|---------|--------------|
| `Document` | Source document metadata | No |
| `Chunk` | Text chunk (fragment of a document) | Yes |
| `__Entity__` | Secondary label on all extracted entities | Yes (after backfill) |
| `Person`, `Place`, etc. | Primary entity labels (schema-defined) | Via `__Entity__` |

### Edge Types

| Type | Source -> Target | Purpose |
|------|-----------------|---------|
| `PART_OF` | Document -> Chunk | Provenance: which document a chunk came from |
| `NEXT_CHUNK` | Chunk -> Chunk | Sequential ordering of chunks |
| `MENTIONED_IN` | Entity -> Chunk | Which chunks mention an entity |
| `RELATES` | Entity -> Entity | All extracted relationships (single edge type) |

All LLM-extracted relationships use the single `RELATES` edge type. The original relationship type (e.g. `WORKS_AT`, `LOCATED_IN`) is preserved as the `rel_type` property on the edge. A `fact` property stores a human-readable fact string for embedding.

### Post-ingestion: Entity Deduplication

Entity deduplication is handled post-ingestion via `deduplicate_entities()` rather than during the pipeline. This allows ingesting multiple documents independently, then deduplicating globally. The method groups entities by `(normalized name, label)` to prevent cross-type merging (e.g. Person "Paris" and Location "Paris" remain separate).

## Retrieval Flow

The retrieval system answers questions by searching the knowledge graph through multiple paths, then generating an answer with the LLM.

```
Question
  |
  v
Keyword Extraction (LLM)
  |
  v
Embed Question (single API call)
  |
  ├── RELATES Edge Vector Search ────────────────────────────┐
  │   -> fact strings + entity entry points                  │
  │                                                          │
  ├── 2-Path Entity Discovery ──────────────────────────────┤
  │   1. Cypher CONTAINS substring match                     │
  │   2. Fulltext search on entity index                     │
  │   + merge entities from RELATES vector search            │
  │                                                          v
  │                                              Top entities (scored + deduped)
  │                                                         |
  │                                              2-Hop Relationship Expansion
  │                                              (1-hop + 2-hop from top entities)
  │                                                         |
  ├── 4-Path Chunk Retrieval ───────────────────────────────┤
  │   1. Fulltext search in chunks                           │
  │   2. Vector search in chunks                             │
  │   3. MENTIONED_IN traversal (entity -> chunk)            │
  │   4. 2-hop: entity -> neighbor -> chunk                  │
  │                                                          v
  │                                              Candidate chunks (cosine reranked)
  │                                                         |
  └──────────────────────────────> Context Assembly
                                    (hint, entities, relationships, facts, passages)
                                          |
                                          v
                                    LLM Generation (RAG prompt)
                                          |
                                          v
                                       Answer
```

### MultiPathRetrieval Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `chunk_top_k` | 15 | Final chunks after reranking |
| `max_entities` | 30 | Total entity cap across all paths |
| `max_relationships` | 20 | Max relationships in context |
| `rel_top_k` | 15 | RELATES edge vector search results |
| `keyword_limit` | 10 | Max keywords extracted from question |

## Storage Layer

### GraphStore

All graph writes go through `GraphStore`, which provides:
- **Batched UNWIND upserts** -- 500 nodes/relationships per batch
- **Label hints** -- relationship MATCH queries use label hints (e.g., MENTIONED_IN matches `__Entity__` -> `Chunk`) for faster lookups
- **Per-item fallback** -- if a batch fails, retries individual items
- **None-id guard** -- filters out entities with None/empty IDs (bad LLM extraction)

### VectorStore

Vector operations go through `VectorStore`:
- **Index creation** -- `CREATE VECTOR INDEX` for Chunk and Entity embeddings, plus RELATES edge embeddings
- **Fulltext index** -- for keyword search on chunk text and entity names
- **Batched embedding** -- embeds texts in bulk, stores via UNWIND
- **Search** -- `db.idx.vector.queryNodes` for similarity search

## Extending the SDK

To add a custom strategy, subclass the relevant ABC:

```python
from graphrag_sdk import ChunkingStrategy, TextChunks
from graphrag_sdk.core.context import Context

class SemanticChunking(ChunkingStrategy):
    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        # Your implementation here
        ...
```

Then pass it to `ingest()`:

```python
await rag.ingest("doc.txt", chunker=SemanticChunking())
```

The same pattern applies to all 6 strategy ABCs: `LoaderStrategy`, `ChunkingStrategy`, `ExtractionStrategy`, `ResolutionStrategy`, `RetrievalStrategy`, `RerankingStrategy`.
