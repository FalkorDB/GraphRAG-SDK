# Architecture

GraphRAG SDK v2 follows a **strategy-based pipeline architecture**. Every algorithmic concern is an abstract base class (ABC) with swappable implementations. The system has two main flows: **ingestion** (document to knowledge graph) and **retrieval** (question to answer).

## Design Principles

- **Strategy Modularity** -- Every pipeline step is an ABC. Swap any implementation without touching other code.
- **Zero-Loss Data** -- Full provenance chain from raw text to graph nodes (Document -> Chunk -> Entity).
- **Production Latency** -- Async-first, connection pooling, batched writes, parallel pipeline steps.
- **Simplicity** -- Single entry point (`GraphRAG`), flat package structure, no meta-programming.

## Ingestion Pipeline

The ingestion pipeline transforms documents into a knowledge graph in 10 steps. Steps 1-7 run sequentially (each depends on the previous), steps 8-11 run in parallel.

```
                    Sequential                              Parallel
  ┌──────────────────────────────────────────┐    ┌──────────────────────┐
  │ 1.Load  2.Chunk  3.Lexical  4.Extract    │    │ 8. Index Facts       │
  │ 5.Prune  6.Resolve  7.Write             │──>│ 9. Synonymy          │
  │                                          │    │ 10. Mentions         │
  └──────────────────────────────────────────┘    │ 11. Index Chunks     │
                                                  └──────────────────────┘
```

### Step-by-Step

| Step | Name | What it does | Strategy ABC |
|------|------|-------------|-------------|
| 1 | **Load** | Reads raw text from a source (file, URL, raw string) | `LoaderStrategy` |
| 2 | **Chunk** | Splits text into overlapping windows | `ChunkingStrategy` |
| 3 | **Lexical Graph** | Creates Document and Chunk nodes with PART_OF and NEXT_CHUNK edges (provenance chain) | Built-in |
| 4 | **Extract** | LLM extracts entities, relationships, facts, and mentions from each chunk | `ExtractionStrategy` |
| 5 | **Prune** | Filters extracted data against the schema (removes off-schema entities/relationships) | Built-in |
| 6 | **Resolve** | Deduplicates entities (exact match or LLM-assisted description merge) | `ResolutionStrategy` |
| 7 | **Write** | Batched MERGE of nodes and relationships into FalkorDB | Built-in |
| 8 | **Index Facts** | Embeds fact triples and stores them as Fact nodes with vector embeddings | Built-in (parallel) |
| 9 | **Synonymy** | Detects synonym entities via embedding similarity, creates SYNONYM edges | Built-in (parallel, skippable) |
| 10 | **Mentions** | Writes MENTIONED_IN edges linking entities to their source chunks | Built-in (parallel) |
| 11 | **Index Chunks** | Embeds chunk text and stores vector embeddings on Chunk nodes | Built-in (parallel) |

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
| `Fact` | Subject-predicate-object triple | Yes |
| `__Entity__` | Secondary label on all extracted entities | Yes (after backfill) |
| `Person`, `Place`, etc. | Primary entity labels (schema-defined) | Via `__Entity__` |

### Edge Types

| Type | Source -> Target | Purpose |
|------|-----------------|---------|
| `PART_OF` | Chunk -> Document | Provenance: which document a chunk came from |
| `NEXT_CHUNK` | Chunk -> Chunk | Sequential ordering of chunks |
| `MENTIONED_IN` | Entity -> Chunk | Which chunks mention an entity |
| `SYNONYM` | Entity -> Entity | Synonym pairs (cosine similarity > threshold) |
| Schema-defined | Entity -> Entity | Extracted relationships (WORKS_AT, LOCATED_IN, etc.) |

## Retrieval Flow

The retrieval system answers questions by searching the knowledge graph through multiple paths, then generating an answer with the LLM.

```
Question
  |
  v
Keyword Extraction (LLM)
  |
  v
Batch Embedding (keywords + question)
  |
  ├── 5-Path Entity Discovery ──────────────────────────────┐
  │   1. Vector search per keyword                          │
  │   2. CONTAINS substring match                           │
  │   3. Fulltext search per keyword                        │
  │   4. Question vector similarity                         │
  │   5. Synonym expansion of found entities                │
  │                                                         v
  │                                              Top entities (scored + deduped)
  │                                                         |
  │                                              2-Hop Relationship Expansion
  │                                              (top 5 entities -> neighbors)
  │                                                         |
  ├── 5-Path Chunk Retrieval ───────────────────────────────┤
  │   1. Fulltext search in chunks                          │
  │   2. Vector search in chunks                            │
  │   3. MENTIONED_IN traversal (entity -> chunk)           │
  │   4. CONTAINS substring in chunks                       │
  │   5. 2-hop: entity -> neighbor -> chunk                 │
  │                                                         v
  │                                              Candidate chunks (cosine reranked)
  │                                                         |
  ├── Fact Retrieval (vector search on Fact nodes) ─────────┤
  │                                                         v
  └──────────────────────────────> Context Assembly
                                    (entities + relationships + facts + passages)
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
| `entity_top_k` | 5 | Max entities per discovery path |
| `chunk_top_k` | 15 | Final chunks after reranking |
| `fact_top_k` | 15 | Facts retrieved via vector search |
| `max_entities` | 30 | Total entity cap across all paths |
| `max_relationships` | 20 | Max relationships in context |
| `keyword_limit` | 10 | Max keywords extracted from question |
| `llm_rerank` | False | Enable LLM-based reranking (adds ~1s latency, marginal accuracy gain) |

## Storage Layer

### GraphStore

All graph writes go through `GraphStore`, which provides:
- **Batched UNWIND upserts** -- 500 nodes/relationships per batch
- **Label hints** -- relationship MATCH queries use label hints (e.g., MENTIONED_IN matches `__Entity__` -> `Chunk`) for faster lookups
- **Per-item fallback** -- if a batch fails, retries individual items
- **None-id guard** -- filters out entities with None/empty IDs (bad LLM extraction)

### VectorStore

Vector operations go through `VectorStore`:
- **Index creation** -- `CREATE VECTOR INDEX` for Chunk, Entity, and Fact embeddings
- **Fulltext index** -- for keyword search on chunk text
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
