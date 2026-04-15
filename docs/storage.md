# Storage — How Data Lives in FalkorDB

The storage layer is the bridge between the SDK's Python objects and FalkorDB's graph database. Two classes handle everything: **GraphStore** manages node and relationship writes, and **VectorStore** manages embeddings, indexes, and search. A third class, **EntityDeduplicator**, handles post-ingestion entity merging.

This document explains how each one works, what Cypher queries they generate, and how to tune them.

---

## The Big Picture

```
SDK Python Objects                    FalkorDB
──────────────────                    ────────
GraphNode          ───GraphStore───>  (:Person {id, name, ...})
GraphRelationship  ───GraphStore───>  ()-[:RELATES {fact, ...}]->()

TextChunks         ───VectorStore──>  (:Chunk {embedding: vecf32(...)})
                   ───VectorStore──>  Vector + Fulltext Indexes

Query vectors      <──VectorStore───  db.idx.vector.queryNodes(...)
Keyword queries    <──VectorStore───  db.idx.fulltext.queryNodes(...)

Duplicate entities ─EntityDeduplicator─> DETACH DELETE + edge remap
```

---

## GraphStore — Writing Nodes and Relationships

`GraphStore` is the single write path for all graph data. It uses parameterized Cypher to prevent injection and batched `UNWIND` for performance.

### Batched UNWIND MERGE

Nodes and relationships are written in batches of **500 items** per query. The pattern:

```cypher
-- Nodes
UNWIND $batch AS item
MERGE (n:`Person` {id: item.id})
SET n += item.properties
SET n:__Entity__

-- Relationships (with label hints)
UNWIND $batch AS item
MATCH (a:`__Entity__` {id: item.start_id}), (b:`__Entity__` {id: item.end_id})
MERGE (a)-[r:`RELATES`]->(b)
SET r += item.properties
```

**Why MERGE, not CREATE?** MERGE is idempotent — if a node or relationship already exists, it updates the properties rather than creating a duplicate. This makes re-ingestion safe.

### Label Hints

Relationship MATCH queries use **label hints** to speed up endpoint lookups. Instead of searching all nodes, FalkorDB only looks at nodes with the specified label:

| Edge Type | Source Label | Target Label |
|-----------|-------------|-------------|
| `PART_OF` | `Document` | `Chunk` |
| `NEXT_CHUNK` | `Chunk` | `Chunk` |
| `MENTIONED_IN` | `__Entity__` | `Chunk` |
| `RELATES` | `__Entity__` | `__Entity__` |

Unknown edge types default to `(__Entity__, __Entity__)`.

### Per-Item Fallback

If a batch upsert fails (e.g., a single malformed property causes the whole batch to error), GraphStore falls back to **per-item upserts**. The behavior differs slightly: for **nodes**, the first per-item failure raises a `DatabaseError` (remaining items in that batch are not attempted); for **relationships**, failures are logged as warnings and processing continues through the batch.

### None-ID Guard

Before writing, nodes with `None` or empty IDs are filtered out. These come from bad LLM extraction (the LLM sometimes returns entities without proper names). The guard prevents phantom nodes from polluting the graph.

### Property Cleaning

All properties go through `_clean_properties()` before writing:

| Input Type | Treatment |
|-----------|-----------|
| `str`, `int`, `float`, `bool` | Stored as-is |
| `list` | Items filtered to primitives only, empty lists dropped |
| `dict` | Serialized to JSON string |
| `None` | Dropped entirely |
| Other | Converted to string |

### Cypher Safety

Node labels and relationship types are sanitized via `sanitize_cypher_label()` to prevent Cypher injection. Properties are applied via parameter maps (e.g., `SET n += item.properties`), so property keys and values are never interpolated into the Cypher string.

---

## VectorStore — Embeddings, Indexes, and Search

`VectorStore` handles everything related to vector and fulltext operations.

### Index Creation

**Vector indexes** use FalkorDB's native vector index syntax:

```
CREATE VECTOR INDEX FOR (n:Chunk) ON (n.embedding)
OPTIONS {dimension:1536, similarityFunction:'cosine'}
```

**Fulltext indexes** use the RediSearch-based fulltext API:

```
CALL db.idx.fulltext.createNodeIndex('Chunk', 'text')
CALL db.idx.fulltext.createNodeIndex('__Entity__', 'name', 'description')
```

**Relationship vector indexes** use the edge index syntax:

```
CREATE VECTOR INDEX FOR ()-[e:`RELATES`]->() ON (e.embedding)
OPTIONS {dimension:1536, similarityFunction:'cosine'}
```

All index creation is idempotent — if the index already exists, the error is silently caught and logged at debug level.

### Chunk Indexing (index_chunks)

When chunks are ingested, their text is embedded and stored:

1. **Batch embed:** All chunk texts are passed to the embedder via a single `aembed_documents` call. The underlying provider controls how these are internally batched (e.g., via a configurable `batch_size`) and may split them across multiple API requests.
2. **Batch write:** Vectors are written to Chunk nodes using UNWIND (500 per batch):
   ```cypher
   UNWIND $batch AS item
   MATCH (c:Chunk {id: item.chunk_id})
   SET c.embedding = vecf32(item.vector)
   ```
3. **Fallback:** If batch embedding fails, chunks are embedded one at a time. If batch writing fails, items are written individually.

### Entity Embedding Backfill (backfill_entity_embeddings)

After all documents are ingested, entity nodes need embeddings for vector search. This is done during `finalize()`:

1. **Query:** Find entities missing embeddings: `WHERE e.embedding IS NULL`
2. **Embed:** Batch-embed entity names via `aembed_documents`
3. **Write:** Store vectors using UNWIND:
   ```cypher
   UNWIND $batch AS item
   MATCH (e:__Entity__ {id: item.eid})
   SET e.embedding = vecf32(item.vector)
   ```
4. **Loop:** Repeat until no more entities with NULL embeddings remain. Each batch naturally returns the next un-embedded set.

### Relationship Embedding (embed_relationships)

RELATES edges with a `fact` property but no embedding are batch-embedded:

1. **Query:** Find edges with `r.embedding IS NULL AND r.fact IS NOT NULL`
2. **Embed:** Batch-embed the `fact` text
3. **Write:** Store vectors on each edge individually (using internal FalkorDB edge IDs):
   ```cypher
   MATCH ()-[r:RELATES]->()
   WHERE id(r) = $rid
   SET r.embedding = vecf32($vector)
   ```

### Search Methods

**Vector search on Chunk nodes:**
```cypher
CALL db.idx.vector.queryNodes('Chunk', 'embedding', $top_k, vecf32($vector))
YIELD node, score
RETURN node.id AS id, node.text AS text, score
ORDER BY score DESC
```

**Vector search on Entity nodes:**
```cypher
CALL db.idx.vector.queryNodes('__Entity__', 'embedding', $top_k, vecf32($vector))
YIELD node, score
RETURN node.id AS id, node.name AS name, node.description AS description, score
ORDER BY score DESC
```

**Vector search on RELATES edges (with fallback):**
```cypher
-- Primary (FalkorDB >= 4.2):
CALL db.idx.vector.queryRelationships('RELATES', 'embedding', $top_k, vecf32($vector))
YIELD relationship AS r, score
RETURN r.src_name, r.rel_type, r.tgt_name, r.fact, score

-- Fallback (Cypher-based cosine distance scan):
MATCH (a:__Entity__)-[r:RELATES]->(b:__Entity__)
WHERE r.embedding IS NOT NULL
WITH a, r, b, vecf32.distance.cosine(r.embedding, vecf32($vector)) AS dist
RETURN r.src_name, r.rel_type, r.tgt_name, r.fact, (1-dist) AS score
ORDER BY dist ASC LIMIT $top_k
```

**Fulltext search:**
```cypher
CALL db.idx.fulltext.queryNodes('Chunk', $query_text)
YIELD node, score
RETURN node.id AS id, node.text AS text, score
ORDER BY score DESC LIMIT $top_k
```

Special characters in fulltext queries are escaped for RediSearch compatibility (commas, brackets, operators, etc.).

### Stored Embedding Optimization

During retrieval, the `rerank_chunks()` function uses **stored embeddings** when possible. Instead of re-embedding all candidate chunks (which would require an expensive API call), it fetches the vectors already stored on Chunk nodes and computes cosine similarity locally. This makes reranking instant when stored embedding coverage is >= 90%.

### ensure_indices()

Creates all 5 standard indexes in one call. Tracks state internally (`_indices_ensured`) to avoid redundant creation. Called automatically after each `ingest()` call, and re-run during `finalize()` (which resets the flag).

---

## EntityDeduplicator — Merging Duplicate Entities

After ingesting multiple documents, the same real-world entity might exist as multiple nodes (e.g., "Alice" from doc 1 and "Alice" from doc 5). The deduplicator merges them.

### Phase 1: Exact Name Match (Always Runs)

1. **Fetch** all `__Entity__` nodes with their primary label (the non-`__Entity__` label)
2. **Group** by `(normalized_name.lower(), label)` — grouping by label prevents cross-type merging (Person "Paris" and Location "Paris" stay separate)
3. **For each group with duplicates:**
   - **Survivor:** The entity with the longest description
   - **Remap edges:** All RELATES and MENTIONED_IN edges from duplicates are redirected to the survivor:
     ```cypher
     -- Outgoing RELATES
     MATCH (dup:__Entity__ {id: $dup_id})-[r:RELATES]->(b:__Entity__)
     WHERE b.id <> $survivor_id
     MERGE (s:__Entity__ {id: $survivor_id})-[nr:RELATES]->(b)
     SET nr += properties(r)
     DELETE r

     -- Incoming RELATES
     MATCH (a:__Entity__)-[r:RELATES]->(dup:__Entity__ {id: $dup_id})
     WHERE a.id <> $survivor_id
     MERGE (a)-[nr:RELATES]->(s:__Entity__ {id: $survivor_id})
     SET nr += properties(r)
     DELETE r

     -- MENTIONED_IN
     MATCH (dup:__Entity__ {id: $dup_id})-[r:MENTIONED_IN]->(c:Chunk)
     MERGE (s:__Entity__ {id: $survivor_id})-[:MENTIONED_IN]->(c)
     DELETE r
     ```
   - **Delete** the duplicate node: `MATCH (e:__Entity__ {id: $dup_id}) DETACH DELETE e`

### Phase 2: Fuzzy Embedding Match (Optional)

Enabled via `fuzzy=True`. Catches near-duplicates that have slightly different names (e.g., "J. Doe" and "Jane Doe"):

1. **Fetch** all surviving entities with their labels
2. **Batch-embed** entity names
3. **Normalize** vectors and compute pairwise cosine similarity in blocks (1000 entities per block to avoid OOM)
4. **Merge** pairs above the similarity threshold (default: 0.9), but only within the same label (no cross-type merging)
5. **Remap and delete** as in Phase 1

---

## FalkorDB-Specific Notes

### Vector Storage

FalkorDB stores vectors as `vecf32` — a native 32-bit float vector type. All vectors in the SDK are stored via:
```cypher
SET n.embedding = vecf32($vector)
```

### Vector Search API

Node vector search uses 4 arguments:
```
db.idx.vector.queryNodes('Label', 'property', $top_k, vecf32($vector))
```

Relationship vector search (FalkorDB >= 4.2):
```
db.idx.vector.queryRelationships('RELATES', 'embedding', $top_k, vecf32($vector))
```

### Graph Deletion

For fast graph deletion, use `GRAPH.DELETE` (the Redis-level command):
```python
await rag.graph_store.delete_all()  # Uses GRAPH.DELETE internally
```
This is much faster than `MATCH (n) DETACH DELETE n` on large graphs.

### Retry Behavior

The connection layer retries transient query failures up to 3 times using exponential backoff with jitter (`retry_delay * 2^attempt * random(0.5, 1.5)`) and employs a circuit breaker to short-circuit repeated failures. Non-transient errors (containing "already indexed", "already exists", or "unknown index") are raised immediately.

---

## File Reference

| File | What it contains |
|------|-----------------|
| [`storage/graph_store.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/src/graphrag_sdk/storage/graph_store.py) | GraphStore — batched MERGE, label hints, statistics, cleanup |
| [`storage/vector_store.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/src/graphrag_sdk/storage/vector_store.py) | VectorStore — index management, chunk/entity/relationship embedding, search |
| [`storage/deduplicator.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/src/graphrag_sdk/storage/deduplicator.py) | EntityDeduplicator — exact + fuzzy dedup, edge remapping |
| [`utils/cypher.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/src/graphrag_sdk/utils/cypher.py) | sanitize_cypher_label() and other Cypher utilities |
| [`core/connection.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/src/graphrag_sdk/core/connection.py) | FalkorDBConnection — async client, connection pooling, retry |
