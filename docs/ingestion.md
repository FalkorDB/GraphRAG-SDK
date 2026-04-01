# Ingestion — How Documents Become a Knowledge Graph

When you call `rag.ingest("document.txt")`, the SDK transforms your raw text into a structured knowledge graph through a **9-step sequential pipeline**. Think of it as an assembly line: each step takes the output of the previous one, refines it, and passes it forward.

This document explains what each step does, why it exists, and how to tune it.

---

## The Big Picture

```
                         Your Document
                              |
                    ┌─────────┴─────────┐
                    v                    v
              1. Load Text         (or pass text= directly)
                    |
              2. Chunk Text
                    |
              3. Build Lexical Graph
              (Document + Chunk nodes, provenance edges)
                    |
              4. Extract Entities & Relationships
                    |
              4b. Quality Filter
              (remove bad nodes/dangling edges)
                    |
              5. Prune Against Schema
              (keep only schema-conforming data)
                    |
              6. Resolve Duplicates
              (merge same-entity mentions)
                    |
              7. Write to Graph
              (batched MERGE into FalkorDB)
                    |
         ┌─────────┴─────────┐
         v                    v
   8. Write Mentions    9. Index Chunks
   (MENTIONED_IN edges)  (embed + store vectors)
         └─────────┬─────────┘
                   v
            Done — IngestionResult
```

Steps 1-7 run **sequentially** (each depends on the previous). Steps 8-9 run **in parallel** since they're independent.

---

## Step-by-Step Explanation

### Step 1 — Load

**What it does:** Reads raw text from a file, URL, or string.

**How:** The `LoaderStrategy` ABC handles this. The SDK auto-detects the loader based on file extension:
- `.pdf` files use `PdfLoader`
- Everything else uses `TextLoader`
- If you pass `text=` directly, the loader step is skipped entirely

**Output:** `DocumentOutput` containing the raw text and a `DocumentInfo` with a unique ID and file path.

**Code:** `LoaderStrategy.load()` in [`ingestion/loaders/base.py`](../graphrag_sdk/src/graphrag_sdk/ingestion/loaders/base.py)

---

### Step 2 — Chunk

**What it does:** Splits the document text into smaller overlapping windows called chunks. Each chunk is small enough for the LLM to process, but large enough to contain meaningful context.

**How:** The default `FixedSizeChunking` uses a sliding window:
- Window size: 1000 characters (configurable)
- Overlap: 100 characters between consecutive chunks
- Step size: `chunk_size - chunk_overlap` = 900 characters

**Why overlap?** Without overlap, an entity mentioned right at the boundary between two chunks might be split across them and lost. Overlap ensures entities near boundaries appear in at least one complete chunk.

**Output:** `TextChunks` — a list of `TextChunk` objects, each with a unique ID (`uid`), the text content, and an index number.

**Code:** `ChunkingStrategy.chunk()` in [`ingestion/chunking_strategies/base.py`](../graphrag_sdk/src/graphrag_sdk/ingestion/chunking_strategies/base.py)

---

### Step 3 — Build Lexical Graph (Mandatory)

**What it does:** Creates the provenance backbone of the knowledge graph — this is how every answer traces back to its source document.

**Creates:**
- 1 **Document** node (with the file path and metadata)
- N **Chunk** nodes (one per text chunk, storing the chunk text and index)
- N **PART_OF** edges (Document → each Chunk)
- N-1 **NEXT_CHUNK** edges (Chunk → next Chunk, preserving reading order)

**The result looks like:**
```
Document
├── PART_OF → Chunk 0
├── PART_OF → Chunk 1
└── PART_OF → Chunk 2

Chunk 0 ──NEXT_CHUNK──> Chunk 1 ──NEXT_CHUNK──> Chunk 2
```

**Why mandatory?** This is the Zero-Loss Data principle — every piece of source material is traceable in the graph. When the retrieval system finds a chunk, it can always trace back to the source document.

**Code:** `IngestionPipeline._build_lexical_graph()` in [`ingestion/pipeline.py`](../graphrag_sdk/src/graphrag_sdk/ingestion/pipeline.py)

---

### Step 4 — Extract Entities & Relationships

**What it does:** The most important step — an LLM reads each chunk and extracts structured knowledge: entities (people, places, organizations, etc.) and the relationships between them.

**How:** The default `GraphExtraction` strategy uses a 2-step process:

1. **Step 1 (NER):** A pluggable entity extractor identifies entities in the text. Default: GLiNER (a local transformer model, no API calls needed). Alternative: `LLMExtractor` (uses the LLM for NER).

2. **Step 2 (Verify + Relationships):** The LLM receives the pre-extracted entities and the original text. It verifies the entities (fixing errors, adding missed ones) and extracts all relationships between them.

For a detailed explanation of the extraction process, see [extraction.md](extraction.md).

**Output:** `GraphData` containing nodes (entities), relationships, and mention records.

**Code:** `ExtractionStrategy.extract()` in [`ingestion/extraction_strategies/base.py`](../graphrag_sdk/src/graphrag_sdk/ingestion/extraction_strategies/base.py)

---

### Step 4b — Quality Filter

**What it does:** Removes bad data that slipped through extraction — nodes with empty or `None` IDs, and relationships whose endpoints don't exist.

**Why:** LLMs sometimes produce malformed output (empty entity names, references to entities that weren't extracted). This step catches those before they reach the graph.

**Code:** `IngestionPipeline._filter_quality()` in [`ingestion/pipeline.py`](../graphrag_sdk/src/graphrag_sdk/ingestion/pipeline.py)

---

### Step 5 — Prune Against Schema

**What it does:** Filters extracted data to only keep entities and relationships that match your schema definition.

**How it works:**
- If your schema defines entity types (e.g., Person, Organization, Location), only entities with those labels pass through
- If your schema defines relationship types, only those relationship types pass through
- Relationships whose endpoints were pruned are also removed
- **Special cases:** `"Unknown"` entities (low-confidence NER) and `"RELATES"` edges (the unified relationship type) always pass through

**Open schema mode:** If you define no entity or relationship types (empty `GraphSchema()`), this step is skipped entirely — everything passes through.

**Code:** `IngestionPipeline._prune()` in [`ingestion/pipeline.py`](../graphrag_sdk/src/graphrag_sdk/ingestion/pipeline.py)

---

### Step 6 — Resolve Duplicates

**What it does:** Merges entities that refer to the same real-world thing. When the LLM extracts "Alice" from chunk 1 and "Alice" from chunk 5, this step recognizes they're the same entity and merges them.

**Default: ExactMatchResolution**
- Groups entities by ID
- Keeps the first occurrence as the survivor
- Merges properties from duplicates into the survivor
- Remaps all relationship endpoints to the survivor
- Deduplicates relationships by `(start_id, type, end_id)`

**Alternative: DescriptionMergeResolution**
- Groups by `(normalized name, label)` — same name but different labels stay separate (e.g., Person "Paris" vs Location "Paris")
- Merges descriptions (concatenation or LLM summarization)
- Used in the benchmark-winning pipeline

For details on resolution strategies, see [strategies.md](strategies.md).

**Code:** `ResolutionStrategy.resolve()` in [`ingestion/resolution_strategies/base.py`](../graphrag_sdk/src/graphrag_sdk/ingestion/resolution_strategies/base.py)

---

### Step 7 — Write to Graph

**What it does:** Persists all the extracted and resolved data into FalkorDB using batched Cypher queries.

**How:**
- Nodes are written via `UNWIND $batch AS item MERGE (n:Label {id: item.id}) SET n += item.properties`
- Relationships are written similarly, using label hints for efficient MATCH operations
- Batch size: 500 items per query
- Entity nodes automatically get the `__Entity__` secondary label (structural nodes like Chunk and Document do not)

For details on the storage layer, see [storage.md](storage.md).

**Code:** `GraphStore.upsert_nodes()` and `GraphStore.upsert_relationships()` in [`storage/graph_store.py`](../graphrag_sdk/src/graphrag_sdk/storage/graph_store.py)

---

### Steps 8 & 9 — Mentions + Index Chunks (Parallel)

These two steps run simultaneously since they're independent:

#### Step 8 — Write Mentions

**What it does:** Creates `MENTIONED_IN` edges linking every entity to every chunk it was extracted from. These edges are critical for retrieval — they let the system find text passages for any entity.

**Details:** Uncapped — every entity-chunk pair gets an edge. Duplicates are deduplicated by `(entity_id, chunk_id)`.

#### Step 9 — Index Chunks

**What it does:** Embeds each chunk's text into a vector and stores it on the Chunk node. These embeddings power the vector similarity search during retrieval.

**How:**
1. Batch-embed all chunk texts in one API call (`aembed_documents`)
2. Write vectors to Chunk nodes via `SET c.embedding = vecf32(vector)`
3. Falls back to sequential embedding if the batch call fails

**Code:**
- Mentions: `IngestionPipeline._write_mentions()` in [`ingestion/pipeline.py`](../graphrag_sdk/src/graphrag_sdk/ingestion/pipeline.py)
- Chunk indexing: `VectorStore.index_chunks()` in [`storage/vector_store.py`](../graphrag_sdk/src/graphrag_sdk/storage/vector_store.py)

---

## Post-Ingestion: finalize()

After all documents are ingested, call `finalize()` to prepare the graph for querying. This is a separate step because some operations (like deduplication) work best when run globally across all documents, not per-document.

**`finalize()` runs 5 steps in order:**

| Step | What it does | Why |
|------|-------------|-----|
| 1. NULL cleanup | Removes entities with `NULL` names | Legacy edge cases from MERGE operations |
| 2. Deduplicate | Global exact-name entity dedup | "Alice" from doc 1 and doc 5 become one node |
| 3. Entity embeddings | Embeds entity names into vectors | Powers entity vector search during retrieval |
| 4. Relationship embeddings | Embeds fact text on RELATES edges | Powers RELATES edge vector search during retrieval |
| 5. Ensure indexes | Creates all 5 standard indexes | Vector + fulltext indexes for search |

```python
# After ingesting ALL documents:
stats = await rag.finalize()
# Returns: {
#     "null_stubs_removed": 0,
#     "entities_deduplicated": 142,
#     "entities_embedded": 3200,
#     "relationships_embedded": 8500,
#     "indexes": { "vector_Chunk": True, "vector___Entity__": True, ... }
# }
```

**Important:** Do not call `finalize()` after each document — call it once after all ingestion is complete. Entity backfill re-scans all entities and is slow when called repeatedly.

---

## Configuration Quick Reference

### Chunking

| Parameter | Default | Benchmark Value | Effect |
|-----------|---------|-----------------|--------|
| `chunk_size` | 1000 | 1500 | Larger = richer extraction context, more LLM tokens |
| `chunk_overlap` | 100 | 200 | Larger = fewer boundary losses, more redundancy |

### Extraction

| Parameter | Default | Effect |
|-----------|---------|--------|
| `entity_extractor` | `GLiNERExtractor()` | Local NER model (fast, no API calls) |
| `coref_resolver` | `None` | Optional coreference resolution (resolves pronouns) |
| `entity_types` | 11 defaults | Custom ontology for your domain |
| `max_concurrency` | LLM default (12) | Parallel LLM calls during extraction |

### Resolution

| Strategy | When to use |
|----------|-------------|
| `ExactMatchResolution` (default) | Fast, deterministic. Good when extraction is consistent |
| `DescriptionMergeResolution` | Multi-document ingestion. Used in the benchmark pipeline |

---

## Performance Notes

- **Slowest step:** Extraction (Step 4) — involves LLM calls for every chunk. Expect ~2-5 seconds per chunk.
- **Fastest step:** Quality filter, prune, and resolve — all in-memory, sub-second.
- **Parallelism:** Steps 8-9 run in parallel. Step 1 NER uses a semaphore (default 12 concurrent calls).
- **Batch size:** The benchmark uses 1500-character chunks. 20 documents (~4.7 MB total) take ~47 minutes to ingest.

---

## File Reference

| File | What it contains |
|------|-----------------|
| [`pipeline.py`](../graphrag_sdk/src/graphrag_sdk/ingestion/pipeline.py) | The 9-step pipeline orchestrator |
| [`api/main.py`](../graphrag_sdk/src/graphrag_sdk/api/main.py) | `GraphRAG.ingest()` and `finalize()` — user-facing API |
| [`loaders/text_loader.py`](../graphrag_sdk/src/graphrag_sdk/ingestion/loaders/text_loader.py) | TextLoader implementation |
| [`loaders/pdf_loader.py`](../graphrag_sdk/src/graphrag_sdk/ingestion/loaders/pdf_loader.py) | PdfLoader implementation |
| [`chunking_strategies/fixed_size.py`](../graphrag_sdk/src/graphrag_sdk/ingestion/chunking_strategies/fixed_size.py) | FixedSizeChunking implementation |
| [`extraction_strategies/graph_extraction.py`](../graphrag_sdk/src/graphrag_sdk/ingestion/extraction_strategies/graph_extraction.py) | 2-step extraction (NER + LLM verify/rels) |
| [`resolution_strategies/exact_match.py`](../graphrag_sdk/src/graphrag_sdk/ingestion/resolution_strategies/exact_match.py) | ExactMatchResolution |
| [`resolution_strategies/description_merge.py`](../graphrag_sdk/src/graphrag_sdk/ingestion/resolution_strategies/description_merge.py) | DescriptionMergeResolution |
| [`storage/graph_store.py`](../graphrag_sdk/src/graphrag_sdk/storage/graph_store.py) | Batched node/relationship writes |
| [`storage/vector_store.py`](../graphrag_sdk/src/graphrag_sdk/storage/vector_store.py) | Chunk embedding + indexing |
