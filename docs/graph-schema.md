# Graph Schema — How the Knowledge Graph is Structured

When you ingest documents into GraphRAG SDK, the system builds a property graph in FalkorDB. This document explains exactly what that graph looks like — what node types exist, what edges connect them, what properties they carry, and what indexes make it searchable.

Understanding the graph structure helps you write custom Cypher queries, debug ingestion quality, and tune retrieval.

---

## The Big Picture

```
┌──────────┐                   ┌──────────┐
│ Document │──PART_OF────────>│  Chunk   │
│          │                   │ (embed.) │
└──────────┘                   └─────┬────┘
                                     │
                              NEXT_CHUNK
                                     │
                               ┌─────v────┐
                               │  Chunk   │
                               │ (embed.) │
                               └──────────┘
                                     ^
                              MENTIONED_IN
                                     │
┌──────────┐                   ┌─────┴────┐
│  Person  │──RELATES────────>│  Org     │
│__Entity__│                   │__Entity__│
│ (embed.) │<──RELATES────────│ (embed.) │
└──────────┘                   └──────────┘
```

The graph has two layers:
1. **Lexical layer** — Document and Chunk nodes with provenance edges (the text you ingested)
2. **Knowledge layer** — Entity nodes with relationship edges (the structured knowledge extracted from that text)

The layers are connected by `MENTIONED_IN` edges, which link entities to the chunks where they were found.

---

## Node Labels

### Document

**Purpose:** Represents a source document that was ingested.

| Property | Type | Description |
|----------|------|-------------|
| `id` | `string` | Unique ID (auto-generated UUID) |
| `path` | `string` | Original file path or source identifier |
| Any metadata | varies | Custom metadata passed via `DocumentInfo` |

**Created in:** Step 3 (Lexical Graph) of the ingestion pipeline.

### Chunk

**Purpose:** A text fragment from a document. Chunks are the atomic unit of retrieval — when the system finds relevant information, it returns chunks.

| Property | Type | Description |
|----------|------|-------------|
| `id` | `string` | Unique ID (auto-generated UUID) |
| `text` | `string` | The chunk's text content |
| `index` | `integer` | Position within the document (0-based) |
| `embedding` | `vecf32` | Vector embedding for semantic search (added in Step 9) |
| `start_char` | `integer` | Start character offset in the source document (if `FixedSizeChunking`) |
| `end_char` | `integer` | End character offset |
| `chunk_size` | `integer` | Configured chunk size |
| `chunk_overlap` | `integer` | Configured overlap |

**Created in:** Step 3 (Lexical Graph). Embedding added in Step 9.

### Entity Nodes (Person, Organization, etc.)

**Purpose:** Extracted knowledge entities. Each entity node has **two labels**: its domain type (e.g., `Person`) and `__Entity__` (a secondary label shared by all extracted entities).

| Property | Type | Description |
|----------|------|-------------|
| `id` | `string` | Deterministic: `"name__type"` (e.g., `"alice__person"`) |
| `name` | `string` | Entity name as extracted |
| `description` | `string` | Rich description from LLM extraction |
| `source_chunk_ids` | `list[string]` | Chunks where this entity was found |
| `spans` | `string` (JSON) | Character offsets: `{chunk_id: [{start, end}]}` |
| `embedding` | `vecf32` | Vector embedding of the entity name (added during `finalize()`) |

**Created in:** Step 7 (Write). `__Entity__` label is automatically added. Embedding is backfilled during `finalize()`.

**Default entity types (11):** Person, Organization, Technology, Product, Location, Date, Event, Concept, Law, Dataset, Method

Entities below the NER confidence threshold get the special label `Unknown`.

---

## Edge Types

### PART_OF (Document -> Chunk)

**Purpose:** Provenance — tracks which document a chunk came from.

| Property | Type | Description |
|----------|------|-------------|
| `index` | `integer` | Chunk position within the document |

### NEXT_CHUNK (Chunk -> Chunk)

**Purpose:** Sequential ordering — preserves the reading order of chunks within a document. Used to fetch neighboring chunks for context expansion.

No additional properties.

### MENTIONED_IN (Entity -> Chunk)

**Purpose:** Co-occurrence — links entities to the chunks where they were extracted. This is a critical edge for retrieval: when you find an entity, you can traverse to its source chunks.

No additional properties. Deduplicated by `(entity_id, chunk_id)`.

### RELATES (Entity -> Entity)

**Purpose:** All extracted relationships between entities. This is the **only** relationship type used for knowledge edges.

| Property | Type | Description |
|----------|------|-------------|
| `rel_type` | `string` | Original relationship type (e.g., `"WORKS_AT"`, `"LOCATED_IN"`) |
| `fact` | `string` | Human-readable fact: `"(Alice, WORKS_AT, Acme Corp): Alice is a senior engineer"` |
| `description` | `string` | Relationship description from LLM |
| `keywords` | `string` | Comma-separated terms for fulltext search |
| `weight` | `float` | Confidence: 1.0 = explicit, 0.5 = implied, 0.2 = weak inference |
| `src_name` | `string` | Source entity name |
| `tgt_name` | `string` | Target entity name |
| `source_chunk_ids` | `list[string]` | Chunks containing evidence for this relationship |
| `spans` | `string` (JSON) | Character offsets of the evidence sentence |
| `embedding` | `vecf32` | Vector embedding of the `fact` text (added during `finalize()`) |

---

## Why a Single RELATES Edge Type?

You might expect separate edge types like `WORKS_AT`, `LOCATED_IN`, and `MARRIED_TO`. Instead, all relationships use the single `RELATES` type with the original type stored in the `rel_type` property. Here's why:

1. **Index efficiency.** Each edge type in FalkorDB needs its own vector index. With potentially hundreds of LLM-generated relationship types, you'd need hundreds of indexes. One `RELATES` type means one vector index that covers all relationships.

2. **Consistent retrieval.** The retrieval system searches all relationships at once via the RELATES edge vector index. Having a single type means one query covers everything.

3. **No information loss.** The original type is preserved in `rel_type` and appears in the `fact` string, so you can still filter by type in custom Cypher queries:
   ```cypher
   MATCH (a:Person)-[r:RELATES {rel_type: "WORKS_AT"}]->(b:Organization)
   RETURN a.name, b.name
   ```

---

## Indexes

The SDK creates 5 standard indexes during `finalize()` (or `ensure_indices()`). All are idempotent — safe to create repeatedly.

### Vector Indexes (3)

| Target | Property | Purpose |
|--------|----------|---------|
| `Chunk` nodes | `embedding` | Semantic search over text passages |
| `__Entity__` nodes | `embedding` | Semantic search over entity names |
| `RELATES` edges | `embedding` | Semantic search over relationship facts |

**Syntax:**
```
CREATE VECTOR INDEX FOR (n:Chunk) ON (n.embedding)
OPTIONS {dimension:1536, similarityFunction:'cosine'}
```

### Fulltext Indexes (2)

| Target | Properties | Purpose |
|--------|------------|---------|
| `Chunk` nodes | `text` | Keyword search over text passages |
| `__Entity__` nodes | `name`, `description` | Keyword search over entity names and descriptions |

**Syntax:**
```
CALL db.idx.fulltext.createNodeIndex('Chunk', 'text')
CALL db.idx.fulltext.createNodeIndex('__Entity__', 'name', 'description')
```

---

## The Provenance Chain

The graph structure ensures complete traceability from any answer back to its source:

```
Answer
  ↑ (generated by LLM from context)
Chunk text passages
  ↑ MENTIONED_IN (entity → chunk where it was found)
Entity relationships (RELATES)
  ↑ extracted from chunks
Chunk nodes
  ↑ PART_OF (document → chunk)
Document node
  ↑ (original source file)
Your Document
```

This is the **Zero-Loss Data** principle: every piece of source material is traceable in the graph. When the retrieval system provides context to the LLM, it can always point back to which document and which chunk the information came from.

---

## Defining Your Own Schema

A `GraphSchema` tells the extraction pipeline which entity and relationship types to look for, and the pruning step uses it to filter non-conforming data.

### Basic Schema

```python
from graphrag_sdk import GraphSchema, EntityType, RelationType

schema = GraphSchema(
    entities=[
        EntityType(label="Person", description="A human being"),
        EntityType(label="Organization", description="A company or institution"),
    ],
    relations=[
        RelationType(label="WORKS_AT", description="Employment relationship"),
    ],
)
```

### Schema with Patterns

Patterns define which source-target pairs are valid for each relationship type.
They are specified directly on `RelationType`:

```python
schema = GraphSchema(
    entities=[
        EntityType(label="Person"),
        EntityType(label="Organization"),
        EntityType(label="Location"),
    ],
    relations=[
        RelationType(label="WORKS_AT", patterns=[("Person", "Organization")]),
        RelationType(label="LOCATED_IN", patterns=[("Organization", "Location")]),
    ],
)
```

A relationship with an empty `patterns` list is allowed between any entity types.

### Open Schema Mode

If you create an empty schema (`GraphSchema()`), the pipeline operates in **open schema mode**:
- The LLM extracts any entities and relationships it finds
- The pruning step is skipped entirely
- The 11 default entity types are used for NER

This is good for exploration. For production, a defined schema produces cleaner, more consistent graphs.

---

## Inspecting the Graph

### Statistics

```python
stats = await rag.graph_store.get_statistics()
# Returns: node_count, edge_count, entity_types, relationship_types,
#          graph_density, embedded_relationship_count, mention_edge_count,
#          relates_edge_count
```

### Raw Cypher Queries

```python
# Find all Person entities
result = await rag.graph_store.query_raw(
    "MATCH (p:Person) RETURN p.name, p.description LIMIT 10"
)

# Find relationships between two entities
result = await rag.graph_store.query_raw(
    "MATCH (a:Person {name: 'Alice'})-[r:RELATES]->(b) "
    "RETURN a.name, r.rel_type, b.name, r.fact"
)

# Count mentions per entity
result = await rag.graph_store.query_raw(
    "MATCH (e:__Entity__)-[m:MENTIONED_IN]->(c:Chunk) "
    "RETURN e.name, count(m) AS mentions "
    "ORDER BY mentions DESC LIMIT 20"
)
```

---

## File Reference

| File | What it contains |
|------|-----------------|
| [`core/models.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/src/graphrag_sdk/core/models.py) | GraphSchema, EntityType, RelationType, SchemaPattern, PropertyType |
| [`storage/graph_store.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/src/graphrag_sdk/storage/graph_store.py) | Node/relationship upserts, label hints, statistics |
| [`storage/vector_store.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/src/graphrag_sdk/storage/vector_store.py) | Index creation, vector search, fulltext search |
| [`ingestion/pipeline.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/src/graphrag_sdk/ingestion/pipeline.py) | Lexical graph construction, pruning logic |
| [`ingestion/extraction_strategies/entity_extractors.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/src/graphrag_sdk/ingestion/extraction_strategies/entity_extractors.py) | Default entity types, compute_entity_id() |
