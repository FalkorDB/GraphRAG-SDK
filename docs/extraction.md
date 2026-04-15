# Extraction — How Text Becomes Entities and Relationships

Extraction is the heart of the ingestion pipeline. It takes raw text chunks and turns them into structured knowledge — named entities (people, places, concepts) and the relationships between them (works at, located in, married to).

GraphRAG SDK uses a **two-step hybrid approach**: a fast local NER model finds the entities first, then an LLM verifies them and extracts relationships. This gives you the speed of local models with the reasoning power of LLMs.

---

## The Big Picture

```
            Text Chunk
                |
     ┌──────────┴──────────┐
     v                      v
  (Optional)           Step 1: Entity NER
  Coreference          ┌─────────────────┐
  Resolution           │  GLiNERExtractor │  (default, local)
  (resolve pronouns)   │  LLMExtractor    │  (API-based)
     |                 │  Custom          │  (your own)
     v                 └────────┬────────┘
  Resolved Text                 |
                        List of entities
                        (name, type, confidence, spans)
                                |
                     Step 2: LLM Verify + Relationships
                     ┌──────────────────────────────────┐
                     │ LLM receives:                     │
                     │   - Pre-extracted entities         │
                     │   - Original text                  │
                     │                                    │
                     │ LLM returns:                       │
                     │   - Verified entities (fixed/added) │
                     │   - Relationships with evidence     │
                     └──────────────┬───────────────────┘
                                    |
                          Aggregate across chunks
                          (dedup entities + relations)
                                    |
                          Convert to GraphData
                          (GraphNode + GraphRelationship)
```

---

## Step 1 — Entity NER

The first step identifies **what things** are mentioned in the text. It's pluggable — you choose which NER backend to use.

### EntityExtractor ABC

All extractors implement the same interface:

```python
class EntityExtractor(ABC):
    @abstractmethod
    async def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        source_chunk_id: str,
    ) -> list[ExtractedEntity]
```

Each extracted entity has:
- **name** — the entity's name as it appears in the text (e.g., "Professor Harmon")
- **type** — one of the allowed entity types (e.g., "Person")
- **description** — a brief description
- **confidence** — how confident the model is (0.0 to 1.0)
- **spans** — character offsets where the entity appears: `{chunk_id: [{start, end}]}`
- **source_chunk_ids** — which chunks mention this entity

### GLiNERExtractor (Default)

A local transformer model that runs on your machine — no API calls needed.

```python
from graphrag_sdk import GLiNERExtractor

extractor = GLiNERExtractor(
    threshold=0.75,                          # confidence threshold (below = "Unknown")
    model_name="urchade/gliner_medium-v2.1", # HuggingFace model
)
```

**How it works:**
1. The model is loaded lazily on first use (thread-safe via `threading.Lock`)
2. Runs inference via `asyncio.to_thread()` to avoid blocking the event loop
3. Returns predictions with character-level spans — more precise than LLM spans
4. Entities below the confidence threshold are labeled `"Unknown"` (not discarded)

**Best for:** Production pipelines where you want fast, cheap NER without API costs.

### LLMExtractor

Uses your LLM for entity extraction via a structured prompt.

```python
from graphrag_sdk import LLMExtractor

extractor = LLMExtractor(
    llm=my_llm,
    threshold=0.75,
)
```

**How it works:** Sends a `NER_PROMPT` to the LLM asking it to extract entities with names, types, descriptions, confidence scores, and character offsets. The response is parsed as JSON.

**Best for:** When you need richer entity descriptions or when GLiNER doesn't perform well on your domain.

### Custom Extractors

Subclass `EntityExtractor` to plug in any NER backend:

```python
from graphrag_sdk import EntityExtractor
from graphrag_sdk.core.models import ExtractedEntity

class SpaCyExtractor(EntityExtractor):
    def __init__(self, model_name="en_core_web_sm"):
        import spacy
        self._nlp = spacy.load(model_name)

    async def extract_entities(self, text, entity_types, source_chunk_id):
        import asyncio
        doc = await asyncio.to_thread(self._nlp, text)
        return [
            ExtractedEntity(
                name=ent.text,
                type=ent.label_,
                description="",
                source_chunk_ids=[source_chunk_id],
                spans={source_chunk_id: [{"start": ent.start_char, "end": ent.end_char}]},
            )
            for ent in doc.ents
        ]
```

---

## Step 2 — LLM Verify + Relationship Extraction

The second step uses the LLM to do two things at once:

1. **Verify entities** — remove false positives from step 1, fix naming errors, and add any entities the NER model missed
2. **Extract relationships** — identify all factual connections between the verified entities

### What the LLM Receives

A structured prompt (`VERIFY_EXTRACT_RELS_PROMPT`) containing:
- The list of entity types
- The pre-extracted entities from step 1 (as JSON)
- The original chunk text

### What the LLM Returns

A JSON object with two arrays:

```json
{
  "entities": [
    {"name": "Alice", "type": "Person", "description": "Senior engineer at Acme Corp", "span_start": 0, "span_end": 5}
  ],
  "relationships": [
    {
      "source": "Alice",
      "target": "Acme Corp",
      "type": "WORKS_AT",
      "description": "Alice is a senior engineer at Acme Corp",
      "keywords": "employment, career",
      "weight": 1.0,
      "span_start": 0,
      "span_end": 42
    }
  ]
}
```

### Metadata Merging

After step 2, GLiNER spans from step 1 are **carried forward** into the verified entities. GLiNER character offsets are more precise than LLM-generated offsets, so step 1 spans take priority. If the LLM found a new entity that GLiNER missed, the LLM's own spans are kept.

### Fallback Behavior

If step 2 fails for a chunk (bad JSON, API error), the pipeline falls back to using step 1 entities without relationships — you still get entities, just no relationships for that chunk.

---

## The Ontology — Entity Types

Every extracted entity is mapped to one of the allowed entity types. The SDK ships with 11 default types:

```
Person, Organization, Technology, Product, Location,
Date, Event, Concept, Law, Dataset, Method
```

Entities that don't match any type (or fall below the confidence threshold) are labeled `"Unknown"`.

### Customizing the Ontology

There are three ways to define entity types, listed by priority:

**1. GraphSchema entities (highest priority):**
```python
from graphrag_sdk import GraphSchema, EntityType

schema = GraphSchema(entities=[
    EntityType(label="Gene", description="A gene or genetic locus"),
    EntityType(label="Disease", description="A disease or condition"),
])
rag = GraphRAG(connection=conn, llm=llm, embedder=embedder, schema=schema)
# Extraction uses: ["Gene", "Disease"]
```

**2. `entity_types` parameter on GraphExtraction:**
```python
extractor = GraphExtraction(
    llm=llm,
    entity_types=["Gene", "Protein", "Disease", "Drug"],
)
```

**3. Defaults (lowest priority):**
```python
extractor = GraphExtraction(llm=llm)
# Uses the 11 default types
```

---

## Entity Name Validation

Not every string the NER model produces is a valid entity. The SDK filters names through quality gates:

| Rule | Rejected Examples |
|------|------------------|
| Length 2-80 characters | Single characters, descriptions masquerading as names |
| Not a pronoun | he, she, they, it, him, her, his, them, who |
| Not a generic reference | narrator, author, reader, person, someone, story, chapter |

The full stoplist includes ~50 pronouns and generic references. See `_ENTITY_STOPLIST` in `entity_extractors.py`.

---

## Entity Aggregation

After extraction runs on all chunks, entities are **deduplicated across chunks** by `(normalized_name.lower(), type.lower())`:

- If the same entity appears in multiple chunks, the one with the **longer description** wins
- `source_chunk_ids` are merged (the entity knows every chunk it appeared in)
- `spans` are merged (character offsets from every chunk)
- Capitalized names are preferred over lowercase

**Example:** "Alice" appears in chunks 3, 7, and 12. After aggregation, there's one entity with `source_chunk_ids = ["chunk_3", "chunk_7", "chunk_12"]` and spans from all three chunks.

---

## Relationship Aggregation

Relationships are similarly deduplicated by `(source.lower(), type.lower(), target.lower())`:

- Longer descriptions win
- `source_chunk_ids` are merged
- Spans are merged across chunks

---

## How Entities Become Graph Nodes

After aggregation, each entity becomes a `GraphNode`:

- **ID:** `compute_entity_id(name, type)` — deterministic: `"alice__person"` (lowercase, spaces replaced with underscores, type-qualified to prevent collisions)
- **Label:** The entity type (e.g., `Person`, `Organization`)
- **Properties:** `name`, `description`, `source_chunk_ids`, and optionally `spans`

---

## How Relationships Become Graph Edges

All relationships become `GraphRelationship` objects with type `"RELATES"`:

- **Type:** Always `"RELATES"` — a single unified edge type
- **Properties:**
  - `rel_type` — the original relationship type (e.g., `"WORKS_AT"`)
  - `fact` — a human-readable fact string: `"(Alice, WORKS_AT, Acme Corp): Alice is a senior engineer at Acme Corp"`
  - `description` — the relationship description
  - `keywords` — comma-separated terms for fulltext search
  - `weight` — confidence (1.0 = explicitly stated, 0.5 = implied)
  - `src_name`, `tgt_name` — endpoint entity names
  - `source_chunk_ids` — provenance
  - `spans` — character offsets of the evidence

**Why a single edge type?** Using one `RELATES` type with a `rel_type` property avoids creating dozens of relationship types in the graph (each needing its own index). The original type is preserved in the `rel_type` property and is used for display and filtering.

---

## Optional: Coreference Resolution

Coreference resolution replaces pronouns with the entities they refer to, **before** extraction runs:

```
Before: "She went to the store. She bought milk."
After:  "Voss went to the store. Voss bought milk."
```

### FastCorefResolver

```python
from graphrag_sdk import FastCorefResolver

extractor = GraphExtraction(
    llm=llm,
    coref_resolver=FastCorefResolver(),  # pip install graphrag-sdk[fastcoref]
)
```

**Model:** `"biu-nlp/lingmess-coref"` (LingMessCoref)

**How it works:**
1. Predict coreference clusters (groups of spans referring to the same entity)
2. Find the canonical mention for each cluster (the longest non-pronoun span)
3. Replace pronouns with canonical mentions, right-to-left to preserve offsets
4. Handles possessives: "her" becomes "Voss's"

**When to use:** When your documents have heavy pronoun usage and extraction is missing entities because of it. Adds ~1-2 seconds per chunk.

---

## Concurrency

- **GLiNER/custom extractors:** Use `asyncio.Semaphore(max_concurrency or 12)` for parallel chunk processing
- **LLM extractors:** Use `llm.abatch_invoke(max_concurrency=...)` for batched LLM calls
- **Step 2 (verify + rels):** Always uses `llm.abatch_invoke()` regardless of step 1 backend

---

## File Reference

| File | What it contains |
|------|-----------------|
| [`graph_extraction.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/src/graphrag_sdk/ingestion/extraction_strategies/graph_extraction.py) | 2-step extraction orchestrator, entity/relation aggregation, GraphData conversion |
| [`entity_extractors.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/src/graphrag_sdk/ingestion/extraction_strategies/entity_extractors.py) | EntityExtractor ABC, GLiNERExtractor, LLMExtractor, entity utilities |
| [`coref_resolvers.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/src/graphrag_sdk/ingestion/extraction_strategies/coref_resolvers.py) | CorefResolver ABC, FastCorefResolver |
| [`base.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/src/graphrag_sdk/ingestion/extraction_strategies/base.py) | ExtractionStrategy ABC |
