# Retrieval — How Questions Get Answered

When you ask GraphRAG a question, it doesn't just search for keywords. It uses **multiple retrieval paths** in parallel — like asking five different experts to find relevant information, then combining their best findings into one answer.

This document explains how each path works, what it contributes, and how they perform individually and together.

---

## The Big Picture

```
                          Your Question
                               |
                    ┌──────────┴──────────┐
                    v                      v
            1. Extract Keywords      2. Embed Question
                    |                      |
         ┌─────────┴──────────────────────┴─────────┐
         v                   v                       v
   3a. RELATES Vector   3b. Text-to-Cypher    4. Entity Discovery
   (fact search)        (graph queries)       (name matching)
         |                   |                       |
         └───────────────────┴───────────────────────┘
                             |
                    5. Relationship Expansion
                             |
                    6. Chunk Retrieval (4 paths)
                    ┌────┬────┬────┬────┐
                    v    v    v    v    |
                 Full- Vector MENT- 2-hop
                 text  search IONED graph
                    └────┴────┴────┴────┘
                             |
                    7. Document Mapping
                             |
              ┌──────────────┴──────────────┐
              v                              v
   8a. Rerank Passages              8b. Filter Facts
   (stored embeddings)              (score threshold)
              |                              |
              └──────────────┬──────────────┘
                             v
                    9. Assemble Context ← Cypher results
                             |               (direct, no reranking)
                             v
                    Final LLM Answer
```

---

## Step-by-Step Explanation

### Step 1 — Keyword Extraction

**What it does:** Pulls out the important words from your question.

**How:** Two methods run together:
- **Simple filter:** Removes common words ("the", "is", "what") and keeps meaningful terms
- **LLM extraction:** Asks the language model to identify proper nouns, names, places, and specific terms

**Example:**
> Question: "What did Professor Harmon discover at the lighthouse?"
>
> Simple keywords: `["professor", "harmon", "discover", "lighthouse"]`
> LLM keywords: `["Professor Harmon", "lighthouse"]`

**Code:** `MultiPathRetrieval._extract_keywords()` in [`multi_path.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/multi_path.py)

---

### Step 2 — Embed the Question

**What it does:** Converts the question text into a numerical vector (a list of numbers) that captures its meaning.

**Why:** This vector is used later to find chunks and facts that are semantically similar to the question — even if they don't share the exact same words.

**Code:** `Embedder.aembed_query()` in [`providers/base.py`](../graphrag_sdk/src/graphrag_sdk/core/providers/base.py)

---

### Step 3a — RELATES Vector Search (Knowledge Graph Facts)

**What it does:** Searches the relationship edges in the graph by meaning similarity. Every relationship between entities (like "Alice WORKS_AT Acme Corp") has been embedded as a vector during ingestion. This step finds the relationships most relevant to the question.

**Returns:** Fact strings like:
```
Alice —[WORKS_AT]→ Acme Corp: Alice is a senior engineer at Acme Corp
```

**Important:** Facts are **scored** by their vector similarity to the question. Low-scoring facts are filtered out (see Step 8b) to reduce noise.

**Code:** `search_relates_edges()` in [`entity_discovery.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/entity_discovery.py)

---

### Step 3b — Text-to-Cypher (Graph Queries)

**What it does:** Asks the LLM to write a database query (in the Cypher language) that can directly answer the question from the graph structure.

**Why it matters:** Some questions need structural information that text search can't provide:
- "How many organizations are in the story?" → needs `COUNT`
- "What connects Alice and the castle?" → needs graph path traversal
- "List all locations mentioned" → needs `MATCH (l:Location) RETURN l.name`

**How it works:**
1. The LLM receives a description of the graph schema (what node types exist, how edges work)
2. It generates a Cypher query tailored to the question
3. The query is validated (read-only, valid labels) and sanitized (adds LIMIT, removes unsupported FalkorDB patterns)
4. If the query executes successfully, the results go **directly** to the final LLM context — they are NOT filtered by the reranker

**Runs in parallel** with step 3a to avoid adding latency. If it fails, the other paths still produce results.

**Code:** `execute_cypher_retrieval()` in [`cypher_generation.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/cypher_generation.py)

---

### Step 4 — Entity Discovery

**What it does:** Finds entities (people, places, organizations, etc.) in the graph that match the question's keywords.

**Two paths:**
- **Path A — Name matching:** Searches entity names using `CONTAINS` (e.g., "lighthouse" matches "The Old Lighthouse"). Runs as a single batched database query for efficiency.
- **Path B — Fulltext search:** Uses the text search index on entity names and descriptions. Good for partial matches and stemming.

Entities found in steps 3a and 3b are also merged in here.

**Code:** `discover_entities()` in [`entity_discovery.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/entity_discovery.py)

---

### Step 5 — Relationship Expansion

**What it does:** Starting from the discovered entities, traverses the graph to find their relationships.

**Two depths:**
- **1-hop:** Direct relationships (Alice → WORKS_AT → Acme Corp)
- **2-hop:** Indirect connections through an intermediate entity (Alice → WORKS_AT → Acme → LOCATED_IN → New York)

**Returns:** Formatted relationship strings with evidence, like:
```
Alice —[WORKS_AT]→ Acme Corp: Alice joined Acme as a senior engineer in 2019
Acme Corp —[LOCATED_IN]→ New York: Acme Corp headquarters is in Manhattan
```

**Code:** `expand_relationships()` in [`relationship_expansion.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/relationship_expansion.py)

---

### Step 6 — Chunk Retrieval (4 Paths)

**What it does:** Finds the actual text passages (chunks) from the original documents that are most relevant to the question. This is the core of passage-based retrieval.

**Four independent paths ensure we don't miss relevant passages:**

| Path | Method | What it finds |
|------|--------|---------------|
| **A — Fulltext** | Keyword search on chunk text | Passages containing exact keyword matches |
| **B — Vector** | Embedding similarity search | Passages with similar meaning (even without shared keywords) |
| **C — MENTIONED_IN** | Graph traversal from entities to their source chunks | Passages where discovered entities were originally extracted |
| **D — 2-hop** | Entity → related entity → chunk | Passages containing entities that are connected to the question's entities |

All four paths contribute to a single pool of candidate chunks.

**Code:** `retrieve_chunks()` in [`chunk_retrieval.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/chunk_retrieval.py)

---

### Step 7 — Document Mapping

**What it does:** Looks up which source document each chunk came from, so the final answer can reference the source.

**Code:** `fetch_chunk_documents()` in [`chunk_retrieval.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/chunk_retrieval.py)

---

### Step 8 — Reranking (Differentiated)

Facts and passages are ranked by **different criteria** because they have different characteristics:

#### 8a — Passage Reranking (Stored Embeddings)

**What it does:** Ranks the candidate chunks by how similar their meaning is to the question, keeping only the top 15.

**How:** Each chunk already has an embedding vector stored in the graph from ingestion. Instead of re-computing embeddings (which would require an expensive API call), we fetch the stored vectors and compute cosine similarity locally. This makes reranking **instant** instead of taking 2-3 seconds.

**Code:** `rerank_chunks()` in [`result_assembly.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/result_assembly.py)

#### 8b — Fact Filtering (Score Threshold)

**What it does:** Filters knowledge graph facts by their vector similarity score from step 3a.

**Why separate?** Facts are short structured strings ("Alice —[WORKS_AT]→ Acme") while passages are long prose paragraphs. Short text has higher cosine similarity variance — a threshold that works for passages would let too many irrelevant facts through. Facts use a higher threshold (0.25) and always keep at least 3 top facts.

**Code:** `filter_facts_by_relevance()` in [`result_assembly.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/result_assembly.py)

---

### Step 9 — Context Assembly

**What it does:** Combines everything into a structured context document that the LLM uses to generate the final answer.

**Sections (in order):**
1. **Answer format hint** — e.g., "This is a yes/no question" for yes/no questions
2. **Graph Query Results** — Direct results from text-to-cypher (bypasses reranking)
3. **Key Entities** — Names and descriptions of relevant entities
4. **Entity Relationships** — How entities connect to each other
5. **Knowledge Graph Facts** — Evidence from relationship embeddings
6. **Source Document Passages** — Ranked text passages with source attribution

The final LLM receives this structured context and generates a natural language answer.

**Code:** `assemble_raw_result()` in [`result_assembly.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/result_assembly.py)

---

## Benchmark Results

All experiments were run on the same pre-built graph (`graphrag_sdk_v2_retrieval_benchmark`) with 100 questions scored by an LLM judge (0-10 scale).

### Enhancement Comparison

| Configuration | Accuracy | vs Baseline | P50 Latency | What changed |
|---|---|---|---|---|
| **Baseline** (MultiPath only) | 83.8% | -- | 4.2s | Current default |
| **+ Text-to-Cypher** | 84.2% | +0.4% | 5.7s | Added graph query path |
| **+ All Enhancements** | **84.8%** | **+1.0%** | 5.7s | Stored rerank + fact filtering + cypher |

### Accuracy by Question Type

| Question Type | Baseline | All Enhancements | Change |
|---|---|---|---|
| Complex Reasoning | 83% | 84% | +1% |
| Contextual Summarize | 84% | **87%** | **+3%** |
| Creative Generation | 80% | 80% | -- |
| Fact Retrieval | 85% | 85% | -- |

Text-to-Cypher helps most with **Contextual Summarize** questions where structured graph relationships provide the context the LLM needs to produce comprehensive summaries.

### Isolated Path Performance

Each retrieval path was tested **in isolation** (only that path, no others) to measure its individual contribution:

| Rank | Retrieval Path | Accuracy Alone | Role in Pipeline |
|------|---------------|---------------|-----------------|
| 1 | **Vector Search** (chunk embeddings) | **77.4%** | The backbone — strongest single path |
| 2 | **Fulltext Search** (keyword matching) | 74.8% | Catches exact keyword matches that vectors miss |
| 3 | **RELATES Vector** (relationship facts) | 70.0% | Provides structured knowledge (who/what/where) |
| 4 | **Graph Traversal** (MENTIONED_IN edges) | 69.1% | Links entities back to their source text |
| 5 | **Text-to-Cypher** (LLM-generated queries) | 53.0% | Weakest alone, but adds unique structural signal |

**Key insight:** No single path matches the combined pipeline (83.8%). The value is in **multi-path fusion** — each path finds information the others miss, and together they cover more ground than any individual approach.

### Latency Profile

| Retrieval Path | P50 Latency | Notes |
|---|---|---|
| RELATES vector search | 1.6s | Fastest — single vector index query |
| Vector chunk search | 2.2s | Fast — single vector index query |
| Fulltext chunk search | 2.7s | Multiple keyword queries |
| Graph traversal | 2.8s | Depends on entity fanout |
| Text-to-Cypher | 4.5s | Includes LLM call for query generation |
| **Full MultiPath pipeline** | **4.2s** | Steps run in parallel where possible |

---

## Configuration

### Toggling Text-to-Cypher

```python
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval

# With text-to-cypher (default)
strategy = MultiPathRetrieval(
    graph_store=rag.graph_store,
    vector_store=rag.vector_store,
    embedder=embedder,
    llm=llm,
    enable_cypher=True,  # default
)

# Without text-to-cypher (lower latency)
strategy = MultiPathRetrieval(
    graph_store=rag.graph_store,
    vector_store=rag.vector_store,
    embedder=embedder,
    llm=llm,
    enable_cypher=False,
)

result = await rag.query("Your question", strategy=strategy)
```

### Tuning Parameters

| Parameter | Default | What it controls |
|---|---|---|
| `chunk_top_k` | 15 | Maximum passages after reranking |
| `max_entities` | 30 | Maximum entities to keep from discovery |
| `max_relationships` | 20 | Maximum relationships to include |
| `rel_top_k` | 15 | RELATES edge vector search results |
| `enable_cypher` | True | Toggle text-to-cypher path |

### Using the External Reranker

The pipeline has built-in reranking (step 8), but you can also apply an external reranker after retrieval:

```python
from graphrag_sdk.retrieval.reranking_strategies.cosine import CosineReranker

reranker = CosineReranker(embedder=embedder, top_k=10)
result = await rag.query("Your question", reranker=reranker)
```

---

## File Reference

| File | What it contains |
|------|-----------------|
| [`multi_path.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/multi_path.py) | Main orchestrator — coordinates all 9 steps |
| [`entity_discovery.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/entity_discovery.py) | RELATES vector search + 2-path entity discovery |
| [`chunk_retrieval.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/chunk_retrieval.py) | 4-path chunk retrieval + document mapping |
| [`relationship_expansion.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/relationship_expansion.py) | 1-hop and 2-hop relationship traversal |
| [`result_assembly.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/result_assembly.py) | Reranking, fact filtering, question hints, context assembly |
| [`cypher_generation.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/cypher_generation.py) | Text-to-Cypher: schema prompt, generation, validation, execution |
| [`base.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/base.py) | RetrievalStrategy abstract base class |
| [`local.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/strategies/local.py) | Simple vector + 1-hop retrieval (alternative strategy) |
| [`cosine.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/reranking_strategies/cosine.py) | External cosine reranker |
| [`router.py`](../graphrag_sdk/src/graphrag_sdk/retrieval/router.py) | Semantic router for conditional strategy selection |
