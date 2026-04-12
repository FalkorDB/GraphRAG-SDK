# FalkorDB GraphRAG SDK — Production Roadmap

> **Vision:** The production-grade open-source GraphRAG framework — accurate, modular, and built on a real graph database.

**Current state (v1.0.0):** 84.8% benchmark accuracy, 549 tests, 9-step ingestion pipeline, multi-path retrieval, strategy pattern throughout, async-first, Apache 2.0.

This roadmap is organized into **5 phases** covering production hardening, document intelligence, intelligent retrieval, agentic capabilities, and scale. Each item is scoped to be actionable as a GitHub issue. Items marked with :handshake: are good for community contributions.

---

## Phase 0: Foundation Hardening

*Make it safe to run in production.*

### 0.1 LLM & Embedding Timeout Enforcement :handshake:

**Problem:** `ainvoke()` in `core/providers/base.py` has retry+backoff but zero timeout. A hanging LLM endpoint blocks the entire pipeline forever. Embedding calls via `asyncio.to_thread` are also unbounded.

**Solution:** Wrap all provider calls with `asyncio.wait_for(coro, timeout=self.timeout)`. Add configurable `timeout` parameter to `LLMInterface` (default 120s) and `Embedder` (default 60s). Raise typed `LLMTimeoutError` / `EmbeddingTimeoutError`.

**Files:** `core/providers/base.py`

---

### 0.2 Error Visibility

**Problem:** 73 `except Exception` blocks across the SDK, zero `logger.error()` or `logger.critical()` calls. Production operators cannot distinguish a retryable hiccup from a database outage.

**Solution:** Audit all exception handlers. Escalate non-recoverable failures to `logger.error()`. Replace generic catches with typed `GraphRAGError` subclasses. Return structured error metadata instead of silently skipping items.

**Files:** All source files (can be split into per-module PRs) :handshake:

---

### 0.3 Async Context Manager :handshake:

**Problem:** `GraphRAG` has no `__aenter__`/`__aexit__`. If users forget cleanup, the FalkorDB `BlockingConnectionPool` (max 16 connections) leaks. In a web server, this exhausts the pool within minutes.

**Solution:** Add `async with GraphRAG(...) as rag:` support. `__aexit__` calls `self._conn.close()`. Add `__del__` warning if pool is not closed.

**Files:** `api/main.py`, `core/connection.py`

---

### 0.4 Latency Budget Enforcement

**Problem:** `Context.latency_budget_ms` and `budget_exceeded` property exist in `core/context.py` but are only checked in one place. The 9-phase retrieval pipeline never checks it — a slow query runs all 9 phases regardless.

**Solution:** Add budget checks before each retrieval phase and each LLM call. When budget is exceeded, return partial results with metadata indicating truncation. Enables predictable query latency for production SLAs.

**Files:** `retrieval/strategies/multi_path.py`, `core/context.py`

---

### 0.5 Integration Tests with Real FalkorDB :handshake:

**Problem:** All 549 tests are mocked. Any Cypher syntax issue, index creation failure, or FalkorDB version incompatibility is invisible until runtime.

**Solution:** Create `docker-compose.test.yml` with FalkorDB service. Add integration tests: connection lifecycle, CRUD, vector index creation, fulltext search, full ingest-then-query roundtrip. Run as separate CI job using GitHub Actions services.

**Files:** New `tests/integration/`, `.github/workflows/ci.yml`

---

### 0.6 Release Automation :handshake:

**Problem:** No PyPI publish workflow, no docs deployment, no automated changelog.

**Solution:** Add `publish.yml` (triggered by GitHub Release, build, PyPI via trusted publisher). Add `docs.yml` for GitHub Pages deployment (MkDocs Material). Add dependabot for dependency updates.

**Files:** New `.github/workflows/publish.yml`, `.github/workflows/docs.yml`

---

## Phase 1: Document Intelligence

*Support every document type. Understand structure, not just text.*

### 1.1 Document Loaders :handshake:

**Problem:** Only Text and PDF loaders. Anyone with HTML, Markdown, DOCX, CSV, or URL-based content must write custom code before they can try the SDK.

**Solution:** Each loader follows the existing `LoaderStrategy` ABC — async `load()` returning `DocumentOutput`. Priority order:

| Loader | Library | Notes |
|--------|---------|-------|
| **HTML** | beautifulsoup4 + trafilatura | Content extraction, strip nav/footer |
| **Markdown** | markdown-it-py | Preserve headings as metadata |
| **DOCX** | python-docx | Table + heading extraction |
| **CSV / Excel** | pandas | Row-as-document or column mapping |
| **URL** | httpx + HTML loader | Fetch then parse pipeline |
| **S3** | boto3 | Presigned URL then delegate to sub-loader |
| **Image** | Vision LLM (GPT-4V / Claude) | OCR + scene description |

Each loader is an independent PR (~50-100 lines each).

**Files:** New files in `ingestion/loaders/`

---

### 1.2 Full PDF Intelligence

**Problem:** Current PDF loader uses `pypdf` which extracts plain text only. Tables become garbled text. Layout, headers, images, and document structure are lost.

**Solution:** Multi-tier PDF processing:

1. **Table extraction** — use `pdfplumber` or `pymupdf` to detect and extract tables as structured data (list of dicts with column headers). Store tables as dedicated `Table` nodes in the graph with `columns`, `row_count` properties and `HAS_TABLE` edges from `Document`.

2. **Layout awareness** — detect sections, headers, captions. Create `Section` nodes with `PART_OF` edges for document hierarchy: `Document -> Section -> Chunk / Table`.

3. **Image extraction** — extract embedded images, pass to Vision LLM for description. Create `Figure` nodes with `caption` and `description` properties.

4. **Metadata extraction** — author, creation date, title, keywords from PDF metadata.

**Files:** `ingestion/loaders/pdf_loader.py` (major rewrite), new `ingestion/loaders/table_parser.py`

---

### 1.3 Structured Data to Graph

**Problem:** Tabular and structured data (CSV, JSON, SQL, Excel) cannot be ingested. These are common in enterprise (financial reports, CRM exports, log files).

**Solution:** New `StructuredDataLoader` and `TableExtractionStrategy`:

- **CSV/Excel:** Each row becomes an entity node. Column headers become property names. Foreign key relationships detected via shared values become `RELATES` edges. Users can provide column mapping config.
- **JSON/JSONL:** Nested structures become graph hierarchy. Objects become nodes, nested objects become `CONTAINS` edges, arrays become multiple edges.
- **SQL query results:** Wrap any DB query result as graph input.
- **Table-aware extraction prompt:** When a chunk contains table content, use a specialized prompt that understands row/column semantics instead of treating it as prose.

**Files:** New `ingestion/loaders/structured_loader.py`, new extraction prompt variant

---

### 1.4 Existing Graph Conversation

**Problem:** The SDK assumes it built the graph. There's no way to connect to a pre-existing FalkorDB graph and use it for RAG without re-ingesting. `ConnectionConfig` can connect, but retrieval strategies expect the `Document -> Chunk -> Entity` structure.

**Solution:**

1. **Schema discovery** — `GraphRAG.discover_schema()` that queries `CALL db.labels()`, `CALL db.relationshipTypes()`, and samples node properties to auto-build a `GraphSchema`. Users can then review and adjust.

2. **External graph retrieval** — new `ExternalGraphRetrieval` strategy that works without `Document`/`Chunk` nodes. Uses entity vector search + relationship expansion directly, skipping chunk retrieval. Falls back to returning entity descriptions + relationship facts as context instead of source chunks.

3. **Hybrid mode** — combine ingested knowledge (with provenance) and external graph data (without provenance) in a single retrieval. External graph entities get a `source: "external"` marker in results.

**Files:** New `core/schema_discovery.py`, new `retrieval/strategies/external_graph.py`, modify `api/main.py`

---

### 1.5 Document Lifecycle (Update / Delete) :handshake:

**Problem:** Only `delete_graph()` for full wipe. No way to remove or update a single document.

**Solution:**
- `rag.delete_document(doc_id)` — delete Document node + Chunk cascade via `PART_OF`, clean up `MENTIONED_IN` edges, optionally remove orphaned entities (entities with zero remaining mentions).
- `rag.update_document(source, doc_id)` — delete + re-ingest atomically.

**Files:** `api/main.py`, `storage/graph_store.py`

---

### 1.6 Streaming Responses :handshake:

**Problem:** `astream()` in `core/providers/base.py:128-131` yields the full response as one chunk — no real streaming.

**Solution:** Implement proper token-by-token streaming in LiteLLM provider (litellm natively supports it). Add `rag.completion_stream(question)` that returns `AsyncIterator[str]`. Retrieval runs fully, then LLM generation streams.

**Files:** `core/providers/base.py`, `api/main.py`

---

## Phase 2: Intelligent Retrieval

*Dynamic traversal, graph-native algorithms, and temporal awareness.*

### 2.1 Dynamic Graph Traversal

**Problem:** Traversal is fixed at 1-hop + 2-hop (hardcoded in `relationship_expansion.py`). A simple fact lookup wastes time on 2-hop expansion. A complex "how are X and Y connected?" query needs deeper exploration but stops at 2 hops.

**Solution:** Adaptive traversal with three mechanisms:

1. **Configurable depth** — expose `max_hops` parameter on `MultiPathRetrieval` (default 2, range 1-5). Users can tune per use case.

2. **Confidence-based expansion** — after each hop, score the relevance of discovered entities (embedding similarity to query). If average score > threshold, continue expanding. If score drops, stop. This is beam search on the graph.

3. **Query-complexity routing** — classify queries into types:
   - *Factual* ("What is X?") -> 1-hop, fast
   - *Relational* ("How is X related to Y?") -> 3+ hops, path finding
   - *Enumerative* ("List all...") -> 1-hop + sibling expansion (partially exists)
   - *Analytical* ("What are the main themes?") -> community summaries
   - *Temporal* ("What happened in 2020?") -> temporal filter + 2-hop

4. **Result-quality feedback** — if initial retrieval returns few entities (<3), automatically retry with +1 hop depth. Prevents empty answers for edge queries.

**Research:** GraphReader (2024) — agent-based graph exploration with adaptive depth.

**Files:** `retrieval/strategies/multi_path.py`, `retrieval/strategies/relationship_expansion.py`, new `retrieval/strategies/adaptive.py`

---

### 2.2 Temporal Knowledge Graph

*First community feature request.*

**Problem:** Zero temporal support. "Who was CEO of X in 2020?" is indistinguishable from "Who is CEO of X?" The graph has no concept of time.

**Design (8 implementation steps):**

**Step 1 — Data model.** Add optional `TemporalProperties` to `GraphNode` and `GraphRelationship`:

```
valid_from: datetime | None    # When this fact became true
valid_to: datetime | None      # When it stopped being true (None = still true)
event_time: datetime | None    # Point-in-time for events
temporal_source: str           # "explicit", "inferred", or "default"
```

Non-breaking: temporal properties are optional, existing graphs work identically.

**Step 2 — Date normalization utility.** Parse diverse temporal expressions: "January 2020", "the 1920s", "Victorian era", "Q3 2024", "last Tuesday" into ISO 8601 ranges. Hardest technical piece — temporal NLP.

**Step 3 — Temporal extraction.** Decorator around existing `GraphExtraction`. After entity+relationship extraction, a separate LLM call associates temporal expressions with specific entities/relationships. Separate call because temporal association is cognitively different from entity identification — merging them degrades both (validated by Microsoft GraphRAG's prompt overload experience).

**Step 4 — Storage.** Store temporal properties on FalkorDB nodes/edges. Create range indexes on `__Entity__.valid_from` and `RELATES.valid_from` for efficient temporal queries. FalkorDB's native property indexes are a natural fit — advantage over competitors implementing temporal filtering in application code.

**Step 5 — Temporal query detection.** Regex + LLM-based detection of temporal intent. Output: `query_time`, `time_range`, or `"no temporal constraint"`.

**Step 6 — Temporal filtering.** Add WHERE clauses to entity and relationship queries:

```cypher
WHERE (e.valid_to IS NULL OR e.valid_to >= $query_time)
  AND (e.valid_from IS NULL OR e.valid_from <= $query_time)
```

**Step 7 — Temporal retrieval strategy.** `TemporalRetrieval` extending `MultiPathRetrieval` with three query modes:
- **Point-in-time snapshot:** "Who was CEO in 2020?" -> filter graph to 2020 state
- **Time range:** "What happened between 2019 and 2021?" -> facts within range
- **Timeline:** "Show the history of X" -> return all temporal versions, ordered

**Step 8 — Tests + benchmark.** Temporal-specific test suite + benchmark questions.

**Research:**
- Graphiti (Zep) — temporal knowledge graphs with episodic memory (direct competitor)
- T-GAP (Jung et al., NAACL 2021) — learning to walk across time for temporal KG completion
- TimeQA (Chen et al., TACL 2023) — benchmark for QA over time-evolving knowledge
- TERO (Xu et al., COLING 2020) — time-aware KG embedding via temporal rotation

**Files:** `core/models.py`, new `utils/temporal.py`, new `ingestion/extraction_strategies/temporal_extraction.py`, `storage/graph_store.py`, `retrieval/strategies/multi_path.py`, new `retrieval/strategies/temporal.py`

---

### 2.3 Community Detection & Hierarchical Summaries

**Problem:** No graph analytics. This is where FalkorDB's graph database advantage should shine vs. LightRAG (NetworkX in-memory) and pure vector RAG.

**Solution:**
1. **Leiden/Louvain clustering** — group entities into communities. FalkorDB can run this server-side instead of pulling the entire graph into Python memory (advantage over Microsoft GraphRAG's approach).
2. **Hierarchical community summaries** — LLM-generated summaries at each hierarchy level, stored as `Community` nodes. Enables global queries ("What are the main themes across all documents?").
3. **PageRank for entity importance** — weight entities during retrieval. More important entities get higher relevance boost and deeper traversal.

**Research:** "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Edge et al., Microsoft, 2024) — the paper that launched the GraphRAG wave.

**Files:** New `analytics/` module, modify `retrieval/strategies/multi_path.py`

---

### 2.4 Query Decomposition

**Problem:** Complex multi-hop questions go through a single retrieval pass. "How did the relationship between X and Y change after Z happened?" requires decomposition into sub-queries.

**Solution:** LLM-based query decomposition into parallel sub-query retrieval then result fusion then final answer. Supports:
- **Compositional:** "Compare X and Y" -> retrieve X, retrieve Y, synthesize
- **Sequential:** "What happened after X?" -> retrieve X, identify time, retrieve events after
- **Conditional:** "If X is true, what about Y?" -> retrieve X, evaluate, branch

**Research:** "Decomposed Prompting" (Khot et al., 2023), "Interleaving Retrieval with Chain-of-Thought Reasoning" (Trivedi et al., 2023)

**Files:** New `retrieval/strategies/decomposed.py`

---

### 2.5 Hybrid Search & Result Fusion

**Problem:** Current retrieval uses vector similarity + fulltext search but combines results with a simple union. No principled fusion.

**Solution:**
- **BM25 integration** for keyword-based retrieval (exact matches, rare terms)
- **Reciprocal Rank Fusion (RRF)** to combine vector, fulltext, BM25, and graph-based results
- **Learned reranking** — cross-encoder reranker (e.g., BGE-reranker) for final result ordering

**Files:** `retrieval/strategies/multi_path.py`, new `retrieval/rerankers/`

---

## Phase 3: Agentic GraphRAG

*Graph as a tool. Agents that reason over knowledge graphs.*

### 3.1 Agentic Retrieval (ReAct-Style)

**Problem:** Current retrieval is a fixed pipeline — no ability to inspect results, decide they're insufficient, and try a different strategy. `LLMResponse.tool_calls` field exists in `core/models.py` but is never used.

**Solution:** ReAct-style agent loop for complex queries:

1. **Tool definitions:**
   - `search_entities(query, filters)` — vector + fulltext entity search
   - `get_neighbors(entity_id, hop_depth, direction)` — graph traversal
   - `search_chunks(query, top_k)` — chunk-level retrieval
   - `run_cypher(query)` — safe read-only Cypher execution (builds on existing `cypher_generation.py`)
   - `get_entity_details(entity_id)` — full entity properties + relationships
   - `get_community_summary(community_id)` — hierarchical summary (when available)
   - `search_temporal(query, time_range)` — time-filtered search (when available)

2. **Agent loop:**

   ```
   while not sufficient_context and budget_remaining:
       action = LLM(question + observations_so_far + available_tools)
       observation = execute_tool(action)
       observations.append(observation)
   answer = LLM(question + all_observations)
   ```

3. **Termination:** Budget-based (max iterations, max tokens, latency budget) + confidence-based (LLM decides it has enough context).

**Research:** "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2023), "Toolformer" (Schick et al., 2023)

**Files:** New `retrieval/strategies/agentic.py`, modify `core/models.py`

---

### 3.2 MCP Server (Model Context Protocol)

**Problem:** No way for external agents (Claude, ChatGPT, custom agents) to use the knowledge graph as a tool. The SDK is code-only.

**Solution:** Expose GraphRAG as an MCP server with tools:

| MCP Tool | Description |
|----------|-------------|
| `graphrag_query` | Full RAG completion — question in, answer + sources out |
| `graphrag_search` | Retrieval only — return entities, relationships, chunks |
| `graphrag_ingest` | Ingest a document into the graph |
| `graphrag_explore` | Browse graph neighborhood of an entity |
| `graphrag_cypher` | Execute read-only Cypher query |
| `graphrag_timeline` | Get temporal history of an entity (when available) |
| `graphrag_stats` | Graph statistics (node/edge counts, entity types) |
| `graphrag_schema` | Return the graph schema |

**Implementation:** Use the `mcp` Python package. Wrap existing `GraphRAG` methods as tool handlers. Support both `stdio` and `sse` transports.

**Impact:** Any MCP-compatible client (Claude Desktop, VS Code Copilot, custom agents) can use the knowledge graph as a tool without writing Python code.

**Files:** New `mcp/` module at package root, `mcp/server.py`, `mcp/tools.py`

---

### 3.3 Dynamic Graph Walk

**Problem:** Current traversal follows a fixed pattern (vector search, entity discovery, 1-hop, 2-hop). It doesn't adapt to graph topology or query semantics.

**Solution:** Intelligent graph walk inspired by GraphReader and random walk with restart:

1. **Importance-weighted traversal** — use PageRank scores (from Phase 2.3) to prioritize which neighbor to expand next. Hub entities (high centrality) get priority.

2. **Beam search on graph** — maintain a beam of K most promising entities at each hop. Score by embedding similarity to query. Prune low-scoring paths. Continue until convergence or budget exhaustion.

3. **Bidirectional search** — for relational queries ("How are X and Y connected?"), start traversal from both entities simultaneously and meet in the middle.

4. **Path scoring** — instead of returning individual entities, return scored paths (chains of entities + relationships). The LLM gets structured reasoning chains, not a bag of facts.

**Research:** "GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models" (Li et al., 2024), "Personal PageRank-based retrieval" (HippoRAG, Gutierrez et al., NeurIPS 2024)

**Files:** New `retrieval/strategies/graph_walk.py`

---

### 3.4 Skill / Tool Library :handshake:

**Problem:** As agentic capabilities grow, users need pre-built skills for common graph reasoning tasks.

**Solution:** Composable skills that combine multiple tools:

- **Entity comparison** — retrieve two entities and their neighborhoods, generate structured comparison
- **Impact analysis** — given an entity, find all entities within N hops and assess relationship strength
- **Contradiction detection** — find entities/relationships with conflicting facts from different sources
- **Gap analysis** — identify entities with few relationships or missing expected connections
- **Timeline reconstruction** — collect all temporal facts about an entity, order chronologically

**Files:** New `skills/` module

---

## Phase 4: Performance & Scale

*Parallelism, memory efficiency, latency reduction.*

### 4.1 Retrieval Parallelization

**Problem:** The 9-phase retrieval pipeline in `multi_path.py` is fully sequential. Steps with no dependency still wait for each other.

**Solution:**

```
Phase 1+2 (parallel): keyword_extraction | query_embedding
Phase 3: RELATES edge vector search (needs embedding)
Phase 4a+4b (parallel): Cypher CONTAINS search | fulltext entity search
Phase 5: relationship expansion (needs entities)
Phase 6a+6b+6c+6d (parallel): fulltext chunks | vector chunks | MENTIONED_IN | 2-hop chunks
Phase 7+8 (parallel): fetch doc names | cosine reranking
Phase 9: context assembly + LLM generation
```

**Estimated impact:** P50 reduction from ~13.8s to ~8s (steps 1+2 alone saves ~2s).

**Files:** `retrieval/strategies/multi_path.py`

---

### 4.2 Ingestion Parallelism

**Problem:** Pipeline processes one document at a time. Steps 1-7 are sequential. No chunk-level extraction parallelism.

**Solution:**

1. **Multi-document parallelism** — `rag.ingest([file1, file2, ...], concurrency=4)`. Each document runs its own pipeline instance. Bounded by semaphore to control LLM API rate.

2. **Chunk-level extraction parallelism** — within a single document, extract entities from multiple chunks concurrently. Currently sequential in `graph_extraction.py`. Add `asyncio.gather` with semaphore for bounded concurrency.

3. **Pipeline streaming** — instead of all chunks waiting for extraction to complete before resolution, stream chunks through the pipeline. Each chunk progresses independently until the resolution step (which needs all entities).

**Files:** `api/main.py`, `ingestion/pipeline.py`, `ingestion/extraction_strategies/graph_extraction.py`

---

### 4.3 Memory Efficiency

**Problem:** All chunks and entities are held in memory simultaneously. Entity resolution computes pairwise similarity (O(n^2)). Mentions writing is uncapped. For large documents (1000+ chunks), this causes memory pressure.

**Solution:**

1. **Generator-based pipeline** — replace list accumulation with async generators where possible. Chunks flow through load, chunk, extract as a stream instead of materializing all at once.

2. **Batched entity resolution** — instead of loading all entities for pairwise comparison, process in label-partitioned batches. Entities of different labels never merge, so partition by label first (already partially done in semantic resolution).

3. **Capped mention writes** — limit `MENTIONED_IN` edge creation to top-K mentions per entity (ranked by relevance) instead of all mentions. Reduces edge explosion for frequently-mentioned entities.

4. **Incremental finalization** — track which entities/relationships were added since last `finalize()`. Only process new items instead of re-scanning the entire graph.

5. **Embedding cache** — hash-based cache for chunk content. If a chunk's text hasn't changed, reuse its stored embedding instead of re-computing.

**Files:** `ingestion/pipeline.py`, `ingestion/resolution_strategies/`, `storage/vector_store.py`, `api/main.py`

---

### 4.4 Latency Reduction

**Problem:** Query P50 is ~5.5s. For real-time chat UX, target is <3s.

**Solution:**

1. **Query result caching** — LRU cache for identical or similar queries. Hash question embedding, cache retrieval results with TTL. Similar questions (cosine > 0.98) return cached results. Currently zero caching exists in retrieval.

2. **Pre-computed entity neighborhoods** — for high-PageRank entities, pre-compute and cache their 2-hop neighborhoods. Eliminates graph traversal for common entity queries.

3. **Speculative retrieval** — during keyword extraction (LLM call), simultaneously start entity vector search with the raw question embedding. If keywords don't improve results, use the speculative results.

4. **Connection pooling optimization** — current pool max is 16. For high-concurrency servers, make configurable. Add connection warmup on startup.

5. **Cypher query optimization** — combine multiple sequential queries into single multi-RETURN Cypher. The `get_statistics()` method runs 5 sequential queries that could be one.

**Files:** `retrieval/strategies/multi_path.py`, `core/connection.py`, `storage/graph_store.py`

---

### 4.5 Multi-tenancy Enforcement

**Problem:** `Context.tenant_id` exists but is never used in any storage query. All tenants see all data.

**Solution:** Two tiers:
- **Graph-level isolation (recommended for most):** Separate FalkorDB graph per tenant via `ConnectionConfig.graph_name`. Simple, strong isolation, no query overhead.
- **Property-level isolation:** For shared graphs, add `tenant_id` property to all nodes, filter all Cypher queries with `WHERE n.tenant_id = $tenant_id`. Enables cross-tenant analytics while maintaining data isolation.

**Files:** `storage/graph_store.py`, `storage/vector_store.py`, `api/main.py`

---

## Phase 5: Research & Advanced Methods

*Push the frontier. Each item is a research direction with paper references.*

### 5.1 HippoRAG Memory Model

Inspired by hippocampal indexing: use Personal PageRank from query-matched entities to discover relevant knowledge, treating the graph as analogous to hippocampal memory. Naturally fits FalkorDB's graph traversal capabilities.

**Paper:** "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models" (Gutierrez et al., NeurIPS 2024)

---

### 5.2 RAPTOR Hierarchical Summaries

Recursive abstractive processing: cluster chunks, summarize clusters, cluster summaries, repeat. Creates a tree of increasingly abstract summaries that enables multi-resolution retrieval.

**Paper:** "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval" (Sarthi et al., 2024)

---

### 5.3 Adaptive Retrieval (Self-RAG)

Not all queries need retrieval. Self-RAG learns when to retrieve, what to retrieve, and whether the retrieved context is useful. Reduces unnecessary retrieval calls and improves answer quality.

**Paper:** "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al., 2023)

---

### 5.4 Knowledge Graph Completion

Predict missing links in the knowledge graph using graph structure. TransE/RotatE embeddings on the knowledge graph itself. Can suggest relationships that extraction missed.

**Paper:** "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space" (Sun et al., ICLR 2019)

---

### 5.5 External Knowledge Linking :handshake:

Link extracted entities to Wikidata/DBpedia for disambiguation and enrichment. Add `SAME_AS` edges and import external properties (coordinates, dates, categories).

**Paper:** "BLINK: Scalable Entity Linking" (Wu et al., EMNLP 2020)

---

### 5.6 Domain-Specific Extraction

Fine-tuned extractors for specialized domains: biomedical (diseases, genes, drugs), legal (statutes, precedents, parties), financial (instruments, metrics, filings). Community-contributed via the strategy pattern.

---

### 5.7 Ontology Learning

Automatically discover entity types and relationship types from the data instead of using the fixed 11-type schema. Starts with a schema-free extraction pass, clusters entity types, proposes an ontology for user approval.

---

### 5.8 Graph of Thoughts

Structured LLM reasoning where intermediate thoughts form a graph (not a chain). Each thought can branch, merge, or loop back. Applied to query decomposition and multi-step reasoning.

**Paper:** "Graph of Thoughts: Solving Elaborate Problems with Large Language Models" (Besta et al., 2023)

---

## Research Backlog — Paper References

| # | Topic | Paper | Year | Relevance |
|---|-------|-------|------|-----------|
| 1 | Community Summaries | "From Local to Global" (Edge et al.) | 2024 | Hierarchical graph summarization — Phase 2.3 |
| 2 | Temporal KG | "T-GAP" (Jung et al., NAACL) | 2021 | Learning temporal graph walks — Phase 2.2 |
| 3 | Hippocampal RAG | "HippoRAG" (Gutierrez et al., NeurIPS) | 2024 | Personal PageRank retrieval — Phase 5.1 |
| 4 | Graph Agent | "GraphReader" (Li et al.) | 2024 | Agent-based graph exploration — Phase 3.1 |
| 5 | Query Decomposition | "Decomposed Prompting" (Khot et al.) | 2023 | Multi-step retrieval — Phase 2.4 |
| 6 | Hierarchical Retrieval | "RAPTOR" (Sarthi et al.) | 2024 | Tree-organized retrieval — Phase 5.2 |
| 7 | Self-Reflective RAG | "Self-RAG" (Asai et al.) | 2023 | Adaptive retrieval — Phase 5.3 |
| 8 | Graph Reasoning | "Graph of Thoughts" (Besta et al.) | 2023 | Structured reasoning — Phase 5.8 |
| 9 | Temporal QA | "TimeQA" (Chen et al., TACL) | 2023 | Temporal benchmark — Phase 2.2 |
| 10 | Entity Linking | "BLINK" (Wu et al., EMNLP) | 2020 | KB linking — Phase 5.5 |

---

## Success Metrics

### Adoption

| Metric | 3 months | 6 months | 12 months |
|--------|----------|----------|-----------|
| GitHub stars | 1,000 | 3,000 | 8,000 |
| PyPI downloads/month | 1,000 | 5,000 | 20,000 |
| External contributors | 10 | 30 | 75 |
| Document loaders | 8 | 12 | 20+ |
| Retrieval strategies | 4 | 7 | 12+ |

### Quality

| Metric | Current | 6 months | 12 months |
|--------|---------|----------|-----------|
| Literary benchmark accuracy | 84.8% | 89% | 92% |
| Temporal benchmark accuracy | N/A | 80% | 88% |
| Integration test coverage | 0% | 60% | 85% |

### Performance

| Metric | Current | 6 months | 12 months |
|--------|---------|----------|-----------|
| Query P50 latency | 5.5s | 3.5s | <2s |
| Ingestion throughput | ~0.4 doc/min | 2 doc/min | 8 doc/min |
| Max graph size tested | 20 docs | 500 docs | 10,000 docs |

---

## Competitive Positioning

### vs. LightRAG (32K stars, NetworkX)

LightRAG uses in-memory NetworkX — it cannot handle corpora that exceed RAM, cannot be shared across processes, and loses data on crash. FalkorDB is a real graph database with disk persistence, concurrent access, vector indexes, and fulltext search. Every feature that uses graph-native capabilities (community detection, temporal range indexes, multi-tenant isolation, agentic traversal) widens this gap.

### vs. Microsoft GraphRAG (community summaries)

Microsoft GraphRAG's innovation is Leiden clustering + hierarchical summaries. Phase 2.3 implements this on FalkorDB, more efficiently (server-side algorithms vs. pulling the entire graph into Python). FalkorDB GraphRAG SDK's multi-path retrieval already outperforms on accuracy (84.8% vs 81.3%).

### vs. Graphiti / Zep (temporal)

Graphiti made temporal awareness its primary differentiator. Phase 2.2 addresses this directly, with FalkorDB's native property indexes for temporal range queries (vs. application-level filtering).

### The compound advantage

No competitor has all four: (1) real graph database, (2) strategy-pattern modularity, (3) multi-path retrieval, (4) agentic graph tools. The strategy pattern is not just architecture — it is the ecosystem strategy. Every community-contributed loader, extractor, or retrieval strategy makes the platform stickier.

---

## Community Contribution Guide

### Good First Issues :handshake:

| Issue | Phase | Difficulty | Skills |
|-------|-------|------------|--------|
| HTML loader | 1.1 | Low | Python, beautifulsoup4 |
| Markdown loader | 1.1 | Low | Python, markdown parsing |
| DOCX loader | 1.1 | Low | Python, python-docx |
| CSV loader | 1.1 | Low | Python, pandas |
| LLM timeout enforcement | 0.1 | Low | Python, asyncio |
| Async context manager | 0.3 | Low | Python, asyncio |
| Error logging audit (per module) | 0.2 | Low | Python, logging |
| Integration test cases | 0.5 | Medium | Python, pytest, Docker |
| YAML config support | — | Medium | Python, Pydantic |
| Retrieval parallelization | 4.1 | Low | Python, asyncio |
| Release automation workflow | 0.6 | Medium | GitHub Actions, PyPI |
| Benchmark questions for new domains | — | Low | Domain knowledge |
| Wikidata entity linking | 5.5 | Medium | Python, APIs |

### Contributor Ladder

- **Newcomer:** Loader, error fix, benchmark question, documentation improvement
- **Regular:** Strategy implementation, config system, integration tests
- **Core:** Temporal extraction, community detection, agentic retrieval, query decomposition
- **Maintainer:** Architecture decisions, release management, code review, roadmap input
