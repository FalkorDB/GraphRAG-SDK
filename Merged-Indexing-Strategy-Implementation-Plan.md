# Merged Indexing Strategy: HippoRAG + LightRAG

> **Goal**: Design a single `ExtractionStrategy` for GraphRAG SDK v2 that combines the strengths of both HippoRAG's and LightRAG's indexing pipelines into one unified approach.
>
> **Sources analyzed**:
> - [`HippoRAG.py`](https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/hipporag/HippoRAG.py) — 1611 lines
> - [`lightrag.py`](https://github.com/HKUDS/LightRAG/blob/main/lightrag/lightrag.py) — 4079 lines
> - [`operate.py`](https://github.com/HKUDS/LightRAG/blob/main/lightrag/operate.py) — `extract_entities`, `merge_nodes_and_edges`, `_process_extraction_result`

---

## 1. How Each System Indexes (Side-by-Side)

### 1.1 HippoRAG Indexing Pipeline

```
Documents
    │
    ▼
┌──────────────────────────┐
│  1. Chunk Embedding       │  Store chunk embeddings (EmbeddingStore)
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  2. OpenIE Extraction     │  LLM extracts (subject, predicate, object) triples
│     (batch_openie)        │  Also extracts NER entities per chunk
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  3. Entity Embedding      │  Embed all unique entity strings
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  4. Fact Embedding        │  Embed all unique triples as strings: "(subj, pred, obj)"
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  5. Graph Construction    │
│    a. Fact edges          │  entity ──[predicate weight]──▶ entity
│    b. Passage edges       │  chunk ──[1.0]──▶ entity (for each entity in chunk)
│    c. Synonymy edges      │  entity ──[cosine sim]──▶ entity (KNN on entity embeddings)
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  6. Persist igraph        │  pickle graph to disk
└──────────────────────────┘
```

**Key data structures:**
- `chunk_embedding_store` — chunk text → embedding
- `entity_embedding_store` — entity string → embedding
- `fact_embedding_store` — triple string → embedding
- `igraph.Graph` — unified graph with entity nodes, chunk nodes, and 3 edge types
- `node_to_node_stats` — edge weight map `(node_key, node_key) → float`
- `ent_node_to_chunk_ids` — entity → set of chunk IDs (provenance)

**What HippoRAG does well:**
- **Fact-level embeddings**: Embeds entire triples `"(Einstein, born_in, Germany)"` as first-class searchable objects. At retrieval time, queries are matched against facts directly, not just entities.
- **Synonymy edges via KNN**: After extraction, runs KNN on entity embeddings to find semantically similar entities (e.g., "US" ↔ "United States") and adds weighted edges. This implicitly resolves entities without an explicit merge step.
- **Personalized PageRank (PPR)**: At retrieval, uses the graph structure to propagate relevance from seed entities/facts to connected chunks. This enables multi-hop reasoning through the graph topology.
- **Passage nodes in the graph**: Chunks are first-class nodes connected to their entities, so PPR naturally flows from entities → chunks.

**What HippoRAG lacks:**
- No entity descriptions or types — entities are bare strings
- No relationship descriptions — predicates are strings without metadata
- No description merging — duplicate entities across chunks are not reconciled
- Open IE only — no schema-guided extraction
- igraph is in-memory only, not a production graph database

---

### 1.2 LightRAG Indexing Pipeline

```
Documents
    │
    ▼
┌──────────────────────────┐
│  1. Chunking              │  Token-based splitting with overlap
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  2. Entity Extraction     │  LLM extracts per chunk:
│     (extract_entities)    │    Entities: (name, type, description)
│                           │    Relations: (src, tgt, keywords, description, weight)
│     + Gleaning            │  Second LLM pass for missed entities
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  3. Two-Phase Merge       │
│  (merge_nodes_and_edges)  │
│                           │
│  Phase 1: Merge entities  │  Deduplicate by name, merge descriptions via LLM
│                           │  summary if descriptions accumulate across chunks
│                           │
│  Phase 2: Merge relations │  Deduplicate by (src, tgt), merge descriptions
│                           │  Update source_id tracking per entity/relation
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  4. Storage Upsert        │
│    a. Graph store         │  Nodes: entity_name → {type, description, source_id}
│    b. Entity VDB          │  Embed: "entity_name\ndescription" → vector
│    c. Relationship VDB    │  Embed: "src\ntgt\ndescription" → vector
│    d. Chunk VDB           │  Embed: chunk text → vector
│    e. KV stores           │  full_docs, text_chunks, full_entities, full_relations
└──────────────────────────┘
```

**Key data structures:**
- `chunk_entity_relation_graph` — `BaseGraphStorage` (NetworkX, Neo4j, etc.)
- `entities_vdb` — vector store for entity embeddings (name + description)
- `relationships_vdb` — vector store for relationship embeddings
- `chunks_vdb` — vector store for chunk embeddings
- `full_entities` / `full_relations` — KV stores mapping doc_id → entity/relation lists
- `entity_chunks` / `relation_chunks` — KV stores tracking chunk_ids per entity/relation
- `source_id` field on every entity/relation — `GRAPH_FIELD_SEP`-joined chunk IDs

**What LightRAG does well:**
- **Rich entity metadata**: Every entity has a `name`, `type`, and `description`. Descriptions are LLM-generated summaries that capture the entity's meaning in context.
- **Description merging via LLM**: When the same entity appears across multiple chunks, their descriptions are merged using a map-reduce LLM summarization strategy. This produces canonical entity descriptions.
- **Gleaning**: A second extraction pass catches entities/relations missed in the first pass.
- **Typed relationships with keywords**: Relations have descriptions and keywords, not just predicate strings.
- **Source ID tracking**: Every entity and relation tracks which chunks it came from (`source_id`), enabling provenance queries.
- **Concurrent processing**: Extraction and merge use semaphore-controlled asyncio for throughput.

**What LightRAG lacks:**
- No fact-level embeddings — triples aren't embedded as first-class objects
- No synonymy detection — similar entities (e.g., "US" vs "United States") aren't linked unless the LLM extracts the same canonical name
- No graph-topology-based retrieval (no PPR) — retrieval is vector similarity only
- No passage nodes in graph — chunks are in a separate VDB, not in the entity-relation graph

---

## 2. The Merged Strategy: Design

### 2.1 Core Idea

Combine LightRAG's **rich typed extraction + description merging** with HippoRAG's **fact embeddings + synonymy edges + graph-topology-aware structure**.

```
Documents
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Step 1: Chunk                                        │
│  (existing SDK FixedSizeChunking or custom)           │
└──────────┬───────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│  Step 2: Extract (LightRAG-style + schema-guided)     │
│  Per chunk, LLM extracts:                             │
│    Entities: (name, type, description)                │
│    Relations: (src, tgt, type, keywords, description) │
│  With gleaning (second pass)                          │
│  With schema constraints (SDK's GraphSchema)          │
└──────────┬───────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│  Step 3: Build Fact Representations (HippoRAG-style)  │
│  For each extracted relation, produce:                │
│    fact_string = f"({src}, {type}, {tgt})"            │
│    fact_embedding = embed(fact_string)                 │
│  Store in fact embedding index                        │
└──────────┬───────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│  Step 4: Merge Entities (LightRAG-style)              │
│  Deduplicate by normalized name                       │
│  Merge descriptions via LLM summarization             │
│  Track source_ids (chunk provenance)                  │
└──────────┬───────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│  Step 5: Synonymy Detection (HippoRAG-style)          │
│  KNN on entity embeddings (name + description)        │
│  Add SYNONYM edges above similarity threshold         │
│  This catches what LLM-based merge misses:            │
│    "US" ↔ "United States", "NYC" ↔ "New York City"   │
└──────────┬───────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│  Step 6: Build Unified Graph                          │
│  Node types:                                          │
│    Entity nodes: {name, type, description, source_id} │
│    Chunk nodes:  {text, index, doc_id}  (from lexical │
│                   graph — already exists in SDK)       │
│  Edge types:                                          │
│    RELATION:   entity ──[type, desc, weight]──▶ entity│
│    MENTIONED:  chunk ──[weight]──▶ entity             │
│    SYNONYM:    entity ──[similarity]──▶ entity        │
│    PART_OF:    doc ──▶ chunk (existing lexical graph) │
│    NEXT_CHUNK: chunk ──▶ chunk (existing)             │
└──────────┬───────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│  Step 7: Index into Vector Stores                     │
│    a. Entity VDB:   embed(name + "\n" + description)  │
│    b. Relation VDB: embed(src + "\n" + tgt + "\n" +   │
│                           description)                │
│    c. Fact VDB:     embed("(src, type, tgt)")         │
│    d. Chunk VDB:    embed(chunk_text)                 │
│  (a, b, d are LightRAG-style; c is HippoRAG-style)   │
└──────────────────────────────────────────────────────┘
```

### 2.2 What Each Repo Contributes

| Capability | Source | Why |
|---|---|---|
| Entity extraction with name, type, description | LightRAG | Richer metadata than bare OpenIE strings |
| Relationship extraction with keywords + description | LightRAG | Enables semantic search over relationships |
| Schema-guided extraction constraints | GraphRAG SDK v2 | Keeps extraction focused on domain schema |
| Gleaning (multi-pass extraction) | LightRAG | Better recall for missed entities |
| Description merging via LLM summarization | LightRAG | Canonical entity descriptions across chunks |
| Fact-level embeddings ("(subj, pred, obj)") | HippoRAG | Enables query → fact matching at retrieval time |
| Synonymy edges via KNN | HippoRAG | Catches entity aliases LLM extraction misses |
| Chunk nodes in the entity graph | HippoRAG | Enables PPR to flow from entities → chunks |
| Passage ↔ Entity edges (MENTIONED) | HippoRAG | Links chunks to their entities in the graph |
| Source ID provenance tracking | LightRAG | Full chunk → entity/relation traceability |
| Concurrent extraction with semaphore | LightRAG | Production throughput |

---

## 3. Implementation Plan for GraphRAG SDK v2

### 3.1 New/Modified Files

```
graphrag_sdk/src/graphrag_sdk/
├── core/
│   └── models.py                          # ADD: FactTriple, SynonymEdge, EntityMention models
│
├── ingestion/
│   ├── extraction_strategies/
│   │   ├── base.py                        # MODIFY: ExtractedData return type to include facts
│   │   ├── schema_guided.py               # MODIFY: Add gleaning + fact generation
│   │   └── merged_extraction.py           # NEW: The merged strategy
│   │
│   ├── resolution_strategies/
│   │   ├── base.py                        # NO CHANGE
│   │   ├── exact_match.py                 # NO CHANGE
│   │   └── description_merge.py           # NEW: LightRAG-style LLM description merging
│   │
│   └── pipeline.py                        # MODIFY: Add synonymy + fact indexing steps
│
├── retrieval/
│   └── strategies/
│       └── ppr_retrieval.py               # NEW: HippoRAG-style PPR retrieval (future)
│
└── storage/
    ├── graph_store.py                     # MODIFY: Support MENTIONED + SYNONYM edge types
    └── vector_store.py                    # MODIFY: Support multiple vector indices
```

### 3.2 New Data Models (`core/models.py`)

```python
class FactTriple(DataModel):
    """A fact triple extracted from text — embedded as a first-class object."""
    subject: str
    predicate: str
    object: str
    source_chunk_id: str
    weight: float = 1.0

    def to_embedding_string(self) -> str:
        """String representation used for embedding."""
        return f"({self.subject}, {self.predicate}, {self.object})"


class SynonymEdge(DataModel):
    """A synonymy link between two entities discovered via embedding similarity."""
    entity_a_id: str
    entity_b_id: str
    similarity: float


class EntityMention(DataModel):
    """Link between a chunk and an entity mentioned within it."""
    chunk_id: str
    entity_id: str
    weight: float = 1.0


class ExtractedEntity(DataModel):
    """A richly-typed entity extracted from text (LightRAG-style)."""
    name: str
    type: str
    description: str
    source_chunk_ids: list[str] = Field(default_factory=list)

    @property
    def embedding_string(self) -> str:
        return f"{self.name}\n{self.description}"


class ExtractedRelation(DataModel):
    """A richly-typed relationship extracted from text (LightRAG-style)."""
    source: str
    target: str
    type: str
    keywords: str = ""
    description: str = ""
    weight: float = 1.0
    source_chunk_ids: list[str] = Field(default_factory=list)

    def to_fact_triple(self) -> FactTriple:
        return FactTriple(
            subject=self.source,
            predicate=self.type,
            object=self.target,
            source_chunk_id=self.source_chunk_ids[0] if self.source_chunk_ids else "",
            weight=self.weight,
        )


class ExtractionOutput(DataModel):
    """Full output from the merged extraction strategy."""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    facts: list[FactTriple] = Field(default_factory=list)
    mentions: list[EntityMention] = Field(default_factory=list)
```

### 3.3 Merged Extraction Strategy

**File**: `ingestion/extraction_strategies/merged_extraction.py`

This is the core new implementation. It combines LightRAG's extraction with HippoRAG's fact generation.

```python
class MergedExtraction(ExtractionStrategy):
    """
    Merged extraction strategy combining:
    - LightRAG: Rich typed entity/relation extraction with gleaning
    - HippoRAG: Fact triple generation for embedding-based fact retrieval

    Pipeline per chunk:
    1. LLM extraction (entities with type+description, relations with keywords+description)
    2. Gleaning pass (re-prompt LLM for missed entities)
    3. Generate FactTriple objects from each relation
    4. Generate EntityMention objects linking chunks → entities
    """

    def __init__(
        self,
        llm: LLMInterface,
        embedder: Embedder,
        *,
        enable_gleaning: bool = True,
        max_concurrency: int = 5,
        entity_types: list[str] | None = None,
    ):
        self.llm = llm
        self.embedder = embedder
        self.enable_gleaning = enable_gleaning
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.entity_types = entity_types

    async def extract(
        self,
        chunks: TextChunks,
        schema: GraphSchema,
        ctx: Context,
    ) -> GraphData:
        """Extract entities, relations, facts, and mentions from all chunks."""

        # Phase 1: Extract per chunk (concurrent with semaphore)
        tasks = [
            self._extract_chunk(chunk, schema, ctx)
            for chunk in chunks.chunks
        ]
        chunk_results: list[ExtractionOutput] = await asyncio.gather(*tasks)

        # Phase 2: Aggregate across chunks
        all_entities, all_relations, all_facts, all_mentions = (
            self._aggregate(chunk_results)
        )

        # Phase 3: Convert to GraphData for downstream pipeline steps
        nodes = [
            GraphNode(
                id=compute_entity_id(e.name),
                label=e.type,
                properties={
                    "name": e.name,
                    "description": e.description,
                    "source_ids": e.source_chunk_ids,
                },
            )
            for e in all_entities.values()
        ]

        relationships = [
            GraphRelationship(
                start_node_id=compute_entity_id(r.source),
                end_node_id=compute_entity_id(r.target),
                type=r.type,
                properties={
                    "description": r.description,
                    "keywords": r.keywords,
                    "weight": r.weight,
                    "source_ids": r.source_chunk_ids,
                },
            )
            for r in all_relations.values()
        ]

        # Attach facts and mentions as metadata for downstream steps
        graph_data = GraphData(nodes=nodes, relationships=relationships)
        graph_data.facts = all_facts           # list[FactTriple]
        graph_data.mentions = all_mentions     # list[EntityMention]

        return graph_data

    async def _extract_chunk(
        self,
        chunk: TextChunk,
        schema: GraphSchema,
        ctx: Context,
    ) -> ExtractionOutput:
        """Extract from a single chunk with semaphore-controlled concurrency."""
        async with self._semaphore:
            # --- LLM Extraction ---
            prompt = self._build_extraction_prompt(chunk.text, schema)
            response = await self.llm.ainvoke(prompt)
            entities, relations = self._parse_extraction_response(response.content, chunk.uid)

            # --- Gleaning (second pass) ---
            if self.enable_gleaning:
                glean_prompt = self._build_gleaning_prompt(chunk.text, response.content, schema)
                glean_response = await self.llm.ainvoke(glean_prompt)
                glean_entities, glean_relations = self._parse_extraction_response(
                    glean_response.content, chunk.uid
                )
                entities = self._merge_chunk_entities(entities, glean_entities)
                relations = self._merge_chunk_relations(relations, glean_relations)

            # --- Generate Facts (HippoRAG-style) ---
            facts = [
                r.to_fact_triple()
                for r in relations
            ]

            # --- Generate Mentions (HippoRAG-style) ---
            mentions = [
                EntityMention(chunk_id=chunk.uid, entity_id=e.name)
                for e in entities
            ]

            return ExtractionOutput(
                entities=entities,
                relations=relations,
                facts=facts,
                mentions=mentions,
            )

    def _merge_chunk_entities(
        self,
        original: list[ExtractedEntity],
        gleaned: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        """Merge gleaned entities into originals, keeping longer descriptions."""
        by_name = {e.name: e for e in original}
        for e in gleaned:
            if e.name in by_name:
                if len(e.description) > len(by_name[e.name].description):
                    by_name[e.name] = e  # Keep better description
            else:
                by_name[e.name] = e
        return list(by_name.values())

    def _merge_chunk_relations(
        self,
        original: list[ExtractedRelation],
        gleaned: list[ExtractedRelation],
    ) -> list[ExtractedRelation]:
        """Merge gleaned relations, keeping longer descriptions."""
        by_key = {(r.source, r.target): r for r in original}
        for r in gleaned:
            key = (r.source, r.target)
            if key in by_key:
                if len(r.description) > len(by_key[key].description):
                    by_key[key] = r
            else:
                by_key[key] = r
        return list(by_key.values())

    def _aggregate(
        self,
        chunk_results: list[ExtractionOutput],
    ) -> tuple[dict, dict, list, list]:
        """Aggregate extraction results across all chunks."""
        all_entities: dict[str, ExtractedEntity] = {}
        all_relations: dict[tuple, ExtractedRelation] = {}
        all_facts: list[FactTriple] = []
        all_mentions: list[EntityMention] = []

        for result in chunk_results:
            for e in result.entities:
                if e.name in all_entities:
                    # Accumulate source_chunk_ids + keep longer description
                    existing = all_entities[e.name]
                    existing.source_chunk_ids.extend(e.source_chunk_ids)
                    if len(e.description) > len(existing.description):
                        existing.description = e.description
                else:
                    all_entities[e.name] = e

            for r in result.relations:
                key = tuple(sorted([r.source, r.target]))
                if key in all_relations:
                    existing = all_relations[key]
                    existing.source_chunk_ids.extend(r.source_chunk_ids)
                    existing.weight += r.weight
                    if len(r.description) > len(existing.description):
                        existing.description = r.description
                else:
                    all_relations[key] = r

            all_facts.extend(result.facts)
            all_mentions.extend(result.mentions)

        return all_entities, all_relations, all_facts, all_mentions
```

### 3.4 Description Merge Resolution Strategy

**File**: `ingestion/resolution_strategies/description_merge.py`

This replaces `ExactMatchResolution` for production use. It performs LightRAG-style LLM-based description summarization when entities appear across many chunks.

```python
class DescriptionMergeResolution(ResolutionStrategy):
    """
    Entity resolution that merges descriptions via LLM summarization.

    From LightRAG's merge_nodes_and_edges pattern:
    - If an entity has < N descriptions and total tokens < threshold → concatenate
    - If an entity has >= N descriptions or exceeds token threshold → LLM summarize
    - Uses map-reduce for very large description lists

    This is the "resolve" step in the SDK pipeline, replacing ExactMatchResolution.
    """

    def __init__(
        self,
        llm: LLMInterface,
        *,
        force_summary_threshold: int = 3,
        max_summary_tokens: int = 500,
        max_context_tokens: int = 2000,
    ):
        self.llm = llm
        self.force_summary_threshold = force_summary_threshold
        self.max_summary_tokens = max_summary_tokens
        self.max_context_tokens = max_context_tokens

    async def resolve(self, graph_data: GraphData, ctx: Context) -> ResolutionResult:
        # Group nodes by normalized name
        entity_groups: dict[str, list[GraphNode]] = defaultdict(list)
        for node in graph_data.nodes:
            canonical = self._normalize(node.properties.get("name", node.id))
            entity_groups[canonical].append(node)

        merged_nodes = []
        merged_count = 0

        for canonical, nodes in entity_groups.items():
            if len(nodes) == 1:
                merged_nodes.append(nodes[0])
            else:
                merged_node = await self._merge_entity_group(canonical, nodes, ctx)
                merged_nodes.append(merged_node)
                merged_count += len(nodes) - 1

        # Remap relationship endpoints to merged node IDs
        id_remap = self._build_id_remap(entity_groups)
        merged_relationships = self._remap_relationships(
            graph_data.relationships, id_remap
        )

        return ResolutionResult(
            nodes=merged_nodes,
            relationships=merged_relationships,
            merged_count=merged_count,
        )

    async def _merge_entity_group(
        self,
        canonical: str,
        nodes: list[GraphNode],
        ctx: Context,
    ) -> GraphNode:
        """Merge multiple nodes into one, summarizing descriptions if needed."""
        descriptions = [
            n.properties.get("description", "")
            for n in nodes
            if n.properties.get("description")
        ]

        all_source_ids = []
        for n in nodes:
            all_source_ids.extend(n.properties.get("source_ids", []))

        if len(descriptions) < self.force_summary_threshold:
            merged_description = " | ".join(descriptions)
        else:
            # LLM summarization (LightRAG map-reduce pattern)
            merged_description = await self._summarize_descriptions(
                canonical, descriptions
            )

        return GraphNode(
            id=nodes[0].id,
            label=nodes[0].label,
            properties={
                "name": canonical,
                "description": merged_description,
                "source_ids": list(set(all_source_ids)),
            },
        )

    async def _summarize_descriptions(
        self,
        entity_name: str,
        descriptions: list[str],
    ) -> str:
        """Map-reduce LLM summarization from LightRAG."""
        prompt = (
            f"Summarize the following descriptions of the entity '{entity_name}' "
            f"into a single coherent description:\n\n"
            + "\n".join(f"- {d}" for d in descriptions)
            + "\n\nSummary:"
        )
        response = await self.llm.ainvoke(prompt)
        return response.content.strip()
```

### 3.5 Synonymy Edge Detection (New Pipeline Step)

**File**: Add as a method in `ingestion/pipeline.py`

This is HippoRAG's `add_synonymy_edges` adapted for the SDK. It runs **after** entity resolution but **before** graph write.

```python
async def _detect_synonymy_edges(
    self,
    resolved_nodes: list[GraphNode],
    ctx: Context,
    *,
    top_k: int = 10,
    similarity_threshold: float = 0.85,
) -> list[GraphRelationship]:
    """
    HippoRAG-style synonymy detection.

    1. Embed all resolved entity nodes (name + description)
    2. Run KNN to find top-K similar entities for each entity
    3. Add SYNONYM edges for pairs above the similarity threshold

    This catches aliases that LLM extraction + resolution missed:
    - "US" ↔ "United States of America"
    - "NYC" ↔ "New York City"
    - "ML" ↔ "Machine Learning"
    """
    entity_nodes = [n for n in resolved_nodes if n.label != "Document" and n.label != "Chunk"]

    if len(entity_nodes) < 2:
        return []

    # Embed all entities
    texts = [
        f"{n.properties.get('name', n.id)}\n{n.properties.get('description', '')}"
        for n in entity_nodes
    ]
    embeddings = [await self.vector_store.embedder.aembed_query(t) for t in texts]
    embeddings_np = np.array(embeddings)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings_np / norms

    # Compute similarity matrix
    sim_matrix = normalized @ normalized.T

    synonym_edges = []
    seen_pairs = set()

    for i, node in enumerate(entity_nodes):
        # Get top-K similarities (excluding self)
        sim_scores = sim_matrix[i]
        top_indices = np.argsort(sim_scores)[::-1][1:top_k + 1]

        for j in top_indices:
            score = float(sim_scores[j])
            if score < similarity_threshold:
                break

            pair = tuple(sorted([node.id, entity_nodes[j].id]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            synonym_edges.append(
                GraphRelationship(
                    start_node_id=node.id,
                    end_node_id=entity_nodes[j].id,
                    type="SYNONYM",
                    properties={"similarity": score},
                )
            )

    ctx.log(f"Synonymy detection: {len(synonym_edges)} SYNONYM edges above {similarity_threshold}")
    return synonym_edges
```

### 3.6 Fact Indexing (New Pipeline Step)

Add to `pipeline.py` between resolve and write:

```python
async def _index_facts(
    self,
    graph_data: GraphData,
    ctx: Context,
) -> None:
    """
    HippoRAG-style fact embedding.

    For each extracted relation, create a fact string "(subject, predicate, object)"
    and embed it into a dedicated fact vector index.

    At retrieval time, queries are compared against fact embeddings to find
    relevant triples, which then seed the PPR walk.
    """
    facts = getattr(graph_data, 'facts', [])
    if not facts:
        # Generate facts from relationships if not already attached
        facts = [
            FactTriple(
                subject=r.properties.get("name", r.start_node_id),
                predicate=r.type,
                object=r.properties.get("name", r.end_node_id),
                source_chunk_id=r.properties.get("source_ids", [""])[0],
            )
            for r in graph_data.relationships
            if r.type != "SYNONYM"
        ]

    if facts:
        fact_strings = [f.to_embedding_string() for f in facts]
        await self.vector_store.index_facts(fact_strings, facts)
        ctx.log(f"Indexed {len(facts)} fact embeddings")
```

### 3.7 Modified Pipeline Flow

The pipeline in `pipeline.py` becomes a **10-step** sequence:

```
1.  Load          — read text from source (existing)
2.  Chunk         — split text (existing)
3.  Lexical Graph — Document → Chunk provenance (existing, MANDATORY)
4.  Extract       — entities + relations + facts + mentions (MODIFIED — uses MergedExtraction)
5.  Prune         — filter against schema (existing)
6.  Resolve       — merge descriptions via LLM (MODIFIED — uses DescriptionMergeResolution)
7.  Synonymy      — KNN-based synonym edge detection (NEW — HippoRAG)
8.  Write         — upsert nodes + rels + synonym edges to graph (MODIFIED)
9.  Index Chunks  — embed chunks in vector store (existing)
10. Index Facts   — embed fact triples in vector store (NEW — HippoRAG)
```

### 3.8 Modified Pipeline Code

```python
async def run(self, source, ctx, *, text=None, document_info=None):
    # Steps 1-3 unchanged...

    # Step 4: Extract (now produces facts + mentions too)
    ctx.log("Step 4/10: Extracting entities, relations & facts")
    graph_data = await self.extractor.extract(chunks, self.schema, ctx)

    # Step 5: Prune against schema
    ctx.log("Step 5/10: Pruning against schema")
    graph_data = self._prune(graph_data, self.schema)

    # Step 6: Resolve (LLM description merge)
    ctx.log("Step 6/10: Resolving & merging descriptions")
    resolved = await self.resolver.resolve(graph_data, ctx)

    # Step 7: Synonymy detection (NEW)
    ctx.log("Step 7/10: Detecting synonym entities")
    synonym_edges = await self._detect_synonymy_edges(resolved.nodes, ctx)

    # Step 8: Write to graph (includes MENTIONED + SYNONYM edges)
    ctx.log("Step 8/10: Writing to graph store")
    await self.graph_store.upsert_nodes(resolved.nodes)
    await self.graph_store.upsert_relationships(resolved.relationships)
    await self.graph_store.upsert_relationships(synonym_edges)

    # Write MENTIONED edges (chunk → entity)
    mentions = getattr(graph_data, 'mentions', [])
    if mentions:
        mention_rels = [
            GraphRelationship(
                start_node_id=m.chunk_id,
                end_node_id=compute_entity_id(m.entity_id),
                type="MENTIONED_IN",
                properties={"weight": m.weight},
            )
            for m in mentions
        ]
        await self.graph_store.upsert_relationships(mention_rels)

    # Step 9: Embed & index chunks
    ctx.log("Step 9/10: Embedding & indexing chunks")
    await self.vector_store.index_chunks(chunks)

    # Step 10: Index facts (NEW)
    ctx.log("Step 10/10: Indexing fact embeddings")
    await self._index_facts(graph_data, ctx)

    return IngestionResult(...)
```

---

## 4. Graph Schema After Indexing

After the merged pipeline completes, the FalkorDB graph contains:

```
Node Types:
  (:Document {path, metadata...})                        ← Lexical graph (existing)
  (:Chunk {text, index, metadata...})                    ← Lexical graph (existing)
  (:Entity {name, type, description, source_ids})        ← Merged extraction (NEW)

Edge Types:
  (Document)-[:PART_OF]->(Chunk)                         ← Lexical graph (existing)
  (Chunk)-[:NEXT_CHUNK]->(Chunk)                         ← Lexical graph (existing)
  (Entity)-[:RELATION {type, desc, keywords, weight}]->(Entity)  ← Extraction (NEW)
  (Chunk)-[:MENTIONED_IN {weight}]->(Entity)             ← HippoRAG-style (NEW)
  (Entity)-[:SYNONYM {similarity}]->(Entity)             ← HippoRAG-style (NEW)

Vector Indices:
  chunk_embeddings     — chunk text vectors              ← Existing
  entity_embeddings    — entity (name + desc) vectors    ← LightRAG-style (NEW)
  relation_embeddings  — relation (src + tgt + desc)     ← LightRAG-style (NEW)
  fact_embeddings      — fact triple string vectors      ← HippoRAG-style (NEW)
```

This graph structure enables:
- **PPR-based retrieval** (HippoRAG): Seed entities from query → PPR walk through RELATION + SYNONYM + MENTIONED_IN edges → rank chunks
- **Local retrieval** (LightRAG): Vector search on entities → 1-hop graph expansion → collect related chunks
- **Global retrieval** (LightRAG): Vector search on relationships → collect related entity summaries
- **Fact-based retrieval** (HippoRAG): Vector search on facts → rerank → seed PPR

---

## 5. Implementation Order

| Phase | Task | Depends On | Estimated Effort |
|-------|------|------------|-----------------|
| **1** | Add new data models to `core/models.py` (`FactTriple`, `SynonymEdge`, `EntityMention`, `ExtractedEntity`, `ExtractedRelation`, `ExtractionOutput`) | Nothing | Small |
| **2** | Implement `MergedExtraction` strategy (`merged_extraction.py`) | Phase 1 | Large — core LLM extraction logic |
| **3** | Implement `DescriptionMergeResolution` strategy (`description_merge.py`) | Phase 1 | Medium — LLM summarization |
| **4** | Add synonymy detection to pipeline (`_detect_synonymy_edges`) | Phase 1 | Medium — KNN + numpy |
| **5** | Add fact indexing to pipeline (`_index_facts`) | Phase 1 | Small |
| **6** | Extend `VectorStore` to support multiple named indices (entity, relation, fact, chunk) | Nothing | Medium |
| **7** | Extend `GraphStore` to support MENTIONED_IN + SYNONYM edge types | Nothing | Small |
| **8** | Modify `IngestionPipeline.run()` to wire the new 10-step flow | Phases 2-7 | Medium |
| **9** | Add PPR-based retrieval strategy (`ppr_retrieval.py`) | Phases 4-8 | Large — retrieval side |
| **10** | Tests for all new components | All | Medium |

**Critical path**: Phase 1 → Phase 2 → Phase 8 (models → extraction → pipeline wiring).

Phases 3, 4, 5, 6, 7 can be developed in parallel once Phase 1 is done.

---

## 6. LLM Prompt Design

### 6.1 Entity + Relation Extraction Prompt (Adapted from LightRAG)

```
Given a text document, identify all entities and relationships.

Entity types to extract: {entity_types}

For each entity, extract:
- entity_name: The name of the entity (normalized, capitalized)
- entity_type: One of the types listed above
- description: A comprehensive description of the entity's role in the text

For each relationship, extract:
- source_entity: The source entity name
- target_entity: The target entity name
- relationship_type: The type of relationship
- relationship_keywords: Key terms describing the relationship
- relationship_description: A description of the relationship

Format each entity as:
("entity"<|#|>entity_name<|#|>entity_type<|#|>description)

Format each relationship as:
("relationship"<|#|>source<|#|>target<|#|>keywords<|#|>description)

End with <|COMPLETE|>

---
Text:
{input_text}
```

### 6.2 Gleaning Prompt

```
The following entities and relationships were already extracted from the text.
Are there any MISSING entities or relationships? If so, add them using the same format.
If nothing is missing, output <|COMPLETE|> immediately.

Already extracted:
{previous_extraction}

---
Text:
{input_text}
```

### 6.3 Description Summarization Prompt (Adapted from LightRAG)

```
You are summarizing descriptions of the {description_type} "{description_name}".
Multiple descriptions have been collected from different sources.
Produce a single, comprehensive summary that captures all key information.
Keep the summary under {summary_length} words.
Language: {language}

Descriptions:
{description_list}

Summary:
```

---

## 7. Configuration

```python
# Default usage — everything works out of the box
rag = GraphRAG(
    connection=ConnectionConfig(host="localhost", graph_name="my_graph"),
    llm=MyLLM(model_name="gpt-4o"),
    embedder=MyEmbedder(),
    schema=GraphSchema(entities=[...], relations=[...]),
)

# Ingest with merged strategy (default in v2)
await rag.ingest("document.pdf")

# Or explicitly configure the merged strategy
await rag.ingest(
    "document.pdf",
    extractor=MergedExtraction(
        llm=my_llm,
        embedder=my_embedder,
        enable_gleaning=True,
        max_concurrency=10,
    ),
    resolver=DescriptionMergeResolution(
        llm=my_llm,
        force_summary_threshold=3,
    ),
)
```

---

## 8. Summary

This merged strategy takes the **best indexing ideas from both repos** and unifies them:

| From HippoRAG | From LightRAG | Combined Result |
|---|---|---|
| Fact embeddings `(s, p, o)` | Rich entity descriptions | Facts are embedded AND entities have descriptions |
| Synonymy edges (KNN) | LLM-based description merge | Entities are merged by LLM AND linked by similarity |
| Chunk nodes in graph | Source ID tracking | Chunks are nodes AND have source_id provenance |
| PPR for retrieval | Vector search for retrieval | Both retrieval modes available |
| igraph (in-memory) | Pluggable graph storage | FalkorDB (production) |

The result is a graph that is richer than either system alone: entities have types and descriptions (LightRAG), facts are searchable vectors (HippoRAG), synonyms are detected and linked (HippoRAG), and descriptions are canonicalized via LLM (LightRAG).
