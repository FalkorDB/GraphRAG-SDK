# Strategy Reference

GraphRAG SDK uses the **Strategy pattern** for every algorithmic concern. Each concern has an abstract base class (ABC) with one or more built-in implementations. You can swap any implementation or write your own.

## Overview

| # | Concern | ABC | Built-in Implementations |
|---|---------|-----|------------------------|
| 1 | Loading | `LoaderStrategy` | `TextLoader`, `PdfLoader` |
| 2 | Chunking | `ChunkingStrategy` | `FixedSizeChunking`, `SentenceTokenCapChunking`, `ContextualChunking`, `CallableChunking` |
| 3 | Extraction | `ExtractionStrategy` | `HybridExtraction` |
| 4 | Resolution | `ResolutionStrategy` | `ExactMatchResolution`, `DescriptionMergeResolution` |
| 5 | Retrieval | `RetrievalStrategy` | `LocalRetrieval`, `MultiPathRetrieval` |
| 6 | Reranking | `RerankingStrategy` | `CosineReranker` |

---

## 1. LoaderStrategy

Reads raw text from a data source.

### ABC

```python
from graphrag_sdk import LoaderStrategy

class LoaderStrategy(ABC):
    @abstractmethod
    async def load(self, source: str, ctx: Context) -> DocumentOutput:
        """Load text from the given source identifier."""
        ...
```

### Built-in: TextLoader

Reads plain text and markdown files.

```python
from graphrag_sdk.ingestion.loaders.text_loader import TextLoader

loader = TextLoader(encoding="utf-8")  # default encoding
```

### Built-in: PdfLoader

Extracts text from PDF files. Requires `pip install graphrag-sdk[pdf]`.

```python
from graphrag_sdk.ingestion.loaders.pdf_loader import PdfLoader

loader = PdfLoader()
```

### Default Behavior

If no loader is specified in `ingest()`:
- `.pdf` files use `PdfLoader`
- Everything else uses `TextLoader`
- If `text=` is passed directly, the loader is skipped

### Writing Your Own

```python
class HtmlLoader(LoaderStrategy):
    async def load(self, source: str, ctx: Context) -> DocumentOutput:
        from bs4 import BeautifulSoup
        with open(source) as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        return DocumentOutput(text=soup.get_text())

await rag.ingest("page.html", loader=HtmlLoader())
```

---

## 2. ChunkingStrategy

Splits document text into overlapping chunks for processing.

### ABC

```python
from graphrag_sdk import ChunkingStrategy

class ChunkingStrategy(ABC):
    @abstractmethod
    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        """Split text into chunks."""
        ...
```

### Built-in: FixedSizeChunking

Fixed-size character windows with configurable overlap.

```python
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking

chunker = FixedSizeChunking(
    chunk_size=1000,    # characters per chunk (default: 1000)
    chunk_overlap=100,  # overlap between chunks (default: 100)
)
```

**Tuning guidance:**
- Default (`1000/100`) works well for general use
- Benchmark-winning config uses `1500/200` for richer extraction context
- Smaller chunks (500) for fine-grained retrieval, larger (2000) for broader context

### Built-in: SentenceTokenCapChunking

Splits at sentence boundaries (never mid-sentence) and enforces a hard token cap per chunk using tiktoken. No LLM or embedder required.

```python
from graphrag_sdk.ingestion.chunking_strategies.sentence_token_cap import SentenceTokenCapChunking

chunker = SentenceTokenCapChunking(
    max_tokens=512,         # max tokens per chunk (default: 512)
    overlap_sentences=2,    # sentences shared between chunks (default: 2)
    encoding_name="cl100k_base",  # tiktoken encoding (default: cl100k_base)
)
```

### Built-in: ContextualChunking

Sentence-boundary chunking with LLM-generated context prefixes prepended to each chunk (Anthropic's contextual retrieval approach). Improves retrieval for cross-chunk co-reference questions.

```python
from graphrag_sdk.ingestion.chunking_strategies.contextual_chunking import ContextualChunking

chunker = ContextualChunking(
    llm=my_llm,
    max_tokens=512,              # token cap per chunk (default: 512)
    overlap_sentences=2,         # sentence overlap (default: 2)
    max_document_tokens=16_000,  # truncation limit for the doc reference in prompts (default: 16000)
)
```

> **Cost note:** generates one LLM call per chunk at ingestion time.

### Built-in: CallableChunking (bring your own framework)

Adapts any `text -> list[str]` function into a chunking strategy. Use this to plug in **any** chunking library -- LlamaIndex, LangChain, Unstructured, spaCy, or your own logic -- without the SDK carrying those dependencies.

Works with sync functions, async functions, and callable classes.

```python
from graphrag_sdk.ingestion.chunking_strategies.callable_chunking import CallableChunking

# Plain function
chunker = CallableChunking(lambda text: text.split("\n\n"))
```

### Writing Your Own

```python
class SentenceChunking(ChunkingStrategy):
    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        import nltk
        sentences = nltk.sent_tokenize(text)
        chunks = []
        for i, sent in enumerate(sentences):
            chunks.append(TextChunk(text=sent, index=i))
        return TextChunks(chunks=chunks)
```

---

## 3. ExtractionStrategy

Extracts entities, relationships, and entity mentions from text chunks.

### ABC

```python
from graphrag_sdk import ExtractionStrategy

class ExtractionStrategy(ABC):
    @abstractmethod
    async def extract(
        self,
        chunks: TextChunks,
        schema: GraphSchema,
        ctx: Context,
    ) -> GraphData:
        """Extract graph data from text chunks."""
        ...
```

### Built-in: HybridExtraction

Composable 2-step extraction with pluggable entity NER and LLM relationship extraction.

**Step 1 -- Entity NER** (pluggable via `EntityExtractor`):
- **GLiNER2** (default): Local transformer NER, no API calls. Returns typed entities with confidence scores and character spans.
- **LLM**: Uses a structured NER prompt. Returns entities with confidence, spans, descriptions.
- **Custom**: Any model implementing `predict_entities(text, labels)`.

**Step 2 -- LLM Verify + Relationship Extraction**:
The LLM receives the pre-extracted entities and original text, verifies entities (removes invalid, adds missed), and extracts relationships with descriptions, keywords, confidence, and evidence spans.

**Fixed Ontology**: All entities are mapped to a fixed set of types (default 11 types). Types not in the ontology become `"Unknown"`. No post-hoc type resolution needed.

```python
from graphrag_sdk import HybridExtraction, EntityExtractor

# Default: GLiNER2 for step 1, LLM for step 2
extractor = HybridExtraction(llm=llm)

# With LLM for step 1 instead of GLiNER2
extractor = HybridExtraction(
    llm=llm,
    entity_extractor=EntityExtractor(llm=llm),
)

# With custom NER model
extractor = HybridExtraction(
    llm=llm,
    entity_extractor=EntityExtractor(model=my_spacy_ner),
)

# With coreference resolution
from graphrag_sdk import FastCorefResolver

extractor = HybridExtraction(
    llm=llm,
    coref_resolver=FastCorefResolver(),  # pip install graphrag-sdk[fastcoref]
)

# Custom entity types
extractor = HybridExtraction(
    llm=llm,
    entity_types=["Gene", "Protein", "Disease", "Drug"],
)
```

**Default entity types:** Person, Organization, Technology, Product, Location, Date, Event, Concept, Law, Dataset, Method.

**EntityExtractor parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | None | Custom NER model with `predict_entities(text, labels)` |
| `llm` | None | LLMInterface for LLM-based NER |
| `threshold` | 0.75 | Confidence threshold -- below this, entities become "Unknown" |
| `gliner_model_name` | `urchade/gliner_medium-v2.1` | GLiNER model (when using default mode) |

If neither `model` nor `llm` is provided, GLiNER2 is used (local, no API calls).

**Graph output:**
- All relationships use `RELATES` edge type. The original type (e.g. `WORKS_AT`) is in `properties["rel_type"]`.
- Entity IDs are type-qualified: `compute_entity_id("Paris", "Location")` -> `"paris__location"`.
- Character spans stored as `properties["spans"]` = `{chunk_id: [{start, end}]}` on both entities and relationships.
- Entity mentions (`MENTIONED_IN` edges) link entities to source chunks.

**When to use:** Default and only built-in extraction strategy. Composable via pluggable entity extractors and optional coreference resolution.

### Writing Your Own

```python
class MyExtraction(ExtractionStrategy):
    async def extract(self, chunks, schema, ctx):
        nodes, rels = [], []
        for chunk in chunks.chunks:
            # Your extraction logic
            ...
        return GraphData(nodes=nodes, relationships=rels)

await rag.ingest("doc.txt", extractor=MyExtraction())
```

---

## 4. ResolutionStrategy

Deduplicates entities that refer to the same real-world thing.

### ABC

```python
from graphrag_sdk import ResolutionStrategy

class ResolutionStrategy(ABC):
    @abstractmethod
    async def resolve(
        self,
        graph_data: GraphData,
        ctx: Context,
    ) -> ResolutionResult:
        """Deduplicate entities in the graph data."""
        ...
```

Returns `ResolutionResult` with deduplicated `nodes`, remapped `relationships`, and `merged_count`.

### Built-in: ExactMatchResolution

Deduplicates by exact property match (default: `id`). Fast, no LLM calls.

```python
from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution

resolver = ExactMatchResolution(
    resolve_property="id",  # property to match on (default: "id")
)
```

**When to use:** Default. Fast and deterministic. Works well when extraction produces consistent entity IDs.

### Built-in: DescriptionMergeResolution

Deduplicates by `(normalized name, label)` -- same-name entities with different labels (e.g. Person "Paris" vs Location "Paris") are kept separate. Merges descriptions:
- If fewer than `force_summary_threshold` descriptions: concatenates them
- If more: uses LLM to summarize into a single description

```python
from graphrag_sdk.ingestion.resolution_strategies.description_merge import DescriptionMergeResolution

resolver = DescriptionMergeResolution(
    llm=llm,                       # LLMInterface for summarization (optional)
    force_summary_threshold=3,     # Trigger LLM summary at this many descriptions (default: 3)
    max_summary_tokens=500,        # Max tokens for LLM summary (default: 500)
)
```

**When to use:** Multi-document ingestion where the same entity appears with different descriptions. Used in the benchmark-winning pipeline.

---

## 5. RetrievalStrategy

Searches the knowledge graph to find context for answering a question. Uses the **Template Method pattern**: `search()` handles validation and formatting, you implement `_execute()`.

### ABC

```python
from graphrag_sdk import RetrievalStrategy

class RetrievalStrategy(ABC):
    def __init__(self, graph_store=None, vector_store=None):
        self.graph_store = graph_store
        self.vector_store = vector_store

    async def search(self, query: str, ctx: Context = None, **kwargs) -> RetrieverResult:
        """Public API: validate -> execute -> format."""
        ...

    @abstractmethod
    async def _execute(self, query: str, ctx: Context, **kwargs) -> RawSearchResult:
        """Implement your search logic here."""
        ...
```

### Built-in: LocalRetrieval

Simple retrieval: vector search on chunks + 1-hop entity traversal.

```python
from graphrag_sdk.retrieval.strategies.local import LocalRetrieval

retriever = LocalRetrieval(
    graph_store=rag.graph_store,
    vector_store=rag.vector_store,
    embedder=embedder,
    top_k=5,                # chunks to retrieve (default: 5)
    include_entities=True,  # include connected entities (default: True)
)
```

**When to use:** Simple use cases, low latency requirements, small graphs.

### Built-in: MultiPathRetrieval

Production-grade retrieval with RELATES edge vector search, 2-path entity discovery, 4-path chunk retrieval, and cosine reranking. This is the **default** and the benchmark-winning strategy.

```python
from graphrag_sdk import MultiPathRetrieval

retriever = MultiPathRetrieval(
    graph_store=rag.graph_store,
    vector_store=rag.vector_store,
    embedder=embedder,
    llm=llm,
    chunk_top_k=15,         # final chunks after reranking (default: 15)
    max_entities=30,        # total entity cap (default: 30)
    max_relationships=20,   # max relationships in context (default: 20)
    rel_top_k=15,           # RELATES edge vector search results (default: 15)
    keyword_limit=10,       # max keywords from question (default: 10)
)

rag = GraphRAG(connection=conn, llm=llm, embedder=embedder, retrieval_strategy=retriever)
```

**When to use:** Default choice. Best accuracy on benchmark. Handles complex multi-hop questions.

### Writing Your Own

```python
class GlobalRetrieval(RetrievalStrategy):
    async def _execute(self, query, ctx, **kwargs):
        # Your custom retrieval logic
        results = await self.vector_store.search(query_vector, top_k=20)
        return RawSearchResult(records=results)

rag = GraphRAG(connection=conn, llm=llm, embedder=embedder, retrieval_strategy=GlobalRetrieval())
```

---

## 6. RerankingStrategy

Reranks retrieval results before they are passed to the LLM for answer generation.

### ABC

```python
from graphrag_sdk import RerankingStrategy

class RerankingStrategy(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        result: RetrieverResult,
        ctx: Context,
    ) -> RetrieverResult:
        """Rerank and filter retrieval results."""
        ...
```

### Built-in: CosineReranker

Reranks by cosine similarity between query embedding and item embeddings.

```python
from graphrag_sdk import CosineReranker

reranker = CosineReranker(
    embedder=embedder,
    top_k=15,          # keep top N results (default: 15)
)

result = await rag.query("question", reranker=reranker)
```

**Note:** `MultiPathRetrieval` already includes cosine reranking internally. The standalone `CosineReranker` is useful when using `LocalRetrieval` or a custom strategy.

### Writing Your Own

```python
class LLMReranker(RerankingStrategy):
    async def rerank(self, query, result, ctx):
        # Score each item with the LLM and sort
        scored = []
        for item in result.items:
            score = await self.llm.ainvoke(f"Rate relevance 0-10: {query} vs {item.content}")
            scored.append((float(score.content), item))
        scored.sort(reverse=True)
        return RetrieverResult(items=[item for _, item in scored[:10]])
```
