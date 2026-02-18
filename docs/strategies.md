# Strategy Reference

GraphRAG SDK uses the **Strategy pattern** for every algorithmic concern. Each concern has an abstract base class (ABC) with one or more built-in implementations. You can swap any implementation or write your own.

## Overview

| # | Concern | ABC | Built-in Implementations |
|---|---------|-----|------------------------|
| 1 | Loading | `LoaderStrategy` | `TextLoader`, `PdfLoader` |
| 2 | Chunking | `ChunkingStrategy` | `FixedSizeChunking` |
| 3 | Extraction | `ExtractionStrategy` | `SchemaGuidedExtraction`, `MergedExtraction` |
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

Extracts entities, relationships, and optionally fact triples from text chunks using an LLM.

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

### Built-in: SchemaGuidedExtraction

LLM-based extraction constrained to the provided schema. Prompts the LLM with entity types, relationship types, and schema patterns.

```python
from graphrag_sdk.ingestion.extraction_strategies.schema_guided import SchemaGuidedExtraction

extractor = SchemaGuidedExtraction(
    llm=llm,              # LLMInterface instance
    chunk_batch_size=1,   # chunks per LLM call (default: 1)
)
```

**When to use:** Default choice. Good accuracy, respects schema constraints, straightforward.

### Built-in: MergedExtraction

Combines two extraction approaches:
- **LightRAG-style**: Rich typed entity extraction with descriptions, delimiter-based parsing
- **HippoRAG-style**: Fact triples (subject-predicate-object), entity mentions, synonym candidates

Attaches `facts`, `mentions`, `extracted_entities`, and `extracted_relations` to the `GraphData` object via Pydantic's `extra="allow"`.

```python
from graphrag_sdk.ingestion.extraction_strategies.merged_extraction import MergedExtraction

extractor = MergedExtraction(
    llm=llm,                    # LLMInterface instance
    embedder=embedder,          # Embedder instance (optional, for fact embedding)
    enable_gleaning=False,      # Second LLM pass for missed entities (default: False)
    max_concurrency=None,       # Override LLM concurrency (default: uses LLM's setting)
)
```

**When to use:** Best accuracy. Used in the benchmark-winning pipeline. Produces richer graphs (facts, mentions, synonyms) but uses more LLM calls.

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

Deduplicates by normalized name (lowercase, stripped). Merges descriptions:
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

Production-grade 5-path retrieval with 2-hop expansion. This is the **default** and the benchmark-winning strategy.

```python
from graphrag_sdk import MultiPathRetrieval

retriever = MultiPathRetrieval(
    graph_store=rag.graph_store,
    vector_store=rag.vector_store,
    embedder=embedder,
    llm=llm,
    entity_top_k=5,         # entities per discovery path (default: 5)
    chunk_top_k=15,         # final chunks after reranking (default: 15)
    fact_top_k=15,          # facts via vector search (default: 15)
    max_entities=30,        # total entity cap (default: 30)
    max_relationships=20,   # max relationships in context (default: 20)
    keyword_limit=10,       # max keywords from question (default: 10)
    llm_rerank=False,       # LLM reranking adds ~1s, marginal gain (default: False)
)

rag = GraphRAG(connection=conn, llm=llm, embedder=embedder, retrieval_strategy=retriever)
```

**Pipeline:**
1. Extract keywords from question (LLM)
2. Batch embed keywords + question
3. 5-path entity discovery (vector, CONTAINS, fulltext, question-vector, synonyms)
4. 2-hop relationship expansion for top entities
5. 5-path chunk retrieval (fulltext, vector, MENTIONED_IN, CONTAINS, 2-hop)
6. Cosine reranking of candidate chunks
7. Fact retrieval via vector search
8. Structured context assembly

**When to use:** Default choice. Best accuracy (88.2% on benchmark). Handles complex multi-hop questions.

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
