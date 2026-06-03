---
title: "Ingestion strategies"
nav_order: 6
parent: "API Reference"
grand_parent: "GraphRAG-SDK"
description: "Loaders, chunkers, extractors, resolvers — the pluggable strategy ABCs and bundled implementations."
---

# Ingestion strategies

Module: `graphrag_sdk`  ·  Submodule: `graphrag_sdk.ingestion`

Strategy ABCs and built-in implementations for the four ingestion stages.

---

## Loaders

### `LoaderStrategy` (ABC)

```python
class LoaderStrategy(ABC):
    @abstractmethod
    async def load(self, source: str, ctx: Context | None = None) -> DocumentOutput: ...
```

| Method | Purpose |
|---|---|
| `load(source, ctx=None) -> DocumentOutput` | **abstract** — read `source` and return text + optional structural elements. |

### Bundled loaders

| Class | Extension | Notes |
|---|---|---|
| `TextLoader` | `.txt` (default) | Passthrough. |
| `MarkdownLoader` | `.md`, `.markdown` | Parses headings/lists into `DocumentElement` tree. Requires `graphrag-sdk[markdown]`. |
| `PdfLoader` | `.pdf` | pypdf-based. Requires `graphrag-sdk[pdf]`. Use `[pdf-fast]` for PyMuPDF + table-aware extraction (AGPL). |

Default auto-selection inside `GraphRAG.ingest` / `update`: `.pdf` → `PdfLoader`, `.md` → `MarkdownLoader`, anything else → `TextLoader`. Pass `loader=` explicitly to override.

---

## Chunkers

### `ChunkingStrategy` (ABC)

```python
class ChunkingStrategy(ABC):
    @abstractmethod
    async def chunk(self, document: DocumentOutput, ctx: Context | None = None) -> TextChunks: ...
```

### Bundled chunkers

| Class | Init | Use when |
|---|---|---|
| `SentenceTokenCapChunking` | `(max_tokens=512, overlap_sentences=2, tokenizer="cl100k_base")` | **Default.** Sentence-aware. Never splits inside a sentence; adjacent chunks share 2 sentences of context so entity mentions are not severed. |
| `FixedSizeChunking` | `(chunk_size, chunk_overlap)` | Character-window chunking. Predictable size; ignores sentence boundaries. |
| `ContextualChunking` | `(llm, base_chunker=None, max_concurrency=...)` | Anthropic-style contextual retrieval — each chunk gets a one-line LLM summary prepended. Higher cost, better long-document retrieval. |
| `CallableChunking` | `(fn: Callable[[DocumentOutput], TextChunks])` | Adapter for an arbitrary function. |

---

## Extractors

### `ExtractionStrategy` (ABC)

```python
class ExtractionStrategy(ABC):
    @abstractmethod
    async def extract(
        self,
        chunks: TextChunks,
        ontology: Ontology,
        ctx: Context | None = None,
    ) -> GraphData: ...
```

### Bundled extractor

#### `GraphExtraction`

The default. Two-step LLM-driven extraction.

```python
class GraphExtraction(ExtractionStrategy):
    def __init__(
        self,
        llm: LLMInterface,
        entity_extractor: EntityExtractor | None = None,
        coref_resolver: CorefResolver | None = None,
        entity_types: list[str] | None = None,
        max_concurrency: int = 4,
    ) -> None
```

| Name | Type | Default | Description |
|---|---|---|---|
| `llm` | `LLMInterface` | — required — | Used for the relation-extraction step. |
| `entity_extractor` | `EntityExtractor \| None` | `GLiNERExtractor()` | Step-1 NER. Pass `LLMExtractor(llm)` for LLM-based entity detection instead. |
| `coref_resolver` | `CorefResolver \| None` | `None` | Optional coreference resolution before extraction — re-anchors pronouns. `FastCorefResolver()` is the bundled implementation. |
| `entity_types` | `list[str] \| None` | from ontology | Whitelist of labels for the entity step. |
| `max_concurrency` | `int` | `4` | Per-extraction concurrency cap. |

### Entity extractors (step 1)

| Class | Notes |
|---|---|
| `GLiNERExtractor` | Local NER model. No API calls. Default. Requires `graphrag-sdk[gliner]` (included in the base install). |
| `LLMExtractor` | LLM-based entity detection. Slower, more expensive, sometimes more accurate. |

### Coref resolvers (optional pre-extraction step)

| Class | Notes |
|---|---|
| `FastCorefResolver` | Coreference resolution via `fastcoref`. Requires `graphrag-sdk[fastcoref]`. |

---

## Resolvers

### `ResolutionStrategy` (ABC)

```python
class ResolutionStrategy(ABC):
    @abstractmethod
    async def resolve(
        self,
        graph_data: GraphData,
        ctx: Context | None = None,
    ) -> ResolutionResult: ...
```

### Bundled resolvers

| Class | Cost | Behaviour |
|---|---|---|
| `ExactMatchResolution(resolve_property="id")` | Zero — groupby on `(label, resolve_property)`. Default. |
| `SemanticResolution(embedder, similarity_threshold=0.85)` | One embedding per candidate. Catches near-duplicates (`"Acme Corp."` vs `"Acme Corporation"`). |
| `LLMVerifiedResolution(embedder, llm, similarity_threshold=0.7)` | One LLM call per candidate pair above threshold. Highest accuracy when entities have rich descriptions. |
| `DescriptionMergeResolution(llm)` | Post-merge step — one LLM call per surviving entity to merge descriptions. Doesn't change membership. |

---

## `IngestionPipeline`

Orchestrates the four strategies. You typically don't construct this yourself — `GraphRAG.ingest` / `update` build it from your overrides.

```python
class IngestionPipeline:
    def __init__(
        self,
        loader: LoaderStrategy,
        chunker: ChunkingStrategy,
        extractor: ExtractionStrategy,
        resolver: ResolutionStrategy,
        graph_store: GraphStore,
        vector_store: VectorStore,
        ontology: Ontology,
    ) -> None

    async def run(
        self,
        source: str,
        ctx: Context,
        *,
        text: str | None = None,
        document_info: DocumentInfo | None = None,
    ) -> IngestionResult
```

---

## `BackfillExecutor`

Internal engine behind `GraphRAG.add_attribute`, `backfill_entity`, `backfill_relation_pattern`. Documented for advanced custom backfills.

```python
class BackfillExecutor:
    def __init__(
        self,
        llm: LLMInterface,
        graph_store: GraphStore,
        concurrency: int = 4,
    ) -> None

    async def run(
        self,
        *,
        op_id: str,
        chunks: AsyncIterator[ChunkContext],
        prompt_builder: Callable[[ChunkContext], str],
        parse_fn: Callable[[str, ChunkContext], Any],
        merge_fn: Callable[[Any, ChunkContext], Awaitable[BackfillMergeStats]],
    ) -> BackfillResult
```

Per-chunk idempotency is via `extracted_ops` markers keyed on `op_id` — a re-run skips chunks already processed in a previous run.

## See also

- [Concepts → Ingestion pipeline](../concepts/ingestion-pipeline) — the four-stage mental model.
- [API Reference → Result types](./result-types) — `IngestionResult`, `ResolutionResult`, `BackfillResult`.
