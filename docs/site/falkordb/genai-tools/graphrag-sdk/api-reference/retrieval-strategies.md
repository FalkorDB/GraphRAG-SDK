---
title: "Retrieval strategies"
nav_order: 7
parent: "API Reference"
grand_parent: "GraphRAG-SDK"
description: "RetrievalStrategy, RerankingStrategy ABCs and the bundled MultiPathRetrieval and CosineReranker."
---

# Retrieval strategies

Module: `graphrag_sdk`  ·  Submodule: `graphrag_sdk.retrieval`

ABCs and built-ins for the retrieval and reranking stages.

---

## `RetrievalStrategy` (ABC)

```python
class RetrievalStrategy(ABC):
    @abstractmethod
    async def search(
        self,
        question: str,
        ctx: Context | None = None,
    ) -> RetrieverResult: ...
```

Subclasses implement `search(question)` and return a `RetrieverResult` of items (content, metadata, score).

---

## `MultiPathRetrieval`

The default retrieval strategy. Combines vector search, full-text search, optional Cypher generation, and graph traversal.

```python
class MultiPathRetrieval(RetrievalStrategy):
    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        embedder: Embedder,
        llm: LLMInterface,
        ontology: Ontology,
        *,
        vector_top_k: int = 8,
        fulltext_top_k: int = 8,
        expand_hops: int = 1,
        enable_cypher: bool = True,
        max_total_items: int = 20,
    ) -> None
```

| Name | Type | Default | Description |
|---|---|---|---|
| `graph_store`, `vector_store`, `embedder`, `llm`, `ontology` | — | — required — | Wired by `GraphRAG`. |
| `vector_top_k` | `int` | `8` | Per-mode result cap for vector search. |
| `fulltext_top_k` | `int` | `8` | Per-mode result cap for full-text search. |
| `expand_hops` | `int` | `1` | Hops to walk from each seeded entity when gathering graph context. `0` disables expansion. |
| `enable_cypher` | `bool` | `True` | Run the LLM-generated-Cypher path. Disable for fully deterministic retrieval. |
| `max_total_items` | `int` | `20` | Cap on items returned after merging. Reranker scores then choose the top. |

### Search modes inside `MultiPathRetrieval`

1. **Vector** — embed the question, top-K chunks and entities by cosine in the HNSW index.
2. **Full-text** — FalkorDB FT search on entity names.
3. **Cypher** — LLM emits a Cypher query against the live ontology; executed read-only.
4. **Graph expansion** — walk `expand_hops` from each seeded entity.

Failures inside any mode are logged and skipped — retrieval degrades gracefully rather than raising.

---

## `RerankingStrategy` (ABC)

```python
class RerankingStrategy(ABC):
    @abstractmethod
    async def rerank(
        self,
        question: str,
        retriever_result: RetrieverResult,
        ctx: Context | None = None,
    ) -> RetrieverResult: ...
```

---

## `CosineReranker`

The default reranker. Re-scores every candidate by cosine similarity to the **question embedding** (not chunk-vs-chunk similarity), giving a single consistent score across passages that came from different retrieval modes.

```python
class CosineReranker(RerankingStrategy):
    def __init__(self, embedder: Embedder, *, top_n: int | None = None) -> None
```

| Name | Type | Default | Description |
|---|---|---|---|
| `embedder` | `Embedder` | — required — | Used to embed the question. |
| `top_n` | `int \| None` | `None` (= keep all) | Truncate to top-N after re-scoring. |

## See also

- [Concepts → Retrieval pipeline](../concepts/retrieval-pipeline) — why these modes are combined.
- [API Reference → GraphRAG](./graphrag) — `retrieve` and `completion` use these strategies.
