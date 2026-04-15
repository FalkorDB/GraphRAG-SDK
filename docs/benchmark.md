# Benchmarking

This guide explains how to evaluate the GraphRAG SDK against academic benchmarks and your own datasets. It covers the evaluation methodology, dataset format, step-by-step reproduction with the SDK API, pipeline configuration options, and our published results on the [GraphRAG-Bench](https://graphrag-bench.github.io) Novel leaderboard.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Datasets](#datasets)
4. [Reproducing with the SDK API](#reproducing-with-the-sdk-api)
5. [Pipeline Configuration](#pipeline-configuration)
6. [GraphRAG-Bench Novel Results](#graphrag-bench-novel-results)

---

## Overview

A benchmark run measures four dimensions:

| Dimension | What it captures |
|-----------|------------------|
| **Accuracy** | Answer quality against ground-truth references (ACC, ROUGE-L, coverage) |
| **Ingestion throughput** | Time to chunk, extract, resolve, and build the knowledge graph |
| **Query latency** | End-to-end time from question submission to final answer |
| **Graph statistics** | Nodes, edges, and chunks produced — a proxy for knowledge density |

### GraphRAG-Bench scoring system

We use the official [GraphRAG-Bench](https://graphrag-bench.github.io) evaluation methodology. The primary leaderboard metric is **ACC** (answer correctness × 100), computed as:

$$\text{ACC} = \bigl(0.75 \times \text{factuality\_F1} + 0.25 \times \text{semantic\_similarity}\bigr) \times 100$$

| Component | How it works |
|-----------|-------------|
| **Factuality F1** | An LLM decomposes both the generated answer and the ground truth into atomic statements, classifies each as TP / FP / FN, and computes F1 |
| **Semantic similarity** | Cosine similarity between answer and reference embeddings, scaled to \[0, 1\] |
| **ROUGE-L** | Longest common subsequence F1 — used for Fact Retrieval and Complex Reasoning |
| **Coverage score** | Fraction of reference facts present in the answer — used for Contextual Summarize and Creative Generation |

---

## Prerequisites

### Infrastructure

Start a FalkorDB instance:

```bash
docker run -p 6379:6379 falkordb/falkordb:latest
```

### Environment Variables

Configure your LLM and embedding provider. Example for Azure OpenAI:

```bash
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
```

Any [LiteLLM-supported provider](https://docs.litellm.ai/docs/providers) works — OpenAI, Anthropic, local models, etc.

### Dependencies

```bash
pip install graphrag-sdk[litellm] gliner
```

---

## Datasets

We evaluate against [GraphRAG-Bench](https://graphrag-bench.github.io), an academic benchmark for graph-based retrieval-augmented generation. The datasets and questions are published in the GraphRAG-Bench project — download them from the official repository and place them in your dataset directory.

### Available datasets

| Dataset | Corpus | Questions | Docs | Questions |
|---------|--------|-----------|-----:|----------:|
| Novel (Full) | `novel.json` | `novel_questions.json` | 20 | 2,010 |
| Novel (Sample 100) | `novel.json` | `novel_questions_sample_100.json` | 20 | 100 |
| Medical | `medical.json` | `medical_questions.json` | 1 | 2,062 |

> **Note:** These files are not included in this repository. Download them from [GraphRAG-Bench](https://graphrag-bench.github.io) and place them in a local directory (e.g., `datasets/`).

### Data format

**Corpus** — a JSON array of documents:

```json
[
  {
    "corpus_name": "Novel-30752",
    "context": "Full text of the document..."
  }
]
```

**Questions** — a JSON array of evaluation items:

```json
[
  {
    "id": "Novel-73586ddc",
    "source": "Novel-44557",
    "question": "Which plant known as Erica vagans is also called...?",
    "answer": "Cornish heath",
    "question_type": "Fact Retrieval",
    "evidence": "The plant known scientifically as Erica vagans...",
    "evidence_relations": ["..."]
  }
]
```

Four question types are evaluated, each with different metrics:

| Question Type | Metrics Used |
|---------------|-------------|
| Fact Retrieval | ROUGE-L + answer\_correctness (ACC) |
| Complex Reasoning | ROUGE-L + answer\_correctness (ACC) |
| Contextual Summarize | answer\_correctness (ACC) + coverage\_score |
| Creative Generation | answer\_correctness (ACC) + coverage\_score |

To benchmark your own domain, create two JSON files following the same schema.

---

## Reproducing with the SDK API

The following walkthrough shows how to reproduce our benchmark results from scratch using the SDK's Python API. Following these exact steps with the same configuration and dataset will produce the results shown in the [GraphRAG-Bench Novel Results](#graphrag-bench-novel-results) section.

### Step 1 — Initialize providers

```python
import asyncio
import json
from graphrag_sdk import ConnectionConfig, GraphRAG, LiteLLM, LiteLLMEmbedder

llm = LiteLLM(
    model="azure/gpt-4o-mini",
    api_key="...",
    api_base="https://your-resource.openai.azure.com/",
    api_version="2024-12-01-preview",
)

embedder = LiteLLMEmbedder(
    model="azure/text-embedding-3-small",
    api_key="...",
    api_base="https://your-resource.openai.azure.com/",
    api_version="2024-12-01-preview",
)
```

### Step 2 — Create a GraphRAG instance

```python
rag = GraphRAG(
    connection=ConnectionConfig(host="localhost", port=6379, graph_name="novel_bench"),
    llm=llm,
    embedder=embedder,
)
```

### Step 3 — Configure the ingestion pipeline

```python
from graphrag_sdk.core.context import Context
from graphrag_sdk.ingestion.chunking_strategies.sentence_token_cap import SentenceTokenCapChunking
from graphrag_sdk.ingestion.extraction_strategies.graph_extraction import GraphExtraction
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import GLiNERExtractor
from graphrag_sdk.ingestion.extraction_strategies.coref_resolvers import FastCorefResolver
from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy
from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution
from graphrag_sdk.ingestion.resolution_strategies.description_merge import DescriptionMergeResolution
from graphrag_sdk.ingestion.resolution_strategies.semantic_resolution import SemanticResolution
from graphrag_sdk.ingestion.resolution_strategies.llm_verified_resolution import LLMVerifiedResolution

chunker = SentenceTokenCapChunking(max_tokens=512, overlap_sentences=2)

extractor = GraphExtraction(
    llm=llm,
    entity_extractor=GLiNERExtractor(model_name="urchade/gliner_medium-v2.1"),
    coref_resolver=FastCorefResolver(),
)
```

### Step 4 — Chain resolution stages

Multiple resolution strategies run in sequence — each stage feeds its output into the next. Create a simple chained resolver:

```python
class ChainedResolution(ResolutionStrategy):
    """Run multiple resolution strategies in sequence."""
    def __init__(self, *stages):
        self._stages = stages

    async def resolve(self, graph_data, ctx):
        for stage in self._stages:
            graph_data = await stage.resolve(graph_data, ctx)
        return graph_data

resolver = ChainedResolution(
    ExactMatchResolution(resolve_property="name"),
    DescriptionMergeResolution(llm=llm),
    SemanticResolution(embedder=embedder, similarity_threshold=0.85),
    LLMVerifiedResolution(llm=llm, embedder=embedder, hard_threshold=0.95, soft_threshold=0.60),
)
```

### Step 5 — Ingest the corpus

```python
corpus = json.load(open("datasets/novel.json"))

for doc in corpus:
    await rag.ingest(
        doc["corpus_name"],
        text=doc["context"],
        chunker=chunker,
        extractor=extractor,
        resolver=resolver,
        ctx=Context(tenant_id=doc["corpus_name"]),
    )
```

### Step 6 — Finalize the graph

```python
await rag.finalize()
```

This removes null/stub entities, deduplicates across documents, embeds all entities and relationships, and creates vector and fulltext indexes in FalkorDB.

### Step 7 — Query

```python
questions = json.load(open("datasets/novel_questions.json"))

results = []
for q in questions:
    result = await rag.completion(q["question"])
    results.append({
        "question": q["question"],
        "answer": result.answer,
        "reference": q["answer"],
        "question_type": q["question_type"],
    })
```

### Step 8 — Evaluate with GraphRAG-Bench metrics

For each question, compute the official [GraphRAG-Bench](https://graphrag-bench.github.io) metrics:

**Answer Correctness (ACC)** — the primary leaderboard metric:

1. The LLM decomposes both the generated answer and the ground truth into atomic statements
2. Each statement is classified as TP (true positive), FP (false positive), or FN (false negative)
3. Factuality F1 is computed from the TP / FP / FN counts
4. Semantic similarity is the cosine similarity between answer and reference embeddings, scaled to [0, 1]
5. Final score: `ACC = 0.75 × factuality_F1 + 0.25 × semantic_similarity`

**ROUGE-L** — longest common subsequence F1 between answer and reference. Applied to Fact Retrieval and Complex Reasoning questions.

**Coverage Score** — the LLM extracts facts from the reference and checks what fraction is covered in the answer. Applied to Contextual Summarize and Creative Generation questions.

After running evaluation across all questions, aggregate the results:

```python
from collections import defaultdict

by_type = defaultdict(list)
for r in results:
    acc = compute_answer_correctness(llm, embedder, r["question"], r["answer"], r["reference"])
    by_type[r["question_type"]].append(acc)

# Per-type and overall ACC
for q_type, scores in by_type.items():
    avg_acc = sum(scores) / len(scores) * 100
    print(f"{q_type}: ACC = {avg_acc:.2f}")

all_scores = [s for scores in by_type.values() for s in scores]
overall_acc = sum(all_scores) / len(all_scores) * 100
print(f"Overall ACC: {overall_acc:.2f}")
```

This produces the accuracy tables, graph statistics, and leaderboard comparison shown in the [results section below](#graphrag-bench-novel-results).

---

## Pipeline Configuration

The ingestion and retrieval pipeline is fully composable. Each stage can be swapped independently.

### Chunking strategies

| Strategy | Description |
|----------|-------------|
| `SentenceTokenCapChunking(max_tokens, overlap_sentences)` | Splits on sentence boundaries with a configurable token cap. Best for most use cases. |

### Extraction strategies

| Strategy | Description |
|----------|-------------|
| `GraphExtraction(llm, entity_extractor=GLiNERExtractor(), coref_resolver=FastCorefResolver())` | Local NER (no API cost) + coreference resolution + LLM for relationships. Best accuracy. |
| `GraphExtraction(llm)` | LLM-only extraction. Higher API cost per document. |

### Resolution strategies

Chain multiple resolvers in sequence using a `ChainedResolution` wrapper (see [Step 4](#step-4--chain-resolution-stages)). Each stage feeds its deduplicated output into the next:

| Strategy | Description |
|----------|-------------|
| `ExactMatchResolution(resolve_property="name")` | Merges entities with identical names. Zero API cost. |
| `DescriptionMergeResolution(llm)` | LLM merges entities with similar descriptions. |
| `SemanticResolution(embedder, similarity_threshold)` | Cosine similarity on embeddings with hnswlib ANN index. No LLM calls. |
| `LLMVerifiedResolution(llm, embedder, hard_threshold, soft_threshold)` | Two-tier: auto-merge above hard threshold, LLM-verify between soft and hard. Uses Louvain community detection. |

**Winning chain** (used in our benchmark):

```python
resolver = ChainedResolution(
    ExactMatchResolution(resolve_property="name"),
    DescriptionMergeResolution(llm=llm),
    SemanticResolution(embedder=embedder, similarity_threshold=0.85),
    LLMVerifiedResolution(llm=llm, embedder=embedder, hard_threshold=0.95, soft_threshold=0.60),
)
```

### Retrieval strategies

| Strategy | Description |
|----------|-------------|
| `MultiPathRetrieval` (default) | Multi-path entity discovery, 2-hop graph expansion, chunk retrieval, cosine rerank. No configuration required. |

### Post-ingestion: `finalize()`

Always call `await rag.finalize()` after ingesting all documents:

- Removes null/stub entities
- Deduplicates across document boundaries
- Embeds all entities and relationships
- Creates vector and fulltext indexes in FalkorDB

---

## GraphRAG-Bench Novel Results

The following results were produced by running the pipeline described above on the complete [GraphRAG-Bench](https://graphrag-bench.github.io) Novel dataset (20 novels, 2,010 questions).

### Configuration

| Parameter | Value |
|-----------|-------|
| LLM | gpt-4o-mini (Azure OpenAI) |
| Embeddings | text-embedding-3-small (Azure OpenAI) |
| Chunking | SentenceTokenCapChunking — max\_tokens=512, overlap\_sentences=2 |
| Extraction | GLiNER v2.1 + FastCoref + LLM relationship extraction |
| Resolution | ExactMatch (name) → DescriptionMerge → Semantic (0.85) → LLMVerified (0.95 / 0.60) |
| Retrieval | MultiPathRetrieval |
| Corpus | `novel.json` — 20 novels, 4.7 MB |
| Questions | `novel_questions.json` — 2,010 questions |

### Accuracy (official GraphRAG-Bench ACC)

| Question Type | ACC (×100) | ROUGE-L | Coverage |
|---------------|----------:|--------:|---------:|
| Fact Retrieval | 65.22 | 35.95 | — |
| Complex Reasoning | 58.63 | 22.39 | — |
| Contextual Summarize | 69.54 | — | 55.21 |
| Creative Generation | 57.08 | — | 44.52 |
| **Overall** | **63.73** | — | — |

### Leaderboard comparison

| System | Fact Retrieval | Complex Reasoning | Contextual Summarize | Creative Generation | Overall |
|--------|------:|------:|------:|------:|------:|
| **FalkorDB GraphRAG-SDK** | **65.22** | **58.63** | **69.54** | **57.08** | **63.73** |
| AutoPrunedRetriever | 45.99 | 62.80 | 83.10 | 62.97 | 63.72 |
| G-Reasoner | 60.07 | 53.92 | 71.28 | 50.48 | 58.94 |
| HippoRAG2 | 60.14 | 53.38 | 64.10 | 48.28 | 56.48 |
| Fast-GraphRAG | 56.95 | 48.55 | 56.41 | 46.18 | 52.02 |
| MS-GraphRAG (local) | 49.29 | 50.93 | 64.40 | 39.10 | 50.93 |
| RAG (w/ rerank) | 60.92 | 42.93 | 51.30 | 38.26 | 48.35 |
| LightRAG | 58.62 | 49.07 | 48.85 | 23.80 | 45.09 |
| HippoRAG | 52.93 | 38.52 | 48.70 | 38.85 | 44.75 |

Source: [graphrag-bench.github.io](https://graphrag-bench.github.io) — Novel leaderboard.

### Graph statistics

| Metric | Value |
|--------|------:|
| Total nodes | 8,765 |
| Total edges | 25,895 |
| Total chunks | 2,782 |
| Documents | 20 |

### Timing

| Phase | Duration |
|-------|--------:|
| Avg. query latency | 3.6 s |

### Evaluation methodology

Scores use the official [GraphRAG-Bench](https://graphrag-bench.github.io) evaluation suite,
ported from `github.com/GraphRAG-Bench/GraphRAG-Benchmark/Evaluation`:

| Component | How it works | Judge LLM |
|-----------|-------------|-----------|
| **answer_correctness** | 0.75 × Factuality F1 + 0.25 × Semantic Similarity. The LLM decomposes both the answer and reference into atomic statements, classifies TP / FP / FN, and computes F1. Semantic similarity is cosine similarity of answer vs reference embeddings. | gpt-4o-mini |
| **rouge_score** | ROUGE-L F1 (used for Fact Retrieval & Complex Reasoning) | — (algorithmic) |
| **coverage_score** | The LLM extracts facts from the reference and checks which are covered in the answer (used for Contextual Summarize & Creative Generation) | gpt-4o-mini |

**ACC** reported on the leaderboard = `answer_correctness × 100`, averaged per question type.
