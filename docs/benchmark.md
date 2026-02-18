# Benchmark: Reproducing the 88.2% Accuracy Result

This document explains how to reproduce the GraphRAG SDK v2 benchmark that achieves **8.82/10 (88.2%)** accuracy on a 100-question evaluation over 20 Project Gutenberg novels.

## Overview

The benchmark measures:
- **Accuracy** -- LLM-as-Judge scoring (0-10) against ground-truth answers
- **Indexing throughput** -- Time to build the knowledge graph from documents
- **Query latency** -- End-to-end time from question to answer
- **Graph statistics** -- Nodes, edges, facts, synonyms, mentions

## Prerequisites

### Infrastructure

```bash
# Start FalkorDB
docker run -p 6379:6379 falkordb/falkordb
```

### Environment Variables

```bash
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_DEPLOYMENT="gpt-4.1"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-ada-002"
```

### Dependencies

```bash
pip install graphrag-sdk[litellm]
```

### Datasets

The benchmark uses two JSON files:

| File | Description |
|------|-------------|
| `Datasets/Corpus/novel.json` | 20 Project Gutenberg novel excerpts (4.7 MB) |
| `Datasets/Questions/novel_questions_sample_100.json` | 100 questions with ground-truth answers (69 KB) |

Each corpus entry has `corpus_name` and `context` fields. Each question entry has `question`, `answer`, and `question_type` fields.

Question types in the evaluation set:

| Type | Count | Description |
|------|-------|-------------|
| Fact Retrieval | 42 | Direct factual questions |
| Complex Reasoning | 37 | Multi-hop or inference-required |
| Contextual Summarization | 18 | Summarize themes, events, or relationships |
| Creative Generation | 3 | Open-ended creative responses |

## Reproducing with the SDK API

Below is a complete, self-contained script that reproduces the benchmark using only the SDK's public API. Copy-paste and run.

```python
"""
Reproduce the GraphRAG SDK v2 benchmark (88.2% accuracy).

Usage:
    python reproduce_benchmark.py                   # Full: index + evaluate
    python reproduce_benchmark.py --query-only      # Skip indexing, evaluate existing graph
"""

import asyncio
import json
import os
import time

from graphrag_sdk import (
    ConnectionConfig,
    EntityType,
    GraphRAG,
    GraphSchema,
    LiteLLM,
    LiteLLMEmbedder,
    RelationType,
    SchemaPattern,
)
from graphrag_sdk.core.context import Context
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
from graphrag_sdk.ingestion.extraction_strategies.merged_extraction import MergedExtraction
from graphrag_sdk.ingestion.resolution_strategies.description_merge import (
    DescriptionMergeResolution,
)

# ── Configuration ─────────────────────────────────────────────────────────

CORPUS_PATH = "Datasets/Corpus/novel.json"
QUESTIONS_PATH = "Datasets/Questions/novel_questions_sample_100.json"
GRAPH_NAME = "graphrag_benchmark"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
SIMILARITY_THRESHOLD = 0.9


# ── Schema (10 entity types, 14 relation types) ──────────────────────────

def create_schema() -> GraphSchema:
    return GraphSchema(
        entities=[
            EntityType(label="Person", description="A human being or fictional character"),
            EntityType(label="Place", description="A geographic location"),
            EntityType(label="Location", description="A specific place or setting"),
            EntityType(label="Character", description="A character in a literary work"),
            EntityType(label="Event", description="A significant event or occurrence"),
            EntityType(label="Object", description="A physical or abstract object"),
            EntityType(label="Concept", description="An abstract idea or theme"),
            EntityType(label="Organization", description="A group, institution, or company"),
            EntityType(label="Work", description="A literary, artistic, or creative work"),
            EntityType(label="Time", description="A time period or temporal reference"),
        ],
        relations=[
            RelationType(label="LOCATED_IN", description="Is located in a place"),
            RelationType(label="RELATED_TO", description="Has a relationship with"),
            RelationType(label="PART_OF", description="Is part of an organization or group"),
            RelationType(label="MARRIED_TO", description="Is married to"),
            RelationType(label="PARENT_OF", description="Is parent of"),
            RelationType(label="CHILD_OF", description="Is child of"),
            RelationType(label="FRIEND_OF", description="Is a friend of"),
            RelationType(label="ENEMY_OF", description="Is an enemy of"),
            RelationType(label="CREATED", description="Created or authored something"),
            RelationType(label="VISITED", description="Visited a place"),
            RelationType(label="MENTIONED_IN", description="Is mentioned in a work"),
            RelationType(label="ASSOCIATED_WITH", description="Is associated with"),
            RelationType(label="WORKS_AT", description="Works at or employed by"),
            RelationType(label="KNOWS", description="Knows or is acquainted with"),
        ],
        patterns=[
            SchemaPattern(source="Person", relationship="LOCATED_IN", target="Place"),
            SchemaPattern(source="Character", relationship="LOCATED_IN", target="Location"),
            SchemaPattern(source="Person", relationship="RELATED_TO", target="Person"),
            SchemaPattern(source="Character", relationship="RELATED_TO", target="Character"),
            SchemaPattern(source="Person", relationship="PART_OF", target="Organization"),
            SchemaPattern(source="Person", relationship="MARRIED_TO", target="Person"),
            SchemaPattern(source="Character", relationship="MARRIED_TO", target="Character"),
            SchemaPattern(source="Person", relationship="PARENT_OF", target="Person"),
            SchemaPattern(source="Person", relationship="CHILD_OF", target="Person"),
            SchemaPattern(source="Person", relationship="FRIEND_OF", target="Person"),
            SchemaPattern(source="Character", relationship="FRIEND_OF", target="Character"),
            SchemaPattern(source="Person", relationship="ENEMY_OF", target="Person"),
            SchemaPattern(source="Person", relationship="CREATED", target="Work"),
            SchemaPattern(source="Person", relationship="VISITED", target="Place"),
            SchemaPattern(source="Person", relationship="MENTIONED_IN", target="Work"),
            SchemaPattern(source="Person", relationship="ASSOCIATED_WITH", target="Event"),
            SchemaPattern(source="Object", relationship="ASSOCIATED_WITH", target="Person"),
        ],
    )


# ── Helpers ───────────────────────────────────────────────────────────────

def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header/footer markers."""
    if "Project Gutenberg" not in text:
        return text
    for marker in ["*** START", "***START"]:
        if marker in text:
            idx = text.find(marker)
            nl = text.find("\n", idx)
            if nl > idx:
                text = text[nl:].strip()
            break
    for marker in ["*** END", "***END", "End of Project Gutenberg"]:
        if marker in text:
            text = text[: text.find(marker)].strip()
            break
    return text


JUDGE_PROMPT = (
    "You are an expert evaluator comparing a generated answer against the ground truth.\n\n"
    "Question: {question}\n"
    "Ground Truth Answer: {ground_truth}\n"
    "Generated Answer: {generated_answer}\n\n"
    "Score the generated answer from 0-10 based on:\n"
    "- Factual correctness compared to ground truth\n"
    "- Completeness of the answer\n"
    "- Relevance to the question\n\n"
    "Return ONLY the numeric score (0-10), nothing else."
)


# ── Main ──────────────────────────────────────────────────────────────────

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="GraphRAG SDK v2 Benchmark Reproduction")
    parser.add_argument("--query-only", action="store_true", help="Skip indexing, query existing graph")
    args = parser.parse_args()

    # ── Step 1: Create providers ──────────────────────────────────────

    llm = LiteLLM(
        model=f"azure/{os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4.1')}",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0.0,
    )
    embedder = LiteLLMEmbedder(
        model=f"azure/{os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-ada-002')}",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )

    # ── Step 2: Create GraphRAG with schema ───────────────────────────

    rag = GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name=GRAPH_NAME),
        llm=llm,
        embedder=embedder,
        schema=create_schema(),
    )

    # ── Step 3: Ingest all 20 documents ───────────────────────────────

    if not args.query_only:
        with open(CORPUS_PATH) as f:
            corpus = json.load(f)

        print(f"Ingesting {len(corpus)} documents...")
        t0 = time.time()

        for i, doc in enumerate(corpus):
            text = strip_gutenberg_boilerplate(doc.get("context", ""))
            source_name = doc.get("corpus_name", f"doc_{i}")
            print(f"  [{i+1}/{len(corpus)}] {source_name} ({len(text):,} chars)")

            await rag.ingest(
                source_name,
                text=text,
                chunker=FixedSizeChunking(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
                extractor=MergedExtraction(llm=llm, embedder=embedder),
                resolver=DescriptionMergeResolution(llm=llm),
                ctx=Context(tenant_id="benchmark"),
            )

        # ── Step 4: Post-ingestion operations ─────────────────────────

        print("Post-ingestion: detecting synonyms...")
        synonym_count = await rag.detect_synonymy(similarity_threshold=SIMILARITY_THRESHOLD)
        print(f"  Created {synonym_count} SYNONYM edges")

        print("Post-ingestion: backfilling entity embeddings...")
        backfilled = await rag.vector_store.backfill_entity_embeddings()
        print(f"  Backfilled {backfilled} entities")

        indexing_time = time.time() - t0
        print(f"\nIndexing complete: {indexing_time:.0f}s ({indexing_time/60:.1f} min)")

    # ── Step 5: Print graph statistics ────────────────────────────────

    stats = await rag.graph_store.get_statistics()
    print(f"\nGraph Statistics:")
    print(f"  Nodes:     {stats['node_count']:,}")
    print(f"  Edges:     {stats['edge_count']:,}")
    print(f"  Facts:     {stats['fact_node_count']:,}")
    print(f"  Synonyms:  {stats['synonym_edge_count']:,}")
    print(f"  Mentions:  {stats['mention_edge_count']:,}")

    # ── Step 6: Evaluate accuracy (LLM-as-Judge) ─────────────────────

    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    print(f"\nEvaluating {len(questions)} questions...")
    scores = []
    latencies = []

    for i, q in enumerate(questions):
        t0 = time.time()
        result = await rag.query(q["question"])
        latency = time.time() - t0

        # LLM-as-Judge scoring
        judge_prompt = JUDGE_PROMPT.format(
            question=q["question"],
            ground_truth=q["answer"],
            generated_answer=result.answer,
        )
        judge_response = llm.invoke(judge_prompt)
        try:
            score = max(0, min(10, int(float(judge_response.content.strip()))))
        except ValueError:
            score = 0

        scores.append(score)
        latencies.append(latency)

        print(f"  [{i+1}/{len(questions)}] score={score}/10  latency={latency:.1f}s  "
              f"type={q.get('question_type', 'unknown')}")

    # ── Step 7: Print results ─────────────────────────────────────────

    mean_score = sum(scores) / len(scores)
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy:     {mean_score:.2f}/10 ({mean_score * 10:.1f}%)")
    print(f"  Questions:    {len(scores)}")
    print(f"  Query P50:    {p50:.2f}s")
    print(f"  Query P95:    {p95:.2f}s")

    # By question type
    from collections import defaultdict
    by_type = defaultdict(list)
    for q, s in zip(questions, scores):
        by_type[q.get("question_type", "unknown")].append(s)
    print(f"\n  By question type:")
    for qt, type_scores in sorted(by_type.items()):
        print(f"    {qt}: {sum(type_scores)/len(type_scores):.2f}/10 ({len(type_scores)} Qs)")


if __name__ == "__main__":
    asyncio.run(main())
```

## Winning Pipeline Configuration

The benchmark uses this specific combination of strategies:

### Ingestion

| Strategy | Class | Key Parameters |
|----------|-------|---------------|
| **Chunking** | `FixedSizeChunking` | `chunk_size=1500`, `chunk_overlap=200` |
| **Extraction** | `MergedExtraction` | Combines LightRAG typed extraction with HippoRAG fact triples + entity mentions |
| **Resolution** | `DescriptionMergeResolution` | LLM-assisted entity deduplication with description merging |
| **Post-ingestion** | `detect_synonymy()` | `similarity_threshold=0.9` -- creates SYNONYM edges between similar entities |
| **Post-ingestion** | `backfill_entity_embeddings()` | Embeds all entity names for vector search |

### Retrieval

| Strategy | Class | Key Parameters |
|----------|-------|---------------|
| **Retrieval** | `MultiPathRetrieval` (default) | 5-path entity discovery, 2-hop expansion, 5-path chunk retrieval |
| **Reranking** | Cosine (built-in) | `llm_rerank=False` -- LLM reranker adds ~1s latency with no accuracy gain |
| **Facts** | Vector search | `fact_top_k=15` |

### Models

| Role | Model | Details |
|------|-------|---------|
| **LLM** | GPT-4.1 (Azure OpenAI) | `temperature=0.0` |
| **Embeddings** | text-embedding-ada-002 | 1536 dimensions |

### Schema

**10 entity types:** Person, Place, Location, Character, Event, Object, Concept, Organization, Work, Time

**14 relationship types:** LOCATED_IN, RELATED_TO, PART_OF, MARRIED_TO, PARENT_OF, CHILD_OF, FRIEND_OF, ENEMY_OF, CREATED, VISITED, MENTIONED_IN, ASSOCIATED_WITH, WORKS_AT, KNOWS

### Key Ingestion API Calls

```python
# Each document: ingest with custom strategies
await rag.ingest(
    source_name,
    text=document_text,
    chunker=FixedSizeChunking(chunk_size=1500, chunk_overlap=200),
    extractor=MergedExtraction(llm=llm, embedder=embedder),
    resolver=DescriptionMergeResolution(llm=llm),
)

# After ALL documents are ingested:
await rag.detect_synonymy(similarity_threshold=0.9)
await rag.vector_store.backfill_entity_embeddings()
```

### Key Query API Call

```python
# Query uses the default MultiPathRetrieval (configured automatically)
result = await rag.query("What happened to character X?")
print(result.answer)

# With context inspection:
result = await rag.query("What happened to character X?", return_context=True)
print(result.retriever_result.items)  # See retrieved entities, facts, passages
```

## Results

### Accuracy

| Metric | Value |
|--------|-------|
| **Overall** | **8.82/10 (88.2%)** |
| Questions | 100 |
| Perfect scores (10) | ~60% |
| Failures (< 7) | ~9% |

### By Question Type

| Question Type | Count | Mean Score |
|---------------|-------|-----------|
| Complex Reasoning | 37 | 8.95/10 |
| Fact Retrieval | 42 | 8.83/10 |
| Contextual Summarization | 18 | 8.61/10 |
| Creative Generation | 3 | 8.33/10 |

### Graph Statistics

| Metric | Value |
|--------|-------|
| Total nodes | 114,849 |
| Total edges | 155,574 |
| Graph density | 1.35 |
| Fact nodes | 71,885 |
| Mention edges | 87,252 |
| Entity types | 14 |
| Relationship types | 15 |

### Performance

| Metric | Value |
|--------|-------|
| Indexing time | ~47 min for 20 docs |
| Throughput | ~0.42 docs/min |
| Query P50 | 5.43s |
| Query P95 | 9.15s |
| Mean query latency | 5.98s |

## Evaluation Methodology

### LLM-as-Judge

Each generated answer is scored against the ground-truth reference by GPT-4.1 using this rubric:

| Score | Meaning |
|-------|---------|
| 10 | Perfect match, complete and accurate |
| 7-9 | Mostly correct, minor omissions |
| 4-6 | Partially correct, key information missing |
| 1-3 | Mostly incorrect but contains some relevant information |
| 0 | Completely wrong or irrelevant |

Criteria: factual correctness, completeness, and relevance to the question.

### Variance

Approximately 52% of questions show score variance between runs due to LLM generation non-determinism (even with temperature=0). The ~88% accuracy represents a practical ceiling for single-pass RAG with GPT-4.1 on this corpus.

## Available Datasets

| Dataset | Corpus | Questions | Domain |
|---------|--------|-----------|--------|
| `novel.json` + `novel_questions_sample_100.json` | 20 docs, 4.7 MB | 100 Qs | Literature (Project Gutenberg) |
| `novel.json` + `novel_questions.json` | 20 docs, 4.7 MB | Full set | Literature |
| `medical.json` + `medical_questions.json` | Medical corpus, 1.1 MB | Full set | Healthcare |
