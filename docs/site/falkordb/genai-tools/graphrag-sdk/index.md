---
title: "GraphRAG-SDK"
nav_order: 1
description: "Build intelligent GraphRAG applications with FalkorDB and LLMs — async-first Python SDK with schema-guided extraction, hybrid retrieval, and incremental updates."
parent: "GenAI Tools"
has_children: true
redirect_from:
  - /graphrag_sdk.html
  - /graphrag_sdk
  - /graphrag-sdk.html
  - /graphrag-sdk
  - /genai-tools/graphrag-sdk.html
---

# GraphRAG-SDK

**Async-first Python SDK for building GraphRAG applications on FalkorDB.**

Ingest raw documents directly into a knowledge graph, query with retrieval that combines vector search, full-text search, Cypher generation, and relationship expansion, and get cited answers — every fact in every answer is traceable back to the source chunk that supported it.

- **PyPI:** `pip install graphrag-sdk[litellm]`
- **GitHub:** [FalkorDB/GraphRAG-SDK](https://github.com/FalkorDB/GraphRAG-SDK)
- **Latest:** v1.2.0 ([changelog](./changelog))

## Where to start

| Goal | Page |
|---|---|
| Run your first ingestion in 5 minutes | [Quickstart](./quickstart) |
| Understand what a knowledge graph + ontology gives you | [Concepts](./concepts/) |
| Solve a specific task (custom schema, PDF ingest, CI integration…) | [Guides](./guides/) |
| Look up a method or class signature | [API Reference](./api-reference/) |
| See what changed in this release | [Changelog](./changelog) |

## What it gives you

- **Schema-guided extraction.** Declare the entity and relation types you care about and the extractor produces a focused, queryable graph. Or skip the schema entirely and run open-world.
- **Auto-discovered ontologies (new in v1.2).** Bootstrap a schema from your corpus with one call — LLM-driven or grounded against a live catalog like DBpedia. See [Concepts → Ontology discovery](./concepts/ontology-discovery).
- **Hybrid retrieval.** Vector search, full-text search, LLM-generated Cypher, and graph-walks combined in one pipeline.
- **Cited answers.** `completion()` returns the retrieval trail alongside the answer — `MENTIONS` edges connect every cited entity to the chunks that mentioned it.
- **Incremental updates.** Re-sync individual documents without rebuilding. `apply_changes(added=..., modified=..., deleted=...)` is the canonical CI hook on PR merge. See [Concepts → Incremental updates](./concepts/incremental-updates).
- **Provider-agnostic.** Any model reachable via [LiteLLM](https://docs.litellm.ai/) works out of the box (OpenAI, Anthropic, Gemini, Azure, Ollama, Groq, Cohere, OpenRouter…). Plug a custom provider in behind the `LLMInterface` / `Embedder` ABCs.
- **Multi-tenant.** `graph_name` on the connection gives you per-tenant isolation inside one FalkorDB instance.

## 60-second example

```python
import asyncio
from graphrag_sdk import GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder

async def main():
    async with GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="my_graph"),
        llm=LiteLLM(model="openai/gpt-4o-mini"),
        embedder=LiteLLMEmbedder(model="openai/text-embedding-3-large"),
    ) as rag:
        await rag.ingest(text="Alice Johnson is a software engineer at Acme Corp in London.")
        await rag.finalize()

        answer = await rag.completion("Where does Alice work?")
        print(answer.answer)

asyncio.run(main())
```

Full walkthrough — install, start FalkorDB, define a schema, query with provenance — in the [Quickstart](./quickstart).

## How it works

1. **Ingest.** Documents (text, Markdown, PDF) are loaded and chunked. An LLM-driven extractor reads each chunk and emits entities and relations, optionally constrained by the ontology you declared. Surface forms of the same entity are merged via a resolver.
2. **Index.** Chunks and entities are embedded and stored alongside the graph. Vector, full-text, and graph-pattern indexes are built.
3. **Retrieve.** A retrieval pipeline combines vector search, full-text search, optional LLM-generated Cypher, and multi-hop graph walks. Results are reranked by cosine similarity to the question.
4. **Generate.** Retrieved context is fed to the LLM along with the question. Every answer is traceable back to source chunks via `MENTIONS` edges.

A deeper walk-through lives under [Concepts](./concepts/).

> 📓 New to knowledge graphs? Start with [Understanding Ontologies and Knowledge Graphs](https://www.falkordb.com/blog/understanding-ontologies-knowledge-graph-schemas/) for the mental model, then come back to [Concepts → Ontology](./concepts/ontology).

## Migrating from v1.1

The v1.2 release renamed the schema vocabulary. The old names still work but emit `DeprecationWarning`:

| Old (≤ v1.1) | New (v1.2+) |
|---|---|
| `GraphSchema` | `Ontology` |
| `EntityType` | `Entity` |
| `RelationType` | `Relation` |
| `PropertyType` | `Attribute` |
| `SchemaModificationNotAllowedError` | `OntologyModificationNotAllowedError` |
| `GraphRAG(..., schema=...)` | `GraphRAG(..., ontology=...)` |
| `rag.schema` (property) | `rag.ontology` |

Replace imports and keyword arguments — no behavioural change. See [Concepts → Ontology](./concepts/ontology) for the new API.

{% include faq_accordion.html
  title="Frequently Asked Questions"
  q1="Do I need a running FalkorDB to use the SDK?"
  a1="Yes. Run one locally with `docker run -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest`, or use [FalkorDB Cloud](https://app.falkordb.cloud) for a managed instance."
  q2="Which Python version is required?"
  a2="Python 3.10 or newer. The SDK uses `match`/`case` and PEP 604 union syntax (`str | None`) in its public types."
  q3="Which LLM and embedder providers are supported?"
  a3="Any model reachable via [LiteLLM](https://docs.litellm.ai/) — that's OpenAI, Azure OpenAI, Anthropic, Gemini, Groq, Cohere, Ollama, OpenRouter, and 100+ others. Plug in a custom provider by subclassing `LLMInterface` and `Embedder`."
  q4="Can I use it without a schema?"
  a4="Yes. Without an ontology the extractor runs open-world — it produces entities and relations as it sees them. You can also auto-discover a draft schema from your corpus with `Ontology.from_sources(...)` (new in v1.2). See [Concepts → Ontology discovery](./concepts/ontology-discovery)."
  q5="Does the SDK support multi-tenancy?"
  a5="Yes. The `graph_name` argument on `ConnectionConfig` gives each tenant a separate graph inside one FalkorDB instance. Indexes and ontology live alongside the data graph at `<graph_name>__ontology`."
%}
