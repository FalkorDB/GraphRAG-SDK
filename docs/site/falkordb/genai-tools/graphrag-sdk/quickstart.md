---
title: "Quickstart"
nav_order: 2
parent: "GraphRAG-SDK"
grand_parent: "GenAI Tools"
description: "Install GraphRAG-SDK, start FalkorDB, ingest a document, and query it with citations — in five minutes."
---

# Quickstart

Five-minute path from `pip install` to a working ingestion and a cited answer.

## 1. Install

```bash
pip install graphrag-sdk[litellm]
```

For PDF ingestion add the `pdf` extra:

```bash
pip install graphrag-sdk[litellm,pdf]
```

The `litellm` extra pulls in [LiteLLM](https://docs.litellm.ai/) so the same code can target OpenAI, Anthropic, Gemini, Azure, Ollama, Groq, Cohere, OpenRouter, and 100+ other providers.

## 2. Start FalkorDB

```bash
docker run -d -p 6379:6379 -p 3000:3000 --name falkordb falkordb/falkordb:latest
```

Port 6379 is the database; port 3000 hosts the web UI (open `http://localhost:3000`).

Or sign up for [FalkorDB Cloud](https://app.falkordb.cloud) and copy the connection string.

## 3. Set your provider key

```bash
export OPENAI_API_KEY="sk-..."
```

For other providers, set the matching variable — `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, etc. See the [LiteLLM providers list](https://docs.litellm.ai/docs/providers).

## 4. Ingest and query

```python
import asyncio
from graphrag_sdk import GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder

async def main():
    async with GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="my_graph"),
        llm=LiteLLM(model="openai/gpt-4o-mini"),
        embedder=LiteLLMEmbedder(model="openai/text-embedding-3-large"),
    ) as rag:
        # Ingest raw text (or pass a file path for .md / .txt / .pdf).
        result = await rag.ingest(
            text="Alice Johnson is a software engineer at Acme Corp in London. "
                 "Acme Corp was founded in 2005 and has offices in London and Berlin.",
            document_id="acme-overview",
        )
        print(f"Nodes: {result.nodes_created}, "
              f"Edges: {result.relationships_created}")

        # Finalize once after all ingests: deduplicates entities, backfills
        # entity/relation embeddings, builds indexes.
        await rag.finalize()

        # Full RAG: retrieve context, generate answer, return both.
        answer = await rag.completion("Where does Alice work and where is that company based?")
        print(answer.answer)

asyncio.run(main())
```

Expected output (model-dependent):

```
Nodes: 4, Edges: 4
Alice works at Acme Corp, which has offices in London and Berlin.
```

## 5. (Optional) Define a schema

Without a schema the extractor runs open-world. With one, the LLM is constrained to your declared entity and relation types — the resulting graph is tighter and more predictable.

```python
from graphrag_sdk import GraphRAG, Ontology, Entity, Relation

ontology = Ontology(
    entities=[
        Entity(label="Person", description="A human being"),
        Entity(label="Organization", description="A company or institution"),
        Entity(label="Location", description="A geographic location"),
    ],
    relations=[
        Relation(label="WORKS_AT", patterns=[("Person", "Organization")]),
        Relation(label="LOCATED_IN", patterns=[("Organization", "Location")]),
    ],
)

async with GraphRAG(
    connection=ConnectionConfig(host="localhost", graph_name="my_graph"),
    llm=LiteLLM(model="openai/gpt-4o-mini"),
    embedder=LiteLLMEmbedder(model="openai/text-embedding-3-large"),
    ontology=ontology,
) as rag:
    ...  # ingest / completion as above
```

Don't know what schema you need yet? Let the SDK draft one from your documents — see [Guides → Auto-discover a schema](./guides/auto-discover-schema).

## 6. (Optional) Inspect retrieval

`completion()` also exposes the retrieval trail used to produce the answer. Pass `return_context=True` to see which chunks supported each fact:

```python
answer = await rag.completion(
    "Where does Alice work?",
    return_context=True,
)
print(answer.answer)
for item in answer.retriever_result.items:
    print(f"- score={item.score:.3f}  {item.content[:80]}…")
```

## Next steps

- [Concepts → Ontology](./concepts/ontology) — entity/relation/attribute mental model.
- [Concepts → Ingestion pipeline](./concepts/ingestion-pipeline) — load → chunk → extract → resolve.
- [Concepts → Retrieval pipeline](./concepts/retrieval-pipeline) — how vector + full-text + Cypher + graph walks combine.
- [Concepts → Incremental updates](./concepts/incremental-updates) — re-sync documents on PR merge without rebuilding.
- [API Reference → GraphRAG](./api-reference/graphrag) — every public method, with parameters and return types.
