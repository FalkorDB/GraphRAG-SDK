<h1 align="center">GraphRAG-SDK</h1>
<h2 align="center">The simplest, most accurate GraphRAG framework built on FalkorDB</h2>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License: Apache 2.0"></a>
  <a href="https://github.com/FalkorDB/GraphRAG-SDK/actions"><img src="https://img.shields.io/github/actions/workflow/status/FalkorDB/GraphRAG-SDK/ci.yml?label=CI" alt="CI"></a>
  <a href="https://discord.gg/6M4QwDXn2w"><img src="https://img.shields.io/discord/1146782921294884966?label=Discord&logo=discord" alt="Discord"></a>
  <a href="https://github.com/FalkorDB/GraphRAG-SDK"><img src="https://img.shields.io/github/stars/FalkorDB/GraphRAG-SDK?style=social" alt="GitHub Stars"></a>
</p>

![knowledge-graph-construction-b](https://github.com/user-attachments/assets/69066899-0168-4e14-b359-f68c5b6c1e75)


Most GraphRAG systems work in demos and break under production constraints. GraphRAG SDK was built from real deployments around a simple idea: the retrieval harness matters more than the model. The result is a modular, benchmark-leading framework with predictable cost and sensible defaults that gets you from raw documents to cited answers quickly.

---

## Benchmarks
| Rank | System | Fact retrieval | Complex | Contextual | Creative | Overall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **`FalkorDB GraphRAG SDK ◄`** | **`65.22`** | **`58.63`** | **`69.54`** | **`57.08`** | **`63.73`** |
| 2 | AutoPrunedRetriever | 45.99 | 62.80 | 83.10 | 62.97 | 63.72 |
| 3 | G-Reasoner | 60.07 | 53.92 | 71.28 | 50.48 | 58.94 |
| 4 | HippoRAG2 | 60.14 | 53.38 | 64.10 | 48.28 | 56.48 |
| 5 | Fast-GraphRAG | 56.95 | 48.55 | 56.41 | 46.18 | 52.02 |
| 6 | MS-GraphRAG (local) | 49.29 | 50.93 | 64.40 | 39.10 | 50.93 |
| 7 | RAG (w rerank) | 60.92 | 42.93 | 51.30 | 38.26 | 48.35 |
| 8 | LightRAG | 58.62 | 49.07 | 48.85 | 23.80 | 45.09 |
| 9 | HippoRAG | 52.93 | 38.52 | 48.70 | 38.85 | 44.75 |

> FalkorDB scored with `gpt-4o-mini` (Azure OpenAI) on the [GraphRAG-Bench](https://graphrag-bench.github.io) Novel dataset — 20 novels, 2,010 questions, automated evaluation (ROUGE-L + answer-correctness with `gpt-4o-mini`). Competitor numbers are sourced from the GraphRAG-Bench published leaderboard. See [docs/benchmark.md](docs/benchmark.md) for full methodology and reproduction instructions.

---

![document-to-provenance-answer-flow-v1](https://github.com/user-attachments/assets/afd1607e-20e1-4954-95f2-274701f5d61d)


## Ingestion & Retrieval Pipeline

| Area | Item | Execution | Description |
| --- | --- | --- | --- |
| Ingestion | 1. Load | Sequential | Read raw text from files (PDF, TXT) or strings. |
| Ingestion | 2. Chunk | Sequential | Split content into overlapping text chunks. |
| Ingestion | 3. Lexical Graph | Sequential | Create `Document` and `Chunk` nodes with provenance edges. |
| Ingestion | 4. Extract | Sequential | Run GLiNER2 local NER and LLM-based relationship extraction. |
| Ingestion | 5. Quality Filter | Sequential | Remove invalid extracted nodes (empty IDs, malformed shape). |
| Ingestion | 6. Prune | Sequential | Filter nodes/relations against the schema; drop orphan relations. |
| Ingestion | 7. Resolve | Sequential | Deduplicate entities (exact match, semantic, LLM-verified). |
| Ingestion | 8. Write | Sequential | Persist graph updates with batched `MERGE` operations in FalkorDB. |
| Ingestion | 9a. Mentions | Parallel | Link entities back to source chunks. |
| Ingestion | 9b. Index | Parallel | Embed and index chunks for retrieval. |
| Retrieval | Vector search | Runtime | Finds semantically similar chunks. |
| Retrieval | Full-text search | Runtime | Matches exact terms and keywords. |
| Retrieval | Cypher queries | Runtime | Executes structured graph lookups. |
| Retrieval | Relationship expansion | Runtime | Traverses connected entities and context. |
| Retrieval | Cosine reranking | Runtime | Reorders candidates by relevance. |

> 💡 Every answer is traceable to its source chunks via `MENTIONS` edges. Pass `return_context=True` to `completion()` to get the retrieval trail alongside the answer.

## Quick Start

### 1. Install and start FalkorDB

```bash
pip install graphrag-sdk[litellm]
docker run -d -p 6379:6379 -p 3000:3000 --name falkordb falkordb/falkordb:latest
export OPENAI_API_KEY="sk-..."
```

> For PDF ingestion, install the `pdf` extra instead: `pip install graphrag-sdk[litellm,pdf]`.

### 2. Ingest a document

```python
import asyncio
from graphrag_sdk import GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder

async def main():
    async with GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="my_graph"),
        llm=LiteLLM(model="openai/gpt-5.4"),
        embedder=LiteLLMEmbedder(model="openai/text-embedding-3-large", dimensions=1536),
    ) as rag:
        # Ingest raw text (pass a file path with the `pdf` extra installed for PDFs)
        result = await rag.ingest(
            "my_doc",
            text="Alice Johnson is a software engineer at Acme Corp in London.",
        )
        print(f"Nodes: {result.nodes_created}, Edges: {result.relationships_created}")

        # Finalize: deduplicate entities, backfill embeddings, create indexes
        await rag.finalize()

        # Full RAG: retrieve + generate
        answer = await rag.completion("Where does Alice work?")
        print(answer.answer)

asyncio.run(main())
```

### 3. Define a schema (optional)

```python
from graphrag_sdk import GraphSchema, EntityType, RelationType, SchemaPattern

schema = GraphSchema(
    entities=[
        EntityType(label="Person", description="A human being"),
        EntityType(label="Organization", description="A company or institution"),
        EntityType(label="Location", description="A geographic location"),
    ],
    relations=[
        RelationType(label="WORKS_AT", description="Is employed by"),
        RelationType(label="LOCATED_IN", description="Is situated in"),
    ],
    patterns=[
        SchemaPattern(source="Person", relationship="WORKS_AT", target="Organization"),
        SchemaPattern(source="Organization", relationship="LOCATED_IN", target="Location"),
    ],
)

async with GraphRAG(
    connection=ConnectionConfig(host="localhost", graph_name="my_graph"),
    llm=LiteLLM(model="openai/gpt-5.4"),
    embedder=LiteLLMEmbedder(model="openai/text-embedding-3-large", dimensions=1536),
    schema=schema,
) as rag:
    ...  # ingest / completion as above
```
---

## Examples

| # | Example | What it demonstrates |
|---|---------|---------------------|
| 1 | [Quick Start](graphrag_sdk/examples/01_quickstart.py) | Minimal ingest + query |
| 2 | [PDF with Schema](graphrag_sdk/examples/02_pdf_with_schema.py) | PDF ingestion with custom entity types |
| 3 | [Custom Strategies](graphrag_sdk/examples/03_custom_strategies.py) | Benchmark-winning pipeline configuration |
| 4 | [Custom Provider](graphrag_sdk/examples/04_custom_provider.py) | Implement your own LLM/Embedder |
| 5 | [Notebook Demo](graphrag_sdk/examples/05_notebook_demo.ipynb) | Interactive walkthrough with provenance inspection |

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Step-by-step tutorial from install to first query |
| [Architecture](docs/architecture.md) | Pipeline design, graph schema, retrieval strategy |
| [Configuration](docs/configuration.md) | Connection, providers, and tuning reference |
| [Strategies](docs/strategies.md) | All ABCs and built-in implementations |
| [Providers](docs/providers.md) | LLM and embedder configuration guide |
| [Benchmark](docs/benchmark.md) | Methodology, results, and reproduction instructions |
| [API Reference](docs/api-reference.md) | Full API documentation |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and code style guidelines.

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

### Community

- [Discord](https://discord.gg/6M4QwDXn2w) -- Ask questions, share what you build
- [GitHub Discussions](https://github.com/FalkorDB/GraphRAG-SDK/discussions) -- Feature ideas, Q&A
- [Issues](https://github.com/FalkorDB/GraphRAG-SDK/issues) -- Bug reports and feature requests

---

## Citation

If you use GraphRAG SDK in your research, please cite:

```bibtex
@software{graphrag_sdk,
  title  = {GraphRAG SDK: A Modular Graph RAG Framework},
  author = {FalkorDB},
  year   = {2026},
  url    = {https://github.com/FalkorDB/GraphRAG-SDK},
}
```

---

## License

[Apache License 2.0](LICENSE)
