> 📦 Still on the v0.x API? Pin the legacy release: `pip install graphrag-sdk==0.8.2`.

<h1 align="center">GraphRAG-SDK</h1>
<h2 align="center">The simplest, most accurate GraphRAG framework built on FalkorDB</h2>

<p align="center"><b>Benchmark-leading accuracy</b> · <b>FalkorDB-fast</b> · <b>Multi-tenant</b> · <b>Graph traversal</b> · <b>5-minute setup</b></p>

<p align="center">
  <a href="https://pypi.org/project/graphrag-sdk/"><img src="https://img.shields.io/pypi/v/graphrag-sdk.svg?label=pypi" alt="PyPI version"></a>
  <a href="https://pepy.tech/projects/graphrag-sdk"><img src="https://img.shields.io/pepy/dt/graphrag-sdk?label=downloads&color=16A534" alt="Downloads"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License: Apache 2.0"></a>
  <a href="https://github.com/FalkorDB/GraphRAG-SDK/actions"><img src="https://img.shields.io/github/actions/workflow/status/FalkorDB/GraphRAG-SDK/ci.yml?label=CI" alt="CI"></a>
  <a href="https://discord.gg/6M4QwDXn2w"><img src="https://img.shields.io/discord/1146782921294884966?label=Discord&logo=discord" alt="Discord"></a>
  <a href="https://github.com/FalkorDB/GraphRAG-SDK"><img src="https://img.shields.io/github/stars/FalkorDB/GraphRAG-SDK?style=social" alt="GitHub Stars"></a>
</p>

![knowledge-graph-construction-b](https://github.com/user-attachments/assets/69066899-0168-4e14-b359-f68c5b6c1e75)


Most GraphRAG systems work in demos and break under production constraints. GraphRAG SDK was built from real deployments around a simple idea: the retrieval harness matters more than the model. The result is a modular, benchmark-leading framework with predictable cost and sensible defaults that gets you from raw documents to cited answers in under 5 minutes.

---

## Benchmarks
| Rank | System | Novel (Multi-Doc) | Medical (Single-Doc) | Overall |
| :--- | :--- | :---: | :---: | :---: |
| **1** | **FalkorDB GraphRAG SDK ◄** | **63.73** | **75.73** | **69.73** |
| 2 | G-Reasoner | 58.94 | 73.30 | 66.12 |
| 3 | AutoPrunedRetriever | 63.72 | 67.00 | 65.36 |
| 4 | HippoRAG2 | 56.48 | 64.85 | 60.67 |
| 5 | Fast-GraphRAG | 52.02 | 64.12 | 58.07 |
| 6 | **RAG (w rerank) (Vector RAG)** | **48.35** | **62.43** | **55.39** |
| 7 | LightRAG | 45.09 | 62.59 | 53.84 |
| 8 | HippoRAG | 44.75 | 59.08 | 51.92 |
| 9 | MS-GraphRAG (local) | 50.93 | 45.16 | 48.05 |

> Overall ACC on [GraphRAG-Bench](https://graphrag-bench.github.io) Novel (20 novels, 2,010 questions) and Medical (1 corpus, 2,062 questions) datasets. FalkorDB scored with `gpt-4o-mini` (Azure OpenAI); competitor numbers are from the published leaderboard. Overall = mean of Novel and Medical ACC. See [docs/benchmark.md](docs/benchmark.md) for per-category breakdowns, methodology, and reproduction instructions.

Vectors match similar chunks. The graph traverses relationships. Every answer cites its source.

---

## Quick Start

### 1. Install and start FalkorDB

```bash
pip install graphrag-sdk[litellm]
docker run -d -p 6379:6379 -p 3000:3000 --name falkordb falkordb/falkordb:latest
export OPENAI_API_KEY="sk-..."
```

> For PDF ingestion, install the `pdf` extra instead: `pip install graphrag-sdk[litellm,pdf]`.
> Ingestion sanitizes unsupported control characters in IDs and string properties before graph upserts, which helps avoid FalkorDB Cypher parse errors on noisy PDFs.

### 2. Ingest a document

```python
import asyncio
from graphrag_sdk import GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder

async def main():
    async with GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="my_graph"),  # graph_name = per-tenant isolation
        llm=LiteLLM(model="openai/gpt-5.5"),
        embedder=LiteLLMEmbedder(model="openai/text-embedding-3-large", dimensions=256),
    ) as rag:
        # Ingest raw text (pass a file path with the `pdf` extra installed for PDFs)
        result = await rag.ingest(
            text="Alice Johnson is a software engineer at Acme Corp in London.",
            document_id="my_doc",
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
from graphrag_sdk import GraphSchema, EntityType, RelationType

schema = GraphSchema(
    entities=[
        EntityType(label="Person", description="A human being"),
        EntityType(label="Organization", description="A company or institution"),
        EntityType(label="Location", description="A geographic location"),
    ],
    relations=[
        RelationType(label="WORKS_AT", description="Is employed by", patterns=[("Person", "Organization")]),
        RelationType(label="LOCATED_IN", description="Is situated in", patterns=[("Organization", "Location")]),
    ],
)

async with GraphRAG(
    connection=ConnectionConfig(host="localhost", graph_name="my_graph"),
    llm=LiteLLM(model="openai/gpt-5.5"),
    embedder=LiteLLMEmbedder(model="openai/text-embedding-3-large", dimensions=256),
    schema=schema,
) as rag:
    ...  # ingest / completion as above
```

<p align="center">
  <b>→ Full walkthrough: <a href="docs/getting-started.md">Getting Started</a></b><br/>
  <b>→ Benchmark-winning recipe: <a href="graphrag_sdk/examples/03_custom_strategies.py">Custom Strategies</a></b>
</p>

---

![document-to-provenance-answer-flow-v1](https://github.com/user-attachments/assets/afd1607e-20e1-4954-95f2-274701f5d61d)


## Ingestion & Retrieval Pipeline

| Area | Step | Cost |
| --- | --- | --- |
| Ingestion | Extract entities & relations | LLM |
| Ingestion | Resolve & deduplicate entities | LLM |
| Ingestion | Embed & index | LLM |
| Retrieval | Vector search | DB |
| Retrieval | Full-text search | DB |
| Retrieval | Text-to-Cypher *(experimental)* | LLM |
| Retrieval | Cypher queries | DB |
| Retrieval | Relationship expansion | DB |
| Retrieval | Cosine reranking | Local |

> 💡 Every answer is traceable to its source chunks via `MENTIONS` edges. Pass `return_context=True` to `completion()` to get the retrieval trail alongside the answer.

---

## Examples

> **Working starters — clone, plug in your source, ship.**

| # | Example | What you'll build |
|---|---------|-------------------|
| 1 | [Quick Start](graphrag_sdk/examples/01_quickstart.py) | Your first ingest-and-query loop in under 30 lines |
| 2 | [PDF with Schema](graphrag_sdk/examples/02_pdf_with_schema.py) | A PDF Q&A bot with your own entity and relation types |
| 3 | [Custom Strategies](graphrag_sdk/examples/03_custom_strategies.py) | The benchmark-winning pipeline, ready to drop in |
| 4 | [Custom Provider](graphrag_sdk/examples/04_custom_provider.py) | Plug in any LLM or embedder behind a clean interface |
| 5 | [Notebook Demo](graphrag_sdk/examples/05_notebook_demo.ipynb) | An interactive walkthrough that shows the provenance trail |

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

## Development Milestones
- 2024-06: First public release
- 2024-Q4: PDF ingestion and multi-provider LLMs
- 2025-Q1–Q2: Pluggable providers and pipeline tuning
- 2025-Q3: Sharper retrieval, deeper test coverage
- 🎉 2026-04: Version 1.0 is released
- 2026-Q2: Production observability; expand ingestion support — tables, structured data
- 2026-Q3: Introduce Agentic GraphRAG; complete PDF ingestion
- 2026-Q4: Smarter retrieval — dynamic traversal, temporal graph

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
