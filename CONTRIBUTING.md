# Contributing to GraphRAG SDK v2

Thank you for your interest in contributing to the GraphRAG SDK v2. This guide covers setup, testing, code conventions, and how to extend the SDK with custom strategies.

---

## 1. Development Setup

Clone the repository and create a virtual environment:

```bash
git clone <repo-url>
cd GraphRAG-SDKv2-DEMO
python -m venv .venv
source .venv/bin/activate
```

Install the SDK in editable mode with development dependencies:

```bash
pip install -e "graphrag_sdk[dev]"
```

You will need a running FalkorDB instance for integration work. The easiest way is via Docker:

```bash
docker run -p 6379:6379 falkordb/falkordb
```

This exposes FalkorDB on the default Redis port (6379). No additional configuration is required for local development.

---

## 2. Running Tests

Run the full test suite with:

```bash
python -m pytest graphrag_sdk/tests/ -q
```

There are 490+ tests covering the ingestion pipeline, the GraphRAG facade, extraction strategies, resolution strategies, retrieval strategies, storage layers, and utilities. All tests use mock providers, so no live LLM or database connection is needed to run them.

---

## 3. Code Style

- **Python 3.10+** is the minimum supported version.
- **Pydantic v2** is used throughout. The core `DataModel` uses `extra="allow"` so that dynamic attributes (such as mention metadata) can be attached at runtime.
- **Async-first design**: all public methods are implemented as `async` coroutines, with thin synchronous wrappers provided for convenience. When contributing new functionality, implement the async version first and add a sync wrapper if needed.
- **Type hints everywhere**: every function signature, return type, and class attribute should be fully annotated. Avoid `Any` unless strictly necessary.

---

## 4. Project Structure

```
graphrag_sdk/
  src/graphrag_sdk/          -- SDK source code
    api/                     -- GraphRAG facade (main.py)
    core/                    -- connection, models, providers, context, exceptions
    ingestion/               -- pipeline, loaders, chunking, extraction strategies, resolution strategies
    retrieval/               -- retrieval and reranking strategies
    storage/                 -- graph_store.py, vector_store.py
    utils/                   -- utilities (Cypher helpers, graph visualization, etc.)
  tests/                     -- all tests (mock-based, no external services required)
docs/                        -- documentation
```

Key entry points:

- `graphrag_sdk/src/graphrag_sdk/api/main.py` -- the `GraphRAG` class that serves as the primary facade for ingestion and querying.
- `graphrag_sdk/src/graphrag_sdk/ingestion/pipeline.py` -- the 9-step ingestion pipeline (Load, Chunk, Lexical Graph, Extract, Prune, Resolve, Write, Mentions, Index Chunks).
- `graphrag_sdk/src/graphrag_sdk/retrieval/` -- retrieval strategies including multi-path retrieval.
- `graphrag_sdk/src/graphrag_sdk/storage/` -- graph store (FalkorDB) and vector store abstractions.

---

## 5. Commit Rules

- **All code changes must achieve greater than 85% benchmark accuracy before they can be committed.** The benchmark is a 100-question evaluation over a 20-document novel corpus. Run it and verify your score before proposing a commit.
- **Always ask before committing.** Do not create commits autonomously. Confirm with the project maintainer that the changes are ready and that benchmark results meet the threshold.
- Write clear, descriptive commit messages that explain *why* the change was made, not just *what* changed.

---

## 6. Adding Strategies

The SDK is designed to be extended via an Abstract Base Class (ABC) pattern. There are three main extension points: extraction strategies, resolution strategies, and retrieval strategies.

### General Pattern

1. Identify the relevant ABC for the kind of strategy you want to add (extraction, resolution, or retrieval).
2. Create a new class that subclasses the ABC.
3. Implement all required abstract methods.
4. Pass your custom strategy to `ingest()` (for extraction/resolution) or to the `GraphRAG` constructor (for retrieval).

### Example: Custom Extraction Strategy

```python
from graphrag_sdk import ExtractionStrategy, GraphData, GraphSchema, TextChunks, Context


class MyCustomExtraction(ExtractionStrategy):
    """A custom extraction strategy that implements domain-specific entity/relation extraction."""

    async def extract(self, chunks: TextChunks, schema: GraphSchema, ctx: Context) -> GraphData:
        # Your extraction logic here.
        # Process the input chunks and return a GraphData containing
        # GraphNode and GraphRelationship objects.
        ...
```

Then use it during ingestion:

```python
from graphrag_sdk import GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder

rag = GraphRAG(
    connection=ConnectionConfig(host="localhost", graph_name="my_graph"),
    llm=LiteLLM(model="azure/gpt-4.1"),
    embedder=LiteLLMEmbedder(model="azure/text-embedding-ada-002"),
)
await rag.ingest("document.txt", extractor=MyCustomExtraction())
```

The same pattern applies to resolution strategies (subclass `ResolutionStrategy`) and retrieval strategies (subclass `RetrievalStrategy` and pass it to the `GraphRAG` constructor or query method).

---

If you have questions or run into issues, open an issue in the repository or reach out to the maintainers.
