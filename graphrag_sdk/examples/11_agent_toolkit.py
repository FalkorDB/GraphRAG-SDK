"""
GraphRAG SDK -- Agent Toolkit
=============================
The framework-neutral agent surface: wrap a GraphRAG in GraphRAGToolkit,
store a few facts, then exercise every tool an agent would call --
graph_schema, graph_search, graph_entity, graph_answer, cypher_read --
and print the machine-readable tool_specs() adapters consume.

Prerequisites:
    docker run -p 6379:6379 falkordb/falkordb
    pip install graphrag-sdk[litellm]
    export OPENAI_API_KEY="sk-..."

More providers: see docs/providers.md (mirror 01_quickstart.py's setup).
"""

import asyncio
import json
import os

from graphrag_sdk import ConnectionConfig, GraphRAG, LiteLLM, LiteLLMEmbedder
from graphrag_sdk.tools import GraphRAGToolkit, ReadOnlyViolation

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("Set OPENAI_API_KEY before running this example.")

FACTS = [
    "Alice Johnson is a software engineer at Acme Corp in London.",
    "Bob Smith is the CTO of Acme Corp and Alice's manager.",
    "Acme Corp is headquartered in Berlin and builds cloud infrastructure.",
]


async def main():
    llm = LiteLLM(model="openai/gpt-5.5")
    embedder = LiteLLMEmbedder(model="openai/text-embedding-3-large", dimensions=256)

    rag = GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="agent_toolkit_demo"),
        llm=llm,
        embedder=embedder,
        embedding_dimension=256,
    )
    toolkit = GraphRAGToolkit(rag, finalize_policy="manual")

    try:
        # 1. Write path: remember facts, then flush once (finalize is O(graph size))
        for i, fact in enumerate(FACTS):
            stored = await toolkit.remember(fact, document_id=f"fact-{i}")
            print(stored.to_llm_text(), "\n")
        await toolkit.flush()

        # 2. graph_schema -- what does the graph contain?
        schema = await toolkit.schema()
        print("=== graph_schema ===\n" + schema.to_llm_text(max_chars=800), "\n")

        # 3. graph_search -- typed retrieval, no generation (the default agent mode)
        search = await toolkit.search("Who works at Acme Corp?", top_k=5)
        print("=== graph_search ===\n" + search.to_llm_text(max_chars=1200), "\n")

        # 4. graph_entity -- one entity's card
        entity = await toolkit.entity("Alice Johnson", hops=2)
        print("=== graph_entity ===\n" + entity.to_llm_text(max_chars=800), "\n")

        # 5. graph_answer -- full RAG with citations
        answer = await toolkit.answer("Who is the CTO of Acme Corp?")
        print("=== graph_answer ===\n" + answer.to_llm_text(max_chars=800), "\n")

        # 6. cypher_read -- guarded read-only escape hatch
        rows = await toolkit.cypher_read(
            "MATCH (e:__Entity__) RETURN e.name ORDER BY e.name LIMIT 5"
        )
        print("=== cypher_read ===\n" + rows.to_llm_text(max_chars=600), "\n")

        # ... and the guard rejecting a write attempt:
        try:
            await toolkit.cypher_read("CREATE (n:Hack) RETURN n")
        except ReadOnlyViolation as exc:
            print(f"Guard blocked write: {exc} (token={exc.offending_token})\n")

        # 7. tool_specs() -- what adapters (pydantic-ai, MCP, ...) consume
        spec = toolkit.tool_specs()[0]
        print("=== tool_specs()[0] ===")
        print(json.dumps(spec.model_dump(), indent=2)[:600], "...")
    finally:
        await rag.delete_all()  # demo cleanup: drop the example graph
        await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
