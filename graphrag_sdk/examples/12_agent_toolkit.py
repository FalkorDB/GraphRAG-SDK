"""
GraphRAG SDK -- Agent Toolkit
=============================
The graph tools your own agent calls: wrap a GraphRAG in GraphRAGToolkit,
ingest a few facts, then run each read tool an agent would use -- status,
lookup_entity, traverse, graph_walk, cypher_query, retrieve, run_skill --
cite the evidence numbers they return, and check the citations.

The same tools are served to MCP clients by ``graphrag-mcp``.

Prerequisites:
    docker run -p 6379:6379 falkordb/falkordb
    pip install graphrag-sdk[litellm]
    export OPENAI_API_KEY="sk-..."

More providers: see docs/providers.md (mirror 01_quickstart.py's setup).
"""

import asyncio
import json
import os
from uuid import uuid4

from graphrag_sdk import ConnectionConfig, GraphRAG, LiteLLM, LiteLLMEmbedder
from graphrag_sdk.tools import GraphRAGToolkit, ToolError

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
        # A fresh graph per run, so the cleanup below never drops one you keep
        connection=ConnectionConfig(
            host="localhost", graph_name=f"agent_toolkit_demo_{uuid4().hex[:8]}"
        ),
        llm=llm,
        embedder=embedder,
        embedding_dimension=256,
    )
    toolkit = GraphRAGToolkit(rag)

    try:
        # 1. Write path: ingest facts, then finalize once (its cost grows with the graph)
        for i, fact in enumerate(FACTS):
            await toolkit.ingest(fact, document_id=f"fact-{i}")
        await toolkit.finalize()

        # 2. Read tools, one session (one evidence numbering) per conversation
        session = toolkit.new_session()
        status = await session.status()
        print("=== status ===\n" + status.to_llm_text(max_chars=800), "\n")

        names = await session.lookup_entity("Alice")
        if not names.matches:
            raise SystemExit("No entity named like 'Alice' was extracted; check the LLM setup.")
        alice = names.matches[0].name
        print(f"=== lookup_entity -> {alice} ===\n")

        walk = await session.traverse(alice, max_depth=2)
        print("=== traverse ===\n" + walk.to_llm_text(max_chars=1200), "\n")

        paths = await session.graph_walk(alice, goal="Berlin")
        print("=== graph_walk ===\n" + paths.to_llm_text(max_chars=800), "\n")

        rows = await session.cypher_query(
            "MATCH ()-[r:RELATES]->() RETURN r.rel_type AS relation, count(*) AS edges "
            "ORDER BY edges DESC"
        )
        print("=== cypher_query ===\n" + rows.to_llm_text(max_chars=600), "\n")

        found = await session.retrieve("Who is the CTO of Acme Corp?", top_k=5)
        print("=== retrieve ===\n" + found.to_llm_text(max_chars=1200), "\n")

        impact = await session.run_skill("impact_analysis", {"entity": "Acme Corp"})
        print("=== run_skill ===\n" + impact.to_llm_text(max_chars=800), "\n")

        # 3. The guard rejects writes, and errors carry a hint for the agent
        try:
            await session.cypher_query("CREATE (n:Hack) RETURN n")
        except ToolError as exc:
            print(f"Guard: {exc}\n")

        # 4. Your agent drafts an answer citing [n]; check it before replying
        if walk.edges:
            edge = walk.edges[0]
            draft = f"{edge.source} {edge.type} {edge.target} [{edge.n}]. Also [999]."
        else:
            draft = f"{alice} is in the graph. Also [999]."
        check = await session.check_citations(draft)
        print(f"grounded={check.grounded} cited={check.cited} invalid={check.invalid}\n")

        # 5. tool_specs() -- what agent frameworks and the MCP server consume
        spec = toolkit.tool_specs()[0]
        print("=== tool_specs()[0] ===")
        print(json.dumps(spec.model_dump(), indent=2)[:600], "...")
    finally:
        await rag.delete_all()  # demo cleanup: drop the example graph
        await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
