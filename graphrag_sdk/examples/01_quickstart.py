"""
GraphRAG SDK -- Quick Start
===============================
Minimal example: ingest a short text and query it.
No external files needed -- just an LLM API key and FalkorDB.

Prerequisites:
    docker run -p 6379:6379 falkordb/falkordb
    pip install graphrag-sdk[litellm]
    export AZURE_OPENAI_API_KEY="..."
    export AZURE_OPENAI_ENDPOINT="..."

Alternative providers (replace the LiteLLM(...) calls below):
    OpenAI direct:   LiteLLM(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    Anthropic:       LiteLLM(model="anthropic/claude-sonnet-4-20250514", api_key=...)
    OpenRouter:      OpenRouterLLM(model="openai/gpt-4o", api_key=...)
    See docs/providers.md for the full configuration guide.
"""

import asyncio
import os

from graphrag_sdk import GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder

TEXT = (
    "Alice Johnson is a software engineer at Acme Corp in London. "
    "She leads the backend team and reports to Bob Smith, the CTO. "
    "Acme Corp was founded in 2015 and specializes in cloud infrastructure. "
    "The company recently opened a new office in Berlin, managed by Clara Wei."
)


async def main():
    # 1. Configure providers
    llm = LiteLLM(
        model=f"azure/{os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4.1')}",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )
    embedder = LiteLLMEmbedder(
        model=f"azure/{os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-ada-002')}",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )

    # 2. Create GraphRAG instance
    rag = GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="quickstart"),
        llm=llm,
        embedder=embedder,
    )

    # 3. Ingest text
    result = await rag.ingest("quickstart_doc", text=TEXT)
    print(f"Ingested: {result.nodes_created} nodes, {result.relationships_created} edges")

    # 4. Retrieve context only (no LLM answer generation)
    context = await rag.retrieve("Where does Alice work?")
    print(f"\nRetrieved {len(context.items)} context items")

    # 5. Full RAG: retrieve + generate answer
    for question in [
        "Where does Alice work?",
        "Who is the CTO of Acme Corp?",
        "When was Acme Corp founded?",
    ]:
        answer = await rag.completion(question)
        print(f"\nQ: {question}")
        print(f"A: {answer.answer}")

    # 6. Multi-turn conversation (native messages to LLM)
    from graphrag_sdk import ChatMessage

    followup = await rag.completion(
        "What is her role there?",
        history=[
            ChatMessage(role="user", content="Where does Alice work?"),
            ChatMessage(role="assistant", content="Alice works at Acme Corp in London."),
        ],
    )
    print(f"\nQ: What is her role there?")
    print(f"A: {followup.answer}")


if __name__ == "__main__":
    asyncio.run(main())
