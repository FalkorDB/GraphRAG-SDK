"""
GraphRAG SDK -- Quick Start
===============================
Minimal example: ingest a short text and query it.
No external files needed -- just an LLM API key and FalkorDB.

Prerequisites:
    docker run -p 6379:6379 falkordb/falkordb
    pip install graphrag-sdk[litellm]

Option A -- OpenAI (simplest, 1 env var):
    export OPENAI_API_KEY="sk-..."

Option B -- Azure OpenAI:
    export AZURE_OPENAI_API_KEY="..."
    export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
    export AZURE_OPENAI_API_VERSION="2024-12-01-preview"

More providers: see docs/providers.md
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


def get_providers():
    """Configure LLM and embedder providers.

    Uncomment the section matching your provider.
    """

    # ── Option A: OpenAI (default) ──────────────────────────────
    llm = LiteLLM(model="openai/gpt-5.5")
    embedder = LiteLLMEmbedder(model="openai/text-embedding-3-large", dimensions=256)

    # ── Option B: Azure OpenAI ──────────────────────────────────
    # llm = LiteLLM(
    #     model=f"azure/{os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4.1')}",
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #     api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #     api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    # )
    # embedder = LiteLLMEmbedder(
    #     model=f"azure/{os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-ada-002')}",
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #     api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #     api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    # )

    return llm, embedder


async def main():
    llm, embedder = get_providers()

    async with GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="quickstart"),
        llm=llm,
        embedder=embedder,
        embedding_dimension=256,
    ) as rag:
        # 1. Ingest text
        result = await rag.ingest(text=TEXT, document_id="quickstart_doc")
        print(f"Ingested: {result.nodes_created} nodes, {result.relationships_created} edges")

        # 2. Finalize (dedup + embeddings + indexes)
        await rag.finalize()

        # 3. Retrieve context only (no LLM answer generation)
        context = await rag.retrieve("Where does Alice work?")
        print(f"\nRetrieved {len(context.items)} context items")

        # 4. Full RAG: retrieve + generate answer
        for question in [
            "Where does Alice work?",
            "Who is the CTO of Acme Corp?",
            "When was Acme Corp founded?",
        ]:
            answer = await rag.completion(question)
            print(f"\nQ: {question}")
            print(f"A: {answer.answer}")

        # 5. Multi-turn conversation (native messages to LLM)
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
