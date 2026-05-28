"""
GraphRAG SDK -- Advanced Docling Loader
=======================================
This example demonstrates how to explicitly instantiate and configure the
DoclingLoader to parse rich document formats, passing advanced options to
the underlying docling DocumentConverter.

Prerequisites:
    docker run -p 6379:6379 falkordb/falkordb
    pip install graphrag-sdk[litellm,docling]

Usage:
    export OPENAI_API_KEY="sk-..."
    python 09_docling_advanced_loader.py
"""

import asyncio
from pathlib import Path

from graphrag_sdk import ConnectionConfig, GraphRAG, LiteLLM, LiteLLMEmbedder
from graphrag_sdk.ingestion.loaders.docling_loader import DoclingLoader


def get_providers():
    llm = LiteLLM(model="openai/gpt-4.1")
    embedder = LiteLLMEmbedder(model="openai/text-embedding-3-small")
    return llm, embedder


async def main():
    llm, embedder = get_providers()

    # Create a dummy markdown file for demonstration purposes
    # Note: In a real scenario, this would be a PDF, DOCX, XLSX, etc.
    dummy_file = Path("sample_docling_input.md")
    dummy_file.write_text(
        "# Advanced Analysis\n\n"
        "## Section 1\n"
        "This is a paragraph inside section 1.\n\n"
        "## Section 2\n"
        "This is another paragraph."
    )

    try:
        # Instantiate DoclingLoader with custom configuration
        # Any **kwargs are passed directly to docling's DocumentConverter
        advanced_loader = DoclingLoader(
            allowed_formats=["md", "docx", "pdf"],
            # Example docling kwargs (pipeline_options, etc. could be added here)
        )

        async with GraphRAG(
            connection=ConnectionConfig(host="localhost", graph_name="docling_demo"),
            llm=llm,
            embedder=embedder,
        ) as rag:
            print("Ingesting document with advanced DoclingLoader...")
            # Explicitly pass the loader to override auto-dispatch
            result = await rag.ingest(
                str(dummy_file),
                loader=advanced_loader,
            )
            print(f"Ingested: {result.nodes_created} nodes, {result.relationships_created} edges")

            print("\nFinalizing graph...")
            await rag.finalize()

            question = "What sections are in the advanced analysis?"
            print(f"\nQ: {question}")
            answer = await rag.completion(question)
            print(f"A: {answer.answer}")

    finally:
        # Cleanup dummy file
        if dummy_file.exists():
            dummy_file.unlink()


if __name__ == "__main__":
    asyncio.run(main())
