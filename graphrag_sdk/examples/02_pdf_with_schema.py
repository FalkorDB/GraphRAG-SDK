"""
GraphRAG SDK v2 -- PDF Ingestion with Schema
==============================================
Ingest a PDF file with a custom schema that constrains entity/relationship extraction.
Shows how to inspect retrieved context alongside the answer.

Prerequisites:
    pip install graphrag-sdk[litellm,pdf]
    docker run -p 6379:6379 falkordb/falkordb

Usage:
    python 02_pdf_with_schema.py path/to/document.pdf
"""

import asyncio
import os
import sys

from graphrag_sdk import (
    ConnectionConfig,
    EntityType,
    GraphRAG,
    GraphSchema,
    LiteLLM,
    LiteLLMEmbedder,
    RelationType,
    SchemaPattern,
)
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking


def create_schema() -> GraphSchema:
    """Define what entities and relationships the LLM should extract."""
    return GraphSchema(
        entities=[
            EntityType(label="Person", description="A human being or character"),
            EntityType(label="Organization", description="A company, institution, or group"),
            EntityType(label="Place", description="A geographic location or setting"),
            EntityType(label="Event", description="A significant occurrence or happening"),
            EntityType(label="Concept", description="An abstract idea or theme"),
        ],
        relations=[
            RelationType(label="WORKS_AT", description="Is employed by an organization"),
            RelationType(label="LOCATED_IN", description="Is physically located in a place"),
            RelationType(label="RELATED_TO", description="Has a general relationship with"),
            RelationType(label="PART_OF", description="Is a member or component of"),
            RelationType(label="PARTICIPATED_IN", description="Took part in an event"),
        ],
        patterns=[
            SchemaPattern(source="Person", relationship="WORKS_AT", target="Organization"),
            SchemaPattern(source="Person", relationship="LOCATED_IN", target="Place"),
            SchemaPattern(source="Organization", relationship="LOCATED_IN", target="Place"),
            SchemaPattern(source="Person", relationship="RELATED_TO", target="Person"),
            SchemaPattern(source="Person", relationship="PART_OF", target="Organization"),
            SchemaPattern(source="Person", relationship="PARTICIPATED_IN", target="Event"),
        ],
    )


async def main():
    if len(sys.argv) < 2:
        print("Usage: python 02_pdf_with_schema.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Providers
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

    # Create GraphRAG with schema constraints
    rag = GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="pdf_demo"),
        llm=llm,
        embedder=embedder,
        schema=create_schema(),
    )

    # Ingest PDF with larger chunks for better context
    print(f"Ingesting {pdf_path}...")
    result = await rag.ingest(
        pdf_path,
        chunker=FixedSizeChunking(chunk_size=1500, chunk_overlap=200),
    )
    print(f"Done: {result.nodes_created} nodes, {result.relationships_created} edges")

    # Query with context inspection
    question = "What are the main topics discussed in this document?"
    print(f"\nQ: {question}")

    answer = await rag.query(question, return_context=True)
    print(f"A: {answer.answer}")

    # Show what was retrieved
    print(f"\nRetrieved {len(answer.retriever_result.items)} context items:")
    for i, item in enumerate(answer.retriever_result.items[:5]):
        print(f"  [{i+1}] (score={item.score:.3f}) {item.content[:100]}...")

    # Show graph stats
    stats = await rag.graph_store.get_statistics()
    print(f"\nGraph: {stats['node_count']} nodes, {stats['edge_count']} edges")
    print(f"Entity types: {stats['entity_types']}")


if __name__ == "__main__":
    asyncio.run(main())
