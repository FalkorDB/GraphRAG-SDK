"""
GraphRAG SDK v2 -- Custom Strategies (Benchmark-Winning Pipeline)
==================================================================
Demonstrates the full pipeline configuration that achieved 88.2% accuracy
on the 20-document novel benchmark. Uses:
  - MergedExtraction (LightRAG + HippoRAG combined)
  - DescriptionMergeResolution (LLM-assisted entity dedup)
  - Post-ingestion synonym detection
  - MultiPathRetrieval (default, configured automatically)

Prerequisites:
    pip install graphrag-sdk[litellm]
    docker run -p 6379:6379 falkordb/falkordb

Usage:
    python 03_custom_strategies.py
"""

import asyncio
import os
import time

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
from graphrag_sdk.core.context import Context
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
from graphrag_sdk.ingestion.extraction_strategies.merged_extraction import MergedExtraction
from graphrag_sdk.ingestion.resolution_strategies.description_merge import DescriptionMergeResolution

# Sample documents (replace with your own)
DOCUMENTS = [
    (
        "doc_1",
        "Marie Curie was a physicist and chemist who conducted pioneering research on "
        "radioactivity. Born in Warsaw, Poland, she moved to Paris to study at the Sorbonne. "
        "She was the first woman to win a Nobel Prize, and the only person to win Nobel Prizes "
        "in two different sciences -- Physics in 1903 and Chemistry in 1911. She worked closely "
        "with her husband Pierre Curie at the University of Paris."
    ),
    (
        "doc_2",
        "Pierre Curie was a French physicist and Nobel laureate. He shared the 1903 Nobel Prize "
        "in Physics with his wife Marie Curie and Henri Becquerel for their research on radiation. "
        "Pierre was a professor at the University of Paris. He tragically died in 1906 in a "
        "street accident in Paris. After his death, Marie took over his teaching position."
    ),
]

SCHEMA = GraphSchema(
    entities=[
        EntityType(label="Person", description="A historical figure or scientist"),
        EntityType(label="Place", description="A city, country, or geographic location"),
        EntityType(label="Organization", description="A university, institution, or award body"),
        EntityType(label="Event", description="A significant event, award, or discovery"),
        EntityType(label="Concept", description="A scientific field or abstract idea"),
    ],
    relations=[
        RelationType(label="LOCATED_IN", description="Is located in a place"),
        RelationType(label="WORKS_AT", description="Works at an institution"),
        RelationType(label="MARRIED_TO", description="Is married to"),
        RelationType(label="RELATED_TO", description="Has a relationship with"),
        RelationType(label="AWARDED", description="Received an award or prize"),
        RelationType(label="RESEARCHED", description="Conducted research on a topic"),
    ],
    patterns=[
        SchemaPattern(source="Person", relationship="LOCATED_IN", target="Place"),
        SchemaPattern(source="Person", relationship="WORKS_AT", target="Organization"),
        SchemaPattern(source="Person", relationship="MARRIED_TO", target="Person"),
        SchemaPattern(source="Person", relationship="RELATED_TO", target="Person"),
        SchemaPattern(source="Person", relationship="AWARDED", target="Event"),
        SchemaPattern(source="Person", relationship="RESEARCHED", target="Concept"),
    ],
)


async def main():
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

    rag = GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="strategies_demo"),
        llm=llm,
        embedder=embedder,
        schema=SCHEMA,
    )

    # Clear previous data
    try:
        await rag.graph_store.delete_all()
    except Exception:
        pass

    # --- Ingestion with custom strategies ---
    print("Ingesting documents with benchmark-winning pipeline...")
    t0 = time.time()

    for source_id, text in DOCUMENTS:
        result = await rag.ingest(
            source_id,
            text=text,
            # Larger chunks capture more context per extraction call
            chunker=FixedSizeChunking(chunk_size=1500, chunk_overlap=200),
            # MergedExtraction: LightRAG typed entities + HippoRAG fact triples & mentions
            extractor=MergedExtraction(llm=llm, embedder=embedder),
            # LLM-assisted deduplication merges entity descriptions
            resolver=DescriptionMergeResolution(llm=llm),
            ctx=Context(tenant_id="demo"),
        )
        print(f"  {source_id}: {result.nodes_created} nodes, {result.relationships_created} edges")

    # --- Post-ingestion: synonym detection ---
    print("\nDetecting synonyms across all entities...")
    synonym_count = await rag.detect_synonymy(similarity_threshold=0.9)
    print(f"  Created {synonym_count} SYNONYM edges")

    # --- Post-ingestion: backfill entity embeddings ---
    print("Backfilling entity embeddings...")
    backfilled = await rag.vector_store.backfill_entity_embeddings()
    print(f"  Backfilled {backfilled} entities")

    elapsed = time.time() - t0
    print(f"\nTotal ingestion time: {elapsed:.1f}s")

    # --- Graph statistics ---
    stats = await rag.graph_store.get_statistics()
    print(f"\nGraph Statistics:")
    print(f"  Nodes:         {stats['node_count']}")
    print(f"  Edges:         {stats['edge_count']}")
    print(f"  Facts:         {stats['fact_node_count']}")
    print(f"  Synonyms:      {stats['synonym_edge_count']}")
    print(f"  Mentions:      {stats['mention_edge_count']}")
    print(f"  Entity types:  {stats['entity_types']}")

    # --- Queries ---
    print("\n" + "=" * 50)
    questions = [
        "What Nobel Prizes did Marie Curie win?",
        "What is the relationship between Marie and Pierre Curie?",
        "Where did Marie Curie study?",
        "What happened to Pierre Curie?",
    ]

    for q in questions:
        result = await rag.query(q)
        print(f"\nQ: {q}")
        print(f"A: {result.answer}")


if __name__ == "__main__":
    asyncio.run(main())
