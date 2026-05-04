"""
GraphRAG SDK -- Document-Aware Markdown Ingestion
===================================================
Ingest a Markdown file using structure-preserving chunking.

Unlike fixed-size or sentence chunking, StructuralChunking groups
content by heading hierarchy.  Each chunk carries ``breadcrumbs``
(e.g. ["Installation", "Prerequisites"]) that become queryable
properties on the Chunk nodes in the knowledge graph.

MarkdownLoader strips raw markup from header content before the text
reaches the LLM, so embeddings are clean and co-reference resolution
works correctly.

Prerequisites:
    pip install graphrag-sdk[litellm,markdown]
    docker run -p 6379:6379 falkordb/falkordb

Usage:
    # Generate a ready-to-use sample document and print its content:
    python 06_markdown_document_aware.py --sample

    # Ingest any markdown file:
    python 06_markdown_document_aware.py path/to/document.md

    # Quick end-to-end demo with the generated sample:
    python 06_markdown_document_aware.py --sample          # creates sample_doc.md
    python 06_markdown_document_aware.py sample_doc.md     # ingests it
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
)
from graphrag_sdk.ingestion.chunking_strategies.structural_chunking import StructuralChunking
from graphrag_sdk.ingestion.loaders.markdown_loader import MarkdownLoader


def create_schema() -> GraphSchema:
    """Generic schema suitable for technical documentation."""
    return GraphSchema(
        entities=[
            EntityType(label="Concept",      description="A technical concept, feature, or abstraction"),
            EntityType(label="Component",    description="A software module, library, or service"),
            EntityType(label="Technology",   description="A programming language, framework, or tool"),
            EntityType(label="Person",       description="An author, contributor, or mentioned individual"),
            EntityType(label="Organization", description="A company, team, or standards body"),
        ],
        relations=[
            RelationType(
                label="DEPENDS_ON",
                description="Requires another component or technology",
                patterns=[("Component", "Component"), ("Component", "Technology")],
            ),
            RelationType(
                label="IMPLEMENTS",
                description="Provides a concrete implementation of a concept",
                patterns=[("Component", "Concept")],
            ),
            RelationType(
                label="CREATED_BY",
                description="Was authored or maintained by a person or organization",
                patterns=[("Component", "Person"), ("Component", "Organization")],
            ),
            RelationType(
                label="RELATED_TO",
                description="Has a general relationship with",
                patterns=[
                    ("Concept", "Concept"),
                    ("Component", "Concept"),
                    ("Technology", "Concept"),
                ],
            ),
        ],
    )


_SAMPLE_DOC = """
# GraphRAG SDK

GraphRAG SDK is an open-source Python library created by FalkorDB for building
knowledge-graph-augmented retrieval systems.

## Core Concepts

### Knowledge Graph

A knowledge graph stores entities and the relationships between them.
GraphRAG SDK uses FalkorDB, a Redis-based graph database that supports
the openCypher query language, as its graph backend.

### Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) grounds language model responses in
external knowledge by injecting retrieved context into the prompt before
generation. GraphRAG SDK extends this pattern with a graph-structured index.

## Components

### IngestionPipeline

The IngestionPipeline orchestrates loading, chunking, entity extraction,
resolution, graph writing, and vector indexing in a fixed nine-step sequence.
It depends on a LoaderStrategy, a ChunkingStrategy, an ExtractionStrategy,
a ResolutionStrategy, a GraphStore, and a VectorStore.

### MarkdownLoader

MarkdownLoader parses Markdown files using markdown-it-py and produces
structured DocumentElement objects that carry heading breadcrumbs.
It was created by the FalkorDB engineering team and implements the
LoaderStrategy interface.

### StructuralChunking

StructuralChunking groups DocumentElement objects by heading hierarchy
into token-bounded chunks. Each chunk stores a breadcrumbs metadata field
that is written as a property on the Chunk node in the knowledge graph,
making section paths directly queryable via Cypher.

### GraphExtraction

GraphExtraction runs a two-step pipeline: local named-entity recognition
powered by GLiNER, followed by LLM-assisted relationship extraction.
It implements the ExtractionStrategy interface and depends on an LLM provider.
"""


def _write_sample_doc(path: str = "sample_doc.md") -> str:
    """Write a compact sample markdown document to *path* and return the path."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(_SAMPLE_DOC.lstrip("\n"))
    return path


async def main():
    if len(sys.argv) < 2 or sys.argv[1] == "--sample":
        out = _write_sample_doc()
        print(f"Sample document written to: {out}")
        print()
        print("Content preview:")
        print("-" * 60)
        print(_SAMPLE_DOC.strip())
        print("-" * 60)
        print()
        print("Run the ingestion with:")
        print(f"  python {sys.argv[0]} {out}")
        sys.exit(0)

    md_path = sys.argv[1]

    # Providers
    llm = LiteLLM(
        model=f"gemini/{os.getenv('GEMINI_MODEL', 'gemini-3.1-pro-preview')}",
        api_key=f"{os.getenv("GEMINI_API_KEY")}",
    )
    embedder = LiteLLMEmbedder(
        model=f"gemini/{os.getenv('GEMINI_MODEL', 'gemini-embedding-002')}",
        api_key=f"{os.getenv("GEMINI_API_KEY")}",
    )

    rag = GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="markdown_demo"),
        llm=llm,
        embedder=embedder,
        schema=create_schema(),
    )

    # delete_all() must come BEFORE ingest().
    # ingest() calls _validate_graph_config() as its first step, which reads the
    # __GraphRAGConfig__ node from FalkorDB and raises ConfigError if the stored
    # embedding model doesn't match the current one.  Wiping the graph here
    # removes that stale node so the new run starts clean.
    try:
        await rag.delete_all()
    except Exception:
        pass  # Graph doesn't exist yet on first run — that's fine

    # Ingest the markdown file.
    # - MarkdownLoader parses heading hierarchy and produces structured elements.
    # - StructuralChunking groups those elements into token-bounded chunks and
    #   attaches breadcrumbs to each chunk's metadata so the graph is queryable
    #   by section path (e.g. MATCH (c:Chunk) WHERE 'Installation' IN c.breadcrumbs).
    print(f"Ingesting {md_path}...")
    result = await rag.ingest(
        md_path,
        loader=MarkdownLoader(),
        chunker=StructuralChunking(max_tokens=512),
    )
    print(f"Done: {result.nodes_created} nodes, {result.relationships_created} edges, "
          f"{result.chunks_indexed} chunks indexed")

    # Post-ingestion: dedup entities, embed entity/relationship descriptions,
    # and build full-text + vector indexes.  Required before retrieval works.
    await rag.finalize()

    # Query the ingested document
    question = "What are the main components or concepts described in this document?"
    print(f"\nQ: {question}")

    answer = await rag.completion(question, return_context=True)
    print(f"A: {answer.answer}")

    # Show retrieved context items
    print(f"\nRetrieved {len(answer.retriever_result.items)} context items:")
    for i, item in enumerate(answer.retriever_result.items[:5]):
        score = item.score if item.score is not None else 0.0
        print(f"  [{i+1}] (score={score:.3f}) {item.content[:120]}...")

    # Graph statistics
    stats = await rag.get_statistics()
    print(f"\nGraph: {stats['node_count']} nodes, {stats['edge_count']} edges")
    print(f"Entity types: {stats['entity_types']}")


if __name__ == "__main__":
    asyncio.run(main())
