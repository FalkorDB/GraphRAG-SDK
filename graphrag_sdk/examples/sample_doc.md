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
