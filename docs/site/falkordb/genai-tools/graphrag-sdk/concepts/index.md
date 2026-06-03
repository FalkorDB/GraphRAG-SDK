---
title: "Concepts"
nav_order: 3
parent: "GraphRAG-SDK"
grand_parent: "GenAI Tools"
has_children: true
description: "Mental models for GraphRAG-SDK — what a knowledge graph is, how ontologies constrain it, how ingestion and retrieval pipelines work, and how incremental updates stay consistent."
---

# Concepts

The mental models behind GraphRAG-SDK. Read these before reaching for the API reference if you're new to graph-based RAG — or if you want to know *why* the pipeline is shaped the way it is.

| Page | When to read |
|---|---|
| [Knowledge graph](./knowledge-graph) | First time using a graph for RAG — what does it give you over a vector DB alone? |
| [Ontology](./ontology) | Designing or curating a schema. |
| [Ontology discovery](./ontology-discovery) | You don't know what schema you need yet. **New in v1.2.** |
| [Ingestion pipeline](./ingestion-pipeline) | Tuning chunking, extraction, or resolution for your corpus. |
| [Retrieval pipeline](./retrieval-pipeline) | Tuning retrieval — vector vs full-text vs Cypher vs graph walks. |
| [Incremental updates](./incremental-updates) | Wiring the SDK into CI / PR-merge / document-watch workflows. |
