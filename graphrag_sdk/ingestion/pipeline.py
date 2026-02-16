# GraphRAG SDK 2.0 — Ingestion: Pipeline Orchestrator
# Pattern: Sequential Pipeline — domain-specific linear orchestrator.
# Flow: Load → Chunk → Lexical Graph (MANDATORY) → Extract → Prune → Resolve → Write + Index
#
# Origin: Neo4j lexical graph + pruning as mandatory steps;
#         User design for domain-specific sequential pipeline over generic DAG.

from __future__ import annotations

import logging
from typing import Any, Optional

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import IngestionError, SchemaValidationError
from graphrag_sdk.core.models import (
    DocumentInfo,
    DocumentOutput,
    GraphData,
    GraphNode,
    GraphRelationship,
    GraphSchema,
    IngestionResult,
    TextChunks,
)
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy
from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Sequential orchestrator for knowledge graph construction.

    Executes the fixed sequence:

    1. **Load** — read text from source via ``LoaderStrategy``
    2. **Chunk** — split text via ``ChunkingStrategy``
    3. **Lexical Graph** — create Document→Chunk provenance (MANDATORY)
    4. **Extract** — extract entities/relationships via ``ExtractionStrategy``
    5. **Prune** — filter against schema (built-in, not a strategy)
    6. **Resolve** — deduplicate entities via ``ResolutionStrategy``
    7. **Write** — upsert to graph store (batched)
    8. **Index** — embed and index chunks in vector store

    The pipeline is intentionally *not* a generic DAG — the fixed sequence
    is debuggable, loggable, and understandable.

    Args:
        loader: Data source loader strategy.
        chunker: Text chunking strategy.
        extractor: Entity/relationship extraction strategy.
        resolver: Entity resolution strategy.
        graph_store: Graph data access object (from ``storage/``).
        vector_store: Vector data access object (from ``storage/``).
        schema: Graph schema for extraction constraints and pruning.

    Example::

        pipeline = IngestionPipeline(
            loader=PdfLoader(),
            chunker=FixedSizeChunking(chunk_size=500),
            extractor=SchemaGuidedExtraction(llm=my_llm),
            resolver=ExactMatchResolution(),
            graph_store=my_graph_store,
            vector_store=my_vector_store,
            schema=my_schema,
        )
        result = await pipeline.run("document.pdf", ctx)
    """

    def __init__(
        self,
        loader: LoaderStrategy,
        chunker: ChunkingStrategy,
        extractor: ExtractionStrategy,
        resolver: ResolutionStrategy,
        graph_store: Any,  # storage.GraphStore — import avoided for layering
        vector_store: Any,  # storage.VectorStore
        schema: GraphSchema | None = None,
        embedder: Any | None = None,  # optional, for future use
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.extractor = extractor
        self.resolver = resolver
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.schema = schema or GraphSchema()
        self.embedder = embedder

    async def run(
        self,
        source: str,
        ctx: Context | None = None,
        *,
        text: str | None = None,
        document_info: DocumentInfo | None = None,
    ) -> IngestionResult:
        """Execute the full ingestion pipeline.

        Either ``source`` or ``text`` must be provided:
        - If ``source`` is given, the loader reads from it.
        - If ``text`` is given directly, the loader step is skipped.

        Args:
            source: Path/URL to load (passed to loader).
            ctx: Execution context (created automatically if None).
            text: Optional raw text (skips loader if provided).
            document_info: Optional pre-built document metadata.

        Returns:
            IngestionResult with statistics about the pipeline run.
        """
        if ctx is None:
            ctx = Context()

        ctx.log("Pipeline starting")

        try:
            # Step 1: Load
            if text is not None:
                document = DocumentOutput(
                    text=text,
                    document_info=document_info or DocumentInfo(),
                )
                ctx.log("Using provided text (loader skipped)")
            else:
                ctx.log("Step 1/8: Loading source")
                document = await self.loader.load(source, ctx)

            # Step 2: Chunk
            ctx.log("Step 2/8: Chunking text")
            chunks = await self.chunker.chunk(document.text, ctx)

            if not chunks.chunks:
                ctx.log("No chunks produced, pipeline complete")
                return IngestionResult(document_info=document.document_info)

            # Step 3: Build lexical graph (MANDATORY — not a strategy)
            ctx.log("Step 3/8: Building lexical graph (provenance chain)")
            await self._build_lexical_graph(document.document_info, chunks, ctx)

            # Step 4: Extract entities & relationships
            ctx.log("Step 4/8: Extracting entities & relationships")
            graph_data = await self.extractor.extract(chunks, self.schema, ctx)

            # Step 4b: Quality filter — remove empty-ID and invalid nodes
            graph_data = self._filter_quality(graph_data)

            # Step 5: Prune against schema
            ctx.log("Step 5/8: Pruning against schema")
            graph_data = self._prune(graph_data, self.schema)

            # Step 6: Resolve duplicate entities
            ctx.log("Step 6/8: Resolving duplicates")
            resolved = await self.resolver.resolve(graph_data, ctx)

            # Step 7: Write to graph (batched)
            ctx.log("Step 7/8: Writing to graph store")
            await self.graph_store.upsert_nodes(resolved.nodes)
            await self.graph_store.upsert_relationships(resolved.relationships)

            # Step 8: Embed & index chunks
            ctx.log("Step 8/8: Embedding & indexing chunks")
            await self.vector_store.index_chunks(chunks)

            result = IngestionResult(
                document_info=document.document_info,
                nodes_created=len(resolved.nodes),
                relationships_created=len(resolved.relationships),
                chunks_indexed=len(chunks.chunks),
                metadata={
                    "merged_entities": resolved.merged_count,
                    "raw_nodes": len(graph_data.nodes),
                    "raw_relationships": len(graph_data.relationships),
                },
            )
            ctx.log(
                f"Pipeline complete: {result.nodes_created} nodes, "
                f"{result.relationships_created} rels, "
                f"{result.chunks_indexed} chunks indexed"
            )
            return result

        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Pipeline failed: {exc}") from exc

    async def _build_lexical_graph(
        self,
        doc_info: DocumentInfo,
        chunks: TextChunks,
        ctx: Context,
    ) -> None:
        """Build the mandatory provenance chain.

        Creates:
        - A Document node
        - A Chunk node for each text chunk
        - Document -[PART_OF]-> Chunk relationships
        - Chunk -[NEXT_CHUNK]-> Chunk sequential relationships

        This is NON-OPTIONAL. The Zero-Loss Data principle requires
        that every piece of source material is traceable in the graph.
        """
        # Document node
        doc_node = GraphNode(
            id=doc_info.uid,
            label="Document",
            properties={
                "path": doc_info.path or "",
                **doc_info.metadata,
            },
        )
        await self.graph_store.upsert_nodes([doc_node])

        # Chunk nodes + PART_OF relationships
        chunk_nodes: list[GraphNode] = []
        part_of_rels: list[GraphRelationship] = []
        next_chunk_rels: list[GraphRelationship] = []

        prev_chunk_id: str | None = None

        for chunk in chunks.chunks:
            chunk_node = GraphNode(
                id=chunk.uid,
                label="Chunk",
                properties={
                    "text": chunk.text,
                    "index": chunk.index,
                    **chunk.metadata,
                },
            )
            chunk_nodes.append(chunk_node)

            # Document -[PART_OF]-> Chunk
            part_of_rels.append(
                GraphRelationship(
                    start_node_id=doc_info.uid,
                    end_node_id=chunk.uid,
                    type="PART_OF",
                    properties={"index": chunk.index},
                )
            )

            # Previous Chunk -[NEXT_CHUNK]-> Current Chunk
            if prev_chunk_id is not None:
                next_chunk_rels.append(
                    GraphRelationship(
                        start_node_id=prev_chunk_id,
                        end_node_id=chunk.uid,
                        type="NEXT_CHUNK",
                    )
                )
            prev_chunk_id = chunk.uid

        await self.graph_store.upsert_nodes(chunk_nodes)
        await self.graph_store.upsert_relationships(part_of_rels + next_chunk_rels)

        ctx.log(
            f"Lexical graph: 1 Document, {len(chunk_nodes)} Chunks, "
            f"{len(part_of_rels)} PART_OF, {len(next_chunk_rels)} NEXT_CHUNK"
        )

    def _filter_quality(self, graph_data: GraphData) -> GraphData:
        """Remove nodes with empty IDs or labels, and dangling relationships.

        This is a defensive quality gate that catches anything the extraction
        strategy missed.
        """
        valid_nodes = [n for n in graph_data.nodes if n.id and n.label]
        removed = len(graph_data.nodes) - len(valid_nodes)

        if removed:
            logger.info(f"Quality filter removed {removed} invalid nodes")

        valid_ids = {n.id for n in valid_nodes}
        valid_rels = [
            r for r in graph_data.relationships
            if r.start_node_id in valid_ids and r.end_node_id in valid_ids
        ]

        removed_rels = len(graph_data.relationships) - len(valid_rels)
        if removed_rels:
            logger.info(f"Quality filter removed {removed_rels} dangling relationships")

        return GraphData(nodes=valid_nodes, relationships=valid_rels)

    def _prune(self, graph_data: GraphData, schema: GraphSchema) -> GraphData:
        """Filter graph data to only include schema-conforming nodes and relationships.

        If the schema has no entity types defined, all data passes through
        (open-schema mode).
        """
        if not schema.entities and not schema.relations:
            return graph_data  # Open schema — no pruning

        # Filter nodes by allowed labels
        allowed_labels = {e.label for e in schema.entities}
        if allowed_labels:
            pruned_nodes = [n for n in graph_data.nodes if n.label in allowed_labels]
        else:
            pruned_nodes = graph_data.nodes

        # Filter relationships by allowed types
        allowed_types = {r.label for r in schema.relations}
        if allowed_types:
            pruned_rels = [
                r for r in graph_data.relationships if r.type in allowed_types
            ]
        else:
            pruned_rels = graph_data.relationships

        # Ensure relationship endpoints exist
        valid_ids = {n.id for n in pruned_nodes}
        pruned_rels = [
            r
            for r in pruned_rels
            if r.start_node_id in valid_ids and r.end_node_id in valid_ids
        ]

        pruned_node_count = len(graph_data.nodes) - len(pruned_nodes)
        pruned_rel_count = len(graph_data.relationships) - len(pruned_rels)
        if pruned_node_count or pruned_rel_count:
            logger.info(f"Pruned {pruned_node_count} nodes, {pruned_rel_count} rels")

        return GraphData(nodes=pruned_nodes, relationships=pruned_rels)
