# GraphRAG SDK 2.0 — Ingestion: Pipeline Orchestrator
# Pattern: Sequential Pipeline — domain-specific linear orchestrator.
# Flow: Load → Chunk → Lexical Graph (MANDATORY) → Extract → Prune → Resolve → Write + Index
#
# Origin: Neo4j lexical graph + pruning as mandatory steps;
#         User design for domain-specific sequential pipeline over generic DAG.

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import numpy as np

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import IngestionError, SchemaValidationError
from graphrag_sdk.core.models import (
    DocumentInfo,
    DocumentOutput,
    EntityMention,
    FactTriple,
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
        skip_synonymy: bool = False,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.extractor = extractor
        self.resolver = resolver
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.schema = schema or GraphSchema()
        self.embedder = embedder
        self._skip_synonymy = skip_synonymy

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
                ctx.log("Step 1/11: Loading source")
                document = await self.loader.load(source, ctx)

            # Step 2: Chunk
            ctx.log("Step 2/11: Chunking text")
            chunks = await self.chunker.chunk(document.text, ctx)

            if not chunks.chunks:
                ctx.log("No chunks produced, pipeline complete")
                return IngestionResult(document_info=document.document_info)

            # Step 3: Build lexical graph (MANDATORY — not a strategy)
            ctx.log("Step 3/11: Building lexical graph (provenance chain)")
            await self._build_lexical_graph(document.document_info, chunks, ctx)

            # Step 4: Extract entities & relationships
            ctx.log("Step 4/11: Extracting entities & relationships")
            graph_data = await self.extractor.extract(chunks, self.schema, ctx)

            # Step 4b: Quality filter — remove empty-ID and invalid nodes
            graph_data = self._filter_quality(graph_data)

            # Step 5: Prune against schema
            ctx.log("Step 5/11: Pruning against schema")
            graph_data = self._prune(graph_data, self.schema)

            # Step 6: Resolve duplicate entities
            ctx.log("Step 6/11: Resolving duplicates")
            resolved = await self.resolver.resolve(graph_data, ctx)

            # Step 7: Write to graph (batched)
            ctx.log("Step 7/11: Writing to graph store")
            await self.graph_store.upsert_nodes(resolved.nodes)
            await self.graph_store.upsert_relationships(resolved.relationships)

            # Steps 8-11: Run in parallel (independent of each other)
            async def _step_index_facts() -> int:
                ctx.log("Step 8/11: Indexing facts")
                return await self._index_facts(graph_data, ctx)

            async def _step_synonymy() -> list[GraphRelationship]:
                if self._skip_synonymy:
                    ctx.log("Step 9/11: Synonymy skipped (post-ingestion mode)")
                    return []
                ctx.log("Step 9/11: Detecting synonymy")
                edges = await self._detect_synonymy_edges(resolved.nodes, ctx)
                if edges:
                    await self.graph_store.upsert_relationships(edges)
                return edges

            async def _step_mentions() -> int:
                ctx.log("Step 10/11: Writing mentions")
                return await self._write_mentions(graph_data, ctx)

            async def _step_index_chunks() -> None:
                ctx.log("Step 11/11: Embedding & indexing chunks")
                await self.vector_store.index_chunks(chunks)

            facts_indexed, synonym_edges, mentions_written, _ = await asyncio.gather(
                _step_index_facts(),
                _step_synonymy(),
                _step_mentions(),
                _step_index_chunks(),
            )

            total_rels = len(resolved.relationships) + len(synonym_edges) + mentions_written
            result = IngestionResult(
                document_info=document.document_info,
                nodes_created=len(resolved.nodes),
                relationships_created=total_rels,
                chunks_indexed=len(chunks.chunks),
                metadata={
                    "merged_entities": resolved.merged_count,
                    "raw_nodes": len(graph_data.nodes),
                    "raw_relationships": len(graph_data.relationships),
                    "facts_indexed": facts_indexed,
                    "synonym_edges_created": len(synonym_edges),
                    "mention_edges_created": mentions_written,
                },
            )
            ctx.log(
                f"Pipeline complete: {result.nodes_created} nodes, "
                f"{result.relationships_created} rels, "
                f"{result.chunks_indexed} chunks indexed, "
                f"{facts_indexed} facts, {len(synonym_edges)} synonyms, "
                f"{mentions_written} mentions"
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

        new_gd = GraphData(nodes=pruned_nodes, relationships=pruned_rels)
        # Propagate extras (facts, mentions) through pruning
        if hasattr(graph_data, "facts"):
            new_gd.facts = graph_data.facts
        if hasattr(graph_data, "mentions"):
            new_gd.mentions = graph_data.mentions
        return new_gd

    def _filter_quality(self, graph_data: GraphData) -> GraphData:
        """Remove nodes with empty IDs or labels, and dangling relationships."""
        valid_nodes = [n for n in graph_data.nodes if n.id and n.label]
        removed = len(graph_data.nodes) - len(valid_nodes)
        if removed:
            logger.info(f"Quality filter removed {removed} invalid nodes")
        valid_ids = {n.id for n in valid_nodes}
        valid_rels = [
            r for r in graph_data.relationships
            if r.start_node_id in valid_ids and r.end_node_id in valid_ids
        ]
        new_gd = GraphData(nodes=valid_nodes, relationships=valid_rels)
        # Propagate extras (facts, mentions) through quality filter
        if hasattr(graph_data, "facts"):
            new_gd.facts = graph_data.facts
        if hasattr(graph_data, "mentions"):
            new_gd.mentions = graph_data.mentions
        return new_gd

    async def _index_facts(self, graph_data: GraphData, ctx: Context) -> int:
        """Index facts as Fact nodes with embeddings for vector search.

        Uses facts attached to graph_data via extra="allow" if available,
        otherwise generates facts from relationships.
        """
        facts: list[FactTriple] = []
        if hasattr(graph_data, "facts") and graph_data.facts:
            facts = graph_data.facts
        elif graph_data.relationships:
            for rel in graph_data.relationships:
                if rel.type == "SYNONYM":
                    continue
                facts.append(FactTriple(
                    subject=rel.start_node_id,
                    predicate=rel.type,
                    object=rel.end_node_id,
                    chunk_id=rel.properties.get("source_chunk_id", ""),
                ))

        if not facts:
            return 0

        fact_strings = [f"({f.subject}, {f.predicate}, {f.object})" for f in facts]
        return await self.vector_store.index_facts(fact_strings, facts)

    async def _detect_synonymy_edges(
        self,
        nodes: list[GraphNode],
        ctx: Context,
        *,
        similarity_threshold: float = 0.9,
    ) -> list[GraphRelationship]:
        """Detect semantically similar entities and create SYNONYM edges.

        Uses numpy vectorized cosine similarity for performance.
        """
        if self.embedder is None:
            return []

        structural_labels = {"Document", "Chunk"}
        entity_nodes = [n for n in nodes if n.label not in structural_labels]

        if len(entity_nodes) < 2:
            return []

        names = [n.properties.get("name", n.id) for n in entity_nodes]
        raw_vectors = await self.embedder.aembed_documents(names)

        # Filter out entities whose embedding failed (None)
        valid = [(node, vec) for node, vec in zip(entity_nodes, raw_vectors) if vec is not None]
        if len(valid) < 2:
            return []
        entity_nodes, vectors = zip(*valid)  # type: ignore[assignment]
        entity_nodes = list(entity_nodes)

        # Numpy vectorized pairwise cosine similarity (block-wise to limit memory)
        mat = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat_normed = mat / norms

        # Block-wise computation avoids N×N allocation (7.4GB for 44K entities)
        BLOCK_SIZE = 1000
        n = len(entity_nodes)
        rows_list: list[int] = []
        cols_list: list[int] = []
        for i_start in range(0, n, BLOCK_SIZE):
            i_end = min(i_start + BLOCK_SIZE, n)
            block = mat_normed[i_start:i_end]
            remaining = mat_normed[i_start:]
            sim_block = block @ remaining.T  # shape: (block_size, n - i_start)
            local_rows, local_cols = np.where(sim_block >= similarity_threshold)
            for lr, lc in zip(local_rows.tolist(), local_cols.tolist()):
                global_i = i_start + lr
                global_j = i_start + lc
                if global_j > global_i:  # upper triangle only
                    rows_list.append(global_i)
                    cols_list.append(global_j)
        rows = np.array(rows_list, dtype=np.intp)
        cols = np.array(cols_list, dtype=np.intp)

        synonym_edges: list[GraphRelationship] = []
        for i, j in zip(rows, cols):
            sim_val = float(mat_normed[i] @ mat_normed[j])
            synonym_edges.append(GraphRelationship(
                start_node_id=entity_nodes[i].id,
                end_node_id=entity_nodes[j].id,
                type="SYNONYM",
                properties={"similarity": sim_val},
            ))

        ctx.log(f"Synonymy: {len(synonym_edges)} SYNONYM edges from {len(entity_nodes)} entities")
        return synonym_edges

    async def _write_mentions(self, graph_data: GraphData, ctx: Context) -> int:
        """Write MENTIONED_IN edges linking entities to their source chunks."""
        mentions: list[EntityMention] = []
        if hasattr(graph_data, "mentions") and graph_data.mentions:
            mentions = graph_data.mentions

        if not mentions:
            return 0

        seen: set[tuple[str, str]] = set()
        mention_rels: list[GraphRelationship] = []
        for m in mentions:
            key = (m.entity_id, m.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            mention_rels.append(
                GraphRelationship(
                    start_node_id=m.entity_id,
                    end_node_id=m.chunk_id,
                    type="MENTIONED_IN",
                )
            )
        await self.graph_store.upsert_relationships(mention_rels)
        ctx.log(f"Wrote {len(mention_rels)} MENTIONED_IN edges")
        return len(mention_rels)
