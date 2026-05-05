# GraphRAG SDK — Ingestion: Pipeline Orchestrator
# Pattern: Sequential Pipeline — domain-specific linear orchestrator.
# Flow: Load → Chunk → Lexical Graph (MANDATORY) → Extract → Prune → Resolve → Write + Index
#
# Origin: Neo4j lexical graph + pruning as mandatory steps;
#         User design for domain-specific sequential pipeline over generic DAG.

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import IngestionError
from graphrag_sdk.core.models import (
    DocumentInfo,
    DocumentOutput,
    EntityMention,
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

# Number of offending (src, tgt) pairs to include verbatim in the
# "pattern mismatch" warning. Enough to spot the inversion; bounded
# so a misconfigured schema can't flood logs.
_PATTERN_MISMATCH_SAMPLE_SIZE = 3


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
    8. **Mentions** — write MENTIONED_IN edges (parallel with step 9)
    9. **Index Chunks** — embed and index chunks in vector store (parallel with step 8)

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
            extractor=GraphExtraction(llm=my_llm),
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
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.extractor = extractor
        self.resolver = resolver
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.schema = schema or GraphSchema()

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
                    document_info=document_info or DocumentInfo(path=source),
                )
                ctx.log("Using provided text (loader skipped)")
            else:
                ctx.log("Step 1/9: Loading source")
                document = await self.loader.load(source, ctx)
                # When the caller supplies a ``document_info`` (e.g. for
                # stable-id ingestion or update()), prefer its uid/path
                # over whatever the loader produced. The loader-side
                # metadata is preserved.
                if document_info is not None:
                    document.document_info = DocumentInfo(
                        uid=document_info.uid,
                        path=document_info.path or document.document_info.path,
                        metadata={
                            **document.document_info.metadata,
                            **document_info.metadata,
                        },
                    )

            # Step 2: Chunk
            ctx.log("Step 2/9: Chunking text")
            chunks = await self.chunker.chunk(document.text, ctx)

            if not chunks.chunks:
                ctx.log("No chunks produced, pipeline complete")
                return IngestionResult(document_info=document.document_info)

            # Step 3: Build lexical graph (MANDATORY — not a strategy).
            # Hash the loaded text so ``GraphRAG.update()`` can short-circuit
            # when content is unchanged. SHA-256 hex; cost is negligible
            # next to extraction.
            content_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
            ctx.log("Step 3/9: Building lexical graph (provenance chain)")
            await self._build_lexical_graph(
                document.document_info, chunks, ctx, content_hash=content_hash
            )

            # Step 4: Extract entities & relationships
            ctx.log("Step 4/9: Extracting entities & relationships")
            graph_data = await self.extractor.extract(chunks, self.schema, ctx)

            # Step 4b: Quality filter — remove empty-ID and invalid nodes
            graph_data = self._filter_quality(graph_data)

            # Step 5: Prune against schema
            ctx.log("Step 5/9: Pruning against schema")
            graph_data = self._prune(graph_data, self.schema)

            # Step 6: Resolve duplicate entities
            ctx.log("Step 6/9: Resolving duplicates")
            resolved = await self.resolver.resolve(graph_data, ctx)

            # Step 6b: Rewrite mentions through the resolver's remap.
            # Without this, MENTIONED_IN edges that reference an entity
            # which was merged away during resolution would silently fail
            # to write — graph_store.upsert_relationships does MATCH (a)
            # MATCH (b) MERGE, and the MATCH on the merged-away id finds
            # nothing. That would silently break update()/delete_document()
            # orphan-cleanup correctness for fuzzy-resolver users (default
            # ExactMatch typically has no merges to apply, so the rewrite
            # is a no-op for the common case).
            if resolved.remap and graph_data.mentions:
                graph_data = self._remap_mentions(graph_data, resolved.remap)

            # Step 7: Write to graph (batched)
            ctx.log("Step 7/9: Writing to graph store")
            await self.graph_store.upsert_nodes(resolved.nodes)
            await self.graph_store.upsert_relationships(resolved.relationships)

            # Steps 8-9: Run in parallel (independent of each other).
            #
            # LOAD-BEARING ORDERING: this gather() must complete before
            # run() returns. v1.1.0's update()/delete_document() orphan-
            # cleanup is race-free under concurrent updates only because
            # the new MENTIONED_IN edges produced in step 8 are persisted
            # to the graph before pipeline.run() returns (and therefore
            # before the caller's cutover begins). Concurrent updates A
            # and B sharing an entity ``e1`` will then always observe
            # ``e1`` to have at least one incident MENTIONED_IN from B's
            # old chunks (pre-cutover) or B's new chunks (post-pipeline.
            # run()), so A's orphan-cleanup never wrongly deletes it.
            #
            # If you defer step 8 to a background task, batch it across
            # pipelines, or skip it under a flag, you must also serialize
            # updates inside apply_changes — the current default
            # update_concurrency=1 is what keeps concurrent updates
            # correct otherwise.
            #
            # Tripwire: tests/test_integration.py::
            #     TestIncrementalUpdateInvariants::
            #     test_concurrent_updates_preserve_shared_entity
            async def _step_mentions() -> int:
                ctx.log("Step 8/9: Writing mentions (uncapped)")
                return await self._write_mentions(graph_data, ctx)

            async def _step_index_chunks() -> None:
                ctx.log("Step 9/9: Embedding & indexing chunks")
                await self.vector_store.index_chunks(chunks)

            mentions_written, _ = await asyncio.gather(
                _step_mentions(),
                _step_index_chunks(),
            )

            total_rels = len(resolved.relationships) + mentions_written
            result = IngestionResult(
                document_info=document.document_info,
                nodes_created=len(resolved.nodes),
                relationships_created=total_rels,
                chunks_indexed=len(chunks.chunks),
                metadata={
                    "merged_entities": resolved.merged_count,
                    "raw_nodes": len(graph_data.nodes),
                    "raw_relationships": len(graph_data.relationships),
                    "mention_edges_created": mentions_written,
                },
            )
            ctx.log(
                f"Pipeline complete: {result.nodes_created} nodes, "
                f"{result.relationships_created} rels, "
                f"{result.chunks_indexed} chunks indexed, "
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
        *,
        content_hash: str | None = None,
    ) -> None:
        """Build the mandatory provenance chain.

        Creates:
        - A Document node
        - A Chunk node for each text chunk
        - Document -[PART_OF]-> Chunk relationships
        - Chunk -[NEXT_CHUNK]-> Chunk sequential relationships

        This is NON-OPTIONAL. The Zero-Loss Data principle requires
        that every piece of source material is traceable in the graph.

        ``content_hash`` is the SHA-256 of the loaded source text. When
        present it is written to the Document node so ``GraphRAG.update()``
        can short-circuit no-op updates without re-running extraction.
        """
        # Document node
        doc_props: dict[str, Any] = {
            "path": doc_info.path or "",
            **doc_info.metadata,
        }
        if content_hash is not None:
            doc_props["content_hash"] = content_hash
        doc_node = GraphNode(
            id=doc_info.uid,
            label="Document",
            properties=doc_props,
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

        Each check runs only when the corresponding schema section is populated:

        - ``schema.entities`` present → nodes whose label is not declared (or
          "Unknown") are dropped; otherwise all nodes pass.
        - ``schema.relations`` present → relationships whose ``rel_type`` is not
          declared are dropped, and each declared relation's ``patterns`` (when
          non-empty) filters by ``(src_label, tgt_label)``; otherwise all
          relationships whose endpoints survived node-pruning pass.

        When both sections are empty the pipeline is in open-schema mode and
        the graph is returned unchanged.
        """
        if not schema.entities and not schema.relations:
            return graph_data

        # --- Node filtering (keep "Unknown" — low-confidence entities) ---
        allowed_labels = {e.label for e in schema.entities}
        if allowed_labels:
            allowed_labels.add("Unknown")
            pruned_nodes = [n for n in graph_data.nodes if n.label in allowed_labels]
        else:
            pruned_nodes = graph_data.nodes

        nodes_by_id = {n.id: n for n in pruned_nodes}

        # --- Build relation catalog ---
        # label -> set of (src, tgt) pairs, or None for "open" (no pattern constraint).
        allowed_rels: dict[str, set[tuple[str, str]] | None] = {}
        for rt in schema.relations:
            allowed_rels[rt.label] = set(rt.patterns) if rt.patterns else None

        # --- Relationship filtering ---
        pruned_rels: list[GraphRelationship] = []
        # Track (src, tgt) pairs dropped for pattern-mismatch (vs unknown rel
        # type or pruned endpoints). Direction inversion is the most common
        # cause and the hardest to debug without a structured warning.
        pattern_mismatches: dict[str, list[tuple[str, str]]] = {}
        for r in graph_data.relationships:
            rel_label = r.properties.get("rel_type", r.type)

            src = nodes_by_id.get(r.start_node_id)
            tgt = nodes_by_id.get(r.end_node_id)
            if src is None or tgt is None:
                continue

            if not allowed_rels:
                pruned_rels.append(r)
                continue

            valid_pairs = allowed_rels.get(rel_label)
            if valid_pairs is None and rel_label in allowed_rels:
                pruned_rels.append(r)
            elif valid_pairs and (src.label, tgt.label) in valid_pairs:
                pruned_rels.append(r)
            elif valid_pairs:
                # Declared rel type, but (src, tgt) doesn't match any pattern.
                pattern_mismatches.setdefault(rel_label, []).append((src.label, tgt.label))

        pruned_node_count = len(graph_data.nodes) - len(pruned_nodes)
        pruned_rel_count = len(graph_data.relationships) - len(pruned_rels)
        if pruned_node_count or pruned_rel_count:
            logger.info(f"Pruned {pruned_node_count} nodes, {pruned_rel_count} rels")

        for rel_label, observed in pattern_mismatches.items():
            sample = observed[:_PATTERN_MISMATCH_SAMPLE_SIZE]
            declared = sorted(allowed_rels[rel_label] or set())
            logger.warning(
                "Pruned %d '%s' relationships due to (source, target) mismatch. "
                "Declared patterns: %s. Observed (sample): %s. "
                "If extraction looks correct, the pattern direction may be inverted.",
                len(observed),
                rel_label,
                declared,
                sample,
            )

        return GraphData(
            nodes=pruned_nodes,
            relationships=pruned_rels,
            mentions=graph_data.mentions,
            extracted_entities=graph_data.extracted_entities,
            extracted_relations=graph_data.extracted_relations,
        )

    def _filter_quality(self, graph_data: GraphData) -> GraphData:
        """Remove nodes with empty IDs or labels, and dangling relationships."""
        valid_nodes = [n for n in graph_data.nodes if n.id and n.label]
        removed = len(graph_data.nodes) - len(valid_nodes)
        if removed:
            logger.info(f"Quality filter removed {removed} invalid nodes")
        valid_ids = {n.id for n in valid_nodes}
        valid_rels = [
            r
            for r in graph_data.relationships
            if r.start_node_id in valid_ids and r.end_node_id in valid_ids
        ]
        new_gd = GraphData(
            nodes=valid_nodes,
            relationships=valid_rels,
            mentions=graph_data.mentions,
            extracted_entities=graph_data.extracted_entities,
            extracted_relations=graph_data.extracted_relations,
        )
        return new_gd

    @staticmethod
    def _remap_mentions(graph_data: GraphData, remap: dict[str, str]) -> GraphData:
        """Rewrite ``graph_data.mentions`` so MENTIONED_IN edges target
        the survivor entity, not a merged-away id.

        Called between the resolver step and the write step whenever the
        resolver reports a non-empty ``remap``. Without this, mentions
        carrying pre-resolution ids silently fail to be written (the
        upsert's MATCH on the merged-away id finds nothing), which would
        invalidate the orphan-cleanup invariant for fuzzy resolvers.
        """
        rewritten: list[EntityMention] = []
        seen: set[tuple[str, str]] = set()
        for m in graph_data.mentions:
            new_id = remap.get(m.entity_id, m.entity_id)
            key = (new_id, m.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            rewritten.append(EntityMention(entity_id=new_id, chunk_id=m.chunk_id))
        return graph_data.model_copy(update={"mentions": rewritten})

    async def _write_mentions(self, graph_data: GraphData, ctx: Context) -> int:
        """Write MENTIONED_IN edges linking entities to their source chunks.

        Every entity connects to every chunk it was extracted from (uncapped).
        With global dedup controlling entity cardinality, uncapped mentions
        provide richer entity-chunk connectivity for retrieval.
        """
        mentions: list[EntityMention] = graph_data.mentions or []

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
        ctx.log(f"Wrote {len(mention_rels)} MENTIONED_IN edges (uncapped)")
        return len(mention_rels)
