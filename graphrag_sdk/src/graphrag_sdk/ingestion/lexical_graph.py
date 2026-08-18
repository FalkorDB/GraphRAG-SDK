# GraphRAG SDK — Ingestion: Lexical Graph Writer
# The provenance chain, shared by every ingest path.
#
# Extracted from IngestionPipeline so a structured pipeline can reuse it without
# subclassing: IngestionPipeline.__init__ requires a chunker and an LLM
# extractor, and a structured source has neither. Sharing a base keeps both
# paths writing byte-identical Document, Chunk and MENTIONED_IN shapes, which is
# what lets one retrieval path serve both.

from __future__ import annotations

from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    DocumentInfo,
    EntityMention,
    GraphData,
    GraphNode,
    GraphRelationship,
    TextChunks,
)


class LexicalGraphWriter:
    """Writes the Document to Chunk provenance chain, and entity mentions.

    Depends on nothing but ``self.graph_store``, so any pipeline can mix it in.

    This is NON-OPTIONAL for every ingest path. The Zero-Loss Data principle
    requires that every piece of source material is traceable in the graph, and
    ``update()`` and ``delete_document()`` are both defined over this chain: a
    record that is not a chunk is a record that leaks on update.
    """

    graph_store: Any

    async def _build_lexical_graph(
        self,
        doc_info: DocumentInfo,
        chunks: TextChunks,
        ctx: Context,
        *,
        content_hash: str | None = None,
        link_sequential: bool = True,
    ) -> None:
        """Build the mandatory provenance chain.

        Creates:
        - A Document node
        - A Chunk node for each text chunk
        - Document -[PART_OF]-> Chunk relationships
        - Chunk -[NEXT_CHUNK]-> Chunk sequential relationships

        This is NON-OPTIONAL. The Zero-Loss Data principle requires
        that every piece of source material is traceable in the graph.

        ``link_sequential`` chains NEXT_CHUNK between consecutive chunks. True
        for prose, where the chunks are consecutive passages of one text. False
        for records, which have no reading order: ``cypher_generation`` tells the
        model NEXT_CHUNK means "the next sequential Chunk", so chaining unrelated
        rows would assert a sequence that does not exist.

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
            if link_sequential and prev_chunk_id is not None:
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
