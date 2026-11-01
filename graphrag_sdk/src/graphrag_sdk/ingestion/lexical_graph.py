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


def merge_mentions(mentions: list[EntityMention]) -> list[EntityMention]:
    """One mention per (entity, chunk). Two that land on one — two names with
    one entity id, or a remap onto a survivor — become one holding both
    mentions' spans and descriptions, each once, in order: its MENTIONED_IN
    edge is the only record of them."""
    spans: dict[tuple[str, str], list[tuple[int, int]]] = {}
    said: dict[tuple[str, str], list[str]] = {}
    for m in mentions:
        key = (m.entity_id, m.chunk_id)
        pairs = spans.setdefault(key, [])
        for pair in zip(m.spans[0::2], m.spans[1::2]):
            if pair not in pairs:
                pairs.append(pair)
        held = said.setdefault(key, [])
        for d in m.descriptions:
            if d not in held:
                held.append(d)
    return [
        EntityMention(
            entity_id=entity_id,
            chunk_id=chunk_id,
            spans=[x for pair in pairs for x in pair],
            descriptions=said[(entity_id, chunk_id)],
        )
        for (entity_id, chunk_id), pairs in spans.items()
    ]


def _reported_short(reported: Any, expected: int) -> bool:
    """True when a store reported writing fewer items than it was handed.

    ``GraphStore.upsert_relationships`` and ``VectorStore.index_chunks`` do
    not raise on per-item failures — they log, skip and return a count — so
    the count is the only signal that a relationship or an embedding is
    missing. Stores that return nothing (``None``, a mock) are taken at their
    word: ``index_chunks`` returns ``None`` when no embedder is configured
    (nothing was attempted, so nothing is missing), and the pipeline cannot
    tell and must not refuse to ever mark a run complete against a
    duck-typed store.
    """
    return isinstance(reported, int) and not isinstance(reported, bool) and reported < expected


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
        link_sequential: bool = True,
    ) -> str | None:
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

        The Document is written without its ``content_hash``. Both pipelines
        record the hash last, through :meth:`_mark_content_hash`, once every
        write has been reported complete: written here, it certified a
        Document whose entities or edges then failed to write, and the next
        ingest of the same content skipped it as unchanged.

        Returns ``None`` when every edge was reported written, else a short
        description of the shortfall for the caller to record.
        """
        # Document node
        doc_node = GraphNode(
            id=doc_info.uid,
            label="Document",
            properties={"path": doc_info.path or "", **doc_info.metadata},
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
        lexical_rels = part_of_rels + next_chunk_rels
        written = await self.graph_store.upsert_relationships(lexical_rels)

        ctx.log(
            f"Lexical graph: 1 Document, {len(chunk_nodes)} Chunks, "
            f"{len(part_of_rels)} PART_OF, {len(next_chunk_rels)} NEXT_CHUNK"
        )
        if _reported_short(written, len(lexical_rels)):
            return f"lexical edges {written}/{len(lexical_rels)}"
        return None

    async def _mark_content_hash(self, doc_uid: str, content_hash: str) -> None:
        """Record ``content_hash`` on an existing Document node.

        The last step of a run whose every write was reported complete, and
        what a later ``ingest()`` of the same content short-circuits on.
        ``upsert_nodes`` merges on id, so this only adds the property; the
        Document node and its ``path`` / metadata were written by
        :meth:`_build_lexical_graph`.
        """
        await self.graph_store.upsert_nodes(
            [GraphNode(id=doc_uid, label="Document", properties={"content_hash": content_hash})]
        )

    async def _write_mentions(self, graph_data: GraphData, ctx: Context) -> tuple[int, str | None]:
        """Write MENTIONED_IN edges linking entities to their source chunks.

        Every entity connects to every chunk it was extracted from (uncapped).
        With global dedup controlling entity cardinality, uncapped mentions
        provide richer entity-chunk connectivity for retrieval.

        Returns ``(edges attempted, shortfall)`` where ``shortfall`` is ``None``
        when the store reported every edge written.
        """
        mentions: list[EntityMention] = graph_data.mentions or []

        if not mentions:
            return 0, None

        mention_rels: list[GraphRelationship] = []
        for m in merge_mentions(mentions):
            props: dict = {}
            if m.spans:
                props["spans"] = list(m.spans)
            if m.descriptions:
                # As on a node: joined for readers, the list kept whole when
                # there is more than one, so each can be taken off on its own.
                props["description"] = " | ".join(m.descriptions)
                if len(m.descriptions) > 1:
                    props["descriptions"] = list(m.descriptions)
            mention_rels.append(
                GraphRelationship(
                    start_node_id=m.entity_id,
                    end_node_id=m.chunk_id,
                    type="MENTIONED_IN",
                    properties=props,
                )
            )
        written = await self.graph_store.upsert_relationships(mention_rels)
        ctx.log(f"Wrote {len(mention_rels)} MENTIONED_IN edges (uncapped)")
        if _reported_short(written, len(mention_rels)):
            return len(mention_rels), f"mentions {written}/{len(mention_rels)}"
        return len(mention_rels), None
