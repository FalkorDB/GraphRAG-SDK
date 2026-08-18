# GraphRAG SDK — Ingestion: Structured Pipeline
# The deterministic write path for records: no model is called anywhere in here.
#
# Deliberately generic. It never names a source, a column or a label, so a new
# source costs a declaration and not code.

from __future__ import annotations

import hashlib
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    DocumentInfo,
    GraphNode,
    GraphRelationship,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import compute_entity_id
from graphrag_sdk.ingestion.lexical_graph import LexicalGraphWriter
from graphrag_sdk.ingestion.loaders.record_loader import RecordBatch, RecordLoaderStrategy
from graphrag_sdk.ingestion.mapping import MappingError, NodeMapping, RecordMapping
from graphrag_sdk.storage.graph_store import ReferenceNode

# Chunk properties the writer owns. A column with one of these names is stored
# under a "col_" prefix rather than silently overwriting the chunk's identity.
_CHUNK_RESERVED = frozenset({"id", "text", "index", "kind", "record_key", "embedding"})


def record_chunk_uid(document_uid: str, record_key: str) -> str:
    """Deterministic chunk id for one record.

    Keyed on the *effective* document uid, never on a canonical one. Keying a
    record chunk on a canonical document id makes a pending update MERGE onto the
    live document's chunks, which the cutover then deletes: measured as three
    chunks before and zero after, with no exception raised.
    """
    digest = hashlib.sha256(f"{document_uid}::{record_key}".encode()).hexdigest()
    return f"rec_{digest[:24]}"


def render_record(record: dict[str, Any]) -> str:
    """Render a record as a sentence.

    ``text`` is the only field chunk embedding and full-text search read, and
    embedders are trained on language. Key-value soup separated by punctuation
    embeds poorly and loses to real prose in vector search, so a record that is
    never retrieved would be a record that may as well not be a chunk.
    """
    parts = [
        f"{key.replace('_', ' ')} {value}"
        for key, value in record.items()
        if value not in (None, "")
    ]
    return ", ".join(parts) + "." if parts else ""


def record_cells(record: dict[str, Any]) -> dict[str, Any]:
    """The record's cells verbatim, to sit on the chunk beside the rendered text.

    The typed projection lives on the entity, where queries and aggregation read
    it. This is the faithful record of what the source said, so the original row
    is recoverable from the graph without parsing the rendered sentence.
    """
    cells: dict[str, Any] = {}
    for key, value in record.items():
        if value in (None, ""):
            continue
        cells[f"col_{key}" if key in _CHUNK_RESERVED else key] = value
    return cells


class StructuredIngestionResult:
    """Counts from one structured ingest."""

    __slots__ = ("records", "chunks", "entities", "references", "edges", "document_id")

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        self.records = 0
        self.chunks = 0
        self.entities = 0
        self.references = 0
        self.edges = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "records": self.records,
            "chunks": self.chunks,
            "entities": self.entities,
            "references": self.references,
            "edges": self.edges,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StructuredIngestionResult({self.as_dict()})"


class StructuredIngestionPipeline(LexicalGraphWriter):
    """Writes a structured source into the graph, deterministically.

    Executes a fixed sequence, none of which consults a model:

    1. **Load** — parse the source into a re-openable record stream
    2. **Validate** — check the mapping against the source's real header
    3. **Lexical graph** — a Document, and one Chunk per record
    4. **Map** — declared columns become typed entity nodes and RELATES edges
    5. **Write** — entities, then references ON CREATE, then edges

    The same input always produces the same graph, because identity comes from a
    declared key and every property type is declared rather than inferred.

    Args:
        loader: How to read the source into records.
        graph_store: Storage layer.
        vector_store: Optional, used only so callers can index the new chunks.
    """

    def __init__(
        self,
        loader: RecordLoaderStrategy,
        graph_store: Any,
        vector_store: Any | None = None,
    ) -> None:
        self.loader = loader
        self.graph_store = graph_store
        self.vector_store = vector_store

    async def run(
        self,
        source: str,
        mapping: RecordMapping,
        ctx: Context | None = None,
        *,
        document_id: str | None = None,
        strict: bool = False,
        link_sequential: bool = False,
    ) -> StructuredIngestionResult:
        """Ingest one structured source.

        Args:
            source: Path or identifier for the source.
            mapping: The declaration.
            ctx: Execution context.
            document_id: Overrides the Document node id.
            strict: Fail when the source has a column the mapping never reads.
                Off by default because ignoring a column is a legitimate choice.
            link_sequential: Whether to chain NEXT_CHUNK between records. False,
                and that is not cosmetic: records have no reading order, and
                text-to-Cypher is told NEXT_CHUNK means "the next sequential
                Chunk", so chaining unrelated rows asserts a sequence that does
                not exist.

        Raises:
            MappingError: If the mapping does not fit the source. Nothing is
                written in that case.
        """
        ctx = ctx or Context()
        batch = await self.loader.load_records(source, ctx)
        doc_info = batch.document_info
        if document_id:
            doc_info = DocumentInfo(
                uid=document_id, path=doc_info.path, metadata=dict(doc_info.metadata)
            )

        problems = mapping.validate_against(batch.columns, strict=strict)
        if problems:
            raise MappingError(f"mapping does not fit {source}:\n  " + "\n  ".join(problems))

        result = StructuredIngestionResult(doc_info.uid)

        # Step 3. Every record becomes a Chunk, through the same writer the prose
        # path uses, so both halves are the same shape.
        chunks = self._record_chunks(batch, mapping, doc_info, result)
        if not chunks.chunks:
            ctx.log(f"{source} produced no records, nothing written")
            return result
        await self._build_lexical_graph(doc_info, chunks, ctx, link_sequential=link_sequential)

        # Steps 4 and 5. Declared columns become entities and edges.
        nodes, references, edges = self._map_records(batch, mapping, doc_info, result)
        await self.graph_store.upsert_nodes(nodes)
        await self.graph_store.upsert_reference_nodes(references)
        await self.graph_store.upsert_relationships(edges)

        ctx.log(
            f"Structured ingest of {doc_info.uid}: {result.records} records, "
            f"{result.entities} entities, {result.references} references, "
            f"{result.edges} edges"
        )
        return result

    # ── internals ───────────────────────────────────────────────

    def _record_chunks(
        self,
        batch: RecordBatch,
        mapping: RecordMapping,
        doc_info: DocumentInfo,
        result: StructuredIngestionResult,
    ) -> TextChunks:
        """One TextChunk per record, carrying the cells alongside the text."""
        anchor = mapping.anchor
        chunks: list[TextChunk] = []
        for index, record in enumerate(batch):
            record_key = str(record.get(anchor.key) or "").strip()
            if not record_key:
                # Without its key a record has no stable identity, so it could
                # not be updated or deleted later. Skipping is the honest choice.
                continue
            chunks.append(
                TextChunk(
                    uid=record_chunk_uid(doc_info.uid, record_key),
                    text=render_record(record),
                    index=index,
                    metadata={
                        "kind": "record",
                        "record_key": record_key,
                        **record_cells(record),
                    },
                )
            )
        result.records = result.chunks = len(chunks)
        return TextChunks(chunks=chunks)

    def _map_records(
        self,
        batch: RecordBatch,
        mapping: RecordMapping,
        doc_info: DocumentInfo,
        result: StructuredIngestionResult,
    ) -> tuple[list[GraphNode], list[ReferenceNode], list[GraphRelationship]]:
        """Second pass: declared columns become nodes and edges.

        This is why RecordBatch hands over a factory. A one-shot iterator is
        empty by the time this pass runs, and nothing raises.
        """
        anchor = mapping.anchor
        nodes: list[GraphNode] = []
        references: list[ReferenceNode] = []
        edges: list[GraphRelationship] = []

        for record in batch:
            record_key = str(record.get(anchor.key) or "").strip()
            if not record_key:
                continue
            chunk_uid = record_chunk_uid(doc_info.uid, record_key)
            ids: dict[str, str] = {}

            for node in mapping.nodes:
                raw_key = record.get(node.key)
                if raw_key in (None, ""):
                    continue
                node_id = compute_entity_id(str(raw_key), node.label)
                if not node_id:
                    continue
                ids[node.alias] = node_id
                if node.reference:
                    # A record that denormalises the referenced entity's name
                    # ("org_id" plus "org_name") can label the stub properly.
                    # Falling back to the raw key would name a node "ORG-42",
                    # which then resolves against nothing and reads as a real
                    # name to whoever queries it.
                    fallback = str(raw_key)
                    if node.name:
                        declared_name = str(record.get(node.name) or "").strip()
                        if declared_name:
                            fallback = declared_name
                    references.append(
                        ReferenceNode(
                            id=node_id,
                            label=node.label,
                            name=fallback,
                            # The key, so the placeholder is joinable by the same
                            # column the mapping declared.
                            properties={node.key: str(raw_key), "is_stub": True},
                        )
                    )
                    result.references += 1
                else:
                    nodes.append(
                        GraphNode(
                            id=node_id,
                            label=node.label,
                            properties=self._node_properties(node, record, str(raw_key)),
                        )
                    )
                    result.entities += 1
                edges.append(
                    GraphRelationship(
                        start_node_id=node_id,
                        end_node_id=chunk_uid,
                        type="MENTIONED_IN",
                    )
                )

            for edge in mapping.edges:
                start, end = ids.get(edge.source), ids.get(edge.target)
                if not start or not end or start == end:
                    # A self loop means both aliases resolved to the same entity,
                    # which is a fact about the data, not an edge worth writing.
                    continue
                properties: dict[str, Any] = {
                    "rel_type": edge.type,
                    "fact": f"({edge.source}, {edge.type}, {edge.target})",
                    "source_chunk_ids": [chunk_uid],
                }
                for prop, column in edge.properties.items():
                    value = column.cast(record.get(column.name))
                    if value is not None:
                        properties[prop] = value
                edges.append(
                    GraphRelationship(
                        start_node_id=start,
                        end_node_id=end,
                        type="RELATES",
                        properties=properties,
                    )
                )
                result.edges += 1

        return nodes, references, edges

    @staticmethod
    def _node_properties(node: NodeMapping, record: dict[str, Any], raw_key: str) -> dict[str, Any]:
        """Typed properties for one entity, plus the alias that lets it resolve."""
        properties: dict[str, Any] = {node.key: raw_key, "is_stub": False}
        if node.name:
            display = record.get(node.name)
            if display not in (None, ""):
                properties["name"] = str(display)
                # This source holds the key AND the name, so it can publish the
                # id an extractor reading prose would independently compute for
                # the same thing. That published id is what lets a keyed node and
                # an extracted node resolve to one, by string equality.
                alias = compute_entity_id(str(display), node.label)
                if alias and alias != compute_entity_id(raw_key, node.label):
                    properties["alias_ids"] = [alias]
        for prop, column in node.properties.items():
            value = column.cast(record.get(column.name))
            if value is not None:
                properties[prop] = value
        return properties
