# GraphRAG SDK 2.0 — Ingestion: Schema-Guided LLM Extraction
# Origin: Neo4j SchemaBuilder + LLMEntityRelationExtractor pattern.
#
# The LLM receives the schema definition and text, then extracts
# entities and relationships constrained to the defined types.

from __future__ import annotations

import logging
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import ExtractionError
from graphrag_sdk.core.models import (
    GraphData,
    GraphNode,
    GraphRelationship,
    GraphSchema,
    TextChunks,
)
from graphrag_sdk.core.providers import LLMInterface
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy

logger = logging.getLogger(__name__)


_EXTRACTION_PROMPT = (
    "You are an expert knowledge graph builder.\n"
    "Extract entities and relationships from the text below.\n"
    "Only use the entity types, relationship types, and patterns defined in the schema.\n\n"
    "## Schema\n"
    "Entity types: {entity_types}\n"
    "Relationship types: {relation_types}\n"
    "Allowed patterns: {patterns}\n\n"
    "## Text\n"
    "{text}\n\n"
    "## Instructions\n"
    "Return a JSON object with two arrays:\n"
    '- "nodes": [{{"id": "<unique_id>", "label": "<entity_type>", '
    '"properties": {{...}}}}]\n'
    '- "relationships": [{{"start_node_id": "<id>", "end_node_id": "<id>", '
    '"type": "<rel_type>", "properties": {{...}}}}]\n\n'
    "Return ONLY valid JSON."
)


class SchemaGuidedExtraction(ExtractionStrategy):
    """Extract entities and relationships using an LLM guided by a schema.

    The LLM is prompted with the schema definition and instructed to
    only extract entities/relationships matching the defined types and
    patterns. This is the primary extraction strategy for v1.

    Args:
        llm: LLM provider for extraction.
        chunk_batch_size: Number of chunks to process per LLM call.
    """

    def __init__(
        self,
        llm: LLMInterface,
        chunk_batch_size: int = 1,
    ) -> None:
        self.llm = llm
        self.chunk_batch_size = chunk_batch_size

    async def extract(
        self,
        chunks: TextChunks,
        schema: GraphSchema,
        ctx: Context,
    ) -> GraphData:
        ctx.log(f"Extracting from {len(chunks.chunks)} chunks (schema-guided)")

        all_nodes: list[GraphNode] = []
        all_relationships: list[GraphRelationship] = []

        entity_types = ", ".join(e.label for e in schema.entities)
        relation_types = ", ".join(r.label for r in schema.relations)
        patterns = "; ".join(
            f"({p.source})-[{p.relationship}]->({p.target})"
            for p in schema.patterns
        )

        for chunk in chunks.chunks:
            if ctx.budget_exceeded:
                ctx.log("Latency budget exceeded, stopping extraction", logging.WARNING)
                break

            prompt = _EXTRACTION_PROMPT.format(
                entity_types=entity_types or "any",
                relation_types=relation_types or "any",
                patterns=patterns or "any",
                text=chunk.text,
            )

            try:
                response = await self.llm.ainvoke(prompt)
                graph_data = self._parse_response(response.content, chunk.uid)
                all_nodes.extend(graph_data.nodes)
                all_relationships.extend(graph_data.relationships)
            except Exception as exc:
                ctx.log(
                    f"Extraction failed for chunk {chunk.index}: {exc}",
                    logging.WARNING,
                )

        ctx.log(
            f"Extracted {len(all_nodes)} nodes, "
            f"{len(all_relationships)} relationships"
        )
        return GraphData(nodes=all_nodes, relationships=all_relationships)

    def _parse_response(self, content: str, source_chunk_id: str) -> GraphData:
        """Parse LLM JSON response into GraphData.

        Attaches ``source_chunk_id`` to every node/relationship for provenance.
        """
        import json

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"LLM returned invalid JSON: {exc}") from exc

        nodes: list[GraphNode] = []
        for n in data.get("nodes", []):
            props = n.get("properties", {})
            props["source_chunk_id"] = source_chunk_id
            nodes.append(
                GraphNode(
                    id=str(n["id"]),
                    label=n["label"],
                    properties=props,
                )
            )

        relationships: list[GraphRelationship] = []
        for r in data.get("relationships", []):
            props = r.get("properties", {})
            props["source_chunk_id"] = source_chunk_id
            relationships.append(
                GraphRelationship(
                    start_node_id=str(r["start_node_id"]),
                    end_node_id=str(r["end_node_id"]),
                    type=r["type"],
                    properties=props,
                )
            )

        return GraphData(nodes=nodes, relationships=relationships)
