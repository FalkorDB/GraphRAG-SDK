# GraphRAG SDK 2.0 — Ingestion: Merged Extraction (HippoRAG + LightRAG)
# Combines LightRAG-style rich typed extraction with HippoRAG-style
# fact triples and entity mentions.

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    EntityMention,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionOutput,
    FactTriple,
    GraphData,
    GraphNode,
    GraphRelationship,
    GraphSchema,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.core.providers import Embedder, LLMInterface
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy

logger = logging.getLogger(__name__)

TUPLE_DELIMITER = "<|#|>"
RECORD_DELIMITER = "##"

_MERGED_EXTRACTION_PROMPT = (
    "You are an expert knowledge graph builder.\n"
    "Extract all entities and relationships from the text below.\n\n"
    "## Schema Constraints\n"
    "Entity types: {entity_types}\n"
    "Relationship types: {relation_types}\n\n"
    "## Output Format\n"
    "Return each extracted item on its own line using the delimited format below.\n"
    "Use '{tuple_delim}' as the field delimiter and '{record_delim}' as the record delimiter.\n\n"
    "For entities:\n"
    '("entity"{tuple_delim}<entity_name>{tuple_delim}<entity_type>{tuple_delim}<entity_description>)'
    "{record_delim}\n"
    "For relationships:\n"
    '("relationship"{tuple_delim}<source_entity>{tuple_delim}<target_entity>'
    "{tuple_delim}<relationship_type>{tuple_delim}<relationship_keywords>"
    "{tuple_delim}<relationship_description>{tuple_delim}<weight>){record_delim}\n\n"
    "## Text\n"
    "{text}\n\n"
    "## Instructions\n"
    "- Extract ALL entities and relationships present in the text.\n"
    "- Entity names should be human-readable identifiers (not UUIDs).\n"
    "- Descriptions should capture key attributes mentioned in the text.\n"
    "- Weight is a float 0-1 indicating confidence/strength.\n"
    "- Return ONLY the delimited tuples, nothing else.\n"
)

_GLEANING_PROMPT = (
    "The following text was previously analysed and some entities/relationships "
    "were extracted. Review the text again and extract any MISSED entities or "
    "relationships that were not captured in the first pass.\n\n"
    "## Already Extracted\n"
    "{already_extracted}\n\n"
    "## Text\n"
    "{text}\n\n"
    "Return additional extractions in the same delimited format. "
    "If nothing was missed, return an empty string.\n"
    "Use '{tuple_delim}' as the field delimiter and '{record_delim}' as the record delimiter.\n"
    "For entities:\n"
    '("entity"{tuple_delim}<entity_name>{tuple_delim}<entity_type>{tuple_delim}<entity_description>)'
    "{record_delim}\n"
    "For relationships:\n"
    '("relationship"{tuple_delim}<source_entity>{tuple_delim}<target_entity>'
    "{tuple_delim}<relationship_type>{tuple_delim}<relationship_keywords>"
    "{tuple_delim}<relationship_description>{tuple_delim}<weight>){record_delim}\n"
)


def compute_entity_id(name: str) -> str:
    """Deterministic entity ID from normalized name."""
    return name.strip().lower().replace(" ", "_")


class MergedExtraction(ExtractionStrategy):
    """Merged extraction combining LightRAG + HippoRAG patterns.

    Per-chunk extraction with optional gleaning (second LLM pass),
    producing rich entities (with descriptions), fact triples, and
    entity mentions.

    Args:
        llm: LLM provider for extraction.
        embedder: Embedder (reserved for future use).
        enable_gleaning: If True, perform a second LLM pass to catch missed entities.
        max_concurrency: Maximum parallel LLM calls.
    """

    def __init__(
        self,
        llm: LLMInterface,
        embedder: Embedder | None = None,
        enable_gleaning: bool = False,
        max_concurrency: int = 12,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.enable_gleaning = enable_gleaning
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def extract(
        self,
        chunks: TextChunks,
        schema: GraphSchema,
        ctx: Context,
    ) -> GraphData:
        ctx.log(
            f"Extracting from {len(chunks.chunks)} chunks (merged strategy, "
            f"gleaning={'on' if self.enable_gleaning else 'off'})"
        )

        entity_types = ", ".join(e.label for e in schema.entities) or "any"
        relation_types = ", ".join(r.label for r in schema.relations) or "any"

        all_entities: list[ExtractedEntity] = []
        all_relations: list[ExtractedRelation] = []
        all_facts: list[FactTriple] = []
        all_mentions: list[EntityMention] = []

        tasks = [
            self._extract_chunk(chunk, entity_types, relation_types, ctx)
            for chunk in chunks.chunks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                ctx.log(f"Chunk extraction error: {result}", logging.WARNING)
                continue
            entities, relations = result
            all_entities.extend(entities)
            all_relations.extend(relations)

        # Generate HippoRAG-style facts and mentions
        for rel in all_relations:
            for chunk_id in rel.source_chunk_ids:
                all_facts.append(rel.to_fact_triple(chunk_id))

        for ent in all_entities:
            ent_id = compute_entity_id(ent.name)
            for chunk_id in ent.source_chunk_ids:
                all_mentions.append(
                    EntityMention(chunk_id=chunk_id, entity_id=ent_id)
                )

        # Aggregate across chunks: dedup entities by normalized name
        merged_entities = self._aggregate_entities(all_entities)
        merged_relations = self._aggregate_relations(all_relations)

        # Convert to GraphData nodes + relationships
        nodes = self._entities_to_nodes(merged_entities)
        relationships = self._relations_to_relationships(merged_relations)

        # Attach extra data via extra="allow"
        graph_data = GraphData(nodes=nodes, relationships=relationships)
        graph_data.facts = all_facts  # type: ignore[attr-defined]
        graph_data.mentions = all_mentions  # type: ignore[attr-defined]
        graph_data.extracted_entities = merged_entities  # type: ignore[attr-defined]
        graph_data.extracted_relations = merged_relations  # type: ignore[attr-defined]

        ctx.log(
            f"Extracted {len(nodes)} nodes, {len(relationships)} relationships, "
            f"{len(all_facts)} facts, {len(all_mentions)} mentions"
        )
        return graph_data

    async def _extract_chunk(
        self,
        chunk: TextChunk,
        entity_types: str,
        relation_types: str,
        ctx: Context,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        """Extract entities and relations from a single chunk."""
        async with self._semaphore:
            if ctx.budget_exceeded:
                ctx.log("Latency budget exceeded, stopping extraction", logging.WARNING)
                return [], []

            prompt = _MERGED_EXTRACTION_PROMPT.format(
                entity_types=entity_types,
                relation_types=relation_types,
                tuple_delim=TUPLE_DELIMITER,
                record_delim=RECORD_DELIMITER,
                text=chunk.text,
            )

            try:
                response = await self.llm.ainvoke(prompt)
                entities, relations = self._parse_delimiter_response(
                    response.content, chunk.uid
                )

                # Optional gleaning pass
                if self.enable_gleaning and (entities or relations):
                    already = response.content
                    gleaning_prompt = _GLEANING_PROMPT.format(
                        already_extracted=already,
                        text=chunk.text,
                        tuple_delim=TUPLE_DELIMITER,
                        record_delim=RECORD_DELIMITER,
                    )
                    gleaning_response = await self.llm.ainvoke(gleaning_prompt)
                    extra_entities, extra_relations = self._parse_delimiter_response(
                        gleaning_response.content, chunk.uid
                    )
                    entities.extend(extra_entities)
                    relations.extend(extra_relations)

                return entities, relations
            except Exception as exc:
                ctx.log(
                    f"Extraction failed for chunk {chunk.index}: {exc}",
                    logging.WARNING,
                )
                return [], []

    def _parse_delimiter_response(
        self,
        content: str,
        source_chunk_id: str,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        """Parse LightRAG-style delimiter format into entities and relations."""
        entities: list[ExtractedEntity] = []
        relations: list[ExtractedRelation] = []

        records = content.split(RECORD_DELIMITER)

        for record in records:
            record = record.strip()
            if not record:
                continue

            # Extract content between parentheses
            match = re.search(r"\((.+)\)", record, re.DOTALL)
            if not match:
                continue
            inner = match.group(1)

            parts = [p.strip().strip('"').strip("'") for p in inner.split(TUPLE_DELIMITER)]

            if len(parts) >= 4 and parts[0].lower() == "entity":
                entities.append(
                    ExtractedEntity(
                        name=parts[1],
                        type=parts[2],
                        description=parts[3],
                        source_chunk_ids=[source_chunk_id],
                    )
                )
            elif len(parts) >= 6 and parts[0].lower() == "relationship":
                weight = 1.0
                if len(parts) >= 7:
                    try:
                        weight = float(parts[6])
                    except (ValueError, IndexError):
                        weight = 1.0
                relations.append(
                    ExtractedRelation(
                        source=parts[1],
                        target=parts[2],
                        type=parts[3],
                        keywords=parts[4],
                        description=parts[5],
                        weight=weight,
                        source_chunk_ids=[source_chunk_id],
                    )
                )

        return entities, relations

    def _aggregate_entities(
        self,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        """Deduplicate entities by normalized name, keeping longer descriptions."""
        seen: dict[str, ExtractedEntity] = {}
        for ent in entities:
            key = ent.name.strip().lower()
            if key in seen:
                existing = seen[key]
                # Keep longer description
                if len(ent.description) > len(existing.description):
                    existing.description = ent.description
                # Accumulate source chunks
                for cid in ent.source_chunk_ids:
                    if cid not in existing.source_chunk_ids:
                        existing.source_chunk_ids.append(cid)
            else:
                seen[key] = ExtractedEntity(
                    name=ent.name,
                    type=ent.type,
                    description=ent.description,
                    source_chunk_ids=list(ent.source_chunk_ids),
                )
        return list(seen.values())

    def _aggregate_relations(
        self,
        relations: list[ExtractedRelation],
    ) -> list[ExtractedRelation]:
        """Deduplicate relations by sorted (source, target)."""
        seen: dict[tuple[str, str, str], ExtractedRelation] = {}
        for rel in relations:
            key = (rel.source.strip().lower(), rel.type.strip().lower(), rel.target.strip().lower())
            if key in seen:
                existing = seen[key]
                for cid in rel.source_chunk_ids:
                    if cid not in existing.source_chunk_ids:
                        existing.source_chunk_ids.append(cid)
                if len(rel.description) > len(existing.description):
                    existing.description = rel.description
            else:
                seen[key] = ExtractedRelation(
                    source=rel.source,
                    target=rel.target,
                    type=rel.type,
                    keywords=rel.keywords,
                    description=rel.description,
                    weight=rel.weight,
                    source_chunk_ids=list(rel.source_chunk_ids),
                )
        return list(seen.values())

    def _entities_to_nodes(
        self,
        entities: list[ExtractedEntity],
    ) -> list[GraphNode]:
        """Convert ExtractedEntity list to GraphNode list."""
        nodes: list[GraphNode] = []
        for ent in entities:
            node_id = compute_entity_id(ent.name)
            nodes.append(
                GraphNode(
                    id=node_id,
                    label=ent.type,
                    properties={
                        "name": ent.name,
                        "description": ent.description,
                        "source_chunk_ids": ent.source_chunk_ids,
                    },
                )
            )
        return nodes

    def _relations_to_relationships(
        self,
        relations: list[ExtractedRelation],
    ) -> list[GraphRelationship]:
        """Convert ExtractedRelation list to GraphRelationship list."""
        relationships: list[GraphRelationship] = []
        for rel in relations:
            relationships.append(
                GraphRelationship(
                    start_node_id=compute_entity_id(rel.source),
                    end_node_id=compute_entity_id(rel.target),
                    type=rel.type,
                    properties={
                        "keywords": rel.keywords,
                        "description": rel.description,
                        "weight": rel.weight,
                        "source_chunk_ids": rel.source_chunk_ids,
                    },
                )
            )
        return relationships
