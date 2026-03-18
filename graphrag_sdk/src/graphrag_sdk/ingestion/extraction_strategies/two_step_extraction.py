# GraphRAG SDK 2.0 — Ingestion: Hybrid Extraction Strategy
# Composable 2-step extraction: pluggable entity NER (step 1) +
# LLM verify/enrich/relationship extraction (step 2).
# Benchmark-winning: Entity F1 80.8%, Relation F1 72.1%.

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    EntityMention,
    ExtractedEntity,
    ExtractedRelation,
    GraphData,
    GraphNode,
    GraphRelationship,
    GraphSchema,
    TextChunks,
)
from graphrag_sdk.core.providers import Embedder, LLMInterface
from graphrag_sdk.ingestion.extraction_strategies._entity_utils import (
    DEFAULT_ENTITY_TYPES,
    compute_entity_id,
    is_valid_entity_name,
    label_for_type,
)
from graphrag_sdk.ingestion.extraction_strategies._prompts import (
    VERIFY_EXTRACT_RELS_PROMPT,
)
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
from graphrag_sdk.ingestion.extraction_strategies.coref_resolvers import CorefResolver
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
    EntityExtractor,
)

logger = logging.getLogger(__name__)


class TwoStepExtraction(ExtractionStrategy):
    """Composable 2-step extraction with pluggable entity NER and coreference.

    **Step 1** — Entity extraction via a pluggable ``EntityExtractor``
    (default: LLM-based NER). Supports LLM, GLiNER2, spaCy backends.

    **Step 2** — LLM verification + relationship extraction. The LLM
    receives the pre-extracted entities and original text, verifies
    entities, and extracts relationships.

    Optionally applies coreference resolution to each chunk before
    extraction (requires a ``CorefResolver`` instance).

    Args:
        llm: LLM provider for step 2 (verify + relationship extraction).
        entity_extractor: Pluggable NER backend. Default: GLiNER2 (local, fast).
            Pass ``EntityExtractor(llm=llm)`` to use LLM for step 1 instead.
        coref_resolver: Optional coreference resolver applied per-chunk.
        embedder: Embedder for type taxonomy resolution.
        entity_types: Entity type labels. Default: DEFAULT_ENTITY_TYPES.
            Overridden by schema.entities if present.
        max_concurrency: Maximum parallel LLM calls.
    """

    def __init__(
        self,
        llm: LLMInterface,
        *,
        entity_extractor: EntityExtractor | None = None,
        coref_resolver: CorefResolver | None = None,
        embedder: Embedder | None = None,
        entity_types: list[str] | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self.llm = llm
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.coref_resolver = coref_resolver
        self.embedder = embedder
        self.entity_types = entity_types or list(DEFAULT_ENTITY_TYPES)
        self._max_concurrency = max_concurrency

    async def extract(
        self,
        chunks: TextChunks,
        schema: GraphSchema,
        ctx: Context,
    ) -> GraphData:
        # Resolve entity types: schema overrides instance default
        entity_types = (
            [e.label for e in schema.entities]
            if schema.entities
            else list(self.entity_types)
        )

        ctx.log(
            f"Extracting from {len(chunks.chunks)} chunks (hybrid, "
            f"extractor={self.entity_extractor.__class__.__name__}, "
            f"coref={'on' if self.coref_resolver else 'off'})"
        )

        # Budget-aware chunk filtering
        active_chunks = []
        for chunk in chunks.chunks:
            if ctx.budget_exceeded:
                ctx.log("Latency budget exceeded, stopping extraction", logging.WARNING)
                break
            active_chunks.append(chunk)

        if not active_chunks:
            return GraphData(nodes=[], relationships=[])

        # ── Optional: Coreference resolution per chunk ──
        chunk_texts: list[str] = []
        if self.coref_resolver is not None:
            for chunk in active_chunks:
                try:
                    resolved = await self.coref_resolver.resolve(chunk.text)
                    chunk_texts.append(resolved)
                except Exception as exc:
                    logger.warning("Coref failed for chunk %s: %s", chunk.uid, exc)
                    chunk_texts.append(chunk.text)
        else:
            chunk_texts = [chunk.text for chunk in active_chunks]

        # ── Step 1: Entity extraction (pluggable) ──
        # For LLM extractors, use abatch_invoke pattern;
        # for local extractors, use asyncio.gather
        chunk_entities: list[list[ExtractedEntity]] = []

        if self.entity_extractor.mode == "llm":
            # Batch through LLM for efficiency
            from graphrag_sdk.ingestion.extraction_strategies._prompts import NER_PROMPT

            prompts = [
                NER_PROMPT.format(
                    entity_types=", ".join(entity_types),
                    text=text,
                )
                for text in chunk_texts
            ]
            batch_kw: dict[str, Any] = {}
            if self._max_concurrency is not None:
                batch_kw["max_concurrency"] = self._max_concurrency

            step1_results = await self.llm.abatch_invoke(prompts, **batch_kw)

            for item in step1_results:
                chunk = active_chunks[item.index]
                if not item.ok:
                    ctx.log(
                        f"Step 1 NER failed for chunk {chunk.index}: {item.error}",
                        logging.WARNING,
                    )
                    chunk_entities.append([])
                    continue
                assert item.response is not None
                parsed = EntityExtractor._parse_llm_response(
                    item.response.content, entity_types, chunk.uid,
                    self.entity_extractor._threshold,
                )
                chunk_entities.append(parsed)
        else:
            # Local extractors (GLiNER2, custom): use asyncio.gather
            tasks = [
                self.entity_extractor.extract_entities(
                    text, entity_types, chunk.uid
                )
                for text, chunk in zip(chunk_texts, active_chunks)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    ctx.log(
                        f"Step 1 NER failed for chunk {active_chunks[i].index}: {result}",
                        logging.WARNING,
                    )
                    chunk_entities.append([])
                else:
                    chunk_entities.append(result)

        # ── Step 2: LLM verify + relationship extraction ──
        step2_prompts: list[str] = []
        step2_indices: list[int] = []  # maps prompt index -> active_chunk index

        for i, (text, ents) in enumerate(zip(chunk_texts, chunk_entities)):
            entities_json = json.dumps(
                [{"name": e.name, "type": e.type, "description": e.description} for e in ents]
            )
            prompt = VERIFY_EXTRACT_RELS_PROMPT.format(
                entity_types=", ".join(entity_types),
                entities_json=entities_json,
                text=text,
            )
            step2_prompts.append(prompt)
            step2_indices.append(i)

        batch_kw2: dict[str, Any] = {}
        if self._max_concurrency is not None:
            batch_kw2["max_concurrency"] = self._max_concurrency

        all_entities: list[ExtractedEntity] = []
        all_relations: list[ExtractedRelation] = []

        if step2_prompts:
            step2_results = await self.llm.abatch_invoke(step2_prompts, **batch_kw2)

            for item in step2_results:
                chunk_idx = step2_indices[item.index]
                chunk = active_chunks[chunk_idx]
                if not item.ok:
                    ctx.log(
                        f"Step 2 verify+rels failed for chunk {chunk.index}: {item.error}",
                        logging.WARNING,
                    )
                    # Fall back to step 1 entities only
                    all_entities.extend(chunk_entities[chunk_idx])
                    continue

                assert item.response is not None
                verified_ents, rels = self._parse_step2_response(
                    item.response.content, entity_types, chunk.uid
                )
                if verified_ents:
                    # Carry over spans/confidence from step 1 entities
                    step1_ents = chunk_entities[chunk_idx]
                    self._merge_step1_metadata(verified_ents, step1_ents)
                    all_entities.extend(verified_ents)
                else:
                    # LLM returned no entities — use step 1 entities
                    all_entities.extend(chunk_entities[chunk_idx])
                all_relations.extend(rels)

        # ── Aggregate across chunks ──
        merged_entities = self._aggregate_entities(all_entities)
        merged_relations = self._aggregate_relations(all_relations)

        # ── Generate mentions ──
        all_mentions: list[EntityMention] = []
        for ent in merged_entities:
            ent_id = compute_entity_id(ent.name, ent.type)
            for chunk_id in ent.source_chunk_ids:
                all_mentions.append(EntityMention(chunk_id=chunk_id, entity_id=ent_id))

        # ── Build entity type map for relationship endpoint resolution ──
        entity_type_map: dict[str, str] = {}
        for ent in merged_entities:
            entity_type_map[ent.name.strip().lower()] = ent.type

        # ── Convert to GraphData ──
        nodes = self._entities_to_nodes(merged_entities)
        relationships = self._relations_to_relationships(merged_relations, entity_type_map)

        graph_data = GraphData(
            nodes=nodes,
            relationships=relationships,
            mentions=all_mentions,
            extracted_entities=merged_entities,
            extracted_relations=merged_relations,
        )

        ctx.log(
            f"Extracted {len(nodes)} nodes, {len(relationships)} relationships, "
            f"{len(all_mentions)} mentions"
        )
        return graph_data

    @staticmethod
    def _merge_step1_metadata(
        verified: list[ExtractedEntity],
        step1: list[ExtractedEntity],
    ) -> None:
        """Carry over spans and confidence from step 1 entities into step 2 verified entities.

        Matches by normalized name. Step 1 spans (from GLiNER2) take
        priority over step 2 spans (from LLM) since GLiNER2 character
        offsets are more precise. Step 2 spans are kept for entities
        that GLiNER2 didn't find.
        """
        # Build lookup: normalized name → step 1 entity
        s1_lookup: dict[str, ExtractedEntity] = {}
        for ent in step1:
            key = ent.name.strip().lower()
            if key not in s1_lookup:
                s1_lookup[key] = ent

        for ent in verified:
            s1 = s1_lookup.get(ent.name.strip().lower())
            if s1 is not None:
                # Step 1 match — use GLiNER2 spans (more precise)
                s1_spans = getattr(s1, "spans", None)
                if s1_spans:
                    ent.spans = s1_spans  # type: ignore[attr-defined]
                s1_conf = getattr(s1, "confidence", None)
                if s1_conf is not None:
                    ent.confidence = s1_conf  # type: ignore[attr-defined]
            # else: entity is new from step 2 — keep its own spans if present

    # ── Step 2 Response Parsing ──────────────────────────────────

    @staticmethod
    def _parse_step2_response(
        content: str,
        entity_types: list[str],
        source_chunk_id: str,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        """Parse the step 2 LLM response (verified entities + relationships)."""
        text = content.strip()
        # Strip markdown fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "Step 2 returned invalid JSON for chunk %s", source_chunk_id
            )
            return [], []

        if not isinstance(data, dict):
            return [], []

        # Parse entities
        entities: list[ExtractedEntity] = []
        for item in data.get("entities", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not is_valid_entity_name(name):
                continue
            raw_type = str(item.get("type", "")).strip()
            # Reject slash/parenthetical compound types
            if "/" in raw_type or "(" in raw_type or ")" in raw_type:
                continue
            etype = label_for_type(raw_type, entity_types)
            description = str(item.get("description", "")).strip()

            # Entity span from step 2
            extra: dict[str, Any] = {}
            span_start = item.get("span_start")
            span_end = item.get("span_end")
            if span_start is not None and span_end is not None:
                try:
                    extra["spans"] = {
                        source_chunk_id: [{"start": int(span_start), "end": int(span_end)}]
                    }
                except (ValueError, TypeError):
                    pass

            entities.append(
                ExtractedEntity(
                    name=name,
                    type=etype,
                    description=description,
                    source_chunk_ids=[source_chunk_id],
                    **extra,
                )
            )

        # Parse relationships
        relations: list[ExtractedRelation] = []
        for item in data.get("relationships", []):
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            rel_type = str(item.get("type", "")).strip()
            if not source or not target or not rel_type:
                continue
            # Reject relations referencing stoplist entities
            if not is_valid_entity_name(source) or not is_valid_entity_name(target):
                continue

            weight = 1.0
            try:
                weight = float(item.get("weight", 1.0))
            except (ValueError, TypeError):
                weight = 1.0

            description = str(item.get("description", "")).strip()
            keywords = str(item.get("keywords", "")).strip()

            # Span: evidence sentence offsets in the chunk
            extra: dict[str, Any] = {}
            span_start = item.get("span_start")
            span_end = item.get("span_end")
            if span_start is not None and span_end is not None:
                try:
                    extra["spans"] = {
                        source_chunk_id: [{"start": int(span_start), "end": int(span_end)}]
                    }
                except (ValueError, TypeError):
                    pass

            relations.append(
                ExtractedRelation(
                    source=source,
                    target=target,
                    type=rel_type,
                    keywords=keywords,
                    description=description,
                    weight=weight,
                    source_chunk_ids=[source_chunk_id],
                    **extra,
                )
            )

        return entities, relations

    # ── Aggregation ──────────────────────────────────────────────

    @staticmethod
    def _aggregate_entities(
        entities: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        """Deduplicate entities by (normalized name, type), keeping longer descriptions.

        Merges ``spans`` dicts across chunks so the final entity carries
        character offsets from every chunk it appeared in.
        """
        seen: dict[tuple[str, str], ExtractedEntity] = {}
        for ent in entities:
            name_key = ent.name.strip().lower()
            if not name_key:
                continue
            key = (name_key, ent.type.strip().lower())
            if key in seen:
                existing = seen[key]
                if ent.name[0].isupper() and not existing.name[0].isupper():
                    existing.name = ent.name
                if len(ent.description) > len(existing.description):
                    existing.description = ent.description
                for cid in ent.source_chunk_ids:
                    if cid not in existing.source_chunk_ids:
                        existing.source_chunk_ids.append(cid)
                # Merge spans: {chunk_id: [{start, end}, ...]}
                ent_spans = getattr(ent, "spans", None) or {}
                if ent_spans:
                    existing_spans = getattr(existing, "spans", None) or {}
                    for chunk_id, offsets in ent_spans.items():
                        existing_spans.setdefault(chunk_id, []).extend(offsets)
                    existing.spans = existing_spans  # type: ignore[attr-defined]
            else:
                seen[key] = ExtractedEntity(
                    name=ent.name,
                    type=ent.type,
                    description=ent.description,
                    source_chunk_ids=list(ent.source_chunk_ids),
                    **({"spans": dict(ent.spans)} if getattr(ent, "spans", None) else {}),
                    **({"confidence": ent.confidence} if getattr(ent, "confidence", None) is not None else {}),
                )
        return list(seen.values())

    @staticmethod
    def _aggregate_relations(
        relations: list[ExtractedRelation],
    ) -> list[ExtractedRelation]:
        """Deduplicate relations by (source, type, target).

        Merges ``spans`` across chunks for provenance tracking.
        """
        seen: dict[tuple[str, str, str], ExtractedRelation] = {}
        for rel in relations:
            key = (
                rel.source.strip().lower(),
                rel.type.strip().lower(),
                rel.target.strip().lower(),
            )
            if key in seen:
                existing = seen[key]
                for cid in rel.source_chunk_ids:
                    if cid not in existing.source_chunk_ids:
                        existing.source_chunk_ids.append(cid)
                if len(rel.description) > len(existing.description):
                    existing.description = rel.description
                # Merge spans
                rel_spans = getattr(rel, "spans", None) or {}
                if rel_spans:
                    existing_spans = getattr(existing, "spans", None) or {}
                    for chunk_id, offsets in rel_spans.items():
                        existing_spans.setdefault(chunk_id, []).extend(offsets)
                    existing.spans = existing_spans  # type: ignore[attr-defined]
            else:
                seen[key] = ExtractedRelation(
                    source=rel.source,
                    target=rel.target,
                    type=rel.type,
                    keywords=rel.keywords,
                    description=rel.description,
                    weight=rel.weight,
                    source_chunk_ids=list(rel.source_chunk_ids),
                    **({"spans": dict(rel.spans)} if getattr(rel, "spans", None) else {}),
                )
        return list(seen.values())

    # ── GraphData Conversion ─────────────────────────────────────

    @staticmethod
    def _entities_to_nodes(
        entities: list[ExtractedEntity],
    ) -> list[GraphNode]:
        nodes: list[GraphNode] = []
        for ent in entities:
            node_id = compute_entity_id(ent.name, ent.type)
            if not node_id:
                continue
            props: dict[str, Any] = {
                "name": ent.name,
                "description": ent.description,
                "source_chunk_ids": ent.source_chunk_ids,
            }
            # Include spans if present (GLiNER2 provides character offsets)
            spans = getattr(ent, "spans", None)
            if spans:
                props["spans"] = spans
            nodes.append(
                GraphNode(
                    id=node_id,
                    label=ent.type,
                    properties=props,
                )
            )
        return nodes

    @staticmethod
    def _relations_to_relationships(
        relations: list[ExtractedRelation],
        entity_type_map: dict[str, str] | None = None,
    ) -> list[GraphRelationship]:
        entity_type_map = entity_type_map or {}
        relationships: list[GraphRelationship] = []
        for rel in relations:
            fact = (
                f"({rel.source}, {rel.type}, {rel.target}): {rel.description}"
                if rel.description
                else f"({rel.source}, {rel.type}, {rel.target})"
            )
            src_type = entity_type_map.get(rel.source.strip().lower(), "")
            tgt_type = entity_type_map.get(rel.target.strip().lower(), "")
            props: dict[str, Any] = {
                "rel_type": rel.type,
                "fact": fact,
                "keywords": rel.keywords,
                "description": rel.description,
                "weight": rel.weight,
                "source_chunk_ids": rel.source_chunk_ids,
                "src_name": rel.source,
                "tgt_name": rel.target,
            }
            spans = getattr(rel, "spans", None)
            if spans:
                props["spans"] = spans
            relationships.append(
                GraphRelationship(
                    start_node_id=compute_entity_id(rel.source, src_type),
                    end_node_id=compute_entity_id(rel.target, tgt_type),
                    type="RELATES",
                    properties=props,
                )
            )
        return relationships
