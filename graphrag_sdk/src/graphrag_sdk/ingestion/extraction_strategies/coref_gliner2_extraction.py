# GraphRAG SDK 2.0 — Ingestion: CorefGLiNER2 Extraction (Zero-LLM)
# Coreference + GLiNER2 for BOTH entities AND relationships — no LLM needed.
#
# Architecture:
#   1. fastcoref resolves pronouns/short names → full entity names
#   2. GLiNER2 extracts entities + relations in a single forward pass
#      using schema-driven multi-task learning
#   3. Merge entities + relations → graph
#
# Key difference from CorefGLiNERLLMExtraction:
#   GLiNER2 handles BOTH entities AND relationships locally.
#   No LLM call is needed at any step → zero API cost, zero latency.
#
# Trade-offs vs CorefGLiNER+LLM:
#   ✅ Zero LLM cost — fully local inference
#   ✅ ~10x faster (~8s vs ~144s for 3 corpora)
#   ✅ Single model for entities + relations (GLiNER2 multi-task)
#   ✅ Relation types are schema-driven (you define what to extract)
#   ✅ Confidence scores for both entities and relations
#   ⚠ May miss very implicit relationships that LLM would catch
#   ⚠ Requires predefined relation types (not open-ended)
#
# Requirements:
#   pip install gliner2 fastcoref
#
# Usage::
#
#   extractor = CorefGLiNER2Extraction()
#   graph_data = await extractor.extract(chunks, schema, ctx)

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
    GraphData,
    GraphNode,
    GraphRelationship,
    GraphSchema,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_GLINER2_MODEL = "fastino/gliner2-base-v1"

_DEFAULT_ENTITY_TYPES = [
    "person",
    "location",
    "organization",
    "artifact",
    "event",
    "concept",
    "facility",
    "work of art",
]

_DEFAULT_RELATION_TYPES = {
    "married_to": "Marriage relationship between two people",
    "daughter_of": "Parent-child relationship where person is daughter of another",
    "son_of": "Parent-child relationship where person is son of another",
    "parent_of": "Parent-child relationship",
    "created": "Person who created, built, or made something",
    "constructed": "Person or entity who constructed or built something",
    "founded": "Person who founded or established an organization",
    "established": "Person who established or founded something",
    "keeper_of": "Person who maintains or manages a facility",
    "maintained": "Person who maintained or cared for something",
    "worked_at": "Person who worked at an organization or facility",
    "directed": "Person who directed or led an organization",
    "commanded": "Person who commanded a vessel or military unit",
    "wrecked_at": "Vessel or ship that was wrecked at a location",
    "collaborated_with": "Person who collaborated or worked together with another",
    "excavated": "Person who excavated or discovered an archaeological site",
    "located_in": "Entity located in a geographic location",
    "acquired": "Entity that acquired or purchased another entity",
    "authored": "Person who wrote or authored a work",
    "studied_at": "Person who studied at an institution",
}

_PRONOUNS = frozenset({
    "he", "she", "it", "they", "we", "i", "me", "him", "her", "us", "them",
    "his", "hers", "its", "their", "our", "my", "this", "that", "these",
    "those", "which", "who", "whom", "what",
})

_MIN_NAME_LEN = 2
_MAX_NAME_LEN = 80
_UNKNOWN_LABEL = "Unknown"


# ── ID helper ─────────────────────────────────────────────────────────────────

def _entity_id(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())


# ── Strategy ─────────────────────────────────────────────────────────────────

class CorefGLiNER2Extraction(ExtractionStrategy):
    """Zero-LLM extraction: CorefGLiNER2 for entities + relations.

    Flow::

        Text → fastcoref (resolve pronouns)
             → GLiNER2 multi-task extraction (entities + relations)
             → typed nodes + directional relationships → Graph

    This is a fully local, zero-cost extraction strategy.
    GLiNER2's 205M parameter model handles entities, classification,
    structured data extraction, and relation extraction in a single pass.

    Args:
        gliner2_model_name: HuggingFace model ID for GLiNER2.
        entity_types: Entity type labels for GLiNER2 NER.
        relation_types: Dict of relation_name → description for GLiNER2.
        enable_coref: Enable/disable coreference resolution.

    Example::

        extractor = CorefGLiNER2Extraction()
        graph_data = await extractor.extract(chunks, schema, ctx)
    """

    def __init__(
        self,
        gliner2_model_name: str = _DEFAULT_GLINER2_MODEL,
        entity_types: list[str] | dict[str, str] | None = None,
        relation_types: dict[str, str] | list[str] | None = None,
        enable_coref: bool = True,
    ) -> None:
        self.gliner2_model_name = gliner2_model_name
        self.enable_coref = enable_coref

        # Entity types — accept list or dict with descriptions
        if entity_types is None:
            self.entity_types = _DEFAULT_ENTITY_TYPES
        elif isinstance(entity_types, dict):
            self.entity_types = entity_types
        else:
            self.entity_types = [t.lower() for t in entity_types]

        # Relation types — accept dict with descriptions or plain list
        if relation_types is None:
            self.relation_types = _DEFAULT_RELATION_TYPES
        elif isinstance(relation_types, dict):
            self.relation_types = relation_types
        else:
            self.relation_types = {r: r.replace("_", " ") for r in relation_types}

        self._gliner2: Any | None = None
        self._coref: Any | None = None

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        if self._gliner2 is None:
            try:
                from gliner2 import GLiNER2
                logger.info(f"Loading GLiNER2 model: {self.gliner2_model_name}")
                self._gliner2 = GLiNER2.from_pretrained(self.gliner2_model_name)
            except ImportError as exc:
                raise ImportError(
                    "gliner2 is required. Install: pip install gliner2"
                ) from exc

        if self.enable_coref and self._coref is None:
            try:
                import transformers
                # LingMessCoref uses Longformer which doesn't support SDPA yet.
                _orig_sdpa = getattr(transformers.LongformerModel, "_sdpa_can_dispatch", None)
                try:
                    if _orig_sdpa is not None:
                        transformers.LongformerModel._sdpa_can_dispatch = (
                            lambda self, *a, **k: False
                        )
                    from fastcoref import LingMessCoref
                    logger.info("Loading LingMessCoref (full-accuracy coreference model)")
                    self._coref = LingMessCoref(device="cpu")
                finally:
                    if _orig_sdpa is not None:
                        transformers.LongformerModel._sdpa_can_dispatch = _orig_sdpa
            except ImportError as exc:
                raise ImportError(
                    "fastcoref is required for coreference. Install: pip install fastcoref"
                ) from exc

    # ── Coreference resolution ────────────────────────────────────────────────

    def _resolve_coreferences(self, text: str) -> str:
        if not self.enable_coref or self._coref is None:
            return text

        preds = self._coref.predict(texts=[text])
        clusters = preds[0].get_clusters(as_strings=True)

        if not clusters:
            return text

        resolved = text
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            canonical = max(cluster, key=len)
            for mention in cluster:
                if mention != canonical and mention.lower() in _PRONOUNS | {
                    "the " + w for w in _PRONOUNS
                }:
                    resolved = resolved.replace(mention, canonical)

        return resolved

    # ── GLiNER2 extraction (sync, runs in thread pool) ────────────────────────

    def _extract_chunk_sync(
        self,
        chunks: list[TextChunk],
    ) -> list[tuple[str, dict[str, list], dict[str, list]]]:
        """Run coref + GLiNER2 on each chunk.

        Returns (resolved_text, entities_dict, relations_dict) per chunk.
        """
        # Build schema once
        schema = self._gliner2.create_schema()

        # Entity types
        if isinstance(self.entity_types, dict):
            schema = schema.entities(self.entity_types)
        else:
            schema = schema.entities(self.entity_types)

        # Relation types
        schema = schema.relations(self.relation_types)

        results = []
        for chunk in chunks:
            text = chunk.text

            # Step 1: Coreference
            resolved_text = self._resolve_coreferences(text)

            # Step 2: GLiNER2 multi-task extraction
            extraction = self._gliner2.extract(resolved_text, schema)

            entities = extraction.get("entities", {})
            relations = extraction.get("relation_extraction", {})

            results.append((resolved_text, entities, relations))

        return results

    # ── Main extraction ───────────────────────────────────────────────────────

    async def extract(
        self,
        chunks: TextChunks,
        schema: GraphSchema,
        ctx: Context,
    ) -> GraphData:
        ctx.log(
            f"CorefGLiNER2: processing {len(chunks.chunks)} chunks "
            f"(coref={'on' if self.enable_coref else 'off'}, "
            f"model={self.gliner2_model_name})"
        )

        self._load_models()

        active_chunks: list[TextChunk] = []
        for chunk in chunks.chunks:
            if ctx.budget_exceeded:
                ctx.log("Latency budget exceeded — stopping extraction", logging.WARNING)
                break
            active_chunks.append(chunk)

        # Run extraction in thread pool (GLiNER2 + coref are CPU-bound)
        loop = asyncio.get_event_loop()
        extraction_results = await loop.run_in_executor(
            None,
            self._extract_chunk_sync,
            active_chunks,
        )

        # ── Aggregate ─────────────────────────────────────────────────────────
        all_entities: dict[str, ExtractedEntity] = {}
        all_relations: dict[tuple[str, str, str], ExtractedRelation] = {}
        all_mentions: list[EntityMention] = []

        for chunk, (_, entities, relations) in zip(active_chunks, extraction_results):
            # Process entities
            for etype, ent_list in entities.items():
                for ent_name in ent_list:
                    if not _is_valid_name(ent_name):
                        continue

                    eid = _entity_id(ent_name)
                    if eid not in all_entities:
                        all_entities[eid] = ExtractedEntity(
                            name=ent_name,
                            type=etype.lower(),
                            description="",
                            source_chunk_ids=[chunk.uid],
                        )
                    else:
                        existing = all_entities[eid]
                        if ent_name[0].isupper() and not existing.name[0].isupper():
                            existing.name = ent_name
                        if chunk.uid not in existing.source_chunk_ids:
                            existing.source_chunk_ids.append(chunk.uid)
                    all_mentions.append(EntityMention(chunk_id=chunk.uid, entity_id=eid))

            # Process relations
            for rel_type, rel_list in relations.items():
                for rel_tuple in rel_list:
                    # GLiNER2 returns tuples (source, target)
                    if isinstance(rel_tuple, (list, tuple)) and len(rel_tuple) >= 2:
                        source = str(rel_tuple[0]).strip()
                        target = str(rel_tuple[1]).strip()
                    else:
                        continue

                    if not _is_valid_name(source) or not _is_valid_name(target):
                        continue

                    rel_type_upper = rel_type.upper()

                    # Ensure both endpoints exist as entities
                    for name in (source, target):
                        eid = _entity_id(name)
                        if eid and eid not in all_entities:
                            all_entities[eid] = ExtractedEntity(
                                name=name,
                                type=_UNKNOWN_LABEL.lower(),
                                description="",
                                source_chunk_ids=[chunk.uid],
                            )
                            all_mentions.append(
                                EntityMention(chunk_id=chunk.uid, entity_id=eid)
                            )

                    key = (
                        _entity_id(source),
                        rel_type_upper,
                        _entity_id(target),
                    )
                    if key not in all_relations:
                        all_relations[key] = ExtractedRelation(
                            source=source,
                            target=target,
                            type=rel_type_upper,
                            keywords=rel_type_upper,
                            description=f"{source} {rel_type.replace('_', ' ')} {target}",
                            weight=1.0,
                            source_chunk_ids=[chunk.uid],
                        )
                    else:
                        if chunk.uid not in all_relations[key].source_chunk_ids:
                            all_relations[key].source_chunk_ids.append(chunk.uid)

        # ── Build GraphData ───────────────────────────────────────────────────
        nodes = [
            GraphNode(
                id=_entity_id(ent.name),
                label=_label_for_type(ent.type),
                properties={
                    "name": ent.name,
                    "entity_type": ent.type,
                    "source_chunk_ids": ent.source_chunk_ids,
                },
            )
            for ent in all_entities.values()
        ]

        relationships = [
            GraphRelationship(
                start_node_id=_entity_id(rel.source),
                end_node_id=_entity_id(rel.target),
                type="RELATES",
                properties={
                    "rel_type": rel.type,
                    "description": rel.description,
                    "keywords": rel.keywords,
                    "weight": rel.weight,
                    "fact": rel.description,
                    "source_chunk_ids": rel.source_chunk_ids,
                    "src_name": rel.source,
                    "tgt_name": rel.target,
                    "predicate": rel.keywords,
                    "type": rel.type,
                    "sentence": rel.description,
                },
            )
            for rel in all_relations.values()
        ]

        # Deduplicate mentions
        seen: set[tuple[str, str]] = set()
        unique_mentions: list[EntityMention] = []
        for m in all_mentions:
            key = (m.entity_id, m.chunk_id)
            if key not in seen:
                seen.add(key)
                unique_mentions.append(m)

        ctx.log(
            f"CorefGLiNER2: {len(nodes)} entities, {len(relationships)} rels, "
            f"{len(unique_mentions)} mentions (zero LLM)"
        )

        return GraphData(
            nodes=nodes,
            relationships=relationships,
            mentions=unique_mentions,
            extracted_entities=list(all_entities.values()),
            extracted_relations=list(all_relations.values()),
        )


# ── Validation helpers ────────────────────────────────────────────────────────

def _is_valid_name(name: str) -> bool:
    name = name.strip()
    if not name:
        return False
    if len(name) < _MIN_NAME_LEN or len(name) > _MAX_NAME_LEN:
        return False
    if name.lower() in _PRONOUNS:
        return False
    return True


def _label_for_type(gliner_type: str) -> str:
    if gliner_type.lower() == "unknown":
        return _UNKNOWN_LABEL
    return "".join(word.capitalize() for word in gliner_type.split())
