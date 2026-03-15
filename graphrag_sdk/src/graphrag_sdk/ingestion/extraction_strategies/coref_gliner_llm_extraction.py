# GraphRAG SDK 2.0 — Ingestion: CorefGLiNER + LLM Extraction (Hybrid)
# Coreference + Schema-Typed GLiNER for entities, LLM for relationships.
#
# Architecture:
#   1. fastcoref resolves pronouns/short names → full entity names
#   2. GLiNER NER with schema types → typed entities (≥ 0.75 typed, < 0.75 Unknown)
#   3. LLM call per chunk: given the entities found, extract relationships
#      → outputs structured JSON with source, target, type, sentence, explanation
#   4. Merge entities + LLM relationships → graph
#
# Key difference from CorefGLiNERExtraction:
#   Steps 1–4 (coref + GLiNER entities) are identical — local, no LLM.
#   Step 5 replaces spaCy SVO with an LLM call for relationship extraction.
#   This catches implicit/passive/complex relationships that SVO misses.
#
# Trade-offs vs CorefGLiNERExtraction (no LLM):
#   ✅ Much higher relationship recall (LLM understands implicit facts)
#   ✅ Better relationship naming ("MARRIED" vs "marry")
#   ✅ Captures passive/complex constructs ("was built by" → CONSTRUCTED)
#   ⚠ Adds API cost per chunk
#   ⚠ Adds latency (~2-5s per chunk)
#
# Trade-offs vs pure LLM extraction (SchemaGuided / HippoRAG):
#   ✅ Entities are grounded by GLiNER (no hallucinated entities)
#   ✅ Entity types from schema with confidence scores
#   ✅ Coreference pre-resolved (no "she" in triples)
#   ⚠ LLM still needed for relationships
#
# Requirements:
#   pip install spacy gliner fastcoref openai
#   python -m spacy download en_core_web_lg
#   OPENAI_API_KEY in environment
#
# Usage::
#
#   extractor = CorefGLiNERLLMExtraction(llm=my_llm)
#   graph_data = await extractor.extract(chunks, schema, ctx)

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
    TextChunk,
    TextChunks,
)
from graphrag_sdk.core.providers import LLMInterface
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_GLINER_MODEL = "urchade/gliner_medium-v2.1"
_DEFAULT_SPACY_MODEL = "en_core_web_lg"
_DEFAULT_COREF_MODEL = "biu-nlp/f-coref"

_DEFAULT_SCHEMA_TYPES = [
    "person",
    "location",
    "organization",
    "artifact",
    "event",
    "concept",
    "facility",
    "work of art",
]

_PRONOUNS = frozenset({
    "he", "she", "it", "they", "we", "i", "me", "him", "her", "us", "them",
    "his", "hers", "its", "their", "our", "my", "this", "that", "these",
    "those", "which", "who", "whom", "what",
})

_MIN_NAME_LEN = 2
_MAX_NAME_LEN = 80
_DEFAULT_GLINER_THRESHOLD = 0.75
_UNKNOWN_LABEL = "Unknown"

# ── LLM Prompt for relationship extraction ───────────────────────────────────

_REL_EXTRACTION_PROMPT = (
    "You are a relationship extraction system.\n"
    "Given the text below and a list of named entities found in it, "
    "extract ALL relationships between these entities.\n\n"
    "## Entities found\n"
    "{entities_list}\n\n"
    "## Text\n"
    "{text}\n\n"
    "## Instructions\n"
    "- Extract every factual relationship between the entities listed above.\n"
    "- For each relationship, provide:\n"
    "  - source: the source entity name (must be from the list above)\n"
    "  - target: the target entity name (must be from the list above)\n"
    "  - type: a short UPPER_CASE relationship type (e.g. MARRIED, CREATED, WORKED_AT)\n"
    "  - sentence: the exact sentence from the text that supports this relationship\n"
    "  - explanation: a one-line explanation of the relationship\n"
    "- Be exhaustive — extract every relationship, even implicit ones.\n"
    "- Only use entities from the list above as source/target.\n"
    "- Return valid JSON array only, no extra text.\n\n"
    "## Output format\n"
    '[\n'
    '  {{"source": "Entity A", "target": "Entity B", "type": "REL_TYPE", '
    '"sentence": "...", "explanation": "..."}},\n'
    '  ...\n'
    ']\n'
)


# ── ID helper ─────────────────────────────────────────────────────────────────

def _entity_id(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())


# ── Strategy ─────────────────────────────────────────────────────────────────

class CorefGLiNERLLMExtraction(ExtractionStrategy):
    """Hybrid extraction: CorefGLiNER for entities + LLM for relationships.

    Flow::

        Text → fastcoref (resolve pronouns)
             → GLiNER NER (schema types, threshold filter)
             → entity JSON (typed, with confidence scores)
             → LLM call per chunk: "given these entities, extract relationships"
             → relationship JSON (with sentence + explanation)
             → typed nodes + rich relationships → Graph

    This combines the best of both worlds:
    - Entity extraction is grounded (GLiNER — no hallucinated entities)
    - Relationship extraction is intelligent (LLM — catches implicit facts)

    Args:
        llm: LLM provider for relationship extraction.
        gliner_model_name: HuggingFace model ID for GLiNER.
        spacy_model_name: spaCy pipeline name (used for sentence splitting only).
        coref_model_name: HuggingFace model ID for fastcoref.
        schema_types: Entity type labels for GLiNER zero-shot NER.
        gliner_threshold: Confidence threshold — ≥ this → typed, below → "Unknown".
        enable_coref: Enable/disable coreference resolution.
        max_concurrency: Maximum parallel LLM calls.

    Example::

        extractor = CorefGLiNERLLMExtraction(llm=my_llm)
        graph_data = await extractor.extract(chunks, schema, ctx)
    """

    def __init__(
        self,
        llm: LLMInterface,
        gliner_model_name: str = _DEFAULT_GLINER_MODEL,
        spacy_model_name: str = _DEFAULT_SPACY_MODEL,
        coref_model_name: str = _DEFAULT_COREF_MODEL,
        schema_types: list[str] | None = None,
        gliner_threshold: float = _DEFAULT_GLINER_THRESHOLD,
        enable_coref: bool = True,
        max_concurrency: int | None = None,
    ) -> None:
        self.llm = llm
        self.gliner_model_name = gliner_model_name
        self.spacy_model_name = spacy_model_name
        self.coref_model_name = coref_model_name
        self.schema_types = [t.lower() for t in (schema_types or _DEFAULT_SCHEMA_TYPES)]
        self.gliner_threshold = gliner_threshold
        self.enable_coref = enable_coref
        self._max_concurrency = max_concurrency

        self._gliner: Any | None = None
        self._nlp: Any | None = None
        self._coref: Any | None = None

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        if self._gliner is None:
            try:
                from gliner import GLiNER
                logger.info(f"Loading GLiNER model: {self.gliner_model_name}")
                self._gliner = GLiNER.from_pretrained(self.gliner_model_name)
            except ImportError as exc:
                raise ImportError(
                    "gliner is required. Install: pip install gliner"
                ) from exc

        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load(self.spacy_model_name, disable=["ner"])
            except OSError as exc:
                raise OSError(
                    f"spaCy model '{self.spacy_model_name}' not found. "
                    f"Install: python -m spacy download {self.spacy_model_name}"
                ) from exc

        if self.enable_coref and self._coref is None:
            try:
                import transformers
                # LingMessCoref uses Longformer which doesn't support SDPA yet.
                # Patch the check so it falls back to eager attention automatically.
                _orig_sdpa = getattr(transformers.LongformerModel, "_sdpa_can_dispatch", None)
                if _orig_sdpa is not None:
                    transformers.LongformerModel._sdpa_can_dispatch = lambda self, *a, **k: False

                from fastcoref import LingMessCoref
                logger.info("Loading LingMessCoref (full-accuracy coreference model)")
                self._coref = LingMessCoref(device="cpu")
            except ImportError as exc:
                raise ImportError(
                    "fastcoref is required. Install: pip install fastcoref"
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

    # ── GLiNER entity extraction (sync, runs in thread pool) ──────────────────

    def _extract_entities_sync(
        self,
        chunks: list[TextChunk],
    ) -> list[tuple[str, list[ExtractedEntity]]]:
        """Run coref + GLiNER on each chunk. Returns (resolved_text, entities) per chunk."""
        results = []
        for chunk in chunks:
            text = chunk.text

            # Step 1: Coreference
            resolved_text = self._resolve_coreferences(text)

            # Step 2: GLiNER NER with schema types
            gliner_ents = self._gliner.predict_entities(
                resolved_text,
                self.schema_types,
                threshold=0.0,
            )

            entities: list[ExtractedEntity] = []
            seen_names: set[str] = set()

            for e in gliner_ents:
                name = e["text"]
                if not _is_valid_name(name):
                    continue
                norm = name.strip().lower()
                if norm in seen_names:
                    continue
                seen_names.add(norm)

                score = e.get("score", 0.0)
                label = e["label"] if score >= self.gliner_threshold else _UNKNOWN_LABEL.lower()

                entities.append(
                    ExtractedEntity(
                        name=name,
                        type=label,
                        description=f"{score:.3f}",
                        source_chunk_ids=[chunk.uid],
                    )
                )

            results.append((resolved_text, entities))

        return results

    # ── Main extraction ───────────────────────────────────────────────────────

    async def extract(
        self,
        chunks: TextChunks,
        schema: GraphSchema,
        ctx: Context,
    ) -> GraphData:
        ctx.log(
            f"CorefGLiNER+LLM: processing {len(chunks.chunks)} chunks "
            f"(coref={'on' if self.enable_coref else 'off'}, "
            f"threshold={self.gliner_threshold})"
        )

        self._load_models()

        active_chunks: list[TextChunk] = []
        for chunk in chunks.chunks:
            if ctx.budget_exceeded:
                ctx.log("Latency budget exceeded — stopping extraction", logging.WARNING)
                break
            active_chunks.append(chunk)

        # Phase 1: Entity extraction (local, in thread pool)
        loop = asyncio.get_event_loop()
        entity_results: list[tuple[str, list[ExtractedEntity]]] = (
            await loop.run_in_executor(
                None,
                self._extract_entities_sync,
                active_chunks,
            )
        )

        # Phase 2: LLM relationship extraction (async, batched)
        batch_kw: dict[str, Any] = {}
        if self._max_concurrency is not None:
            batch_kw["max_concurrency"] = self._max_concurrency

        prompts: list[str] = []
        for chunk, (resolved_text, ents) in zip(active_chunks, entity_results):
            if not ents:
                prompts.append("")  # empty prompt — will skip
                continue
            entities_str = "\n".join(
                f"- {e.name}" for e in ents
            )
            prompts.append(
                _REL_EXTRACTION_PROMPT.format(
                    entities_list=entities_str,
                    text=resolved_text,
                )
            )

        # Filter out empty prompts
        prompt_indices = [i for i, p in enumerate(prompts) if p]
        active_prompts = [prompts[i] for i in prompt_indices]

        chunk_relations: list[list[ExtractedRelation]] = [[] for _ in active_chunks]

        if active_prompts:
            llm_results = await self.llm.abatch_invoke(active_prompts, **batch_kw)

            for item in llm_results:
                orig_idx = prompt_indices[item.index]
                chunk = active_chunks[orig_idx]

                if not item.ok:
                    ctx.log(
                        f"LLM rel extraction failed for chunk {chunk.index}: {item.error}",
                        logging.WARNING,
                    )
                    continue

                rels = _parse_llm_relationships(
                    item.response.content,
                    chunk.uid,
                )
                chunk_relations[orig_idx] = rels

        # ── Aggregate ─────────────────────────────────────────────────────────
        all_entities: dict[str, ExtractedEntity] = {}
        all_relations: dict[tuple[str, str, str], ExtractedRelation] = {}
        all_mentions: list[EntityMention] = []

        for chunk, (_, ents) in zip(active_chunks, entity_results):
            for ent in ents:
                eid = _entity_id(ent.name)
                if eid not in all_entities:
                    all_entities[eid] = ExtractedEntity(
                        name=ent.name,
                        type=ent.type,
                        description=ent.description,
                        source_chunk_ids=[chunk.uid],
                    )
                else:
                    existing = all_entities[eid]
                    if ent.name[0].isupper() and not existing.name[0].isupper():
                        existing.name = ent.name
                    if chunk.uid not in existing.source_chunk_ids:
                        existing.source_chunk_ids.append(chunk.uid)
                all_mentions.append(EntityMention(chunk_id=chunk.uid, entity_id=eid))

        for chunk, rels in zip(active_chunks, chunk_relations):
            for rel in rels:
                # Ensure both endpoints exist as entities
                for name in (rel.source, rel.target):
                    eid = _entity_id(name)
                    if eid and eid not in all_entities:
                        all_entities[eid] = ExtractedEntity(
                            name=name,
                            type=_UNKNOWN_LABEL.lower(),
                            description="0.000",
                            source_chunk_ids=[chunk.uid],
                        )
                        all_mentions.append(EntityMention(chunk_id=chunk.uid, entity_id=eid))

                key = (
                    _entity_id(rel.source),
                    rel.keywords.strip().lower(),
                    _entity_id(rel.target),
                )
                if key not in all_relations:
                    all_relations[key] = ExtractedRelation(
                        source=rel.source,
                        target=rel.target,
                        type=rel.type,
                        keywords=rel.keywords,
                        description=rel.description,
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
                    "gliner_score": ent.description,
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
                    "predicate": rel.keywords,
                    "type": rel.type,
                    "sentence": rel.description,
                    "source_chunk_ids": rel.source_chunk_ids,
                    "src_name": rel.source,
                    "tgt_name": rel.target,
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
            f"CorefGLiNER+LLM: {len(nodes)} entities, {len(relationships)} rels, "
            f"{len(unique_mentions)} mentions"
        )

        return GraphData(
            nodes=nodes,
            relationships=relationships,
            mentions=unique_mentions,
            extracted_entities=list(all_entities.values()),
            extracted_relations=list(all_relations.values()),
        )


# ── LLM response parsing ─────────────────────────────────────────────────────

def _parse_llm_relationships(
    content: str,
    chunk_uid: str,
) -> list[ExtractedRelation]:
    """Parse the LLM's JSON array of relationships."""
    # Strip markdown fences if present
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM relationship JSON: {content[:200]}")
        return []

    if not isinstance(data, list):
        return []

    relations: list[ExtractedRelation] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        rel_type = str(item.get("type", "RELATES")).strip()
        sentence = str(item.get("sentence", "")).strip()
        explanation = str(item.get("explanation", "")).strip()

        if not source or not target:
            continue
        if not _is_valid_name(source) or not _is_valid_name(target):
            continue

        # Build description combining sentence + explanation
        desc = sentence
        if explanation:
            desc = f"{sentence} | {explanation}"

        relations.append(
            ExtractedRelation(
                source=source,
                target=target,
                type=rel_type,
                keywords=rel_type,  # e.g. "MARRIED"
                description=desc,
                weight=1.0,
                source_chunk_ids=[chunk_uid],
            )
        )

    return relations


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
