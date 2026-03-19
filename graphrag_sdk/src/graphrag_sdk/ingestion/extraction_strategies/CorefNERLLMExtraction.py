# GraphRAG SDK 2.0 — Ingestion: Coref + NER + LLM Verify & Relate
#
# Architecture:
#   1. fastcoref resolves pronouns/short names → full entity names
#   2. NER model extracts entities only (single forward pass)
#      Supported backends: "gliner2" (default), "gliner", "spacy"
#   3. Single LLM call per chunk:
#        - VERIFY + CLEAN the NER entities (merge aliases, fix types)
#        - EXTRACT all relationships from text using verified entities
#   4. Merge verified entities + LLM relationships → graph
#
# Trade-offs:
#   ✅ NER seeds entities locally (grounded, no hallucination)
#   ✅ LLM verifies/cleans entities (merges aliases like "Dr. Voss" → "Eleanor Voss")
#   ✅ LLM extracts relationships (catches implicit ones NER misses)
#   ✅ Single LLM call per chunk (cheaper than 2-step)
#   ✅ Swappable NER backend — default gliner2, also supports gliner, spacy
#   ⚠ Adds API cost per chunk
#   ⚠ Slightly slower than zero-LLM
#
# Requirements:
#   pip install fastcoref
#   pip install gliner2          # for ner_backend="gliner2" (default)
#   pip install gliner           # for ner_backend="gliner"
#   pip install spacy            # for ner_backend="spacy"
#   LLM provider (e.g. OpenAI API key)
#
# Usage::
#
#   # Default (gliner2)
#   extractor = CorefNERLLMExtraction(llm=my_llm)
#
#   # GLiNER v1
#   extractor = CorefNERLLMExtraction(llm=my_llm, ner_backend="gliner")
#
#   # spaCy
#   extractor = CorefNERLLMExtraction(llm=my_llm, ner_backend="spacy",
#                                      ner_model_name="en_core_web_trf")
#
#   graph_data = await extractor.extract(chunks, schema, ctx)

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Literal

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

NERBackend = Literal["gliner2", "gliner", "spacy"]

_DEFAULT_NER_BACKEND: NERBackend = "gliner2"

_DEFAULT_MODELS: dict[str, str] = {
    "gliner2": "fastino/gliner2-base-v1",
    "gliner":  "urchade/gliner_mediumv2.1",
    "spacy":   "en_core_web_trf",
}

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

_PRONOUNS = frozenset({
    "he", "she", "it", "they", "we", "i", "me", "him", "her", "us", "them",
    "his", "hers", "its", "their", "our", "my", "this", "that", "these",
    "those", "which", "who", "whom", "what",
})

_MIN_NAME_LEN = 2
_MAX_NAME_LEN = 80
_UNKNOWN_LABEL = "Unknown"

# ── LLM Verify + Relate Prompt ────────────────────────────────────────────────

_VERIFY_AND_RELATE_PROMPT = (
    "You are a knowledge graph quality assurance system.\n"
    "A local NER model extracted entities from the text below.\n"
    "Your job is to VERIFY, CLEAN, and then EXTRACT all relationships.\n\n"
    "## Original Text\n"
    "{text}\n\n"
    "## Extracted Entities\n"
    "{entities_json}\n\n"
    "## Instructions\n"
    "1. VERIFY: Remove any entities that are wrong, nonsensical, or common nouns.\n"
    "2. CLEAN: Merge duplicates that refer to the same real-world thing\n"
    "   (e.g. 'Dr. Voss' and 'Eleanor Voss' → one entity with the full name).\n"
    "   Fix entity types if wrong.\n"
    "3. EXTRACT: Find ALL relationships clearly stated in the text between\n"
    "   the verified entities. Be exhaustive. Use short UPPER_CASE types\n"
    "   (e.g. MARRIED_TO, FOUNDED, WORKED_AT, COMMANDED, DAUGHTER_OF).\n"
    "   Source and target MUST exactly match a verified entity name.\n\n"
    "Return valid JSON only, no extra text:\n"
    '{{\n'
    '  "entities": [\n'
    '    {{"name": "...", "type": "...", "description": "..."}},\n'
    '    ...\n'
    '  ],\n'
    '  "relationships": [\n'
    '    {{"source": "...", "target": "...", "type": "...", "description": "..."}},\n'
    '    ...\n'
    '  ]\n'
    '}}\n'
)


# ── ID helper ──────────────────────────────────────────────────────────────────

def _entity_id(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())


# ── Strategy ──────────────────────────────────────────────────────────────────

class CorefNERLLMExtraction(ExtractionStrategy):
    """Hybrid extraction: Coref + swappable NER backend + LLM verify & relate.

    Flow::

        Text → fastcoref (resolve pronouns)
             → NER backend (entities only, single forward pass)
             → LLM call per chunk:
                 - VERIFY + CLEAN entities (merge aliases, fix types)
                 - EXTRACT all relationships from verified entities
             → verified nodes + discovered relationships → Graph

    Args:
        llm: LLM provider for verify + relationship extraction.
        ner_backend: NER backend to use. One of "gliner2" (default), "gliner", "spacy".
        ner_model_name: HuggingFace / spaCy model ID. Defaults to backend's recommended model.
        entity_types: Entity type labels for NER. Ignored for spaCy (uses built-in types).
        enable_coref: Enable/disable fastcoref coreference resolution.
        max_concurrency: Maximum parallel LLM calls.

    Example::

        # Default — GLiNER2
        extractor = CorefNERLLMExtraction(llm=my_llm)

        # GLiNER v1
        extractor = CorefNERLLMExtraction(llm=my_llm, ner_backend="gliner")

        # spaCy
        extractor = CorefNERLLMExtraction(llm=my_llm, ner_backend="spacy")

        graph_data = await extractor.extract(chunks, schema, ctx)
    """

    def __init__(
        self,
        llm: LLMInterface,
        ner_backend: NERBackend = _DEFAULT_NER_BACKEND,
        ner_model_name: str | None = None,
        entity_types: list[str] | None = None,
        enable_coref: bool = True,
        max_concurrency: int | None = None,
    ) -> None:
        self.llm = llm
        self.ner_backend = ner_backend
        self.ner_model_name = ner_model_name or _DEFAULT_MODELS[ner_backend]
        self.enable_coref = enable_coref
        self._max_concurrency = max_concurrency
        self.entity_types = entity_types if entity_types is not None else _DEFAULT_ENTITY_TYPES

        self._ner: Any | None = None
        self._coref: Any | None = None

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        if self._ner is None:
            self._ner = self._load_ner()

        if self.enable_coref and self._coref is None:
            try:
                import transformers
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

    def _load_ner(self) -> Any:
        if self.ner_backend == "gliner2":
            try:
                from gliner2 import GLiNER2
                logger.info(f"Loading GLiNER2 model: {self.ner_model_name}")
                return GLiNER2.from_pretrained(self.ner_model_name)
            except ImportError as exc:
                raise ImportError(
                    "gliner2 is required. Install: pip install gliner2"
                ) from exc

        elif self.ner_backend == "gliner":
            try:
                from gliner import GLiNER
                logger.info(f"Loading GLiNER model: {self.ner_model_name}")
                return GLiNER.from_pretrained(self.ner_model_name)
            except ImportError as exc:
                raise ImportError(
                    "gliner is required. Install: pip install gliner"
                ) from exc

        elif self.ner_backend == "spacy":
            try:
                import spacy
                logger.info(f"Loading spaCy model: {self.ner_model_name}")
                return spacy.load(self.ner_model_name)
            except ImportError as exc:
                raise ImportError(
                    "spacy is required. Install: pip install spacy"
                ) from exc
            except OSError as exc:
                raise OSError(
                    f"spaCy model '{self.ner_model_name}' not found. "
                    f"Install: python -m spacy download {self.ner_model_name}"
                ) from exc

        else:
            raise ValueError(
                f"Unknown ner_backend '{self.ner_backend}'. "
                f"Choose from: 'gliner2', 'gliner', 'spacy'."
            )

    # ── Coreference resolution ─────────────────────────────────────────────────

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

    # ── NER extraction (sync, runs in thread pool) ─────────────────────────────

    def _extract_ner_sync(
        self,
        chunks: list[TextChunk],
    ) -> list[tuple[str, dict[str, list]]]:
        """Run coref + NER on each chunk.

        Returns list of (resolved_text, entities_dict) per chunk.
        entities_dict format: {entity_type: [name, ...]}
        """
        results = []
        for chunk in chunks:
            resolved_text = self._resolve_coreferences(chunk.text)
            entities = self._run_ner(resolved_text)
            results.append((resolved_text, entities))
        return results

    def _run_ner(self, text: str) -> dict[str, list]:
        """Run NER on text, return {entity_type: [name, ...]}."""
        if self.ner_backend == "gliner2":
            schema = self._ner.create_schema()
            schema = schema.entities(self.entity_types)
            extraction = self._ner.extract(text, schema)
            return extraction.get("entities", {})

        elif self.ner_backend == "gliner":
            raw = self._ner.predict_entities(text, self.entity_types)
            entities: dict[str, list] = {}
            for ent in raw:
                label = ent.get("label", "unknown")
                name = ent.get("text", "").strip()
                if name:
                    entities.setdefault(label, []).append(name)
            return entities

        elif self.ner_backend == "spacy":
            doc = self._ner(text)
            entities: dict[str, list] = {}
            for ent in doc.ents:
                label = ent.label_.lower()
                entities.setdefault(label, []).append(ent.text.strip())
            return entities

        return {}

    # ── Main extraction ────────────────────────────────────────────────────────

    async def extract(
        self,
        chunks: TextChunks,
        schema: GraphSchema,
        ctx: Context,
    ) -> GraphData:
        ctx.log(
            f"CorefNER+LLM: processing {len(chunks.chunks)} chunks "
            f"(ner_backend={self.ner_backend}, model={self.ner_model_name}, "
            f"coref={'on' if self.enable_coref else 'off'})"
        )

        self._load_models()

        active_chunks: list[TextChunk] = []
        for chunk in chunks.chunks:
            if ctx.budget_exceeded:
                ctx.log("Latency budget exceeded — stopping extraction", logging.WARNING)
                break
            active_chunks.append(chunk)

        # ── Phase 1: NER — entities only (local, thread pool) ─────────────────
        loop = asyncio.get_event_loop()
        ner_results = await loop.run_in_executor(
            None,
            self._extract_ner_sync,
            active_chunks,
        )

        # ── Phase 2: LLM verify+clean entities + extract relationships ─────────
        batch_kw: dict[str, Any] = {}
        if self._max_concurrency is not None:
            batch_kw["max_concurrency"] = self._max_concurrency

        prompts: list[str] = []
        for chunk, (resolved_text, entities) in zip(active_chunks, ner_results):
            ent_list = []
            for etype, names in entities.items():
                for name in names:
                    if _is_valid_name(name):
                        ent_list.append({"name": name, "type": etype})

            if not ent_list:
                prompts.append("")
                continue

            prompts.append(
                _VERIFY_AND_RELATE_PROMPT.format(
                    text=resolved_text,
                    entities_json=json.dumps(ent_list, ensure_ascii=False),
                )
            )

        prompt_indices = [i for i, p in enumerate(prompts) if p]
        active_prompts = [prompts[i] for i in prompt_indices]

        llm_data: list[dict] = [{} for _ in active_chunks]

        if active_prompts:
            llm_results = await self.llm.abatch_invoke(active_prompts, **batch_kw)

            for item in llm_results:
                orig_idx = prompt_indices[item.index]
                chunk = active_chunks[orig_idx]

                if not item.ok:
                    ctx.log(
                        f"LLM call failed for chunk {chunk.index}: {item.error}",
                        logging.WARNING,
                    )
                    continue

                parsed = _parse_llm_response(item.response.content, chunk.uid)
                llm_data[orig_idx] = parsed

        # ── Aggregate ──────────────────────────────────────────────────────────
        all_entities: dict[str, ExtractedEntity] = {}
        all_relations: dict[tuple[str, str, str], ExtractedRelation] = {}
        all_mentions: list[EntityMention] = []

        for chunk_idx, chunk in enumerate(active_chunks):
            data = llm_data[chunk_idx]
            _, raw_ner_entities = ner_results[chunk_idx]

            verified_entities = data.get("entities", [])
            if not verified_entities:
                # Fallback: use raw NER entities
                for etype, names in raw_ner_entities.items():
                    for name in names:
                        if _is_valid_name(name):
                            verified_entities.append({"name": name, "type": etype, "description": ""})

            for ent in verified_entities:
                name = str(ent.get("name", "")).strip()
                etype = str(ent.get("type", "unknown")).strip().lower()
                desc = str(ent.get("description", "")).strip()

                if not _is_valid_name(name):
                    continue

                eid = _entity_id(name)
                if eid not in all_entities:
                    all_entities[eid] = ExtractedEntity(
                        name=name,
                        type=etype,
                        description=desc,
                        source_chunk_ids=[chunk.uid],
                    )
                else:
                    existing = all_entities[eid]
                    if name[0].isupper() and not existing.name[0].isupper():
                        existing.name = name
                    if desc and not existing.description:
                        existing.description = desc
                    if chunk.uid not in existing.source_chunk_ids:
                        existing.source_chunk_ids.append(chunk.uid)

                all_mentions.append(EntityMention(chunk_id=chunk.uid, entity_id=eid))

            for rel in data.get("relationships", []):
                source = str(rel.get("source", "")).strip()
                target = str(rel.get("target", "")).strip()
                rel_type = str(rel.get("type", "RELATES")).strip().upper()
                desc = str(rel.get("description", "")).strip()

                if not _is_valid_name(source) or not _is_valid_name(target):
                    continue
                if source == target:
                    continue

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

                if not desc:
                    desc = f"{source} {rel_type.replace('_', ' ').lower()} {target}"

                key = (_entity_id(source), rel_type, _entity_id(target))
                if key not in all_relations:
                    all_relations[key] = ExtractedRelation(
                        source=source,
                        target=target,
                        type=rel_type,
                        keywords=rel_type,
                        description=desc,
                        weight=1.0,
                        source_chunk_ids=[chunk.uid],
                    )
                else:
                    existing_rel = all_relations[key]
                    if desc and not existing_rel.description:
                        existing_rel.description = desc
                    if chunk.uid not in existing_rel.source_chunk_ids:
                        existing_rel.source_chunk_ids.append(chunk.uid)

        # ── Build GraphData ────────────────────────────────────────────────────
        nodes = [
            GraphNode(
                id=_entity_id(ent.name),
                label=_label_for_type(ent.type),
                properties={
                    "name": ent.name,
                    "entity_type": ent.type,
                    "description": ent.description,
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
                },
            )
            for rel in all_relations.values()
        ]

        seen: set[tuple[str, str]] = set()
        unique_mentions: list[EntityMention] = []
        for m in all_mentions:
            mkey = (m.entity_id, m.chunk_id)
            if mkey not in seen:
                seen.add(mkey)
                unique_mentions.append(m)

        ctx.log(
            f"CorefNER+LLM: {len(nodes)} entities, {len(relationships)} rels, "
            f"{len(unique_mentions)} mentions "
            f"(backend={self.ner_backend}, model={self.ner_model_name})"
        )

        return GraphData(
            nodes=nodes,
            relationships=relationships,
            mentions=unique_mentions,
            extracted_entities=list(all_entities.values()),
            extracted_relations=list(all_relations.values()),
        )


# ── LLM response parsing ──────────────────────────────────────────────────────

def _parse_llm_response(content: str, chunk_uid: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse LLM response for chunk %s: %s (response_length=%d)",
            chunk_uid,
            exc,
            len(content),
        )
        return {}

    if not isinstance(data, dict):
        return {}

    return data


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


def _label_for_type(ner_type: str) -> str:
    if ner_type.lower() == "unknown":
        return _UNKNOWN_LABEL
    return "".join(word.capitalize() for word in ner_type.split())
