# GraphRAG SDK 2.0 — Ingestion: Merged Extraction (HippoRAG + LightRAG)
# Combines LightRAG-style rich typed extraction with HippoRAG-style
# fact triples and entity mentions.

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

import numpy as np

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    EntityMention,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionOutput,
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


def compute_entity_id(name: str, entity_type: str = "") -> str:
    """Deterministic entity ID from normalized name and optional type.

    When ``entity_type`` is provided, a ``__type`` suffix is appended to
    prevent cross-type collisions (e.g. Person "Paris" vs Location "Paris"
    produce different IDs: ``paris__person`` vs ``paris__location``).

    When ``entity_type`` is empty, returns just the normalized name for
    backwards compatibility.
    """
    base = name.strip().lower().replace(" ", "_")
    if entity_type:
        return f"{base}__{entity_type.strip().lower()}"
    return base


def _normalize_type_label(raw: str) -> str:
    """Normalize type string by lowercasing and removing separators.

    Collapses trivial formatting variants:
    "Data Type" / "DataType" / "data_type" / "data-type" → "datatype"
    """
    s = raw.strip().lower()
    return re.sub(r"[\s_\-/]+", "", s)


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
        type_resolution_threshold: Cosine similarity threshold for merging
            semantically similar type labels (0-1). Default 0.90 is tuned
            for ada-002 embeddings where short type labels have a high
            baseline similarity (~0.75-0.85). Set to 1.0 to disable
            embedding clustering (surface normalization still applies).
        consolidate_by_name: If True, entities with the same name but
            different types are consolidated to the dominant (most frequent)
            type. This eliminates duplicate nodes caused by LLM type
            inconsistency across chunks. Default True.
    """

    def __init__(
        self,
        llm: LLMInterface,
        embedder: Embedder | None = None,
        enable_gleaning: bool = False,
        max_concurrency: int | None = None,
        type_resolution_threshold: float = 0.90,
        consolidate_by_name: bool = True,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.enable_gleaning = enable_gleaning
        self._max_concurrency = max_concurrency  # None = use LLM's default
        self._type_resolution_threshold = type_resolution_threshold
        self._consolidate_by_name = consolidate_by_name

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
        all_mentions: list[EntityMention] = []

        # Filter chunks respecting budget
        active_chunks = []
        for chunk in chunks.chunks:
            if ctx.budget_exceeded:
                ctx.log("Latency budget exceeded, stopping extraction", logging.WARNING)
                break
            active_chunks.append(chunk)

        # ── Phase 1: Build all extraction prompts and batch-invoke ──
        prompts = [
            _MERGED_EXTRACTION_PROMPT.format(
                entity_types=entity_types,
                relation_types=relation_types,
                tuple_delim=TUPLE_DELIMITER,
                record_delim=RECORD_DELIMITER,
                text=chunk.text,
            )
            for chunk in active_chunks
        ]

        batch_kw: dict[str, Any] = {}
        if self._max_concurrency is not None:
            batch_kw["max_concurrency"] = self._max_concurrency

        phase1_results = await self.llm.abatch_invoke(prompts, **batch_kw)

        # Parse Phase 1 results
        # Parallel arrays: per-chunk entities/relations and raw response content
        chunk_entities: list[list[ExtractedEntity]] = []
        chunk_relations: list[list[ExtractedRelation]] = []
        phase1_contents: list[str] = []

        for item in phase1_results:
            chunk = active_chunks[item.index]
            if not item.ok:
                ctx.log(
                    f"Extraction failed for chunk {chunk.index}: {item.error}",
                    logging.WARNING,
                )
                chunk_entities.append([])
                chunk_relations.append([])
                phase1_contents.append("")
                continue

            entities, relations = self._parse_delimiter_response(
                item.response.content, chunk.uid
            )
            chunk_entities.append(entities)
            chunk_relations.append(relations)
            phase1_contents.append(item.response.content)

        # ── Phase 2: Gleaning (optional) ──
        if self.enable_gleaning:
            gleaning_indices: list[int] = []
            gleaning_prompts: list[str] = []
            for i, chunk in enumerate(active_chunks):
                if chunk_entities[i] or chunk_relations[i]:
                    gleaning_indices.append(i)
                    gleaning_prompts.append(
                        _GLEANING_PROMPT.format(
                            already_extracted=phase1_contents[i],
                            text=chunk.text,
                            tuple_delim=TUPLE_DELIMITER,
                            record_delim=RECORD_DELIMITER,
                        )
                    )

            if gleaning_prompts:
                phase2_results = await self.llm.abatch_invoke(
                    gleaning_prompts, **batch_kw
                )
                for item in phase2_results:
                    orig_idx = gleaning_indices[item.index]
                    chunk = active_chunks[orig_idx]
                    if not item.ok:
                        ctx.log(
                            f"Gleaning failed for chunk {chunk.index}: {item.error}",
                            logging.WARNING,
                        )
                        continue
                    extra_ents, extra_rels = self._parse_delimiter_response(
                        item.response.content, chunk.uid
                    )
                    chunk_entities[orig_idx].extend(extra_ents)
                    chunk_relations[orig_idx].extend(extra_rels)

        # ── Collect all entities and relations ──
        for ents in chunk_entities:
            all_entities.extend(ents)
        for rels in chunk_relations:
            all_relations.extend(rels)

        # ── Resolve type taxonomy (open-schema dedup) ──
        all_entities = await self._resolve_type_taxonomy(all_entities, ctx)

        # Generate HippoRAG-style mentions
        for ent in all_entities:
            ent_id = compute_entity_id(ent.name, ent.type)
            for chunk_id in ent.source_chunk_ids:
                all_mentions.append(
                    EntityMention(chunk_id=chunk_id, entity_id=ent_id)
                )

        # Aggregate across chunks: dedup entities by normalized name
        merged_entities = self._aggregate_entities(all_entities)
        merged_relations = self._aggregate_relations(all_relations)

        # Build entity type map for relationship endpoint resolution
        entity_type_map: dict[str, str] = {}
        for ent in merged_entities:
            entity_type_map[ent.name.strip().lower()] = ent.type

        # Convert to GraphData nodes + relationships
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

    # Quality thresholds for entity name validation
    _MIN_ENTITY_NAME_LEN = 2  # single-char names are noise
    _MAX_ENTITY_NAME_LEN = 80  # descriptions masquerading as names

    def _parse_delimiter_response(
        self,
        content: str,
        source_chunk_id: str,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        """Parse LightRAG-style delimiter format into entities and relations.

        Applies quality filtering:
        - Entities with empty/whitespace-only names or types are skipped.
        - Entity names shorter than 2 chars or longer than 80 chars are skipped.
        - Relations with empty source/target are skipped.
        """
        entities: list[ExtractedEntity] = []
        relations: list[ExtractedRelation] = []
        skipped = 0

        records = content.split(RECORD_DELIMITER)

        for record in records:
            record = record.strip()
            if not record:
                continue

            # Extract content between outermost parentheses
            stripped = record.strip()
            if stripped.startswith("(") and stripped.endswith(")"):
                inner = stripped[1:-1]
            else:
                match = re.search(r"\((.+)\)", record, re.DOTALL)
                if not match:
                    continue
                inner = match.group(1)

            parts = [p.strip().strip('"').strip("'") for p in inner.split(TUPLE_DELIMITER)]

            if len(parts) >= 4 and parts[0].lower() == "entity":
                name = parts[1].strip()
                etype = parts[2].strip()

                # Quality gate: reject empty, too-short, or too-long names
                if not name or not etype:
                    skipped += 1
                    continue
                if len(name) < self._MIN_ENTITY_NAME_LEN:
                    skipped += 1
                    continue
                if len(name) > self._MAX_ENTITY_NAME_LEN:
                    skipped += 1
                    continue

                entities.append(
                    ExtractedEntity(
                        name=name,
                        type=etype,
                        description=parts[3].strip(),
                        source_chunk_ids=[source_chunk_id],
                    )
                )
            elif len(parts) >= 6 and parts[0].lower() == "relationship":
                source = parts[1].strip()
                target = parts[2].strip()
                rel_type = parts[3].strip()

                # Quality gate: reject relations with empty endpoints or type
                if not source or not target or not rel_type:
                    skipped += 1
                    continue

                weight = 1.0
                if len(parts) >= 7:
                    try:
                        weight = float(parts[6])
                    except (ValueError, IndexError):
                        weight = 1.0
                relations.append(
                    ExtractedRelation(
                        source=source,
                        target=target,
                        type=rel_type,
                        keywords=parts[4].strip(),
                        description=parts[5].strip(),
                        weight=weight,
                        source_chunk_ids=[source_chunk_id],
                    )
                )

        if skipped:
            logger.debug(f"Skipped {skipped} low-quality records from chunk {source_chunk_id}")

        return entities, relations

    def _aggregate_entities(
        self,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        """Deduplicate entities by (normalized name, type), keeping longer descriptions."""
        seen: dict[tuple[str, str], ExtractedEntity] = {}
        for ent in entities:
            name_key = ent.name.strip().lower()
            if not name_key:
                continue  # skip empty names that slipped through
            key = (name_key, ent.type.strip().lower())
            if key in seen:
                existing = seen[key]
                # Prefer properly-capitalized name
                if ent.name[0].isupper() and not existing.name[0].isupper():
                    existing.name = ent.name
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

    async def _resolve_type_taxonomy(
        self,
        entities: list[ExtractedEntity],
        ctx: Context,
    ) -> list[ExtractedEntity]:
        """Resolve inconsistent type labels into canonical types.

        Two-phase approach:
        1. Surface normalization — collapse formatting variants (free, <1ms)
        2. Embedding clustering — merge semantically similar types (single API call)
        """
        # Collect raw type frequencies
        type_freq: dict[str, int] = defaultdict(int)
        for ent in entities:
            type_freq[ent.type] += 1

        unique_types = set(type_freq.keys())
        if len(unique_types) < 2:
            return entities

        # ── Phase 1: Surface normalization ──
        # Group raw types by their normalized form
        norm_groups: dict[str, list[str]] = defaultdict(list)
        for raw_type in unique_types:
            norm_groups[_normalize_type_label(raw_type)].append(raw_type)

        surface_remap: dict[str, str] = {}
        surface_merges = 0
        for _norm_key, variants in norm_groups.items():
            if len(variants) <= 1:
                continue
            # Pick the most frequent variant; on tie prefer Title Case
            canonical = max(
                variants,
                key=lambda v: (type_freq[v], v[0].isupper() if v else False, v),
            )
            for v in variants:
                if v != canonical:
                    surface_remap[v] = canonical
                    surface_merges += 1

        # Apply surface remapping
        if surface_remap:
            for ent in entities:
                if ent.type in surface_remap:
                    ent.type = surface_remap[ent.type]

        # ── Phase 2: Embedding clustering ──
        embedding_merges = 0
        if self.embedder is not None and self._type_resolution_threshold < 1.0:
            # Recompute unique types after surface pass
            type_freq_post: dict[str, int] = defaultdict(int)
            for ent in entities:
                type_freq_post[ent.type] += 1
            canonical_types = sorted(type_freq_post.keys())

            if len(canonical_types) >= 2:
                try:
                    embeddings = await self.embedder.aembed_documents(canonical_types)
                    emb_matrix = np.array(embeddings, dtype=np.float32)

                    # Normalize rows for cosine similarity via dot product
                    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    emb_matrix = emb_matrix / norms

                    sim_matrix = emb_matrix @ emb_matrix.T

                    # Union-Find clustering
                    n = len(canonical_types)
                    parent = list(range(n))

                    def find(x: int) -> int:
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x

                    def union(a: int, b: int) -> None:
                        ra, rb = find(a), find(b)
                        if ra != rb:
                            parent[ra] = rb

                    for i in range(n):
                        for j in range(i + 1, n):
                            if sim_matrix[i, j] >= self._type_resolution_threshold:
                                union(i, j)

                    # Group by cluster root
                    clusters: dict[int, list[int]] = defaultdict(list)
                    for i in range(n):
                        clusters[find(i)].append(i)

                    embed_remap: dict[str, str] = {}
                    for members in clusters.values():
                        if len(members) <= 1:
                            continue
                        # Pick most frequent type as canonical
                        cluster_canonical = max(
                            members,
                            key=lambda idx: (
                                type_freq_post[canonical_types[idx]],
                                canonical_types[idx][0].isupper() if canonical_types[idx] else False,
                                canonical_types[idx],
                            ),
                        )
                        canonical_label = canonical_types[cluster_canonical]
                        for idx in members:
                            if idx != cluster_canonical:
                                embed_remap[canonical_types[idx]] = canonical_label
                                embedding_merges += 1

                    if embed_remap:
                        for ent in entities:
                            if ent.type in embed_remap:
                                ent.type = embed_remap[ent.type]

                except Exception as e:
                    ctx.log(
                        f"Type taxonomy embedding clustering failed, "
                        f"using surface-only resolution: {e}",
                        logging.WARNING,
                    )

        # ── Phase 3: Name-based dominant-type consolidation ──
        name_merges = 0
        if self._consolidate_by_name:
            # Group entities by normalized name
            name_groups: dict[str, list[ExtractedEntity]] = defaultdict(list)
            for ent in entities:
                name_groups[ent.name.strip().lower()].append(ent)

            name_remap: dict[str, dict[str, str]] = {}  # {norm_name: {old_type: new_type}}
            for norm_name, group in name_groups.items():
                # Count type frequencies within this name group
                group_type_freq: dict[str, int] = defaultdict(int)
                for ent in group:
                    group_type_freq[ent.type] += 1

                if len(group_type_freq) < 2:
                    continue  # single type, nothing to consolidate

                # Pick the most frequent type; tie-break: prefer Title Case, then alphabetical
                dominant_type = max(
                    group_type_freq,
                    key=lambda t: (
                        group_type_freq[t],
                        t[0].isupper() if t else False,
                        t,
                    ),
                )

                for old_type in group_type_freq:
                    if old_type != dominant_type:
                        name_remap.setdefault(norm_name, {})[old_type] = dominant_type
                        name_merges += group_type_freq[old_type]

            # Apply name-based remapping
            if name_remap:
                for ent in entities:
                    norm_name = ent.name.strip().lower()
                    if norm_name in name_remap and ent.type in name_remap[norm_name]:
                        ent.type = name_remap[norm_name][ent.type]

        # Compute final type count
        final_types = len({ent.type for ent in entities})
        ctx.log(
            f"Type resolution: {len(unique_types)} raw types -> "
            f"{final_types} canonical "
            f"(surface={surface_merges}, embedding={embedding_merges}, "
            f"name={name_merges})"
        )

        return entities

    def _entities_to_nodes(
        self,
        entities: list[ExtractedEntity],
    ) -> list[GraphNode]:
        """Convert ExtractedEntity list to GraphNode list.

        Skips entities whose computed ID is empty (defensive check).
        """
        nodes: list[GraphNode] = []
        for ent in entities:
            node_id = compute_entity_id(ent.name, ent.type)
            if not node_id:
                continue
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
        entity_type_map: dict[str, str] | None = None,
    ) -> list[GraphRelationship]:
        """Convert ExtractedRelation list to GraphRelationship list.

        All LLM-extracted relationships use the single ``RELATES`` edge type.
        The original relationship type is preserved as the ``rel_type`` property,
        and a ``fact`` property stores a human-readable fact string for embedding.

        Args:
            relations: Extracted relations to convert.
            entity_type_map: Mapping from normalized entity name to type,
                used to compute type-qualified entity IDs for endpoints.
        """
        entity_type_map = entity_type_map or {}
        relationships: list[GraphRelationship] = []
        for rel in relations:
            fact = f"({rel.source}, {rel.type}, {rel.target}): {rel.description}" if rel.description else f"({rel.source}, {rel.type}, {rel.target})"
            src_type = entity_type_map.get(rel.source.strip().lower(), "")
            tgt_type = entity_type_map.get(rel.target.strip().lower(), "")
            relationships.append(
                GraphRelationship(
                    start_node_id=compute_entity_id(rel.source, src_type),
                    end_node_id=compute_entity_id(rel.target, tgt_type),
                    type="RELATES",
                    properties={
                        "rel_type": rel.type,
                        "fact": fact,
                        "keywords": rel.keywords,
                        "description": rel.description,
                        "weight": rel.weight,
                        "source_chunk_ids": rel.source_chunk_ids,
                        "src_name": rel.source,
                        "tgt_name": rel.target,
                    },
                )
            )
        return relationships
