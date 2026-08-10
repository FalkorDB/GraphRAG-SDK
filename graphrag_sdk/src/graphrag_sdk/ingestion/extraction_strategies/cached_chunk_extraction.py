# GraphRAG SDK — Ingestion: Chunk-Level Extraction Cache
# Pattern: Decorator — wraps any ExtractionStrategy with a graph-backed cache.

"""Chunk-level extraction cache for document updates.

Wraps any :class:`ExtractionStrategy` so that chunks whose text is
byte-identical to a chunk already stored for the same document skip LLM
extraction entirely: their entities, relationships, and mentions are rebuilt
from the live graph and remapped onto the new chunk uid. Only genuinely
new/changed chunks are passed to the inner extractor.

The graph itself is the cache — nothing is stored anywhere new. Reads go
through :class:`~graphrag_sdk.storage.graph_store.GraphStore` accessors
(``get_document_chunk_texts``, ``get_entities_mentioned_in_chunks``,
``get_relationships_for_chunks``), so the persistence schema stays owned by
the storage layer.

Safe by construction:

- Entity ids are deterministic, so cached nodes MERGE onto the existing
  entities (``SET n += props`` preserves embeddings and enrichment).
- Entity ``source_chunk_ids`` are re-emitted with this document's old chunk
  ids remapped to the new chunk uids (old ids die in the cutover; other
  documents' ids pass through), so citation resolution keeps working.
- Relationship upserts UNION ``source_chunk_ids`` (see
  ``GraphStore.upsert_relationships``); the ``update()`` stale-edge cleanup
  then strips the old chunk ids.
- Mentions are re-emitted against the new chunk uid, so the ``update()``
  orphan cleanup keeps every entity that unchanged chunks still support.
- A cache failure falls back to real extraction (fail-open): the worst case is
  paying for LLM calls that could have been skipped, never data loss. The
  fallback is per chunk where possible — a chunk that cannot be rebuilt
  faithfully (a stored entity carrying no concrete label, or an edge whose
  endpoint no cached chunk mentions) is extracted on its own, leaving the
  rest of the document cached.
  The exception is an unreachable graph (``DatabaseUnavailableError``) or an
  exhausted latency budget, which propagate — extraction only needs the LLM,
  so it would bill in full and then fail at the write phase regardless. A
  plain ``DatabaseError`` (the server answered and rejected the query) does
  fall open: the rest of the pipeline is unaffected.

Semantics to be aware of (documented on ``GraphRAG.update()``):

- The cache mirrors the live graph. If entities were manually deleted from
  the graph, an update with caching enabled will NOT resurrect them from
  unchanged chunks (full re-extraction would). Pass
  ``cache_unchanged_chunks=False`` to force a full rebuild.
- If the ontology, prompts, or model changed since ingest, unchanged chunks
  keep their previously extracted data. Disable the cache to re-extract
  under the new configuration.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import DatabaseUnavailableError, LatencyBudgetExceededError
from graphrag_sdk.core.models import (
    EntityMention,
    GraphData,
    GraphNode,
    GraphRelationship,
    Ontology,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy

if TYPE_CHECKING:
    from graphrag_sdk.storage.graph_store import GraphStore

logger = logging.getLogger(__name__)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedChunkExtraction(ExtractionStrategy):
    """Extraction strategy that reuses graph data for unchanged chunks.

    Decorates a real extraction strategy: chunks whose text hash matches an
    existing chunk of ``document_id`` are rebuilt from the graph; the rest
    are forwarded to ``inner``. Designed for ``GraphRAG.update()`` — the
    facade wires it automatically when ``cache_unchanged_chunks=True``.

    After ``extract()`` returns, ``cached_chunk_count`` and
    ``extracted_chunk_count`` report the split so callers can surface cache
    effectiveness (``update()`` copies them into ``UpdateResult.metadata``).

    Args:
        inner: The real extraction strategy for new/changed chunks.
        graph_store: Graph store to read previously extracted data from.
        document_id: Id of the Document node being updated.
    """

    def __init__(
        self,
        inner: ExtractionStrategy,
        graph_store: GraphStore,
        document_id: str,
    ) -> None:
        if not document_id or not document_id.strip():
            raise ValueError("'document_id' must be a non-empty string")
        self._inner = inner
        self._graph_store = graph_store
        self._document_id = document_id
        # Filled during extract() so the caller can report cache stats.
        self.cached_chunk_count = 0
        self.extracted_chunk_count = 0

    async def _old_chunks_by_hash(self, exclude: set[str]) -> tuple[dict[str, str], set[str]]:
        """Return (sha256(chunk text) -> old chunk id, all old chunk ids).

        Ids in ``exclude`` — the uids of the chunks currently being written —
        are dropped. ``update()`` writes under a pending document id, so the
        read and write sets are disjoint there, but that is a property of the
        caller, not of this class. If they ever overlap (wired into a path
        where the read id is the write id, or the pending indirection
        changes), a chunk would hash to *itself*: the rebuild would find no
        mentions for a chunk nothing has been extracted from yet, emit empty
        ``GraphData``, and skip the extractor — dropping entities silently,
        with no exception for the fail-open path to catch and cache stats
        still reporting a hit. Excluding them degrades to normal extraction.
        """
        by_hash: dict[str, str] = {}
        all_ids: set[str] = set()
        for cid, text in await self._graph_store.get_document_chunk_texts(self._document_id):
            if cid in exclude:
                continue
            all_ids.add(cid)
            # First occurrence wins; duplicates map to the same content anyway.
            by_hash.setdefault(_sha256(text), cid)
        return by_hash, all_ids

    @staticmethod
    def _usable_chunks(
        old_ids: list[str],
        ent_rows: list[Any],
        rel_rows: list[Any],
    ) -> set[str]:
        """Largest subset of cached chunks that can be rebuilt faithfully.

        A chunk is unusable when it mentions an entity carrying no concrete
        label (``upsert_nodes`` would MERGE on a fallback label and mint a
        duplicate node), or when it cites a relationship whose endpoint no
        remaining cached chunk mentions. In both cases
        ``IngestionPipeline._filter_quality`` would drop the affected edges,
        the rebuilt provenance would never land, and the post-cutover sweep
        would then delete facts the unchanged chunk still supports.

        Dropping a chunk removes the entities only it mentioned, which can
        strand edges cited by other chunks, so this iterates to a fixpoint.
        Only the affected chunks are excluded — the rest of the document
        still gets its cache.
        """
        ents_by_chunk: dict[str, set[str]] = {}
        label_by_ent: dict[str, str | None] = {}
        for row in ent_rows:
            ents_by_chunk.setdefault(row.chunk_id, set()).add(row.entity_id)
            if not label_by_ent.get(row.entity_id):
                label_by_ent[row.entity_id] = row.label

        good = set(old_ids)
        while good:
            node_ids = {e for c in good for e in ents_by_chunk.get(c, ())}
            bad: set[str] = set()
            for c in good:
                if any(not label_by_ent.get(e) for e in ents_by_chunk.get(c, ())):
                    bad.add(c)
            for rel in rel_rows:
                if rel.chunk_id in good and rel.chunk_id not in bad:
                    if rel.start_entity_id not in node_ids or rel.end_entity_id not in node_ids:
                        bad.add(rel.chunk_id)
            if not bad:
                break
            good -= bad
        return good

    async def _graph_data_from_cache(
        self, pairs: list[tuple[str, str]], all_old_ids: set[str]
    ) -> tuple[GraphData, set[str]]:
        """Rebuild GraphData for all unchanged chunks from the live graph.

        Args:
            pairs: ``(old_chunk_id, new_chunk_uid)`` for every cached chunk.
                Multiple new chunks may map to the same old chunk (duplicate
                text within the document) — every new uid gets its own
                mentions and provenance.
            all_old_ids: every pre-update chunk id of this document — used to
                drop provenance pointing at chunks that die in the cutover.

        Returns:
            ``(graph_data, unusable_old_ids)``. Chunks in ``unusable_old_ids``
            could not be rebuilt faithfully and are excluded from the result;
            the caller sends their new chunks to real extraction instead.
        """
        # old chunk id -> ALL new chunk uids that reuse its extraction.
        id_map: dict[str, list[str]] = {}
        for old_id, new_uid in pairs:
            id_map.setdefault(old_id, []).append(new_uid)
        old_ids = list(id_map)

        ent_rows = await self._graph_store.get_entities_mentioned_in_chunks(old_ids)
        rel_rows = await self._graph_store.get_relationships_for_chunks(old_ids)

        good = self._usable_chunks(old_ids, ent_rows, rel_rows)
        unusable = set(old_ids) - good
        if unusable:
            logger.warning(
                "Cache cannot faithfully rebuild %d of %d unchanged chunk(s); "
                "extracting those instead",
                len(unusable),
                len(old_ids),
            )
        if not good:
            return GraphData(nodes=[], relationships=[], mentions=[]), unusable

        id_map = {old: uids for old, uids in id_map.items() if old in good}

        mentions: list[EntityMention] = []
        ent_props: dict[str, dict[str, Any]] = {}
        ent_labels: dict[str, str | None] = {}
        ent_sources: dict[str, list[str]] = {}

        for row in ent_rows:
            new_uids = id_map.get(row.chunk_id)
            if not new_uids:
                continue
            for new_uid in new_uids:
                mentions.append(EntityMention(chunk_id=new_uid, entity_id=row.entity_id))
            srcs = ent_sources.setdefault(row.entity_id, [])
            if row.entity_id not in ent_props:
                # Only emit non-empty fields — SET n += props would null-out
                # existing values otherwise.
                props: dict[str, Any] = {}
                if row.name:
                    props["name"] = row.name
                if row.type:
                    props["type"] = row.type
                if row.description:
                    props["description"] = row.description
                ent_props[row.entity_id] = props
                ent_labels[row.entity_id] = row.label
                # Remap provenance onto post-update chunk ids: this doc's
                # old ids map to their replacement uid(s) or die in the
                # cutover; other documents' chunk ids pass through.
                for old in row.source_chunk_ids:
                    if old in all_old_ids:
                        mapped_uids = id_map.get(old, [])
                    else:
                        mapped_uids = [old]
                    for mapped in mapped_uids:
                        if mapped not in srcs:
                            srcs.append(mapped)
            for new_uid in new_uids:
                if new_uid not in srcs:
                    srcs.append(new_uid)

        nodes: list[GraphNode] = []
        for eid, props in ent_props.items():
            label = ent_labels[eid]
            if not label:
                # _usable_chunks already excluded every chunk mentioning a
                # label-less entity, so reaching here means the row set
                # changed underneath us. Skip defensively rather than mint a
                # duplicate node on a fallback label.
                continue
            props["source_chunk_ids"] = ent_sources[eid]
            nodes.append(GraphNode(id=eid, label=label, properties=props))

        relationships: list[GraphRelationship] = []
        if ent_props:
            node_ids = {n.id for n in nodes}
            rel_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
            for rel in rel_rows:
                new_uids = id_map.get(rel.chunk_id)
                if not new_uids:
                    continue
                key = (rel.start_entity_id, rel.end_entity_id)
                if key[0] not in node_ids or key[1] not in node_ids:
                    # Same as above: the fixpoint guarantees both endpoints
                    # are present, so this is unreachable for a consistent
                    # read. Dropping the edge here is safe because the chunk
                    # that cites it would already have been excluded.
                    continue
                props = rel_by_pair.get(key)
                if props is None:
                    props = {"source_chunk_ids": []}
                    if rel.rel_type:
                        props["rel_type"] = rel.rel_type
                    if rel.description:
                        props["description"] = rel.description
                    if rel.fact:
                        props["fact"] = rel.fact
                    if rel.src_name:
                        props["src_name"] = rel.src_name
                    if rel.tgt_name:
                        props["tgt_name"] = rel.tgt_name
                    rel_by_pair[key] = props
                for new_uid in new_uids:
                    if new_uid not in props["source_chunk_ids"]:
                        props["source_chunk_ids"].append(new_uid)
            relationships = [
                GraphRelationship(start_node_id=s, end_node_id=e, type="RELATES", properties=p)
                for (s, e), p in rel_by_pair.items()
            ]

        return GraphData(nodes=nodes, relationships=relationships, mentions=mentions), unusable

    @staticmethod
    def _merge(parts: list[GraphData]) -> GraphData:
        """Merge cached and freshly extracted GraphData.

        Later parts win on conflicting properties (fresh extraction follows
        cached parts, so updated descriptions take precedence) — EXCEPT
        ``source_chunk_ids``, which is a union: an entity or fact present in
        both a cached and an extracted chunk must keep both provenances.

        Relationships are keyed on ``(start, type, end)`` for the same reason
        nodes are keyed on id. Concatenating them instead would leave two
        rows for one fact, and ``ExactMatchResolution`` keeps only the first
        and rebuilds from it — discarding the fresh part's corrected
        properties *and* its provenance, which the post-cutover sweep would
        later collect as an unsupported edge. Full re-extraction never hits
        that because ``_aggregate_relations`` collapses duplicates upstream;
        merging here keeps the cached path equivalent.
        """
        nodes_by_id: dict[str, GraphNode] = {}
        rels_by_key: dict[tuple[str, str, str], GraphRelationship] = {}
        mentions: list[EntityMention] = []
        extracted_entities = []
        extracted_relations = []

        def _union_sources(old_props: dict[str, Any], new_props: dict[str, Any]) -> dict[str, Any]:
            merged = {**old_props, **new_props}
            old_src = old_props.get("source_chunk_ids") or []
            new_src = new_props.get("source_chunk_ids") or []
            if old_src or new_src:
                merged["source_chunk_ids"] = list(old_src) + [
                    c for c in new_src if c not in old_src
                ]
            return merged

        for part in parts:
            for node in part.nodes:
                existing = nodes_by_id.get(node.id)
                if existing is None:
                    nodes_by_id[node.id] = node
                else:
                    merged = _union_sources(existing.properties or {}, node.properties or {})
                    label = node.label or existing.label
                    nodes_by_id[node.id] = GraphNode(id=node.id, label=label, properties=merged)
            for rel in part.relationships:
                key = (rel.start_node_id, rel.type, rel.end_node_id)
                current = rels_by_key.get(key)
                if current is None:
                    rels_by_key[key] = rel
                else:
                    merged = _union_sources(current.properties or {}, rel.properties or {})
                    rels_by_key[key] = GraphRelationship(
                        start_node_id=rel.start_node_id,
                        end_node_id=rel.end_node_id,
                        type=rel.type,
                        properties=merged,
                    )
            mentions.extend(part.mentions)
            extracted_entities.extend(part.extracted_entities)
            extracted_relations.extend(part.extracted_relations)
        return GraphData(
            nodes=list(nodes_by_id.values()),
            relationships=list(rels_by_key.values()),
            mentions=mentions,
            extracted_entities=extracted_entities,
            extracted_relations=extracted_relations,
        )

    async def extract(self, chunks: TextChunks, ontology: Ontology, ctx: Context) -> GraphData:
        """Split chunks into cached vs. new, rebuild the former from the
        graph, extract the latter with the inner strategy, and merge."""
        try:
            old_by_hash, all_old_ids = await self._old_chunks_by_hash(
                {c.uid for c in chunks.chunks}
            )
        except (DatabaseUnavailableError, LatencyBudgetExceededError):
            # Fail-open covers a cache that can't answer, not a graph that
            # isn't there. Extracting every chunk needs only the LLM, so it
            # would succeed and bill in full before the write phase hit the
            # same dead connection and failed anyway. A plain DatabaseError
            # means the server answered and rejected the query — the rest of
            # the pipeline still works, so that one falls open below.
            raise
        except Exception as exc:
            logger.warning("Chunk cache lookup failed, extracting everything: %s", exc)
            old_by_hash, all_old_ids = {}, set()

        cached: list[tuple[TextChunk, str]] = []  # (new chunk, old chunk id)
        to_extract: list[TextChunk] = []
        for chunk in chunks.chunks:
            old_id = old_by_hash.get(_sha256(chunk.text))
            if old_id is not None:
                cached.append((chunk, old_id))
            else:
                to_extract.append(chunk)

        self.cached_chunk_count = len(cached)
        self.extracted_chunk_count = len(to_extract)
        if ctx:
            ctx.log(
                f"update cache: {len(cached)} unchanged chunk(s) reused, "
                f"{len(to_extract)} chunk(s) sent to extraction"
            )

        parts: list[GraphData] = []
        if cached:
            try:
                data, unusable = await self._graph_data_from_cache(
                    [(old_id, chunk.uid) for chunk, old_id in cached], all_old_ids
                )
                if unusable:
                    # Only the chunks that could not be rebuilt go to the
                    # LLM; the rest of the document keeps its cache.
                    to_extract.extend(chunk for chunk, old_id in cached if old_id in unusable)
                    moved = sum(1 for _, old_id in cached if old_id in unusable)
                    self.cached_chunk_count -= moved
                    self.extracted_chunk_count += moved
                parts.append(data)
            except (DatabaseUnavailableError, LatencyBudgetExceededError):
                # Same reasoning as the lookup above: an unreachable graph
                # can't be recovered from by doing more LLM work.
                raise
            except Exception as exc:
                # Cache miss must never lose data — fall back to real extraction.
                logger.warning(
                    "Cache rebuild failed, extracting %d cached chunk(s) instead: %s",
                    len(cached),
                    exc,
                )
                to_extract.extend(chunk for chunk, _ in cached)
                self.extracted_chunk_count += self.cached_chunk_count
                self.cached_chunk_count = 0

        if to_extract:
            parts.append(await self._inner.extract(TextChunks(chunks=to_extract), ontology, ctx))

        return self._merge(parts)
