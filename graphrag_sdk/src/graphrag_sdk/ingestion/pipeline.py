# GraphRAG SDK — Ingestion: Pipeline Orchestrator
# Pattern: Sequential Pipeline — domain-specific linear orchestrator.
# Flow: Load → Chunk → Lexical Graph (MANDATORY) → Extract → Prune → Resolve → Write + Index
#
# Origin: Neo4j lexical graph + pruning as mandatory steps;
#         User design for domain-specific sequential pipeline over generic DAG.

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import IngestionError
from graphrag_sdk.core.models import (
    DocumentInfo,
    DocumentOutput,
    EntityMention,
    GraphData,
    GraphRelationship,
    IngestionResult,
    Ontology,
    TextChunks,
    ensure_no_pending_marker,
    stable_document_id,
)
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
from graphrag_sdk.ingestion.lexical_graph import LexicalGraphWriter, _reported_short
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy
from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy

logger = logging.getLogger(__name__)

# Number of offending (src, tgt) pairs to include verbatim in the
# "pattern mismatch" warning. Enough to spot the inversion; bounded
# so a misconfigured ontology can't flood logs.
_PATTERN_MISMATCH_SAMPLE_SIZE = 3


def _has_explicit(info: DocumentInfo, field: str) -> bool:
    """True when the caller set ``field`` on ``info`` (and to a non-empty value).

    ``DocumentInfo.uid`` has a ``uuid4()`` default, so a truthiness test cannot
    tell "the caller chose this id" from "nobody did"; Pydantic's
    ``model_fields_set`` can.
    """
    return field in info.model_fields_set and bool(getattr(info, field))


def _resolve_identity(
    *,
    loaded: DocumentInfo,
    supplied: DocumentInfo | None,
    derived_uid: str,
    source: str,
) -> DocumentInfo:
    """Merge the loader's ``DocumentInfo`` with the caller's into the identity
    the run will write under.

    Precedence per field: caller's explicitly-set value, then the derived id
    (for ``uid``) or the loader's value / ``source`` (for ``path``). Metadata is
    the loader's overlaid with the caller's. The loader's own ``uid`` is never
    used — no shipped loader sets one, so it is always the random default.
    """
    if supplied is not None and _has_explicit(supplied, "uid"):
        uid = supplied.uid
    else:
        uid = derived_uid
    if supplied is not None and _has_explicit(supplied, "path"):
        path = supplied.path
    else:
        path = loaded.path or source
    metadata = {**loaded.metadata, **(supplied.metadata if supplied is not None else {})}
    return DocumentInfo(uid=uid, path=path, metadata=metadata)


def _assign_deterministic_chunk_uids(doc_info: DocumentInfo, chunks: TextChunks) -> None:
    """Replace random chunk UIDs with a content-derived, stable identity.

    ``TextChunk.uid`` defaults to ``uuid4()``, so every ingest of the *same*
    document minted brand-new chunk IDs. The lexical graph MERGEs chunks on
    that ID, so nothing ever matched and re-ingesting silently duplicated the
    whole chunk layer.

    Measured on the benchmark corpus before this fix — ingesting one unchanged
    document three times in a row::

        round 1: 17 Chunks, 171 MENTIONED_IN
        round 2: 34 Chunks, 343 MENTIONED_IN
        round 3: 51 Chunks, 513 MENTIONED_IN

    Entities were unaffected (94 -> 95) because those MERGE on a normalised
    name, which *is* stable. Only the lexical layer duplicated.

    The UID is derived from the owning document's UID, the chunk index and a
    hash of the chunk text. Including the document UID keeps two documents that
    happen to share a paragraph as distinct chunks; including the text means an
    edited document produces new chunks rather than silently overwriting the
    old text under a recycled ``doc:index`` key.

    Chunkers that deliberately assign their own meaningful UIDs are not
    special-cased: this runs after chunking and overwrites unconditionally, so
    identity is decided in exactly one place.

    Chunkers that enrich the text (``ContextualChunking`` prepends an
    LLM-written summary) record the untouched source in
    ``metadata["original_chunk"]``; that is what is hashed, so a differently
    worded summary on the next run does not mint a new id for the same text.

    Because ``index`` is part of the key, inserting a paragraph early in an
    edited document re-ids every chunk after it. That is deliberate — it keeps
    ``doc:index`` a true position — but it means an *edited* file grows the
    lexical layer under plain ``ingest()``; only a byte-identical file is a
    no-op (see the unchanged-document short-circuit in ``run``).

    ``GraphRAG.update()`` runs the pipeline under a transient
    ``<id>__pending__<hex>`` Document, so the chunks it writes are keyed on
    that pending id, not the final one. This is load-bearing, not an
    oversight: the pending and live documents coexist until the cutover, and
    the cutover deletes every chunk ``PART_OF`` the live document — chunks
    keyed on the final id would MERGE onto the live nodes and be deleted
    along with them. After ``update()`` the Document carries the new
    ``content_hash``, so a later ``ingest()`` of that content is a no-op and
    never re-derives ids; only an ``ingest()`` of *edited* content re-mints
    them (as it does for any edited file).
    """
    doc_key = doc_info.uid or doc_info.path or ""
    for chunk in chunks.chunks:
        base = chunk.metadata.get("original_chunk")
        text = base if isinstance(base, str) and base else chunk.text
        digest = hashlib.sha256(f"{doc_key}\x00{chunk.index}\x00{text}".encode()).hexdigest()
        chunk.uid = f"chunk-{digest[:32]}"


class IngestionPipeline(LexicalGraphWriter):
    """Sequential orchestrator for knowledge graph construction.

    Executes the fixed sequence:

    1. **Load** — read text from source via ``LoaderStrategy``
    2. **Chunk** — split text via ``ChunkingStrategy``
    3. **Lexical Graph** — create Document→Chunk provenance (MANDATORY)
    4. **Extract** — extract entities/relationships via ``ExtractionStrategy``
    5. **Prune** — filter against ontology (built-in, not a strategy)
    6. **Resolve** — deduplicate entities via ``ResolutionStrategy``
    7. **Write** — upsert to graph store (batched)
    8. **Mentions** — write MENTIONED_IN edges (parallel with step 9)
    9. **Index Chunks** — embed and index chunks in vector store (parallel with step 8)

    The pipeline is intentionally *not* a generic DAG — the fixed sequence
    is debuggable, loggable, and understandable.

    Args:
        loader: Data source loader strategy.
        chunker: Text chunking strategy.
        extractor: Entity/relationship extraction strategy.
        resolver: Entity resolution strategy.
        graph_store: Graph data access object (from ``storage/``).
        vector_store: Vector data access object (from ``storage/``).
        ontology: Graph ontology for extraction constraints and pruning.

    Example::

        pipeline = IngestionPipeline(
            loader=PdfLoader(),
            chunker=FixedSizeChunking(chunk_size=500),
            extractor=GraphExtraction(llm=my_llm),
            resolver=ExactMatchResolution(),
            graph_store=my_graph_store,
            vector_store=my_vector_store,
            ontology=my_schema,
        )
        result = await pipeline.run("document.pdf", ctx)
    """

    def __init__(
        self,
        loader: LoaderStrategy,
        chunker: ChunkingStrategy,
        extractor: ExtractionStrategy,
        resolver: ResolutionStrategy,
        graph_store: Any,  # storage.GraphStore — import avoided for layering
        vector_store: Any,  # storage.VectorStore
        ontology: Ontology | None = None,
        *,
        schema: Ontology | None = None,  # DEPRECATED: use ``ontology=`` instead
    ) -> None:
        # Back-compat: legacy ``schema=`` kwarg forwards to ``ontology=``.
        if schema is not None:
            import warnings

            if ontology is not None:
                raise TypeError(
                    "IngestionPipeline() received both `ontology=` and `schema=`. "
                    "Use `ontology=` only; `schema=` is deprecated."
                )
            warnings.warn(
                "The `schema=` keyword argument on IngestionPipeline has been "
                "renamed to `ontology=` (graphrag_sdk v1.2+). Update your call "
                "site — the alias will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            ontology = schema

        self.loader = loader
        self.chunker = chunker
        self.extractor = extractor
        self.resolver = resolver
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.ontology = ontology or Ontology()

    async def run(
        self,
        source: str,
        ctx: Context | None = None,
        *,
        text: str | None = None,
        document_info: DocumentInfo | None = None,
    ) -> IngestionResult:
        """Execute the full ingestion pipeline.

        Either ``source`` or ``text`` must be provided:
        - If ``source`` is given, the loader reads from it.
        - If ``text`` is given directly, the loader step is skipped.

        Args:
            source: Path/URL to load (passed to loader).
            ctx: Execution context (created automatically if None).
            text: Optional raw text (skips loader if provided).
            document_info: Optional pre-built document metadata.

        Returns:
            IngestionResult with statistics about the pipeline run.
        """
        if ctx is None:
            ctx = Context()

        # Identity the pipeline will use when the caller does not pin one
        # (no ``document_info``, or one without an explicit ``uid``). Text
        # mode hashes the text; file mode applies ``stable_document_id``
        # (normalised path, URIs verbatim). Computed up front so the
        # reserved-marker check fails fast, before any I/O, and is not
        # wrapped in ``IngestionError`` — same ``ValueError`` the facade
        # raises for an explicit ``document_id``.
        if text is not None:
            derived_uid = f"text-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
        else:
            derived_uid = stable_document_id(source)
        if document_info is None or not _has_explicit(document_info, "uid"):
            # Only *derived* ids are checked: ``GraphRAG.update()`` passes
            # its ``<id>__pending__<hex>`` DocumentInfo explicitly and must
            # get through.
            ensure_no_pending_marker(derived_uid)

        ctx.log("Pipeline starting")

        try:
            # Step 1: Load
            if text is not None:
                document = DocumentOutput(text=text, document_info=DocumentInfo(path=source))
                ctx.log("Using provided text (loader skipped)")
            else:
                ctx.log("Step 1/9: Loading source")
                document = await self.loader.load(source, ctx)

            # Loaders leave ``DocumentInfo.uid`` at its ``uuid4()`` default,
            # so without a stable identity every run of the same file was a
            # new Document — and, since the chunk ids include the document
            # id, a new set of chunks. Resolve the identity here so the
            # pipeline is idempotent on its own and not only through the
            # facade: the caller's explicitly-set fields win, then the
            # derived id / loader path, and metadata is merged (caller over
            # loader). ``model_fields_set`` is what tells an explicit
            # ``uid=`` apart from the random default — a
            # ``DocumentInfo(path=...)`` must not clobber the derived id
            # with a fresh uuid.
            document.document_info = _resolve_identity(
                loaded=document.document_info,
                supplied=document_info,
                derived_uid=derived_uid,
                source=source,
            )

            # Hash the loaded text. Written to the Document node at the END of
            # a successful run (step 9b) so ``GraphRAG.update()`` and the
            # short-circuit below can recognise unchanged content. SHA-256
            # hex; cost is negligible next to extraction.
            #
            # Assumes the loader returns deterministic text for the same
            # source. Loaders that inject non-deterministic content
            # (timestamps, randomized ordering, run-id watermarks, etc.)
            # will produce a different hash on every run and the no-op
            # short-circuit will never fire — correct, just not optimal.
            content_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()

            # Unchanged re-ingest short-circuit — before chunking, so a
            # no-op costs one graph lookup and nothing else (contextual and
            # semantic chunkers make provider calls per chunk). Stable chunk
            # UIDs made the lexical layer idempotent, but extraction still
            # re-ran and, being LLM work, named a few entities differently
            # each time: measured +3 to +16 entity nodes, +18 to +50 RELATES
            # and +30 MENTIONED_IN per re-ingest of one unchanged document.
            # Same rule update() uses for its no-op: a stored Document whose
            # content_hash matches means every chunk, entity and edge is
            # already there — the hash is only written once a run has
            # completed with every write reported in full, so a half-finished
            # ingest is never mistaken for a finished one. Changed text (new
            # hash) takes the full path; a changed ontology or strategy with
            # the same text is ``GraphRAG.update(force=True)``'s job.
            #
            # ``graph_store`` is duck-typed (``Any`` — see ``__init__``); the
            # pipeline only ever required ``upsert_nodes`` /
            # ``upsert_relationships``. A store without ``get_document_record``
            # simply never short-circuits (the previous behaviour) rather than
            # failing every ingest with ``AttributeError``.
            doc_uid = document.document_info.uid
            get_record = getattr(self.graph_store, "get_document_record", None)
            if doc_uid and get_record is not None:
                existing = await get_record(doc_uid)
                if existing is not None and existing.content_hash == content_hash:
                    ctx.log(
                        f"Document '{doc_uid}' already ingested with identical content; "
                        f"skipping. Use GraphRAG.update(..., force=True) to re-extract "
                        f"unchanged content with a new ontology, chunker or extractor."
                    )
                    return IngestionResult(
                        document_info=document.document_info,
                        chunks_indexed=0,
                        metadata={"skipped_unchanged": True, "content_hash": content_hash},
                    )

            # Step 2: Chunk
            ctx.log("Step 2/9: Chunking text")
            chunks = await self.chunker.chunk_document(document, ctx)

            if not chunks.chunks:
                ctx.log("No chunks produced, pipeline complete")
                return IngestionResult(document_info=document.document_info)

            _assign_deterministic_chunk_uids(document.document_info, chunks)

            # Writes that report a shortfall instead of raising (see
            # ``_reported_short``). Any entry here means the graph is not a
            # faithful copy of this document, so the content hash is *not*
            # recorded and the next ingest repairs it by re-running.
            incomplete_writes: list[str] = []

            # Step 3: Build lexical graph (MANDATORY — not a strategy).
            ctx.log("Step 3/9: Building lexical graph (provenance chain)")
            lexical_shortfall = await self._build_lexical_graph(document.document_info, chunks, ctx)
            if lexical_shortfall:
                incomplete_writes.append(lexical_shortfall)

            # Step 4: Extract entities & relationships
            ctx.log("Step 4/9: Extracting entities & relationships")
            graph_data = await self.extractor.extract(chunks, self.ontology, ctx)

            # Step 4b: Quality filter — remove empty-ID and invalid nodes
            graph_data = self._filter_quality(graph_data)

            # Step 5: Prune against ontology
            ctx.log("Step 5/9: Pruning against ontology")
            graph_data = self._prune(graph_data, self.ontology)

            # Step 6: Resolve duplicate entities
            ctx.log("Step 6/9: Resolving duplicates")
            resolved = await self.resolver.resolve(graph_data, ctx)

            # Step 6b: Rewrite mentions through the resolver's remap.
            # Without this, MENTIONED_IN edges that reference an entity
            # which was merged away during resolution would silently fail
            # to write — graph_store.upsert_relationships does MATCH (a)
            # MATCH (b) MERGE, and the MATCH on the merged-away id finds
            # nothing. That would silently break update()/delete_document()
            # orphan-cleanup correctness for any resolver that merges
            # entities (ExactMatch when same-id duplicates exist;
            # SemanticResolution and LLMVerifiedResolution always). The
            # ``if resolved.remap`` guard makes this a no-op when the
            # resolver returned an empty remap.
            if resolved.remap and graph_data.mentions:
                graph_data = self._remap_mentions(graph_data, resolved.remap)

            # Step 7: Write to graph (batched)
            ctx.log("Step 7/9: Writing to graph store")
            # ``upsert_nodes`` is deliberately not gated on its count: it
            # raises ``DatabaseError`` on a real write failure, and the only
            # nodes it drops silently are those whose id sanitises to empty.
            # A re-run would drop those identically, so counting them would
            # withhold ``content_hash`` for this document forever rather than
            # flag something the next ingest can repair.
            await self.graph_store.upsert_nodes(resolved.nodes)
            rels_written = await self.graph_store.upsert_relationships(resolved.relationships)
            if _reported_short(rels_written, len(resolved.relationships)):
                incomplete_writes.append(
                    f"relationships {rels_written}/{len(resolved.relationships)}"
                )

            # ╔══════════════════════════════════════════════════════════╗
            # ║  LOAD-BEARING — DO NOT MAKE STEP 8 ASYNCHRONOUS WITH     ║
            # ║  RESPECT TO run() RETURNING. READ BEFORE EDITING.        ║
            # ╚══════════════════════════════════════════════════════════╝
            #
            # Steps 8-9 run in parallel with each other (they are
            # independent), but the gather() MUST complete before run()
            # returns. v1.1.0's update()/delete_document() orphan-cleanup
            # is race-free under concurrent updates only because the new
            # MENTIONED_IN edges produced in step 8 are persisted to the
            # graph before pipeline.run() returns — and therefore before
            # the caller's cutover begins. Concurrent updates A and B
            # sharing an entity ``e1`` will then always observe ``e1`` to
            # have at least one incident MENTIONED_IN edge from B's old
            # chunks (pre-cutover) or B's new chunks (post-pipeline.run()),
            # so A's orphan-cleanup never wrongly deletes it.
            #
            # If you defer step 8 to a background task, batch it across
            # pipelines, or skip it under a flag, you must also serialize
            # updates inside apply_changes — the current default
            # update_concurrency=1 is what keeps concurrent updates
            # correct otherwise.
            #
            # Tripwire: tests/test_integration.py::
            #     TestIncrementalUpdateInvariants::
            #     test_concurrent_updates_preserve_shared_entity
            async def _step_mentions() -> tuple[int, str | None]:
                ctx.log("Step 8/9: Writing mentions (uncapped)")
                return await self._write_mentions(graph_data, ctx)

            async def _step_index_chunks() -> str | None:
                ctx.log("Step 9/9: Embedding & indexing chunks")
                indexed = await self.vector_store.index_chunks(chunks)
                if _reported_short(indexed, len(chunks.chunks)):
                    return f"chunks indexed {indexed}/{len(chunks.chunks)}"
                return None

            (mentions_written, mentions_shortfall), index_shortfall = await asyncio.gather(
                _step_mentions(),
                _step_index_chunks(),
            )
            incomplete_writes.extend(s for s in (mentions_shortfall, index_shortfall) if s)

            # Step 9b: only now record the content hash. Writing it in step 3
            # meant a failure in extraction or the graph write left a
            # Document that looked complete, and the next ingest skipped it
            # (Copilot review on #309). A partial run therefore retries in
            # full; only a finished run is recognised as unchanged. "Finished"
            # includes the writes that swallow per-item errors: a RELATES
            # edge dropped by a transient graph error or a chunk left without
            # an embedding by a rate-limited embedder must not be certified
            # complete, or it would never be repaired (galshubeli on #309).
            result_metadata: dict[str, Any] = {
                "merged_entities": resolved.merged_count,
                "raw_nodes": len(graph_data.nodes),
                "raw_relationships": len(graph_data.relationships),
                "mention_edges_created": mentions_written,
                # Per-chunk extraction report (see ``GraphData``). Lives
                # in metadata so it also survives ``GraphRAG.update()``,
                # which forwards metadata but rebuilds the typed fields.
                "extraction": {
                    "chunks_attempted": graph_data.chunks_attempted,
                    "chunks_skipped": graph_data.chunks_skipped,
                    "failed_chunks": list(graph_data.failed_chunks),
                    "relation_failed_chunks": list(graph_data.relation_failed_chunks),
                    "extraction_failed": graph_data.extraction_failed,
                },
            }
            # The extractor reports per-chunk failures instead of raising, so
            # a partial extraction must withhold the hash the same way a
            # partial write does — otherwise the next ingest would skip a
            # document whose chunks never produced their entities.
            if graph_data.failed_chunks or graph_data.relation_failed_chunks:
                incomplete_writes.append(
                    f"extraction failed for {len(graph_data.failed_chunks)} chunk(s), "
                    f"relations for {len(graph_data.relation_failed_chunks)} chunk(s)"
                )
            if graph_data.chunks_skipped:
                incomplete_writes.append(
                    f"extraction skipped {graph_data.chunks_skipped} chunk(s) (latency budget)"
                )
            if incomplete_writes:
                result_metadata["incomplete_writes"] = incomplete_writes
                ctx.log(
                    "Some writes were reported incomplete "
                    f"({'; '.join(incomplete_writes)}); content_hash not recorded — "
                    "the next ingest of this document re-runs in full"
                )
                logger.warning(
                    "Ingest of '%s' left incomplete writes (%s); not marking content_hash",
                    document.document_info.uid,
                    "; ".join(incomplete_writes),
                )
            else:
                await self._mark_content_hash(document.document_info.uid, content_hash)

            total_rels = len(resolved.relationships) + mentions_written
            result = IngestionResult(
                document_info=document.document_info,
                nodes_created=len(resolved.nodes),
                relationships_created=total_rels,
                chunks_indexed=len(chunks.chunks),
                metadata=result_metadata,
            )
            ctx.log(
                f"Pipeline complete: {result.nodes_created} nodes, "
                f"{result.relationships_created} rels, "
                f"{result.chunks_indexed} chunks indexed, "
                f"{mentions_written} mentions"
            )
            return result

        except IngestionError:
            raise
        except Exception as exc:
            logger.error("Pipeline failed with unexpected error: %s", exc)
            logger.debug("Pipeline failure details", exc_info=True)
            raise IngestionError(f"Pipeline failed: {exc}") from exc

    def _prune(self, graph_data: GraphData, ontology: Ontology) -> GraphData:
        """Filter graph data to only include ontology-conforming nodes and relationships.

        Each check runs only when the corresponding ontology section is populated:

        - ``ontology.entities`` present → nodes whose label is not declared (or
          "Unknown") are dropped; otherwise all nodes pass.
        - ``ontology.relations`` present → relationships whose ``rel_type`` is not
          declared are dropped, and each declared relation's ``patterns`` (when
          non-empty) filters by ``(src_label, tgt_label)``; otherwise all
          relationships whose endpoints survived node-pruning pass.

        When both sections are empty the pipeline is in open-ontology mode and
        the graph is returned unchanged.
        """
        if not ontology.entities and not ontology.relations:
            return graph_data

        # --- Node filtering (keep "Unknown" — low-confidence entities) ---
        allowed_labels = {e.label for e in ontology.entities}
        if allowed_labels:
            allowed_labels.add("Unknown")
            pruned_nodes = [n for n in graph_data.nodes if n.label in allowed_labels]
        else:
            pruned_nodes = graph_data.nodes

        nodes_by_id = {n.id: n for n in pruned_nodes}

        # --- Build relation catalog ---
        # label -> set of (src, tgt) pairs, or None for "open" (no pattern constraint).
        allowed_rels: dict[str, set[tuple[str, str]] | None] = {}
        for rt in ontology.relations:
            allowed_rels[rt.label] = set(rt.patterns) if rt.patterns else None

        # --- Relationship filtering ---
        pruned_rels: list[GraphRelationship] = []
        # Track (src, tgt) pairs dropped for pattern-mismatch (vs unknown rel
        # type or pruned endpoints). Direction inversion is the most common
        # cause and the hardest to debug without a structured warning.
        pattern_mismatches: dict[str, list[tuple[str, str]]] = {}
        for r in graph_data.relationships:
            rel_label = r.properties.get("rel_type", r.type)

            src = nodes_by_id.get(r.start_node_id)
            tgt = nodes_by_id.get(r.end_node_id)
            if src is None or tgt is None:
                continue

            if not allowed_rels:
                pruned_rels.append(r)
                continue

            valid_pairs = allowed_rels.get(rel_label)
            if valid_pairs is None and rel_label in allowed_rels:
                pruned_rels.append(r)
            elif valid_pairs and (src.label, tgt.label) in valid_pairs:
                pruned_rels.append(r)
            elif valid_pairs:
                # Declared rel type, but (src, tgt) doesn't match any pattern.
                pattern_mismatches.setdefault(rel_label, []).append((src.label, tgt.label))

        pruned_node_count = len(graph_data.nodes) - len(pruned_nodes)
        pruned_rel_count = len(graph_data.relationships) - len(pruned_rels)
        if pruned_node_count or pruned_rel_count:
            logger.info(f"Pruned {pruned_node_count} nodes, {pruned_rel_count} rels")

        for rel_label, observed in pattern_mismatches.items():
            sample = observed[:_PATTERN_MISMATCH_SAMPLE_SIZE]
            declared = sorted(allowed_rels[rel_label] or set())
            logger.warning(
                "Pruned %d '%s' relationships due to (source, target) mismatch. "
                "Declared patterns: %s. Observed (sample): %s. "
                "If extraction looks correct, the pattern direction may be inverted.",
                len(observed),
                rel_label,
                declared,
                sample,
            )

        # ``model_copy`` rather than rebuilding field-by-field, so report
        # fields (``chunks_attempted``, ``failed_chunks``, ...) survive.
        return graph_data.model_copy(update={"nodes": pruned_nodes, "relationships": pruned_rels})

    def _filter_quality(self, graph_data: GraphData) -> GraphData:
        """Remove nodes with empty IDs or labels, and dangling relationships."""
        valid_nodes = [n for n in graph_data.nodes if n.id and n.label]
        removed = len(graph_data.nodes) - len(valid_nodes)
        if removed:
            logger.info(f"Quality filter removed {removed} invalid nodes")
        valid_ids = {n.id for n in valid_nodes}
        valid_rels = [
            r
            for r in graph_data.relationships
            if r.start_node_id in valid_ids and r.end_node_id in valid_ids
        ]
        return graph_data.model_copy(update={"nodes": valid_nodes, "relationships": valid_rels})

    @staticmethod
    def _remap_mentions(graph_data: GraphData, remap: dict[str, str]) -> GraphData:
        """Rewrite ``graph_data.mentions`` so MENTIONED_IN edges target
        the survivor entity, not a merged-away id.

        Called between the resolver step and the write step whenever the
        resolver reports a non-empty ``remap``. Without this, mentions
        carrying pre-resolution ids silently fail to be written (the
        upsert's MATCH on the merged-away id finds nothing), which would
        invalidate the orphan-cleanup invariant for fuzzy resolvers.

        Two-stage resolvers (``SemanticResolution``, ``LLMVerifiedResolution``)
        merge dicts from successive phases without flattening, so the
        remap can contain transitive chains like ``{A: B, B: C}`` where
        a single ``remap.get(A)`` returns ``B`` — itself a merged-away
        id. We follow each chain to its terminal survivor instead, so
        MENTIONED_IN edges always land on a node that exists in the
        graph. The ``visited`` guard makes a malformed cyclic remap
        terminate at the cycle's entry point rather than loop forever.
        """
        rewritten: list[EntityMention] = []
        seen: set[tuple[str, str]] = set()
        for m in graph_data.mentions:
            new_id = m.entity_id
            visited: set[str] = set()
            while new_id in remap and new_id not in visited:
                visited.add(new_id)
                new_id = remap[new_id]
            key = (new_id, m.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            rewritten.append(EntityMention(entity_id=new_id, chunk_id=m.chunk_id))
        return graph_data.model_copy(update={"mentions": rewritten})
