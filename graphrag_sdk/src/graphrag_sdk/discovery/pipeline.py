# GraphRAG SDK — Discovery: pipeline orchestrator
#
# Implements ``Ontology.from_sources`` by composing existing
# primitives (loaders, chunkers, LLM) with the discovery-only pieces
# in this package (prompts, instructor wrapper, validators).
#
# Pipeline shape:
#
#   for each source:
#     1. Load → DocumentOutput               (existing LoaderStrategy)
#     2. Chunk → TextChunks                  (existing ChunkingStrategy)
#     3. Sample N chunks                     (random.sample inline)
#     4. Doc-summary LLM call (anchors 5)    (extract_with_retry, soft-fail)
#     5. Per-chunk proposal LLM calls        (extract_with_retry, soft-fail)
#     6. Per-doc merge                       (existing Ontology.merge)
#   7. Corpus merge across docs               (existing Ontology.merge)
#   8. Normalization LLM call                 (extract_with_retry, soft-fail)
#   9. Final merge with existing              (existing Ontology.merge)
#
# Policy: the validation-retry wrapper is strict (raises on exhausted
# retries). The pipeline catches those failures so one noisy chunk does
# not sink the whole draft. Failed units are logged. Total counts are
# surfaced through ``ctx.log`` for traceability but not returned as a
# stats struct — the v1 contract is just "return an Ontology".

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    Attribute,
    DocumentOutput,
    Entity,
    Ontology,
    Relation,
    TextChunk,
    TextChunks,
    _PROPERTY_TYPES,
    _RESERVED_ATTRIBUTE_NAMES,
    _SDK_MANAGED_ATTRIBUTE_NAMES,
)
from graphrag_sdk.discovery.instructor import extract_with_retry
from graphrag_sdk.discovery.prompts import (
    SYSTEM_PROMPT,
    chunk_proposal_prompt,
    doc_summary_prompt,
    normalization_prompt,
)
from graphrag_sdk.discovery.proposal import (
    ChunkProposal,
    DocSummary,
    NormalizedDraft,
    OntologyDiscoveryError,
    SchemaExtensionProposal,
    _ProposedAttribute,
    _ProposedEntity,
    _ProposedRelation,
)
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy
from graphrag_sdk.ingestion.chunking_strategies.sentence_token_cap import (
    SentenceTokenCapChunking,
)
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy
from graphrag_sdk.ingestion.loaders.markdown_loader import MarkdownLoader
from graphrag_sdk.ingestion.loaders.pdf_loader import PdfLoader
from graphrag_sdk.ingestion.loaders.text_loader import TextLoader

if TYPE_CHECKING:
    from graphrag_sdk.core.providers.base import LLMInterface

logger = logging.getLogger(__name__)

# Max characters of a document fed to the summary step. The summary is
# anchoring context, not exhaustive — feeding the whole doc burns
# tokens for no quality lift.
_SUMMARY_TEXT_CAP = 4000


# ── Source loading (duplicates GraphRAG._default_loader_for) ────────


def _default_loader_for(source: str) -> LoaderStrategy:
    """Mirror of ``GraphRAG._default_loader_for``.

    Duplicated here so the discovery module does not import the
    ``GraphRAG`` facade — keeping ``Ontology.from_sources`` a pure
    function with no DB-connection dependency.
    """
    lower = source.lower()
    if lower.endswith(".pdf"):
        return PdfLoader()
    if lower.endswith(".md"):
        return MarkdownLoader()
    return TextLoader()


# ── Conversion: proposal → Ontology fragment ────────────────────────


def _attr_from_proposed(p: _ProposedAttribute) -> Attribute:
    return Attribute(name=p.name, type=p.type, description=p.description)


def _entity_from_proposed(p: _ProposedEntity) -> Entity:
    return Entity(
        label=p.label,
        description=p.description,
        properties=[_attr_from_proposed(a) for a in p.properties],
    )


def _relation_from_proposed(p: _ProposedRelation) -> Relation:
    return Relation(
        label=p.label,
        description=p.description,
        patterns=list(p.patterns),
        properties=[_attr_from_proposed(a) for a in p.properties],
    )


def _proposal_to_ontology(
    entities: list[_ProposedEntity], relations: list[_ProposedRelation]
) -> Ontology:
    """Lift validated proposal pieces into an ``Ontology``.

    Pydantic validators on ``Attribute`` will still reject anything the
    discovery validator missed (e.g. a property type that snuck through
    as ``"int"`` instead of ``"INTEGER"``), so we let those raise — a
    Pydantic error here is a real bug in our validator.
    """
    return Ontology(
        entities=[_entity_from_proposed(e) for e in entities],
        relations=[_relation_from_proposed(r) for r in relations],
    )


# ── Semantic validators (extra_validate for extract_with_retry) ─────


def _validate_proposal(
    proposal: ChunkProposal | NormalizedDraft,
) -> list[str]:
    """Semantic checks beyond what Pydantic enforces on _ProposedX.

    Catches the failure modes that ``Ontology`` itself only warns about
    (pattern labels referencing missing entities, attribute names that
    would shadow system keys) plus the property-type allow-list.
    """
    errors: list[str] = []
    declared = {e.label for e in proposal.entities}
    # Reserved names that are *truly* internal (extraction would conflict).
    # SDK-managed names like 'name' are allowed in the schema as documentation
    # and filtered out of the extraction prompt downstream.
    forbidden = _RESERVED_ATTRIBUTE_NAMES - _SDK_MANAGED_ATTRIBUTE_NAMES

    # Entity-level attribute checks.
    for e in proposal.entities:
        for a in e.properties:
            normalized = (a.type or "STRING").strip().upper()
            if normalized not in _PROPERTY_TYPES:
                errors.append(
                    f"Entity '{e.label}' attribute '{a.name}' has type "
                    f"'{a.type}' — must be one of {sorted(_PROPERTY_TYPES)}."
                )
            if a.name in forbidden:
                errors.append(
                    f"Entity '{e.label}' attribute '{a.name}' uses a reserved "
                    f"name that would shadow an SDK-written value — rename it."
                )

    # Relation-level checks.
    for r in proposal.relations:
        for src, tgt in r.patterns:
            missing = [lbl for lbl in (src, tgt) if lbl not in declared]
            if missing:
                errors.append(
                    f"Relation '{r.label}' pattern ({src}, {tgt}) references "
                    f"entity label(s) not declared in this proposal: "
                    f"{', '.join(missing)}. Add them to 'entities' or drop the pattern."
                )
        for a in r.properties:
            normalized = (a.type or "STRING").strip().upper()
            if normalized not in _PROPERTY_TYPES:
                errors.append(
                    f"Relation '{r.label}' attribute '{a.name}' has type "
                    f"'{a.type}' — must be one of {sorted(_PROPERTY_TYPES)}."
                )
            # Relations have no `name` field, so all reserved names remain
            # forbidden on relation-level attributes (no managed-name allow-list).
            if a.name in _RESERVED_ATTRIBUTE_NAMES:
                errors.append(
                    f"Relation '{r.label}' attribute '{a.name}' uses a reserved "
                    f"name that would shadow an SDK-written value — rename it."
                )

    return errors


def _ensure_sdk_managed_attributes(ontology: Ontology) -> Ontology:
    """Guarantee every entity declares the SDK-managed attributes (``name``).

    Discovery is allowed to produce an ontology that doesn't mention
    ``name``; we add it here so the saved schema honestly reflects what
    every extracted entity will carry. The added attribute is placed
    first and carries a description explaining the SDK-managed semantic.

    The extraction layer (``_render_attribute_block`` in
    ``ingestion/extraction_strategies/graph_extraction.py``) filters
    SDK-managed names back out so the LLM is never asked to extract
    them as per-entity properties.
    """
    if not ontology.entities:
        return ontology

    name_attribute = Attribute(
        name="name",
        type="STRING",
        description=(
            "Entity identifier. Auto-populated during extraction from "
            "the entity span detected in the source text; not extracted "
            "as a per-entity attribute by the LLM."
        ),
    )

    new_entities: list[Entity] = []
    changed = False
    for entity in ontology.entities:
        if any(a.name == "name" for a in entity.properties):
            new_entities.append(entity)
            continue
        changed = True
        new_entities.append(
            Entity(
                label=entity.label,
                description=entity.description,
                properties=[name_attribute, *entity.properties],
            )
        )

    if not changed:
        return ontology

    return Ontology(entities=new_entities, relations=list(ontology.relations))


# ── Per-step calls ──────────────────────────────────────────────────


async def _run_doc_summary(
    llm: "LLMInterface",
    document: DocumentOutput,
    *,
    boundaries: str | None,
    max_retries: int,
    doc_id: str,
) -> DocSummary:
    """First-pass per-doc LLM call. Anchors the per-chunk pass.

    On failure, returns an empty ``DocSummary`` so the pipeline keeps
    going — every chunk-level call still works without the anchoring
    frame, just less accurately.
    """
    text = document.text[:_SUMMARY_TEXT_CAP]
    try:
        return await extract_with_retry(
            llm,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=doc_summary_prompt(text, boundaries=boundaries),
            response_model=DocSummary,
            max_retries=max_retries,
            chunk_id=f"summary:{doc_id}",
        )
    except OntologyDiscoveryError as exc:
        logger.warning(
            "Doc summary failed for %s after %d attempts; continuing "
            "without anchor: %s",
            doc_id,
            exc.attempts,
            exc.last_error,
        )
        return DocSummary()


async def _run_chunk_proposal(
    llm: "LLMInterface",
    chunk: TextChunk,
    *,
    doc_summary: DocSummary,
    boundaries: str | None,
    existing: Ontology | None,
    max_retries: int,
) -> ChunkProposal | None:
    """Per-chunk LLM call wrapped in the validation-retry loop.

    Returns ``None`` on exhausted retries so the doc-level merge can
    simply skip this chunk — soft-fail at the pipeline level, strict at
    the wrapper level.
    """
    schema = ChunkProposal.model_json_schema()
    try:
        return await extract_with_retry(
            llm,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=chunk_proposal_prompt(
                chunk.text,
                doc_summary=doc_summary,
                boundaries=boundaries,
                existing=existing,
                response_schema=schema,
            ),
            response_model=ChunkProposal,
            extra_validate=_validate_proposal,
            max_retries=max_retries,
            chunk_id=chunk.uid,
        )
    except OntologyDiscoveryError as exc:
        logger.warning(
            "Chunk proposal failed for chunk %s after %d attempts; "
            "skipping. Last error: %s",
            chunk.uid,
            exc.attempts,
            exc.last_error,
        )
        return None


async def _run_normalization(
    llm: "LLMInterface",
    merged: Ontology,
    *,
    existing: Ontology | None,
    max_retries: int,
) -> Ontology:
    """Final normalization LLM call.

    On failure, falls back to the un-normalized merged draft — better
    to ship a noisier draft than to fail discovery entirely.
    """
    if not merged.entities and not merged.relations:
        return merged
    schema = NormalizedDraft.model_json_schema()
    draft_json = merged.model_dump_json()
    try:
        normalized = await extract_with_retry(
            llm,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=normalization_prompt(
                draft_json, existing=existing, response_schema=schema
            ),
            response_model=NormalizedDraft,
            extra_validate=_validate_proposal,
            max_retries=max_retries,
            chunk_id="normalize",
        )
    except OntologyDiscoveryError as exc:
        logger.warning(
            "Normalization failed after %d attempts; returning the merged "
            "draft un-normalized. Last error: %s",
            exc.attempts,
            exc.last_error,
        )
        return merged
    return _proposal_to_ontology(normalized.entities, normalized.relations)


# ── Per-doc orchestration ───────────────────────────────────────────


async def _process_document(
    llm: "LLMInterface",
    source: str,
    *,
    loader: LoaderStrategy | None,
    chunker: ChunkingStrategy,
    ctx: Context,
    sample_chunks_per_doc: int,
    boundaries: str | None,
    existing: Ontology | None,
    max_retries: int,
    chunk_sem: asyncio.Semaphore,
    rng: random.Random,
) -> Ontology:
    """Load + chunk + sample + summarize + per-chunk extract + merge.

    ``rng`` is per-document — derived from the user-supplied seed and the
    source string by the caller — so chunk sampling is deterministic
    regardless of how the asyncio scheduler interleaves doc tasks. Sharing
    one mutable Random across concurrent coroutines would make the same
    ``seed`` produce different samples across runs.
    """
    used_loader = loader or _default_loader_for(source)
    document = await used_loader.load(source, ctx)
    text_chunks: TextChunks = await chunker.chunk_document(document, ctx)

    if not text_chunks.chunks:
        ctx.log(f"discovery: no chunks produced for {source!r}", level=logging.INFO)
        return Ontology()

    # Sample.
    pool = text_chunks.chunks
    sample_size = min(sample_chunks_per_doc, len(pool))
    sampled = rng.sample(pool, sample_size) if sample_size < len(pool) else list(pool)

    # Doc-summary anchor — gated by the same semaphore so the documented
    # ``concurrency`` cap genuinely bounds *total* in-flight LLM calls. Each
    # source costs one summary call here plus N chunk calls below; without
    # the gate, fan-out is doc_count × (1 + N) instead of the advertised cap.
    async with chunk_sem:
        doc_summary = await _run_doc_summary(
            llm,
            document,
            boundaries=boundaries,
            max_retries=max_retries,
            doc_id=source,
        )

    # Per-chunk proposals under the same shared semaphore.
    async def _bounded(chunk: TextChunk) -> ChunkProposal | None:
        async with chunk_sem:
            return await _run_chunk_proposal(
                llm,
                chunk,
                doc_summary=doc_summary,
                boundaries=boundaries,
                existing=existing,
                max_retries=max_retries,
            )

    chunk_results = await asyncio.gather(*[_bounded(c) for c in sampled])

    # Per-doc merge: turn each proposal into an Ontology fragment and union them.
    doc_ontology = Ontology()
    for proposal in chunk_results:
        if proposal is None:
            continue
        try:
            fragment = _proposal_to_ontology(proposal.entities, proposal.relations)
        except Exception as exc:
            logger.warning(
                "Discarding chunk proposal that failed Pydantic conversion "
                "(should not happen — validator missed something): %s",
                exc,
            )
            continue
        doc_ontology = doc_ontology.merge(fragment)

    return doc_ontology


# ── Public entry point ──────────────────────────────────────────────


async def discover_ontology(
    sources: str | list[str],
    llm: "LLMInterface",
    *,
    boundaries: str | None = None,
    existing: Ontology | None = None,
    sample_chunks_per_doc: int = 3,
    max_retries: int = 3,
    concurrency: int = 4,
    chunker: ChunkingStrategy | None = None,
    loader: LoaderStrategy | None = None,
    ctx: Context | None = None,
    seed: int | None = None,
) -> Ontology:
    """Implementation of :py:meth:`Ontology.from_sources`.

    Args:
        sources: Single source string or a list of them. Same union the
            ``GraphRAG.ingest`` file-mode source argument accepts. The
            default loader is auto-selected by file extension.
        llm: Any ``LLMInterface``.
        boundaries: Free-text scope hint passed into every prompt.
        existing: Optional structured prior. When provided, the chunk
            and normalization prompts include this as a soft controlled
            vocabulary, and the returned ontology is merged with it.
        sample_chunks_per_doc: How many chunks per document to send to
            the per-chunk proposal step.
        max_retries: Retry budget per individual LLM call inside the
            instructor wrapper.
        concurrency: Max in-flight LLM calls across all docs.
        chunker: Override the default ``SentenceTokenCapChunking``.
        loader: Override the per-source auto-detected loader.
        ctx: Optional execution context for logging / tenancy.
        seed: Optional RNG seed for deterministic chunk sampling
            (useful in tests).
    """
    if sample_chunks_per_doc < 1:
        raise ValueError("sample_chunks_per_doc must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    sources_list = [sources] if isinstance(sources, str) else list(sources)
    if not sources_list:
        return existing or Ontology()

    used_chunker = chunker or SentenceTokenCapChunking()
    used_ctx = ctx or Context()
    chunk_sem = asyncio.Semaphore(concurrency)
    used_ctx.log(
        f"discovery: starting from_sources on {len(sources_list)} source(s), "
        f"sample_chunks_per_doc={sample_chunks_per_doc}, "
        f"concurrency={concurrency}",
        level=logging.INFO,
    )

    # Derive a per-source RNG so chunk sampling is deterministic regardless
    # of asyncio task interleaving. Sharing one Random across concurrent
    # _process_document tasks would make the same ``seed`` produce different
    # samples across runs.
    def _per_source_rng(src: str) -> random.Random:
        if seed is None:
            return random.Random()
        return random.Random(f"{seed}:{src}")

    per_doc = await asyncio.gather(
        *[
            _process_document(
                llm,
                src,
                loader=loader,
                chunker=used_chunker,
                ctx=used_ctx,
                sample_chunks_per_doc=sample_chunks_per_doc,
                boundaries=boundaries,
                existing=existing,
                max_retries=max_retries,
                chunk_sem=chunk_sem,
                rng=_per_source_rng(src),
            )
            for src in sources_list
        ]
    )

    # Corpus-level merge.
    merged = Ontology()
    for doc_ont in per_doc:
        merged = merged.merge(doc_ont)

    # Normalization pass.
    normalized = await _run_normalization(
        llm, merged, existing=existing, max_retries=max_retries
    )

    # Final merge with existing prior, if provided.
    if existing is not None:
        normalized = existing.merge(normalized)

    # Guarantee schema honesty: every entity declares its `name` attribute,
    # even though extraction fills it via the top-level identifier rather
    # than the per-entity attributes dict.
    normalized = _ensure_sdk_managed_attributes(normalized)

    used_ctx.log(
        f"discovery: produced {len(normalized.entities)} entity type(s), "
        f"{len(normalized.relations)} relation type(s)",
        level=logging.INFO,
    )
    return normalized


# ── Layer B: schema extension proposal ──────────────────────────────


def _diff_ontologies(
    existing: Ontology, discovered: Ontology
) -> SchemaExtensionProposal:
    """Compute additions from ``existing`` to ``discovered``.

    Pure structural diff, no LLM. Identity is by label for entities and
    relations, by ``(owner_label, attribute_name)`` for attributes, and
    by ``(rel_label, src, tgt)`` for relation patterns.

    The proposal carries only additions — labels in ``existing`` that
    are missing from ``discovered`` do not appear. We intentionally do
    not propose deletions, because absence-from-this-corpus is not
    evidence of irrelevance.
    """
    existing_entity_labels = {e.label for e in existing.entities}
    existing_entities_by_label = {e.label: e for e in existing.entities}
    existing_relation_labels = {r.label for r in existing.relations}
    existing_relations_by_label = {r.label: r for r in existing.relations}

    new_entities: list[Entity] = []
    new_attributes: list[tuple[str, Attribute]] = []
    for entity in discovered.entities:
        if entity.label not in existing_entity_labels:
            new_entities.append(entity)
            continue
        existing_entity = existing_entities_by_label[entity.label]
        existing_attr_names = {a.name for a in existing_entity.properties}
        for attr in entity.properties:
            if attr.name in _SDK_MANAGED_ATTRIBUTE_NAMES:
                # SDK-managed names are conceptually present on every entity
                # even when not explicitly declared, so they never count as
                # a new addition the user must apply.
                continue
            if attr.name not in existing_attr_names:
                new_attributes.append((entity.label, attr))

    new_relations: list[Relation] = []
    new_patterns: list[tuple[str, str, str]] = []
    for relation in discovered.relations:
        if relation.label not in existing_relation_labels:
            new_relations.append(relation)
            continue
        existing_relation = existing_relations_by_label[relation.label]
        existing_pattern_set = set(existing_relation.patterns)
        for src, tgt in relation.patterns:
            if (src, tgt) not in existing_pattern_set:
                new_patterns.append((relation.label, src, tgt))
        # Surface relation-property additions on already-known relation
        # labels (mirrors the entity-side property diff above). Without
        # this, discovery on a corpus that introduces a new property to
        # an existing relation would drop it silently from the proposal.
        # The v1 mutation API does not yet apply relation-attribute
        # changes — the field is surfaced for visibility; applying is the
        # caller's responsibility once that landing lands.
        existing_rel_attr_names = {a.name for a in existing_relation.properties}
        for attr in relation.properties:
            if attr.name not in existing_rel_attr_names:
                new_attributes.append((relation.label, attr))

    return SchemaExtensionProposal(
        new_entities=new_entities,
        new_relations=new_relations,
        new_patterns=new_patterns,
        new_attributes=new_attributes,
    )


async def suggest_extensions(
    existing: Ontology,
    sources: str | list[str],
    llm: "LLMInterface",
    *,
    boundaries: str | None = None,
    sample_chunks_per_doc: int = 3,
    max_retries: int = 3,
    concurrency: int = 4,
    chunker: ChunkingStrategy | None = None,
    loader: LoaderStrategy | None = None,
    ctx: Context | None = None,
    seed: int | None = None,
) -> SchemaExtensionProposal:
    """Implementation of :py:meth:`GraphRAG.suggest_schema_extensions`.

    Runs the discovery pipeline with ``existing`` as a soft controlled
    vocabulary, then diffs the result against ``existing`` to produce
    additions only.
    """
    discovered = await discover_ontology(
        sources,
        llm,
        boundaries=boundaries,
        existing=existing,
        sample_chunks_per_doc=sample_chunks_per_doc,
        max_retries=max_retries,
        concurrency=concurrency,
        chunker=chunker,
        loader=loader,
        ctx=ctx,
        seed=seed,
    )
    proposal = _diff_ontologies(existing, discovered)
    proposal.sources_scanned = (
        [sources] if isinstance(sources, str) else list(sources)
    )
    return proposal
