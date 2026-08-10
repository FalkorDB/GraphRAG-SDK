# Local-only adversarial edge coverage for the chunk-level extraction cache.
#
# The PR's own integration tests script the LLM with `relationships: []`, so
# the relationship-rebuild path is never exercised against a real FalkorDB.
# These tests close that gap and add the equivalence property that matters
# most in production: a cached update and a full-extraction update must
# converge on the SAME graph.
#
# Run: RUN_INTEGRATION=1 FALKOR_PORT=6399 pytest tests/test_cached_chunk_extraction_edges.py

from __future__ import annotations

from typing import Any

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    EntityMention,
    GraphData,
    GraphNode,
    GraphRelationship,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
    compute_entity_id,
)
from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution

SEP = "\n@@@\n"


class SepChunking(ChunkingStrategy):
    """Split on an explicit separator so chunk boundaries are exact.

    FixedSizeChunking makes "which chunks are byte-identical" a function of
    character arithmetic; that hides real cache behaviour behind padding
    tricks. Here the test author states it directly.
    """

    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        parts = [p for p in text.split(SEP) if p.strip()]
        return TextChunks(
            chunks=[TextChunk(text=p, index=i, metadata={}) for i, p in enumerate(parts)]
        )


class ScriptedExtractor(ExtractionStrategy):
    """Content-keyed extractor: the chunk text *is* the script.

    Line grammar (one directive per line):
        E|<name>|<type>|<description>
        R|<src name>|<tgt name>|<rel type>|<description>

    Deterministic and side-effect free, so the same chunk text always yields
    the same GraphData — which is what makes the cached-vs-uncached
    equivalence assertion meaningful. Mirrors GraphExtraction's id/property
    construction exactly (compute_entity_id, label=type, `fact` string) so
    cached rows MERGE onto the same nodes.
    """

    def __init__(self) -> None:
        self.seen_chunk_texts: list[str] = []

    async def extract(self, chunks: TextChunks, ontology, ctx) -> GraphData:
        nodes: dict[str, GraphNode] = {}
        rels: list[GraphRelationship] = []
        mentions: list[EntityMention] = []
        types: dict[str, str] = {}

        for chunk in chunks.chunks:
            self.seen_chunk_texts.append(chunk.text)
            for line in chunk.text.splitlines():
                line = line.strip()
                if line.startswith("E|"):
                    _, name, etype, desc = line.split("|", 3)
                    types[name.strip().lower()] = etype
                    eid = compute_entity_id(name, etype)
                    existing = nodes.get(eid)
                    srcs = list(existing.properties["source_chunk_ids"]) if existing else []
                    if chunk.uid not in srcs:
                        srcs.append(chunk.uid)
                    nodes[eid] = GraphNode(
                        id=eid,
                        label=etype,
                        properties={
                            "name": name,
                            "type": etype,
                            "description": desc,
                            "source_chunk_ids": srcs,
                        },
                    )
                    mentions.append(EntityMention(chunk_id=chunk.uid, entity_id=eid))

        # Second pass so relationship endpoints can reference entities
        # declared in any chunk of this batch (matches the real extractor,
        # which resolves types across the whole extraction).
        for chunk in chunks.chunks:
            for line in chunk.text.splitlines():
                line = line.strip()
                if line.startswith("R|"):
                    _, src, tgt, rtype, desc = line.split("|", 4)
                    fact = f"({src}, {rtype}, {tgt}): {desc}" if desc else f"({src}, {rtype}, {tgt})"
                    rels.append(
                        GraphRelationship(
                            start_node_id=compute_entity_id(src, types.get(src.strip().lower(), "")),
                            end_node_id=compute_entity_id(tgt, types.get(tgt.strip().lower(), "")),
                            type="RELATES",
                            properties={
                                "rel_type": rtype,
                                "fact": fact,
                                "description": desc,
                                "source_chunk_ids": [chunk.uid],
                                "src_name": src,
                                "tgt_name": tgt,
                            },
                        )
                    )

        return GraphData(nodes=list(nodes.values()), relationships=rels, mentions=mentions)


# ── Snapshot helpers ────────────────────────────────────────────────


async def _snapshot(rag) -> dict[str, Any]:
    """Content-addressed view of the graph, free of volatile chunk uids.

    Chunk uids are regenerated on every update, so a raw dump can never be
    compared across runs. Everything here is keyed by chunk *text* instead,
    which is exactly the equivalence we care about.
    """
    q = rag._graph_store.query_raw

    r = await q(
        "MATCH (e:__Entity__) RETURN e.name, e.type, e.description ORDER BY e.name, e.type"
    )
    entities = sorted(tuple(row) for row in (r.result_set or []))

    r = await q(
        "MATCH (a:__Entity__)-[rel:RELATES]->(b:__Entity__) "
        "RETURN a.name, b.name, rel.rel_type, rel.fact ORDER BY a.name, b.name"
    )
    relationships = sorted(tuple(row) for row in (r.result_set or []))

    r = await q(
        "MATCH (e:__Entity__)-[:MENTIONED_IN]->(c:Chunk) RETURN e.name, c.text"
    )
    mentions = sorted(tuple(row) for row in (r.result_set or []))

    r = await q("MATCH (:Document)-[:PART_OF]->(c:Chunk) RETURN c.text ORDER BY c.index")
    chunk_texts = sorted(row[0] for row in (r.result_set or []))

    # Entity provenance, resolved from volatile uids back to chunk text.
    r = await q(
        "MATCH (e:__Entity__) WHERE e.source_chunk_ids IS NOT NULL "
        "UNWIND e.source_chunk_ids AS cid "
        "OPTIONAL MATCH (c:Chunk {id: cid}) RETURN e.name, c.text"
    )
    provenance = sorted(tuple(row) for row in (r.result_set or []))

    return {
        "entities": entities,
        "relationships": relationships,
        "mentions": mentions,
        "chunks": chunk_texts,
        "provenance": provenance,
    }


async def _live_chunk_ids(rag, document_id: str) -> set[str]:
    r = await rag._graph_store.query_raw(
        "MATCH (:Document {id: $d})-[:PART_OF]->(c:Chunk) RETURN collect(c.id)",
        {"d": document_id},
    )
    return set((r.result_set or [[[]]])[0][0] or [])


async def _assert_no_dangling_provenance(rag) -> None:
    """No entity or RELATES edge may cite a chunk id that no longer exists.

    Dangling provenance is the classic silent corruption from an id remap
    bug: retrieval still "works" but citations resolve to nothing.
    """
    r = await rag._graph_store.query_raw("MATCH (c:Chunk) RETURN collect(c.id)")
    live = set((r.result_set or [[[]]])[0][0] or [])

    r = await rag._graph_store.query_raw(
        "MATCH (e:__Entity__) WHERE e.source_chunk_ids IS NOT NULL "
        "RETURN e.name, e.source_chunk_ids"
    )
    for name, srcs in r.result_set or []:
        stale = set(srcs or []) - live
        assert not stale, f"entity {name!r} cites dead chunks: {stale}"

    r = await rag._graph_store.query_raw(
        "MATCH ()-[rel:RELATES]->() WHERE rel.source_chunk_ids IS NOT NULL "
        "RETURN rel.fact, rel.source_chunk_ids"
    )
    for fact, srcs in r.result_set or []:
        stale = set(srcs or []) - live
        assert not stale, f"RELATES {fact!r} cites dead chunks: {stale}"


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def make_rag(real_falkordb_rag_factory):
    from tests.conftest import MockLLM

    def _make():
        return real_falkordb_rag_factory(llm=MockLLM(), resolver=ExactMatchResolution())

    return _make


# ── Document versions ───────────────────────────────────────────────

C_ALICE = "E|Alice|Person|Engineer at Acme\nE|Acme|Organization|Tech company\nR|Alice|Acme|WORKS_AT|Alice is employed by Acme"
C_CAROL = "E|Carol|Person|Manager in Berlin\nE|Berlin|Location|German city\nR|Carol|Berlin|BASED_IN|Carol runs the Berlin office"
C_DAVE = "E|Dave|Person|Manager in Munich\nE|Munich|Location|German city\nR|Dave|Munich|BASED_IN|Dave runs the Munich office"
C_EVE = "E|Eve|Person|Designer at Acme\nE|Acme|Organization|Tech company\nR|Eve|Acme|WORKS_AT|Eve is employed by Acme"


@pytest.mark.asyncio
@pytest.mark.integration
class TestChunkCacheEdges:
    async def test_cached_and_uncached_updates_converge(self, make_rag):
        """THE property test: caching is an optimisation, not a semantic change.

        Same ingest, same update, run twice on two isolated graphs — once
        with the cache and once without. Every entity, relationship,
        mention and provenance edge must match. Any divergence here is a
        correctness bug in the remap, not a tuning issue.
        """
        v1 = SEP.join([C_ALICE, C_CAROL])
        v2 = SEP.join([C_ALICE, C_DAVE])

        snapshots = {}
        for cached in (True, False):
            rag = make_rag()
            ex = ScriptedExtractor()
            await rag.ingest(
                text=v1, document_id="doc", chunker=SepChunking(), extractor=ex,
                resolver=ExactMatchResolution(),
            )
            await rag.update(
                text=v2, document_id="doc", chunker=SepChunking(), extractor=ex,
                resolver=ExactMatchResolution(), cache_unchanged_chunks=cached,
            )
            snapshots[cached] = await _snapshot(rag)
            await _assert_no_dangling_provenance(rag)

        assert snapshots[True] == snapshots[False], (
            "cached update diverged from full re-extraction:\n"
            f"cached  : {snapshots[True]}\n"
            f"uncached: {snapshots[False]}"
        )

    async def test_relationship_rebuilt_from_cache_survives_update(self, make_rag):
        """Relationship rebuild end-to-end — the gap in the PR's own tests.

        The unchanged chunk carries a RELATES edge. After a cached update
        that edge must still exist, keep its rel_type/fact, and cite only
        live chunks.
        """
        v1 = SEP.join([C_ALICE, C_CAROL])
        v2 = SEP.join([C_ALICE, C_DAVE])

        rag = make_rag()
        ex = ScriptedExtractor()
        await rag.ingest(text=v1, document_id="doc", chunker=SepChunking(), extractor=ex,
                         resolver=ExactMatchResolution())

        before = len(ex.seen_chunk_texts)
        result = await rag.update(
            text=v2, document_id="doc", chunker=SepChunking(), extractor=ex,
            resolver=ExactMatchResolution(), cache_unchanged_chunks=True,
        )
        after = ex.seen_chunk_texts[before:]

        assert result.metadata["cache_stats"] == {"cached_chunks": 1, "extracted_chunks": 1}
        assert after == [C_DAVE], f"only the changed chunk may be extracted, got {after}"

        r = await rag._graph_store.query_raw(
            "MATCH (a:__Entity__ {name:'Alice'})-[rel:RELATES]->(b:__Entity__ {name:'Acme'}) "
            "RETURN rel.rel_type, rel.fact, rel.source_chunk_ids"
        )
        assert r.result_set, "cached relationship was lost by the update"
        rel_type, fact, srcs = r.result_set[0]
        assert rel_type == "WORKS_AT"
        assert "Alice is employed by Acme" in fact
        live = await _live_chunk_ids(rag, "doc")
        assert srcs and set(srcs) <= live, f"relationship provenance is stale: {set(srcs) - live}"

        # Replaced chunk's relationship must be gone.
        r = await rag._graph_store.query_raw(
            "MATCH (:__Entity__ {name:'Carol'})-[rel:RELATES]->() RETURN count(rel)"
        )
        assert (r.result_set[0][0] if r.result_set else 0) == 0
        await _assert_no_dangling_provenance(rag)

    async def test_reordered_chunks_are_all_cached(self, make_rag):
        """Reordering paragraphs changes the document hash but no chunk text.

        Every chunk must be a cache hit and zero extraction must happen —
        this is the common 'section moved' docs PR.
        """
        v1 = SEP.join([C_ALICE, C_CAROL, C_DAVE])
        v2 = SEP.join([C_DAVE, C_ALICE, C_CAROL])

        rag = make_rag()
        ex = ScriptedExtractor()
        await rag.ingest(text=v1, document_id="doc", chunker=SepChunking(), extractor=ex,
                         resolver=ExactMatchResolution())
        before = len(ex.seen_chunk_texts)

        result = await rag.update(
            text=v2, document_id="doc", chunker=SepChunking(), extractor=ex,
            resolver=ExactMatchResolution(), cache_unchanged_chunks=True,
        )

        assert result.no_op is False, "reordering changes the doc hash, so this is a real update"
        assert result.metadata["cache_stats"] == {"cached_chunks": 3, "extracted_chunks": 0}
        assert ex.seen_chunk_texts[before:] == [], "zero LLM extraction expected"

        snap = await _snapshot(rag)
        assert {e[0] for e in snap["entities"]} == {
            "Alice", "Acme", "Carol", "Berlin", "Dave", "Munich",
        }
        assert len(snap["relationships"]) == 3
        await _assert_no_dangling_provenance(rag)

    async def test_duplicate_identical_chunks_each_get_provenance(self, make_rag):
        """Two byte-identical chunks map to ONE old chunk id.

        Both new uids must receive their own mention and appear in the
        entity's provenance — a naive dict remap would keep only one.
        """
        v1 = SEP.join([C_ALICE, C_CAROL])
        v2 = SEP.join([C_ALICE, C_ALICE, C_DAVE])

        rag = make_rag()
        ex = ScriptedExtractor()
        await rag.ingest(text=v1, document_id="doc", chunker=SepChunking(), extractor=ex,
                         resolver=ExactMatchResolution())

        result = await rag.update(
            text=v2, document_id="doc", chunker=SepChunking(), extractor=ex,
            resolver=ExactMatchResolution(), cache_unchanged_chunks=True,
        )
        assert result.metadata["cache_stats"] == {"cached_chunks": 2, "extracted_chunks": 1}

        r = await rag._graph_store.query_raw(
            "MATCH (:__Entity__ {name:'Alice'})-[:MENTIONED_IN]->(c:Chunk) RETURN count(c)"
        )
        assert r.result_set[0][0] == 2, "each duplicate chunk needs its own mention"

        r = await rag._graph_store.query_raw(
            "MATCH (e:__Entity__ {name:'Alice'}) RETURN e.source_chunk_ids"
        )
        srcs = set(r.result_set[0][0] or [])
        live = await _live_chunk_ids(rag, "doc")
        assert len(srcs) == 2 and srcs <= live
        await _assert_no_dangling_provenance(rag)

    async def test_other_document_provenance_untouched(self, make_rag):
        """An entity shared with a second document must keep that document's
        chunk ids after a cached update of the first."""
        rag = make_rag()
        ex = ScriptedExtractor()
        await rag.ingest(text=SEP.join([C_ALICE, C_CAROL]), document_id="docA",
                         chunker=SepChunking(), extractor=ex, resolver=ExactMatchResolution())
        await rag.ingest(text=C_EVE, document_id="docB",
                         chunker=SepChunking(), extractor=ex, resolver=ExactMatchResolution())

        b_ids = await _live_chunk_ids(rag, "docB")

        await rag.update(
            text=SEP.join([C_ALICE, C_DAVE]), document_id="docA", chunker=SepChunking(),
            extractor=ex, resolver=ExactMatchResolution(), cache_unchanged_chunks=True,
        )

        r = await rag._graph_store.query_raw(
            "MATCH (e:__Entity__ {name:'Acme'}) RETURN e.source_chunk_ids"
        )
        srcs = set(r.result_set[0][0] or [])
        assert b_ids and b_ids <= srcs, (
            f"docB provenance was dropped by docA's cached update: missing {b_ids - srcs}"
        )
        assert srcs & (await _live_chunk_ids(rag, "docA")), "docA provenance missing"
        await _assert_no_dangling_provenance(rag)

    async def test_removed_chunk_orphans_cleaned_under_cache(self, make_rag):
        """Shrinking a document must still orphan-clean, cache or not."""
        rag = make_rag()
        ex = ScriptedExtractor()
        await rag.ingest(text=SEP.join([C_ALICE, C_CAROL, C_DAVE]), document_id="doc",
                         chunker=SepChunking(), extractor=ex, resolver=ExactMatchResolution())

        result = await rag.update(
            text=C_ALICE, document_id="doc", chunker=SepChunking(), extractor=ex,
            resolver=ExactMatchResolution(), cache_unchanged_chunks=True,
        )
        assert result.metadata["cache_stats"] == {"cached_chunks": 1, "extracted_chunks": 0}

        snap = await _snapshot(rag)
        names = {e[0] for e in snap["entities"]}
        assert names == {"Alice", "Acme"}, f"orphans survived: {names}"
        assert len(snap["relationships"]) == 1
        await _assert_no_dangling_provenance(rag)

    async def test_repeated_cached_updates_are_stable(self, make_rag):
        """Idempotency: the same cached update applied repeatedly must not
        accumulate duplicate entities, mentions or provenance entries."""
        v1 = SEP.join([C_ALICE, C_CAROL])
        v2 = SEP.join([C_ALICE, C_DAVE])

        rag = make_rag()
        ex = ScriptedExtractor()
        await rag.ingest(text=v1, document_id="doc", chunker=SepChunking(), extractor=ex,
                         resolver=ExactMatchResolution())

        snaps = []
        for _ in range(3):
            await rag.update(text=v2, document_id="doc", chunker=SepChunking(), extractor=ex,
                             resolver=ExactMatchResolution(), cache_unchanged_chunks=True)
            await rag.update(text=v1, document_id="doc", chunker=SepChunking(), extractor=ex,
                             resolver=ExactMatchResolution(), cache_unchanged_chunks=True)
            snaps.append(await _snapshot(rag))
            await _assert_no_dangling_provenance(rag)

        assert snaps[0] == snaps[1] == snaps[2], "cached updates are not idempotent"

    async def test_noop_short_circuit_reports_no_cache_stats(self, make_rag):
        """Identical content still takes the document-hash fast path; the
        cache flag must not disturb it."""
        v1 = SEP.join([C_ALICE, C_CAROL])
        rag = make_rag()
        ex = ScriptedExtractor()
        await rag.ingest(text=v1, document_id="doc", chunker=SepChunking(), extractor=ex,
                         resolver=ExactMatchResolution())
        before = len(ex.seen_chunk_texts)

        result = await rag.update(text=v1, document_id="doc", chunker=SepChunking(),
                                  extractor=ex, resolver=ExactMatchResolution(),
                                  cache_unchanged_chunks=True)

        assert result.no_op is True
        assert "cache_stats" not in result.metadata
        assert ex.seen_chunk_texts[before:] == []
