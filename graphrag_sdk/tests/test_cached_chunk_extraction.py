"""Tests for ingestion/extraction_strategies/cached_chunk_extraction.py.

Unit tests use a fake graph store + recording inner extractor and cover the
full split/remap/merge/fallback matrix. Integration tests (RUN_INTEGRATION=1,
real FalkorDB) prove end-to-end that unchanged chunks skip LLM extraction —
the scripted LLM is strict, so any unexpected extraction raises loudly.
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    ChunkEntityRow,
    ChunkRelationshipRow,
    EntityMention,
    GraphData,
    GraphNode,
    Ontology,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
from graphrag_sdk.ingestion.extraction_strategies.cached_chunk_extraction import (
    CachedChunkExtraction,
)
from graphrag_sdk.storage.graph_store import GraphStore


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FakeGraphStore:
    """Minimal in-memory stand-in for the three cache read accessors."""

    def __init__(
        self,
        chunk_texts: list[tuple[str, str]] | None = None,
        entity_rows: list[ChunkEntityRow] | None = None,
        rel_rows: list[ChunkRelationshipRow] | None = None,
    ) -> None:
        self.chunk_texts = chunk_texts or []
        self.entity_rows = entity_rows or []
        self.rel_rows = rel_rows or []

    async def get_document_chunk_texts(self, document_id: str) -> list[tuple[str, str]]:
        return self.chunk_texts

    async def get_entities_mentioned_in_chunks(
        self, chunk_ids: list[str]
    ) -> list[ChunkEntityRow]:
        wanted = set(chunk_ids)
        return [r for r in self.entity_rows if r.chunk_id in wanted]

    async def get_relationships_for_chunks(
        self, chunk_ids: list[str]
    ) -> list[ChunkRelationshipRow]:
        wanted = set(chunk_ids)
        return [r for r in self.rel_rows if r.chunk_id in wanted]


class RecordingExtractor(ExtractionStrategy):
    """Inner extractor that records every call and returns a canned result."""

    def __init__(self, result: GraphData | None = None) -> None:
        self.calls: list[TextChunks] = []
        self.result = result or GraphData()

    async def extract(self, chunks: TextChunks, ontology: Ontology, ctx: Context) -> GraphData:
        self.calls.append(chunks)
        return self.result


@pytest.fixture
def ontology() -> Ontology:
    return Ontology()


@pytest.fixture
def ctx() -> Context:
    return Context()


def _chunk(text: str, index: int = 0, uid: str | None = None) -> TextChunk:
    kwargs = {"text": text, "index": index}
    if uid is not None:
        kwargs["uid"] = uid
    return TextChunk(**kwargs)


class TestConstructor:
    def test_rejects_empty_document_id(self):
        with pytest.raises(ValueError, match="document_id"):
            CachedChunkExtraction(RecordingExtractor(), FakeGraphStore(), "")

    def test_rejects_whitespace_document_id(self):
        with pytest.raises(ValueError, match="document_id"):
            CachedChunkExtraction(RecordingExtractor(), FakeGraphStore(), "   ")


class TestSplit:
    async def test_all_chunks_cached_inner_never_called(self, ontology, ctx):
        store = FakeGraphStore(
            chunk_texts=[("old-1", "alpha"), ("old-2", "beta")],
            entity_rows=[
                ChunkEntityRow(
                    chunk_id="old-1",
                    entity_id="alice__person",
                    label="Person",
                    name="Alice",
                    type="Person",
                    description="Engineer",
                    source_chunk_ids=["old-1"],
                ),
                ChunkEntityRow(
                    chunk_id="old-2",
                    entity_id="acme__organization",
                    label="Organization",
                    name="Acme",
                    type="Organization",
                    source_chunk_ids=["old-2"],
                ),
            ],
        )
        inner = RecordingExtractor()
        strategy = CachedChunkExtraction(inner, store, "doc-1")

        new_chunks = TextChunks(
            chunks=[_chunk("alpha", 0, "new-1"), _chunk("beta", 1, "new-2")]
        )
        result = await strategy.extract(new_chunks, ontology, ctx)

        assert inner.calls == []
        assert strategy.cached_chunk_count == 2
        assert strategy.extracted_chunk_count == 0
        assert {m.chunk_id for m in result.mentions} == {"new-1", "new-2"}
        assert {m.entity_id for m in result.mentions} == {
            "alice__person",
            "acme__organization",
        }
        by_id = {n.id: n for n in result.nodes}
        assert by_id["alice__person"].label == "Person"
        assert by_id["alice__person"].properties["source_chunk_ids"] == ["new-1"]
        assert by_id["alice__person"].properties["description"] == "Engineer"
        assert by_id["acme__organization"].properties["source_chunk_ids"] == ["new-2"]

    async def test_all_chunks_new_everything_extracted(self, ontology, ctx):
        store = FakeGraphStore(chunk_texts=[("old-1", "something else entirely")])
        node = GraphNode(id="e1", label="Person", properties={"name": "Bob"})
        inner = RecordingExtractor(GraphData(nodes=[node]))
        strategy = CachedChunkExtraction(inner, store, "doc-1")

        new_chunks = TextChunks(chunks=[_chunk("alpha"), _chunk("beta", 1)])
        result = await strategy.extract(new_chunks, ontology, ctx)

        assert len(inner.calls) == 1
        assert [c.text for c in inner.calls[0].chunks] == ["alpha", "beta"]
        assert strategy.cached_chunk_count == 0
        assert strategy.extracted_chunk_count == 2
        assert result.nodes == [node]

    async def test_mixed_split_only_changed_chunk_extracted(self, ontology, ctx):
        store = FakeGraphStore(
            chunk_texts=[("old-1", "unchanged text")],
            entity_rows=[
                ChunkEntityRow(
                    chunk_id="old-1",
                    entity_id="e-cached",
                    label="Person",
                    name="Cached",
                    source_chunk_ids=["old-1"],
                )
            ],
        )
        inner = RecordingExtractor(
            GraphData(nodes=[GraphNode(id="e-new", label="Person", properties={})])
        )
        strategy = CachedChunkExtraction(inner, store, "doc-1")

        new_chunks = TextChunks(
            chunks=[_chunk("unchanged text", 0, "new-1"), _chunk("brand new", 1, "new-2")]
        )
        result = await strategy.extract(new_chunks, ontology, ctx)

        assert len(inner.calls) == 1
        assert [c.text for c in inner.calls[0].chunks] == ["brand new"]
        assert strategy.cached_chunk_count == 1
        assert strategy.extracted_chunk_count == 1
        ids = {n.id for n in result.nodes}
        assert ids == {"e-cached", "e-new"}

    async def test_empty_input_returns_empty_graph_data(self, ontology, ctx):
        inner = RecordingExtractor()
        strategy = CachedChunkExtraction(inner, FakeGraphStore(), "doc-1")
        result = await strategy.extract(TextChunks(chunks=[]), ontology, ctx)
        assert inner.calls == []
        assert result.nodes == [] and result.mentions == [] and result.relationships == []
        assert strategy.cached_chunk_count == 0
        assert strategy.extracted_chunk_count == 0


class TestProvenanceRemap:
    async def test_own_old_ids_remap_other_docs_pass_through(self, ontology, ctx):
        """source_chunk_ids: this doc's cached old id → new uid; this doc's
        dropped old id → removed; other documents' chunk ids → unchanged."""
        store = FakeGraphStore(
            chunk_texts=[("old-1", "kept"), ("old-2", "dropped content")],
            entity_rows=[
                ChunkEntityRow(
                    chunk_id="old-1",
                    entity_id="e1",
                    label="Person",
                    name="Alice",
                    source_chunk_ids=["old-1", "old-2", "other-doc-chunk"],
                )
            ],
        )
        inner = RecordingExtractor()
        strategy = CachedChunkExtraction(inner, store, "doc-1")

        new_chunks = TextChunks(
            chunks=[_chunk("kept", 0, "new-1"), _chunk("replacement", 1, "new-2")]
        )
        result = await strategy.extract(new_chunks, ontology, ctx)

        node = next(n for n in result.nodes if n.id == "e1")
        srcs = node.properties["source_chunk_ids"]
        assert "new-1" in srcs  # remapped
        assert "other-doc-chunk" in srcs  # passes through
        assert "old-1" not in srcs and "old-2" not in srcs  # old ids die

    async def test_duplicate_chunk_texts_all_get_mentions(self, ontology, ctx):
        """Two identical new chunks map to one old chunk — every new uid must
        get its own mention and appear in provenance (the server-side
        original collapsed these; the SDK port must not)."""
        store = FakeGraphStore(
            chunk_texts=[("old-1", "same text")],
            entity_rows=[
                ChunkEntityRow(
                    chunk_id="old-1",
                    entity_id="e1",
                    label="Person",
                    name="Alice",
                    source_chunk_ids=["old-1"],
                )
            ],
        )
        strategy = CachedChunkExtraction(RecordingExtractor(), store, "doc-1")

        new_chunks = TextChunks(
            chunks=[_chunk("same text", 0, "new-1"), _chunk("same text", 1, "new-2")]
        )
        result = await strategy.extract(new_chunks, ontology, ctx)

        assert strategy.cached_chunk_count == 2
        assert {m.chunk_id for m in result.mentions} == {"new-1", "new-2"}
        node = next(n for n in result.nodes if n.id == "e1")
        assert set(node.properties["source_chunk_ids"]) == {"new-1", "new-2"}


class TestRelationshipRebuild:
    async def test_relationships_reemitted_with_remapped_provenance(self, ontology, ctx):
        store = FakeGraphStore(
            chunk_texts=[("old-1", "alpha")],
            entity_rows=[
                ChunkEntityRow(
                    chunk_id="old-1",
                    entity_id="a",
                    label="Person",
                    name="A",
                    source_chunk_ids=["old-1"],
                ),
                ChunkEntityRow(
                    chunk_id="old-1",
                    entity_id="b",
                    label="Person",
                    name="B",
                    source_chunk_ids=["old-1"],
                ),
            ],
            rel_rows=[
                ChunkRelationshipRow(
                    chunk_id="old-1",
                    start_entity_id="a",
                    end_entity_id="b",
                    rel_type="KNOWS",
                    description="A knows B",
                    fact="(A, KNOWS, B)",
                    src_name="A",
                    tgt_name="B",
                )
            ],
        )
        strategy = CachedChunkExtraction(RecordingExtractor(), store, "doc-1")

        result = await strategy.extract(
            TextChunks(chunks=[_chunk("alpha", 0, "new-1")]), ontology, ctx
        )

        assert len(result.relationships) == 1
        rel = result.relationships[0]
        assert rel.type == "RELATES"
        assert rel.start_node_id == "a" and rel.end_node_id == "b"
        assert rel.properties["rel_type"] == "KNOWS"
        assert rel.properties["fact"] == "(A, KNOWS, B)"
        assert rel.properties["source_chunk_ids"] == ["new-1"]

    async def test_relationship_between_same_pair_deduped(self, ontology, ctx):
        """Two cached chunks supporting the same (a, b) edge produce ONE
        relationship whose provenance lists both new uids."""
        rows = [
            ChunkEntityRow(
                chunk_id=cid,
                entity_id=eid,
                label="Person",
                name=eid,
                source_chunk_ids=[cid],
            )
            for cid in ("old-1", "old-2")
            for eid in ("a", "b")
        ]
        store = FakeGraphStore(
            chunk_texts=[("old-1", "alpha"), ("old-2", "beta")],
            entity_rows=rows,
            rel_rows=[
                ChunkRelationshipRow(
                    chunk_id="old-1", start_entity_id="a", end_entity_id="b", rel_type="KNOWS"
                ),
                ChunkRelationshipRow(
                    chunk_id="old-2", start_entity_id="a", end_entity_id="b", rel_type="KNOWS"
                ),
            ],
        )
        strategy = CachedChunkExtraction(RecordingExtractor(), store, "doc-1")

        result = await strategy.extract(
            TextChunks(chunks=[_chunk("alpha", 0, "new-1"), _chunk("beta", 1, "new-2")]),
            ontology,
            ctx,
        )

        assert len(result.relationships) == 1
        assert set(result.relationships[0].properties["source_chunk_ids"]) == {
            "new-1",
            "new-2",
        }


class TestLabelLessEntity:
    async def test_label_less_entity_skips_node_but_keeps_mention(self, ontology, ctx):
        store = FakeGraphStore(
            chunk_texts=[("old-1", "alpha")],
            entity_rows=[
                ChunkEntityRow(
                    chunk_id="old-1",
                    entity_id="ghost",
                    label=None,
                    name="Ghost",
                    source_chunk_ids=["old-1"],
                )
            ],
        )
        strategy = CachedChunkExtraction(RecordingExtractor(), store, "doc-1")

        result = await strategy.extract(
            TextChunks(chunks=[_chunk("alpha", 0, "new-1")]), ontology, ctx
        )

        assert result.nodes == []  # no node write — would mint a duplicate
        assert result.mentions == [EntityMention(chunk_id="new-1", entity_id="ghost")]


class TestFailOpen:
    async def test_chunk_lookup_failure_extracts_everything(self, ontology, ctx):
        store = FakeGraphStore()
        store.get_document_chunk_texts = AsyncMock(side_effect=RuntimeError("boom"))
        inner = RecordingExtractor()
        strategy = CachedChunkExtraction(inner, store, "doc-1")

        await strategy.extract(
            TextChunks(chunks=[_chunk("alpha"), _chunk("beta", 1)]), ontology, ctx
        )

        assert len(inner.calls) == 1
        assert len(inner.calls[0].chunks) == 2
        assert strategy.cached_chunk_count == 0
        assert strategy.extracted_chunk_count == 2

    async def test_cache_rebuild_failure_falls_back_to_extraction(self, ontology, ctx):
        store = FakeGraphStore(chunk_texts=[("old-1", "alpha")])
        store.get_entities_mentioned_in_chunks = AsyncMock(side_effect=RuntimeError("boom"))
        inner = RecordingExtractor()
        strategy = CachedChunkExtraction(inner, store, "doc-1")

        await strategy.extract(
            TextChunks(chunks=[_chunk("alpha", 0), _chunk("beta", 1)]), ontology, ctx
        )

        # Both the changed chunk AND the failed cached chunk get extracted.
        assert len(inner.calls) == 1
        assert sorted(c.text for c in inner.calls[0].chunks) == ["alpha", "beta"]
        assert strategy.cached_chunk_count == 0
        assert strategy.extracted_chunk_count == 2

    async def test_stats_reset_between_calls(self, ontology, ctx):
        store = FakeGraphStore(chunk_texts=[("old-1", "alpha")])
        strategy = CachedChunkExtraction(RecordingExtractor(), store, "doc-1")

        await strategy.extract(TextChunks(chunks=[_chunk("alpha")]), ontology, ctx)
        assert (strategy.cached_chunk_count, strategy.extracted_chunk_count) == (1, 0)

        await strategy.extract(TextChunks(chunks=[_chunk("other")]), ontology, ctx)
        assert (strategy.cached_chunk_count, strategy.extracted_chunk_count) == (0, 1)


class TestMerge:
    async def test_entity_in_cached_and_extracted_chunk_unions_provenance(
        self, ontology, ctx
    ):
        """Same entity in an unchanged chunk (cached) and a changed chunk
        (extracted): fresh properties win, provenance is the union."""
        store = FakeGraphStore(
            chunk_texts=[("old-1", "unchanged")],
            entity_rows=[
                ChunkEntityRow(
                    chunk_id="old-1",
                    entity_id="e1",
                    label="Person",
                    name="Alice",
                    description="Old description",
                    source_chunk_ids=["old-1"],
                )
            ],
        )
        fresh = GraphData(
            nodes=[
                GraphNode(
                    id="e1",
                    label="Person",
                    properties={
                        "name": "Alice",
                        "description": "New description",
                        "source_chunk_ids": ["new-2"],
                    },
                )
            ],
            mentions=[EntityMention(chunk_id="new-2", entity_id="e1")],
        )
        strategy = CachedChunkExtraction(RecordingExtractor(fresh), store, "doc-1")

        result = await strategy.extract(
            TextChunks(chunks=[_chunk("unchanged", 0, "new-1"), _chunk("changed", 1, "new-2")]),
            ontology,
            ctx,
        )

        assert len(result.nodes) == 1
        node = result.nodes[0]
        assert node.properties["description"] == "New description"  # fresh wins
        assert set(node.properties["source_chunk_ids"]) == {"new-1", "new-2"}  # union
        assert {m.chunk_id for m in result.mentions} == {"new-1", "new-2"}

    async def test_extracted_entities_and_relations_pass_through(self, ontology, ctx):
        from graphrag_sdk.core.models import ExtractedEntity

        fresh = GraphData(
            extracted_entities=[ExtractedEntity(name="Alice", type="Person")],
        )
        strategy = CachedChunkExtraction(
            RecordingExtractor(fresh), FakeGraphStore(), "doc-1"
        )
        result = await strategy.extract(
            TextChunks(chunks=[_chunk("anything")]), ontology, ctx
        )
        assert len(result.extracted_entities) == 1


class TestGraphStoreCacheAccessors:
    """Unit tests for the three GraphStore read accessors backing the cache."""

    @pytest.fixture
    def graph_store(self, mock_connection):
        return GraphStore(mock_connection)

    async def test_get_document_chunk_texts_filters_bad_rows(
        self, graph_store, mock_connection
    ):
        from unittest.mock import MagicMock

        mock_connection.query = AsyncMock(
            return_value=MagicMock(
                result_set=[
                    ["c1", "hello"],
                    [None, "orphan text"],  # missing id → skipped
                    ["c2", None],  # missing text → skipped
                    ["c3", 42],  # non-string text → skipped
                ]
            )
        )
        rows = await graph_store.get_document_chunk_texts("doc-1")
        assert rows == [("c1", "hello")]
        cypher = mock_connection.query.call_args[0][0]
        assert "PART_OF" in cypher and "c.text" in cypher

    async def test_get_entities_mentioned_in_chunks_maps_rows(
        self, graph_store, mock_connection
    ):
        from unittest.mock import MagicMock

        mock_connection.query = AsyncMock(
            return_value=MagicMock(
                result_set=[
                    [
                        "c1",
                        "alice__person",
                        ["Person", "__Entity__"],
                        "Alice",
                        "Person",
                        "Engineer",
                        ["c1", "x"],
                    ],
                    ["c1", None, ["Person"], "NoId", "Person", None, []],  # skipped
                ]
            )
        )
        rows = await graph_store.get_entities_mentioned_in_chunks(["c1"])
        assert len(rows) == 1
        row = rows[0]
        assert row.entity_id == "alice__person"
        assert row.label == "Person"  # __Entity__ filtered out
        assert row.source_chunk_ids == ["c1", "x"]

    async def test_get_entities_label_none_when_no_concrete_label(
        self, graph_store, mock_connection
    ):
        """No fallback to e.type: a MERGE on a label the node doesn't carry
        would mint a duplicate node — the consumer must skip the node write
        (label=None triggers CachedChunkExtraction's skip guard)."""
        from unittest.mock import MagicMock

        mock_connection.query = AsyncMock(
            return_value=MagicMock(
                result_set=[["c1", "e1", ["__Entity__"], "X", "Widget", None, None]]
            )
        )
        rows = await graph_store.get_entities_mentioned_in_chunks(["c1"])
        assert rows[0].label is None

    async def test_get_relationships_for_chunks_maps_rows(
        self, graph_store, mock_connection
    ):
        from unittest.mock import MagicMock

        mock_connection.query = AsyncMock(
            return_value=MagicMock(
                result_set=[
                    # Edge supported by c1 and an out-of-batch chunk: one row
                    # per matching cid, foreign provenance ignored.
                    ["a", "b", "KNOWS", "desc", "fact", "A", "B", ["c1", "other"]],
                    [None, "b", "KNOWS", None, None, None, None, ["c1"]],  # skipped
                ]
            )
        )
        rows = await graph_store.get_relationships_for_chunks(["c1"])
        assert len(rows) == 1
        assert rows[0].chunk_id == "c1"
        assert rows[0].start_entity_id == "a"
        assert rows[0].rel_type == "KNOWS"
        cypher = mock_connection.query.call_args[0][0]
        assert "RELATES" in cypher and "source_chunk_ids" in cypher
        # Single edge scan per batch — no per-chunk UNWIND re-scan.
        assert "UNWIND" not in cypher

    async def test_get_relationships_one_row_per_matching_chunk(
        self, graph_store, mock_connection
    ):
        from unittest.mock import MagicMock

        mock_connection.query = AsyncMock(
            return_value=MagicMock(
                result_set=[
                    ["a", "b", "KNOWS", None, None, None, None, ["c1", "c2", "foreign"]],
                ]
            )
        )
        rows = await graph_store.get_relationships_for_chunks(["c1", "c2"])
        assert {(r.chunk_id, r.start_entity_id, r.end_entity_id) for r in rows} == {
            ("c1", "a", "b"),
            ("c2", "a", "b"),
        }

    async def test_empty_chunk_ids_no_query(self, graph_store, mock_connection):
        assert await graph_store.get_entities_mentioned_in_chunks([]) == []
        assert await graph_store.get_relationships_for_chunks([]) == []
        mock_connection.query.assert_not_called()


# ── Integration: real FalkorDB (RUN_INTEGRATION=1) ───────────────


def _first_n_chars_stable(part1: str, size: int) -> str:
    """Sanity helper: ensure part1 is exactly ``size`` chars so FixedSize
    chunking yields a byte-identical first chunk across updates."""
    assert len(part1) == size, f"part1 must be exactly {size} chars, got {len(part1)}"
    return part1


@pytest.mark.asyncio
@pytest.mark.integration
class TestCachedUpdateIntegration:
    """End-to-end proof against real FalkorDB. The scripted LLM is strict:
    an LLM call for an unchanged chunk raises, so passing tests ARE the
    proof that caching skips extraction."""

    CHUNK = 64

    def _texts(self):
        # part1 is padded to exactly CHUNK chars → chunk 1 is byte-identical
        # across both versions; only chunk 2 changes. FixedSizeChunking cuts
        # pure character windows without stripping, so padding is stable.
        part1 = _first_n_chars_stable(
            "Alice works at Acme Corporation with her colleague Bob today.".ljust(
                self.CHUNK, "|"
            ),
            self.CHUNK,
        )
        v1 = part1 + "Carol manages the Berlin office of Acme Corporation."
        v2 = part1 + "Dave now manages the Munich office of Acme Corporation."
        return v1, v2

    def _chunker(self):
        from graphrag_sdk.ingestion.chunking_strategies.fixed_size import (
            FixedSizeChunking,
        )

        return FixedSizeChunking(chunk_size=self.CHUNK, chunk_overlap=0)

    async def test_cached_update_skips_llm_for_unchanged_chunk(
        self, real_falkordb_rag_factory, scripted_llm
    ):
        from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
            ExactMatchResolution,
        )

        v1, v2 = self._texts()
        # Scripted responses: 2 for ingest (2 chunks), 1 for update (only
        # the changed chunk). strict=True → a 4th call raises.
        llm = scripted_llm(
            [
                ("Alice", "Person", "Engineer at Acme"),
                ("Acme Corporation", "Organization", "Tech company"),
            ],
            [("Carol", "Person", "Manager in Berlin")],
            [("Dave", "Person", "Manager in Munich")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=ExactMatchResolution())

        await rag.ingest(text=v1, document_id="doc-cache", chunker=self._chunker())

        result = await rag.update(
            text=v2,
            document_id="doc-cache",
            chunker=self._chunker(),
            cache_unchanged_chunks=True,
        )

        assert result.no_op is False
        assert result.metadata["cache_stats"] == {
            "cached_chunks": 1,
            "extracted_chunks": 1,
        }

        # Entity from the unchanged chunk survives; entity from the replaced
        # chunk is orphan-cleaned; new entity present.
        async def count(name):
            r = await rag._graph_store.query_raw(
                "MATCH (e:__Entity__) WHERE e.name = $n RETURN count(e)", {"n": name}
            )
            return r.result_set[0][0] if r.result_set else 0

        assert await count("Alice") == 1, "unchanged-chunk entity must survive"
        assert await count("Carol") == 0, "replaced-chunk entity must be orphan-cleaned"
        assert await count("Dave") == 1, "new-chunk entity must be extracted"

        # Provenance: Alice's source_chunk_ids must point only at LIVE chunks.
        r = await rag._graph_store.query_raw(
            "MATCH (e:__Entity__) WHERE e.name = 'Alice' RETURN e.source_chunk_ids"
        )
        alice_srcs = r.result_set[0][0] or []
        r = await rag._graph_store.query_raw(
            "MATCH (:Document {id: 'doc-cache'})-[:PART_OF]->(c:Chunk) RETURN collect(c.id)"
        )
        live_chunk_ids = set(r.result_set[0][0] or [])
        assert alice_srcs, "Alice must retain chunk provenance"
        assert set(alice_srcs) <= live_chunk_ids, (
            f"stale provenance survived the update: {set(alice_srcs) - live_chunk_ids}"
        )

        # Alice's MENTIONED_IN edge must target a live chunk.
        r = await rag._graph_store.query_raw(
            "MATCH (e:__Entity__ {name: 'Alice'})-[:MENTIONED_IN]->(c:Chunk) "
            "RETURN collect(c.id)"
        )
        assert set(r.result_set[0][0] or []) <= live_chunk_ids

    async def test_manually_deleted_entity_not_resurrected(
        self, real_falkordb_rag_factory, scripted_llm
    ):
        from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
            ExactMatchResolution,
        )

        v1, v2 = self._texts()
        llm = scripted_llm(
            [("Alice", "Person", "Engineer"), ("Bob", "Person", "Colleague")],
            [("Carol", "Person", "Manager")],
            [("Dave", "Person", "Manager")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=ExactMatchResolution())
        await rag.ingest(text=v1, document_id="doc-del", chunker=self._chunker())

        # Curate: manually delete Bob from the graph.
        await rag._graph_store.query_raw(
            "MATCH (e:__Entity__) WHERE e.name = 'Bob' DETACH DELETE e"
        )

        await rag.update(
            text=v2,
            document_id="doc-del",
            chunker=self._chunker(),
            cache_unchanged_chunks=True,
        )

        r = await rag._graph_store.query_raw(
            "MATCH (e:__Entity__) WHERE e.name = 'Bob' RETURN count(e)"
        )
        assert r.result_set[0][0] == 0, (
            "cached update must respect manual deletion, not resurrect Bob"
        )
        # But Alice (also from the unchanged chunk, not deleted) survives.
        r = await rag._graph_store.query_raw(
            "MATCH (e:__Entity__) WHERE e.name = 'Alice' RETURN count(e)"
        )
        assert r.result_set[0][0] == 1

    async def test_default_off_extracts_all_chunks(
        self, real_falkordb_rag_factory, scripted_llm
    ):
        """cache_unchanged_chunks defaults to False → both chunks hit the
        LLM on update (4 scripted responses, all consumed)."""
        from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
            ExactMatchResolution,
        )

        v1, v2 = self._texts()
        llm = scripted_llm(
            [("Alice", "Person", "Engineer")],
            [("Carol", "Person", "Manager")],
            [("Alice", "Person", "Engineer")],
            [("Dave", "Person", "Manager")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=ExactMatchResolution())
        await rag.ingest(text=v1, document_id="doc-off", chunker=self._chunker())

        result = await rag.update(text=v2, document_id="doc-off", chunker=self._chunker())

        assert "cache_stats" not in result.metadata
        r = await rag._graph_store.query_raw(
            "MATCH (e:__Entity__) WHERE e.name = 'Dave' RETURN count(e)"
        )
        assert r.result_set[0][0] == 1

    async def test_no_op_short_circuit_unaffected(
        self, real_falkordb_rag_factory, scripted_llm
    ):
        from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
            ExactMatchResolution,
        )

        v1, _ = self._texts()
        llm = scripted_llm(
            [("Alice", "Person", "Engineer")],
            [("Carol", "Person", "Manager")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=ExactMatchResolution())
        await rag.ingest(text=v1, document_id="doc-noop", chunker=self._chunker())

        result = await rag.update(
            text=v1,
            document_id="doc-noop",
            chunker=self._chunker(),
            cache_unchanged_chunks=True,
        )
        assert result.no_op is True
