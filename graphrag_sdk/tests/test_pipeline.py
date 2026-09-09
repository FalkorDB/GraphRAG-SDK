"""Tests for ingestion/pipeline.py — the sequential orchestrator."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import IngestionError
from graphrag_sdk.core.models import (
    DocumentInfo,
    DocumentOutput,
    Entity,
    GraphData,
    GraphNode,
    GraphRelationship,
    Ontology,
    Relation,
    ResolutionResult,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.ingestion.pipeline import IngestionPipeline
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy
from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy


# ── Stub strategies ─────────────────────────────────────────────


class StubLoader(LoaderStrategy):
    def __init__(self, text: str = "Test content for pipeline.") -> None:
        self._text = text

    async def load(self, source: str, ctx: Context) -> DocumentOutput:
        return DocumentOutput(
            text=self._text,
            document_info=DocumentInfo(path=source),
        )


class StubChunker(ChunkingStrategy):
    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        # Split by sentence
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        return TextChunks(
            chunks=[
                TextChunk(text=s, index=i, uid=f"chunk-{i}")
                for i, s in enumerate(sentences)
            ]
        )


class StubExtractor(ExtractionStrategy):
    async def extract(self, chunks, ontology, ctx):
        return GraphData(
            nodes=[GraphNode(id="e1", label="Entity", properties={"name": "Test"})],
            relationships=[],
        )


class StubResolver(ResolutionStrategy):
    async def resolve(self, graph_data, ctx):
        return ResolutionResult(
            nodes=graph_data.nodes,
            relationships=graph_data.relationships,
            merged_count=0,
        )


# ── Tests ───────────────────────────────────────────────────────


class TestIngestionPipeline:
    def _make_pipeline(
        self,
        mock_graph_store,
        mock_vector_store,
        text="Alice works at Acme Corp. Bob is her colleague.",
        ontology=None,
    ):
        return IngestionPipeline(
            loader=StubLoader(text),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            ontology=ontology or Ontology(),
        )

    async def test_full_run(self, ctx, mock_graph_store, mock_vector_store):
        pipeline = self._make_pipeline(mock_graph_store, mock_vector_store)
        result = await pipeline.run("test.txt", ctx)
        assert result.nodes_created >= 1
        assert result.chunks_indexed >= 1
        # Verify graph_store was called
        assert mock_graph_store.upsert_nodes.called
        assert mock_graph_store.upsert_relationships.called
        assert mock_vector_store.index_chunks.called

    async def test_run_with_text_param(self, ctx, mock_graph_store, mock_vector_store):
        """When text= is passed, loader is skipped."""
        pipeline = self._make_pipeline(mock_graph_store, mock_vector_store)
        result = await pipeline.run("ignored.txt", ctx, text="Direct text input.")
        assert result.chunks_indexed >= 1

    async def test_run_with_text_preserves_source_as_path(self, ctx, mock_graph_store, mock_vector_store):
        """When text= is passed without document_info, source is used as the path."""
        pipeline = self._make_pipeline(mock_graph_store, mock_vector_store)
        result = await pipeline.run("my_doc", ctx, text="Direct text input.")
        assert result.document_info.path == "my_doc"

    async def test_run_creates_lexical_graph(self, ctx, mock_graph_store, mock_vector_store):
        """Mandatory lexical graph creates Document + Chunk nodes + PART_OF rels."""
        pipeline = self._make_pipeline(mock_graph_store, mock_vector_store)
        await pipeline.run("test.txt", ctx)
        # upsert_nodes called multiple times: doc, chunks, extracted entities
        assert mock_graph_store.upsert_nodes.call_count >= 2
        # Check that rels include PART_OF
        rel_calls = mock_graph_store.upsert_relationships.call_args_list
        all_rels = []
        for call in rel_calls:
            all_rels.extend(call[0][0])
        part_of_rels = [r for r in all_rels if r.type == "PART_OF"]
        assert len(part_of_rels) > 0

    async def test_run_creates_next_chunk_links(self, ctx, mock_graph_store, mock_vector_store):
        """Pipeline creates NEXT_CHUNK between sequential chunks."""
        pipeline = self._make_pipeline(
            mock_graph_store, mock_vector_store,
            text="First. Second. Third.",
        )
        await pipeline.run("test.txt", ctx)
        rel_calls = mock_graph_store.upsert_relationships.call_args_list
        all_rels = []
        for call in rel_calls:
            all_rels.extend(call[0][0])
        next_chunk_rels = [r for r in all_rels if r.type == "NEXT_CHUNK"]
        assert len(next_chunk_rels) == 2  # 3 chunks → 2 NEXT_CHUNK

    async def test_empty_chunks_short_circuits(self, ctx, mock_graph_store, mock_vector_store):
        """If chunker produces nothing, pipeline returns early."""
        pipeline = self._make_pipeline(mock_graph_store, mock_vector_store, text="")
        result = await pipeline.run("empty.txt", ctx)
        assert result.nodes_created == 0
        assert result.chunks_indexed == 0

    async def test_default_context(self, mock_graph_store, mock_vector_store):
        pipeline = self._make_pipeline(mock_graph_store, mock_vector_store)
        result = await pipeline.run("test.txt")  # no ctx → default
        assert result.nodes_created >= 0

    async def test_schema_pruning(self, ctx, mock_graph_store, mock_vector_store):
        """Pruning keeps Unknown + valid rel_type, drops non-ontology labels/types."""
        ontology = Ontology(
            entities=[Entity(label="Person")],
            relations=[Relation(label="KNOWS")],
        )

        class ExtractorWithMixedLabels(ExtractionStrategy):
            async def extract(self, chunks, ontology, ctx):
                return GraphData(
                    nodes=[
                        GraphNode(id="p1", label="Person", properties={"name": "Alice"}),
                        GraphNode(id="x1", label="Unknown", properties={"name": "???"}),
                        GraphNode(id="a1", label="Alien", properties={"name": "Zorg"}),
                    ],
                    relationships=[
                        GraphRelationship(
                            start_node_id="p1", end_node_id="x1",
                            type="RELATES", properties={"rel_type": "KNOWS"},
                        ),
                        GraphRelationship(
                            start_node_id="p1", end_node_id="a1",
                            type="RELATES", properties={"rel_type": "WRONG"},
                        ),
                    ],
                )

        pipeline = IngestionPipeline(
            loader=StubLoader("Test"),
            chunker=StubChunker(),
            extractor=ExtractorWithMixedLabels(),
            resolver=StubResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            ontology=ontology,
        )
        result = await pipeline.run("test.txt", ctx)
        # Person + Unknown survive; Alien pruned
        assert result.nodes_created == 2
        # KNOWS survives; WRONG pruned (+ Alien endpoint gone)
        assert result.relationships_created == 1

    async def test_pipeline_wraps_exception(self, ctx, mock_graph_store, mock_vector_store, caplog):
        """Non-IngestionError exceptions get wrapped."""
        class FailingLoader(LoaderStrategy):
            async def load(self, source, ctx):
                raise RuntimeError("unexpected!")

        pipeline = IngestionPipeline(
            loader=FailingLoader(),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            ontology=Ontology(),
        )
        with caplog.at_level("ERROR", logger="graphrag_sdk.ingestion.pipeline"):
            with pytest.raises(IngestionError, match="Pipeline failed"):
                await pipeline.run("test.txt", ctx)
        assert "Pipeline failed with unexpected error" in caplog.text

    async def test_pipeline_writes_content_hash(self, ctx, mock_graph_store, mock_vector_store):
        """v1.1.0: Document node carries SHA-256 of the loaded text so
        ``GraphRAG.update()`` can short-circuit no-op updates without
        re-running extraction.
        """
        import hashlib

        text = "Stable content for hashing."
        expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        pipeline = self._make_pipeline(mock_graph_store, mock_vector_store, text=text)

        await pipeline.run("test.txt", ctx)

        # The Document node is written in step 3 (path + metadata) and the
        # content_hash is added by a second, final upsert once the run has
        # completed — so a half-finished ingest never carries a hash.
        calls = mock_graph_store.upsert_nodes.call_args_list
        doc_calls = [
            (i, n) for i, call in enumerate(calls) for n in call[0][0] if n.label == "Document"
        ]
        assert len(doc_calls) == 2, "expected the Document upsert, then the hash upsert"
        (i_first, first), (i_last, last) = doc_calls
        assert "content_hash" not in first.properties
        assert last.properties == {"content_hash": expected_hash}
        assert i_last == len(calls) - 1, "hash must be the last node write of the run"

    async def test_pipeline_uses_provided_document_info_uid(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        """v1.1.0 precursor: when ``document_info`` carries a custom uid,
        the Document node is anchored to that id rather than a fresh UUID.
        This is what makes ``update()`` and ``delete_document()`` work.
        """
        pipeline = self._make_pipeline(mock_graph_store, mock_vector_store)
        custom_info = DocumentInfo(uid="my-stable-id", path="docs/a.md")

        await pipeline.run("docs/a.md", ctx, document_info=custom_info)

        doc_nodes: list[GraphNode] = []
        for call in mock_graph_store.upsert_nodes.call_args_list:
            for n in call[0][0]:
                if n.label == "Document":
                    doc_nodes.append(n)
        # step-3 Document write + the end-of-run content_hash write
        assert len(doc_nodes) == 2
        assert {n.id for n in doc_nodes} == {"my-stable-id"}
        assert doc_nodes[0].properties.get("path") == "docs/a.md"

    async def test_pipeline_remaps_mentions_through_resolver_remap(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        """v1.1.0: when the resolver merges entities (returns a non-empty
        ``remap``), the pipeline rewrites ``graph_data.mentions`` so
        MENTIONED_IN edges target the survivor entity, not the merged-away
        id. Without this, fuzzy resolvers silently drop mention edges via
        MATCH-not-found in upsert_relationships, breaking orphan-cleanup
        correctness for update()/delete_document().
        """
        from graphrag_sdk.core.models import EntityMention, ResolutionResult

        class _MergingResolver(ResolutionStrategy):
            async def resolve(self, graph_data, ctx):
                # Pretend the extractor produced two distinct ids ``e_alice_a``
                # and ``e_alice_b`` for the same real entity, and the resolver
                # merged them into ``e_alice_canonical``.
                return ResolutionResult(
                    nodes=[
                        GraphNode(
                            id="e_alice_canonical",
                            label="Person",
                            properties={"name": "Alice"},
                        )
                    ],
                    relationships=[],
                    merged_count=2,
                    remap={
                        "e_alice_a": "e_alice_canonical",
                        "e_alice_b": "e_alice_canonical",
                    },
                )

        class _MentioningExtractor(ExtractionStrategy):
            async def extract(self, chunks, ontology, ctx):
                return GraphData(
                    nodes=[
                        GraphNode(id="e_alice_a", label="Person"),
                        GraphNode(id="e_alice_b", label="Person"),
                    ],
                    relationships=[],
                    mentions=[
                        EntityMention(entity_id="e_alice_a", chunk_id="chunk-0"),
                        EntityMention(entity_id="e_alice_b", chunk_id="chunk-0"),
                    ],
                )

        pipeline = IngestionPipeline(
            loader=StubLoader(),
            chunker=StubChunker(),
            extractor=_MentioningExtractor(),
            resolver=_MergingResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            ontology=Ontology(),
        )
        await pipeline.run("test.txt", ctx)

        # Find the MENTIONED_IN relationships that were upserted.
        mention_rels: list[GraphRelationship] = []
        for call in mock_graph_store.upsert_relationships.call_args_list:
            for r in call[0][0]:
                if r.type == "MENTIONED_IN":
                    mention_rels.append(r)

        # Both extracted ids merged → both mentions point at the survivor,
        # and the duplicate (canonical, chunk-0) pair is collapsed to one.
        assert len(mention_rels) == 1, (
            f"expected 1 deduplicated MENTIONED_IN, got {len(mention_rels)}"
        )
        assert mention_rels[0].start_node_id == "e_alice_canonical"
        assert mention_rels[0].end_node_id == "chunk-0"

    async def test_pipeline_remaps_mentions_through_chained_resolver_remap(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        """v1.1.0 follow-up: two-stage resolvers (SemanticResolution,
        LLMVerifiedResolution) merge per-phase remap dicts without
        flattening, so the combined dict can carry chains like
        ``{a: b, b: c}`` where ``b`` was itself merged away in a later
        phase. A single-hop ``remap.get(a)`` would point the mention at
        ``b`` — a non-existent node — and the MENTIONED_IN edge would
        silently fail to write. The fix follows each chain to its
        terminal survivor.

        This is an end-to-end check via the pipeline, mirroring the
        single-hop test above but with chained merges.
        """
        from graphrag_sdk.core.models import EntityMention, ResolutionResult

        class _ChainingResolver(ResolutionStrategy):
            async def resolve(self, graph_data, ctx):
                # Phase 1 merged a -> b; phase 2 merged b -> c. The
                # combined remap is left un-flattened on purpose to
                # reproduce the production code path.
                return ResolutionResult(
                    nodes=[
                        GraphNode(id="c", label="Person", properties={"name": "Alice"})
                    ],
                    relationships=[],
                    merged_count=2,
                    remap={"a": "b", "b": "c"},
                )

        class _ChainingExtractor(ExtractionStrategy):
            async def extract(self, chunks, ontology, ctx):
                # Mention points at the head of the chain (a). After
                # remap-following it must land on c, not b.
                return GraphData(
                    nodes=[
                        GraphNode(id="a", label="Person"),
                        GraphNode(id="b", label="Person"),
                        GraphNode(id="c", label="Person"),
                    ],
                    relationships=[],
                    mentions=[
                        EntityMention(entity_id="a", chunk_id="chunk-0"),
                        EntityMention(entity_id="b", chunk_id="chunk-0"),
                    ],
                )

        pipeline = IngestionPipeline(
            loader=StubLoader(),
            chunker=StubChunker(),
            extractor=_ChainingExtractor(),
            resolver=_ChainingResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            ontology=Ontology(),
        )
        await pipeline.run("test.txt", ctx)

        mention_rels: list[GraphRelationship] = []
        for call in mock_graph_store.upsert_relationships.call_args_list:
            for r in call[0][0]:
                if r.type == "MENTIONED_IN":
                    mention_rels.append(r)

        # Both mentions follow the chain to c, then dedupe on (c, chunk-0).
        assert len(mention_rels) == 1, (
            f"expected 1 chain-collapsed MENTIONED_IN, got {len(mention_rels)}"
        )
        assert mention_rels[0].start_node_id == "c", (
            "mention must follow the full chain a -> b -> c, not stop at b "
            "(b was merged away and writing to it silently MATCH-fails)"
        )
        assert mention_rels[0].end_node_id == "chunk-0"


class TestUnchangedReingestShortCircuit:
    """Re-ingesting a document whose content hash is already stored is a no-op.

    Stable chunk UIDs (Bug #12) made the chunk layer idempotent, but
    extraction still re-ran and, being LLM work, named a few entities
    differently each time: measured +3 to +16 entity nodes, +18 to +50
    RELATES and +30 MENTIONED_IN per re-ingest of one unchanged document.
    """

    def _pipeline(self, mock_graph_store, mock_vector_store, extractor):
        return IngestionPipeline(
            loader=StubLoader("Alice works at Acme Corp."),
            chunker=StubChunker(),
            extractor=extractor,
            resolver=StubResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            ontology=Ontology(),
        )

    @staticmethod
    def _stored(content_hash):
        from graphrag_sdk.core.models import DocumentRecord

        return AsyncMock(return_value=DocumentRecord(path="test.txt", content_hash=content_hash))

    async def test_identical_content_skips_extraction_and_writes(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        import hashlib

        extractor = StubExtractor()
        extractor.extract = AsyncMock(wraps=extractor.extract)
        digest = hashlib.sha256(b"Alice works at Acme Corp.").hexdigest()
        mock_graph_store.get_document_record = self._stored(digest)

        pipeline = self._pipeline(mock_graph_store, mock_vector_store, extractor)
        result = await pipeline.run("test.txt", ctx, document_info=DocumentInfo(uid="doc-1"))

        extractor.extract.assert_not_called()
        mock_graph_store.upsert_nodes.assert_not_called()
        mock_graph_store.upsert_relationships.assert_not_called()
        assert result.metadata["skipped_unchanged"] is True
        assert result.nodes_created == 0

    async def test_unchanged_document_never_reaches_the_chunker(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        """The check runs before step 2: contextual / semantic chunkers make a
        provider call per chunk, so a no-op must not chunk. ``chunks_indexed``
        reports the work actually done (none)."""
        import hashlib

        chunker = StubChunker()
        chunker.chunk_document = AsyncMock(wraps=chunker.chunk_document)
        digest = hashlib.sha256(b"Alice works at Acme Corp.").hexdigest()
        mock_graph_store.get_document_record = self._stored(digest)
        pipeline = self._pipeline(mock_graph_store, mock_vector_store, StubExtractor())
        pipeline.chunker = chunker

        result = await pipeline.run("test.txt", ctx, document_info=DocumentInfo(uid="doc-1"))

        chunker.chunk_document.assert_not_called()
        mock_vector_store.index_chunks.assert_not_called()
        assert result.metadata["skipped_unchanged"] is True
        assert result.chunks_indexed == 0

    async def test_skip_fires_without_caller_supplied_document_info(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        """Direct ``IngestionPipeline.run(source)`` — no facade, no
        ``document_info`` — must still recognise an unchanged document: the
        pipeline derives the Document id from the source path itself."""
        import hashlib

        extractor = StubExtractor()
        extractor.extract = AsyncMock(wraps=extractor.extract)
        digest = hashlib.sha256(b"Alice works at Acme Corp.").hexdigest()
        mock_graph_store.get_document_record = self._stored(digest)

        pipeline = self._pipeline(mock_graph_store, mock_vector_store, extractor)
        result = await pipeline.run("./docs/../test.txt", ctx)

        mock_graph_store.get_document_record.assert_awaited_once_with("test.txt")
        extractor.extract.assert_not_called()
        assert result.metadata["skipped_unchanged"] is True

    async def test_content_hash_is_written_only_after_a_successful_run(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        """A failed extraction must not leave a Document that looks complete,
        or the next ingest would skip it forever."""
        extractor = StubExtractor()
        extractor.extract = AsyncMock(side_effect=RuntimeError("provider down"))
        mock_graph_store.get_document_record = AsyncMock(return_value=None)
        pipeline = self._pipeline(mock_graph_store, mock_vector_store, extractor)

        from graphrag_sdk.core.exceptions import IngestionError

        with pytest.raises(IngestionError):
            await pipeline.run("test.txt", ctx, document_info=DocumentInfo(uid="doc-1"))

        written = [
            n
            for call in mock_graph_store.upsert_nodes.call_args_list
            for n in call[0][0]
            if n.label == "Document"
        ]
        assert written, "the Document node itself is still written in step 3"
        assert all("content_hash" not in n.properties for n in written)

    async def test_changed_content_takes_the_full_path(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        extractor = StubExtractor()
        extractor.extract = AsyncMock(wraps=extractor.extract)
        mock_graph_store.get_document_record = self._stored("0" * 64)

        pipeline = self._pipeline(mock_graph_store, mock_vector_store, extractor)
        result = await pipeline.run("test.txt", ctx, document_info=DocumentInfo(uid="doc-1"))

        extractor.extract.assert_called_once()
        assert "skipped_unchanged" not in result.metadata

    async def test_new_document_takes_the_full_path(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        extractor = StubExtractor()
        extractor.extract = AsyncMock(wraps=extractor.extract)
        mock_graph_store.get_document_record = AsyncMock(return_value=None)

        pipeline = self._pipeline(mock_graph_store, mock_vector_store, extractor)
        await pipeline.run("test.txt", ctx, document_info=DocumentInfo(uid="doc-1"))

        extractor.extract.assert_called_once()


class TestRemapMentionsUnit:
    """Direct unit tests for ``IngestionPipeline._remap_mentions`` covering
    the chain-following contract independently of the full pipeline."""

    def _gd(self, mentions):
        from graphrag_sdk.core.models import EntityMention

        return GraphData(
            nodes=[],
            relationships=[],
            mentions=[EntityMention(**m) for m in mentions],
        )

    def test_single_hop_remap(self):
        out = IngestionPipeline._remap_mentions(
            self._gd([{"entity_id": "a", "chunk_id": "k1"}]),
            {"a": "b"},
        )
        assert out.mentions[0].entity_id == "b"

    def test_chain_followed_to_terminal_survivor(self):
        """{a: b, b: c} — looking up a must yield c, not b."""
        out = IngestionPipeline._remap_mentions(
            self._gd([{"entity_id": "a", "chunk_id": "k1"}]),
            {"a": "b", "b": "c"},
        )
        assert out.mentions[0].entity_id == "c"

    def test_chain_dedupes_on_terminal_survivor(self):
        """Two mentions at different chain heads collapse to one edge if
        they end at the same survivor in the same chunk."""
        out = IngestionPipeline._remap_mentions(
            self._gd(
                [
                    {"entity_id": "a", "chunk_id": "k1"},
                    {"entity_id": "b", "chunk_id": "k1"},
                ]
            ),
            {"a": "b", "b": "c"},
        )
        assert len(out.mentions) == 1
        assert out.mentions[0].entity_id == "c"

    def test_cycle_terminates(self):
        """Malformed cyclic remap must not loop forever; visited-guard
        stops the traversal at the cycle entry."""
        out = IngestionPipeline._remap_mentions(
            self._gd([{"entity_id": "a", "chunk_id": "k1"}]),
            {"a": "b", "b": "a"},
        )
        # Acceptable terminal: either a or b (the loop breaks on
        # revisit). What matters is that the call returns rather than
        # spinning forever.
        assert out.mentions[0].entity_id in {"a", "b"}

    def test_unmapped_id_passes_through(self):
        """Mention pointing at an id not in the remap is written as-is."""
        out = IngestionPipeline._remap_mentions(
            self._gd([{"entity_id": "z", "chunk_id": "k1"}]),
            {"a": "b"},
        )
        assert out.mentions[0].entity_id == "z"


class TestPruneMethod:
    def test_prune_open_schema(self):
        """Empty ontology = open mode, nothing pruned."""
        pipeline = IngestionPipeline(
            loader=StubLoader(),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=MagicMock(),
            vector_store=MagicMock(),
            ontology=Ontology(),
        )
        data = GraphData(
            nodes=[GraphNode(id="a", label="Anything")],
            relationships=[
                GraphRelationship(start_node_id="a", end_node_id="a", type="SELF"),
            ],
        )
        result = pipeline._prune(data, Ontology())
        assert len(result.nodes) == 1
        assert len(result.relationships) == 1

    def test_prune_removes_invalid_labels(self):
        pipeline = IngestionPipeline(
            loader=StubLoader(),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=MagicMock(),
            vector_store=MagicMock(),
        )
        ontology = Ontology(entities=[Entity(label="Person")])
        data = GraphData(
            nodes=[
                GraphNode(id="p", label="Person"),
                GraphNode(id="x", label="Unknown"),
                GraphNode(id="z", label="Alien"),
            ],
            relationships=[],
        )
        result = pipeline._prune(data, ontology)
        # Person + Unknown survive; Alien pruned
        assert len(result.nodes) == 2
        labels = {n.label for n in result.nodes}
        assert labels == {"Person", "Unknown"}

    def test_prune_removes_orphaned_rels(self):
        pipeline = IngestionPipeline(
            loader=StubLoader(),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=MagicMock(),
            vector_store=MagicMock(),
        )
        ontology = Ontology(
            entities=[Entity(label="A")],
            relations=[Relation(label="LINK")],
        )
        data = GraphData(
            nodes=[
                GraphNode(id="a", label="A"),
                GraphNode(id="b", label="B"),  # will be pruned
            ],
            relationships=[
                GraphRelationship(start_node_id="a", end_node_id="b", type="LINK"),
            ],
        )
        result = pipeline._prune(data, ontology)
        assert len(result.nodes) == 1
        assert len(result.relationships) == 0  # rel removed because 'b' is pruned

    def test_prune_enforces_relation_patterns(self):
        pipeline = IngestionPipeline(
            loader=StubLoader(),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=MagicMock(),
            vector_store=MagicMock(),
        )
        ontology = Ontology(
            entities=[
                Entity(label="Person"),
                Entity(label="Company"),
            ],
            relations=[
                Relation(label="WORKS_AT", patterns=[("Person", "Company")]),
            ],
        )
        data = GraphData(
            nodes=[
                GraphNode(id="p", label="Person"),
                GraphNode(id="c", label="Company"),
            ],
            relationships=[
                GraphRelationship(
                    start_node_id="p", end_node_id="c",
                    type="RELATES", properties={"rel_type": "WORKS_AT"},
                ),
                GraphRelationship(
                    start_node_id="c", end_node_id="p",
                    type="RELATES", properties={"rel_type": "WORKS_AT"},
                ),
            ],
        )
        result = pipeline._prune(data, ontology)
        assert len(result.relationships) == 1
        assert result.relationships[0].start_node_id == "p"

    def test_prune_open_relation_patterns(self):
        """Relation with empty patterns allows any direction."""
        pipeline = IngestionPipeline(
            loader=StubLoader(),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=MagicMock(),
            vector_store=MagicMock(),
        )
        ontology = Ontology(
            entities=[Entity(label="Person")],
            relations=[Relation(label="KNOWS")],
        )
        data = GraphData(
            nodes=[
                GraphNode(id="a", label="Person"),
                GraphNode(id="b", label="Person"),
            ],
            relationships=[
                GraphRelationship(
                    start_node_id="a", end_node_id="b",
                    type="RELATES", properties={"rel_type": "KNOWS"},
                ),
            ],
        )
        result = pipeline._prune(data, ontology)
        assert len(result.relationships) == 1

    def test_prune_logs_pattern_mismatch_warning(self, caplog):
        """A2: pattern mismatches must emit a structured per-type warning."""
        import logging

        pipeline = IngestionPipeline(
            loader=StubLoader(),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=MagicMock(),
            vector_store=MagicMock(),
        )
        ontology = Ontology(
            entities=[Entity(label="Person"), Entity(label="Company")],
            relations=[
                Relation(label="WORKS_AT", patterns=[("Person", "Company")]),
            ],
        )
        # Three inverted (Company -> Person) extractions — triggers the warning.
        data = GraphData(
            nodes=[
                GraphNode(id="p", label="Person"),
                GraphNode(id="c", label="Company"),
            ],
            relationships=[
                GraphRelationship(
                    start_node_id="c", end_node_id="p",
                    type="RELATES", properties={"rel_type": "WORKS_AT"},
                )
                for _ in range(3)
            ],
        )
        with caplog.at_level(logging.WARNING, logger="graphrag_sdk.ingestion.pipeline"):
            result = pipeline._prune(data, ontology)

        assert len(result.relationships) == 0
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        msg = next((r.getMessage() for r in warnings if "WORKS_AT" in r.getMessage()), None)
        assert msg is not None
        # Names the offending pair, the declared pattern, and a hint
        assert "Company" in msg and "Person" in msg
        assert "[('Person', 'Company')]" in msg
        assert "inverted" in msg.lower()

    def test_prune_pattern_mismatch_sample_is_bounded(self, caplog):
        """A2: warning must sample, not flood, on large mismatch counts."""
        import logging

        pipeline = IngestionPipeline(
            loader=StubLoader(),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=MagicMock(),
            vector_store=MagicMock(),
        )
        ontology = Ontology(
            entities=[Entity(label="Person"), Entity(label="Company")],
            relations=[
                Relation(label="WORKS_AT", patterns=[("Person", "Company")]),
            ],
        )
        data = GraphData(
            nodes=[
                GraphNode(id="p", label="Person"),
                GraphNode(id="c", label="Company"),
            ],
            relationships=[
                GraphRelationship(
                    start_node_id="c", end_node_id="p",
                    type="RELATES", properties={"rel_type": "WORKS_AT"},
                )
                for _ in range(50)
            ],
        )
        with caplog.at_level(logging.WARNING, logger="graphrag_sdk.ingestion.pipeline"):
            pipeline._prune(data, ontology)

        msg = next(
            (r.getMessage() for r in caplog.records
             if r.levelno == logging.WARNING and "WORKS_AT" in r.getMessage()),
            None,
        )
        assert msg is not None
        # Total count is reported, but the sampled list does not contain 50 entries.
        assert "Pruned 50" in msg
        assert msg.count("('Company', 'Person')") <= 3


class TestReingestIdempotencyEndToEnd:
    """The behaviour the PR exists for, exercised through ``pipeline.run`` with
    the defaults a direct caller gets — no ``document_info``, loader-provided
    ``DocumentInfo`` with only ``path`` set."""

    def _pipeline(self, mock_graph_store, mock_vector_store, text):
        return IngestionPipeline(
            loader=StubLoader(text),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            ontology=Ontology(),
        )

    @staticmethod
    def _chunk_ids(mock_graph_store):
        return sorted(
            n.id
            for call in mock_graph_store.upsert_nodes.call_args_list
            for n in call[0][0]
            if n.label == "Chunk"
        )

    async def test_two_runs_of_the_same_file_write_the_same_chunk_ids(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        mock_graph_store.get_document_record = AsyncMock(return_value=None)
        text = "Alice works at Acme. Bob works at Beta. Carol runs Gamma."
        first = self._pipeline(mock_graph_store, mock_vector_store, text)
        await first.run("report.txt", ctx)
        ids_a = self._chunk_ids(mock_graph_store)
        doc_a = {n.id for c in mock_graph_store.upsert_nodes.call_args_list for n in c[0][0] if n.label == "Document"}

        mock_graph_store.upsert_nodes.reset_mock()
        second = self._pipeline(mock_graph_store, mock_vector_store, text)
        await second.run("report.txt", ctx)
        ids_b = self._chunk_ids(mock_graph_store)
        doc_b = {n.id for c in mock_graph_store.upsert_nodes.call_args_list for n in c[0][0] if n.label == "Document"}

        assert ids_a and ids_a == ids_b
        assert doc_a == doc_b == {"report.txt"}

    async def test_different_files_with_the_same_text_get_different_chunk_ids(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        mock_graph_store.get_document_record = AsyncMock(return_value=None)
        text = "Alice works at Acme. Bob works at Beta."
        await self._pipeline(mock_graph_store, mock_vector_store, text).run("a.txt", ctx)
        ids_a = self._chunk_ids(mock_graph_store)
        mock_graph_store.upsert_nodes.reset_mock()
        await self._pipeline(mock_graph_store, mock_vector_store, text).run("b.txt", ctx)
        ids_b = self._chunk_ids(mock_graph_store)
        assert ids_a and not set(ids_a) & set(ids_b)

    async def test_text_mode_without_document_info_is_stable_too(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        mock_graph_store.get_document_record = AsyncMock(return_value=None)
        p = self._pipeline(mock_graph_store, mock_vector_store, "unused")
        await p.run("ignored", ctx, text="Alice works at Acme. Bob works at Beta.")
        ids_a = self._chunk_ids(mock_graph_store)
        mock_graph_store.upsert_nodes.reset_mock()
        await p.run("ignored", ctx, text="Alice works at Acme. Bob works at Beta.")
        assert ids_a and ids_a == self._chunk_ids(mock_graph_store)


class TestDocumentInfoIdentityMerge:
    """A caller-supplied ``DocumentInfo`` is merged field by field, and only
    fields the caller actually set win. ``DocumentInfo.uid`` has a ``uuid4()``
    default, so ``DocumentInfo(path=...)`` must not smuggle a random id past
    the derived one (Copilot / CodeRabbit on #309)."""

    def _pipeline(self, mock_graph_store, mock_vector_store):
        mock_graph_store.get_document_record = AsyncMock(return_value=None)
        return IngestionPipeline(
            loader=StubLoader("Alice works at Acme. Bob works at Beta."),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            ontology=Ontology(),
        )

    async def test_path_only_document_info_takes_the_derived_id(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        p = self._pipeline(mock_graph_store, mock_vector_store)
        first = await p.run("./docs/../report.txt", ctx, document_info=DocumentInfo(path="docs/report.txt"))
        second = await p.run("./docs/../report.txt", ctx, document_info=DocumentInfo(path="docs/report.txt"))

        assert first.document_info.uid == second.document_info.uid == "report.txt"
        assert first.document_info.path == "docs/report.txt"

    async def test_explicit_uid_still_wins(self, ctx, mock_graph_store, mock_vector_store):
        p = self._pipeline(mock_graph_store, mock_vector_store)
        result = await p.run("report.txt", ctx, document_info=DocumentInfo(uid="pinned"))
        assert result.document_info.uid == "pinned"
        assert result.document_info.path == "report.txt", "path falls back to the loader's"

    async def test_text_mode_path_only_document_info_takes_the_text_hash(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        import hashlib

        p = self._pipeline(mock_graph_store, mock_vector_store)
        text = "Alice works at Acme."
        result = await p.run("label", ctx, text=text, document_info=DocumentInfo(path="label"))
        assert result.document_info.uid == f"text-{hashlib.sha256(text.encode()).hexdigest()[:16]}"

    async def test_metadata_is_merged_caller_over_loader(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        class MetaLoader(StubLoader):
            async def load(self, source, ctx):
                return DocumentOutput(
                    text=self._text,
                    document_info=DocumentInfo(path=source, metadata={"a": 1, "b": 1}),
                )

        p = self._pipeline(mock_graph_store, mock_vector_store)
        p.loader = MetaLoader("Alice works at Acme.")
        result = await p.run("r.txt", ctx, document_info=DocumentInfo(metadata={"b": 2, "c": 3}))
        assert result.document_info.metadata == {"a": 1, "b": 2, "c": 3}

    async def test_derived_id_with_pending_marker_is_rejected_before_any_io(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        """Direct pipeline callers bypass ``GraphRAG``'s guard; a file called
        ``foo__pending__bar.txt`` would otherwise be picked up by
        ``find_pending("foo")`` and rolled back as an interrupted update."""
        p = self._pipeline(mock_graph_store, mock_vector_store)
        p.loader.load = AsyncMock(wraps=p.loader.load)

        with pytest.raises(ValueError, match="reserved substring '__pending__'"):
            await p.run("foo__pending__bar.txt", ctx)

        p.loader.load.assert_not_called()
        mock_graph_store.upsert_nodes.assert_not_called()

    async def test_explicit_pending_id_is_still_accepted(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        """``GraphRAG.update()`` passes its pending id explicitly and must get through."""
        p = self._pipeline(mock_graph_store, mock_vector_store)
        result = await p.run("r.txt", ctx, document_info=DocumentInfo(uid="r.txt__pending__ab12cd34"))
        assert result.document_info.uid == "r.txt__pending__ab12cd34"


class TestDuckTypedGraphStore:
    """``graph_store`` is ``Any`` by design; a store that only implements the
    write surface the pipeline always required must still work
    (galshubeli on #309)."""

    async def test_store_without_get_document_record_ingests_without_the_skip(
        self, ctx, mock_vector_store
    ):
        class MinimalStore:
            def __init__(self):
                self.nodes = []
                self.rels = []

            async def upsert_nodes(self, nodes):
                self.nodes.extend(nodes)
                return len(nodes)

            async def upsert_relationships(self, rels):
                self.rels.extend(rels)
                return len(rels)

        store = MinimalStore()
        p = IngestionPipeline(
            loader=StubLoader("Alice works at Acme."),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=store,
            vector_store=mock_vector_store,
            ontology=Ontology(),
        )
        result = await p.run("r.txt", ctx)
        assert result.chunks_indexed == 1
        assert "skipped_unchanged" not in result.metadata
        assert any("content_hash" in n.properties for n in store.nodes if n.label == "Document")


class TestContentHashRequiresCompleteWrites:
    """``upsert_relationships`` and ``index_chunks`` swallow per-item failures
    and return a count. A run whose count came up short must not be stamped
    with ``content_hash``, or the missing edges / embeddings would never be
    repaired — every later ingest would skip the document (galshubeli on #309)."""

    def _pipeline(self, mock_graph_store, mock_vector_store, extractor=None):
        mock_graph_store.get_document_record = AsyncMock(return_value=None)
        return IngestionPipeline(
            loader=StubLoader("Alice works at Acme. Bob works at Beta. Carol runs Gamma."),
            chunker=StubChunker(),
            extractor=extractor or StubExtractor(),
            resolver=StubResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            ontology=Ontology(),
        )

    @staticmethod
    def _hash_written(mock_graph_store) -> bool:
        return any(
            "content_hash" in n.properties
            for call in mock_graph_store.upsert_nodes.call_args_list
            for n in call[0][0]
            if n.label == "Document"
        )

    async def test_complete_run_records_the_hash(self, ctx, mock_graph_store, mock_vector_store):
        result = await self._pipeline(mock_graph_store, mock_vector_store).run("r.txt", ctx)
        assert self._hash_written(mock_graph_store)
        assert "incomplete_writes" not in result.metadata

    async def test_short_relationship_count_withholds_the_hash(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        mock_graph_store.upsert_relationships = AsyncMock(side_effect=lambda rels: max(len(rels) - 1, 0))
        result = await self._pipeline(mock_graph_store, mock_vector_store).run("r.txt", ctx)

        assert not self._hash_written(mock_graph_store)
        assert any(s.startswith("lexical edges") for s in result.metadata["incomplete_writes"])
        # The run itself still reports what it attempted.
        assert result.chunks_indexed == 3

    async def test_short_mention_count_withholds_the_hash(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        from graphrag_sdk.core.models import EntityMention

        class MentionExtractor(StubExtractor):
            async def extract(self, chunks, ontology, ctx):
                return GraphData(
                    nodes=[GraphNode(id="e1", label="Entity", properties={"name": "Alice"})],
                    relationships=[],
                    mentions=[EntityMention(entity_id="e1", chunk_id=c.uid) for c in chunks.chunks],
                )

        def _drop_one_mention(rels):
            return len(rels) - 1 if rels and rels[0].type == "MENTIONED_IN" else len(rels)

        mock_graph_store.upsert_relationships = AsyncMock(side_effect=_drop_one_mention)
        result = await self._pipeline(
            mock_graph_store, mock_vector_store, extractor=MentionExtractor()
        ).run("r.txt", ctx)

        assert not self._hash_written(mock_graph_store)
        assert result.metadata["incomplete_writes"] == ["mentions 2/3"]

    async def test_unembedded_chunks_withhold_the_hash(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        """0 from ``index_chunks`` — every embedding call failed — means the
        document is absent from chunk vector search. (No embedder at all is
        ``None``, covered by ``test_no_embedder_is_not_a_shortfall``.)"""
        mock_vector_store.index_chunks = AsyncMock(return_value=0)
        result = await self._pipeline(mock_graph_store, mock_vector_store).run("r.txt", ctx)

        assert not self._hash_written(mock_graph_store)
        assert result.metadata["incomplete_writes"] == ["chunks indexed 0/3"]

    async def test_stores_that_report_nothing_are_taken_at_their_word(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        """Duck-typed stores returning ``None`` give no shortfall signal, so the
        pipeline must not refuse to ever mark such a run complete."""
        mock_graph_store.upsert_relationships = AsyncMock(return_value=None)
        mock_vector_store.index_chunks = AsyncMock(return_value=None)
        result = await self._pipeline(mock_graph_store, mock_vector_store).run("r.txt", ctx)

        assert self._hash_written(mock_graph_store)
        assert "incomplete_writes" not in result.metadata

    async def test_next_ingest_after_a_short_run_takes_the_full_path(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        """The point of withholding the hash: the follow-up ingest repairs."""
        from graphrag_sdk.core.models import DocumentRecord

        mock_vector_store.index_chunks = AsyncMock(return_value=0)
        p = self._pipeline(mock_graph_store, mock_vector_store)
        await p.run("r.txt", ctx)
        # The graph now holds the Document with no content_hash, as the run left it.
        mock_graph_store.get_document_record = AsyncMock(
            return_value=DocumentRecord(path="r.txt", content_hash=None)
        )
        mock_vector_store.index_chunks = AsyncMock(side_effect=lambda chunks: len(chunks.chunks))
        mock_graph_store.upsert_nodes.reset_mock()

        result = await p.run("r.txt", ctx)

        assert "skipped_unchanged" not in result.metadata
        assert self._hash_written(mock_graph_store)

    async def test_no_embedder_is_not_a_shortfall(self, ctx, mock_graph_store, mock_connection):
        """Graph-only deployments: ``VectorStore.index_chunks`` returns ``None``
        when no embedder is configured — nothing was attempted, so nothing is
        missing — and the hash must still be recorded or the unchanged-document
        skip would never fire for them (galshubeli on #309)."""
        from graphrag_sdk.storage.vector_store import VectorStore

        store = VectorStore(mock_connection, embedder=None)
        result = await self._pipeline(mock_graph_store, store).run("r.txt", ctx)

        assert self._hash_written(mock_graph_store)
        assert "incomplete_writes" not in result.metadata
        assert result.chunks_indexed == 3


class TestDeterministicChunkUids:
    """Bug #12 — re-ingesting an unchanged document duplicated its chunks.

    ``TextChunk.uid`` defaulted to ``uuid4()``, so the lexical graph's
    ``MERGE (c:Chunk {id: ...})`` never matched an existing chunk. Measured on
    the benchmark corpus, ingesting one unchanged file three times:
    17 -> 34 -> 51 Chunks and 171 -> 343 -> 513 MENTIONED_IN. After the fix the
    same three rounds hold at 17 Chunks / 170 MENTIONED_IN.
    """

    @staticmethod
    def _chunks(texts, doc_uid="doc-1"):
        from graphrag_sdk.core.models import DocumentInfo, TextChunk, TextChunks
        from graphrag_sdk.ingestion.pipeline import _assign_deterministic_chunk_uids

        info = DocumentInfo(uid=doc_uid, path="/tmp/a.txt")
        chunks = TextChunks(
            chunks=[TextChunk(text=t, index=i) for i, t in enumerate(texts)]
        )
        _assign_deterministic_chunk_uids(info, chunks)
        return [c.uid for c in chunks.chunks]

    def test_same_document_yields_same_uids(self):
        a = self._chunks(["alpha", "beta"])
        b = self._chunks(["alpha", "beta"])
        assert a == b

    def test_uids_replace_the_random_default(self):
        from graphrag_sdk.core.models import TextChunk

        assert self._chunks(["alpha"])[0] != TextChunk(text="alpha", index=0).uid

    def test_different_text_yields_different_uid(self):
        assert self._chunks(["alpha"]) != self._chunks(["alpha edited"])

    def test_different_index_yields_different_uid(self):
        """Two chunks with identical text must stay distinct nodes."""
        uids = self._chunks(["same", "same"])
        assert uids[0] != uids[1]

    def test_different_document_yields_different_uid(self):
        assert self._chunks(["alpha"], "doc-1") != self._chunks(["alpha"], "doc-2")

    def test_uids_are_unique_within_a_document(self):
        uids = self._chunks([f"chunk {i}" for i in range(50)])
        assert len(set(uids)) == 50

    def test_contextual_chunks_hash_the_original_text(self):
        """``ContextualChunking`` prepends an LLM summary; the id must come from
        ``metadata["original_chunk"]`` so a reworded summary keeps the id."""
        from graphrag_sdk.ingestion.pipeline import _assign_deterministic_chunk_uids

        def enriched(summary):
            chunks = TextChunks(
                chunks=[
                    TextChunk(
                        text=f"{summary}\n\nThe lighthouse was built in 1896.",
                        index=0,
                        metadata={"original_chunk": "The lighthouse was built in 1896."},
                    )
                ]
            )
            _assign_deterministic_chunk_uids(DocumentInfo(uid="doc-1"), chunks)
            return chunks.chunks[0].uid

        assert enriched("Context: about a lighthouse.") == enriched("Summary: a lighthouse's history.")
        plain = TextChunks(chunks=[TextChunk(text="The lighthouse was built in 1896.", index=0)])
        _assign_deterministic_chunk_uids(DocumentInfo(uid="doc-1"), plain)
        assert plain.chunks[0].uid == enriched("anything")


class TestStableDocumentId:
    def test_paths_are_normalised(self):
        from graphrag_sdk.core.models import stable_document_id

        assert stable_document_id("./docs/../a.md") == "a.md"
        assert stable_document_id("docs//a.md") == "docs/a.md"

    def test_uris_are_kept_verbatim(self):
        """``os.path.normpath`` would turn ``https://`` into ``https:/`` and
        resolve ``..`` inside a query string, merging distinct URLs."""
        from graphrag_sdk.core.models import stable_document_id

        for uri in (
            "https://example.test/doc?path=a/../b",
            "https://example.test/b",
            "s3://bucket/key//with/./dots",
            "file:///tmp/../etc/x",
        ):
            assert stable_document_id(uri) == uri
        assert stable_document_id("https://example.test/doc?path=a/../b") != stable_document_id(
            "https://example.test/b"
        )

    async def test_pipeline_uses_the_uri_verbatim_as_document_id(
        self, ctx, mock_graph_store, mock_vector_store
    ):
        mock_graph_store.get_document_record = AsyncMock(return_value=None)
        pipeline = IngestionPipeline(
            loader=StubLoader("Alice works at Acme."),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            ontology=Ontology(),
        )
        await pipeline.run("https://example.test/doc?path=a/../b", ctx)
        mock_graph_store.get_document_record.assert_awaited_once_with(
            "https://example.test/doc?path=a/../b"
        )
