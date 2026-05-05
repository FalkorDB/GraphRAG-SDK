"""Tests for ingestion/pipeline.py — the sequential orchestrator."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import IngestionError
from graphrag_sdk.core.models import (
    DocumentInfo,
    DocumentOutput,
    EntityType,
    GraphData,
    GraphNode,
    GraphRelationship,
    GraphSchema,
    RelationType,
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
    async def extract(self, chunks, schema, ctx):
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
        schema=None,
    ):
        return IngestionPipeline(
            loader=StubLoader(text),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            schema=schema or GraphSchema(),
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
        """Pruning keeps Unknown + valid rel_type, drops non-schema labels/types."""
        schema = GraphSchema(
            entities=[EntityType(label="Person")],
            relations=[RelationType(label="KNOWS")],
        )

        class ExtractorWithMixedLabels(ExtractionStrategy):
            async def extract(self, chunks, schema, ctx):
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
            schema=schema,
        )
        result = await pipeline.run("test.txt", ctx)
        # Person + Unknown survive; Alien pruned
        assert result.nodes_created == 2
        # KNOWS survives; WRONG pruned (+ Alien endpoint gone)
        assert result.relationships_created == 1

    async def test_pipeline_wraps_exception(self, ctx, mock_graph_store, mock_vector_store):
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
            schema=GraphSchema(),
        )
        with pytest.raises(IngestionError, match="Pipeline failed"):
            await pipeline.run("test.txt", ctx)

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

        # Find the Document node among all upsert_nodes calls.
        doc_nodes: list[GraphNode] = []
        for call in mock_graph_store.upsert_nodes.call_args_list:
            for n in call[0][0]:
                if n.label == "Document":
                    doc_nodes.append(n)
        assert len(doc_nodes) == 1, "expected exactly one Document upsert"
        assert doc_nodes[0].properties.get("content_hash") == expected_hash

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
        assert len(doc_nodes) == 1
        assert doc_nodes[0].id == "my-stable-id"
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
            async def extract(self, chunks, schema, ctx):
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
            schema=GraphSchema(),
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


class TestPruneMethod:
    def test_prune_open_schema(self):
        """Empty schema = open mode, nothing pruned."""
        pipeline = IngestionPipeline(
            loader=StubLoader(),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=MagicMock(),
            vector_store=MagicMock(),
            schema=GraphSchema(),
        )
        data = GraphData(
            nodes=[GraphNode(id="a", label="Anything")],
            relationships=[
                GraphRelationship(start_node_id="a", end_node_id="a", type="SELF"),
            ],
        )
        result = pipeline._prune(data, GraphSchema())
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
        schema = GraphSchema(entities=[EntityType(label="Person")])
        data = GraphData(
            nodes=[
                GraphNode(id="p", label="Person"),
                GraphNode(id="x", label="Unknown"),
                GraphNode(id="z", label="Alien"),
            ],
            relationships=[],
        )
        result = pipeline._prune(data, schema)
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
        schema = GraphSchema(
            entities=[EntityType(label="A")],
            relations=[RelationType(label="LINK")],
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
        result = pipeline._prune(data, schema)
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
        schema = GraphSchema(
            entities=[
                EntityType(label="Person"),
                EntityType(label="Company"),
            ],
            relations=[
                RelationType(label="WORKS_AT", patterns=[("Person", "Company")]),
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
        result = pipeline._prune(data, schema)
        assert len(result.relationships) == 1
        assert result.relationships[0].start_node_id == "p"

    def test_prune_open_relation_patterns(self):
        """RelationType with empty patterns allows any direction."""
        pipeline = IngestionPipeline(
            loader=StubLoader(),
            chunker=StubChunker(),
            extractor=StubExtractor(),
            resolver=StubResolver(),
            graph_store=MagicMock(),
            vector_store=MagicMock(),
        )
        schema = GraphSchema(
            entities=[EntityType(label="Person")],
            relations=[RelationType(label="KNOWS")],
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
        result = pipeline._prune(data, schema)
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
        schema = GraphSchema(
            entities=[EntityType(label="Person"), EntityType(label="Company")],
            relations=[
                RelationType(label="WORKS_AT", patterns=[("Person", "Company")]),
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
            result = pipeline._prune(data, schema)

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
        schema = GraphSchema(
            entities=[EntityType(label="Person"), EntityType(label="Company")],
            relations=[
                RelationType(label="WORKS_AT", patterns=[("Person", "Company")]),
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
            pipeline._prune(data, schema)

        msg = next(
            (r.getMessage() for r in caplog.records
             if r.levelno == logging.WARNING and "WORKS_AT" in r.getMessage()),
            None,
        )
        assert msg is not None
        # Total count is reported, but the sampled list does not contain 50 entries.
        assert "Pruned 50" in msg
        assert msg.count("('Company', 'Person')") <= 3
