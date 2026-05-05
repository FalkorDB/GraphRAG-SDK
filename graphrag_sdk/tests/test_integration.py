"""Integration tests — verify top-level imports and cross-module interactions."""
from __future__ import annotations

import pytest


class TestTopLevelImports:
    """Verify all public API exports are importable."""

    def test_version(self):
        import re

        from graphrag_sdk import __version__

        # Accept stable (1.0.0) and pre-release (1.0.0rc1, 1.0.0a1, 1.0.0b1) forms.
        assert re.match(r"^1\.\d+\.\d+(rc\d+|a\d+|b\d+)?$", __version__), (
            f"__version__ {__version__!r} does not look like a valid 1.x release"
        )

    def test_facade(self):
        from graphrag_sdk import GraphRAG
        assert GraphRAG is not None

    def test_core_models(self):
        from graphrag_sdk import (
            DataModel, GraphNode, GraphRelationship, TextChunk, TextChunks,
            DocumentInfo, DocumentOutput, EntityType, RelationType,
            GraphSchema, GraphData, ResolutionResult,
            RetrieverResult, RetrieverResultItem, RagResult, IngestionResult,
            SearchType,
        )
        # All should be importable
        assert DataModel is not None
        assert SearchType.VECTOR == "vector"

    def test_core_contracts(self):
        from graphrag_sdk import (
            Embedder, LLMInterface, Context, ConnectionConfig,
            FalkorDBConnection, GraphRAGError,
        )
        assert Embedder is not None
        assert LLMInterface is not None

    def test_strategy_abcs(self):
        from graphrag_sdk import (
            LoaderStrategy, ChunkingStrategy, ExtractionStrategy,
            ResolutionStrategy, RetrievalStrategy, RerankingStrategy,
        )
        assert LoaderStrategy is not None

    def test_pipeline(self):
        from graphrag_sdk import IngestionPipeline
        assert IngestionPipeline is not None

    def test_storage(self):
        from graphrag_sdk import GraphStore, VectorStore
        assert GraphStore is not None
        assert VectorStore is not None


class TestCrossCuttingConcerns:
    """Test interactions across module boundaries."""

    def test_context_flows_through_chunking(self):
        """Context is accepted by every strategy interface."""
        from graphrag_sdk.core.context import Context
        from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy
        import inspect
        sig = inspect.signature(ChunkingStrategy.chunk)
        assert "ctx" in sig.parameters

    def test_context_flows_through_extraction(self):
        from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
        import inspect
        sig = inspect.signature(ExtractionStrategy.extract)
        assert "ctx" in sig.parameters

    def test_context_flows_through_resolution(self):
        from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy
        import inspect
        sig = inspect.signature(ResolutionStrategy.resolve)
        assert "ctx" in sig.parameters

    def test_context_flows_through_retrieval(self):
        from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy
        import inspect
        sig = inspect.signature(RetrievalStrategy.search)
        assert "ctx" in sig.parameters

    def test_context_flows_through_reranking(self):
        from graphrag_sdk.retrieval.reranking_strategies.base import RerankingStrategy
        import inspect
        sig = inspect.signature(RerankingStrategy.rerank)
        assert "ctx" in sig.parameters


class TestSubmoduleImports:
    """Verify concrete implementations are importable."""

    def test_text_loader(self):
        from graphrag_sdk.ingestion.loaders.text_loader import TextLoader
        assert TextLoader is not None

    def test_pdf_loader(self):
        from graphrag_sdk.ingestion.loaders.pdf_loader import PdfLoader
        assert PdfLoader is not None

    def test_fixed_size_chunking(self):
        from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
        assert FixedSizeChunking is not None

    def test_graph_extraction(self):
        from graphrag_sdk.ingestion.extraction_strategies.graph_extraction import GraphExtraction
        assert GraphExtraction is not None

    def test_exact_match_resolution(self):
        from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution
        assert ExactMatchResolution is not None

    def test_local_retrieval(self):
        from graphrag_sdk.retrieval.strategies.local import LocalRetrieval
        assert LocalRetrieval is not None

    def test_semantic_router(self):
        from graphrag_sdk.retrieval.router import SemanticRouter
        assert SemanticRouter is not None

    def test_tracer(self):
        from graphrag_sdk.telemetry.tracer import Tracer, Span
        assert Tracer is not None
        assert Span is not None

    def test_graph_visualizer(self):
        from graphrag_sdk.utils.graph_viz import GraphVisualizer
        assert GraphVisualizer is not None


# ── v1.1.0 incremental-update invariants (real FalkorDB) ────────
#
# These tests are env-gated: they only run when ``RUN_INTEGRATION=1``
# is set. They drive ``update`` / ``delete_document`` / ``apply_changes``
# against a real FalkorDB and assert the load-bearing correctness
# property — scoped orphan cleanup preserves shared entities.
#
# Each test is parametrized across two resolvers:
#
#   - ``ExactMatchResolution`` — the default; mention IDs and node IDs
#     align without remapping.
#   - ``SemanticResolution(embedder=…)`` — fuzzy resolver. Tripwire for
#     the v1.1.0 mention-remap fix; without that fix, MENTIONED_IN
#     edges silently fail to write for any merged entity, breaking
#     orphan-cleanup correctness for fuzzy-resolver users.
#
# All assertions go through direct Cypher (``MATCH … RETURN count(e)``)
# rather than retrieval, to keep ranking variance out of the test.


def _resolver_params(embedder):
    """Return [ExactMatch, SemanticResolution] for parametrize."""
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
        ExactMatchResolution,
    )
    from graphrag_sdk.ingestion.resolution_strategies.semantic_resolution import (
        SemanticResolution,
    )
    return [
        pytest.param(
            ExactMatchResolution(),
            id="ExactMatchResolution",
        ),
        pytest.param(
            SemanticResolution(embedder=embedder),
            id="SemanticResolution",
        ),
    ]


async def _entity_count(rag, name: str) -> int:
    """Count :__Entity__ nodes by name. Used as the assertion primitive."""
    result = await rag._graph_store.query_raw(
        "MATCH (e:__Entity__) WHERE e.name = $name RETURN count(e) AS n",
        {"name": name},
    )
    if not result.result_set:
        return 0
    return result.result_set[0][0]


@pytest.mark.asyncio
class TestIncrementalUpdateInvariants:
    """v1.1.0 load-bearing correctness: scoped orphan cleanup preserves
    entities shared across documents. Run against real FalkorDB.

    Parametrized across resolver families to exercise both the default
    code path and the mention-remap path used by fuzzy resolvers.
    """

    @pytest.fixture
    def resolvers(self, embedder):
        # pytest can't directly parametrize a fixture inside a class via
        # a helper that takes an arg, so we expose the list and let
        # individual tests use it via the resolver_index parametrize.
        return _resolver_params(embedder)

    @pytest.mark.parametrize("resolver_index", [0, 1], ids=["ExactMatch", "Semantic"])
    async def test_shared_entity_preserved_when_one_doc_deleted(
        self, real_falkordb_rag_factory, scripted_llm, resolvers, resolver_index
    ):
        """doc-A and doc-B both mention Alice. Deleting doc-A must not
        touch Alice — she's still referenced by doc-B's chunks.
        """
        resolver = resolvers[resolver_index].values[0]
        # Two ingest calls → 2 entity sets scripted into the LLM.
        llm = scripted_llm(
            [("Alice", "Person", "Software engineer at Acme"),
             ("Acme Corp", "Organization", "A tech company")],
            [("Alice", "Person", "Software engineer at Acme"),
             ("Bob", "Person", "Alice's colleague")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)

        await rag.ingest(
            text="Alice works at Acme Corp.",
            document_id="doc-A",
            resolver=resolver,
        )
        await rag.ingest(
            text="Alice and Bob collaborate.",
            document_id="doc-B",
            resolver=resolver,
        )

        # Sanity: pre-deletion, all three are in the graph.
        assert await _entity_count(rag, "Alice") >= 1
        assert await _entity_count(rag, "Acme Corp") >= 1
        assert await _entity_count(rag, "Bob") >= 1

        await rag.delete_document("doc-A")

        # Alice is still mentioned by doc-B → preserved.
        assert await _entity_count(rag, "Alice") >= 1, (
            "Shared entity must survive when one referencing doc is deleted "
            "— the load-bearing correctness property of v1.1.0."
        )
        # Acme Corp was only in doc-A → orphaned.
        assert await _entity_count(rag, "Acme Corp") == 0
        # Bob was only in doc-B → preserved.
        assert await _entity_count(rag, "Bob") >= 1

    @pytest.mark.parametrize("resolver_index", [0, 1], ids=["ExactMatch", "Semantic"])
    async def test_entity_orphaned_when_only_doc_deleted(
        self, real_falkordb_rag_factory, scripted_llm, resolvers, resolver_index
    ):
        resolver = resolvers[resolver_index].values[0]
        llm = scripted_llm(
            [("Bob", "Person", "Standalone entity")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)

        await rag.ingest(
            text="Bob is a person.", document_id="doc-A", resolver=resolver
        )
        assert await _entity_count(rag, "Bob") >= 1

        await rag.delete_document("doc-A")

        assert await _entity_count(rag, "Bob") == 0, (
            "Entity must be orphan-deleted when its only referencing doc is removed."
        )

    @pytest.mark.parametrize("resolver_index", [0, 1], ids=["ExactMatch", "Semantic"])
    async def test_entity_orphaned_when_update_drops_mention(
        self, real_falkordb_rag_factory, scripted_llm, resolvers, resolver_index
    ):
        """Update replaces content. Entity from old content with no other
        references must be removed; entity introduced by new content present."""
        resolver = resolvers[resolver_index].values[0]
        llm = scripted_llm(
            [("Carol", "Person", "Original entity")],
            [("Other", "Person", "Replacement entity")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)

        await rag.ingest(
            text="Carol is here.", document_id="doc-A", resolver=resolver
        )
        assert await _entity_count(rag, "Carol") >= 1

        await rag.update(
            text="A different person named Other appears.",
            document_id="doc-A",
            resolver=resolver,
        )

        assert await _entity_count(rag, "Carol") == 0, (
            "Entity orphaned by an update must be removed."
        )
        assert await _entity_count(rag, "Other") >= 1

    @pytest.mark.parametrize("resolver_index", [0, 1], ids=["ExactMatch", "Semantic"])
    async def test_concurrent_updates_preserve_shared_entity(
        self, real_falkordb_rag_factory, scripted_llm, resolvers, resolver_index
    ):
        """Tripwire for the pipeline-ordering invariant.

        doc-A and doc-B both mention Dave. Both are concurrently updated
        with new text that *also* mentions Dave. Dave must survive both
        updates — orphan cleanup on each side observes Dave's
        MENTIONED_IN from the other side's old chunks (pre-cutover) or
        new chunks (post-pipeline.run()).

        If ``apply_changes(update_concurrency=…)`` is silently raised
        without re-verifying this invariant, OR if a future refactor
        defers MENTIONED_IN writes past pipeline.run() return, this
        test fails.
        """
        import asyncio

        resolver = resolvers[resolver_index].values[0]
        llm = scripted_llm(
            [("Dave", "Person", "Shared by both docs"),
             ("Xena", "Person", "doc-A only")],   # initial doc-A
            [("Dave", "Person", "Shared by both docs"),
             ("Yara", "Person", "doc-B only")],   # initial doc-B
            [("Dave", "Person", "Still in updated A"),
             ("Xena", "Person", "Still in A")],   # updated doc-A
            [("Dave", "Person", "Still in updated B"),
             ("Yara", "Person", "Still in B")],   # updated doc-B
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)

        await rag.ingest(
            text="Dave knows Xena.", document_id="doc-A", resolver=resolver
        )
        await rag.ingest(
            text="Dave likes Yara.", document_id="doc-B", resolver=resolver
        )
        assert await _entity_count(rag, "Dave") >= 1

        # Now run concurrent updates, both still mentioning Dave.
        await asyncio.gather(
            rag.update(
                text="Dave still knows Xena, but more.",
                document_id="doc-A",
                resolver=resolver,
            ),
            rag.update(
                text="Dave still likes Yara, in a new way.",
                document_id="doc-B",
                resolver=resolver,
            ),
        )

        assert await _entity_count(rag, "Dave") >= 1, (
            "Concurrent updates that both still mention a shared entity "
            "must NOT orphan-delete it. If this fails, either the "
            "pipeline step-8 ordering invariant has been broken or "
            "apply_changes update_concurrency was raised above 1 "
            "without re-verifying this property."
        )
