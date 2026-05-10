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


def _resolver_param_ids():
    return ["ExactMatch", "Semantic"]


@pytest.fixture(params=_resolver_param_ids())
def resolver(request, embedder):
    """Direct parametrize over the two resolvers (replaces the old
    ``resolvers`` + ``resolver_index`` index dance). Each test that
    declares this fixture runs once per resolver."""
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
        ExactMatchResolution,
    )
    from graphrag_sdk.ingestion.resolution_strategies.semantic_resolution import (
        SemanticResolution,
    )
    if request.param == "ExactMatch":
        return ExactMatchResolution()
    return SemanticResolution(embedder=embedder)


async def _entity_count(rag, name: str) -> int:
    """Count :__Entity__ nodes by name.

    Used by integration tests with ``== 1`` assertions: a healthy graph
    has exactly one entity per unique name (ExactMatch dedupes by id;
    Semantic merges by similarity). A returned count >1 is a resolver
    bug surfacing — do not weaken to ``>= 1`` to make a flaky test pass.
    """
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

    async def test_shared_entity_preserved_when_one_doc_deleted(
        self, real_falkordb_rag_factory, scripted_llm, resolver
    ):
        """doc-A and doc-B both mention Alice. Deleting doc-A must not
        touch Alice — she's still referenced by doc-B's chunks.
        """
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

        # Sanity: pre-deletion, all three are in the graph exactly once.
        assert await _entity_count(rag, "Alice") == 1
        assert await _entity_count(rag, "Acme Corp") == 1
        assert await _entity_count(rag, "Bob") == 1

        await rag.delete_document("doc-A")

        # Alice is still mentioned by doc-B → preserved.
        assert await _entity_count(rag, "Alice") == 1, (
            "Shared entity must survive when one referencing doc is deleted "
            "— the load-bearing correctness property of v1.1.0."
        )
        # Acme Corp was only in doc-A → orphaned.
        assert await _entity_count(rag, "Acme Corp") == 0
        # Bob was only in doc-B → preserved.
        assert await _entity_count(rag, "Bob") == 1

    async def test_entity_orphaned_when_only_doc_deleted(
        self, real_falkordb_rag_factory, scripted_llm, resolver
    ):
        llm = scripted_llm(
            [("Bob", "Person", "Standalone entity")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)

        await rag.ingest(
            text="Bob is a person.", document_id="doc-A", resolver=resolver
        )
        assert await _entity_count(rag, "Bob") == 1

        await rag.delete_document("doc-A")

        assert await _entity_count(rag, "Bob") == 0, (
            "Entity must be orphan-deleted when its only referencing doc is removed."
        )

    async def test_entity_orphaned_when_update_drops_mention(
        self, real_falkordb_rag_factory, scripted_llm, resolver
    ):
        """Update replaces content. Entity from old content with no other
        references must be removed; entity introduced by new content present."""
        llm = scripted_llm(
            [("Carol", "Person", "Original entity")],
            [("Other", "Person", "Replacement entity")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)

        await rag.ingest(
            text="Carol is here.", document_id="doc-A", resolver=resolver
        )
        assert await _entity_count(rag, "Carol") == 1

        await rag.update(
            text="A different person named Other appears.",
            document_id="doc-A",
            resolver=resolver,
        )

        assert await _entity_count(rag, "Carol") == 0, (
            "Entity orphaned by an update must be removed."
        )
        assert await _entity_count(rag, "Other") == 1

    async def test_concurrent_updates_preserve_shared_entity(
        self, real_falkordb_rag_factory, scripted_llm, resolver
    ):
        """Tripwire for the pipeline-ordering invariant.

        doc-A and doc-B both mention Dave. Both are updated via
        ``asyncio.gather`` with new text that *also* mentions Dave.
        Dave must survive both updates — orphan cleanup on each side
        observes Dave's MENTIONED_IN from the other side's old chunks
        (pre-cutover) or new chunks (post-pipeline.run()).

        Note on "concurrent": this test exercises **asyncio-coroutine
        interleaving**, not necessarily DB-level parallelism — both
        updates share the same FalkorDB connection and the driver may
        serialize Cypher per-connection. The orphan-cleanup invariant
        the test protects is correct under either model, so the test
        is still meaningful, but a future driver change that exposes
        true parallelism could surface additional races this test
        wouldn't catch.

        If ``apply_changes(update_concurrency=…)`` is silently raised
        without re-verifying this invariant, OR if a future refactor
        defers MENTIONED_IN writes past pipeline.run() return, this
        test fails.
        """
        import asyncio

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
        assert await _entity_count(rag, "Dave") == 1

        # Now interleave two updates, both still mentioning Dave.
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

        assert await _entity_count(rag, "Dave") == 1, (
            "Coroutine-interleaved updates that both still mention a shared "
            "entity must NOT orphan-delete it. If this fails, either the "
            "pipeline step-8 ordering invariant has been broken or "
            "apply_changes update_concurrency was raised above 1 "
            "without re-verifying this property."
        )

    # ── D8: schema-mode coverage ──────────────────────────────────
    # The four invariant tests above all run with schema=None (open
    # mode). Schema-constrained extraction (with explicit entity types
    # and relation patterns) takes a different code path through the
    # pipeline's _prune step, so we add one schema-mode test that
    # exercises the same shared-entity-preserved invariant under that
    # configuration.

    async def test_shared_entity_preserved_under_schema_mode(
        self, real_falkordb_rag_factory, scripted_llm, resolver
    ):
        """Same invariant as test_shared_entity_preserved_when_one_doc_deleted,
        but with an explicit GraphSchema. Catches regressions where
        schema pruning interacts badly with the orphan-cleanup
        candidate snapshot."""
        from graphrag_sdk.core.models import (
            EntityType,
            GraphSchema,
            RelationType,
        )

        schema = GraphSchema(
            entities=[
                EntityType(label="Person", description="A human"),
                EntityType(label="Organization", description="A company"),
            ],
            relations=[
                RelationType(
                    label="WORKS_AT",
                    patterns=[("Person", "Organization")],
                ),
            ],
        )
        llm = scripted_llm(
            [("Alice", "Person", "Engineer at Acme"),
             ("Acme Corp", "Organization", "A tech company")],
            [("Alice", "Person", "Engineer at Acme"),
             ("Bob", "Person", "Colleague")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, schema=schema)

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
        assert await _entity_count(rag, "Alice") == 1

        await rag.delete_document("doc-A")

        # Same invariant under schema mode: Alice survives via doc-B.
        assert await _entity_count(rag, "Alice") == 1, (
            "Schema-mode orphan cleanup must preserve shared entities "
            "exactly like open mode."
        )
        assert await _entity_count(rag, "Acme Corp") == 0
        assert await _entity_count(rag, "Bob") == 1

    # ── D2: crash-recovery end-to-end ─────────────────────────────
    # Unit tests verify Phase 0's recovery branch with mocks. This is
    # the only test that verifies the actual claim the PR makes:
    # "FalkorDB persists the ready_to_commit marker across a connection
    # drop, and Phase 0 on next call replays the cutover."
    #
    # We can't truly kill the Python process inside a test, but we can
    # construct an equivalent post-crash state by hand:
    #   1. Ingest doc-A normally → reaches FINAL.
    #   2. Manually create a __pending__ Document for doc-A with new
    #      content under it, then call mark_pending_committed — this
    #      reproduces the exact persisted state of an update() that
    #      crashed between Phase 4 (commit) and Phase 5 (rollforward).
    #   3. Drop the GraphRAG (close the connection — simulates process
    #      death) and open a fresh one against the same graph_name.
    #   4. Call rag2.update(...) for doc-A. Phase 0 must detect the
    #      COMMITTED pending and roll it forward instead of either
    #      ignoring it (silent data loss) or starting a fresh update
    #      that competes with the leftover.

    async def test_committed_pending_replays_after_simulated_crash(
        self, real_falkordb_rag_factory, scripted_llm, resolver
    ):
        """Crash-safety end-to-end: a persisted COMMITTED marker must
        survive a connection drop and be rolled forward on next call."""
        from graphrag_sdk.api.main import GraphRAG
        from graphrag_sdk.core.connection import ConnectionConfig

        # Three LLM calls scripted: initial ingest of doc-A (1 call =
        # 2 responses for two-step extraction), and a follow-up update
        # of doc-A. The simulated-crash pending content is written via
        # direct Cypher (not pipeline), so no LLM scripting needed for
        # the pending itself.
        llm = scripted_llm(
            [("Original", "Person", "Initial entity")],
            [("Recovered", "Person", "Should appear after rollforward")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        graph_name = rag._conn.graph_name

        await rag.ingest(
            text="Original is here.", document_id="doc-A", resolver=resolver
        )
        assert await _entity_count(rag, "Original") == 1

        # ── Simulate the persisted state of a crashed update() between
        # Phase 4 and Phase 5: a __pending__ Document with chunks +
        # ready_to_commit=true, alongside the still-live original doc.
        pending_id = "doc-A__pending__sim00001"
        await rag._graph_store._conn.query(
            "CREATE (p:Document {id: $pid, path: 'doc-A', "
            "content_hash: 'sim-new-hash'})",
            {"pid": pending_id},
        )
        await rag._graph_store._conn.query(
            "MATCH (p:Document {id: $pid}) "
            "CREATE (p)-[:PART_OF]->(:Chunk {"
            "  id: 'sim-chunk-1', "
            "  text: 'Recovered content from rolled-forward pending'"
            "})",
            {"pid": pending_id},
        )
        # Flip the commit marker — this is the exact instant of "crash
        # AFTER commit, BEFORE rollforward".
        committed = await rag._graph_store.mark_pending_committed(pending_id)
        assert committed == 1

        # ── Drop everything (process death). The pending COMMITTED
        # state is now persisted in FalkorDB, awaiting recovery.
        await rag.close()

        # ── Fresh GraphRAG against the same graph_name (process restart).
        rag2 = GraphRAG(
            connection=ConnectionConfig(
                host=os.getenv("FALKOR_HOST", "localhost"),
                port=int(os.getenv("FALKOR_PORT", "6379")),
                username=os.getenv("FALKOR_USERNAME") or None,
                password=os.getenv("FALKOR_PASSWORD") or None,
                graph_name=graph_name,
            ),
            llm=llm,
            embedder=rag.embedder,
            embedding_dimension=rag.embedder.dimension,
        )
        try:
            # Calling update() on doc-A triggers Phase 0 → finds the
            # COMMITTED pending → rolls forward (without running our
            # update's pipeline, which would conflict).
            #
            # Note: this update will ITSELF run a pipeline after
            # Phase 0 completes the rollforward, so the final state
            # is the post-update content. The recovery is verified by
            # the absence of the original "Original" entity AND by
            # find_pending returning None at the end.
            await rag2.update(
                text="Final content after recovery and update.",
                document_id="doc-A",
                resolver=resolver,
            )

            # CRITICAL: the COMMITTED pending was rolled forward by
            # Phase 0 — there must be no leftover pending Document.
            still_pending = await rag2._graph_store.find_pending("doc-A")
            assert still_pending is None, (
                "Phase 0 must consume the COMMITTED pending. A leftover "
                "indicates either the recovery branch didn't fire or the "
                "rollforward didn't complete the rename — both are silent "
                "corruption."
            )
            # The original entity is gone (we updated past it).
            assert await _entity_count(rag2, "Original") == 0
        finally:
            try:
                await rag2._graph_store.delete_all()
            except Exception:
                pass
            try:
                await rag2.close()
            except Exception:
                pass
