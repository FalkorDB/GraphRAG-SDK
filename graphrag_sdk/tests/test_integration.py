"""Integration tests — verify top-level imports and cross-module interactions."""

from __future__ import annotations

import os

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
            DataModel,
            SearchType,
        )

        # All should be importable
        assert DataModel is not None
        assert SearchType.VECTOR == "vector"

    def test_core_contracts(self):
        from graphrag_sdk import (
            Embedder,
            LLMInterface,
        )

        assert Embedder is not None
        assert LLMInterface is not None

    def test_strategy_abcs(self):
        from graphrag_sdk import (
            LoaderStrategy,
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
        import inspect

        from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy

        sig = inspect.signature(ChunkingStrategy.chunk)
        assert "ctx" in sig.parameters

    def test_context_flows_through_extraction(self):
        import inspect

        from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy

        sig = inspect.signature(ExtractionStrategy.extract)
        assert "ctx" in sig.parameters

    def test_context_flows_through_resolution(self):
        import inspect

        from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy

        sig = inspect.signature(ResolutionStrategy.resolve)
        assert "ctx" in sig.parameters

    def test_context_flows_through_retrieval(self):
        import inspect

        from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy

        sig = inspect.signature(RetrievalStrategy.search)
        assert "ctx" in sig.parameters

    def test_context_flows_through_reranking(self):
        import inspect

        from graphrag_sdk.retrieval.reranking_strategies.base import RerankingStrategy

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
        from graphrag_sdk.telemetry.tracer import Span, Tracer

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
@pytest.mark.integration
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
            [
                ("Alice", "Person", "Software engineer at Acme"),
                ("Acme Corp", "Organization", "A tech company"),
            ],
            [
                ("Alice", "Person", "Software engineer at Acme"),
                ("Bob", "Person", "Alice's colleague"),
            ],
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

        await rag.ingest(text="Bob is a person.", document_id="doc-A", resolver=resolver)
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

        await rag.ingest(text="Carol is here.", document_id="doc-A", resolver=resolver)
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
            [
                ("Dave", "Person", "Shared by both docs"),
                ("Xena", "Person", "doc-A only"),
            ],  # initial doc-A
            [
                ("Dave", "Person", "Shared by both docs"),
                ("Yara", "Person", "doc-B only"),
            ],  # initial doc-B
            [
                ("Dave", "Person", "Still in updated A"),
                ("Xena", "Person", "Still in A"),
            ],  # updated doc-A
            [
                ("Dave", "Person", "Still in updated B"),
                ("Yara", "Person", "Still in B"),
            ],  # updated doc-B
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)

        await rag.ingest(text="Dave knows Xena.", document_id="doc-A", resolver=resolver)
        await rag.ingest(text="Dave likes Yara.", document_id="doc-B", resolver=resolver)
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

    # ── D8: ontology-mode coverage ──────────────────────────────────
    # The four invariant tests above all run with ontology=None (open
    # mode). Schema-constrained extraction (with explicit entity types
    # and relation patterns) takes a different code path through the
    # pipeline's _prune step, so we add one ontology-mode test that
    # exercises the same shared-entity-preserved invariant under that
    # configuration.

    async def test_shared_entity_preserved_under_schema_mode(
        self, real_falkordb_rag_factory, scripted_llm, resolver
    ):
        """Same invariant as test_shared_entity_preserved_when_one_doc_deleted,
        but with an explicit Ontology. Catches regressions where
        ontology pruning interacts badly with the orphan-cleanup
        candidate snapshot."""
        from graphrag_sdk.core.models import (
            Entity,
            Ontology,
            Relation,
        )

        ontology = Ontology(
            entities=[
                Entity(label="Person", description="A human"),
                Entity(label="Organization", description="A company"),
            ],
            relations=[
                Relation(
                    label="WORKS_AT",
                    patterns=[("Person", "Organization")],
                ),
            ],
        )
        llm = scripted_llm(
            [
                ("Alice", "Person", "Engineer at Acme"),
                ("Acme Corp", "Organization", "A tech company"),
            ],
            [("Alice", "Person", "Engineer at Acme"), ("Bob", "Person", "Colleague")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)

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

        # Same invariant under ontology mode: Alice survives via doc-B.
        assert await _entity_count(rag, "Alice") == 1, (
            "Schema-mode orphan cleanup must preserve shared entities exactly like open mode."
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
        graph_name = rag._conn.config.graph_name

        await rag.ingest(text="Original is here.", document_id="doc-A", resolver=resolver)
        assert await _entity_count(rag, "Original") == 1

        # ── Simulate the persisted state of a crashed update() between
        # Phase 4 and Phase 5: a __pending__ Document with chunks +
        # ready_to_commit=true, alongside the still-live original doc.
        pending_id = "doc-A__pending__sim00001"
        await rag._graph_store._conn.query(
            "CREATE (p:Document {id: $pid, path: 'doc-A', content_hash: 'sim-new-hash'})",
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

    # ── Crash-safe orphan + stale-RELATES cleanup (post-fix) ──────
    # These tests pin the invariants Naseem's PR #247 review flagged:
    # (1) orphan cleanup survives a crash between commit and cleanup;
    # (2) RELATES facts whose only provenance is the updated/deleted
    #     doc's chunks are removed even when endpoint entities live on;
    # (3) deduplicator's RELATES remap unions source_chunk_ids rather
    #     than overwriting (otherwise the cleanup in (2) silently
    #     under-cleans after any dedup pass).

    async def test_orphans_cleaned_after_crash_between_commit_and_cleanup(
        self, real_falkordb_rag_factory, scripted_llm, resolver
    ):
        """Crash after Phase 4 (commit marker set) but before Phase 6
        (orphan cleanup). On next call, Phase 0's COMMITTED-pending
        rollforward must also run the cleanup — otherwise orphan
        entities persist forever (the pre-fix bug)."""
        from graphrag_sdk.api.main import GraphRAG
        from graphrag_sdk.core.connection import ConnectionConfig

        llm = scripted_llm(
            [("Ghost", "Person", "Will be orphaned by simulated crash")],
            [("Survivor", "Person", "Replacement after recovery")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        graph_name = rag._conn.config.graph_name

        await rag.ingest(text="Ghost lives here.", document_id="doc-A", resolver=resolver)
        assert await _entity_count(rag, "Ghost") == 1

        # Simulate the persisted state of an update() that committed but
        # crashed before Phase 5/6: a __pending__ Document with chunks +
        # ready_to_commit=true AND cleanup_candidates persisted.
        pending_id = "doc-A__pending__crash001"
        candidate_ids = [
            row[0]
            for row in (
                await rag._graph_store._conn.query(
                    "MATCH (e:__Entity__)-[:MENTIONED_IN]->(:Chunk)<-[:PART_OF]-"
                    "(:Document {id: 'doc-A'}) RETURN DISTINCT e.id AS eid",
                    {},
                )
            ).result_set
        ]
        old_chunk_ids = [
            row[0]
            for row in (
                await rag._graph_store._conn.query(
                    "MATCH (:Document {id: 'doc-A'})-[:PART_OF]->(c:Chunk) RETURN c.id AS cid",
                    {},
                )
            ).result_set
        ]
        assert candidate_ids, "test setup: doc-A should have entity candidates"
        await rag._graph_store._conn.query(
            "CREATE (p:Document {id: $pid, path: 'doc-A', content_hash: 'h'})",
            {"pid": pending_id},
        )
        await rag._graph_store._conn.query(
            "MATCH (p:Document {id: $pid}) "
            "CREATE (p)-[:PART_OF]->(:Chunk {id: 'crash-chunk-1', text: 'new'})",
            {"pid": pending_id},
        )
        await rag._graph_store.set_pending_cleanup_state(pending_id, candidate_ids, old_chunk_ids)
        committed = await rag._graph_store.mark_pending_committed(pending_id)
        assert committed == 1

        await rag.close()

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
            # Trigger Phase 0 via update(). Recovery must:
            # 1. Roll forward the COMMITTED pending → doc-A canonical content
            #    is the simulated-crash chunk.
            # 2. Run post-cutover cleanup → Ghost is orphan-deleted because
            #    it had no MENTIONED_IN edge from the crash-chunk.
            # Then the fresh update overwrites doc-A with Survivor content.
            await rag2.update(
                text="Survivor replaces everything.",
                document_id="doc-A",
                resolver=resolver,
            )
            assert await _entity_count(rag2, "Ghost") == 0, (
                "Phase 0 recovery must run orphan cleanup after rolling "
                "forward the committed pending. Pre-fix, candidate ids "
                "lived only in Python memory and were lost on crash — "
                "leaving Ghost permanently orphaned."
            )
        finally:
            try:
                await rag2._graph_store.delete_all()
            except Exception:
                pass
            try:
                await rag2.close()
            except Exception:
                pass

    async def test_orphans_cleaned_after_crash_between_rollforward_and_cleanup(
        self, real_falkordb_rag_factory, scripted_llm, resolver
    ):
        """Crash AFTER rollforward_cutover but before _run_post_cutover_cleanup.
        The canonical Document survives with cleanup_candidates set; Phase 0's
        CLEANUP_PENDING branch (the no-pending-but-state-present case) must
        resume the cleanup."""
        from graphrag_sdk.api.main import GraphRAG
        from graphrag_sdk.core.connection import ConnectionConfig

        llm = scripted_llm(
            [("Stale", "Person", "Should be cleaned up post-rollforward")],
            [("Fresh", "Person", "Replacement")],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        graph_name = rag._conn.config.graph_name

        await rag.ingest(text="Stale was here.", document_id="doc-B", resolver=resolver)
        assert await _entity_count(rag, "Stale") == 1

        # Snapshot what cleanup would do, delete chunks (simulating
        # rollforward), then attach cleanup state to the doc — mimicking
        # "rollforward succeeded but cleanup never ran."
        candidate_ids = await rag._graph_store.get_document_entity_candidates("doc-B")
        old_chunk_ids = await rag._graph_store.get_document_chunk_ids("doc-B")
        assert candidate_ids, "test setup: doc-B should have candidates"
        # Delete the old chunks (this is what rollforward_cutover's
        # step 1+2 does).
        await rag._graph_store.delete_document_chunks("doc-B")
        # Stash cleanup state on the live doc (what set_pending_cleanup_state
        # would have written + rollforward renamed onto the canonical id).
        await rag._graph_store.set_pending_cleanup_state("doc-B", candidate_ids, old_chunk_ids)

        await rag.close()

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
            # delete_document also runs Phase 0 first. Either route must
            # consume the leftover cleanup state. Use update() so the
            # final state is well-defined and we can assert Stale is gone.
            await rag2.update(
                text="Fresh is here now.",
                document_id="doc-B",
                resolver=resolver,
                if_missing="ingest",
            )
            assert await _entity_count(rag2, "Stale") == 0, (
                "CLEANUP_PENDING recovery branch must run when a live "
                "doc has cleanup_candidates set but no pending — i.e. "
                "the crash happened between rollforward and cleanup."
            )
            state = await rag2._graph_store.get_cleanup_state("doc-B")
            assert state is None, (
                "cleanup state must be cleared after recovery — otherwise "
                "every subsequent call would re-run the cleanup."
            )
        finally:
            try:
                await rag2._graph_store.delete_all()
            except Exception:
                pass
            try:
                await rag2.close()
            except Exception:
                pass

    async def test_stale_relates_removed_when_shared_entity_doc_updated(
        self, real_falkordb_rag_factory, scripted_llm, resolver
    ):
        """Two docs both mention Alice and Bob, but only doc-A asserts
        ``Alice KNOWS Bob``. Updating doc-A to drop that fact must
        remove the RELATES edge even though Alice and Bob remain
        (still mentioned by doc-B). Pre-fix, the edge would survive
        because delete_orphan_entities only touches nodes."""
        import json

        from .conftest import MockLLM

        # Hand-script the verify+rels (step 2) response per source.
        # Step 1 NER is handled locally by GLiNER (default extractor),
        # so each ingest/update consumes exactly ONE MockLLM response.
        # Default `scripted_llm` produces entities-only responses; this
        # test needs an explicit RELATES on the first doc.
        def _step2(entities, relationships=()):
            return json.dumps(
                {
                    "entities": [{"name": n, "type": t, "description": d} for n, t, d in entities],
                    "relationships": list(relationships),
                }
            )

        entities = [
            ("Alice", "Person", "p1"),
            ("Bob", "Person", "p2"),
        ]
        doc_a = _step2(
            entities,
            [
                {
                    "source": "Alice",
                    "target": "Bob",
                    "type": "KNOWS",
                    "description": "Alice knows Bob",
                    "keywords": "social",
                    "weight": 0.9,
                }
            ],
        )
        doc_b = _step2(entities)
        doc_a_update = _step2(entities)  # update drops the relationship
        llm = MockLLM(responses=[doc_a, doc_b, doc_a_update], strict=True)
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)

        await rag.ingest(text="Alice knows Bob.", document_id="doc-A", resolver=resolver)
        await rag.ingest(text="Alice and Bob exist.", document_id="doc-B", resolver=resolver)

        # Pre-update: one RELATES edge between Alice and Bob.
        r = await rag._graph_store._conn.query(
            "MATCH (a:__Entity__ {name: 'Alice'})-[r:RELATES]-(b:__Entity__ {name: 'Bob'}) "
            "RETURN count(r) AS n",
            {},
        )
        assert r.result_set[0][0] == 1, "test setup: expected one RELATES edge from doc-A"

        await rag.update(
            text="Alice and Bob are mentioned but no longer related.",
            document_id="doc-A",
            resolver=resolver,
        )

        # Both entities preserved (doc-B still mentions them).
        assert await _entity_count(rag, "Alice") == 1
        assert await _entity_count(rag, "Bob") == 1
        # But the stale RELATES is gone — its only provenance was
        # doc-A's old chunks, which the cutover removed.
        r = await rag._graph_store._conn.query(
            "MATCH (a:__Entity__ {name: 'Alice'})-[r:RELATES]-(b:__Entity__ {name: 'Bob'}) "
            "RETURN count(r) AS n",
            {},
        )
        assert r.result_set[0][0] == 0, (
            "RELATES edge whose only source_chunk_ids were doc-A's old chunks "
            "must be deleted by post-cutover stale-RELATES cleanup. Pre-fix, "
            "delete_orphan_entities only touched nodes — the edge survived."
        )

    async def test_dedup_remap_unions_source_chunk_ids(self, real_falkordb_rag_factory, embedder):
        """When deduplicator merges two entities, the survivor's RELATES
        edge ``source_chunk_ids`` provenance must UNION the duplicate's
        contribution rather than overwrite it. Pre-fix, ``SET nr +=
        properties(r)`` clobbered the survivor's list, breaking the
        stale-RELATES cleanup invariant on any subsequent update/delete.
        """
        from graphrag_sdk.storage.deduplicator import EntityDeduplicator

        rag = real_falkordb_rag_factory(
            llm=None,  # dedup doesn't call the LLM
            resolver=None,
        )
        conn = rag._graph_store._conn

        # Hand-construct two entities with the same normalized name+label
        # (so dedup will merge them) and outgoing RELATES to a common
        # target — each edge with a distinct source_chunk_ids list.
        await conn.query(
            "CREATE (a1:__Entity__:Person {id: 'alice-1', name: 'Alice', "
            "description: 'longer description that wins survivor'})",
            {},
        )
        await conn.query(
            "CREATE (a2:__Entity__:Person {id: 'alice-2', name: 'Alice', description: 'short'})",
            {},
        )
        await conn.query(
            "CREATE (b:__Entity__:Person {id: 'bob', name: 'Bob', description: ''})",
            {},
        )
        await conn.query(
            "MATCH (a:__Entity__ {id: 'alice-1'}), (b:__Entity__ {id: 'bob'}) "
            "CREATE (a)-[:RELATES {source_chunk_ids: ['chunk-A'], rel_type: 'KNOWS'}]->(b)",
            {},
        )
        await conn.query(
            "MATCH (a:__Entity__ {id: 'alice-2'}), (b:__Entity__ {id: 'bob'}) "
            "CREATE (a)-[:RELATES {source_chunk_ids: ['chunk-B'], rel_type: 'KNOWS'}]->(b)",
            {},
        )

        dedup = EntityDeduplicator(rag._graph_store, embedder)
        merged = await dedup.deduplicate(fuzzy=False)
        assert merged >= 1, "test setup: dedup should have merged alice-1/alice-2"

        # Survivor edge should carry BOTH chunk ids — the union.
        result = await conn.query(
            "MATCH (:__Entity__ {name: 'Alice'})-[r:RELATES]->(:__Entity__ {name: 'Bob'}) "
            "RETURN r.source_chunk_ids AS scids",
            {},
        )
        assert result.result_set, "survivor RELATES edge should exist"
        scids = result.result_set[0][0] or []
        assert set(scids) == {"chunk-A", "chunk-B"}, (
            f"dedup remap must UNION source_chunk_ids — got {scids}. "
            "Pre-fix, SET nr += properties(r) overwrote the survivor's "
            "list with the duplicate's, silently dropping provenance "
            "and breaking subsequent stale-RELATES cleanup."
        )

    async def test_relates_provenance_union_on_shared_fact_ingest(
        self, real_falkordb_rag_factory, scripted_llm, resolver
    ):
        """Two docs both extract ``Alice KNOWS Bob`` — the RELATES edge's
        ``source_chunk_ids`` must UNION both docs' chunks, not overwrite.
        Deleting one doc must leave the edge intact with the surviving
        doc's chunk in the provenance list.

        Pre-fix, ``upsert_relationships`` did ``SET r += item.properties``
        on MERGE-found edges, so doc-B's ingest replaced doc-A's chunk in
        ``source_chunk_ids``. ``delete_document("doc-B")`` then deleted
        the edge entirely even though doc-A still supports it. The fix
        special-cases RELATES in the upsert path to mirror the dedup
        remap's union idiom.
        """
        import json

        from .conftest import MockLLM

        # Both docs produce identical Alice KNOWS Bob extraction.
        def _step2_with_knows():
            return json.dumps(
                {
                    "entities": [
                        {"name": "Alice", "type": "Person", "description": "p1"},
                        {"name": "Bob", "type": "Person", "description": "p2"},
                    ],
                    "relationships": [
                        {
                            "source": "Alice",
                            "target": "Bob",
                            "type": "KNOWS",
                            "description": "Alice knows Bob",
                            "keywords": "social",
                            "weight": 0.9,
                        }
                    ],
                }
            )

        llm = MockLLM(responses=[_step2_with_knows(), _step2_with_knows()], strict=True)
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)

        await rag.ingest(text="Alice knows Bob.", document_id="doc-A", resolver=resolver)
        await rag.ingest(text="Alice also knows Bob.", document_id="doc-B", resolver=resolver)

        # After both ingests, the RELATES edge should carry BOTH chunk ids.
        result = await rag._graph_store._conn.query(
            "MATCH (:__Entity__ {name: 'Alice'})-[r:RELATES]->(:__Entity__ {name: 'Bob'}) "
            "RETURN r.source_chunk_ids AS scids",
            {},
        )
        assert result.result_set, "RELATES edge should exist after both ingests"
        scids_after_ingest = set(result.result_set[0][0] or [])
        assert len(scids_after_ingest) == 2, (
            f"upsert_relationships must UNION source_chunk_ids on shared "
            f"facts — got {scids_after_ingest!r} (expected 2 distinct chunk ids). "
            "Pre-fix, SET r += item.properties overwrote doc-A's chunk with "
            "doc-B's on the second ingest."
        )

        # Now delete doc-B. Its chunk leaves source_chunk_ids; the edge
        # survives because doc-A's chunk is still in the list.
        await rag.delete_document("doc-B")

        result = await rag._graph_store._conn.query(
            "MATCH (:__Entity__ {name: 'Alice'})-[r:RELATES]->(:__Entity__ {name: 'Bob'}) "
            "RETURN r.source_chunk_ids AS scids",
            {},
        )
        assert result.result_set, (
            "RELATES edge must SURVIVE delete_document('doc-B') because doc-A "
            "still supports the fact. Pre-fix, the edge would be removed: "
            "doc-B's ingest had clobbered doc-A's chunk in source_chunk_ids, "
            "so post-delete the list emptied and the edge dropped."
        )
        scids_after_delete = set(result.result_set[0][0] or [])
        assert len(scids_after_delete) == 1, (
            f"After deleting doc-B, RELATES.source_chunk_ids should contain "
            f"exactly one chunk (doc-A's) — got {scids_after_delete!r}."
        )
