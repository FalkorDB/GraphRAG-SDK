"""Unit tests for LLM-backed ontology evolution + discovery.

Exercises:
- :py:class:`BackfillExecutor` (concurrency, idempotency markers, partial failure)
- ``GraphRAG.add_attribute`` — atomic declare + LLM backfill + commit
  (the invariant-enforcing core)
- ``GraphRAG.backfill_entity`` — opportunistic: scan chunks for missed instances
- ``GraphRAG.backfill_relation_pattern`` — opportunistic: discover missed edges
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.api.main import GraphRAG
from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.models import Attribute, Entity, Ontology, Relation
from graphrag_sdk.ingestion.backfill import (
    BackfillExecutor,
    BackfillMergeStats,
    ChunkContext,
)
from graphrag_sdk.storage.ontology_store import OntologyStore

from .conftest import MockLLM

# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def small_ontology() -> Ontology:
    return Ontology(
        entities=[
            Entity(
                label="Person",
                properties=[
                    Attribute(name="age", type="INTEGER"),
                    Attribute(name="email", type="STRING"),
                ],
            ),
            Entity(label="Company"),
        ],
        relations=[
            Relation(label="WORKS_AT", patterns=[("Person", "Company")]),
        ],
    )


@pytest.fixture
def rag(embedder, small_ontology, mock_graph_store):
    """GraphRAG with mocked stores. LLM is configured per-test."""
    conn = MagicMock(spec=FalkorDBConnection)
    conn.config = ConnectionConfig()
    rag = GraphRAG(connection=conn, llm=MockLLM(), embedder=embedder, embedding_dimension=8)
    rag._graph_store = mock_graph_store
    rag._ontology_store = MagicMock(spec=OntologyStore)
    rag._ontology_store.load = AsyncMock(return_value=small_ontology)
    rag._ontology_store.add_entity_property = AsyncMock()
    rag._ontology_initialized = True
    rag._global_ontology = small_ontology
    rag.ontology = small_ontology
    # Backfill scope helpers
    rag._graph_store.list_chunks_for_attribute_backfill = AsyncMock(return_value=[])
    rag._graph_store.list_chunks_for_entity_backfill = AsyncMock(return_value=[])
    rag._graph_store.list_chunks_for_relation_pattern_backfill = AsyncMock(return_value=[])
    rag._graph_store.mark_chunk_extracted = AsyncMock()
    rag._graph_store.set_node_property_by_id = AsyncMock()
    rag._graph_store.count_chunks_marked_with_op = AsyncMock(return_value=0)
    return rag


# ── BackfillExecutor unit tests ─────────────────────────────────


class TestBackfillExecutor:
    @pytest.mark.asyncio
    async def test_marks_chunks_on_success(self, small_ontology, mock_graph_store):
        llm = MockLLM(responses=['{"results": {}}'])
        executor = BackfillExecutor(llm, mock_graph_store, concurrency=1)
        mock_graph_store.mark_chunk_extracted = AsyncMock()

        async def chunks():
            yield ChunkContext(chunk_id="c1", chunk_text="hi", payload={}, ontology=small_ontology)

        async def merge_fn(parsed, ctx):
            return BackfillMergeStats(values_filled=2)

        result = await executor.run(
            op_id="op:test",
            chunks=chunks(),
            prompt_builder=lambda c: "prompt",
            parse_fn=lambda t, c: {},
            merge_fn=merge_fn,
        )
        mock_graph_store.mark_chunk_extracted.assert_awaited_once_with("c1", "op:test")
        assert result.values_filled == 2
        assert result.chunks_scanned == 1
        assert result.failed_chunks == []
        assert result.llm_calls == 1

    @pytest.mark.asyncio
    async def test_live_task_count_stays_bounded(self, small_ontology, mock_graph_store):
        """Worker-pool pattern: live ``asyncio.Task`` count stays O(concurrency)
        regardless of corpus size. The previous implementation created one
        task per chunk up-front, which scales linearly with the corpus
        and pins memory until the LLM drains."""
        import asyncio as _asyncio

        chunk_count = 200
        concurrency = 4
        llm = MockLLM(responses=['{"results": {}}'])
        executor = BackfillExecutor(llm, mock_graph_store, concurrency=concurrency)
        mock_graph_store.mark_chunk_extracted = AsyncMock()

        peak = 0

        async def chunks():
            nonlocal peak
            for i in range(chunk_count):
                # Each yield happens after the previous chunks have been
                # consumed by workers, so this samples the live count
                # roughly when one chunk is in flight.
                peak = max(peak, len(_asyncio.all_tasks()))
                yield ChunkContext(
                    chunk_id=f"c{i}",
                    chunk_text="",
                    payload={},
                    ontology=small_ontology,
                )

        async def merge_fn(parsed, ctx):
            return BackfillMergeStats(values_filled=1)

        await executor.run(
            op_id="op:bounded",
            chunks=chunks(),
            prompt_builder=lambda c: "p",
            parse_fn=lambda t, c: {},
            merge_fn=merge_fn,
        )
        # Sanity ceiling: at most concurrency + producer + test driver +
        # a small constant. If the executor regresses to one-task-per-chunk
        # this leaps to ~chunk_count.
        assert peak < chunk_count, f"task count exploded: peak={peak}, chunks={chunk_count}"
        assert peak <= concurrency + 8

    @pytest.mark.asyncio
    async def test_failures_dont_poison_run(self, small_ontology, mock_graph_store):
        llm = MockLLM(responses=["bad", '{"results": {}}'])
        executor = BackfillExecutor(llm, mock_graph_store, concurrency=1)

        async def chunks():
            yield ChunkContext(chunk_id="c1", chunk_text="", payload={}, ontology=small_ontology)
            yield ChunkContext(chunk_id="c2", chunk_text="", payload={}, ontology=small_ontology)

        def parse(text, ctx):
            if text == "bad":
                raise ValueError("nope")
            return {}

        async def merge_fn(parsed, ctx):
            return BackfillMergeStats(values_filled=1)

        result = await executor.run(
            op_id="op:fail",
            chunks=chunks(),
            prompt_builder=lambda c: "p",
            parse_fn=parse,
            merge_fn=merge_fn,
        )
        assert result.failed_chunks == ["c1"]
        assert result.chunks_scanned == 1
        assert result.values_filled == 1


# ── add_attribute (atomic declare + LLM backfill + commit) ──────


class TestAddAttributeAtomic:
    """``add_attribute`` is atomic: LLM backfill → ontology graph write
    as commit point. On hard failure the ontology stays at its pre-call
    state; the data graph may be partially mutated but is idempotent on
    retry via chunk markers."""

    @pytest.mark.asyncio
    async def test_relation_owner_raises_not_implemented(self, rag):
        from graphrag_sdk.core.models import Attribute

        with pytest.raises(NotImplementedError, match="relation owners"):
            await rag.add_attribute("WORKS_AT", Attribute(name="since"))

    @pytest.mark.asyncio
    async def test_already_declared_raises(self, rag):
        """Type changes go through drop+add, not a silent retype on add."""
        from graphrag_sdk.core.models import Attribute

        with pytest.raises(ValueError, match="already declared"):
            await rag.add_attribute("Person", Attribute(name="age", type="STRING"))

    @pytest.mark.asyncio
    async def test_unknown_label_raises(self, rag):
        from graphrag_sdk.core.models import Attribute

        with pytest.raises(ValueError, match="Unknown ontology label"):
            await rag.add_attribute("Alien", Attribute(name="x"))

    @pytest.mark.asyncio
    async def test_happy_path_backfills_then_commits(self, rag):
        from graphrag_sdk.core.models import Attribute

        attr = Attribute(name="role", type="STRING")
        rag.llm = MockLLM(responses=[json.dumps({"results": {"Alice": "engineer"}})])
        rag._graph_store.list_chunks_for_attribute_backfill = AsyncMock(
            return_value=[
                {
                    "chunk_id": "c1",
                    "chunk_text": "Alice is an engineer.",
                    "entities": [{"id": "alice", "name": "Alice"}],
                }
            ]
        )
        result = await rag.add_attribute("Person", attr)
        # Data write happened.
        rag._graph_store.set_node_property_by_id.assert_awaited_once_with(
            "Person", "alice", "role", "engineer"
        )
        rag._graph_store.mark_chunk_extracted.assert_awaited_once_with(
            "c1", "add_attribute:Person:role:STRING"
        )
        # Ontology commit happened LAST.
        rag._ontology_store.add_entity_property.assert_awaited_once_with("Person", attr)
        # EvolutionResult returned with backfill stats.
        from graphrag_sdk.ingestion.backfill import EvolutionResult

        assert isinstance(result, EvolutionResult)
        assert result.values_filled == 1
        assert result.chunks_scanned == 1

    @pytest.mark.asyncio
    async def test_hard_failure_does_not_commit_ontology(self, rag):
        """If any chunk hard-fails (malformed JSON beyond retries),
        OntologyEvolutionError is raised and the ontology graph write is
        skipped. Schema stays at pre-call state; data graph may be
        partially mutated but a retry of the same add_attribute call is
        idempotent (chunk markers)."""
        from graphrag_sdk.core.models import Attribute
        from graphrag_sdk.ingestion.backfill import OntologyEvolutionError

        rag.llm = MockLLM(responses=["this is not json"])
        rag._graph_store.list_chunks_for_attribute_backfill = AsyncMock(
            return_value=[{"chunk_id": "c-bad", "chunk_text": "", "entities": []}]
        )
        with pytest.raises(OntologyEvolutionError) as excinfo:
            await rag.add_attribute("Person", Attribute(name="role"))
        assert excinfo.value.failed_chunks == ["c-bad"]
        # Commit point NOT reached.
        rag._ontology_store.add_entity_property.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_op_id_is_deterministic_from_signature(self, rag):
        """Re-running the same add_attribute call generates the same op_id,
        so chunk markers from a partial prior run are respected on retry."""
        from graphrag_sdk.core.models import Attribute

        rag.llm = MockLLM(responses=[json.dumps({"results": {}})])
        rag._graph_store.list_chunks_for_attribute_backfill = AsyncMock(
            return_value=[{"chunk_id": "c1", "chunk_text": "", "entities": []}]
        )
        await rag.add_attribute("Person", Attribute(name="role"))
        # The mark call carries the op_id; assert it matches the signature.
        rag._graph_store.mark_chunk_extracted.assert_awaited_with(
            "c1", "add_attribute:Person:role:STRING"
        )

    @pytest.mark.asyncio
    async def test_op_id_includes_type_so_drop_add_rescans(self, rag):
        """The documented type-change pattern is drop_attribute + add_attribute
        with the new type. Without the type in op_id, chunk markers from
        the prior backfill would short-circuit the second add — committing
        the new schema while leaving the data graph at the old type.
        Regression test for PR #268 comment #3319054109."""
        from graphrag_sdk.core.models import Attribute

        rag.llm = MockLLM(responses=[json.dumps({"results": {}})])
        rag._graph_store.list_chunks_for_attribute_backfill = AsyncMock(
            return_value=[{"chunk_id": "c1", "chunk_text": "", "entities": []}]
        )
        await rag.add_attribute("Person", Attribute(name="role", type="STRING"))
        first_op = rag._graph_store.mark_chunk_extracted.await_args.args[1]

        # Pretend a drop + re-add with new type.
        rag._graph_store.mark_chunk_extracted.reset_mock()
        rag.llm = MockLLM(responses=[json.dumps({"results": {}})])
        # Simulate "role is no longer declared" by mutating the fixture
        # ontology in place (the actual drop runs in production).
        person = next(
            e for e in rag._global_ontology.entities if e.label == "Person"
        )
        person.properties = [p for p in person.properties if p.name != "role"]
        await rag.add_attribute("Person", Attribute(name="role", type="INTEGER"))
        second_op = rag._graph_store.mark_chunk_extracted.await_args.args[1]

        assert first_op != second_op, (
            f"op_id must differ between type changes; got {first_op!r} both times"
        )
        assert first_op.endswith(":STRING")
        assert second_op.endswith(":INTEGER")

    @pytest.mark.asyncio
    async def test_id_only_entity_lookup_matches(self, rag):
        """LLM may echo the entity id when the entity has no name (the prompt
        falls back to id). The merge must match either."""
        from graphrag_sdk.core.models import Attribute

        rag.llm = MockLLM(responses=[json.dumps({"results": {"alice-id-7": "engineer"}})])
        rag._graph_store.list_chunks_for_attribute_backfill = AsyncMock(
            return_value=[
                {
                    "chunk_id": "c1",
                    "chunk_text": "...",
                    "entities": [{"id": "alice-id-7", "name": None}],
                }
            ]
        )
        result = await rag.add_attribute("Person", Attribute(name="role", type="STRING"))
        assert result.values_filled == 1
        rag._graph_store.set_node_property_by_id.assert_awaited_once_with(
            "Person", "alice-id-7", "role", "engineer"
        )


# ── backfill_entity ─────────────────────────────────────────────


class TestBackfillEntity:
    @pytest.mark.asyncio
    async def test_rejects_unknown_label(self, rag):
        with pytest.raises(ValueError, match="Unknown entity"):
            await rag.backfill_entity("Alien")

    @pytest.mark.asyncio
    async def test_rejects_bad_scope(self, rag):
        with pytest.raises(ValueError, match="scope must be"):
            await rag.backfill_entity("Person", scope="weird")

    @pytest.mark.asyncio
    async def test_creates_entity_and_mention_edge(self, rag):
        rag.llm = MockLLM(
            responses=[
                json.dumps(
                    {"entities": [{"name": "Bob", "description": "A founder", "attributes": {}}]}
                )
            ]
        )
        rag._graph_store.list_chunks_for_entity_backfill = AsyncMock(
            return_value=[{"chunk_id": "c1", "chunk_text": "Bob runs Acme"}]
        )
        rag._graph_store.upsert_nodes = AsyncMock(return_value=1)
        rag._graph_store.upsert_relationships = AsyncMock(return_value=1)
        result = await rag.backfill_entity("Person", scope="all")
        rag._graph_store.upsert_nodes.assert_awaited_once()
        rag._graph_store.upsert_relationships.assert_awaited_once()
        # The MENTIONED_IN edge points the entity at the chunk.
        rel_call = rag._graph_store.upsert_relationships.await_args.args[0]
        assert rel_call[0].type == "MENTIONED_IN"
        assert rel_call[0].end_node_id == "c1"
        assert result.values_filled == 1


# ── backfill_relation_pattern ───────────────────────────────────


class TestBackfillRelationPattern:
    @pytest.mark.asyncio
    async def test_rejects_unsupported_scope(self, rag):
        with pytest.raises(ValueError, match="unsupported scope"):
            await rag.backfill_relation_pattern("WORKS_AT", "Person", "Company", scope="all")

    @pytest.mark.asyncio
    async def test_pair_lookup_is_tuple_keyed_no_id_mismatch(self, rag):
        """Two candidate pairs sharing the same src_name must not produce a
        crossed edge: ``(Alice, Acme)`` and ``(Alice, Globex)`` in the same
        chunk used to be lookup-collapsible; the merge now keys by the
        full ``(src_name, tgt_name)`` tuple so each pair stays paired."""
        rag.llm = MockLLM(
            responses=[
                json.dumps(
                    {"links": [{"src": "Alice", "tgt": "Globex", "description": "joined later"}]}
                )
            ]
        )
        rag._graph_store.list_chunks_for_relation_pattern_backfill = AsyncMock(
            return_value=[
                {
                    "chunk_id": "c1",
                    "chunk_text": "Alice was at Acme, then Globex.",
                    "pairs": [
                        {
                            "src_id": "alice",
                            "src_name": "Alice",
                            "tgt_id": "acme",
                            "tgt_name": "Acme",
                        },
                        {
                            "src_id": "alice",
                            "src_name": "Alice",
                            "tgt_id": "globex",
                            "tgt_name": "Globex",
                        },
                    ],
                }
            ]
        )
        rag._graph_store.upsert_relationships = AsyncMock(return_value=1)
        result = await rag.backfill_relation_pattern("WORKS_AT", "Person", "Company")
        assert result.values_filled == 1
        edge = rag._graph_store.upsert_relationships.await_args.args[0][0]
        # Must be Alice → Globex, NOT Alice → Acme (would happen if name maps
        # collided).
        assert edge.start_node_id == "alice"
        assert edge.end_node_id == "globex"

    @pytest.mark.asyncio
    async def test_no_source_chunk_ids_on_non_relates(self, rag):
        """Backfill edges of arbitrary type must not write source_chunk_ids:
        ``GraphStore.upsert_relationships`` only unions provenance for
        RELATES, so writing it on (e.g.) WORKS_AT would be silently
        overwritten by future MERGE calls."""
        rag.llm = MockLLM(
            responses=[json.dumps({"links": [{"src": "Alice", "tgt": "Acme", "description": "x"}]})]
        )
        rag._graph_store.list_chunks_for_relation_pattern_backfill = AsyncMock(
            return_value=[
                {
                    "chunk_id": "c1",
                    "chunk_text": "...",
                    "pairs": [
                        {
                            "src_id": "alice",
                            "src_name": "Alice",
                            "tgt_id": "acme",
                            "tgt_name": "Acme",
                        }
                    ],
                }
            ]
        )
        rag._graph_store.upsert_relationships = AsyncMock(return_value=1)
        await rag.backfill_relation_pattern("WORKS_AT", "Person", "Company")
        edge = rag._graph_store.upsert_relationships.await_args.args[0][0]
        assert "source_chunk_ids" not in edge.properties

    @pytest.mark.asyncio
    async def test_links_recognised_pairs(self, rag):
        rag.llm = MockLLM(
            responses=[
                json.dumps(
                    {"links": [{"src": "Alice", "tgt": "Acme", "description": "joined 2020"}]}
                )
            ]
        )
        rag._graph_store.list_chunks_for_relation_pattern_backfill = AsyncMock(
            return_value=[
                {
                    "chunk_id": "c1",
                    "chunk_text": "Alice joined Acme.",
                    "pairs": [
                        {
                            "src_id": "alice",
                            "src_name": "Alice",
                            "tgt_id": "acme",
                            "tgt_name": "Acme",
                        }
                    ],
                }
            ]
        )
        rag._graph_store.upsert_relationships = AsyncMock(return_value=1)
        result = await rag.backfill_relation_pattern("WORKS_AT", "Person", "Company")
        assert result.values_filled == 1
        rag._graph_store.upsert_relationships.assert_awaited_once()
        rel = rag._graph_store.upsert_relationships.await_args.args[0][0]
        assert rel.type == "WORKS_AT"
        assert rel.start_node_id == "alice"
        assert rel.end_node_id == "acme"


# Under the strict alignment design, retype_attribute and the
# backfill_attribute_semantic that supported it are gone. Type changes
# go through drop_attribute + add_attribute; the LLM re-derives values
# from the chunks. The Attribute-Semantic tests are removed; coverage
# of the LLM-coercion path lives in test_attribute_prompt for the
# parser side and in TestAddAttributeAtomic for the end-to-end shape.
