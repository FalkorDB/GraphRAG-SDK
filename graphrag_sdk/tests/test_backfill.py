"""Unit tests for Group-3 (LLM-driven backfill) ontology evolution.

Exercises:
- :py:class:`BackfillExecutor` (concurrency, idempotency markers, partial failure)
- ``GraphRAG.backfill_attribute`` (scope → LLM → merge → marker)
- ``GraphRAG.backfill_entity`` (full chunk re-scan, new MENTIONED_IN edges)
- ``GraphRAG.backfill_relation_pattern`` (candidate-pair filter)
- ``GraphRAG.backfill_attribute_semantic`` (no-chunk LLM coercion)

The LLM is always ``MockLLM(strict=True)`` so the second run of an
idempotent backfill catches itself if it tries to make extra calls.
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
    rag._ontology_store.retype_property = AsyncMock()
    rag._ontology_initialized = True
    rag._global_ontology = small_ontology
    rag.ontology = small_ontology
    # Backfill scope helpers
    rag._graph_store.list_chunks_for_attribute_backfill = AsyncMock(return_value=[])
    rag._graph_store.list_chunks_for_entity_backfill = AsyncMock(return_value=[])
    rag._graph_store.list_chunks_for_relation_pattern_backfill = AsyncMock(return_value=[])
    rag._graph_store.list_node_values_for_semantic_coerce = AsyncMock(return_value=[])
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


# ── backfill_attribute ──────────────────────────────────────────


class TestBackfillAttribute:
    @pytest.mark.asyncio
    async def test_rejects_unsupported_scope(self, rag):
        with pytest.raises(ValueError, match="unsupported scope"):
            await rag.backfill_attribute("Person", "age", scope="all")

    @pytest.mark.asyncio
    async def test_rejects_unknown_attribute(self, rag):
        with pytest.raises(ValueError, match="Unknown attribute"):
            await rag.backfill_attribute("Person", "phone")

    @pytest.mark.asyncio
    async def test_fills_value_and_marks_chunk(self, rag):
        rag.llm = MockLLM(responses=[json.dumps({"results": {"Alice": "42"}})])
        rag._graph_store.list_chunks_for_attribute_backfill = AsyncMock(
            return_value=[
                {
                    "chunk_id": "c1",
                    "chunk_text": "Alice is 42 years old.",
                    "entities": [{"id": "alice", "name": "Alice"}],
                }
            ]
        )
        result = await rag.backfill_attribute("Person", "age")
        assert result.values_filled == 1
        assert result.dropped_for_coercion == 0
        rag._graph_store.set_node_property_by_id.assert_awaited_once_with(
            "Person", "alice", "age", 42
        )
        rag._graph_store.mark_chunk_extracted.assert_awaited_once_with(
            "c1", "backfill_attribute:Person:age"
        )

    @pytest.mark.asyncio
    async def test_op_id_is_deterministic(self, rag):
        rag.llm = MockLLM(responses=[json.dumps({"results": {}})])
        rag._graph_store.list_chunks_for_attribute_backfill = AsyncMock(
            return_value=[{"chunk_id": "c1", "chunk_text": "", "entities": []}]
        )
        result = await rag.backfill_attribute("Person", "email")
        assert result.operation_id == "backfill_attribute:Person:email"

    @pytest.mark.asyncio
    async def test_lookup_keys_by_both_name_and_id(self, rag):
        """If the LLM echoes the entity id (because the entity had no name and
        the prompt fell back to id), the merge must still match — keying by
        name alone would silently drop the value."""
        rag.llm = MockLLM(responses=[json.dumps({"results": {"alice-id-7": "42"}})])
        rag._graph_store.list_chunks_for_attribute_backfill = AsyncMock(
            return_value=[
                {
                    "chunk_id": "c1",
                    "chunk_text": "...",
                    # Entity has no name — only an id.
                    "entities": [{"id": "alice-id-7", "name": None}],
                }
            ]
        )
        result = await rag.backfill_attribute("Person", "age")
        assert result.values_filled == 1
        rag._graph_store.set_node_property_by_id.assert_awaited_once_with(
            "Person", "alice-id-7", "age", 42
        )

    @pytest.mark.asyncio
    async def test_chunks_skipped_reflects_prior_run(self, rag):
        """``chunks_skipped`` should count chunks already marked from a
        previous run — i.e. work this run didn't have to redo."""
        rag.llm = MockLLM(responses=[json.dumps({"results": {}})])
        rag._graph_store.list_chunks_for_attribute_backfill = AsyncMock(
            return_value=[{"chunk_id": "c1", "chunk_text": "", "entities": []}]
        )
        # Pretend 5 prior chunks are already marked.
        rag._graph_store.count_chunks_marked_with_op = AsyncMock(return_value=6)
        result = await rag.backfill_attribute("Person", "age")
        # 1 scanned this run, 6 marked total → 5 skipped from prior runs.
        assert result.chunks_skipped == 5

    @pytest.mark.asyncio
    async def test_malformed_json_lands_in_failed_chunks(self, rag):
        """Malformed JSON must raise from the parser so the BackfillExecutor
        adds the chunk to ``failed_chunks`` instead of marking it done."""
        rag.llm = MockLLM(responses=["this is not json"])
        rag._graph_store.list_chunks_for_attribute_backfill = AsyncMock(
            return_value=[{"chunk_id": "c-bad", "chunk_text": "", "entities": []}]
        )
        result = await rag.backfill_attribute("Person", "age")
        assert result.failed_chunks == ["c-bad"]
        rag._graph_store.mark_chunk_extracted.assert_not_awaited()


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


# ── backfill_attribute_semantic ─────────────────────────────────


class TestBackfillAttributeSemantic:
    @pytest.mark.asyncio
    async def test_coerces_node_values(self, rag):
        rag.llm = MockLLM(responses=[json.dumps({"value": 12})])
        rag._graph_store.list_node_values_for_semantic_coerce = AsyncMock(
            return_value=[("alice", "twelve")]
        )
        result = await rag.backfill_attribute_semantic("Person", "age", "INTEGER")
        rag._graph_store.set_node_property_by_id.assert_awaited_once_with(
            "Person", "alice", "age", 12
        )
        rag._ontology_store.retype_property.assert_awaited_once_with(
            "entity", "Person", "age", "INTEGER"
        )
        assert result.values_filled == 1
        assert result.dropped_for_coercion == 0

    @pytest.mark.asyncio
    async def test_drops_unconvertible(self, rag):
        rag.llm = MockLLM(responses=[json.dumps({"value": None})])
        rag._graph_store.list_node_values_for_semantic_coerce = AsyncMock(
            return_value=[("alice", "gibberish")]
        )
        result = await rag.backfill_attribute_semantic("Person", "age", "INTEGER")
        rag._graph_store.set_node_property_by_id.assert_awaited_once_with(
            "Person", "alice", "age", None
        )
        assert result.dropped_for_coercion == 1
        assert result.values_filled == 0

    @pytest.mark.asyncio
    async def test_relation_owner_rejected(self, rag):
        with pytest.raises(ValueError, match="relation owners"):
            await rag.backfill_attribute_semantic("WORKS_AT", "since", "DATE")

    @pytest.mark.asyncio
    async def test_failures_land_in_failed_node_ids(self, rag):
        """Per-node failures in semantic backfill must populate
        ``failed_node_ids`` (not ``failed_chunks`` — these aren't chunks)
        so callers retry against the right surface."""

        async def _boom(prompt, **kwargs):
            raise RuntimeError("LLM down")

        rag.llm.ainvoke = _boom  # type: ignore[method-assign]
        rag._graph_store.list_node_values_for_semantic_coerce = AsyncMock(
            return_value=[("alice", "twelve")]
        )
        result = await rag.backfill_attribute_semantic("Person", "age", "INTEGER")
        assert result.failed_node_ids == ["alice"]
        assert result.failed_chunks == []
