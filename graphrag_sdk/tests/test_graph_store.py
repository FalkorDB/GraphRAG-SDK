"""Tests for storage/graph_store.py — Repository pattern for graph operations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.connection import FalkorDBConnection
from graphrag_sdk.core.exceptions import DatabaseError
from graphrag_sdk.core.models import GraphNode, GraphRelationship
from graphrag_sdk.storage.graph_store import GraphStore


@pytest.fixture
def graph_store(mock_connection):
    return GraphStore(mock_connection)


class TestGraphStoreUpsertNodes:
    async def test_upsert_single_node(self, graph_store, mock_connection):
        nodes = [GraphNode(id="n1", label="Person", properties={"name": "Alice"})]
        result = await graph_store.upsert_nodes(nodes)
        assert result == 1
        mock_connection.query.assert_called_once()
        cypher = mock_connection.query.call_args[0][0]
        assert "UNWIND" in cypher
        assert "MERGE" in cypher
        assert "Person" in cypher
        assert "__Entity__" in cypher

    async def test_upsert_multiple_nodes(self, graph_store, mock_connection):
        nodes = [
            GraphNode(id="n1", label="Person", properties={"name": "Alice"}),
            GraphNode(id="n2", label="Company", properties={"name": "Acme"}),
        ]
        result = await graph_store.upsert_nodes(nodes)
        assert result == 2
        assert mock_connection.query.call_count == 2

    async def test_upsert_empty_list(self, graph_store, mock_connection):
        result = await graph_store.upsert_nodes([])
        assert result == 0
        mock_connection.query.assert_not_called()

    async def test_upsert_raises_on_error(self, graph_store, mock_connection):
        mock_connection.query = AsyncMock(side_effect=Exception("db error"))
        with pytest.raises(DatabaseError, match="Node upsert failed"):
            await graph_store.upsert_nodes([GraphNode(id="x", label="T", properties={})])

    async def test_upsert_passes_id_in_batch_param(self, graph_store, mock_connection):
        await graph_store.upsert_nodes([GraphNode(id="test-id", label="X", properties={})])
        params = mock_connection.query.call_args[0][1]
        assert params["batch"][0]["id"] == "test-id"

    async def test_upsert_sanitizes_control_chars_in_batch_params(
        self, graph_store, mock_connection
    ):
        await graph_store.upsert_nodes(
            [GraphNode(id="id\x00\x01", label="Chunk", properties={"text": "A\x00B\x01C"})]
        )
        params = mock_connection.query.call_args[0][1]
        assert params["batch"][0]["id"] == "id"
        assert params["batch"][0]["properties"]["text"] == "ABC"

    async def test_upsert_sanitizes_control_chars_in_fallback_params(
        self, graph_store, mock_connection
    ):
        """Per-item fallback path should also use sanitized IDs and properties."""
        mock_connection.query = AsyncMock(side_effect=[Exception("batch fail"), MagicMock()])
        await graph_store.upsert_nodes(
            [GraphNode(id="id\x00\x01", label="X", properties={"t": "A\x00B"})]
        )
        # Second call is the per-item fallback
        fallback_params = mock_connection.query.call_args_list[1][0][1]
        assert fallback_params["id"] == "id"
        assert fallback_params["properties"]["t"] == "AB"


class TestGraphStoreUpsertRelationships:
    async def test_upsert_relationship(self, graph_store, mock_connection):
        rels = [
            GraphRelationship(
                start_node_id="a", end_node_id="b", type="KNOWS", properties={"since": 2020}
            )
        ]
        result = await graph_store.upsert_relationships(rels)
        assert result == 1
        cypher = mock_connection.query.call_args[0][0]
        assert "MERGE" in cypher
        assert "KNOWS" in cypher
        # Unknown rel type defaults to __Entity__ labels
        assert "`__Entity__`" in cypher

    async def test_upsert_sanitizes_control_chars_in_batch_params(
        self, graph_store, mock_connection
    ):
        rels = [
            GraphRelationship(
                start_node_id="a\x00\x01",
                end_node_id="b\x00\x01",
                type="RELATES",
                properties={"note": "A\x00B\x01C"},
            )
        ]
        await graph_store.upsert_relationships(rels)
        params = mock_connection.query.call_args[0][1]
        assert params["batch"][0]["start_id"] == "a"
        assert params["batch"][0]["end_id"] == "b"
        assert params["batch"][0]["properties"]["note"] == "ABC"

    async def test_upsert_drops_rels_with_empty_sanitized_ids(self, graph_store, mock_connection):
        rels = [
            GraphRelationship(start_node_id="\x00", end_node_id="valid", type="R"),
            GraphRelationship(start_node_id="ok", end_node_id="also-ok", type="R"),
        ]
        result = await graph_store.upsert_relationships(rels)
        assert result == 1
        params = mock_connection.query.call_args[0][1]
        assert len(params["batch"]) == 1
        assert params["batch"][0]["start_id"] == "ok"

    async def test_upsert_empty_relationships(self, graph_store, mock_connection):
        result = await graph_store.upsert_relationships([])
        assert result == 0

    async def test_upsert_rel_error_continues(self, graph_store, mock_connection):
        """Relationship upsert logs warning but continues.

        With UNWIND batching, rels are grouped by type. Each type group
        gets one UNWIND call. If that fails, per-item fallback runs.
        R1 type: batch fails → individual fallback also fails → logged
        R2 type: batch succeeds.
        """
        mock_connection.query = AsyncMock(
            side_effect=[Exception("fail"), Exception("fail again"), MagicMock()]
        )
        rels = [
            GraphRelationship(start_node_id="a", end_node_id="b", type="R1"),
            GraphRelationship(start_node_id="c", end_node_id="d", type="R2"),
        ]
        # R1 batch fails (call 1), R1 individual fails (call 2), R2 batch succeeds (call 3)
        result = await graph_store.upsert_relationships(rels)
        assert result == 1  # only R2 batch succeeded


class TestGraphStoreGetConnectedEntities:
    async def test_get_entities(self, graph_store, mock_connection):
        result_mock = MagicMock()
        result_mock.result_set = [
            ["e1", ["Person"], {"name": "Alice"}],
            ["e2", ["Company"], {"name": "Acme"}],
        ]
        mock_connection.query = AsyncMock(return_value=result_mock)
        entities = await graph_store.get_connected_entities("chunk-1")
        assert len(entities) == 2
        assert entities[0]["id"] == "e1"
        assert entities[1]["labels"] == ["Company"]

    async def test_get_entities_empty(self, graph_store, mock_connection):
        result_mock = MagicMock()
        result_mock.result_set = []
        mock_connection.query = AsyncMock(return_value=result_mock)
        entities = await graph_store.get_connected_entities("nonexistent")
        assert entities == []

    async def test_get_entities_error_returns_empty(self, graph_store, mock_connection):
        mock_connection.query = AsyncMock(side_effect=Exception("db error"))
        entities = await graph_store.get_connected_entities("chunk-1")
        assert entities == []

    async def test_custom_max_hops(self, graph_store, mock_connection):
        result_mock = MagicMock()
        result_mock.result_set = []
        mock_connection.query = AsyncMock(return_value=result_mock)
        await graph_store.get_connected_entities("c1", max_hops=3)
        cypher = mock_connection.query.call_args[0][0]
        assert "*1..3" in cypher


class TestGraphStoreQueryRaw:
    async def test_raw_query(self, graph_store, mock_connection):
        await graph_store.query_raw("MATCH (n) RETURN n LIMIT 10")
        mock_connection.query.assert_called_once_with("MATCH (n) RETURN n LIMIT 10", None)

    async def test_raw_query_with_params(self, graph_store, mock_connection):
        await graph_store.query_raw("MATCH (n {id: $id})", {"id": "test"})
        mock_connection.query.assert_called_once_with("MATCH (n {id: $id})", {"id": "test"})


class TestGraphStoreDeleteAll:
    async def test_delete_all(self, graph_store, mock_connection):
        mock_connection.delete_graph = AsyncMock()
        await graph_store.delete_all()
        mock_connection.delete_graph.assert_called_once()

    async def test_delete_all_fallback(self, graph_store, mock_connection):
        """Falls back to DETACH DELETE if delete_graph() raises."""
        mock_connection.delete_graph = AsyncMock(side_effect=Exception("no graph"))
        await graph_store.delete_all()
        cypher = mock_connection.query.call_args[0][0]
        assert "DETACH DELETE" in cypher


class TestCleanProperties:
    def test_removes_none(self):
        result = GraphStore._clean_properties({"a": 1, "b": None, "c": "ok"})
        assert "b" not in result
        assert result["a"] == 1
        assert result["c"] == "ok"

    def test_preserves_primitives(self):
        result = GraphStore._clean_properties({"s": "text", "i": 42, "f": 3.14, "b": True})
        assert result == {"s": "text", "i": 42, "f": 3.14, "b": True}

    def test_strips_control_chars_from_strings(self):
        result = GraphStore._clean_properties({"text": "a\x00b\x01c\t\n\r"})
        assert result["text"] == "abc\t\n\r"

    def test_preserves_lists(self):
        result = GraphStore._clean_properties({"tags": ["a", "b"]})
        assert result["tags"] == ["a", "b"]

    def test_strips_control_chars_from_list_strings(self):
        result = GraphStore._clean_properties({"tags": ["a\x00", "b\x01", 3]})
        assert result["tags"] == ["a", "b", 3]

    def test_converts_objects_to_str(self):
        result = GraphStore._clean_properties({"obj": {"nested": True}})
        assert isinstance(result["obj"], str)

    def test_empty_dict(self):
        assert GraphStore._clean_properties({}) == {}


class TestRelationshipLabelHints:
    async def test_known_rel_type_uses_label_hints(self, graph_store, mock_connection):
        """Known relationship types should use label hints in MATCH."""
        rels = [
            GraphRelationship(
                start_node_id="doc1",
                end_node_id="chunk1",
                type="PART_OF",
                properties={"index": 0},
            )
        ]
        await graph_store.upsert_relationships(rels)
        cypher = mock_connection.query.call_args[0][0]
        assert "`Document`" in cypher
        assert "`Chunk`" in cypher

    async def test_unknown_rel_type_defaults_to_entity(self, graph_store, mock_connection):
        """Unknown relationship types should default to __Entity__ labels."""
        rels = [
            GraphRelationship(
                start_node_id="a",
                end_node_id="b",
                type="CUSTOM_REL",
                properties={},
            )
        ]
        await graph_store.upsert_relationships(rels)
        cypher = mock_connection.query.call_args[0][0]
        assert "`__Entity__`" in cypher

    async def test_mentioned_in_uses_correct_labels(self, graph_store, mock_connection):
        """MENTIONED_IN should use __Entity__ → Chunk labels."""
        rels = [
            GraphRelationship(
                start_node_id="entity1",
                end_node_id="chunk1",
                type="MENTIONED_IN",
                properties={},
            )
        ]
        await graph_store.upsert_relationships(rels)
        cypher = mock_connection.query.call_args[0][0]
        assert "`__Entity__`" in cypher
        assert "`Chunk`" in cypher


class TestGraphStoreDocumentLifecycle:
    """v1.1.0: Cypher-layer methods used by GraphRAG.update() / delete_document()."""

    async def test_get_document_record_returns_dict(self, graph_store, mock_connection):
        result_mock = MagicMock()
        result_mock.result_set = [["docs/a.md", "abc123"]]
        mock_connection.query = AsyncMock(return_value=result_mock)

        record = await graph_store.get_document_record("docs/a.md")
        assert record == {"path": "docs/a.md", "content_hash": "abc123"}
        cypher = mock_connection.query.call_args[0][0]
        assert "MATCH (d:Document {id: $id})" in cypher
        assert "content_hash" in cypher

    async def test_get_document_record_returns_none_when_missing(
        self, graph_store, mock_connection
    ):
        result_mock = MagicMock()
        result_mock.result_set = []
        mock_connection.query = AsyncMock(return_value=result_mock)

        record = await graph_store.get_document_record("ghost")
        assert record is None

    async def test_get_document_record_handles_pre_1_1_0_docs(
        self, graph_store, mock_connection
    ):
        """Documents ingested before v1.1.0 lack content_hash; return None
        for the hash (the update() short-circuit then falls through to a
        full update — fail-safe)."""
        result_mock = MagicMock()
        result_mock.result_set = [["docs/old.md", None]]
        mock_connection.query = AsyncMock(return_value=result_mock)

        record = await graph_store.get_document_record("docs/old.md")
        assert record == {"path": "docs/old.md", "content_hash": None}

    async def test_get_document_entity_candidates_returns_distinct_ids(
        self, graph_store, mock_connection
    ):
        result_mock = MagicMock()
        result_mock.result_set = [["e1"], ["e2"], ["e3"]]
        mock_connection.query = AsyncMock(return_value=result_mock)

        candidates = await graph_store.get_document_entity_candidates("docs/a.md")
        assert candidates == ["e1", "e2", "e3"]
        cypher = mock_connection.query.call_args[0][0]
        # Critical: must traverse MENTIONED_IN → Chunk → PART_OF → Document.
        assert "MENTIONED_IN" in cypher
        assert "PART_OF" in cypher
        assert "DISTINCT" in cypher

    async def test_cleanup_pending_documents_skips_committed(
        self, graph_store, mock_connection
    ):
        """v1.1.0 state-machine: cleanup MUST NOT delete a pending whose
        ready_to_commit=true — that pending was committed by a prior call
        that crashed before completing the cutover. Discarding it would
        be silent data loss; the next call's Phase 0 must roll forward."""
        result_mock = MagicMock()
        result_mock.result_set = [[2]]
        mock_connection.query = AsyncMock(return_value=result_mock)

        n = await graph_store.cleanup_pending_documents("docs/a.md")
        assert n == 2
        cypher = mock_connection.query.call_args[0][0]
        assert "STARTS WITH" in cypher
        # Critical: WHERE clause excludes committed pendings.
        assert "ready_to_commit IS NULL OR p.ready_to_commit = false" in cypher
        assert "DETACH DELETE" in cypher
        params = mock_connection.query.call_args[0][1]
        assert params["prefix"] == "docs/a.md__pending__"

    async def test_find_pending_returns_committed_state(
        self, graph_store, mock_connection
    ):
        """find_pending reports COMMITTED when ready_to_commit is truthy."""
        result_mock = MagicMock()
        result_mock.result_set = [["docs/a.md__pending__abc12345", True, "newhash"]]
        mock_connection.query = AsyncMock(return_value=result_mock)

        out = await graph_store.find_pending("docs/a.md")
        assert out is not None
        state, pid, hash_ = out
        assert state == "COMMITTED"
        assert pid == "docs/a.md__pending__abc12345"
        assert hash_ == "newhash"

    async def test_find_pending_returns_written_state(
        self, graph_store, mock_connection
    ):
        """find_pending reports WRITTEN when there's no commit marker."""
        result_mock = MagicMock()
        result_mock.result_set = [["docs/a.md__pending__deadbeef", None, None]]
        mock_connection.query = AsyncMock(return_value=result_mock)

        out = await graph_store.find_pending("docs/a.md")
        assert out is not None
        state, pid, _ = out
        assert state == "WRITTEN"
        assert pid == "docs/a.md__pending__deadbeef"

    async def test_find_pending_returns_none_when_no_pending(
        self, graph_store, mock_connection
    ):
        result_mock = MagicMock()
        result_mock.result_set = []
        mock_connection.query = AsyncMock(return_value=result_mock)
        assert await graph_store.find_pending("docs/a.md") is None

    async def test_mark_pending_committed_is_single_atomic_statement(
        self, graph_store, mock_connection
    ):
        """The commit point must be ONE Cypher statement. Splitting it across
        multiple round-trips would re-introduce the original atomicity bug."""
        result_mock = MagicMock()
        result_mock.result_set = [[1]]
        mock_connection.query = AsyncMock(return_value=result_mock)

        n = await graph_store.mark_pending_committed("docs/a.md__pending__abc12345")
        assert n == 1
        # Exactly one query, no chained statements (no semicolons that would
        # split into multiple statements per FalkorDB execution).
        assert mock_connection.query.await_count == 1
        cypher = mock_connection.query.call_args[0][0]
        assert "SET p.ready_to_commit = true" in cypher

    async def test_rollforward_cutover_runs_three_idempotent_ops(
        self, graph_store, mock_connection
    ):
        """Rollforward is delete-chunks → delete-doc → rename-pending.
        Each op is safe to replay (idempotent)."""
        results = [
            MagicMock(result_set=[[5]]),  # 1. delete chunks
            MagicMock(result_set=[]),     # 2. delete old doc
            MagicMock(result_set=[]),     # 3. rename pending + clear marker
        ]
        mock_connection.query = AsyncMock(side_effect=results)

        chunks_removed = await graph_store.rollforward_cutover(
            pending_id="docs/a.md__pending__abc12345",
            real_id="docs/a.md",
            path="docs/a.md",
            content_hash="newhash",
        )
        assert chunks_removed == 5
        assert mock_connection.query.await_count == 3

        rename_cypher = mock_connection.query.await_args_list[2][0][0]
        assert "SET p.id = $real_id" in rename_cypher
        assert "p.path = $path" in rename_cypher
        assert "p.content_hash = $hash" in rename_cypher
        # Rename also clears the commit marker — otherwise the FINAL
        # state would still report as a pending Document.
        assert "p.ready_to_commit = NULL" in rename_cypher

    async def test_rollforward_idempotent_when_chunks_already_gone(
        self, graph_store, mock_connection
    ):
        """If a prior rollforward attempt completed step 1 (delete chunks)
        and crashed, the replay's delete-chunks must be a no-op (count 0),
        not an error."""
        results = [
            MagicMock(result_set=[[0]]),  # 1. delete chunks → already gone
            MagicMock(result_set=[]),
            MagicMock(result_set=[]),
        ]
        mock_connection.query = AsyncMock(side_effect=results)

        chunks_removed = await graph_store.rollforward_cutover(
            pending_id="docs/a.md__pending__abc12345",
            real_id="docs/a.md",
            path="docs/a.md",
            content_hash="newhash",
        )
        assert chunks_removed == 0
        assert mock_connection.query.await_count == 3

    async def test_delete_document_chunks_and_node(self, graph_store, mock_connection):
        results = [
            MagicMock(result_set=[[3]]),  # 1. delete chunks
            MagicMock(result_set=[]),     # 2. delete document node
        ]
        mock_connection.query = AsyncMock(side_effect=results)

        chunks_removed = await graph_store.delete_document_chunks_and_node("docs/a.md")
        assert chunks_removed == 3
        assert mock_connection.query.await_count == 2

    async def test_delete_orphan_entities_skips_when_empty(
        self, graph_store, mock_connection
    ):
        """No candidates → no Cypher (saves a roundtrip)."""
        n = await graph_store.delete_orphan_entities([])
        assert n == 0
        mock_connection.query.assert_not_called()

    async def test_delete_orphan_entities_filters_by_mentioned_in(
        self, graph_store, mock_connection
    ):
        """The WHERE clause must filter to entities with NO remaining
        MENTIONED_IN — that's how shared entities (still mentioned by
        other documents) are preserved."""
        result_mock = MagicMock()
        result_mock.result_set = [[1]]  # one orphan deleted
        mock_connection.query = AsyncMock(return_value=result_mock)

        n = await graph_store.delete_orphan_entities(["e1", "e2", "e3"])
        assert n == 1
        cypher = mock_connection.query.call_args[0][0]
        assert "WHERE NOT (e)-[:MENTIONED_IN]->(:Chunk)" in cypher
        assert "DETACH DELETE" in cypher
        params = mock_connection.query.call_args[0][1]
        assert params["ids"] == ["e1", "e2", "e3"]

    async def test_delete_orphan_entities_batches_large_lists(
        self, graph_store, mock_connection
    ):
        """A list larger than the batch size must split into multiple
        round-trips so the params payload stays bounded."""
        # 1200 ids with batch size 500 → 3 calls (500 + 500 + 200)
        ids = [f"e{i}" for i in range(1200)]
        results = [
            MagicMock(result_set=[[2]]),
            MagicMock(result_set=[[3]]),
            MagicMock(result_set=[[1]]),
        ]
        mock_connection.query = AsyncMock(side_effect=results)

        n = await graph_store.delete_orphan_entities(ids)
        assert n == 6
        assert mock_connection.query.await_count == 3
