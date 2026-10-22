"""Tests for storage/graph_store.py — Repository pattern for graph operations."""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.exceptions import DatabaseError
from graphrag_sdk.core.models import GraphNode, GraphRelationship
from graphrag_sdk.storage.graph_store import GraphStore, ReferenceNode


@pytest.fixture
def graph_store(mock_connection):
    return GraphStore(mock_connection)


def _is_index_housekeeping(cypher: str) -> bool:
    return "CREATE INDEX" in cypher or "db.indexes()" in cypher


def upsert_calls(mock_connection):
    """The write queries a call made, excluding the indexes it ensures first.

    ``upsert_nodes`` range-indexes a label the first time it writes to it,
    because a MERGE can only use an index on the label in its own pattern, and
    reads what the graph already indexes once per instance. Asserting a raw call
    count here would make every test in this class a tripwire for that detail.
    """
    return [
        call
        for call in mock_connection.query.call_args_list
        if not _is_index_housekeeping(call[0][0])
    ]


class TestGraphStoreUpsertNodes:
    async def test_upsert_single_node(self, graph_store, mock_connection):
        nodes = [GraphNode(id="n1", label="Person", properties={"name": "Alice"})]
        result = await graph_store.upsert_nodes(nodes)
        assert result == 1
        writes = upsert_calls(mock_connection)
        assert len(writes) == 1
        cypher = writes[0][0][0]
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
        # One write per label, and an id index ensured for each.
        assert len(upsert_calls(mock_connection)) == 2

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
        params = upsert_calls(mock_connection)[0][0][1]
        assert params["batch"][0]["id"] == "test-id"

    async def test_upsert_sanitizes_control_chars_in_batch_params(
        self, graph_store, mock_connection
    ):
        await graph_store.upsert_nodes(
            [GraphNode(id="id\x00\x01", label="Chunk", properties={"text": "A\x00B\x01C"})]
        )
        params = upsert_calls(mock_connection)[0][0][1]
        assert params["batch"][0]["id"] == "id"
        assert params["batch"][0]["properties"]["text"] == "ABC"

    async def test_upsert_sanitizes_control_chars_in_fallback_params(
        self, graph_store, mock_connection
    ):
        """Per-item fallback path should also use sanitized IDs and properties."""

        # Fail the batch write specifically, rather than "the first query": the
        # upsert also ensures an id index, and positional side effects would put
        # the failure on the wrong call.
        async def fail_the_batch(cypher, params=None):
            if "UNWIND" in cypher:
                raise Exception("batch fail")
            return MagicMock()

        mock_connection.query = AsyncMock(side_effect=fail_the_batch)
        await graph_store.upsert_nodes(
            [GraphNode(id="id\x00\x01", label="X", properties={"t": "A\x00B"})]
        )
        fallback = [
            call
            for call in mock_connection.query.call_args_list
            if "UNWIND" not in call[0][0] and not _is_index_housekeeping(call[0][0])
        ]
        assert len(fallback) == 1, "the per-item fallback should have run once"
        fallback_params = fallback[0][0][1]
        assert fallback_params["id"] == "id"
        assert fallback_params["properties"]["t"] == "AB"


class TestUpsertNodesKeepsProvenance:
    """Entity-level ``source_chunk_ids`` is a union across writes, never a replace.

    Two documents mentioning one entity both leave their chunks on it; the
    ``MENTIONED_IN`` edges already said so and the property must agree.
    """

    async def test_an_entity_write_unions_source_chunk_ids(self, graph_store, mock_connection):
        await graph_store.upsert_nodes(
            [GraphNode(id="acme", label="Organization", properties={"source_chunk_ids": ["c2"]})]
        )
        cypher = upsert_calls(mock_connection)[0][0][0]
        assert "coalesce(n.source_chunk_ids, []) AS old" in cypher
        assert "SET n += item.properties" in cypher
        assert "n.source_chunk_ids = CASE WHEN size(contrib) = 0" in cypher
        assert "old + [c IN contrib WHERE NOT c IN old]" in cypher
        assert cypher.index("SET n += item.properties") < cypher.index("n.source_chunk_ids = CASE")

    async def test_a_structural_write_replaces_as_before(self, graph_store, mock_connection):
        await graph_store.upsert_nodes(
            [GraphNode(id="c1", label="Chunk", properties={"source_chunk_ids": ["c1"]})]
        )
        cypher = upsert_calls(mock_connection)[0][0][0]
        assert "SET n += item.properties" in cypher
        assert "source_chunk_ids = CASE" not in cypher

    async def test_the_per_item_fallback_unions_too(self, graph_store, mock_connection):
        async def fail_the_batch(cypher, params=None):
            if "UNWIND" in cypher:
                raise Exception("batch fail")
            return MagicMock()

        mock_connection.query = AsyncMock(side_effect=fail_the_batch)
        await graph_store.upsert_nodes(
            [GraphNode(id="acme", label="Organization", properties={"name": "Acme"})]
        )
        fallback = [
            call[0][0]
            for call in mock_connection.query.call_args_list
            if "UNWIND" not in call[0][0] and not _is_index_housekeeping(call[0][0])
        ]
        assert len(fallback) == 1
        assert "n.source_chunk_ids = CASE WHEN size(contrib) = 0" in fallback[0]


class TestOneKeyTwiceInOneCall:
    """A batch holding one key twice gave engine-dependent values and dropped
    one write's chunks, so writes sharing a key are folded before sending."""

    async def test_duplicate_entity_ids_become_one_row(self, graph_store, mock_connection):
        count = await graph_store.upsert_nodes(
            [
                GraphNode(
                    id="x", label="Person", properties={"name": "A", "source_chunk_ids": ["c1"]}
                ),
                GraphNode(id="y", label="Person", properties={"name": "Y"}),
                GraphNode(
                    id="x",
                    label="Person",
                    properties={"name": "B", "age": 3, "source_chunk_ids": ["c2", "c1"]},
                ),
                GraphNode(id="x", label="Person", properties={"source_chunk_ids": []}),
            ]
        )
        batch = upsert_calls(mock_connection)[0][0][1]["batch"]
        assert batch == [
            {"id": "x", "properties": {"name": "B", "age": 3, "source_chunk_ids": ["c1", "c2"]}},
            {"id": "y", "properties": {"name": "Y"}},
        ]
        assert count == 4

    async def test_a_structural_label_keeps_the_last_list(self, graph_store, mock_connection):
        await graph_store.upsert_nodes(
            [
                GraphNode(id="c", label="Chunk", properties={"source_chunk_ids": ["a"]}),
                GraphNode(id="c", label="Chunk", properties={"source_chunk_ids": ["b"]}),
            ]
        )
        batch = upsert_calls(mock_connection)[0][0][1]["batch"]
        assert batch == [{"id": "c", "properties": {"source_chunk_ids": ["b"]}}]

    async def test_duplicate_relates_keys_become_one_row(self, graph_store, mock_connection):
        def rel(rel_type, chunks, **extra):
            return GraphRelationship(
                start_node_id="a",
                end_node_id="b",
                type="RELATES",
                properties={"rel_type": rel_type, "source_chunk_ids": chunks, **extra},
            )

        count = await graph_store.upsert_relationships(
            [rel("KNOWS", ["c1"], w=1), rel("WORKS_AT", ["c9"]), rel("KNOWS", ["c2"], w=2)]
        )
        batch = upsert_calls(mock_connection)[0][0][1]["batch"]
        assert [item["properties"] for item in batch] == [
            {"rel_type": "KNOWS", "source_chunk_ids": ["c1", "c2"], "w": 2},
            {"rel_type": "WORKS_AT", "source_chunk_ids": ["c9"]},
        ]
        assert count == 3

    async def test_rel_type_keys_compare_like_the_merge(self, graph_store, mock_connection):
        """FalkorDB's MERGE treats 5 and 5.0 as one key but keeps true apart
        from 1 and "5" apart from 5; the fold must agree or a batch still
        carries one MERGE key twice."""

        def rel(rel_type, chunk):
            return GraphRelationship(
                start_node_id="a",
                end_node_id="b",
                type="RELATES",
                properties={"rel_type": rel_type, "source_chunk_ids": [chunk]},
            )

        await graph_store.upsert_relationships(
            [rel(5, "c1"), rel(5.0, "c2"), rel(True, "c3"), rel(1, "c4"), rel("5", "c5")]
        )
        batch = upsert_calls(mock_connection)[0][0][1]["batch"]
        assert [item["properties"]["source_chunk_ids"] for item in batch] == [
            ["c1", "c2"],  # 5 and 5.0: one key
            ["c3"],  # true
            ["c4"],  # 1
            ["c5"],  # "5"
        ]

    async def test_other_edge_types_are_keyed_by_the_pair(self, graph_store, mock_connection):
        await graph_store.upsert_relationships(
            [
                GraphRelationship(
                    start_node_id="a", end_node_id="b", type="LIKES", properties={"w": 1}
                ),
                GraphRelationship(
                    start_node_id="a", end_node_id="b", type="LIKES", properties={"w": 2}
                ),
            ]
        )
        batch = upsert_calls(mock_connection)[0][0][1]["batch"]
        assert batch == [{"start_id": "a", "end_id": "b", "properties": {"w": 2}}]

    @staticmethod
    def _fail_batches(mock_connection):
        """Fail every UNWIND batch, so each key goes through the per-item path."""

        async def query(cypher, params=None):
            if "UNWIND" in cypher:
                raise Exception("batch fail")
            result = MagicMock()
            result.result_set = []
            return result

        mock_connection.query = AsyncMock(side_effect=query)

    @staticmethod
    def _per_item_calls(mock_connection):
        return [
            call[0]
            for call in mock_connection.query.call_args_list
            if "UNWIND" not in call[0][0] and not _is_index_housekeeping(call[0][0])
        ]

    async def test_the_per_item_fallback_sends_each_node_key_once(
        self, graph_store, mock_connection
    ):
        """A failed batch falls back to one query per row; a key that appeared
        twice must still be one write carrying both writes' chunks."""
        self._fail_batches(mock_connection)

        count = await graph_store.upsert_nodes(
            [
                GraphNode(
                    id="x", label="Person", properties={"name": "A", "source_chunk_ids": ["c1"]}
                ),
                GraphNode(
                    id="x", label="Person", properties={"name": "B", "source_chunk_ids": ["c2"]}
                ),
            ]
        )

        calls = self._per_item_calls(mock_connection)
        assert [params["properties"] for _, params in calls] == [
            {"name": "B", "source_chunk_ids": ["c1", "c2"]}
        ]
        assert count == 2

    async def test_the_per_item_fallback_sends_each_edge_key_once(
        self, graph_store, mock_connection
    ):
        self._fail_batches(mock_connection)

        def rel(rel_type, chunks):
            return GraphRelationship(
                start_node_id="a",
                end_node_id="b",
                type="RELATES",
                properties={"rel_type": rel_type, "source_chunk_ids": chunks},
            )

        count = await graph_store.upsert_relationships(
            [rel("KNOWS", ["c1"]), rel("WORKS_AT", ["c9"]), rel("KNOWS", ["c2"])]
        )

        calls = self._per_item_calls(mock_connection)
        assert sorted(
            (params["properties"]["rel_type"], params["properties"]["source_chunk_ids"])
            for _, params in calls
        ) == [("KNOWS", ["c1", "c2"]), ("WORKS_AT", ["c9"])]
        assert count == 3

    def test_repeats_extend_one_list_and_reuse_one_seen_set(self):
        """Copying and scanning the chunk list on every repeat cost O(n²): one
        edge repeated on 100,000 table rows took 28 s, blocking the event loop.
        Each repeat must extend the same list and reuse the same set."""
        merged, seen = GraphStore._fold_properties(
            {"source_chunk_ids": ["c1"]}, {"source_chunk_ids": ["c2"]}, union_provenance=True
        )
        chunk_ids = merged["source_chunk_ids"]

        merged, seen_again = GraphStore._fold_properties(
            merged, {"source_chunk_ids": ["c3", "c1"]}, union_provenance=True, seen=seen
        )

        assert seen_again is seen
        assert merged["source_chunk_ids"] is chunk_ids
        assert chunk_ids == ["c1", "c2", "c3"]
        assert seen == {"c1", "c2", "c3"}

    async def test_a_key_repeated_on_every_row_keeps_every_chunk_in_order(
        self, graph_store, mock_connection
    ):
        rels = [
            GraphRelationship(
                start_node_id="store",
                end_node_id="region",
                type="RELATES",
                properties={"rel_type": "IN_REGION", "source_chunk_ids": [f"row{i}"]},
            )
            for i in range(5_000)
        ]

        count = await graph_store.upsert_relationships(rels)

        batch = upsert_calls(mock_connection)[0][0][1]["batch"]
        assert len(batch) == 1
        assert batch[0]["properties"]["source_chunk_ids"] == [f"row{i}" for i in range(5_000)]
        assert count == 5_000


class TestUpsertReferenceNodes:
    """A link's target is keyed always and named only ON CREATE."""

    REFERENCE = ReferenceNode(
        "acme",
        "Organization",
        "Acme",
        {
            "employees__org_id": "O1",
            "entity_key": "O1",
            "is_stub": True,
            "source_chunk_ids": ["rows"],
        },
    )

    async def test_an_existing_node_gets_the_key_and_keeps_its_name(
        self, graph_store, mock_connection
    ):
        await graph_store.upsert_reference_nodes([self.REFERENCE])
        cypher, params = mock_connection.query.call_args[0]
        assert "ON CREATE SET n += item.properties" in cypher
        assert "ON MATCH SET n += item.claims" in cypher
        assert "n.entity_key = coalesce(n.entity_key, item.properties.entity_key)" in cypher
        assert "n.is_stub = coalesce(n.is_stub, true)" in cypher
        item = params["batch"][0]
        assert item["name"] == "Acme"
        assert item["claims"] == {"employees__org_id": "O1"}, (
            "the one claim a reference may add to an existing node is its signed key: never"
            " the name, not the unsigned identity it gets via coalesce, and not provenance,"
            " which is unioned like every other write's"
        )
        assert "n.source_chunk_ids = CASE WHEN size(contrib) = 0" in cypher

    async def test_empty_input_writes_nothing(self, graph_store, mock_connection):
        assert await graph_store.upsert_reference_nodes([]) == 0
        mock_connection.query.assert_not_called()


class TestStripStaleProvenance:
    async def test_removes_the_deleted_chunks_from_the_candidates(
        self, graph_store, mock_connection
    ):
        result_mock = MagicMock()
        result_mock.result_set = [[3]]
        mock_connection.query = AsyncMock(return_value=result_mock)

        trimmed = await graph_store.strip_stale_provenance(["e1", "e2"], ["c1", "c2"])

        assert trimmed == 3
        cypher, params = mock_connection.query.call_args[0]
        assert "UNWIND $ids AS eid" in cypher
        assert "any(c IN e.source_chunk_ids WHERE c IN $old_chunks)" in cypher
        assert (
            "SET e.source_chunk_ids = [c IN e.source_chunk_ids WHERE NOT c IN $old_chunks]"
        ) in cypher
        assert params["ids"] == ["e1", "e2"]
        assert params["old_chunks"] == ["c1", "c2"]

    async def test_nothing_to_do_is_no_query(self, graph_store, mock_connection):
        assert await graph_store.strip_stale_provenance([], ["c1"]) == 0
        assert await graph_store.strip_stale_provenance(["e1"], []) == 0
        mock_connection.query.assert_not_called()


class TestReleaseEntityKeys:
    async def test_identity_goes_unless_another_table_still_keys_the_node(
        self, graph_store, mock_connection
    ):
        removed, demoted = MagicMock(result_set=[[2]]), MagicMock(result_set=[[1]])
        mock_connection.query = AsyncMock(side_effect=[removed, demoted])

        released = await graph_store.release_entity_keys(
            "Organization", owned_by=["orgs__org_id"], referenced_by=["grants__org_id"]
        )

        assert released == 3
        remove, demote = [call[0][0] for call in mock_connection.query.call_args_list]
        assert "n.entity_key IS NOT NULL" in remove
        assert "n.`orgs__org_id` IS NULL AND n.`grants__org_id` IS NULL" in remove
        assert "REMOVE n.entity_key, n.is_stub" in remove
        assert "n.is_stub = false" in demote
        assert "n.`orgs__org_id` IS NULL" in demote
        assert "grants__org_id" not in demote, "a link elsewhere does not keep a node a row"
        assert "SET n.is_stub = true" in demote

    async def test_no_other_table_means_every_keyed_node_is_released(
        self, graph_store, mock_connection
    ):
        await graph_store.release_entity_keys("Organization", owned_by=[], referenced_by=[])
        remove, demote = [call[0][0] for call in mock_connection.query.call_args_list]
        assert "REMOVE n.entity_key, n.is_stub" in remove
        assert " AND n." not in remove.split("WHERE", 1)[1].split("REMOVE")[0]
        assert "SET n.is_stub = true" in demote

    async def test_scoped_to_ids_touches_only_those_nodes(self, graph_store, mock_connection):
        await graph_store.release_entity_keys(
            "Person", owned_by=["hr__employee_id"], referenced_by=[], ids=["a", "b"]
        )
        remove, demote = mock_connection.query.call_args_list
        assert " AND n.id IN $ids " in remove[0][0]
        assert remove[0][1] == {"ids": ["a", "b"]}
        assert " AND n.id IN $ids " in demote[0][0]
        assert demote[0][1] == {"ids": ["a", "b"]}

    async def test_an_empty_scope_is_not_the_whole_label(self, graph_store, mock_connection):
        released = await graph_store.release_entity_keys(
            "Person", owned_by=[], referenced_by=[], ids=[]
        )
        assert released == 0
        mock_connection.query.assert_not_called()


class TestReconcileKeyedIdentity:
    """A table may only move the nodes it wrote, or the placeholders left for it."""

    async def test_only_stubs_and_this_tables_rows_qualify(self, graph_store, mock_connection):
        mock_connection.query = AsyncMock(return_value=MagicMock(result_set=[]))

        await graph_store.reconcile_keyed_identity(
            "Person", "hr__employee_id", [("1", "alice_smith__person")]
        )

        owned, placeholders, contested = [
            call[0][0]
            for call in mock_connection.query.call_args_list
            if not _is_index_housekeeping(call[0][0])
        ]
        assert "{`hr__employee_id`: it.k}" in owned and "n.id <> it.new_id" in owned, (
            "a row is claimed by the value of the key as this table signs it. entity_key is"
            " one slot every table writes: a person hr.csv numbers 1 and crm.csv numbers 2"
            " holds whichever came last, and matching on it let hr.csv's row 2 take her"
        )
        assert "IS NOT NULL" not in owned, "existence of the signed key is not a match"
        assert "{entity_key: it.k}" in placeholders and "n.is_stub = true" in placeholders, (
            "a placeholder knows only the key a link gave it"
        )
        assert "NOT coalesce(n.is_stub, false)" in contested and (
            "n.`hr__employee_id` IS NULL OR n.`hr__employee_id` <> it.k" in contested
        ), (
            "a placeholder a link parked on because two real nodes answer to the key is"
            " not claimed for whichever table re-syncs first; this table's own row for"
            " the key is not such a rival"
        )

    async def test_the_key_it_matches_on_is_indexed_first(self, graph_store, mock_connection):
        mock_connection.query = AsyncMock(return_value=MagicMock(result_set=[]))

        await graph_store.reconcile_keyed_identity("Person", "hr__employee_id", [("1", "x")])
        await graph_store.resolve_by_entity_key("Person", ["1"])

        indexes = [
            call[0][0]
            for call in mock_connection.query.call_args_list
            if "CREATE INDEX" in call[0][0]
        ]
        # One statement: FalkorDB 6.0.2 sometimes left the last of several
        # back-to-back CREATE INDEX statements on one label empty.
        assert indexes == [
            "CREATE INDEX FOR (n:`Person`) ON (n.id, n.entity_key, n.`hr__employee_id`)",
        ], "both lookups run once per keyed row; indexed once per label, not per call"

    async def test_an_index_the_graph_already_has_is_not_created_again(
        self, graph_store, mock_connection
    ):
        """Reopening a graph used to fail one CREATE per label and log it as an error."""
        listing = MagicMock(result_set=[["Person", ["id", "entity_key"]], ["Chunk", ["uid"]]])
        mock_connection.query = AsyncMock(
            side_effect=lambda cypher, *a, **k: (
                listing if "YIELD label, properties" in cypher else MagicMock(result_set=[])
            )
        )

        await graph_store.resolve_by_entity_key("Person", ["1"])
        await graph_store.resolve_by_entity_key("Organization", ["O1"])

        queries = [call[0][0] for call in mock_connection.query.call_args_list]
        assert sum("YIELD label, properties" in q for q in queries) == 1, "read once per instance"
        assert [q for q in queries if "CREATE INDEX" in q] == [
            "CREATE INDEX FOR (n:`Organization`) ON (n.id, n.entity_key)",
        ]

    async def test_a_legacy_description_vector_is_not_carried_to_the_new_node(
        self, graph_store, mock_connection
    ):
        """An earlier build cached description vectors on entity nodes; folding
        such a node into another during ingest must not copy them across."""
        keep = {"id": "new"}
        dup = {
            "id": "old",
            "description_embedding": [0.1, 0.2],
            "description_embedding_hash": "h",
            "role": "engineer",
        }
        mock_connection.query = AsyncMock(
            side_effect=lambda cypher, *a, **k: MagicMock(
                result_set=[[keep, dup]] if "RETURN properties(k)" in cypher else []
            )
        )

        await graph_store._carry_then_delete("old", "new")

        (carry,) = [
            call[0][1]["carry"]
            for call in mock_connection.query.call_args_list
            if "SET k += $carry" in call[0][0]
        ]
        assert carry == {"role": "engineer"}

    async def test_nothing_to_reconcile_touches_nothing(self, graph_store, mock_connection):
        assert await graph_store.reconcile_keyed_identity("Person", "hr__employee_id", []) == {}
        mock_connection.query.assert_not_called()


class TestDropNodeProperty:
    async def test_unscoped_is_the_whole_label(self, graph_store, mock_connection):
        mock_connection.query = AsyncMock(return_value=MagicMock(result_set=[[4]]))
        touched = await graph_store.drop_node_property("Person", "hr__age")
        assert touched == 4
        cypher, params = mock_connection.query.call_args[0]
        assert "MATCH (n:`Person`) WHERE n.`hr__age` IS NOT NULL REMOVE n.`hr__age`" in cypher
        assert params is None

    async def test_scoped_to_ids(self, graph_store, mock_connection):
        await graph_store.drop_node_property("Person", "hr__age", ids=["x"])
        cypher, params = mock_connection.query.call_args[0]
        assert "IS NOT NULL AND n.id IN $ids REMOVE" in cypher
        assert params == {"ids": ["x"]}

    async def test_an_empty_scope_touches_nothing(self, graph_store, mock_connection):
        assert await graph_store.drop_node_property("Person", "hr__age", ids=[]) == 0
        mock_connection.query.assert_not_called()


class TestEntitiesOutsideDocument:
    async def test_reads_the_candidates_the_document_no_longer_mentions(
        self, graph_store, mock_connection
    ):
        mock_connection.query = AsyncMock(return_value=MagicMock(result_set=[["gone"], [None]]))
        left = await graph_store.entities_outside_document(["gone", "kept"], "people.csv")
        assert left == ["gone"]
        cypher, params = mock_connection.query.call_args[0]
        assert "NOT (e)-[:MENTIONED_IN]->(:Chunk)<-[:PART_OF]-(:Document {id: $doc})" in cypher
        assert params == {"ids": ["gone", "kept"], "doc": "people.csv"}

    async def test_no_candidates_no_query(self, graph_store, mock_connection):
        assert await graph_store.entities_outside_document([], "people.csv") == []
        mock_connection.query.assert_not_called()


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

    async def test_relationship_source_ownership_is_unioned(self, graph_store, mock_connection):
        """In the batch write and in the per-row fallback a failed batch takes.

        ``SET r += ...`` alone replaces the list, so a second relationship
        table writing an edge would take it from the first, and the first
        table's next re-ingest or drop would delete an edge the second still
        lists.
        """

        async def fail_batch(cypher, params=None):
            if "UNWIND $batch AS item" in cypher:
                raise Exception("fail")
            return MagicMock()

        mock_connection.query = AsyncMock(side_effect=fail_batch)
        await graph_store.upsert_relationships(
            [
                GraphRelationship(
                    start_node_id="a",
                    end_node_id="b",
                    type="RELATES",
                    properties={
                        "rel_type": "CONSUMED_BATCH",
                        "structured_sources": ["consumed_batch"],
                    },
                )
            ]
        )
        batch, fallback = (call[0][0] for call in upsert_calls(mock_connection))
        for cypher, incoming in ((batch, "item.properties"), (fallback, "$properties")):
            assert "coalesce(r.structured_sources, []) AS old_sources" in cypher
            assert f"coalesce({incoming}.structured_sources, []) AS sources" in cypher
            assert (
                "r.structured_sources = old_sources + [s IN sources WHERE NOT s IN old_sources]"
            ) in cypher
            assert cypher.index("AS old_sources") < cypher.index("SET r +=")
            assert cypher.index("SET r +=") < cypher.index("r.structured_sources = old_sources")
            assert "old_snapshot" not in cypher

    async def test_snapshot_number_never_moves_back(self, graph_store, mock_connection):
        """A run that started earlier and writes later keeps the newer run's number."""

        async def fail_batch(cypher, params=None):
            if "UNWIND $batch AS item" in cypher:
                raise Exception("fail")
            return MagicMock()

        mock_connection.query = AsyncMock(side_effect=fail_batch)
        await graph_store.upsert_relationships(
            [
                GraphRelationship(
                    start_node_id="a",
                    end_node_id="b",
                    type="RELATES",
                    properties={
                        "rel_type": "CONSUMED_BATCH",
                        "structured_sources": ["consumed_batch"],
                        "relationship_mapping_consumed_batch": 7,
                    },
                )
            ],
            snapshot_property="relationship_mapping_consumed_batch",
        )
        batch, fallback = (call[0][0] for call in upsert_calls(mock_connection))
        for cypher, incoming in ((batch, "item.properties"), (fallback, "$properties")):
            assert "r.`relationship_mapping_consumed_batch` AS old_snapshot" in cypher
            assert cypher.index("AS old_snapshot") < cypher.index("SET r +=")
            assert (
                "SET r.`relationship_mapping_consumed_batch` = CASE WHEN old_snapshot > "
                f"{incoming}.`relationship_mapping_consumed_batch` THEN old_snapshot "
                f"ELSE {incoming}.`relationship_mapping_consumed_batch` END"
            ) in cypher

    async def test_null_signed_relationship_property_is_removed(self, graph_store, mock_connection):
        """A blank cell is sent as null, which ``SET r += ...`` removes.

        A REMOVE clause per combination of blank cells split a 500-row batch
        into up to one query per row: 64 combinations with six optional
        columns.
        """
        await graph_store.upsert_relationships(
            [
                GraphRelationship(
                    start_node_id="a",
                    end_node_id=end,
                    type="RELATES",
                    properties={
                        "rel_type": "CONSUMED_BATCH",
                        "consumed_batch__consumed_on": on,
                        "consumed_batch__shift": shift,
                        "note": None,
                    },
                )
                for end, on, shift in (
                    ("b1", None, "night"),
                    ("b2", "2026-09-20", None),
                    ("b3", None, None),
                    ("b4", "2026-09-21", "day"),
                )
            ]
        )
        (write,) = upsert_calls(mock_connection)
        cypher, params = write[0]
        assert "REMOVE" not in cypher
        assert "SET r += item.properties" in cypher
        assert [item["properties"] for item in params["batch"]] == [
            {
                "rel_type": "CONSUMED_BATCH",
                "consumed_batch__consumed_on": None,
                "consumed_batch__shift": "night",
            },
            {
                "rel_type": "CONSUMED_BATCH",
                "consumed_batch__consumed_on": "2026-09-20",
                "consumed_batch__shift": None,
            },
            {
                "rel_type": "CONSUMED_BATCH",
                "consumed_batch__consumed_on": None,
                "consumed_batch__shift": None,
            },
            {
                "rel_type": "CONSUMED_BATCH",
                "consumed_batch__consumed_on": "2026-09-21",
                "consumed_batch__shift": "day",
            },
        ], "only signed (table) properties are cleared; other blanks are left alone as before"

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

        # Fail R1's writes by type rather than by position: the upsert also
        # ensures indexes first, and positional side effects would land on those.
        async def fail_r1(cypher, params=None):
            if not _is_index_housekeeping(cypher) and "`R1`" in cypher:
                raise Exception("fail")
            return MagicMock()

        mock_connection.query = AsyncMock(side_effect=fail_r1)
        rels = [
            GraphRelationship(start_node_id="a", end_node_id="b", type="R1"),
            GraphRelationship(start_node_id="c", end_node_id="d", type="R2"),
        ]
        result = await graph_store.upsert_relationships(rels)
        assert result == 1  # only R2 batch succeeded


class TestRelationshipEndpointResolution:
    async def test_property_lookup_is_batched(self, graph_store, mock_connection):
        keys = [f"K-{index}" for index in range(1001)]
        await graph_store.resolve_by_property("Unit", "units__unit_id", keys)
        lookups = [
            call
            for call in mock_connection.query.call_args_list
            if "UNWIND $keys AS k" in call[0][0]
        ]
        assert [len(call[0][1]["keys"]) for call in lookups] == [500, 500, 1]

    @staticmethod
    def _answer(mock_connection, *found, stripped=None, snapshot=None):
        """Index housekeeping finds nothing. Each cleanup read finds the next
        count of stale edges, then none; each change strips what was found
        (``stripped`` overrides that)."""
        found = iter(found)
        last = {"n": 0}

        async def answer(cypher, params=None, **kwargs):
            if _is_index_housekeeping(cypher):
                return MagicMock(result_set=[])
            if "max(r." in cypher:
                return MagicMock(result_set=[[snapshot]])
            if "LIMIT $limit" in cypher:
                last["n"] = next(found, 0)
                return MagicMock(result_set=[[index, index] for index in range(last["n"])])
            if "UNWIND $edges" in cypher:
                return MagicMock(result_set=[[last["n"] if stripped is None else stripped]])
            return MagicMock(result_set=[])

        mock_connection.query = AsyncMock(side_effect=answer)

    @staticmethod
    def _work(mock_connection):
        return [
            call[0]
            for call in mock_connection.query.call_args_list
            if not _is_index_housekeeping(call[0][0])
        ]

    async def test_snapshot_cleanup_removes_only_one_sources_contribution(
        self, graph_store, mock_connection
    ):
        self._answer(mock_connection, 2)
        deleted = await graph_store.finalize_relationship_snapshot(
            "consumed_batch",
            "relationship_mapping_consumed_batch",
            1_700_000_000_000,
            ["consumed_batch__consumed_on", "consumed_batch__shift"],
        )
        assert deleted == 2
        (find, find_params), (strip, strip_params), (again, _) = self._work(mock_connection)
        assert again == find, "reads again until nothing stale is left"
        # Only rows an older run left are stale; a newer run's are not.
        for cypher, params in ((find, find_params), (strip, strip_params)):
            assert "coalesce(r.`relationship_mapping_consumed_batch`, 0) < $snapshot" in cypher
            assert params["snapshot"] == 1_700_000_000_000
            assert params["signature"] == "consumed_batch"
        assert strip_params["edges"] == [{"a": 0, "r": 0}, {"a": 1, "r": 1}]
        # The change finds each edge again by id and checks it is still stale ...
        assert "WHERE id(a) = edge.a" in strip and "WHERE id(r) = edge.r" in strip
        assert "any(s IN r.structured_sources WHERE s = $signature)" in strip
        # ... strips the source, its marker and every signed property ...
        assert "SET r.structured_sources = rest" in strip
        assert "[s IN r.structured_sources WHERE s <> $signature] AS rest" in strip
        assert (
            "REMOVE r.`relationship_mapping_consumed_batch`, r.`consumed_batch__consumed_on`, "
            "r.`consumed_batch__shift`"
        ) in strip
        # ... and deletes an edge nothing else supports.
        assert "size(rest) = 0 AND size(coalesce(r.source_chunk_ids, [])) = 0" in strip
        assert "FOREACH (x IN unsupported | DELETE x)" in strip

    async def test_snapshot_queries_read_only_the_sources_edges(self, graph_store, mock_connection):
        """Each query used to read every RELATES edge in the graph, documents' too."""
        self._answer(mock_connection, 1, 0, 1, snapshot=7)
        await graph_store.next_relationship_snapshot(
            "relationship_mapping_consumed_batch", "consumed_batch"
        )
        await graph_store.finalize_relationship_snapshot(
            "consumed_batch", "relationship_mapping_consumed_batch", 7, []
        )
        await graph_store.drop_relationship_source("consumed_batch", [])
        await graph_store.clear_relationship_source_property(
            "consumed_batch", "CONSUMED_BATCH", "consumed_batch__shift"
        )
        creates = [
            call[0][0]
            for call in mock_connection.query.call_args_list
            if "CREATE INDEX" in call[0][0]
        ]
        assert creates == ["CREATE INDEX FOR ()-[r:RELATES]-() ON (r.structured_sources)"], (
            "one index shared by every table, created once"
        )
        work = self._work(mock_connection)
        reads = [(cypher, params) for cypher, params in work if "UNWIND $edges" not in cypher]
        assert len(reads) == 6, "the snapshot, two reads per cleanup, the property removal"
        for cypher, params in reads:
            # Not wrapped in coalesce(): FalkorDB 4.x then skips the index.
            assert "WHERE $signature IN r.structured_sources" in cypher
            assert "coalesce(r.structured_sources" not in cypher
            assert params["signature"] == "consumed_batch"
        for cypher, _ in work:
            if "SET r.structured_sources" in cypher:
                # FalkorDB 6.0.2 crashed setting the indexed list on edges a
                # LIMIT left it still reading through that index.
                assert "$signature IN r.structured_sources" not in cypher

    async def test_a_large_cleanup_runs_in_bounded_chunks(self, graph_store, mock_connection):
        """Dropping a 1M-edge table was one write that held the graph for seconds."""
        chunk = graph_store._RELATIONSHIP_CLEANUP_BATCH
        self._answer(mock_connection, chunk, chunk, 3)
        deleted = await graph_store.drop_relationship_source("consumed_batch", [])
        assert deleted == 2 * chunk + 3
        work = self._work(mock_connection)
        finds = [params for cypher, params in work if "LIMIT $limit" in cypher]
        strips = [params for cypher, params in work if "UNWIND $edges" in cypher]
        assert [params["limit"] for params in finds] == [chunk] * 4, "until a read finds none"
        assert [len(params["edges"]) for params in strips] == [chunk, chunk, 3]

    async def test_a_cleanup_that_strips_nothing_stops(self, graph_store, mock_connection):
        """An edge the read keeps finding but the change cannot strip would loop."""
        self._answer(mock_connection, 5, 5, 5, stripped=0)
        assert await graph_store.drop_relationship_source("consumed_batch", []) == 0
        assert len(self._work(mock_connection)) == 2

    async def test_snapshot_number_is_the_clock_or_one_past_the_newest(
        self, graph_store, mock_connection
    ):
        self._answer(mock_connection, snapshot=1_700_000_000_001)
        number = await graph_store.next_relationship_snapshot(
            "relationship_mapping_consumed_batch", "consumed_batch"
        )
        assert number == 1_700_000_000_001
        ((cypher, _),) = self._work(mock_connection)
        assert "max(r.`relationship_mapping_consumed_batch`) AS newest" in cypher
        assert "timestamp() AS now" in cypher
        assert "THEN now ELSE newest + 1 END" in cypher

    async def test_dropping_a_source_outranks_every_run(self, graph_store, mock_connection):
        self._answer(mock_connection, 1)
        await graph_store.drop_relationship_source("consumed_batch", [])
        assert {params["snapshot"] for _, params in self._work(mock_connection)} == {2**63 - 1}


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

    @pytest.mark.parametrize(("max_hops", "ordered"), [(1, True), (2, False), (5, False)])
    async def test_only_one_hop_is_ordered(self, graph_store, mock_connection, max_hops, ordered):
        """An order needs every path before the LIMIT; beyond one hop the
        paths multiply, and 3 hops over 40k nodes took 16.6 s on 4.x."""
        result_mock = MagicMock()
        result_mock.result_set = []
        mock_connection.query = AsyncMock(return_value=result_mock)
        await graph_store.get_connected_entities("c1", max_hops=max_hops)
        cypher = mock_connection.query.call_args[0][0]
        assert ("ORDER BY id LIMIT 50" in cypher) is ordered
        assert cypher.endswith("LIMIT 50")


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

    async def test_get_document_record_returns_typed_record(self, graph_store, mock_connection):
        from graphrag_sdk.core.models import DocumentRecord

        result_mock = MagicMock()
        result_mock.result_set = [["docs/a.md", "abc123"]]
        mock_connection.query = AsyncMock(return_value=result_mock)

        record = await graph_store.get_document_record("docs/a.md")
        assert isinstance(record, DocumentRecord)
        assert record.path == "docs/a.md"
        assert record.content_hash == "abc123"
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

    async def test_get_document_record_handles_pre_1_1_0_docs(self, graph_store, mock_connection):
        """Documents ingested before v1.1.0 lack content_hash; the typed
        record carries None for the hash (the update() short-circuit then
        falls through to a full update — fail-safe)."""
        from graphrag_sdk.core.models import DocumentRecord

        result_mock = MagicMock()
        result_mock.result_set = [["docs/old.md", None]]
        mock_connection.query = AsyncMock(return_value=result_mock)

        record = await graph_store.get_document_record("docs/old.md")
        assert isinstance(record, DocumentRecord)
        assert record.path == "docs/old.md"
        assert record.content_hash is None

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

    async def test_cleanup_pending_documents_skips_committed(self, graph_store, mock_connection):
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

    async def test_find_pending_returns_committed_state(self, graph_store, mock_connection):
        """find_pending checks for COMMITTED first; a hit returns immediately
        and the WRITTEN fallback query is never issued."""
        results = [
            # Query 1: WHERE p.ready_to_commit = true → hit
            MagicMock(result_set=[["docs/a.md__pending__abc12345", "newhash"]]),
        ]
        mock_connection.query = AsyncMock(side_effect=results)

        out = await graph_store.find_pending("docs/a.md")
        assert out is not None
        state, pid, hash_ = out
        assert state == "COMMITTED"
        assert pid == "docs/a.md__pending__abc12345"
        assert hash_ == "newhash"
        # Only the COMMITTED-first query ran — no fallback.
        assert mock_connection.query.await_count == 1
        cypher = mock_connection.query.await_args_list[0][0][0]
        assert "p.ready_to_commit = true" in cypher

    async def test_find_pending_returns_written_state(self, graph_store, mock_connection):
        """When no COMMITTED pending exists, the second query falls back to
        any non-committed pending and labels it WRITTEN."""
        results = [
            MagicMock(result_set=[]),  # 1: no COMMITTED
            MagicMock(result_set=[["docs/a.md__pending__deadbeef", None]]),  # 2: WRITTEN
        ]
        mock_connection.query = AsyncMock(side_effect=results)

        out = await graph_store.find_pending("docs/a.md")
        assert out is not None
        state, pid, _ = out
        assert state == "WRITTEN"
        assert pid == "docs/a.md__pending__deadbeef"
        assert mock_connection.query.await_count == 2
        # Second query explicitly excludes COMMITTED so it can never
        # accidentally promote a committed pending into a "discard" path.
        fallback_cypher = mock_connection.query.await_args_list[1][0][0]
        assert "ready_to_commit IS NULL OR p.ready_to_commit = false" in fallback_cypher

    async def test_find_pending_prefers_committed_over_lexicographically_first(
        self, graph_store, mock_connection
    ):
        """Regression guard: under compounded crashes, the graph can hold both
        a WRITTEN and a COMMITTED pending for the same id. The earlier
        ``ORDER BY p.id LIMIT 1`` could return the WRITTEN one and silently
        let the caller take the rollback path while the COMMITTED data
        sat unattended. We now query for COMMITTED specifically first."""
        results = [
            # COMMITTED query returns the committed one (regardless of id sort).
            MagicMock(result_set=[["docs/a.md__pending__zzzzzzzz", "h"]]),
        ]
        mock_connection.query = AsyncMock(side_effect=results)

        out = await graph_store.find_pending("docs/a.md")
        assert out is not None
        assert out[0] == "COMMITTED"
        assert out[1] == "docs/a.md__pending__zzzzzzzz"

    async def test_find_pending_returns_none_when_no_pending(self, graph_store, mock_connection):
        results = [
            MagicMock(result_set=[]),  # 1: no COMMITTED
            MagicMock(result_set=[]),  # 2: no WRITTEN either
        ]
        mock_connection.query = AsyncMock(side_effect=results)
        assert await graph_store.find_pending("docs/a.md") is None
        assert mock_connection.query.await_count == 2

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

    async def test_rollforward_cutover_runs_precondition_then_three_idempotent_ops(
        self, graph_store, mock_connection
    ):
        """Rollforward is precondition-check → delete-chunks → delete-doc →
        rename-pending. Steps 1-3 are idempotent on replay; the precondition
        guards against deleting the live doc when the pending is missing."""
        results = [
            MagicMock(result_set=[[1]]),  # 0. precondition: pending exists
            MagicMock(result_set=[[5]]),  # 1. delete chunks
            MagicMock(result_set=[]),  # 2. delete old doc
            MagicMock(result_set=[]),  # 3. rename pending + remove marker
        ]
        mock_connection.query = AsyncMock(side_effect=results)

        chunks_removed = await graph_store.rollforward_cutover(
            pending_id="docs/a.md__pending__abc12345",
            real_id="docs/a.md",
            path="docs/a.md",
            content_hash="newhash",
        )
        assert chunks_removed == 5
        assert mock_connection.query.await_count == 4

        precheck_cypher = mock_connection.query.await_args_list[0][0][0]
        assert "RETURN count(p)" in precheck_cypher

        rename_cypher = mock_connection.query.await_args_list[3][0][0]
        assert "SET p.id = $real_id" in rename_cypher
        assert "p.path = $path" in rename_cypher
        assert "p.content_hash = $hash" in rename_cypher
        # Rename also drops the commit marker via REMOVE — otherwise the
        # FINAL state would still report as a pending Document.
        assert "REMOVE p.ready_to_commit" in rename_cypher

    async def test_rollforward_without_hash_removes_content_hash(
        self, graph_store, mock_connection
    ):
        """``content_hash=None`` promotes the pending uncertified: the hash
        is REMOVEd (never set to ``""``/null-ish) so the canonical Document
        stays eligible for repair on the next ingest/update."""
        results = [
            MagicMock(result_set=[[1]]),
            MagicMock(result_set=[[2]]),
            MagicMock(result_set=[]),
            MagicMock(result_set=[]),
        ]
        mock_connection.query = AsyncMock(side_effect=results)

        chunks_removed = await graph_store.rollforward_cutover(
            pending_id="docs/a.md__pending__abc12345",
            real_id="docs/a.md",
            path="docs/a.md",
            content_hash=None,
        )
        assert chunks_removed == 2

        rename_cypher, rename_params = mock_connection.query.await_args_list[3][0]
        assert "SET p.id = $real_id" in rename_cypher
        assert "p.path = $path" in rename_cypher
        assert "p.content_hash = $hash" not in rename_cypher
        assert "REMOVE p.ready_to_commit, p.content_hash" in rename_cypher
        assert "hash" not in rename_params

    async def test_rollforward_aborts_if_pending_missing(self, graph_store, mock_connection):
        """Precondition: if the pending Document is missing, refuse to
        proceed. Without this guard the live document would be deleted
        and the rename would silently no-op, losing the data."""
        from graphrag_sdk.core.exceptions import DatabaseError

        results = [
            MagicMock(result_set=[[0]]),  # 0. precondition: 0 nodes match → missing
        ]
        mock_connection.query = AsyncMock(side_effect=results)

        with pytest.raises(DatabaseError, match="not found"):
            await graph_store.rollforward_cutover(
                pending_id="docs/a.md__pending__missing",
                real_id="docs/a.md",
                path="docs/a.md",
                content_hash="h",
            )
        # Critical: only the precheck query ran; no destructive ops.
        assert mock_connection.query.await_count == 1

    async def test_rollforward_idempotent_when_chunks_already_gone(
        self, graph_store, mock_connection
    ):
        """If a prior rollforward attempt completed step 1 (delete chunks)
        and crashed, the replay's delete-chunks must be a no-op (count 0),
        not an error. Precondition still passes because step 3 (rename)
        hasn't run yet."""
        results = [
            MagicMock(result_set=[[1]]),  # 0. precondition: pending exists
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
        assert mock_connection.query.await_count == 4

    async def test_delete_document_chunks_and_node(self, graph_store, mock_connection):
        results = [
            MagicMock(result_set=[[3]]),  # 1. delete chunks
            MagicMock(result_set=[]),  # 2. delete document node
        ]
        mock_connection.query = AsyncMock(side_effect=results)

        chunks_removed = await graph_store.delete_document_chunks_and_node("docs/a.md")
        assert chunks_removed == 3
        assert mock_connection.query.await_count == 2

    async def test_delete_orphan_entities_skips_when_empty(self, graph_store, mock_connection):
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

    async def test_delete_orphan_entities_batches_large_lists(self, graph_store, mock_connection):
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

    async def test_delete_stale_relationships_keeps_table_backed_edges(
        self, graph_store, mock_connection
    ):
        """An edge a relationship table lists is not the document's to delete.

        A table-only edge has ``source_chunk_ids = []``, so without the
        ``structured_sources`` check any document cleanup that touches one of
        its endpoints deleted it.
        """
        mock_connection.query = AsyncMock(return_value=MagicMock(result_set=[[0]]))
        await graph_store.delete_stale_relationships(["unit-1"], ["chunk-1"])
        strip, delete = (call[0][0] for call in mock_connection.query.call_args_list)
        assert "SET r.source_chunk_ids = remaining" in strip
        assert "DELETE r" in delete
        assert "AND size(r.source_chunk_ids) = 0 " in delete
        assert "AND size(coalesce(r.structured_sources, [])) = 0 " in delete


class TestMergedLabelsBecomeRealLabels:
    """Labels absorbed during resolution must be queryable as Cypher labels.

    Resolution collapses duplicates into one node and records the losers'
    labels in a ``merged_labels`` property, because ``GraphNode.label`` holds a
    single string. Measured against a real FalkorDB, a property is not a type:
    a Person node that absorbed ``Engineer`` was invisible to
    ``MATCH (n:Engineer)`` (0 rows) while the property lookup found it. These
    tests pin the promotion of that property to real labels.
    """

    @staticmethod
    def _label_queries(mock_connection) -> list[str]:
        # The upsert query also contains ``SET n:__Entity__``, so the promotion
        # query is identified by being a MATCH rather than a MERGE.
        return [
            c[0][0]
            for c in mock_connection.query.call_args_list
            if " SET n:" in c[0][0] and "MERGE" not in c[0][0]
        ]

    async def test_absorbed_label_is_added_as_a_label(self, graph_store, mock_connection):
        await graph_store.upsert_nodes(
            [
                GraphNode(
                    id="n1",
                    label="Person",
                    properties={"name": "Ada", "merged_labels": "Engineer"},
                )
            ]
        )
        queries = self._label_queries(mock_connection)
        assert len(queries) == 1, "expected one label-promotion query"
        assert "`Engineer`" in queries[0]

    async def test_node_without_merged_labels_issues_no_extra_query(
        self, graph_store, mock_connection
    ):
        """Control: the promotion must not fire for ordinary nodes."""
        await graph_store.upsert_nodes(
            [GraphNode(id="n1", label="Person", properties={"name": "Ada"})]
        )
        assert self._label_queries(mock_connection) == []

    async def test_several_absorbed_labels_are_all_added(self, graph_store, mock_connection):
        await graph_store.upsert_nodes(
            [
                GraphNode(
                    id="n1",
                    label="Person",
                    properties={"merged_labels": "Engineer | Author"},
                )
            ]
        )
        queries = self._label_queries(mock_connection)
        assert len(queries) == 1
        assert "`Author`" in queries[0] and "`Engineer`" in queries[0]

    async def test_the_survivors_own_label_is_not_re_added(self, graph_store, mock_connection):
        """``Person`` is already on the node; re-setting it would be noise."""
        await graph_store.upsert_nodes(
            [
                GraphNode(
                    id="n1",
                    label="Person",
                    properties={"merged_labels": "Person | Engineer"},
                )
            ]
        )
        queries = self._label_queries(mock_connection)
        assert len(queries) == 1
        assert queries[0].count(":`") == 1, "Person should not be set twice"
        assert "`Engineer`" in queries[0]

    async def test_backticks_in_a_label_cannot_inject_cypher(self, graph_store, mock_connection):
        """The label is interpolated, so it must be sanitized like every other.

        Safety here means the payload cannot escape its backtick-quoted
        identifier, so the check strips every quoted span and asserts nothing
        executable survives outside them.
        """
        await graph_store.upsert_nodes(
            [
                GraphNode(
                    id="n1",
                    label="Person",
                    properties={"merged_labels": "Eng` DETACH DELETE n //"},
                )
            ]
        )
        queries = self._label_queries(mock_connection)
        assert len(queries) == 1
        outside_identifiers = re.sub(r"`[^`]*`", "", queries[0])
        assert "DETACH" not in outside_identifiers
        assert queries[0].count("`") % 2 == 0, "unbalanced backticks means an escape"

    async def test_structural_labels_are_never_promoted(self, graph_store, mock_connection):
        """An entity must not become a ``Document`` or ``Chunk``.

        Those labels select structural nodes in listing and cleanup queries;
        stamping one onto an entity would make it show up there.
        """
        await graph_store.upsert_nodes(
            [
                GraphNode(
                    id="n1",
                    label="Person",
                    properties={"merged_labels": "Document | Chunk | __Entity__ | Engineer"},
                )
            ]
        )
        queries = self._label_queries(mock_connection)
        assert len(queries) == 1
        assert "`Engineer`" in queries[0]
        for bad in ("`Document`", "`Chunk`", "`__Entity__`"):
            assert bad not in queries[0]

    async def test_backticked_structural_labels_are_also_rejected(
        self, graph_store, mock_connection
    ):
        """The guard must run on the sanitized label, not the raw one.

        ``sanitize_cypher_label`` strips backticks, so checking membership
        first lets ```Document``` through and then turns it into
        ``Document`` — landing a structural label on an entity by the very
        path the guard exists to close.
        """
        await graph_store.upsert_nodes(
            [
                GraphNode(
                    id="n1",
                    label="Person",
                    properties={"merged_labels": "`Document` | `Chunk` | `__Entity__` | Engineer"},
                )
            ]
        )
        queries = self._label_queries(mock_connection)
        assert len(queries) == 1
        assert "`Engineer`" in queries[0]
        for bad in ("`Document`", "`Chunk`", "`__Entity__`"):
            assert bad not in queries[0]

    async def test_label_promotion_is_batched(self, graph_store, mock_connection):
        """Every other write in this class chunks at ``_BATCH_SIZE``; so must this.

        Unbatched, one failure drops the promoted labels for every node that
        shares the label set at once.
        """
        count = graph_store._BATCH_SIZE * 2 + 1
        await graph_store.upsert_nodes(
            [
                GraphNode(
                    id=f"n{i}",
                    label="Person",
                    properties={"merged_labels": "Engineer"},
                )
                for i in range(count)
            ]
        )
        queries = [
            c
            for c in mock_connection.query.call_args_list
            if " SET n:" in c[0][0] and "MERGE" not in c[0][0]
        ]
        assert len(queries) == 3
        sizes = [len(c[0][1]["ids"]) for c in queries]
        assert sizes == [graph_store._BATCH_SIZE, graph_store._BATCH_SIZE, 1]
        assert sum(sizes) == count

    async def test_promotion_failure_does_not_fail_the_upsert(self, graph_store, mock_connection):
        """A label is an enrichment; losing it must not lose the node."""

        async def flaky(cypher, params=None):
            if "UNWIND $ids" in cypher:
                raise Exception("label add failed")
            return MagicMock(result_set=[])

        mock_connection.query = AsyncMock(side_effect=flaky)
        result = await graph_store.upsert_nodes(
            [GraphNode(id="n1", label="Person", properties={"merged_labels": "Engineer"})]
        )
        assert result == 1


class TestGraphStoreIdIndex:
    """Bug #9 — range index on ``id`` for every label we MERGE/MATCH by id.

    Measured against FalkorDB v4.18.0 writing 50K nodes with this class's own
    MERGE: unindexed the last batch was 477x slower than the first (4.6 ms ->
    2194.8 ms, 111 s total); indexed it stayed flat (8.8 -> 8.5 ms, 0.69 s).
    Driving the real ``GraphStore`` rather than raw Cypher showed 8.1x less
    total time at 20K nodes. This is the "gets slower as the graph grows"
    complaint.
    """

    @staticmethod
    def index_calls(mock_connection):
        return [c[0][0] for c in mock_connection.query.call_args_list if "CREATE INDEX" in c[0][0]]

    async def test_creates_index_for_node_label_and_entity(self, graph_store, mock_connection):
        await graph_store.upsert_nodes([GraphNode(id="n1", label="Person", properties={})])
        idx = self.index_calls(mock_connection)
        assert "CREATE INDEX FOR (n:`Person`) ON (n.id, n.entity_key)" in idx
        # MERGE targets :Person but relationship MATCHes target :__Entity__.
        assert "CREATE INDEX FOR (n:`__Entity__`) ON (n.id, n.entity_key)" in idx

    async def test_structural_labels_get_index_but_not_entity(self, graph_store, mock_connection):
        await graph_store.upsert_nodes([GraphNode(id="c1", label="Chunk", properties={})])
        idx = self.index_calls(mock_connection)
        assert "CREATE INDEX FOR (n:`Chunk`) ON (n.id, n.entity_key)" in idx
        assert not any("`__Entity__`" in q for q in idx)

    async def test_index_created_once_per_label(self, graph_store, mock_connection):
        for i in range(5):
            await graph_store.upsert_nodes([GraphNode(id=f"n{i}", label="Person", properties={})])
        idx = self.index_calls(mock_connection)
        assert idx.count("CREATE INDEX FOR (n:`Person`) ON (n.id, n.entity_key)") == 1

    async def test_relationship_upsert_indexes_both_endpoints(self, graph_store, mock_connection):
        await graph_store.upsert_relationships(
            [
                GraphRelationship(
                    start_node_id="e1",
                    end_node_id="c1",
                    type="MENTIONED_IN",
                    properties={},
                )
            ]
        )
        idx = self.index_calls(mock_connection)
        assert "CREATE INDEX FOR (n:`__Entity__`) ON (n.id, n.entity_key)" in idx
        assert "CREATE INDEX FOR (n:`Chunk`) ON (n.id, n.entity_key)" in idx

    async def test_index_failure_does_not_break_the_write(self, graph_store, mock_connection):
        """A missing index is slow; a raised exception would be data loss."""

        def side_effect(cypher, params=None, **kwargs):
            if "CREATE INDEX" in cypher:
                raise Exception("index unsupported")
            return MagicMock()

        mock_connection.query = AsyncMock(side_effect=side_effect)
        result = await graph_store.upsert_nodes([GraphNode(id="n1", label="Person", properties={})])
        assert result == 1

    async def test_index_failure_is_retried_on_next_write(self, graph_store, mock_connection):
        """A transient CREATE INDEX failure must not disable indexing for the
        life of the store — the label is memoised only once the query succeeds."""
        attempts = {"n": 0}

        def side_effect(cypher, params=None, **kwargs):
            if "CREATE INDEX" in cypher:
                attempts["n"] += 1
                if attempts["n"] == 1:
                    raise Exception("connection blip")
            return MagicMock()

        mock_connection.query = AsyncMock(side_effect=side_effect)
        await graph_store.upsert_nodes([GraphNode(id="c1", label="Chunk", properties={})])
        await graph_store.upsert_nodes([GraphNode(id="c2", label="Chunk", properties={})])
        await graph_store.upsert_nodes([GraphNode(id="c3", label="Chunk", properties={})])
        idx = self.index_calls(mock_connection)
        # First attempt failed, second succeeded, third was served from the memo.
        assert idx.count("CREATE INDEX FOR (n:`Chunk`) ON (n.id, n.entity_key)") == 2

    async def test_delete_all_resets_index_memo(self, graph_store, mock_connection):
        """GRAPH.DELETE drops the indexes, so a re-ingest on the same store
        must recreate them."""
        mock_connection.delete_graph = AsyncMock()
        await graph_store.upsert_nodes([GraphNode(id="n1", label="Person", properties={})])
        await graph_store.delete_all()
        await graph_store.upsert_nodes([GraphNode(id="n2", label="Person", properties={})])
        idx = self.index_calls(mock_connection)
        assert idx.count("CREATE INDEX FOR (n:`Person`) ON (n.id, n.entity_key)") == 2
        assert idx.count("CREATE INDEX FOR (n:`__Entity__`) ON (n.id, n.entity_key)") == 2

    async def test_delete_all_fallback_also_resets_index_memo(self, graph_store, mock_connection):
        """Clearing on the DETACH DELETE fallback is harmless (one idempotent
        round trip per label) and keeps the two paths behaving the same."""
        mock_connection.delete_graph = AsyncMock(side_effect=Exception("no GRAPH.DELETE"))
        await graph_store.upsert_nodes([GraphNode(id="n1", label="Person", properties={})])
        await graph_store.delete_all()
        await graph_store.upsert_nodes([GraphNode(id="n2", label="Person", properties={})])
        idx = self.index_calls(mock_connection)
        assert idx.count("CREATE INDEX FOR (n:`Person`) ON (n.id, n.entity_key)") == 2
        assert any("DETACH DELETE" in c[0][0] for c in mock_connection.query.call_args_list)


class TestNewIndexIsReadyBeforeItsFirstQuery:
    """``CREATE INDEX`` returns at once; FalkorDB builds the index afterwards.

    A lookup sent straight after it was planned without the index: on a 1M-node
    label the first 500-key lookup scanned the label once per key and timed out.
    And on FalkorDB 6.0.2, several CREATE INDEX statements sent back to back for
    one label sometimes left the last field's index empty while it reported
    OPERATIONAL, so every key looked missing.
    """

    @staticmethod
    def _graph(mock_connection, *, listings=(), statuses=(), refuse=0):
        """``listings``/``statuses`` answer successive index reads; the first
        ``refuse`` CREATE statements fail as FalkorDB fails one naming a field
        that is already indexed."""
        listings, statuses = iter(listings), iter(statuses)
        refused = {"n": 0}

        async def answer(cypher, params=None, **kwargs):
            if "YIELD label, properties" in cypher:
                return MagicMock(result_set=next(listings, []))
            if "YIELD label, status" in cypher:
                return MagicMock(result_set=[[next(statuses, "OPERATIONAL")]])
            if "CREATE INDEX" in cypher and refused["n"] < refuse:
                refused["n"] += 1
                raise DatabaseError("Attribute 'id' is already indexed")
            return MagicMock(result_set=[])

        mock_connection.query = AsyncMock(side_effect=answer)

    @staticmethod
    def _steps(mock_connection):
        steps = []
        for call in mock_connection.query.call_args_list:
            cypher = call[0][0]
            if "CREATE INDEX" in cypher:
                steps.append(cypher)
            elif "YIELD label, status" in cypher:
                steps.append("status")
            elif "UNWIND $keys AS k" in cypher:
                steps.append("lookup")
        return steps

    @pytest.fixture(autouse=True)
    def _no_sleeping(self, monkeypatch):
        monkeypatch.setattr("graphrag_sdk.storage.graph_store.asyncio.sleep", AsyncMock())

    async def test_the_lookup_waits_until_the_new_index_is_built(
        self, graph_store, mock_connection
    ):
        building = "[Indexing] 599/1000: UNDER CONSTRUCTION"
        self._graph(mock_connection, statuses=[building, building, "OPERATIONAL"])

        await graph_store.resolve_by_property("Unit", "code", ["U-1"])

        assert self._steps(mock_connection) == [
            "CREATE INDEX FOR (n:`Unit`) ON (n.id, n.entity_key, n.`code`)",
            "status",
            "status",
            "status",
            "lookup",
        ]
        status = next(
            call for call in mock_connection.query.call_args_list if "status" in call[0][0]
        )
        assert status[0][1] == {"label": "Unit", "entity_type": "NODE"}

    async def test_an_index_already_in_place_is_not_waited_for(self, graph_store, mock_connection):
        self._graph(mock_connection, listings=[[["Unit", ["id", "entity_key", "code"]]]])

        await graph_store.resolve_by_property("Unit", "code", ["U-1"])
        await graph_store.resolve_by_property("Unit", "code", ["U-2"])

        assert self._steps(mock_connection) == ["lookup", "lookup"]

    async def test_the_wait_is_bounded(self, graph_store, mock_connection, monkeypatch, caplog):
        monkeypatch.setattr("graphrag_sdk.storage.graph_store._INDEX_READY_WAIT_S", 0.0)
        self._graph(mock_connection, statuses=["[Indexing] 1/9: UNDER CONSTRUCTION"] * 99)

        with caplog.at_level("WARNING", logger="graphrag_sdk.storage.graph_store"):
            found = await graph_store.resolve_by_property("Unit", "code", ["U-1"])

        assert found == {}
        assert self._steps(mock_connection)[-1] == "lookup", "a slow build only costs speed"
        assert "still building" in caplog.text

    async def test_an_already_indexed_field_is_read_back_and_the_rest_created(
        self, graph_store, mock_connection
    ):
        """FalkorDB refuses the whole statement when one field is indexed, and
        creates none of them; another instance indexed ``id`` in between."""
        self._graph(mock_connection, listings=[[], [["Unit", ["id"]]]], refuse=1)

        await graph_store.resolve_by_property("Unit", "code", ["U-1"])

        assert self._steps(mock_connection) == [
            "CREATE INDEX FOR (n:`Unit`) ON (n.id, n.entity_key, n.`code`)",
            "CREATE INDEX FOR (n:`Unit`) ON (n.entity_key, n.`code`)",
            "status",
            "lookup",
        ]
        assert {("Unit", "id"), ("Unit", "entity_key"), ("Unit", "code")} <= graph_store._indexed

    async def test_already_indexed_does_not_mark_fields_it_did_not_create(
        self, graph_store, mock_connection
    ):
        """The refusal names one field; the others were not created either."""
        self._graph(mock_connection, refuse=99)

        await graph_store.resolve_by_property("Unit", "code", ["U-1"])

        assert not {("Unit", "entity_key"), ("Unit", "code")} & graph_store._indexed, (
            "not remembered, so the next write asks again"
        )
        creates = [step for step in self._steps(mock_connection) if "CREATE" in step]
        assert len(creates) == 3, "bounded retries"
