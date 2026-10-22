"""Real-FalkorDB tests for storage/graph_store.py.

Skipped unless ``RUN_INTEGRATION=1``. Pins behaviour that differs between the
C engine (FalkorDB 4.x) and the Rust engine (6.0.x).
"""

from __future__ import annotations

import os
import random
from uuid import uuid4

import pytest

from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.exceptions import DatabaseError
from graphrag_sdk.core.models import GraphNode, GraphRelationship
from graphrag_sdk.storage.graph_store import GraphStore

from .conftest import falkordb_test_server


@pytest.fixture
async def store():
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("Set RUN_INTEGRATION=1 to run real-FalkorDB integration tests")
    connection = FalkorDBConnection(
        ConnectionConfig(
            **falkordb_test_server(),
            graph_name=f"test_gs_{uuid4().hex[:8]}",
            retry_delay=0.01,
        )
    )
    yield GraphStore(connection)
    try:
        await connection.delete_graph()
    finally:
        await connection.close()


class TestResultsDoNotDependOnEngineRowOrder:
    """FalkorDB 4.x and 6.0.x list unordered rows differently (newest-first vs
    oldest-first), so anything cut by a LIMIT or a cap must be ordered."""

    async def test_connected_entities_beyond_the_limit_are_the_lowest_ids(self, store):
        ids = [f"m{i:02d}" for i in range(60)]
        # Neither creation order nor its reverse is id order.
        shuffled = ids[:]
        random.Random(7).shuffle(shuffled)
        await store.upsert_nodes(
            [GraphNode(id="big", label="Chunk", properties={"text": "t"})]
            + [GraphNode(id=i, label="Thing", properties={"name": i}) for i in shuffled]
        )
        await store.upsert_relationships(
            [
                GraphRelationship(start_node_id=i, end_node_id="big", type="MENTIONED_IN")
                for i in shuffled
            ]
        )

        rows = await store.get_connected_entities("big", max_hops=1)

        assert [r["id"] for r in rows] == ids[:50]

    async def test_a_capped_backfill_chunk_keeps_the_first_pairs_by_id(self, store):
        people = ["p3", "p1", "p4", "p2"]
        companies = ["cz", "ca"]
        await store.upsert_nodes(
            [GraphNode(id="c", label="Chunk", properties={"text": "t"})]
            + [GraphNode(id=p, label="Person", properties={"name": p}) for p in people]
            + [GraphNode(id=c, label="Company", properties={"name": c}) for c in companies]
        )
        await store.upsert_relationships(
            [
                GraphRelationship(start_node_id=e, end_node_id="c", type="MENTIONED_IN")
                for e in people + companies
            ]
        )

        rows = await store.list_chunks_for_relation_pattern_backfill(
            "Person", "Company", op_id="op", pairs_per_chunk=3
        )

        assert [(p["src_id"], p["tgt_id"]) for p in rows[0]["pairs"]] == [
            ("p1", "ca"),
            ("p1", "cz"),
            ("p2", "ca"),
        ]

    async def test_sample_nodes_are_the_lowest_ids(self, store):
        from graphrag_sdk.utils.graph_viz import GraphVisualizer

        # Neither creation order nor its reverse is id order.
        await store.upsert_nodes(
            [
                GraphNode(id=f"n{i}", label="Thing", properties={"name": f"n{i}"})
                for i in (3, 0, 5, 1, 4, 2)
            ]
        )
        viz = GraphVisualizer(store._conn)

        sample = await viz.sample_nodes(label="Thing", limit=3)
        unlabelled = await viz.sample_nodes(limit=3)

        assert [row["node"].properties["id"] for row in sample] == ["n0", "n1", "n2"]
        assert [row["node"].properties["id"] for row in unlabelled] == ["n0", "n1", "n2"]


class TestOneKeyTwiceInOneCall:
    async def test_an_entity_written_twice_keeps_both_chunks_and_the_last_values(self, store):
        await store.upsert_nodes(
            [
                GraphNode(
                    id="x", label="Person", properties={"name": "A", "source_chunk_ids": ["c1"]}
                ),
                GraphNode(
                    id="x", label="Person", properties={"name": "B", "source_chunk_ids": ["c2"]}
                ),
            ]
        )

        rows = (
            await store.query_raw("MATCH (n:Person {id: 'x'}) RETURN n.name, n.source_chunk_ids")
        ).result_set
        assert rows == [["B", ["c1", "c2"]]]

    async def test_a_fact_written_twice_keeps_both_chunks(self, store):
        await store.upsert_nodes(
            [GraphNode(id=i, label="Person", properties={"name": i}) for i in ("a", "b")]
        )
        await store.upsert_relationships(
            [
                GraphRelationship(
                    start_node_id="a",
                    end_node_id="b",
                    type="RELATES",
                    properties={"rel_type": "KNOWS", "source_chunk_ids": [c], "w": w},
                )
                for c, w in (("c1", 1), ("c2", 2))
            ]
        )

        rows = (
            await store.query_raw(
                "MATCH (:Person {id: 'a'})-[r:RELATES]->(:Person {id: 'b'}) "
                "RETURN r.rel_type, r.w, r.source_chunk_ids"
            )
        ).result_set
        assert rows == [["KNOWS", 2, ["c1", "c2"]]]

    async def test_numerically_equal_rel_types_are_one_fact(self, store):
        """MERGE treats rel_type 5 and 5.0 as one key; both writes' chunks stay."""
        await store.upsert_nodes(
            [GraphNode(id=i, label="Person", properties={"name": i}) for i in ("a", "b")]
        )
        await store.upsert_relationships(
            [
                GraphRelationship(
                    start_node_id="a",
                    end_node_id="b",
                    type="RELATES",
                    properties={"rel_type": rel_type, "source_chunk_ids": [chunk]},
                )
                for rel_type, chunk in ((5, "c1"), (5.0, "c2"), (True, "c3"))
            ]
        )

        rows = (
            await store.query_raw(
                "MATCH (:Person {id: 'a'})-[r:RELATES]->(:Person {id: 'b'}) "
                "RETURN r.source_chunk_ids ORDER BY size(r.source_chunk_ids) DESC"
            )
        ).result_set
        assert rows == [[["c1", "c2"]], [["c3"]]]


class TestReadOnlyQueries:
    """A query sent read-only, as text-to-Cypher sends the model's query,
    cannot change the graph: the database itself refuses it."""

    @pytest.mark.parametrize(
        "write",
        [
            "CREATE (:Intruder {id: 'x'})",
            "MATCH (n:Doc) SET n.path = 'changed'",
            "MATCH (n:Doc) DETACH DELETE n",
            "MERGE (:Intruder {id: 'y'})",
        ],
    )
    async def test_a_write_is_refused_and_the_graph_is_unchanged(self, store, write):
        await store.query_raw("CREATE (:Doc {id: 'd1', path: 'docs/a.md'})")

        with pytest.raises(DatabaseError, match="read-only"):
            await store.query_raw(write, read_only=True)

        rows = await store.query_raw("MATCH (n) RETURN labels(n), n.path")
        assert rows.result_set == [[["Doc"], "docs/a.md"]]

    async def test_a_read_with_a_named_path_runs(self, store):
        await store.query_raw(
            "CREATE (:Doc {id: 'd1', path: 'docs/a.md'})-[:LINKS]->(:Doc {id: 'd2'})"
        )
        rows = await store.query_raw(
            "MATCH path = (a:Doc)-[:LINKS]->(b:Doc) WHERE a.path = 'docs/a.md' "
            "RETURN b.id, length(path) LIMIT 5",
            read_only=True,
        )
        assert rows.result_set == [["d2", 1]]
