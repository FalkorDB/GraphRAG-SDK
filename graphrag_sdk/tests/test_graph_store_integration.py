"""Real-FalkorDB tests for storage/graph_store.py.

Skipped unless ``RUN_INTEGRATION=1``. Pins behaviour that differs between the
C engine (FalkorDB 4.x) and the Rust engine (6.0.x).
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.models import GraphNode, GraphRelationship
from graphrag_sdk.storage.graph_store import GraphStore


@pytest.fixture
async def store():
    if not os.getenv("RUN_INTEGRATION"):
        pytest.skip("Set RUN_INTEGRATION=1 to run real-FalkorDB integration tests")
    connection = FalkorDBConnection(
        ConnectionConfig(
            host=os.getenv("FALKOR_HOST", "localhost"),
            port=int(os.getenv("FALKOR_PORT", "6379")),
            username=os.getenv("FALKOR_USERNAME") or None,
            password=os.getenv("FALKOR_PASSWORD") or None,
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
        await store.upsert_nodes(
            [GraphNode(id="big", label="Chunk", properties={"text": "t"})]
            + [GraphNode(id=i, label="Thing", properties={"name": i}) for i in ids]
        )
        await store.upsert_relationships(
            [
                GraphRelationship(start_node_id=i, end_node_id="big", type="MENTIONED_IN")
                for i in ids
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

    async def test_sample_nodes_returns_the_first_created(self, store):
        from graphrag_sdk.utils.graph_viz import GraphVisualizer

        await store.upsert_nodes(
            [GraphNode(id=f"n{i}", label="Thing", properties={"name": f"n{i}"}) for i in range(6)]
        )
        viz = GraphVisualizer(store._conn)

        sample = await viz.sample_nodes(label="Thing", limit=3)

        assert [row["node"].properties["id"] for row in sample] == ["n0", "n1", "n2"]
