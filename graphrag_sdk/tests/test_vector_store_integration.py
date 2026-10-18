"""Real-FalkorDB tests for storage/vector_store.py.

Index DDL, procedure availability and error replies differ between the C
engine (4.x) and the Rust engine (6.0.x); these tests pin the behaviour the
SDK relies on against whichever server ``$FALKOR_HOST:$FALKOR_PORT`` runs.
Skipped unless ``RUN_INTEGRATION=1``.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.storage.vector_store import VectorStore


@pytest.fixture
async def conn():
    if not os.getenv("RUN_INTEGRATION"):
        pytest.skip("Set RUN_INTEGRATION=1 to run real-FalkorDB integration tests")
    connection = FalkorDBConnection(
        ConnectionConfig(
            host=os.getenv("FALKOR_HOST", "localhost"),
            port=int(os.getenv("FALKOR_PORT", "6379")),
            username=os.getenv("FALKOR_USERNAME") or None,
            password=os.getenv("FALKOR_PASSWORD") or None,
            graph_name=f"test_vs_{uuid4().hex[:8]}",
            retry_delay=0.01,
        )
    )
    yield connection
    try:
        await connection.delete_graph()
    finally:
        await connection.close()


async def _index_types(conn: FalkorDBConnection, label: str) -> dict[str, list[str]]:
    result = await conn.query(
        "CALL db.indexes() YIELD label, types WHERE label = $label RETURN types",
        {"label": label},
    )
    return dict(result.result_set[0][0]) if result.result_set else {}


class TestFulltextIndexes:
    async def test_ensure_indices_creates_both_fulltext_indexes(self, conn):
        await conn.query(
            "CREATE (:Chunk {id: 'c1', text: 'the quick brown fox'}), "
            "(:__Entity__ {id: 'e1', name: 'Fox', description: 'a quick animal'})"
        )
        store = VectorStore(conn, embedding_dimension=8)

        result = await store.ensure_indices()

        assert result["fulltext_Chunk"] is True
        assert result["fulltext_Entity"] is True
        assert "FULLTEXT" in (await _index_types(conn, "Chunk")).get("text", [])
        entity = await _index_types(conn, "__Entity__")
        assert "FULLTEXT" in entity.get("name", [])
        assert "FULLTEXT" in entity.get("description", [])
        assert [r["id"] for r in await store.fulltext_search_chunks("fox")] == ["c1"]
        assert [r["id"] for r in await store.fulltext_search_entities("Fox")] == ["e1"]
