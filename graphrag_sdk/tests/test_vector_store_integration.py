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

    async def test_ensure_indices_is_idempotent(self, conn):
        await VectorStore(conn, embedding_dimension=8).ensure_indices()

        # A fresh store has not cached the "already ensured" flag, so it
        # re-issues every CREATE and must read "already indexed" as success.
        again = await VectorStore(conn, embedding_dimension=8).ensure_indices()

        assert all(again.values()), again
        assert await conn._breaker.allow_request()


class TestServerRejections:
    async def test_rejected_statements_fail_once_and_keep_the_breaker_closed(self, conn):
        from graphrag_sdk.core.exceptions import DatabaseError, DatabaseUnavailableError

        # Statements each engine rejects, with wording that differs between
        # the C engine (4.x) and the Rust engine (6.0.x).
        rejected = [
            "CALL db.idx.fulltext.createNodeIndex('L', 'a', 'b')",  # 6.0.x arity
            "CALL db.nope()",
            "RETURN zz",
            "RETURN 1 / 0",
            "MERGE (n:L {id: null}) RETURN n",
            "DROP INDEX FOR (n:Nope) ON (n.x)",
        ]
        for statement in rejected * 2:  # more failures than the breaker threshold
            try:
                await conn.query(statement)
            except DatabaseUnavailableError as exc:
                pytest.fail(f"{statement!r} was retried as an outage: {exc}")
            except DatabaseError:
                pass

        assert await conn._breaker.allow_request()
        assert (await conn.query("RETURN 1")).result_set == [[1]]


class TestDropVectorIndexes:
    async def test_drop_removes_each_vector_index_and_keeps_fulltext(self, conn):
        await conn.query(
            "CREATE (:Chunk {id: 'c1', text: 'the quick brown fox'})"
            "-[:RELATES]->(:__Entity__ {id: 'e1', name: 'Fox'})"
        )
        store = VectorStore(conn, embedding_dimension=8)
        await store.ensure_indices()

        await store.drop_chunk_vector_index()
        await store.drop_entity_vector_index()
        await store.drop_relates_vector_index()

        assert (await _index_types(conn, "Chunk")).get("embedding") is None
        assert (await _index_types(conn, "__Entity__")).get("embedding") is None
        assert (await _index_types(conn, "RELATES")).get("embedding") is None
        assert "FULLTEXT" in (await _index_types(conn, "Chunk")).get("text", [])

        # The same store recreates them.
        result = await store.ensure_indices()
        assert result["vector_Chunk"] and result["vector_Entity"] and result["vector_RELATES"]
        assert "VECTOR" in (await _index_types(conn, "Chunk")).get("embedding", [])


class TestRelationshipSearch:
    async def _graph_with_embedded_edges(self, conn):
        await conn.query(
            "CREATE (a:__Entity__ {id: 'a', name: 'Alice'}), "
            "(b:__Entity__ {id: 'b', name: 'Acme'}), (c:__Entity__ {id: 'c', name: 'Bob'}), "
            "(a)-[:RELATES {rel_type: 'WORKS_AT', src_name: 'Alice', tgt_name: 'Acme', "
            "fact: 'Alice works at Acme', embedding: vecf32([1.0, 0.0, 0.0])}]->(b), "
            "(c)-[:RELATES {rel_type: 'KNOWS', src_name: 'Bob', tgt_name: 'Alice', "
            "fact: 'Bob knows Alice', embedding: vecf32([0.0, 1.0, 0.0])}]->(a)"
        )

    async def test_edges_are_found_without_a_relates_index(self, conn):
        await self._graph_with_embedded_edges(conn)
        store = VectorStore(conn, embedding_dimension=3)  # no index created

        results = await store.search_relationships([1.0, 0.0, 0.0], top_k=2)

        assert [r["fact"] for r in results] == ["Alice works at Acme", "Bob knows Alice"]
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-6)

    async def test_edges_are_found_through_the_index(self, conn):
        await self._graph_with_embedded_edges(conn)
        store = VectorStore(conn, embedding_dimension=3)
        assert await store.create_relates_vector_index()

        results = await store.search_relationships([0.0, 1.0, 0.0], top_k=1)

        assert [r["fact"] for r in results] == ["Bob knows Alice"]

    async def test_an_empty_graph_returns_nothing(self, conn):
        store = VectorStore(conn, embedding_dimension=3)
        assert await store.search_relationships([1.0, 0.0, 0.0]) == []
        assert await conn._breaker.allow_request()


class TestEntityFulltextRanking:
    async def test_the_named_entity_outranks_entities_that_only_mention_it(self, conn):
        # Every description mentions the company, so the fulltext score cannot
        # tell the company from the rest: it is 0 for all of them.
        await conn.query(
            "UNWIND range(2000, 2011) AS y "
            "CREATE (:__Entity__ {id: toString(y) + '__date', name: toString(y), "
            "description: 'The year Northwind Robotics did something'})"
        )
        await conn.query(
            "CREATE (:__Entity__ {id: 'northwind_robotics__organization', "
            "name: 'Northwind Robotics', description: 'A robotics company in Rotterdam'}), "
            "(:__Entity__ {id: 'nw_lab__organization', name: 'Northwind Robotics Lab', "
            "description: 'Research arm of Northwind Robotics'})"
        )
        store = VectorStore(conn, embedding_dimension=8)
        await store.ensure_indices()

        exact = await store.fulltext_search_entities("Northwind Robotics", top_k=3)
        assert [r["id"] for r in exact][:2] == [
            "northwind_robotics__organization",
            "nw_lab__organization",
        ]
        lower = await store.fulltext_search_entities("northwind robotics", top_k=1)
        assert lower[0]["id"] == "northwind_robotics__organization"
        # A non-string name neither breaks the search nor matches.
        await conn.query(
            "CREATE (:__Entity__ {id: 'odd', name: ['Northwind Robotics'], "
            "description: 'Northwind Robotics alias list'})"
        )
        assert (await store.fulltext_search_entities("Northwind Robotics", top_k=1))[0][
            "id"
        ] == "northwind_robotics__organization"
        await conn.query("MATCH (n {id: 'odd'}) DELETE n")
        # Entities that only mention it are ordered by score, then id.
        rest = await store.fulltext_search_entities("Northwind Robotics", top_k=14)
        assert [r["id"] for r in rest][2:] == [f"{y}__date" for y in range(2000, 2012)]
