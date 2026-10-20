"""Real-FalkorDB tests for storage/vector_store.py.

Index DDL, procedure availability and error replies differ between the C
engine (4.x) and the Rust engine (6.0.x); these tests pin the behaviour the
SDK relies on against whichever server ``$FALKOR_HOST:$FALKOR_PORT`` runs.
Skipped unless ``RUN_INTEGRATION=1``.
"""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest

from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.storage.vector_store import VectorStore


@pytest.fixture
async def conn():
    if os.getenv("RUN_INTEGRATION") != "1":
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


async def _engine_major(conn: FalkorDBConnection) -> int:
    """Major version of the FalkorDB graph module: 4 is the C engine, 6 the Rust engine."""
    from redis.asyncio import Redis

    config = conn.config
    client = Redis(
        host=config.host, port=config.port, username=config.username, password=config.password
    )
    try:
        modules = await client.module_list()
    finally:
        await client.aclose()
    graph = next(m for m in modules if m[b"name"] == b"graph")
    return int(graph[b"ver"]) // 10000


async def _index_types(conn: FalkorDBConnection, label: str) -> dict[str, list[str]]:
    result = await conn.query(
        "CALL db.indexes() YIELD label, types WHERE label = $label RETURN types",
        {"label": label},
    )
    return dict(result.result_set[0][0]) if result.result_set else {}


async def _wait_until_operational(conn: FalkorDBConnection, label: str) -> None:
    """Wait for the indexes on ``label`` to finish building. 4.x answers a
    vector query sent while its index is under construction with an error,
    which sends the search down the scan instead."""
    for _ in range(200):
        result = await conn.query(
            "CALL db.indexes() YIELD label, status WHERE label = $label RETURN status",
            {"label": label},
        )
        if result.result_set and all(row[0] == "OPERATIONAL" for row in result.result_set):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"indexes on {label} still building after 10s")


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

    async def test_an_index_on_name_alone_still_gets_description(self, conn):
        await conn.query(
            "CREATE (:__Entity__ {id: 'e1', name: 'Fox', description: 'a quick animal'})"
        )
        await conn.query("CREATE FULLTEXT INDEX FOR (n:__Entity__) ON (n.name)")
        store = VectorStore(conn, embedding_dimension=8)

        assert await store.create_entity_fulltext_index() is True

        entity = await _index_types(conn, "__Entity__")
        assert "FULLTEXT" in entity.get("description", [])
        assert [r["id"] for r in await store.fulltext_search_entities("animal")] == ["e1"]

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

        # Statements the server rejects, with wording that differs between
        # the C engine (4.x) and the Rust engine (6.0.x).
        rejected = [
            "CALL db.nope()",
            "RETURN zz",
            "RETURN 1 / 0",
            "MERGE (n:L {id: null}) RETURN n",
            "DROP INDEX FOR (n:Nope) ON (n.x)",
            "RETURN timeout",  # quotes a word a temporary reply also uses
        ]
        if await _engine_major(conn) >= 6:
            # The C engine accepts the multi-argument form; only 6.x rejects it.
            rejected.append("CALL db.idx.fulltext.createNodeIndex('L', 'a', 'b')")
        for statement in rejected * 2:  # more failures than the breaker threshold
            try:
                await conn.query(statement)
            except DatabaseUnavailableError as exc:
                pytest.fail(f"{statement!r} was retried as an outage: {exc}")
            except DatabaseError:
                # The expected outcome: rejected once, as a plain DatabaseError.
                continue
            pytest.fail(f"{statement!r} was expected to be rejected")

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

    async def test_edges_are_found_through_the_index(self, conn, monkeypatch):
        await self._graph_with_embedded_edges(conn)
        store = VectorStore(conn, embedding_dimension=3)
        assert await store.create_relates_vector_index()
        await _wait_until_operational(conn, "RELATES")
        sent: list[str] = []
        real_query = conn.query

        async def recording_query(cypher, *args, **kwargs):
            sent.append(cypher)
            return await real_query(cypher, *args, **kwargs)

        monkeypatch.setattr(conn, "query", recording_query)

        results = await store.search_relationships([0.0, 1.0, 0.0], top_k=1)

        assert [r["fact"] for r in results] == ["Bob knows Alice"]
        # The scan finds the same edge, so only this shows the index answered.
        assert not any("cosineDistance" in cypher for cypher in sent)

    async def test_an_index_another_store_dropped_does_not_hide_edges(self, conn):
        await self._graph_with_embedded_edges(conn)
        first = VectorStore(conn, embedding_dimension=3)
        await first.ensure_indices()
        await VectorStore(conn, embedding_dimension=3).drop_relates_vector_index()
        # The drop logs and swallows its errors; without this a failed drop
        # leaves the index answering, and the test never reaches the scan.
        assert (await _index_types(conn, "RELATES")).get("embedding") is None

        results = await first.search_relationships([1.0, 0.0, 0.0], top_k=1)

        assert [r["fact"] for r in results] == ["Alice works at Acme"]
        # The fallback warns "run ensure_indices()"; on this store, which ran
        # it once already, that must recreate the index, not return early.
        assert (await first.ensure_indices())["vector_RELATES"]
        assert "VECTOR" in (await _index_types(conn, "RELATES")).get("embedding", [])

    async def test_a_node_index_named_relates_does_not_hide_edges(self, conn):
        await self._graph_with_embedded_edges(conn)
        await conn.query("CREATE (:RELATES {embedding: vecf32([0.0, 0.0, 1.0])})")
        await conn.query(
            "CREATE VECTOR INDEX FOR (n:RELATES) ON (n.embedding) "
            "OPTIONS {dimension:3, similarityFunction:'cosine'}"
        )
        store = VectorStore(conn, embedding_dimension=3)  # no index on the edges

        results = await store.search_relationships([1.0, 0.0, 0.0], top_k=1)

        assert [r["fact"] for r in results] == ["Alice works at Acme"]

    async def test_an_empty_graph_returns_nothing(self, conn):
        store = VectorStore(conn, embedding_dimension=3)
        assert await store.search_relationships([1.0, 0.0, 0.0]) == []
        assert await conn._breaker.allow_request()


class TestIndexBuilds:
    async def test_a_search_after_the_build_wait_finds_the_chunks(self, conn):
        """CREATE VECTOR INDEX returns before the build over existing chunks
        ends; until then 4.x rejects a search and 6.0.x answers with no rows,
        so a search right after ensure_indices() found nothing."""
        rows = [
            {"id": f"c{i}", "v": [1.0 + (i * 7 + d) % 13 for d in range(128)]} for i in range(3000)
        ]
        for start in range(0, len(rows), 500):
            await conn.query(
                "UNWIND $rows AS r CREATE (:Chunk {id: r.id, text: r.id, embedding: vecf32(r.v)})",
                {"rows": rows[start : start + 500]},
            )
        store = VectorStore(conn, embedding_dimension=128)
        assert await store.create_chunk_vector_index()

        await store._wait_for_index_builds()

        assert len(await store.search_chunks([1.0] * 128, top_k=5)) == 5
