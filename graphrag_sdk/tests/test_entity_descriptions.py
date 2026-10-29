"""An entity's description: joined across mentions, bounded by finalize(),
rebuilt from its mention edges after a delete. Entities keep no chunk lists;
their MENTIONED_IN edges say which chunks mention them, where, and what each
says the entity is.

The real-FalkorDB tests are skipped unless ``RUN_INTEGRATION=1``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.models import EntityMention, GraphData, GraphNode, LLMResponse
from graphrag_sdk.core.providers.base import LLMBatchItem
from graphrag_sdk.ingestion.lexical_graph import LexicalGraphWriter
from graphrag_sdk.storage.graph_store import (
    MAX_DESCRIPTION_MEMBERS,
    SUMMARISE_OVER_MEMBERS,
    GraphStore,
)

from .conftest import falkordb_test_server


class SummaryLLM:
    """Answers every summary prompt with one fixed text, recording the prompts."""

    def __init__(self, text: str = "One summary.") -> None:
        self.text = text
        self.prompts: list[str] = []

    async def abatch_invoke(self, prompts, **kw):
        self.prompts.extend(prompts)
        return [
            LLMBatchItem(index=i, response=LLMResponse(content=self.text))
            for i in range(len(prompts))
        ]


def _result(rows):
    return SimpleNamespace(result_set=rows)


def _writes(conn):
    return [c for c in conn.query.call_args_list if "SET e.description = $desc" in c[0][0]]


# ── unit ────────────────────────────────────────────────────────────────────


class TestEntityWrite:
    async def test_a_write_joins_descriptions_up_to_the_cap(self, mock_connection):
        store = GraphStore(mock_connection)
        await store.upsert_nodes(
            [GraphNode(id="acme", label="Organization", properties={"description": "A maker."})]
        )
        cypher, params = next(
            c[0] for c in mock_connection.query.call_args_list if "UNWIND $batch" in c[0][0]
        )
        assert "d_old + [d IN d_new WHERE NOT d IN d_old])[..$cap]" in cypher
        assert "n.descriptions = CASE WHEN size(d_all) > 1" in cypher
        assert params["cap"] == MAX_DESCRIPTION_MEMBERS

    def test_two_writes_of_one_entity_in_a_call_keep_both_descriptions(self):
        merged, _ = GraphStore._fold_properties(
            {"description": "CEO of Acme."},
            {"description": "Founded Acme in 2015."},
            union_provenance=False,
            join_descriptions=True,
        )
        assert merged["descriptions"] == ["CEO of Acme.", "Founded Acme in 2015."]
        assert merged["description"] == "CEO of Acme. | Founded Acme in 2015."


class TestSummaries:
    async def test_an_entity_over_the_limit_gets_one_description(self, mock_connection):
        members = [f"fact {i}" for i in range(SUMMARISE_OVER_MEMBERS + 1)]
        current = " | ".join(members)
        mock_connection.query = AsyncMock(
            side_effect=[_result([["alice", "Alice", current, members]]), _result([[1]])]
        )
        llm = SummaryLLM("Alice leads Acme.")
        written = await GraphStore(mock_connection).summarise_long_descriptions(llm)
        assert written == 1
        assert "- fact 5" in llm.prompts[0]
        (write,) = _writes(mock_connection)
        params = write[0][1]
        # Written only if the node still holds what was summarised.
        assert params["old"] == current
        assert params["desc"] == "Alice leads Acme." and params["members"] == ["Alice leads Acme."]

    async def test_nothing_over_the_limit_makes_no_llm_call(self, mock_connection):
        llm = SummaryLLM()
        assert await GraphStore(mock_connection).summarise_long_descriptions(llm) == 0
        assert llm.prompts == []

    async def test_a_failed_summary_writes_nothing(self, mock_connection):
        mock_connection.query = AsyncMock(
            return_value=_result([["a", "A", "x | y", ["x", "y", "z", "u", "v", "w"]]])
        )
        llm = MagicMock()
        llm.abatch_invoke = AsyncMock(
            return_value=[LLMBatchItem(index=0, error=RuntimeError("timeout"))]
        )
        assert await GraphStore(mock_connection).summarise_long_descriptions(llm) == 0
        assert _writes(mock_connection) == []


class TestRebuildAfterDelete:
    async def test_the_description_becomes_what_the_remaining_edges_say(self, mock_connection):
        mock_connection.query = AsyncMock(
            side_effect=[
                _result([["alice", "Alice", "kept | deleted", ["kept", "also kept"]]]),
                _result([[1]]),
            ]
        )
        assert await GraphStore(mock_connection).rebuild_entity_descriptions(["alice"]) == 1
        (write,) = _writes(mock_connection)
        assert write[0][1]["desc"] == "kept | also kept"
        assert write[0][1]["old"] == "kept | deleted"


class TestMentionEdges:
    async def test_a_mention_edge_carries_its_spans_and_description(self):
        writer = LexicalGraphWriter()
        writer.graph_store = MagicMock()
        writer.graph_store.upsert_relationships = AsyncMock(return_value=1)
        data = GraphData(
            mentions=[
                EntityMention(
                    chunk_id="c1", entity_id="alice", spans=[3, 8], description="CEO of Acme."
                )
            ]
        )
        await writer._write_mentions(data, MagicMock())
        (rels,) = writer.graph_store.upsert_relationships.call_args[0]
        assert rels[0].type == "MENTIONED_IN"
        assert rels[0].properties == {"spans": [3, 8], "description": "CEO of Acme."}


# ── real FalkorDB ───────────────────────────────────────────────────────────


@pytest.fixture
async def store():
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("Set RUN_INTEGRATION=1 to run real-FalkorDB integration tests")
    connection = FalkorDBConnection(
        ConnectionConfig(
            **falkordb_test_server(),
            graph_name=f"test_desc_{uuid4().hex[:8]}",
            retry_delay=0.01,
        )
    )
    yield GraphStore(connection)
    try:
        await connection.delete_graph()
    finally:
        await connection.close()


async def _entity(store, eid="alice"):
    rows = (
        await store.query_raw(
            "MATCH (e:__Entity__ {id: $id}) RETURN e.description, e.descriptions, "
            "e.source_chunk_ids, e.spans",
            {"id": eid},
        )
    ).result_set
    return rows[0] if rows else None


def _alice(description: str) -> GraphNode:
    return GraphNode(
        id="alice",
        label="Person",
        properties={
            "name": "Alice",
            "description": description,
            "source_chunk_ids": ["c1"],
            "spans": {"c1": [{"start": 0, "end": 5}]},
        },
    )


class TestOnRealFalkorDB:
    async def test_a_second_document_adds_to_the_description(self, store):
        await store.upsert_nodes([_alice("CEO of Acme.")])
        await store.upsert_nodes([_alice("Founded Acme in 2015.")])
        await store.upsert_nodes([_alice("CEO of Acme.")])  # said again: not repeated
        desc, descs, chunks, spans = await _entity(store)
        assert descs == ["CEO of Acme.", "Founded Acme in 2015."]
        assert desc == "CEO of Acme. | Founded Acme in 2015."
        # Which chunks and where are on the MENTIONED_IN edges, not here.
        assert chunks is None and spans is None

    async def test_a_node_holds_at_most_the_cap(self, store):
        for i in range(MAX_DESCRIPTION_MEMBERS + 5):
            await store.upsert_nodes([_alice(f"fact {i}")])
        _, descs, _, _ = await _entity(store)
        assert len(descs) == MAX_DESCRIPTION_MEMBERS

    async def test_finalizes_summary_replaces_a_long_list(self, store):
        for i in range(SUMMARISE_OVER_MEMBERS + 1):
            await store.upsert_nodes([_alice(f"fact {i}")])
        assert await store.summarise_long_descriptions(SummaryLLM("Alice leads Acme.")) == 1
        desc, descs, _, _ = await _entity(store)
        assert desc == "Alice leads Acme." and descs is None
        # The next mention joins onto the summary.
        await store.upsert_nodes([_alice("Lives in Berlin.")])
        desc, descs, _, _ = await _entity(store)
        assert descs == ["Alice leads Acme.", "Lives in Berlin."]

    async def test_after_a_delete_nothing_the_chunk_said_stays(self, store):
        await store.upsert_nodes(
            [GraphNode(id=c, label="Chunk", properties={"text": c}) for c in ("c1", "c2")]
        )
        await store.upsert_nodes([_alice("CEO of Acme.")])
        await store.upsert_nodes([_alice("Arrested in 2026.")])
        await store.query_raw(
            "MATCH (e:__Entity__ {id: 'alice'}), (c1:Chunk {id: 'c1'}), (c2:Chunk {id: 'c2'}) "
            "CREATE (e)-[:MENTIONED_IN {description: 'CEO of Acme.', spans: [0, 5]}]->(c1), "
            "(e)-[:MENTIONED_IN {description: 'Arrested in 2026.', spans: [0, 5]}]->(c2)",
            {},
        )
        await store.query_raw("MATCH (c:Chunk {id: 'c2'}) DETACH DELETE c", {})
        assert await store.rebuild_entity_descriptions(["alice"]) == 1
        desc, descs, _, _ = await _entity(store)
        assert desc == "CEO of Acme." and descs is None
