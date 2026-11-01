"""An entity's description: joined across mentions, bounded by finalize(), and
cleared of what a deleted or updated document said. Entities keep no chunk
lists; their MENTIONED_IN edges say which chunks mention them, where, and what
each says the entity is.

The real-FalkorDB tests are skipped unless ``RUN_INTEGRATION=1``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import EntityMention, GraphData, GraphNode, LLMResponse
from graphrag_sdk.core.providers.base import LLMBatchItem
from graphrag_sdk.ingestion.lexical_graph import LexicalGraphWriter
from graphrag_sdk.ingestion.pipeline import IngestionPipeline
from graphrag_sdk.storage.graph_store import (
    MAX_DESCRIPTION_MEMBERS,
    MAX_SUMMARY_CHARS,
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
    return [c for c in conn.query.call_args_list if "SET e.description = CASE" in c[0][0]]


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
        # A node already holding the cap (a merge can leave more) is not cut.
        assert "CASE WHEN size(d_old) >= $cap THEN d_old" in cypher
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


class TestRemappedMentions:
    """Ingest-time resolution remaps mentions onto survivors; a mention's
    spans and description are the only record of what its chunk said."""

    def test_a_remap_keeps_what_every_mention_said(self):
        data = GraphData(
            mentions=[
                EntityMention(entity_id="a", chunk_id="c1", spans=[0, 5], description="CEO"),
                EntityMention(entity_id="b", chunk_id="c2", spans=[9, 12], description="Other"),
            ]
        )
        out = IngestionPipeline._remap_mentions(data, {"x": "y"})
        assert [(m.entity_id, m.spans, m.description) for m in out.mentions] == [
            ("a", [0, 5], "CEO"),
            ("b", [9, 12], "Other"),
        ]

    def test_two_mentions_of_one_chunk_become_one_with_both(self):
        data = GraphData(
            mentions=[
                EntityMention(entity_id="alice", chunk_id="c1", spans=[0, 5], description="CEO"),
                EntityMention(entity_id="dr-a", chunk_id="c1", spans=[40, 45], description="Dr."),
                EntityMention(entity_id="dr-a", chunk_id="c1", spans=[0, 5], description="CEO"),
            ]
        )
        (m,) = IngestionPipeline._remap_mentions(data, {"dr-a": "alice"}).mentions
        assert (m.entity_id, m.chunk_id) == ("alice", "c1")
        assert m.spans == [0, 5, 40, 45]
        assert m.description == "CEO | Dr."


class TestSummaries:
    async def test_an_entity_over_the_limit_gets_one_marked_description(self, mock_connection):
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
        cypher, params = write[0]
        # Written only if the node still holds what was summarised.
        assert params["old"] == current
        assert params["members"] == ["Alice leads Acme."]
        assert "description_summarised = true" in cypher

    async def test_a_summary_is_cut_to_the_limit(self, mock_connection):
        members = [f"fact {i}" for i in range(SUMMARISE_OVER_MEMBERS + 1)]
        mock_connection.query = AsyncMock(
            side_effect=[_result([["a", "A", "x", members]]), _result([[1]])]
        )
        await GraphStore(mock_connection).summarise_long_descriptions(SummaryLLM("word " * 500))
        (write,) = _writes(mock_connection)
        assert len(write[0][1]["members"][0]) <= MAX_SUMMARY_CHARS + 1

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


class TestTakingARemovedDocumentOff:
    async def test_only_what_the_removed_chunks_said_is_taken_off(self, mock_connection):
        mock_connection.query = AsyncMock(
            side_effect=[
                # alice: not summarised, no undescribed edge
                _result([["alice", "Legacy | From C", ["Legacy", "From C"], False, False]]),
                _result([]),  # no remaining edge still says "From C"
                _result([[1]]),
            ]
        )
        store = GraphStore(mock_connection)
        assert await store.rebuild_entity_descriptions(["alice"], {"alice": {"From C"}}) == 1
        (write,) = _writes(mock_connection)
        assert write[0][1]["members"] == ["Legacy"]

    async def test_an_entity_the_removed_chunks_said_nothing_about_is_not_read(
        self, mock_connection
    ):
        store = GraphStore(mock_connection)
        assert await store.rebuild_entity_descriptions(["bob"], {"alice": {"x"}}) == 0
        mock_connection.query.assert_not_called()


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
            "e.source_chunk_ids, e.spans, e.description_summarised",
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


async def _mentions(store, *edges):
    """Chunks, and Alice's MENTIONED_IN edges to them: ``(chunk, description)``,
    ``None`` for an edge written by an earlier version (no description)."""
    for chunk, description in edges:
        await store.query_raw(
            "MERGE (c:Chunk {id: $c}) WITH c MATCH (e:__Entity__ {id: 'alice'}) "
            "CREATE (e)-[m:MENTIONED_IN]->(c) SET m.description = $d, m.spans = [0, 5]",
            {"c": chunk, "d": description},
        )


async def _remove(store, doc, chunks):
    """What delete_document and update() do to one document's chunks: record
    what they said, delete them, take it off."""
    await store.query_raw("CREATE (:Document {id: $id})", {"id": doc})
    await store.capture_cleanup_said(doc, chunks)
    await store.query_raw("MATCH (c:Chunk) WHERE c.id IN $ids DETACH DELETE c", {"ids": chunks})
    return await store.rebuild_entity_descriptions(["alice"], await store.get_cleanup_said(doc))


class TestOnRealFalkorDB:
    async def test_a_second_document_adds_to_the_description(self, store):
        await store.upsert_nodes([_alice("CEO of Acme.")])
        await store.upsert_nodes([_alice("Founded Acme in 2015.")])
        await store.upsert_nodes([_alice("CEO of Acme.")])  # said again: not repeated
        desc, descs, chunks, spans, _ = await _entity(store)
        assert descs == ["CEO of Acme.", "Founded Acme in 2015."]
        assert desc == "CEO of Acme. | Founded Acme in 2015."
        # Which chunks and where are on the MENTIONED_IN edges, not here.
        assert chunks is None and spans is None

    async def test_a_node_holds_at_most_the_cap_and_is_never_cut(self, store):
        for i in range(MAX_DESCRIPTION_MEMBERS + 5):
            await store.upsert_nodes([_alice(f"fact {i}")])
        _, descs, _, _, _ = await _entity(store)
        assert len(descs) == MAX_DESCRIPTION_MEMBERS
        # A node a merge left over the cap keeps what it holds.
        over = [f"merged {i}" for i in range(MAX_DESCRIPTION_MEMBERS + 10)]
        await store.query_raw(
            "MATCH (e:__Entity__ {id: 'alice'}) SET e.descriptions = $d, e.description = 'x'",
            {"d": over},
        )
        await store.upsert_nodes([_alice("new fact")])
        _, descs, _, _, _ = await _entity(store)
        assert descs == over

    async def test_finalizes_summary_replaces_a_long_list(self, store):
        for i in range(SUMMARISE_OVER_MEMBERS + 1):
            await store.upsert_nodes([_alice(f"fact {i}")])
        assert await store.summarise_long_descriptions(SummaryLLM("Alice leads Acme.")) == 1
        desc, descs, _, _, summarised = await _entity(store)
        assert (desc, descs, summarised) == ("Alice leads Acme.", None, True)
        # The next mention joins onto the summary.
        await store.upsert_nodes([_alice("Lives in Berlin.")])
        _, descs, _, _, _ = await _entity(store)
        assert descs == ["Alice leads Acme.", "Lives in Berlin."]

    async def test_a_removed_document_is_taken_off_and_older_text_is_kept(self, store):
        """Naseem's repro: text from a document ingested by an earlier version
        (its edge records nothing) survives the delete of another document."""
        for d in ("Legacy fact", "From B", "From C"):
            await store.upsert_nodes([_alice(d)])
        await _mentions(store, ("c0", None), ("cB", "From B"), ("cC", "From C"))
        assert await _remove(store, "docC", ["cC"]) == 1
        desc, descs, _, _, _ = await _entity(store)
        assert descs == ["Legacy fact", "From B"]

    async def test_deleted_text_goes_when_only_undescribed_edges_remain(self, store):
        """A table row's or an older version's mention edges say nothing; the
        prose that is deleted must still leave the description."""
        for d in ("Legacy fact", "Prose said"):
            await store.upsert_nodes([_alice(d)])
        await _mentions(store, ("c0", None), ("c1", "Prose said"))
        await _remove(store, "doc1", ["c1"])
        desc, descs, _, _, _ = await _entity(store)
        assert (desc, descs) == ("Legacy fact", None)

    async def test_text_a_remaining_chunk_still_says_stays(self, store):
        """An update keeps an unchanged chunk under a new id: what it says is
        removed with the old chunk and said again by the new one."""
        await store.upsert_nodes([_alice("Same text.")])
        await _mentions(store, ("old", "Same text."), ("new", "Same text."))
        assert await _remove(store, "doc", ["old"]) == 0
        desc, _, _, _, _ = await _entity(store)
        assert desc == "Same text."

    async def test_a_summary_fed_by_a_removed_chunk_is_rebuilt_from_its_edges(self, store):
        await store.upsert_nodes([_alice("A said this.")])
        await _mentions(store, ("cA", "A said this."), ("cB", "B said this."))
        await store.query_raw(
            "MATCH (e:__Entity__ {id: 'alice'}) "
            "SET e.description = 'Summary of A and B.', e.description_summarised = true"
        )
        await _remove(store, "docB", ["cB"])
        desc, descs, _, _, summarised = await _entity(store)
        assert (desc, descs, summarised) == ("A said this.", None, None)

    async def test_a_summary_with_an_undescribed_edge_left_is_kept(self, store):
        await store.upsert_nodes([_alice("x")])
        await _mentions(store, ("c0", None), ("cB", "B said this."))
        await store.query_raw(
            "MATCH (e:__Entity__ {id: 'alice'}) "
            "SET e.description = 'Summary.', e.description_summarised = true"
        )
        assert await _remove(store, "docB", ["cB"]) == 0
        desc, _, _, _, summarised = await _entity(store)
        assert (desc, summarised) == ("Summary.", True)


# ── end to end: update() and a resumed delete ───────────────────────────────


def _alice_doc(rag_factory, scripted_llm, *descriptions):
    llm = scripted_llm(*[[("Alice", "Person", d)] for d in descriptions])
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution

    return rag_factory(llm=llm, resolver=ExactMatchResolution())


async def _alice_description(rag):
    rows = (
        await rag._graph_store.query_raw(
            "MATCH (e:__Entity__ {name: 'Alice'}) RETURN e.description, e.descriptions"
        )
    ).result_set
    return rows[0] if rows else None


class TestEndToEnd:
    async def test_an_update_takes_the_old_versions_text_off(
        self, real_falkordb_rag_factory, scripted_llm
    ):
        rag = _alice_doc(real_falkordb_rag_factory, scripted_llm, "CEO of Acme.", "Joined Initech.")
        await rag.ingest(text="Alice runs Acme.", document_id="bio")
        await rag.update(text="Alice now works at Initech.", document_id="bio")
        desc, descs = await _alice_description(rag)
        assert (desc, descs) == ("Joined Initech.", None)

    async def test_a_resumed_delete_still_takes_its_text_off(
        self, real_falkordb_rag_factory, scripted_llm
    ):
        rag = _alice_doc(
            real_falkordb_rag_factory, scripted_llm, "CEO of Acme.", "Lives in Berlin."
        )
        await rag.ingest(text="Alice runs Acme.", document_id="work")
        await rag.ingest(text="Alice lives in Berlin.", document_id="home")
        store = rag._graph_store
        # The delete crashes right after its commit marker and chunk delete.
        candidates = await store.get_document_entity_candidates("home")
        chunks = await store.get_document_chunk_ids("home")
        assert await store.mark_document_pending_delete("home", candidates, chunks) == 1
        await store.delete_document_chunks("home")
        await rag._resume_pending_delete("home", Context())
        desc, descs = await _alice_description(rag)
        assert (desc, descs) == ("CEO of Acme.", None)
