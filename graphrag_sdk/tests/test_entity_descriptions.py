"""An entity's description: joined across mentions, bounded by finalize(), and
cleared of what a deleted or updated document said. Entities keep no chunk
lists; their MENTIONED_IN edges say which chunks mention them, where, and what
each says the entity is.

The real-FalkorDB tests are skipped unless ``RUN_INTEGRATION=1``.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.descriptions import description_piece, join_description, split_description
from graphrag_sdk.core.models import (
    EntityMention,
    GraphData,
    GraphNode,
    GraphRelationship,
    LLMResponse,
)
from graphrag_sdk.core.providers.base import LLMBatchItem
from graphrag_sdk.ingestion.lexical_graph import LexicalGraphWriter
from graphrag_sdk.ingestion.pipeline import IngestionPipeline
from graphrag_sdk.storage.graph_store import (
    _DESCRIPTION_READS,
    MAX_DESCRIPTION_MEMBERS,
    MAX_SUMMARY_CHARS,
    SUMMARISE_OVER_MEMBERS,
    GraphStore,
)

from .conftest import MockLLM, falkordb_test_server


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
        assert merged["description"] == "CEO of Acme. | Founded Acme in 2015."
        assert "descriptions" not in merged


class TestRemappedMentions:
    """Ingest-time resolution remaps mentions onto survivors; a mention's
    spans and description are the only record of what its chunk said."""

    def test_a_remap_keeps_what_every_mention_said(self):
        data = GraphData(
            mentions=[
                EntityMention(entity_id="a", chunk_id="c1", spans=[0, 5], descriptions=["CEO"]),
                EntityMention(entity_id="b", chunk_id="c2", spans=[9, 12], descriptions=["Other"]),
            ]
        )
        out = IngestionPipeline._remap_mentions(data, {"x": "y"})
        assert [(m.entity_id, m.spans, m.descriptions) for m in out.mentions] == [
            ("a", [0, 5], ["CEO"]),
            ("b", [9, 12], ["Other"]),
        ]

    def test_two_mentions_of_one_chunk_become_one_with_both(self):
        data = GraphData(
            mentions=[
                EntityMention(entity_id="alice", chunk_id="c1", spans=[0, 5], descriptions=["CEO"]),
                EntityMention(
                    entity_id="dr-a", chunk_id="c1", spans=[40, 45], descriptions=["Dr. | MD"]
                ),
                EntityMention(entity_id="dr-a", chunk_id="c1", spans=[0, 5], descriptions=["CEO"]),
            ]
        )
        (m,) = IngestionPipeline._remap_mentions(data, {"dr-a": "alice"}).mentions
        assert (m.entity_id, m.chunk_id) == ("alice", "c1")
        assert m.spans == [0, 5, 40, 45]
        # Each kept whole: one holding " | " is not split into two.
        assert m.descriptions == ["CEO", "Dr. | MD"]


class TestSummaries:
    async def test_an_entity_over_the_limit_gets_one_marked_description(self, mock_connection):
        members = [f"fact {i}" for i in range(SUMMARISE_OVER_MEMBERS + 1)]
        current = " | ".join(members)
        # No list on the node: its pieces are its one string, split at "|".
        mock_connection.query = AsyncMock(
            side_effect=[_result([["alice", "Alice", current, None]]), _result([[1]])]
        )
        llm = SummaryLLM("Alice leads Acme.")
        written = await GraphStore(mock_connection).summarise_long_descriptions(llm)
        assert written == 1
        assert "- fact 0" in llm.prompts[0] and "- fact 5" in llm.prompts[0]
        scan = mock_connection.query.call_args_list[0][0][0]
        # Only descriptions of more than one piece are split and counted.
        assert "e.description CONTAINS '|'" in scan and "split(" in scan
        (write,) = _writes(mock_connection)
        cypher, params = write[0]
        # Written only if the node still holds what was summarised.
        assert params["old"] == current
        assert params["desc"] == "Alice leads Acme."
        assert "e.descriptions = null" in cypher
        assert "description_summarised = true" in cypher

    async def test_a_summary_holding_the_separator_stays_one_piece(self, mock_connection):
        members = [f"fact {i}" for i in range(SUMMARISE_OVER_MEMBERS + 1)]
        mock_connection.query = AsyncMock(
            side_effect=[_result([["a", "A", " | ".join(members), None]]), _result([[1]])]
        )
        await GraphStore(mock_connection).summarise_long_descriptions(SummaryLLM("CEO | founder"))
        (write,) = _writes(mock_connection)
        assert write[0][1]["desc"] == "CEO \u00a6 founder"

    async def test_a_summary_is_cut_to_the_limit(self, mock_connection):
        members = [f"fact {i}" for i in range(SUMMARISE_OVER_MEMBERS + 1)]
        mock_connection.query = AsyncMock(
            side_effect=[_result([["a", "A", "x", members]]), _result([[1]])]
        )
        await GraphStore(mock_connection).summarise_long_descriptions(SummaryLLM("word " * 500))
        (write,) = _writes(mock_connection)
        assert len(write[0][1]["desc"]) <= MAX_SUMMARY_CHARS

    async def test_a_summary_full_of_pipes_is_cut_after_they_become_stand_ins(
        self, mock_connection
    ):
        """Each "|" stored as " ¦ " is longer: measured on the raw reply, a
        999-character summary was stored at 1,499."""
        members = [f"fact {i}" for i in range(SUMMARISE_OVER_MEMBERS + 1)]
        mock_connection.query = AsyncMock(
            side_effect=[_result([["a", "A", "x", members]]), _result([[1]])]
        )
        summary = ("a|b " * 250).strip()
        assert len(summary) < MAX_SUMMARY_CHARS
        await GraphStore(mock_connection).summarise_long_descriptions(SummaryLLM(summary))
        (write,) = _writes(mock_connection)
        stored = write[0][1]["desc"]
        assert len(stored) <= MAX_SUMMARY_CHARS and "|" not in stored

    async def test_nothing_over_the_limit_makes_no_llm_call(self, mock_connection):
        llm = SummaryLLM()
        assert await GraphStore(mock_connection).summarise_long_descriptions(llm) == 0
        assert llm.prompts == []

    async def test_a_failed_summary_writes_nothing_and_is_logged(self, mock_connection, caplog):
        mock_connection.query = AsyncMock(
            return_value=_result([["a", "A", "x | y", ["x", "y", "z", "u", "v", "w"]]])
        )
        llm = MagicMock()
        llm.abatch_invoke = AsyncMock(
            return_value=[LLMBatchItem(index=0, error=RuntimeError("timeout"))]
        )
        failures: list[str] = []
        store = GraphStore(mock_connection)
        assert await store.summarise_long_descriptions(llm, failures=failures) == 0
        assert _writes(mock_connection) == []
        assert "Could not summarise 1 long entity description" in caplog.text
        # finalize() hands these on as FinalizeResult.description_summary_failures.
        assert failures == ["a (timeout)"]

    async def test_a_page_that_raises_is_listed_and_the_pass_goes_on(
        self, mock_connection, monkeypatch
    ):
        """What earlier pages wrote stays counted; the pass does not raise."""
        monkeypatch.setattr("graphrag_sdk.storage.graph_store.SUMMARY_PAGE", 1)
        members = [f"fact {i}" for i in range(SUMMARISE_OVER_MEMBERS + 1)]
        mock_connection.query = AsyncMock(
            side_effect=[
                _result([["a", "A", "x", members], ["b", "B", "y", members]]),
                _result([[1]]),
            ]
        )
        llm = SummaryLLM("A summary.")
        calls = 0
        real = llm.abatch_invoke

        async def second_page_fails(prompts, **kw):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("rate limited")
            return await real(prompts, **kw)

        llm.abatch_invoke = second_page_fails
        failures: list[str] = []
        store = GraphStore(mock_connection)
        assert await store.summarise_long_descriptions(llm, failures=failures) == 1
        assert failures == ["b (rate limited)"]


class TestTakingARemovedDocumentOff:
    async def test_only_what_the_removed_chunks_said_is_taken_off(self, mock_connection):
        mock_connection.query = AsyncMock(
            side_effect=[
                # alice: not summarised, no undescribed edge, no list
                _result([["alice", "Legacy | From C", None, False, False]]),
                _result([]),  # no remaining edge still says "From C"
                _result([[1]]),
            ]
        )
        store = GraphStore(mock_connection)
        assert await store.rebuild_entity_descriptions(["alice"], {"alice": {"From C"}}) == 1
        (write,) = _writes(mock_connection)
        assert write[0][1]["desc"] == "Legacy"

    async def test_an_entity_the_removed_chunks_said_nothing_about_is_not_read(
        self, mock_connection
    ):
        store = GraphStore(mock_connection)
        assert await store.rebuild_entity_descriptions(["bob"], {"alice": {"x"}}) == 0
        mock_connection.query.assert_not_called()

    async def test_an_entity_written_meanwhile_is_read_again(self, mock_connection):
        mock_connection.query = AsyncMock(
            side_effect=[
                _result([["alice", "Legacy | From C", ["Legacy", "From C"], False, False]]),
                _result([]),
                _result([[0]]),  # something else wrote it since the read
                _result(
                    [["alice", "Legacy | From C | New", ["Legacy", "From C", "New"], False, False]]
                ),
                _result([]),
                _result([[1]]),
            ]
        )
        store = GraphStore(mock_connection)
        assert await store.rebuild_entity_descriptions(["alice"], {"alice": {"From C"}}) == 1
        assert _writes(mock_connection)[-1][0][1]["desc"] == "Legacy | New"

    async def test_one_that_keeps_changing_is_logged_not_raised(
        self, mock_connection, monkeypatch, caplog
    ):
        """The delete or update has already taken effect: failing it would
        report a failure for a change that happened."""
        monkeypatch.setattr("graphrag_sdk.storage.graph_store._DESCRIPTION_READ_WAIT_S", 0)
        mock_connection.query = AsyncMock(
            side_effect=[
                _result([["alice", "Legacy | From C", ["Legacy", "From C"], False, False]]),
                _result([]),
                _result([[0]]),
            ]
            * _DESCRIPTION_READS
        )
        store = GraphStore(mock_connection)
        assert await store.rebuild_entity_descriptions(["alice"], {"alice": {"From C"}}) == 0
        assert "kept changing" in caplog.text and "alice" in caplog.text


class TestMentionEdges:
    async def test_a_mention_edge_carries_its_spans_and_description(self):
        writer = LexicalGraphWriter()
        writer.graph_store = MagicMock()
        writer.graph_store.upsert_relationships = AsyncMock(return_value=2)
        data = GraphData(
            mentions=[
                EntityMention(
                    chunk_id="c1", entity_id="alice", spans=[3, 8], descriptions=["CEO of Acme."]
                ),
                EntityMention(
                    chunk_id="c2", entity_id="alice", descriptions=["CEO", "Founder \u00a6 Acme"]
                ),
                # A custom extractor's text with "|" in it: read as a node's
                # description is, so the edge's pieces match the node's.
                EntityMention(chunk_id="c3", entity_id="alice", descriptions=["CEO | Acme"]),
            ]
        )
        await writer._write_mentions(data, MagicMock())
        (rels,) = writer.graph_store.upsert_relationships.call_args[0]
        assert rels[0].type == "MENTIONED_IN"
        assert rels[0].properties == {"spans": [3, 8], "description": "CEO of Acme."}
        # More than one: one string, as on a node.
        assert rels[1].properties == {"description": "CEO | Founder \u00a6 Acme"}
        assert rels[2].properties == {"description": "CEO | Acme"}

    async def test_two_names_with_one_id_in_a_chunk_make_one_edge_with_both(self):
        """``New York`` and ``New_York`` get one entity id; the second
        mention's spans and description were dropped."""
        writer = LexicalGraphWriter()
        writer.graph_store = MagicMock()
        writer.graph_store.upsert_relationships = AsyncMock(return_value=1)
        data = GraphData(
            mentions=[
                EntityMention(chunk_id="c1", entity_id="ny", spans=[0, 8], descriptions=["A city"]),
                EntityMention(
                    chunk_id="c1", entity_id="ny", spans=[20, 28], descriptions=["A state"]
                ),
            ]
        )
        await writer._write_mentions(data, MagicMock())
        (rels,) = writer.graph_store.upsert_relationships.call_args[0]
        assert len(rels) == 1
        assert rels[0].properties == {"spans": [0, 8, 20, 28], "description": "A city | A state"}


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
    """``(description, pieces, source_chunk_ids, spans, summarised)``. The
    node keeps no list (checked here); ``pieces`` is its one string split at
    "|" when it holds more than one piece, ``None`` when it holds one."""
    rows = (
        await store.query_raw(
            "MATCH (e:__Entity__ {id: $id}) RETURN e.description, e.descriptions, "
            "e.source_chunk_ids, e.spans, e.description_summarised",
            {"id": eid},
        )
    ).result_set
    if not rows:
        return None
    desc, stored_list, chunks, spans, summarised = rows[0]
    assert stored_list is None, "a write left a descriptions list on the node"
    pieces = split_description(desc)
    return desc, pieces if len(pieces) > 1 else None, chunks, spans, summarised


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
    """Chunks, and Alice's MENTIONED_IN edges to them: ``(chunk, description)``
    — a list for an edge holding more than one piece (stored as one string, as
    the SDK writes it), ``None`` for an edge written by an earlier version (no
    description)."""
    for chunk, said in edges:
        ds = said if isinstance(said, list) else [] if said is None else [said]
        await store.query_raw(
            "MERGE (c:Chunk {id: $c}) WITH c MATCH (e:__Entity__ {id: 'alice'}) "
            "CREATE (e)-[m:MENTIONED_IN]->(c) "
            "SET m.description = $d, m.spans = [0, 5], m.said_version = $v",
            {
                "c": chunk,
                "d": join_description(ds) or None,
                "v": None if said is None else 1,
            },
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
            "MATCH (e:__Entity__ {id: 'alice'}) SET e.description = $d",
            {"d": join_description(over)},
        )
        await store.upsert_nodes([_alice("new fact")])
        _, descs, _, _, _ = await _entity(store)
        assert descs == over

    async def test_an_earlier_versions_list_is_read_and_dropped(self, store):
        """A node an earlier version wrote keeps its pieces in a list beside
        the string, one of them holding " | ". The next write reads the list
        (every "|" separating pieces, as in that version's strings), adds to
        it, and leaves one string and no list."""
        await store.upsert_nodes([_alice("x")])
        await store.query_raw(
            "MATCH (e:__Entity__ {id: 'alice'}) "
            "SET e.descriptions = ['CEO', 'Founder | Acme'], e.description = 'CEO | Founder | Acme'"
        )
        await store.upsert_nodes([_alice("Board member")])
        desc, descs, _, _, _ = await _entity(store)
        assert descs == ["CEO", "Founder", "Acme", "Board member"]
        assert desc == "CEO | Founder | Acme | Board member"

    async def test_an_earlier_versions_list_without_text_does_not_hide_the_string(self, store):
        await store.upsert_nodes([_alice("x")])
        await store.query_raw(
            "MATCH (e:__Entity__ {id: 'alice'}) "
            "SET e.descriptions = [' '], e.description = 'Legacy fact'"
        )
        await store.upsert_nodes([_alice("New fact")])
        desc, _, _, _, _ = await _entity(store)
        assert desc == "Legacy fact | New fact"

    async def test_a_repeated_piece_is_kept_once_before_the_cap(self, store):
        """A description written straight to the store with one piece twenty
        times: kept once, so the cap does not cut the piece after it."""
        await store.upsert_nodes([_alice(" | ".join(["CEO"] * MAX_DESCRIPTION_MEMBERS + ["Founder"]))])
        desc, _, _, _, _ = await _entity(store)
        assert desc == "CEO | Founder"

    async def test_a_custom_extractors_text_with_a_pipe_comes_off_on_delete(self, store):
        """A custom extractor gives the node and its mention the same text,
        "CEO | Acme". The node reads it as two pieces; the mention edge is
        written the same way, so deleting the chunk takes both off."""
        await store.upsert_nodes([_alice("CEO | Acme")])
        await store.upsert_nodes([_alice("Other fact")])
        await store.query_raw("CREATE (:Chunk {id: 'c1'}), (:Chunk {id: 'c2'})")
        writer = LexicalGraphWriter()
        writer.graph_store = store
        await writer._write_mentions(
            GraphData(
                mentions=[
                    EntityMention(chunk_id="c1", entity_id="alice", descriptions=["CEO | Acme"]),
                    EntityMention(chunk_id="c2", entity_id="alice", descriptions=["Other fact"]),
                ]
            ),
            MagicMock(),
        )
        assert await _remove(store, "doc", ["c1"]) == 1
        desc, _, _, _, _ = await _entity(store)
        assert desc == "Other fact"

    async def test_an_earlier_versions_long_list_is_summarised(self, store):
        """Its string may hold no "|" (stale, or one piece); the list's six
        pieces still count."""
        await store.upsert_nodes([_alice("x")])
        await store.query_raw(
            "MATCH (e:__Entity__ {id: 'alice'}) "
            "SET e.descriptions = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5'], e.description = 'f0'"
        )
        llm = SummaryLLM("One summary.")
        assert await store.summarise_long_descriptions(llm) == 1
        assert "- f5" in llm.prompts[0]
        desc, _, _, _, summarised = await _entity(store)
        assert (desc, summarised) == ("One summary.", True)

    async def test_a_newline_beside_a_pipe_does_not_drop_what_another_chunk_says(self, store):
        """An earlier version wrote "CEO |\nFounded Acme in 1998." on the node
        and on two chunks' edges. Deleting one chunk keeps both pieces: the
        other edge still says them, and the server reads its pieces with the
        same whitespace rule as Python."""
        await store.upsert_nodes([_alice("x")])
        text = "CEO |\nFounded Acme in 1998."
        await store.query_raw(
            "MATCH (e:__Entity__ {id: 'alice'}) SET e.description = $t "
            "CREATE (e)-[:MENTIONED_IN {description: $t, said_version: 1}]->(:Chunk {id: 'c1'}), "
            "(e)-[:MENTIONED_IN {description: $t, said_version: 1}]->(:Chunk {id: 'c2'})",
            {"t": text},
        )
        assert await _remove(store, "doc", ["c1"]) == 0
        desc, _, _, _, _ = await _entity(store)
        assert split_description(desc) == ["CEO", "Founded Acme in 1998."]

    async def test_a_summary_counts_each_piece_once(self, store):
        """Repeats are not pieces: "A | B" written three times holds two."""
        await store.upsert_nodes([_alice("x")])
        await store.query_raw(
            "MATCH (e:__Entity__ {id: 'alice'}) SET e.description = 'A | B | A | B | A | B'"
        )
        llm = SummaryLLM()
        assert await store.summarise_long_descriptions(llm) == 0
        assert llm.prompts == []

    async def test_fulltext_finds_each_word_of_a_piece_that_held_pipes(self, store):
        """The index splits words at "|" and spaces, not at "¦": the stand-in
        is written with a space each side, so a compact table row's cells stay
        words of their own."""
        await store.query_raw("CREATE FULLTEXT INDEX FOR (n:__Entity__) ON (n.description)")
        await store.upsert_nodes([_alice(description_piece("|Tim Cook|CEO|Apple|"))])
        for word in ("Apple", "Tim", "CEO"):
            found = await store.query_raw(
                "CALL db.idx.fulltext.queryNodes('__Entity__', $w) YIELD node RETURN node.id",
                {"w": word},
            )
            assert found.result_set == [["alice"]], word

    async def test_a_delete_takes_off_text_an_earlier_version_kept_whole(self, store):
        """An earlier version kept "Founder | Acme" whole: in the node's list,
        and as the one description of the chunk's edge (no list). Both read as
        the pieces "Founder" and "Acme", so deleting the chunk takes them off."""
        await store.upsert_nodes([_alice("x")])
        await store.query_raw(
            "MATCH (e:__Entity__ {id: 'alice'}) "
            "SET e.descriptions = ['Legacy fact', 'Founder | Acme'], "
            "e.description = 'Legacy fact | Founder | Acme'"
        )
        await store.query_raw(
            "MATCH (e:__Entity__ {id: 'alice'}) "
            "CREATE (e)-[:MENTIONED_IN {spans: [0, 5]}]->(:Chunk {id: 'c0'}), "
            "(e)-[:MENTIONED_IN {description: 'Founder | Acme', said_version: 1}]->(:Chunk {id: 'c1'})"
        )
        assert await _remove(store, "doc", ["c1"]) == 1
        desc, _, _, _, _ = await _entity(store)
        assert desc == "Legacy fact"

    async def test_a_cleanup_an_earlier_version_recorded_still_takes_its_text_off(self, store):
        """A delete an earlier version started recorded "Founder | Acme" whole
        on the Document; resumed now, the record is read as the pieces the
        node's list is read as."""
        await store.upsert_nodes([_alice("x")])
        await store.query_raw(
            "MATCH (e:__Entity__ {id: 'alice'}) "
            "SET e.descriptions = ['Legacy fact', 'Founder | Acme'], "
            "e.description = 'Legacy fact | Founder | Acme' "
            "CREATE (e)-[:MENTIONED_IN {spans: [0, 5]}]->(:Chunk {id: 'c0'}) "
            "CREATE (:Document {id: 'doc', cleanup_said_ids: ['alice'], "
            "cleanup_said: ['Founder | Acme']})"
        )
        said = await store.get_cleanup_said("doc")
        assert said == {"alice": {"Founder", "Acme"}}
        assert await store.rebuild_entity_descriptions(["alice"], said) == 1
        desc, _, _, _, _ = await _entity(store)
        assert desc == "Legacy fact"

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

    async def test_a_description_holding_the_separator_goes_whole(self, store):
        """A table row's evidence holds " | "; the SDK writes it as one piece
        (its "|" made "¦"), and an edge holding two keeps them apart, so each
        matches the node's."""
        for d in ("Legacy fact", "| Alice | CEO |", "Founder | Acme", "Board member"):
            await store.upsert_nodes([_alice(description_piece(d))])
        await _mentions(
            store,
            ("c0", None),
            ("c1", "| Alice | CEO |"),
            ("c2", ["Founder | Acme", "Board member"]),
        )
        await _remove(store, "doc", ["c1", "c2"])
        desc, descs, _, _, _ = await _entity(store)
        assert (desc, descs) == ("Legacy fact", None)

    async def test_a_full_node_is_refilled_from_the_remaining_edges(self, store):
        """A full node keeps a later mention's description on its edge only;
        once the removed document frees places, that description takes one,
        instead of the entity being left with none."""
        said_by_a = [f"A fact {i}" for i in range(MAX_DESCRIPTION_MEMBERS)]
        for d in [*said_by_a, "From B"]:
            await store.upsert_nodes([_alice(d)])
        _, descs, _, _, _ = await _entity(store)
        assert "From B" not in descs
        await _mentions(store, *[(f"cA{i}", d) for i, d in enumerate(said_by_a)], ("cB", "From B"))
        await _remove(store, "docA", [f"cA{i}" for i in range(MAX_DESCRIPTION_MEMBERS)])
        desc, descs, _, _, _ = await _entity(store)
        assert (desc, descs) == ("From B", None)

    async def test_a_description_written_meanwhile_is_kept_and_the_text_still_goes(self, store):
        for d in ("Legacy fact", "From C"):
            await store.upsert_nodes([_alice(d)])
        await _mentions(store, ("c0", None), ("cC", "From C"))
        real, calls = store._remaining_said, []

        async def ingest_meanwhile(eid, **kw):
            calls.append(kw)
            if len(calls) == 1:
                await store.upsert_nodes([_alice("Written meanwhile")])
            return await real(eid, **kw)

        store._remaining_said = ingest_meanwhile
        assert await _remove(store, "docC", ["cC"]) == 1
        _, descs, _, _, _ = await _entity(store)
        assert descs == ["Legacy fact", "Written meanwhile"]
        assert len(calls) == 2  # read again after the first write was skipped

    async def test_a_second_write_of_a_mention_edge_adds_to_it(self, store):
        """A backfill of an entity extraction already found in a chunk writes
        that mention edge again: it adds its description and spans."""
        await store.upsert_nodes([_alice("CEO")])
        await store.query_raw("CREATE (:Chunk {id: 'c1'})")

        def edge(**props):
            return GraphRelationship(
                start_node_id="alice", end_node_id="c1", type="MENTIONED_IN", properties=props
            )

        await store.upsert_relationships(
            [edge(spans=[0, 5], description=join_description(["CEO", "Founder | Acme"]))]
        )
        await store.upsert_relationships(
            [edge(description="Backfilled"), edge(spans=[0, 5, 9, 14])]
        )
        rows = (
            await store.query_raw(
                "MATCH (:__Entity__ {id: 'alice'})-[m:MENTIONED_IN]->(:Chunk {id: 'c1'}) "
                "RETURN m.spans, m.description, m.descriptions"
            )
        ).result_set
        assert rows == [[[0, 5, 9, 14], "CEO | Founder \u00a6 Acme | Backfilled", None]]

    async def test_one_call_writing_a_mention_edge_twice_keeps_every_span_once(self, store):
        await store.upsert_nodes([_alice("CEO")])
        await store.query_raw(
            "CREATE (c:Chunk {id: 'c1'}) WITH c MATCH (e:__Entity__ {id: 'alice'}) "
            "CREATE (e)-[:MENTIONED_IN {description: ''}]->(c)"
        )

        def edge(**props):
            return GraphRelationship(
                start_node_id="alice", end_node_id="c1", type="MENTIONED_IN", properties=props
            )

        # Two writes of the edge in one call, one repeating a pair of its own.
        await store.upsert_relationships(
            [edge(spans=[0, 5], description="CEO"), edge(spans=[9, 14, 9, 14, 0, 5])]
        )
        rows = (
            await store.query_raw(
                "MATCH (:__Entity__ {id: 'alice'})-[m:MENTIONED_IN]->(:Chunk {id: 'c1'}) "
                "RETURN m.spans, m.description, m.descriptions"
            )
        ).result_set
        # The blank description the edge held is not joined in as " | CEO".
        assert rows == [[[0, 5, 9, 14], "CEO", None]]
        # One write (nothing to fold) repeating a pair: the query keeps it once.
        await store.upsert_relationships([edge(spans=[20, 25, 20, 25])])
        rows = (
            await store.query_raw(
                "MATCH (:__Entity__ {id: 'alice'})-[m:MENTIONED_IN]->(:Chunk {id: 'c1'}) "
                "RETURN m.spans"
            )
        ).result_set
        assert rows == [[[0, 5, 9, 14, 20, 25]]]

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

    async def test_the_text_beside_a_kept_summary_still_goes(self, store):
        """Naseem's repro: the resolver marks a node summarised while list
        members stay beside the summary. An undescribed edge keeps the summary,
        but the deleted chunk's own member must still go."""
        for d in ("Summary.", "A said.", "B said."):
            await store.upsert_nodes([_alice(d)])
        await store.query_raw(
            "MATCH (e:__Entity__ {id: 'alice'}) SET e.description_summarised = true"
        )
        await _mentions(store, ("c0", None), ("cA", "A said."), ("cB", "B said."))
        assert await _remove(store, "docA", ["cA"]) == 1
        _, descs, _, _, summarised = await _entity(store)
        assert (descs, summarised) == (["Summary.", "B said."], True)

    async def test_a_table_rows_edge_does_not_keep_a_summary(self, store):
        """A table row's edge never says anything and holds no text: a
        summarised entity with one is still rebuilt from its described edges."""
        await store.upsert_nodes([_alice("x")])
        await store.query_raw(
            "CREATE (c:Chunk {id: 'row1', record_key: 'E-1'}) WITH c "
            "MATCH (e:__Entity__ {id: 'alice'}) CREATE (e)-[:MENTIONED_IN]->(c)"
        )
        await _mentions(store, ("cA", "A said this."), ("cB", "B said this."))
        await store.query_raw(
            "MATCH (e:__Entity__ {id: 'alice'}) "
            "SET e.description = 'Summary of A and B.', e.description_summarised = true"
        )
        await _remove(store, "docB", ["cB"])
        desc, descs, _, _, summarised = await _entity(store)
        assert (desc, descs, summarised) == ("A said this.", None, None)

    async def test_what_two_chunks_said_alike_is_recorded_once(self, store):
        await store.upsert_nodes([_alice("CEO")])
        await _mentions(store, ("c1", "CEO"), ("c2", "CEO"))
        await store.query_raw("CREATE (:Document {id: 'doc'})")
        await store.capture_cleanup_said("doc", ["c1", "c2"])
        rows = (
            await store.query_raw("MATCH (d:Document {id: 'doc'}) RETURN d.cleanup_said")
        ).result_set
        assert rows == [[["CEO"]]]

    async def test_an_earlier_versions_edge_a_backfill_described_still_keeps_old_text(self, store):
        """Naseem's repro: a backfill adds a description to an edge an earlier
        version wrote. The text that chunk put on the node then was never
        recorded, so the node must not be rebuilt from its edges."""
        for d in ("Legacy: founded Acme in 1999.", "New doc 0.", "New doc 1."):
            await store.upsert_nodes([_alice(d)])
        await store.query_raw(
            "MATCH (e:__Entity__ {id: 'alice'}) SET e.description_summarised = true"
        )
        await _mentions(store, ("c0", None), ("c1", "New doc 0."), ("c2", "New doc 1."))
        await store.upsert_nodes([_alice("Backfill: age 52.")])
        await store.upsert_relationships(
            [
                GraphRelationship(
                    start_node_id="alice",
                    end_node_id="c0",
                    type="MENTIONED_IN",
                    properties={"description": "Backfill: age 52."},
                )
            ]
        )
        await _remove(store, "doc1", ["c2"])
        _, descs, _, _, summarised = await _entity(store)
        assert descs == ["Legacy: founded Acme in 1999.", "New doc 0.", "Backfill: age 52."]
        assert summarised is True

    async def test_a_summary_nothing_was_taken_from_stays(self, store):
        """Naseem's repro: update() removes an unchanged chunk too, whose new
        copy says the same; the summary must not be thrown away for it."""
        await store.upsert_nodes([_alice("x")])
        await _mentions(store, ("old", "Same text."), ("new", "Same text."), ("c2", "Other."))
        await store.query_raw(
            "MATCH (e:__Entity__ {id: 'alice'}) "
            "SET e.description = 'Careful summary.', e.description_summarised = true"
        )
        assert await _remove(store, "doc", ["old"]) == 0
        desc, _, _, _, summarised = await _entity(store)
        assert (desc, summarised) == ("Careful summary.", True)

    async def test_a_kept_summary_is_not_refilled_from_edges(self, store):
        said = [f"fact {i}" for i in range(MAX_DESCRIPTION_MEMBERS)]
        for d in ["Summary.", *said[:-1]]:
            await store.upsert_nodes([_alice(d)])
        await store.query_raw(
            "MATCH (e:__Entity__ {id: 'alice'}) SET e.description_summarised = true"
        )
        await _mentions(store, ("c0", None), *[(f"c{i + 1}", d) for i, d in enumerate(said)])
        await _remove(store, "doc", ["c1"])
        _, descs, _, _, _ = await _entity(store)
        # fact 0 went; the edge-only fact 19 was not added beside the summary.
        assert descs == ["Summary.", *said[1:-1]]

    async def test_an_edge_keeps_at_most_the_cap_trimmed(self, store):
        """An incomplete ingest run again rewrites one chunk's edge with new
        wordings; the edge stops taking them at the cap, as a node does."""
        await store.upsert_nodes([_alice("x")])
        await store.query_raw("CREATE (:Chunk {id: 'c1'})")
        for i in range(MAX_DESCRIPTION_MEMBERS + 5):
            await store.upsert_relationships(
                [
                    GraphRelationship(
                        start_node_id="alice",
                        end_node_id="c1",
                        type="MENTIONED_IN",
                        properties={"description": f" wording {i} "},
                    )
                ]
            )
        rows = (
            await store.query_raw(
                "MATCH (:__Entity__ {id: 'alice'})-[m:MENTIONED_IN]->() "
                "RETURN m.description, m.descriptions, m.said_version"
            )
        ).result_set
        ((desc, stored_list, version),) = rows
        assert split_description(desc) == [f"wording {i}" for i in range(MAX_DESCRIPTION_MEMBERS)]
        assert stored_list is None
        assert version == 1

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


class ByChunkLLM(MockLLM):
    """Answers the extraction step by the chunk in its prompt, not by call
    order: one ingest extracts its chunks concurrently."""

    def __init__(self, answers: dict[str, list[tuple[str, str, str]]]) -> None:
        super().__init__()
        self.answers = answers
        self.asked: list[str] = []

    def invoke(self, prompt: str, **kwargs):
        for key, entities in self.answers.items():
            if key in prompt:
                self.asked.append(key)
                found = [{"name": n, "type": t, "description": d} for n, t, d in entities]
                return LLMResponse(content=json.dumps({"entities": found, "relationships": []}))
        return LLMResponse(content='{"entities": [], "relationships": []}')


async def _alice_description(rag):
    rows = (
        await rag._graph_store.query_raw(
            "MATCH (e:__Entity__ {name: 'Alice'}) RETURN e.description, e.descriptions"
        )
    ).result_set
    return rows[0] if rows else None


class TestEndToEnd:
    async def test_a_cached_update_extracts_again_a_chunk_an_earlier_version_wrote(
        self, real_falkordb_rag_factory
    ):
        """Naseem's repro: copied from the cache, an unchanged chunk's edge
        from an earlier version became a marked edge holding nothing, and a
        later unrelated delete rebuilt the node without that chunk's text."""
        from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
        from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
            ExactMatchResolution,
        )

        legacy = "Legacy: works at Acme."
        llm = ByChunkLLM(
            {
                "Alice works at Acme": [("Alice", "Person", legacy)],
                "Carol manages": [("Carol", "Person", "Berlin manager.")],
                "Dave manages": [("Dave", "Person", "Munich manager.")],
                **{
                    f"fact number {i} ": [("Alice", "Person", f"New fact {i}.")]
                    for i in range(1, 6)
                },
            }
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=ExactMatchResolution())
        store = rag._graph_store
        chunker = FixedSizeChunking(chunk_size=64, chunk_overlap=0)
        part1 = "Alice works at Acme Corporation in the city today.".ljust(64, "|")
        await rag.ingest(
            text=part1 + "Carol manages the Berlin office.", document_id="bio", chunker=chunker
        )
        # Alice's edges as an earlier version wrote them: no description, no mark.
        await store.query_raw(
            "MATCH (:__Entity__ {name: 'Alice'})-[m:MENTIONED_IN]->() "
            "SET m.description = null, m.descriptions = null, m.said_version = null"
        )
        for i in range(1, 6):
            await rag.ingest(text=f"Alice fact number {i} is recorded.", document_id=f"n{i}")
        await store.query_raw(
            "MATCH (e:__Entity__ {name: 'Alice'}) SET e.description = $s, "
            "e.descriptions = null, e.description_summarised = true",
            {"s": "Summary, with what the legacy chunk said."},
        )

        result = await rag.update(
            text=part1 + "Dave manages the Munich office.",
            document_id="bio",
            chunker=chunker,
            cache_unchanged_chunks=True,
        )
        # The unchanged chunk is extracted again, not copied.
        assert result.metadata["cache_stats"] == {"cached_chunks": 0, "extracted_chunks": 2}
        assert llm.asked.count("Alice works at Acme") == 2

        await rag.delete_document("n1")
        desc, _ = await _alice_description(rag)
        assert legacy in desc and "New fact 1." not in desc

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
