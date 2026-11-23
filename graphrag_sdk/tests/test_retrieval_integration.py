"""Real-FalkorDB tests for the retrieval queries.

The retrieval paths swallow a failed query (it is logged at debug and the
path returns less), so a query an engine rejects only shows as missing
results. These run the queries on a seeded graph and check what comes back.
Gated on RUN_INTEGRATION=1; CI runs them on both FalkorDB engines.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from .conftest import rows

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION") != "1",
        reason="Set RUN_INTEGRATION=1 to run real-FalkorDB tests",
    ),
]

QV = [1.0, 0.0, 0.0, 0.0]
NEAR = "vecf32([1.0, 0.0, 0.0, 0.0])"

SEED = f"""
CREATE (acme:__Entity__:Organization {{id: 'acme', name: 'Acme'}}),
       (:__Entity__:Person {{id: 'alice', name: 'Alice', aliases: ['Ally']}}),
       (:__Entity__:Person {{id: 'bob', name: 'Bob', aliases: 'Bobby'}}),
       (widget:__Entity__:Product {{id: 'widget', name: 'Widget'}}),
       (live:Document {{id: 'doc1'}}),
       (pend:Document {{id: 'doc1__pending__ab12'}}),
       (gone:Document {{id: 'doc2', pending_delete: true}}),
       (c_live:Chunk {{id: 'c_live', text: 'live text', embedding: {NEAR}}}),
       (c_row:Chunk {{id: 'c_row', text: 'row text'}}),
       (c_pend:Chunk {{id: 'c_pend', text: 'pending text', embedding: {NEAR}}}),
       (c_gone:Chunk {{id: 'c_gone', text: 'deleted text', embedding: {NEAR}}}),
       (live)-[:PART_OF]->(c_live), (live)-[:PART_OF]->(c_row),
       (pend)-[:PART_OF]->(c_pend), (gone)-[:PART_OF]->(c_gone),
       (acme)-[:MENTIONED_IN]->(c_live), (acme)-[:MENTIONED_IN]->(c_row),
       (acme)-[:MENTIONED_IN]->(c_pend), (acme)-[:MENTIONED_IN]->(c_gone),
       (acme)-[:RELATES {{rel_type: 'SUPPLIES', consumed__qty: 7}}]->(widget)
"""

# Twelve embedded document edges: more than the walk keeps per entity.
NEIGHBOURS = f"""
MATCH (acme:__Entity__ {{id: 'acme'}})
UNWIND range(0, 11) AS i
CREATE (n:__Entity__:Person {{id: 'n' + toString(i), name: 'Person ' + toString(i)}}),
       (acme)-[:RELATES {{rel_type: 'EMPLOYS', description: 'Acme employs them.',
                          embedding: {NEAR}}}]->(n)
"""

# A 2-hop chunk: Acme -> Person 0 -> chunk, one live, one pending.
TWO_HOP = f"""
MATCH (n:__Entity__ {{id: 'n0'}}), (live:Document {{id: 'doc1'}}),
      (pend:Document {{id: 'doc1__pending__ab12'}})
CREATE (c2:Chunk {{id: 'c2_live', text: 'two hop live', embedding: {NEAR}}}),
       (p2:Chunk {{id: 'c2_pend', text: 'two hop pending', embedding: {NEAR}}}),
       (live)-[:PART_OF]->(c2), (pend)-[:PART_OF]->(p2),
       (n)-[:MENTIONED_IN]->(c2), (n)-[:MENTIONED_IN]->(p2)
"""


@pytest.fixture
async def seeded(graph):
    for cypher in (SEED, NEIGHBOURS, TWO_HOP):
        await rows(graph, cypher)
    return graph


async def test_a_string_aliases_value_does_not_break_name_search(seeded):
    from graphrag_sdk.retrieval.strategies.entity_discovery import discover_entities

    found, _ = await discover_entities(
        seeded, None, ["Alice", "Ally", "Bobby"], ["Alice", "Ally", "Bobby"], use_fulltext=False
    )
    # Bob's aliases is a plain string: it counts as one alias and does not
    # fail the query for every other entity.
    assert {"alice", "bob"} <= set(found)


async def test_the_prompt_alias_pattern_reads_a_string_aliases(seeded):
    # The pattern the text-to-Cypher prompt teaches, on a keyword that misses
    # every name, so the alias side is evaluated for Bob's string aliases.
    aliases = (
        "CASE typeOf(e.aliases) WHEN 'List' THEN e.aliases "
        "WHEN 'String' THEN [e.aliases] ELSE [] END"
    )
    found = await rows(
        seeded,
        "MATCH (e:__Entity__) WHERE toLower(e.name) CONTAINS 'bobb' "
        f"OR any(a IN {aliases} WHERE toLower(a) CONTAINS 'bobb') RETURN e.id",
    )
    assert [r[0] for r in found] == ["bob"]


async def test_text_to_cypher_resolves_names_and_both_alias_shapes(seeded):
    from graphrag_sdk.retrieval.strategies.cypher_generation import _resolve_entities

    found = await _resolve_entities(seeded, ["acme", "Ally", "bobby"], None)
    assert set(found) == {"acme", "alice", "bob"}


async def test_the_walk_keeps_a_table_edge_beside_closer_document_edges(seeded):
    from graphrag_sdk.retrieval.strategies.relationship_expansion import expand_relationships

    lines = await expand_relationships(
        seeded, [("acme", {"name": "Acme"})], max_relationships=20, query_vector=QV
    )
    assert any("Widget" in line and "qty" in line for line in lines), lines
    assert sum("EMPLOYS" in line for line in lines) == 10  # the per-entity cap


async def test_the_agent_walk_keeps_a_table_edge_beside_closer_document_edges(seeded):
    from graphrag_sdk.tools.graph_ops import expand_triples

    triples = await expand_triples(seeded, ["acme"], hops=1, cap=10, query_vector=QV)
    assert len(triples) == 10
    widget = [t for t in triples if t["target"] == "Widget"]
    assert widget and "consumed__qty: 7" in widget[0]["fact"]


async def test_graph_paths_leave_out_chunks_of_inactive_documents(seeded):
    from graphrag_sdk.retrieval.strategies.chunk_retrieval import retrieve_chunks

    vector_store = MagicMock()
    vector_store.fulltext_search_chunks = AsyncMock(return_value=[])
    vector_store.search_chunks = AsyncMock(return_value=[])
    chunks, _, _ = await retrieve_chunks(
        vector_store, seeded, "q", QV, [], [], [("acme", {"name": "Acme"})]
    )
    assert set(chunks) == {"c_live", "c_row", "c2_live"}


async def test_inactive_chunk_ids_finds_pending_and_deleted_documents(seeded):
    from graphrag_sdk.retrieval.strategies.chunk_retrieval import inactive_chunk_ids

    ids = ["c_live", "c_row", "c_pend", "c_gone", "c2_live", "c2_pend"]
    assert set(await inactive_chunk_ids(seeded, ids)) == {"c_pend", "c_gone", "c2_pend"}
