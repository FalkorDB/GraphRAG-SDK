"""Table-driven tests for the tools read-only Cypher guard."""

from __future__ import annotations

import pytest

from graphrag_sdk.core.exceptions import ReadOnlyViolation
from graphrag_sdk.tools.cypher_guard import apply_limit, ensure_read_only

BENIGN = [
    "MATCH (n:Person) RETURN n.name LIMIT 5",
    "MATCH (n) WHERE n.name = 'DELETE ME' RETURN n",  # write kw inside string
    'MATCH (n) WHERE n.note = "please MERGE later" RETURN n',
    "MATCH (n) WHERE n.created > 2020 RETURN n",  # substring identifier
    "MATCH (a)-[r:RELATES]->(b) RETURN a.name, r.rel_type, b.name",
    "UNWIND $ids AS i MATCH (n {id: i}) RETURN n.name",
    "WITH 1 AS x RETURN x",
    "RETURN 1",
    "CALL db.labels()",
    "CALL db.idx.fulltext.queryNodes('Entity', 'alice') YIELD node RETURN node.name",
    "MATCH (n) // trailing comment\nRETURN n.name",
    "MATCH (n) /* block comment */ RETURN n LIMIT 3",
    "MATCH (n) WHERE n.url = 'http://x.com//path' RETURN n",  # // inside string
    "MATCH (n) WHERE n.x = 'semi;colon' RETURN n",  # ; inside string
    "MATCH (n) RETURN n;",  # single trailing semicolon
    "UNWIND $kws AS kw CALL { WITH kw MATCH (e:__Entity__) WHERE e.name = kw "
    "RETURN e LIMIT 1 } RETURN e",  # CALL subquery
    "OPTIONAL MATCH (n:Chunk) RETURN count(n)",
    "MATCH (n) RETURN n.`create`",  # backtick identifier
]

REJECTED = [  # (query, offending_token) — token = FIRST check that fires:
    # start-keyword check runs before the write scan, and the write scan
    # checks tokens in tuple order (DELETE before DETACH, SET before FOREACH).
    ("CREATE (n:Person {name:'X'})", "CREATE"),
    ("MATCH (n) SET n.x = 1 RETURN n", "SET"),
    ("MATCH (n) DETACH DELETE n", "DELETE"),
    ("MATCH (n) DELETE n", "DELETE"),
    ("MERGE (n:Person {name:'X'}) RETURN n", "MERGE"),
    ("merge(n) return n", "MERGE"),
    ("MATCH (n) REMOVE n.x RETURN n", "REMOVE"),
    ("DROP INDEX ON :Person(name)", "DROP"),
    ("LOAD CSV FROM 'file:///x' AS row RETURN row", "LOAD"),  # start-keyword check
    ("load    csv from 'x' as r return r", "LOAD"),  # start-keyword check
    ("MATCH (n) WITH n LOAD CSV FROM 'x' AS r RETURN r", "LOAD CSV"),  # embedded LOAD CSV
    ("MATCH (n) FOREACH (x IN [1] | SET n.y = x)", "SET"),
    ("Cr/**/eate (n)", "CREATE"),  # comment-split keyword reassembles
    ("ＣＲＥＡＴＥ (n)", "CREATE"),  # fullwidth unicode
    ("CALL db.idx.fulltext.createNodeIndex('L','p')", "CALL db.idx.fulltext.createnodeindex"),
    ("CALL apoc.load.json('u')", "CALL apoc.load.json"),
    ("MATCH (n) RETURN n; MATCH (m) RETURN m", ";"),  # multi-statement
    ("PROFILE MATCH (n) CREATE (m) RETURN n", "PROFILE"),  # start-keyword check fires first
    ("EXPLAIN MATCH (n) RETURN n", "EXPLAIN"),  # disallowed start keyword
    ("", "empty"),
    ("   ", "empty"),
]


@pytest.mark.parametrize("query", BENIGN)
def test_benign_queries_pass(query):
    ensure_read_only(query)  # must not raise


@pytest.mark.parametrize("query,token", REJECTED)
def test_write_queries_rejected_with_token(query, token):
    with pytest.raises(ReadOnlyViolation) as exc_info:
        ensure_read_only(query)
    if token != "empty":
        assert exc_info.value.offending_token is not None
        assert token.lower() in exc_info.value.offending_token.lower()
    assert str(exc_info.value)  # actionable message


def test_apply_limit_injects_when_absent():
    q, injected = apply_limit("MATCH (n) RETURN n", 100)
    assert injected and q.endswith("LIMIT 100")


def test_apply_limit_respects_existing():
    q, injected = apply_limit("MATCH (n) RETURN n LIMIT 3", 100)
    assert not injected and q == "MATCH (n) RETURN n LIMIT 3"


def test_apply_limit_ignores_limit_inside_string():
    q, injected = apply_limit("MATCH (n) WHERE n.x = 'no LIMIT here' RETURN n", 50)
    assert injected and q.endswith("LIMIT 50")


def test_apply_limit_strips_trailing_semicolon():
    q, injected = apply_limit("MATCH (n) RETURN n;", 10)
    assert injected and ";" not in q
