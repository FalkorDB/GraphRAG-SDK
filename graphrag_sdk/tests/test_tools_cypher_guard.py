"""Tests for graphrag_sdk.tools.cypher_guard: the read-only Cypher guard."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from graphrag_sdk.core.models import Ontology, Relation
from graphrag_sdk.tools.cypher_guard import (
    INTERNAL_NODE_DENYLIST,
    STRUCTURAL_RELATIONSHIPS,
    GraphSchema,
    enforce_row_cap,
    introspect_schema,
    is_read_only_cypher,
    mask_internal_labels,
    validate_read_query,
)


def result(rows: list[list[Any]]) -> SimpleNamespace:
    return SimpleNamespace(result_set=rows, header=[])


class FakeGraphStore:
    """Answers query_raw by the first matching (substring, responder) rule."""

    def __init__(self, rules: list[tuple[str, Callable[[str], Any]]] | None = None):
        self.rules = rules or []
        self.calls: list[dict[str, Any]] = []

    async def query_raw(self, cypher, params=None, *, read_only=False, timeout=None):
        self.calls.append({"cypher": cypher, "read_only": read_only})
        for needle, responder in self.rules:
            if needle in cypher:
                out = responder(cypher)
                if isinstance(out, Exception):
                    raise out
                return out
        return result([])


SCHEMA_RULES = [
    ("db.labels()", lambda c: result([["__Entity__"], ["Person"], ["Company"], ["Chunk"]])),
    ("db.relationshipTypes()", lambda c: result([["RELATES"], ["MENTIONED_IN"]])),
    ("db.propertyKeys()", lambda c: result([["id"], ["name"], ["description"], ["age"]])),
]

ONTOLOGY = Ontology(relations=[Relation(label="WORKS_AT")])

SCHEMA = GraphSchema(
    labels=["__Entity__", "Person", "Company"],
    relationship_types=["RELATES", "MENTIONED_IN"],
    property_keys=["id", "name", "age"],
    semantic_relations=["WORKS_AT"],
)


# ── Structural checks (no schema) ────────────────────────────────


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n:Person) RETURN n.name LIMIT 5",
        "MATCH (n) WHERE n.name = 'DELETE ME' RETURN n",
        'MATCH (n) WHERE n.note = "please MERGE later" RETURN n',
        "MATCH (n) WHERE n.created > 2020 RETURN n",
        "MATCH (a)-[r:RELATES]->(b) RETURN a.name, r.rel_type, b.name",
        "UNWIND $ids AS i MATCH (n {id: i}) RETURN n.name",
        "WITH 1 AS x RETURN x",
        "RETURN 1",
        "OPTIONAL MATCH (n:Chunk) RETURN count(n)",
        "MATCH (n) /* block comment */ RETURN n LIMIT 3",
        "MATCH (n) WHERE n.url = 'http://x.com//path' RETURN n",
        "MATCH (n) WHERE n.x = 'semi;colon' RETURN n",
        "MATCH (n) RETURN n;",
        # Indexing and slicing by literal are not computed property access.
        "MATCH (n) RETURN labels(n)[0] AS l",
        "MATCH (n) RETURN collect(n.name)[0..3] AS first",
        "MATCH (n) WHERE n.name IN ['a', 'b'] RETURN [x IN collect(n) WHERE x:Person | x.name]",
        "MATCH (n) // a comment with a 'quote\nRETURN n.name",
        "MATCH (p:Person) WHERE p.name CONTAINS 'call; delete /* x' RETURN p.name",
        "MATCH (n) RETURN [node IN [n] WHERE node:Person | node.name] AS x",
        "MATCH p = (a)-[*1..2]-(b) RETURN nodes(p)[1].name",
        "MATCH (n) WHERE n.text CONTAINS 'embedding' RETURN n.name",
    ],
)
def test_reads_are_accepted(cypher):
    assert validate_read_query(cypher) == [], cypher


@pytest.mark.parametrize(
    "cypher",
    [
        # A comment or a quote cannot hide a procedure, a write or a vector read.
        "MATCH (n) // '\nCALL db.labels() YIELD label // '\nRETURN label",
        "MATCH (c:Chunk) // RETURN '\nRETURN c.embedding AS v // '\n",
        "MATCH (n) /* ' */ DELETE n /* ' */ RETURN 1",
        # Properties read by computed name.
        "MATCH (c:Chunk) RETURN c['embedding']",
        "MATCH (c:Chunk) RETURN [k IN keys(c) | c[k]]",
        "MATCH (c:Chunk) RETURN properties(c)['embedding']",
        "MATCH (n) RETURN n[$p]",
        "MATCH (c:Chunk) RETURN `c`['embedding']",
        "MATCH (c:Chunk) WITH c AS `IN` RETURN `IN`['embedding']",
        "MATCH (c:Chunk) RETURN vec.euclideanDistance(`c`['embedding'], vecf32([0.1]))",
        # An internal label named inside a string still selects the node.
        "MATCH (n) WHERE '__GraphRAGConfig__' IN labels(n) RETURN n",
        # Unterminated tokens.
        "MATCH (n) RETURN n /* unterminated",
        "MATCH (n) RETURN 'unterminated",
        "MATCH (n) RETURN `unterminated",
        # Keywords split by a comment or written in fullwidth do not start a read.
        "Cr/**/eate (n)",
        "ＣＲＥＡＴＥ (n)",
        "PROFILE MATCH (n) CREATE (m) RETURN n",
        "EXPLAIN MATCH (n) RETURN n",
        "DROP INDEX ON :Person(name)",
        "LOAD CSV FROM 'file:///x' AS row RETURN row",
        "load    csv from 'x' as r return r",
        "MATCH (n) WITH n LOAD CSV FROM 'x' AS r RETURN r",
        "merge(n) return n",
        # No procedure calls, read-only ones included.
        "CALL db.labels()",
        "CALL db.idx.fulltext.queryNodes('Entity', 'alice') YIELD node RETURN node.name",
        "CALL db.idx.fulltext.createNodeIndex('L','p')",
        "CALL apoc.load.json('u')",
        "UNWIND $kws AS kw CALL { WITH kw MATCH (e:__Entity__) WHERE e.name = kw "
        "RETURN e LIMIT 1 } RETURN e",
        "",
        "   ",
        "// only a comment",
    ],
)
def test_bypasses_are_rejected(cypher):
    assert validate_read_query(cypher), cypher


@pytest.mark.parametrize(
    ("cypher", "fragment"),
    [
        ("CREATE (n:Person) RETURN n", "must start with"),
        ("MATCH (n) DETACH DELETE n", "'DETACH' is not allowed"),
        ("MATCH (n) SET n.age = 1 RETURN n", "'SET' is not allowed"),
        ("MATCH (n) CALL db.labels() YIELD label RETURN label", "'CALL' is not allowed"),
        ("MATCH (n) RETURN n; MATCH (m) RETURN m", "one statement"),
        ("MATCH (n:Person)", "RETURN clause"),
        ("MATCH (n) RETURN n.embedding", "embedding"),
        ("MATCH (n) RETURN n.description_embedding", "embedding"),
        ("MATCH (c:__GraphRAGConfig__) RETURN c", "internal"),
        ("MATCH (c:Chunk) RETURN c['embedding']", "computed name"),
        ("MATCH (n) RETURN 'x", "unterminated string"),
        ("MATCH (n) RETURN `x", "unterminated identifier"),
        ("MATCH (n) /* x", "unterminated comment"),
        ("   ", "empty"),
    ],
)
def test_rejection_reasons(cypher, fragment):
    errors = validate_read_query(cypher)
    assert any(fragment in e for e in errors), errors


def test_without_schema_only_structural_checks_apply():
    assert validate_read_query("MATCH (p:Robot) RETURN p.salary") == []
    assert validate_read_query("MATCH (n) DELETE n")


def test_errors_are_not_repeated():
    errors = validate_read_query("MATCH (n) SET n.a = 1 SET n.b = 2 RETURN n")
    assert len(errors) == len(set(errors))


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n) RETURN n.delete",
        "MATCH (n {set: 1}) RETURN n.remove AS x",
        "MATCH (n) RETURN n.`create`",
        "MATCH (n) WHERE n.merge = $set RETURN n",
        "MATCH (n) RETURN n.call",
        "MATCH (p) RETURN p.limit",
        "MATCH (p {limit: 5}) RETURN p.skip AS s",
        "MATCH (p) RETURN 1 AS `limit`",
        "MATCH (a {name:'x'}),(b {name:'y'}) "
        "WITH a, b, a.match AS m, shortestPath((a)-[*]->(b)) AS p RETURN length(p)",
    ],
)
def test_keywords_used_as_names_are_not_clauses(cypher):
    assert validate_read_query(cypher) == [], cypher
    assert is_read_only_cypher(cypher)


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n) SET n.x = 1 RETURN n",
        "MATCH (n) SET n:Label RETURN n",
        "MATCH (n)\nSET\nn.a = 1 RETURN n",
        "MATCH (n) DETACH DELETE n RETURN 1",
        "MATCH (n) REMOVE n:L RETURN n",
        "MATCH (n) FOREACH (x IN [1] | SET n.a = 1) RETURN n",
        "MATCH (n) CALL db.labels() YIELD label RETURN label",
        "MATCH (n) WITH n MERGE (m:A) RETURN m",
        "MATCH (n) // x\nCREATE (m) RETURN n",
        'MATCH (n) LOAD CSV FROM "x" AS l RETURN l',
    ],
)
def test_real_writes_are_refused(cypher):
    assert validate_read_query(cypher)
    assert not is_read_only_cypher(cypher)


def test_shortest_path_in_match_is_rejected_with_the_supported_form():
    ends = "MATCH (a {name:'x'}),(b {name:'y'}) "
    bad = ends + "WITH a,b MATCH p = shortestPath((a)-[*]-(b)) RETURN length(p)"
    errors = validate_read_query(bad)
    assert any("WITH shortestPath((a)-[*]->(b)) AS p" in e for e in errors), errors
    good = ends + "WITH shortestPath((a)-[*]->(b)) AS p RETURN length(p)"
    assert validate_read_query(good) == []
    # The cap never changes what the query means.
    assert "shortestPath((a)-[*]->(b))" in enforce_row_cap(good, 5)
    assert "allShortestPaths(" in enforce_row_cap(
        "MATCH (a),(b) WITH a,b MATCH p = allShortestPaths((a)-[*]->(b)) RETURN length(p)", 5
    )


# ── Schema checks ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cypher",
    [
        (
            "MATCH (p:Person)-[r:RELATES]->(c:Company) WHERE r.rel_type = 'WORKS_AT' "
            "RETURN c.name, count(p) AS n"
        ),
        "MATCH (p:Person) WHERE p.name = 'call center: delete' RETURN p.name",
        "MATCH (n) RETURN [x IN collect(n) WHERE x:Person | x.name] AS names",
        "MATCH (n)-[r:RELATES]-(m) WHERE r.rel_type IN ['WORKS_AT'] RETURN m.name",
        "MATCH (n) RETURN n {.name, .age} AS m",
        "UNWIND [1, 2] AS x RETURN x",
        "MATCH (n:__Entity__) RETURN n.name, n.description",
        "MATCH (d:Document)-[:MENTIONED_IN]-(c:Chunk) RETURN count(c)",
        "MATCH (p:Person|Company) RETURN p.name",
    ],
)
def test_schema_accepts_valid_reads(cypher):
    assert validate_read_query(cypher, SCHEMA) == [], cypher


@pytest.mark.parametrize(
    ("cypher", "fragment"),
    [
        ("MATCH (p:Robot) RETURN p.name", "unknown label or relationship type 'Robot'"),
        ("MATCH (p:Person|Robot) RETURN p.name", "unknown label or relationship type 'Robot'"),
        ("MATCH (p)-[r:OWNS|RELATES]->(c) RETURN c.name", "'OWNS'"),
        ("MATCH (p:Person) RETURN p.salary", "unknown property 'salary'"),
        ("MATCH (p:Person)-[r:RELATES]->(c) RETURN c.name", "must constrain r.rel_type"),
        ("MATCH (p)-[:RELATES]->(c) RETURN c.name", "must bind a variable"),
        ("MATCH (p)-[:WORKS_AT]->(c) RETURN c.name", "stored as RELATES"),
        (
            "MATCH (p)-[r:RELATES]->(c) WHERE r.rel_type = 'OWNS' RETURN c.name",
            "unknown relation 'OWNS'",
        ),
        (
            "MATCH (p)-[r:RELATES]->(c) WHERE r.rel_type IN ['WORKS_AT', 'OWNS'] RETURN c.name",
            "unknown relation 'OWNS'",
        ),
    ],
)
def test_schema_rejects(cypher, fragment):
    errors = validate_read_query(cypher, SCHEMA)
    assert any(fragment in e for e in errors), errors


def test_map_keys_and_strings_are_not_labels():
    cypher = "MATCH (p:Person {name: 'Robot: X'}) RETURN p.name"
    assert validate_read_query(cypher, SCHEMA) == []


def test_list_comprehension_label_is_not_mangled():
    schema = GraphSchema(labels=["Person"], relationship_types=["RELATES"], property_keys=["name"])
    cypher = "MATCH (n) RETURN [node IN [n] WHERE node:Person | node.name] AS x"
    assert validate_read_query(cypher, schema) == []


def test_empty_schema_lists_skip_their_checks():
    schema = GraphSchema(semantic_relations=["WORKS_AT"])
    assert validate_read_query("MATCH (p:Robot) RETURN p.salary", schema) == []


def test_good_generated_query_passes_and_is_capped():
    good = (
        "MATCH (p:Person)-[r:RELATES]->(c:Company) WHERE r.rel_type = 'WORKS_AT' "
        "RETURN c.name AS company, count(p) AS people LIMIT 1000"
    )
    assert validate_read_query(good, SCHEMA) == []
    capped = enforce_row_cap(good, 25)
    assert capped.endswith("LIMIT 25") and "LIMIT 1000" not in capped


class TestRequireRelType:
    def test_untyped_bound_edge_is_allowed_when_not_required(self):
        cypher = "MATCH (a)-[r:RELATES]->(b) RETURN r.rel_type, count(*) AS n"
        assert any("must constrain" in e for e in validate_read_query(cypher, SCHEMA))
        assert validate_read_query(cypher, SCHEMA, require_rel_type=False) == []

    def test_unbound_edge_is_allowed_when_not_required(self):
        cypher = "MATCH (a)-[:RELATES]->(b) RETURN a.name, b.name"
        assert any("must bind" in e for e in validate_read_query(cypher, SCHEMA))
        assert validate_read_query(cypher, SCHEMA, require_rel_type=False) == []

    def test_given_rel_type_must_still_be_declared(self):
        cypher = "MATCH (a)-[r:RELATES]->(b) WHERE r.rel_type = 'OWNS' RETURN b.name"
        errors = validate_read_query(cypher, SCHEMA, require_rel_type=False)
        assert any("unknown relation 'OWNS'" in e for e in errors), errors

    def test_relation_as_edge_type_is_still_redirected(self):
        cypher = "MATCH (p)-[:WORKS_AT]->(c) RETURN c.name"
        errors = validate_read_query(cypher, SCHEMA, require_rel_type=False)
        assert any("stored as RELATES" in e for e in errors), errors

    def test_structural_checks_still_apply(self):
        assert validate_read_query("MATCH (n) DELETE n", SCHEMA, require_rel_type=False)


# ── Row cap ──────────────────────────────────────────────────────


class TestRowCap:
    def test_every_limit_is_capped(self):
        assert enforce_row_cap("MATCH (n) WITH n LIMIT 100000 RETURN n LIMIT 5", 25) == (
            "MATCH (n) WITH n LIMIT 25 RETURN n LIMIT 5"
        )

    def test_missing_limit_is_added_and_semicolon_dropped(self):
        assert enforce_row_cap("MATCH (n) RETURN n;", 10) == "MATCH (n) RETURN n\nLIMIT 10"

    def test_existing_smaller_limit_is_kept(self):
        assert enforce_row_cap("MATCH (n) RETURN n LIMIT 3", 100) == "MATCH (n) RETURN n LIMIT 3"

    def test_cap_must_be_positive(self):
        with pytest.raises(ValueError):
            enforce_row_cap("MATCH (n) RETURN n", 0)

    def test_inner_limit_does_not_stand_in_for_the_cap(self):
        assert enforce_row_cap("MATCH (n) WITH n LIMIT 5 MATCH (n)--(m) RETURN m.name", 25) == (
            "MATCH (n) WITH n LIMIT 5 MATCH (n)--(m) RETURN m.name\nLIMIT 25"
        )
        assert enforce_row_cap("MATCH (n) WITH n LIMIT 5 RETURN n.name ORDER BY n.name", 25) == (
            "MATCH (n) WITH n LIMIT 5 RETURN n.name ORDER BY n.name\nLIMIT 25"
        )

    def test_each_union_part_is_capped(self):
        assert enforce_row_cap(
            "MATCH (n) RETURN n.name AS x UNION MATCH (m) RETURN m.name AS x", 9
        ) == ("MATCH (n) RETURN n.name AS x LIMIT 9 UNION MATCH (m) RETURN m.name AS x\nLIMIT 9")
        union = (
            "MATCH (n) RETURN n.name AS x LIMIT {} UNION ALL MATCH (m) RETURN m.name AS x LIMIT 3"
        )
        assert enforce_row_cap(union.format(50), 9) == union.format(9)

    def test_keyword_names_never_move_the_cap(self):
        assert enforce_row_cap("MATCH (p:P) RETURN p.union AS union", 25) == (
            "MATCH (p:P) RETURN p.union AS union\nLIMIT 25"
        )
        assert enforce_row_cap("MATCH (p) WITH p LIMIT 2 RETURN p.`return`", 25) == (
            "MATCH (p) WITH p LIMIT 2 RETURN p.`return`\nLIMIT 25"
        )

    def test_comment_cannot_hide_a_missing_limit(self):
        assert enforce_row_cap("MATCH (n) RETURN n.name // LIMIT 5", 25) == (
            "MATCH (n) RETURN n.name\nLIMIT 25"
        )

    def test_limit_inside_a_string_is_data(self):
        out = enforce_row_cap("MATCH (n) WHERE n.name CONTAINS 'LIMIT 100' RETURN n LIMIT 500", 25)
        assert out == "MATCH (n) WHERE n.name CONTAINS 'LIMIT 100' RETURN n LIMIT 25"
        out = enforce_row_cap("MATCH (n) WHERE n.x = 'no LIMIT here' RETURN n", 50)
        assert out.endswith("\nLIMIT 50") and "'no LIMIT here'" in out

    def test_comments_are_dropped_from_what_runs(self):
        out = enforce_row_cap("MATCH (n) /* hi */ RETURN n LIMIT 3", 25)
        assert "/*" not in out and out.endswith("LIMIT 3")

    @pytest.mark.parametrize("cypher", ["MATCH (n) RETURN 'x", "MATCH (n) RETURN `x", "/* x"])
    def test_unterminated_input_raises(self, cypher):
        with pytest.raises(ValueError):
            enforce_row_cap(cypher, 25)


# ── is_read_only_cypher ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("cypher", "ok"),
    [
        ("MATCH (n) RETURN n", True),
        ("MATCH (n) WHERE n.name = 'call center' RETURN n", True),
        ('MATCH (n) WHERE n.name CONTAINS "delete" RETURN n // don\'t', True),
        ("CREATE (n)", False),
        ("MATCH (n) DELETE n", False),
        ("MERGE (n)", False),
        ("MATCH (n) SET n.x = 1", False),
        ("MATCH (n) // don't\nCREATE (m) // isn't\nRETURN n", False),
        ("MATCH (n) /* it's */ DELETE n", False),
        ("MATCH (n) FOREACH (x IN [1] | CREATE (:A)) RETURN n", False),
        ("CALL db.labels()", False),
        ("MATCH (n) RETURN 'unterminated", False),
        ("MATCH (n) /* unterminated", False),
    ],
)
def test_is_read_only_cypher(cypher, ok):
    assert is_read_only_cypher(cypher) is ok


# ── Masking / constants ──────────────────────────────────────────


def test_mask_internal_labels():
    assert mask_internal_labels("(e:__Entity__)") == "(e:[internal])"
    assert mask_internal_labels("__Entity__ Corp | 1") == "[internal] Corp | 1"
    assert mask_internal_labels("no internals here") == "no internals here"


def test_constants():
    assert "__GraphRAGConfig__" in INTERNAL_NODE_DENYLIST
    assert {"MENTIONED_IN", "PART_OF", "SAME_AS"} <= STRUCTURAL_RELATIONSHIPS
    assert "RELATES" not in STRUCTURAL_RELATIONSHIPS


def test_allowed_names_include_entity_and_reserved_labels():
    allowed = GraphSchema(labels=["Person"], relationship_types=["RELATES"]).allowed_names()
    assert {"Person", "RELATES", "__Entity__", "Chunk", "Document"} <= allowed


# ── Introspection ────────────────────────────────────────────────


async def test_introspection_reads_the_live_schema_read_only():
    store = FakeGraphStore(SCHEMA_RULES)
    schema = await introspect_schema(store, ONTOLOGY)
    assert schema.labels == ["__Entity__", "Person", "Company", "Chunk"]
    assert schema.relationship_types == ["RELATES", "MENTIONED_IN"]
    assert schema.property_keys == ["id", "name", "description", "age"]
    assert schema.semantic_relations == ["WORKS_AT"]
    assert all(c["read_only"] for c in store.calls)


async def test_introspection_without_ontology():
    schema = await introspect_schema(FakeGraphStore(SCHEMA_RULES))
    assert schema.semantic_relations == []


async def test_introspection_failure_leaves_list_empty():
    store = FakeGraphStore([("db.labels()", lambda c: RuntimeError("down")), *SCHEMA_RULES[1:]])
    schema = await introspect_schema(store)
    assert schema.labels == []
    assert schema.relationship_types == ["RELATES", "MENTIONED_IN"]


async def test_introspection_store_without_read_only_keyword():
    class OldStore:
        def __init__(self):
            self.calls: list[str] = []

        async def query_raw(self, cypher, params=None):
            self.calls.append(cypher)
            return result([["Person"], [None]])

    store = OldStore()
    schema = await introspect_schema(store)
    assert schema.labels == ["Person"]
    assert len(store.calls) == 3
