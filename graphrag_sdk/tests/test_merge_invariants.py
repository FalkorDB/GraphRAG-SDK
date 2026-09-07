"""Every merge, everywhere, obeys three rules:

1. the survivor keeps EVERY member's description as a list (``descriptions``)
   plus the ``" | "``-joined string (``description``) that search reads;
2. the survivor carries every member's label (``merged_labels`` at ingest →
   promoted to Cypher labels on write; set directly by the judge);
3. relationship endpoints are re-pointed to the survivor before any loser is
   removed, so no edge is lost.

Covered here: the shared helper and ExactMatchResolution (ingest). Later
layers add LLMVerifiedResolution, the finalize exact phase and the judge.
"""

from __future__ import annotations

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, GraphNode, GraphRelationship
from graphrag_sdk.ingestion.resolution_strategies.base import (
    description_list,
    set_merged_descriptions,
)
from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution


def _n(nid, label, name, desc=None, **props):
    p = {"name": name, **props}
    if desc is not None:
        p["description"] = desc
    return GraphNode(id=nid, label=label, properties=p)


class TestHelpers:
    def test_description_list_from_single_string(self):
        assert description_list({"description": "a"}) == ["a"]

    def test_description_list_splits_legacy_joined_string(self):
        assert description_list({"description": "a | b | a"}) == ["a", "b"]

    def test_description_list_prefers_the_list(self):
        assert description_list({"descriptions": ["x", "y"], "description": "x | y"}) == ["x", "y"]

    def test_set_merged_descriptions_writes_both_forms(self):
        s = _n("s", "Person", "A", "d1")
        got = set_merged_descriptions(s, [_n("b", "Person", "a", "d2"), _n("c", "Person", "a", "d1")])
        assert got == ["d1", "d2"]
        assert s.properties["descriptions"] == ["d1", "d2"]
        assert s.properties["description"] == "d1 | d2"

    def test_chained_merges_flatten(self):
        s = _n("s", "Person", "A", "d1")
        set_merged_descriptions(s, [_n("b", "Person", "a", "d2")])
        set_merged_descriptions(s, [_n("c", "Person", "a", "d3")])
        assert s.properties["descriptions"] == ["d1", "d2", "d3"]


class TestExactMatchAtIngest:
    async def test_three_rules(self):
        nodes = [
            _n("p1", "Person", "Alice", "engineer"),
            _n("p2", "Person", "alice", "born 1970"),
            _n("p3", "Person", "ALICE", "lives in Paris"),
            _n("x", "Location", "Paris", "city"),
        ]
        rels = [
            GraphRelationship(start_node_id="p2", end_node_id="x", type="RELATES", properties={}),
            GraphRelationship(start_node_id="x", end_node_id="p3", type="RELATES", properties={}),
        ]
        res = await ExactMatchResolution(llm=None, cross_label_merge=False).resolve(
            GraphData(nodes=nodes, relationships=rels), Context()
        )
        surv = next(n for n in res.nodes if n.label == "Person")
        assert res.merged_count == 2
        # rule 1
        assert surv.properties["descriptions"] == ["engineer", "born 1970", "lives in Paris"]
        assert surv.properties["description"] == "engineer | born 1970 | lives in Paris"
        # rule 3: both edges now touch the survivor, none dropped
        ends = {(r.start_node_id, r.end_node_id) for r in res.relationships}
        assert ends == {(surv.id, "x"), ("x", surv.id)}
        assert all(r.start_node_id in {surv.id, "x"} and r.end_node_id in {surv.id, "x"} for r in res.relationships)

    async def test_different_labels_are_not_merged_without_llm(self):
        nodes = [_n("a", "Person", "Paris", "prince"), _n("b", "Location", "Paris", "city")]
        res = await ExactMatchResolution(llm=None, cross_label_merge=False).resolve(
            GraphData(nodes=nodes, relationships=[]), Context()
        )
        assert {n.id for n in res.nodes} == {"a", "b"}
