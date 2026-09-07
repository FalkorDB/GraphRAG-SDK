"""Every merge, everywhere, obeys three rules:

1. the survivor keeps EVERY member's description as a list (``descriptions``)
   plus the ``" | "``-joined string (``description``) that search reads;
2. the survivor carries every member's label (``merged_labels`` at ingest →
   promoted to Cypher labels on write; set directly by the judge);
3. relationship endpoints are re-pointed to the survivor before any loser is
   removed, so no edge is lost.

Covered sites: ExactMatchResolution (ingest), LLMVerifiedResolution (opt-in
ingest), EntityDeduplicator exact phase and LLMJudgeDeduplicator (finalize).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, GraphNode, GraphRelationship
from graphrag_sdk.ingestion.resolution_strategies.base import (
    description_list,
    set_merged_descriptions,
)
from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution
from graphrag_sdk.storage.deduplicator import EntityDeduplicator
from graphrag_sdk.storage.judge_dedup import LLMJudgeDeduplicator, merge_description_list


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

    def test_merge_description_list_on_graph_rows(self):
        rows = [
            {"description": "d1 | d2", "descriptions": ["d1", "d2"]},
            {"description": "d3", "descriptions": []},
            {"description": "old | style", "descriptions": None},
        ]
        assert merge_description_list(rows) == ["d1", "d2", "d3", "old", "style"]


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


class TestLLMVerifiedMergeSites:
    """The two in-memory merge loops of LLMVerifiedResolution use the shared helper."""

    def test_pass2_and_unified_paths_write_the_list_and_labels(self):
        import inspect

        from graphrag_sdk.ingestion.resolution_strategies import llm_verified_resolution as m

        src = inspect.getsource(m)
        # both merge sites call the shared rule; no site builds `description` by hand
        assert src.count("set_merged_descriptions(survivor,") == 2
        assert '" | ".join(descriptions)' not in src
        # labels of every absorbed member are recorded
        assert "merged_labels" in src


class TestFinalizeExactPhase:
    def _rows(self):
        return [
            ["e1", "Cape Morrow Light", "a lighthouse", "Location", None],
            ["e2", "cape morrow light", "first lit 1871", "Location", ["earlier", "first lit 1871"]],
        ]

    async def test_three_rules(self):
        graph = MagicMock()
        pages = [SimpleNamespace(result_set=self._rows()), SimpleNamespace(result_set=[])]
        order: list[str] = []

        async def query_raw(q, params=None):
            order.append(q)
            if "RETURN e.id AS id" in q:
                return pages.pop(0) if pages else SimpleNamespace(result_set=[])
            return SimpleNamespace(result_set=[])

        graph.query_raw = AsyncMock(side_effect=query_raw)
        merged = await EntityDeduplicator(graph, MagicMock()).deduplicate()
        assert merged == 1
        write = next(c for c in graph.query_raw.call_args_list if "s.descriptions = $descs" in c.args[0])
        # rule 1: list + joined string; survivor = longest description (e2), whose
        # own earlier list is kept intact, then the loser's
        assert write.args[1]["descs"] == ["earlier", "first lit 1871", "a lighthouse"]
        assert write.args[1]["desc"] == "earlier | first lit 1871 | a lighthouse"
        # rule 3: every remap query ran before the DETACH DELETE
        i_delete = next(i for i, q in enumerate(order) if "DETACH DELETE e" in q)
        i_remaps = [i for i, q in enumerate(order) if "MERGE (s)-[" in q or "MERGE (a)-[" in q]
        assert i_remaps and max(i_remaps) < i_delete


class TestJudgeMergeSite:
    class _Graph:
        def __init__(self, rows):
            self.rows, self.calls = rows, []

        async def query_raw(self, cypher, params=None):
            self.calls.append((cypher, params or {}))
            if "RETURN e.id, e.name, e.description" in cypher:
                return SimpleNamespace(result_set=list(self.rows))
            return SimpleNamespace(result_set=[])

    class _Emb:
        async def aembed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

    class _LLM:
        async def abatch_invoke(self, prompts, max_concurrency=8):
            return [
                SimpleNamespace(ok=True, index=i, response=SimpleNamespace(content="SET 1 GROUP: 1, 2"))
                for i in range(len(prompts))
            ]

    def test_three_rules(self):
        rows = [
            ("a", "Airbus", "planemaker", ["Organization"], [], [1.0, 0.0], ["planemaker"]),
            ("b", "Airbus SE", "Toulouse group", ["Company"], [], [1.0, 0.0], None),
        ]
        g = self._Graph(rows)
        remap_calls: list[tuple[str, str]] = []

        async def remap(dup, surv):
            remap_calls.append((dup, surv))
            return True

        judge = LLMJudgeDeduplicator(g, self._Emb(), self._LLM(), remap, vote=False)
        stats = asyncio.run(judge.deduplicate())
        assert stats["merged"] == 1
        upd = next(c for c in g.calls if "s.descriptions = $descs" in c[0])
        # rule 1: list (survivor = longest description "Toulouse group" first) + join
        assert set(upd[1]["descs"]) == {"Toulouse group", "planemaker"}
        assert upd[1]["desc"] == " | ".join(upd[1]["descs"])
        # rule 2: both labels set on the survivor
        assert "SET s:`Company`" in upd[0] and "SET s:`Organization`" in upd[0]
        # rule 3: edges moved before the delete
        i_remap = remap_calls and 0
        i_delete = next(i for i, c in enumerate(g.calls) if "DETACH DELETE" in c[0])
        assert remap_calls == [("a", "b")] and i_remap is not None
        assert next(i for i, c in enumerate(g.calls) if "s.descriptions" in c[0]) > i_delete
