"""Every merge, everywhere, obeys three rules:

1. the survivor keeps EVERY member's description as a list (``descriptions``)
   plus the ``" | "``-joined string (``description``) that search reads;
2. the survivor carries every member's label (``merged_labels`` at ingest →
   promoted to Cypher labels on write; at finalize ``_absorb`` sets the Cypher
   labels and records them in ``merged_labels`` in the same statement);
3. relationship endpoints are re-pointed to the survivor before any loser is
   removed, so no edge is lost.

Covered here: the shared helper, ExactMatchResolution (ingest) and both
LLMVerifiedResolution merge loops. Later layers add the finalize exact phase
and the judge.
"""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, GraphNode, GraphRelationship, LLMResponse
from graphrag_sdk.ingestion.resolution_strategies.base import (
    description_list,
    flatten_remap,
    remap_relationships,
    set_merged_descriptions,
)
from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution
from graphrag_sdk.ingestion.resolution_strategies.llm_verified_resolution import (
    LLMVerifiedResolution,
)
from graphrag_sdk.storage.deduplicator import EntityDeduplicator
from graphrag_sdk.storage.judge_dedup import LLMJudgeDeduplicator

from .conftest import MockLLM
from .test_llm_verified_resolution import ControlledEmbedder, _angle


class _AlwaysYes(MockLLM):
    """Confirms every pair, in both the single and the batched reply format."""

    def __init__(self) -> None:
        super().__init__(responses=["YES"])

    def invoke(self, prompt: str, **kwargs):
        self._call_index += 1
        numbers = re.findall(r"--- Pair (\d+) ---", prompt)
        if numbers:
            return LLMResponse(
                content="\n".join(f"{n}. A vs B | rule: test | YES" for n in numbers)
            )
        return LLMResponse(content="YES\nsame entity")


def _identical_pair(
    a_label="Person",
    b_label="Person",
    a_name="Alice Moreau",
    b_name="A. Moreau",
    a_desc="D1",
    b_desc=None,
    a_props=None,
    b_props=None,
):
    """Two nodes whose embeddings are identical (cosine 1.0), so the pair is a
    candidate at any floor and the merge itself is what the test observes."""
    a = _n("a", a_label, a_name, a_desc, **(a_props or {}))
    b = _n("b", b_label, b_name, b_desc, **(b_props or {}))
    vec = _angle(0.0)
    embedder = ControlledEmbedder({}, default_dim=4)
    embedder._vectors = {}
    embedder.embed_query = lambda text, **kw: vec  # every node lands on one point
    return GraphData(nodes=[a, b], relationships=[]), embedder


def _cluster_with_a_merged_member():
    """A survivor with a plain `description` plus a dup already carrying a
    `descriptions` list, the shape Phase 1 produces before the unified stage."""
    surv = _n("s", "Person", "Alicia Moreau", "D3")
    dup = _n("d", "Person", "Alice Moreau", None)
    dup.properties["descriptions"] = ["D1", "D2"]
    dup.properties["description"] = "D1 | D2"
    vec = _angle(0.0)
    embedder = ControlledEmbedder({}, default_dim=4)
    embedder.embed_query = lambda text, **kw: vec
    return GraphData(nodes=[surv, dup], relationships=[]), embedder


def _n(nid, label, name, desc=None, **props):
    p = {"name": name, **props}
    if desc is not None:
        p["description"] = desc
    return GraphNode(id=nid, label=label, properties=p)


class TestHelpers:
    def test_description_list_from_single_string(self):
        assert description_list({"description": "a"}) == ["a"]

    def test_description_list_keeps_a_lone_description_whole(self):
        """A node holding only ``description`` contributes it as one member.

        Splitting on " | " would recover the members of a pre-``descriptions``
        survivor, but it cannot tell that node from a fresh one whose single
        description contains the separator — and inventing members there buys
        LLM summary calls (they are counted against
        ``force_summary_threshold``) that nothing asked for.
        """
        assert description_list({"description": "CEO | founder of Acme"}) == [
            "CEO | founder of Acme"
        ]

    def test_description_list_prefers_the_members_list(self):
        assert description_list({"descriptions": ["a", "b", "a"], "description": "a | b | a"}) == [
            "a",
            "b",
        ]

    def test_description_list_prefers_the_list(self):
        assert description_list({"descriptions": ["x", "y"], "description": "x | y"}) == ["x", "y"]

    def test_set_merged_descriptions_writes_both_forms(self):
        s = _n("s", "Person", "A", "d1")
        got = set_merged_descriptions(
            s, [_n("b", "Person", "a", "d2"), _n("c", "Person", "a", "d1")]
        )
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
        assert all(
            r.start_node_id in {surv.id, "x"} and r.end_node_id in {surv.id, "x"}
            for r in res.relationships
        )

    async def test_different_labels_are_not_merged_without_llm(self):
        nodes = [_n("a", "Person", "Paris", "prince"), _n("b", "Location", "Paris", "city")]
        res = await ExactMatchResolution(llm=None, cross_label_merge=False).resolve(
            GraphData(nodes=nodes, relationships=[]), Context()
        )
        assert {n.id for n in res.nodes} == {"a", "b"}


class TestLLMVerifiedMergeSites:
    """The two in-memory merge loops of LLMVerifiedResolution obey the rules.

    Driven through ``resolve`` rather than read out of the source. The previous
    version of this file asserted ``src.count("set_merged_descriptions(...)")
    == 2``, which is true no matter WHERE in the loop the call sits — and both
    loops called it after the property copy, which is exactly what loses the
    survivor's own description. A source-text invariant cannot see ordering.
    """

    async def test_unified_stage_keeps_the_survivors_own_description(self, ctx):
        """Rule 1 across phases: a dup carrying a `descriptions` LIST must not
        overwrite the survivor's single `description` before the merge reads it.
        """
        gd, embedder = _cluster_with_a_merged_member()
        llm = _AlwaysYes()
        res = await LLMVerifiedResolution(llm=llm, embedder=embedder, unified_stage=True).resolve(
            gd, Context()
        )
        surv = res.nodes[0]
        assert len(res.nodes) == 1
        assert set(surv.properties["descriptions"]) == {"D1", "D2", "D3"}
        for d in ("D1", "D2", "D3"):
            assert d in surv.properties["description"]

    async def test_unified_stage_unions_source_chunk_ids(self, ctx):
        """Provenance is a merge invariant too: the survivor always has a
        source_chunk_ids, so "copy the keys it lacks" never carried the dup's.
        """
        gd, embedder = _identical_pair(
            a_props={"source_chunk_ids": ["c1"]}, b_props={"source_chunk_ids": ["c2"]}
        )
        res = await LLMVerifiedResolution(
            llm=_AlwaysYes(), embedder=embedder, unified_stage=True
        ).resolve(gd, Context())
        assert len(res.nodes) == 1
        assert set(res.nodes[0].properties["source_chunk_ids"]) == {"c1", "c2"}

    async def test_survivor_label_is_never_listed_as_absorbed(self, ctx):
        """A dup that earlier absorbed a Person hands `merged_labels="Person"`
        to a Person survivor; the survivor must not claim to have absorbed
        its own label.
        """
        gd, embedder = _identical_pair(
            a_label="Person",
            b_label="Engineer",
            b_props={"merged_labels": "Person"},
        )
        res = await LLMVerifiedResolution(
            llm=_AlwaysYes(), embedder=embedder, unified_stage=True, label_family_gate=False
        ).resolve(gd, Context())
        assert len(res.nodes) == 1
        absorbed = [p for p in res.nodes[0].properties["merged_labels"].split(" | ") if p]
        assert res.nodes[0].label not in absorbed
        assert "Engineer" in absorbed

    async def test_cross_label_pass_keeps_the_survivors_own_description(self, ctx):
        """The same rule on the PASS 2 path (unified_stage=False)."""
        gd, embedder = _identical_pair(
            a_label="Person",
            a_desc="D3",
            b_label="Engineer",
            b_props={"descriptions": ["D1", "D2"], "description": "D1 | D2"},
        )
        res = await LLMVerifiedResolution(
            llm=_AlwaysYes(),
            embedder=embedder,
            unified_stage=False,
            cross_label_merge=True,
            cross_label_min_descriptions=0,
            label_family_gate=False,
            cross_label_vote=False,
        ).resolve(gd, Context())
        assert len(res.nodes) == 1
        assert set(res.nodes[0].properties["descriptions"]) == {"D1", "D2", "D3"}


class TestFinalizeExactPhase:
    """Rule 1 and 3 through ``EntityDeduplicator._absorb``, the write every
    finalize phase (exact, resolver, judge) merges through."""

    def _rows(self):
        # id, name, description, label, aliases, is_stub, degree
        return [
            ["e1", "Cape Morrow Light", "a lighthouse", "Location", None, None, 0],
            ["e2", "cape morrow light", "earlier | first lit 1871", "Location", None, None, 0],
        ]

    async def test_three_rules(self):
        graph = MagicMock()
        pages = [SimpleNamespace(result_set=self._rows()), SimpleNamespace(result_set=[])]
        order: list[str] = []

        async def query_raw(q, params=None):
            order.append(q)
            if "RETURN e.id AS id" in q:
                return pages.pop(0) if pages else SimpleNamespace(result_set=[])
            if "DETACH DELETE dup RETURN s.id" in q:
                return SimpleNamespace(result_set=[[params["survivor_id"]]])
            return SimpleNamespace(result_set=[])

        graph.query_raw = AsyncMock(side_effect=query_raw)
        merged = await EntityDeduplicator(graph, MagicMock()).deduplicate()
        assert merged == 1
        write = next(
            c for c in graph.query_raw.call_args_list if "s.descriptions = $descs" in c.args[0]
        )
        # rule 1: list + joined string; survivor = longest description (e2), whose
        # own earlier members are kept intact, then the loser's
        assert write.args[1]["survivor_id"] == "e2"
        assert write.args[1]["descs"] == ["earlier", "first lit 1871", "a lighthouse"]
        assert write.args[1]["desc"] == "earlier | first lit 1871 | a lighthouse"
        # rule 3: every remap query ran before the DETACH DELETE
        i_delete = next(i for i, q in enumerate(order) if "DETACH DELETE dup" in q)
        i_remaps = [i for i, q in enumerate(order) if "MERGE (s)-[" in q or "MERGE (a)-[" in q]
        assert i_remaps and max(i_remaps) < i_delete


class TestJudgeMergeSite:
    """The judge decides identity only; the merge is ``EntityDeduplicator``'s
    ``_absorb`` — one statement carrying the descriptions list, aliases and
    the loser's labels, after every edge is remapped."""

    class _Graph:
        def __init__(self, rows):
            self.rows, self.calls = rows, []
            # id, name, description, label, aliases, is_stub, degree
            self.page = [(r[0], r[1], r[2], r[3][0], r[4], None, 0) for r in rows]

        async def query_raw(self, cypher, params=None):
            self.calls.append((cypher, params or {}))
            if "SKIP" in cypher:
                rows = self.page if params.get("offset", 0) == 0 else []
                return SimpleNamespace(result_set=list(rows))
            if "RETURN e.id, e.name, e.description" in cypher:
                return SimpleNamespace(result_set=list(self.rows))
            if "DETACH DELETE dup RETURN s.id" in cypher:
                return SimpleNamespace(result_set=[[params["survivor_id"]]])
            if "SAME_AS" in cypher:
                return SimpleNamespace(result_set=[[params["a"]]])
            return SimpleNamespace(result_set=[])

    class _Emb:
        async def aembed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

    class _LLM:
        async def abatch_invoke(self, prompts, max_concurrency=8):
            return [
                SimpleNamespace(
                    ok=True, index=i, response=SimpleNamespace(content="SET 1 GROUP: 1, 2")
                )
                for i in range(len(prompts))
            ]

    def test_three_rules(self):
        rows = [
            ("a", "Airbus", "planemaker", ["Organization"], [], [1.0, 0.0], ["planemaker"]),
            ("b", "Airbus SE", "Toulouse group", ["Company"], [], [1.0, 0.0], None),
        ]
        g = self._Graph(rows)
        dd = EntityDeduplicator(g, self._Emb())
        merged = asyncio.run(dd.deduplicate(judge_llm=self._LLM(), judge_vote=False))
        assert merged == 1 and dd.last_judge_stats["merged"] == 1
        (upd,) = [c for c in g.calls if "DETACH DELETE dup RETURN s.id" in c[0]]
        # rule 1: list (survivor = longest description "Toulouse group" first) + join
        assert upd[1]["survivor_id"] == "b" and upd[1]["dup_id"] == "a"
        assert upd[1]["descs"] == ["Toulouse group", "planemaker"]
        assert upd[1]["desc"] == " | ".join(upd[1]["descs"])
        assert upd[1]["aliases"] == ["Airbus"]
        # rule 2: the loser's label lands on the survivor in the same statement
        assert "SET s:`Organization`" in upd[0] and "SET s:`Company`" not in upd[0]
        # rule 3: edges moved before the delete
        i_delete = next(i for i, c in enumerate(g.calls) if "DETACH DELETE" in c[0])
        i_remaps = [
            i for i, c in enumerate(g.calls) if "MERGE (s)-[" in c[0] or "MERGE (a)-[" in c[0]
        ]
        assert i_remaps and max(i_remaps) < i_delete
        assert all(c[1].get("dup_id") == "a" for c in g.calls if "MERGE (s)-[" in c[0])

    def test_the_judge_needs_a_merge_hook(self):
        """``merge_group`` is a required positional parameter with no default:
        the judge cannot be built without a merge site, and builds with one."""
        import inspect

        param = inspect.signature(LLMJudgeDeduplicator).parameters["merge_group"]
        assert param.default is inspect.Parameter.empty
        assert param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        # The signature check above is the "cannot be built without one" half;
        # a literal too-few-arguments call would only trip static analysis.
        judge = LLMJudgeDeduplicator(self._Graph([]), self._Emb(), self._LLM(), MagicMock())
        assert isinstance(judge, LLMJudgeDeduplicator)


class TestRemapChainsAreFlattened:
    """Rule 3, across passes: a multi-pass resolver records each hop
    separately (``dup -> A``, then ``A -> B`` once A is itself merged). A
    single lookup would re-point an edge at A, which the later pass removed,
    leaving it dangling on a node that is not in the graph.
    """

    def test_a_two_hop_chain_lands_on_the_final_survivor(self):
        rels = [
            GraphRelationship(start_node_id="dup", end_node_id="x", type="RELATES", properties={})
        ]
        out = remap_relationships(rels, {"dup": "A", "A": "B"})
        assert [(r.start_node_id, r.end_node_id) for r in out] == [("B", "x")]

    def test_both_endpoints_are_flattened(self):
        rels = [
            GraphRelationship(start_node_id="dup", end_node_id="e1", type="RELATES", properties={})
        ]
        out = remap_relationships(rels, {"dup": "A", "A": "B", "e1": "e2", "e2": "e3"})
        assert [(r.start_node_id, r.end_node_id) for r in out] == [("B", "e3")]

    def test_a_flat_mapping_is_unchanged(self):
        rels = [
            GraphRelationship(start_node_id="d", end_node_id="x", type="RELATES", properties={})
        ]
        out = remap_relationships(rels, {"d": "s"})
        assert [(r.start_node_id, r.end_node_id) for r in out] == [("s", "x")]

    def test_a_cyclic_mapping_terminates(self):
        rels = [
            GraphRelationship(start_node_id="a", end_node_id="x", type="RELATES", properties={})
        ]
        out = remap_relationships(rels, {"a": "b", "b": "a"})
        assert len(out) == 1

    def test_flatten_remap_collapses_every_key(self):
        assert flatten_remap({"d1": "A", "A": "B", "d2": "B"}) == {
            "d1": "B",
            "A": "B",
            "d2": "B",
        }

    def test_chained_endpoints_collapse_to_one_edge(self):
        """Two edges that flatten onto the same pair dedup, as they should."""
        rels = [
            GraphRelationship(start_node_id="dup", end_node_id="x", type="RELATES", properties={}),
            GraphRelationship(start_node_id="A", end_node_id="x", type="RELATES", properties={}),
        ]
        out = remap_relationships(rels, {"dup": "A", "A": "B"})
        assert [(r.start_node_id, r.end_node_id) for r in out] == [("B", "x")]
