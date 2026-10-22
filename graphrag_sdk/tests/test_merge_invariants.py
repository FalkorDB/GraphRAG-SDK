"""Every merge, everywhere, obeys three rules:

1. the survivor keeps EVERY member's description as a list (``descriptions``)
   plus the ``" | "``-joined string (``description``) that search reads;
2. the survivor carries every member's label (``merged_labels`` at ingest →
   promoted to Cypher labels on write; at finalize the merge query sets the
   Cypher labels ``EntityDeduplicator._fold`` computed and records them in
   ``merged_labels`` in the same statement);
3. relationship endpoints are re-pointed to the survivor before any loser is
   removed, so no edge is lost.

Covered here: the shared helper, ExactMatchResolution (ingest), both
LLMVerifiedResolution merge loops, and finalize's merge write with the judge
that decides it.
"""

from __future__ import annotations

import re

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
from graphrag_sdk.storage.dedup_judges import DedupJudge, LLMJudge
from graphrag_sdk.storage.merge_query import absorb_query

from .conftest import MockLLM
from .dedup_sim import AllSameJudge, NameOracleLLM, simulate
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


class TestFinalizeMergeFold:
    """Rules 1 and 3 through ``EntityDeduplicator._fold`` — the write every
    finalize phase (exact, adoption, judge) merges through, reached here by
    judged merges — and the merge query that carries it."""

    # id, name, description, label
    _ROWS = [
        ["e1", "Cape Morrow Light", "a lighthouse", "Location"],
        ["e2", "cape morrow light", "earlier | first lit 1871", "Location"],
    ]
    # e2 is a survivor of an earlier merge: it holds the members list as well
    # as the joined text, the shape every merge writes.
    _STORED = {"e2": {"props": {"descriptions": ["earlier", "first lit 1871"]}}}

    @staticmethod
    def _description_writes(sim):
        return [w for w in sim.writes if "s.descriptions = $descs" in w.sets]

    async def test_three_rules(self):
        sim = await simulate(self._ROWS, stored=self._STORED, judge=AllSameJudge())
        assert sim.merged == 1
        (write,) = self._description_writes(sim)
        # rule 1: list + joined string; survivor = longest description (e2), whose
        # own earlier members are kept intact, then the loser's
        assert write.op.survivor == "e2"
        assert write.params["descs"] == ["earlier", "first lit 1871", "a lighthouse"]
        assert write.params["desc"] == "earlier | first lit 1871 | a lighthouse"
        # rule 3: the one merge query moves every edge before its delete
        q = absorb_query([("RELATES", "out"), ("RELATES", "in")], write.sets, write.labels)
        assert "MERGE (s)-[nr:`RELATES`" in q and "MERGE (b)-[nr:`RELATES`" in q
        assert q.rindex("MERGE ") < q.index("DETACH DELETE d")

    async def test_a_lone_description_is_one_member_even_with_the_separator_in_it(self):
        """Rule 1 by ``description_list``'s rule: a node holding only
        ``description`` is ONE member, not split on ``" | "``. Nothing tells a
        pre-``descriptions`` survivor from a fresh node whose single text
        contains the separator, and splitting the fresh one invents members
        that were never written — persisted, and counted toward
        ``force_summary_threshold``."""
        rows = [
            ["e1", "Maya Ellison", "CEO | founder of Acme", "Person"],
            ["e2", "maya ellison", "born 1970", "Person"],
        ]
        sim = await simulate(rows, judge=AllSameJudge())
        assert sim.merged == 1
        (write,) = self._description_writes(sim)
        assert write.op.survivor == "e1"
        assert write.params["descs"] == ["CEO | founder of Acme", "born 1970"]
        assert write.params["desc"] == "CEO | founder of Acme | born 1970"

    async def test_the_same_description_on_both_nodes_still_records_the_member(self):
        """Rule 1 when the text does not change: two nodes with the same
        description and no list. The joined string is the survivor's own, so
        a write gated on the text alone skipped the list — and the deleted
        node's member was never recorded. The list is written whenever the
        graph's list is not these members."""
        rows = [
            ["e1", "Alice", "an engineer", "Person"],
            ["e2", "alice", "an engineer", "Person"],
        ]
        sim = await simulate(rows, judge=AllSameJudge())
        assert sim.merged == 1
        (write,) = self._description_writes(sim)
        assert write.params["descs"] == ["an engineer"]
        assert write.params["desc"] == "an engineer"

    async def test_a_list_left_short_by_an_earlier_writer_is_brought_up_to_date(self):
        """The survivor's text already holds the duplicate's member, but the
        stored list does not: joined text unchanged, list out of step. The
        write is gated on the list as well, so the member lands in it."""
        rows = [
            ["e1", "Cape Morrow Light", "a lighthouse | first lit 1871", "Location"],
            ["e2", "cape morrow light", "first lit 1871", "Location"],
        ]
        sim = await simulate(
            rows,
            stored={"e1": {"props": {"descriptions": ["a lighthouse"]}}},
            judge=AllSameJudge(),
        )
        assert sim.merged == 1
        (write,) = self._description_writes(sim)
        assert write.op.survivor == "e1"
        assert write.params["descs"] == ["a lighthouse", "first lit 1871"]
        assert write.params["desc"] == "a lighthouse | first lit 1871"

    async def test_an_empty_stored_list_does_not_hide_the_survivors_description(self):
        """``description_list`` prefers the list wherever it is present; an
        empty one would make the survivor's own text vanish from the merge.
        The text is still a member — the duplicate's copy is about to go."""
        rows = [
            ["e1", "Alice", "an engineer", "Person"],
            ["e2", "alice", "born 1970", "Person"],
        ]
        sim = await simulate(
            rows, stored={"e1": {"props": {"descriptions": []}}}, judge=AllSameJudge()
        )
        assert sim.merged == 1
        (write,) = self._description_writes(sim)
        assert write.params["descs"] == ["an engineer", "born 1970"]

    def test_a_fold_computed_from_a_stale_read_is_not_written(self):
        """The fold is built from what the graph held when the merge read it.
        Written over a node changed since, it would replace that change — so
        the query re-checks every field the fold was computed from, and writes
        nothing (the merge reads again) when one moved."""
        q = absorb_query([], ["s.description = $desc"], [])
        for field in ("s_desc", "s_descs", "s_aliases", "s_merged", "d_desc", "d_aliases"):
            assert f"${field}" in q
        assert "AS fresh" in q and "(present AND fresh AND unhandled = 0) AS ok" in q


class TestJudgeMergeSite:
    """The judge decides identity only; the merge is ``EntityDeduplicator``'s
    — one fold carrying the descriptions list, aliases and the loser's labels,
    applied by the one query that moves every edge."""

    async def test_three_rules(self):
        rows = [
            ["a", "Airbus", "planemaker", ["Organization"]],
            ["b", "Airbus SE", "Toulouse group", ["Company"]],
        ]
        sim = await simulate(
            rows,
            stored={"a": {"props": {"descriptions": ["planemaker"]}}},
            judge=NameOracleLLM(rounds=["SET 1 GROUP: 1, 2\nEND"]),
        )
        assert sim.merged == 1 and sim.dedup.last_judge_stats["merged"] == 1
        (write,) = sim.writes
        # rule 1: list (survivor = longest description "Toulouse group" first) + join
        assert (write.op.survivor, write.op.dup) == ("b", "a")
        assert write.params["descs"] == ["Toulouse group", "planemaker"]
        assert write.params["desc"] == " | ".join(write.params["descs"])
        assert write.params["aliases"] == ["Airbus"]
        # rule 2: the loser's label lands on the survivor in the same query
        assert write.labels == ["Organization"]
        q = absorb_query([("RELATES", "out")], write.sets, write.labels)
        assert "SET s:`Organization`" in q and "SET s:`Company`" not in q
        assert sim.nodes["b"]["labels"] == ["Company", "Organization"]

    def test_the_judge_writes_nothing(self):
        """A judge has no merge site to call, nor a graph: it is given sets and
        returns verdicts, and the deduplicator decides and writes every merge
        from a sealed plan."""
        import inspect

        assert list(inspect.signature(DedupJudge.judge).parameters) == ["self", "sets", "retry"]
        params = set(inspect.signature(LLMJudge).parameters)
        assert "llm" in params and not params & {"graph", "graph_store", "store"}


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
