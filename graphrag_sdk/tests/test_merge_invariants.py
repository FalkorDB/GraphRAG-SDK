"""Every merge, everywhere, obeys three rules:

1. the survivor keeps EVERY member's description as a list (``descriptions``)
   plus the ``" | "``-joined string (``description``) that search reads;
2. the survivor carries every member's label (``merged_labels`` at ingest →
   promoted to Cypher labels on write; set directly by the judge);
3. relationship endpoints are re-pointed to the survivor before any loser is
   removed, so no edge is lost.

Covered here: the shared helper, ExactMatchResolution (ingest) and both
LLMVerifiedResolution merge loops. Later layers add the finalize exact phase
and the judge.
"""

from __future__ import annotations

import re

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, GraphNode, GraphRelationship, LLMResponse
from graphrag_sdk.ingestion.resolution_strategies.base import (
    description_list,
    set_merged_descriptions,
)
from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution
from graphrag_sdk.ingestion.resolution_strategies.llm_verified_resolution import (
    LLMVerifiedResolution,
)

from .test_llm_verified_resolution import ControlledEmbedder, _angle
from .conftest import MockLLM


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
        res = await LLMVerifiedResolution(
            llm=llm, embedder=embedder, unified_stage=True
        ).resolve(gd, Context())
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


@pytest.fixture
def ctx():
    return Context()
