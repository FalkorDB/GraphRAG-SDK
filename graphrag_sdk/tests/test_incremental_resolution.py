"""Tests for IncrementalResolution — resolve a batch against the existing graph."""

from __future__ import annotations

import json

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, GraphNode, LLMResponse
from graphrag_sdk.core.providers import Embedder
from graphrag_sdk.core.providers.base import LLMBatchItem
from graphrag_sdk.ingestion.resolution_strategies.incremental_resolution import (
    IncrementalResolution,
    normalize_name,
)

from .conftest import MockLLM


class CapturingLLM(MockLLM):
    """Records the prompts sent to ``abatch_invoke`` for assertions."""

    def __init__(self, response: str) -> None:
        super().__init__(responses=[response])
        self.prompts: list[str] = []

    async def abatch_invoke(self, prompts, **kw):
        self.prompts.extend(prompts)
        return [
            LLMBatchItem(index=i, response=LLMResponse(content=self._responses[0]))
            for i in range(len(prompts))
        ]


class WordEmbedder(Embedder):
    """Deterministic bag-of-words embedder — shared words → high cosine."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    @property
    def model_name(self) -> str:
        return "word-embedder"

    def embed_query(self, text: str, **kw):
        toks = str(text).lower().split()
        vec = [0.0] * 64
        for t in toks:
            self._vocab.setdefault(t, len(self._vocab) % 64)
            vec[self._vocab[t] % 64] += 1.0
        return vec or [0.0] * 64


class ScriptedEmbedder(Embedder):
    """Returns a fixed vector per exact text, for precise cosine control."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    @property
    def model_name(self) -> str:
        return "scripted"

    def embed_query(self, text: str, **kw):
        return self._table.get(str(text), [1.0, 0.0])


def _ctx():
    return Context(tenant_id="t", latency_budget_ms=5000.0)


# Existing graph nodes (candidates the retriever can return).
GAL_SH = GraphNode(
    id="gal_sh__person",
    label="Person",
    properties={"name": "Gal Sh", "description": "Engineer at FalkorDB."},
)
GAL_BR = GraphNode(
    id="gal_br__person",
    label="Person",
    properties={"name": "Gal Br", "description": "Designer at another firm."},
)


def make_retriever(returns):
    async def retriever(name, description, k):
        return returns

    return retriever


class TestIncrementalResolution:
    async def test_links_new_entities_into_existing_node_and_rejects_lookalike(self):
        """gal / gal.sh / Gal Shubeli → merge INTO existing Gal Sh;
        Gal Kurland → new; Gal Br (candidate) → rejected."""
        batch = [
            GraphNode(
                id="gal__person",
                label="Person",
                properties={"name": "gal", "description": "works at FalkorDB"},
            ),
            GraphNode(
                id="gal.sh__person",
                label="Person",
                properties={"name": "gal.sh", "description": "FalkorDB engineer"},
            ),
            GraphNode(
                id="gal_shubeli__person",
                label="Person",
                properties={"name": "Gal Shubeli", "description": "engineer at FalkorDB"},
            ),
            GraphNode(
                id="gal_kurland__person",
                label="Person",
                properties={"name": "Gal Kurland", "description": "researcher elsewhere"},
            ),
        ]
        # LLM sees refs 1..4 = new (gal, gal.sh, Gal Shubeli, Gal Kurland),
        # refs 5,6 = graph (Gal Sh, Gal Br).
        decision = json.dumps(
            {
                "groups": [
                    {
                        "members": [1, 2, 3, 5],
                        "target": 5,
                        "canonical": "Gal Shubeli",
                        "type": "Person",
                        "description": "Engineer at FalkorDB.",
                    },
                    {
                        "members": [4],
                        "target": "new",
                        "canonical": "Gal Kurland",
                        "type": "Person",
                        "description": "A researcher.",
                    },
                ]
            }
        )
        resolver = IncrementalResolution(
            llm=MockLLM(responses=[decision]),
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([GAL_SH, GAL_BR]),
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())

        ids = {n.id for n in res.nodes}
        # Merged entity carries the EXISTING graph id, and Gal Kurland is new.
        assert "gal_sh__person" in ids, "new mentions should link into existing Gal Sh"
        assert "gal_kurland__person" in ids, "Gal Kurland stays a new entity"
        assert len(res.nodes) == 2
        # Gal Br was a candidate only — never written as a batch node.
        assert "gal_br__person" not in ids
        # All three gal-variants remap onto the existing node.
        for old in ("gal__person", "gal.sh__person", "gal_shubeli__person"):
            assert res.remap.get(old) == "gal_sh__person"
        merged = next(n for n in res.nodes if n.id == "gal_sh__person")
        assert merged.properties["name"] == "Gal Shubeli"

    async def test_no_candidates_means_new_entity_no_llm(self):
        """A survivor with no graph candidates is created as new — LLM untouched."""
        batch = [
            GraphNode(
                id="novel__person",
                label="Person",
                properties={"name": "Nadia Q", "description": "brand new person"},
            )
        ]

        class BoomLLM(MockLLM):
            def invoke(self, *a, **k):
                raise AssertionError("LLM must not be called when there are no candidates")

        resolver = IncrementalResolution(
            llm=BoomLLM(responses=[""]),
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([]),  # nothing similar in graph
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())
        assert len(res.nodes) == 1
        assert res.nodes[0].id == "novel__person"

    async def test_free_merge_collapses_same_name_before_llm(self):
        """Same-name/different-type homograph merges for free in stage 1."""
        batch = [
            GraphNode(
                id="graphrag__concept",
                label="Concept",
                properties={"name": "GraphRAG", "description": "graph based RAG technique"},
            ),
            GraphNode(
                id="graphrag__technology",
                label="Technology",
                properties={"name": "GraphRAG", "description": "graph based RAG technique"},
            ),
        ]
        resolver = IncrementalResolution(
            llm=MockLLM(responses=[""]),
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([]),  # no graph yet
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())
        assert len(res.nodes) == 1, "GraphRAG Concept+Technology should free-merge"
        assert res.remap.get("graphrag__technology") == "graphrag__concept"

    async def test_immutable_conflict_flags_review(self):
        """Merging nodes that disagree on an immutable prop flags _needs_review."""
        batch = [
            GraphNode(
                id="acme_a__org",
                label="Org",
                properties={"name": "Acme", "description": "a company", "founded": "2001"},
            ),
            GraphNode(
                id="acme_b__org",
                label="Org",
                properties={"name": "Acme", "description": "a company", "founded": "1998"},
            ),
        ]
        resolver = IncrementalResolution(
            llm=MockLLM(responses=[""]),
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([]),
            immutable_props=("founded",),
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())
        assert len(res.nodes) == 1
        assert res.nodes[0].properties.get("_needs_review") is True
        conflicts = res.nodes[0].properties.get("_merge_conflicts")
        # Recorded as strings so the graph store (which drops lists of dicts) persists them.
        assert conflicts and all(isinstance(c, str) for c in conflicts)
        assert any("founded" in c for c in conflicts)

    async def test_genuine_homograph_stays_separate(self):
        """Same name, different type, divergent descriptions → NOT free-merged."""
        batch = [
            GraphNode(
                id="paris__location",
                label="Location",
                properties={"name": "Paris", "description": "capital city france europe"},
            ),
            GraphNode(
                id="paris__person",
                label="Person",
                properties={"name": "Paris", "description": "american media personality celebrity"},
            ),
        ]
        resolver = IncrementalResolution(
            llm=MockLLM(responses=[""]),
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([]),
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())
        assert len(res.nodes) == 2, "distinct homographs must stay separate"

    async def test_malformed_llm_response_leaves_pile_untouched(self):
        """An unparseable partition → fail-safe: no merges applied."""
        batch = [
            GraphNode(id="xylo__t", label="T", properties={"name": "Xylo", "description": "d"})
        ]
        resolver = IncrementalResolution(
            llm=MockLLM(responses=["not valid json"]),  # consulted (has a candidate)
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([GAL_SH]),
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())
        assert len(res.nodes) == 1
        assert res.nodes[0].id == "xylo__t"

    async def test_linking_into_existing_node_preserves_its_label(self):
        """Graph writes are label-scoped, so linking into an existing node must
        keep that node's label even when the LLM returns a conflicting type —
        otherwise the store would create a second same-id node."""
        batch = [
            GraphNode(
                id="gal_var__person",
                label="Person",
                properties={"name": "gal.sh", "description": "FalkorDB engineer"},
            ),
        ]
        # ref 1 = new (gal.sh), ref 2 = graph (Gal Sh / Person); LLM says "Robot".
        decision = json.dumps(
            {
                "groups": [
                    {
                        "members": [1, 2],
                        "target": 2,
                        "canonical": "Gal Shubeli",
                        "type": "Robot",
                        "description": "Engineer.",
                    },
                ]
            }
        )
        resolver = IncrementalResolution(
            llm=MockLLM(responses=[decision]),
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([GAL_SH]),
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())
        merged = next(n for n in res.nodes if n.id == "gal_sh__person")
        assert merged.label == "Person", "existing node's label must be preserved on link"

    async def test_string_or_float_target_still_links(self):
        """LLM refs emitted as strings/floats still resolve and link correctly."""
        batch = [
            GraphNode(
                id="gal_var__person",
                label="Person",
                properties={"name": "gal", "description": "engineer"},
            ),
        ]
        decision = json.dumps(
            {
                "groups": [
                    {
                        "members": ["1", 2.0],
                        "target": "2",  # str + float, as some LLMs emit
                        "canonical": "Gal Sh",
                        "type": "Person",
                        "description": "d",
                    },
                ]
            }
        )
        resolver = IncrementalResolution(
            llm=MockLLM(responses=[decision]),
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([GAL_SH]),
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())
        assert res.remap.get("gal_var__person") == "gal_sh__person"

    async def test_retriever_failure_degrades_to_new_entity(self):
        """A retriever that raises must not abort resolution — the entity is
        simply treated as new (fail toward splitting)."""

        async def boom(name, description, k):
            raise RuntimeError("graph store unavailable")

        batch = [
            GraphNode(id="solo__t", label="T", properties={"name": "Solo", "description": "d"}),
        ]
        resolver = IncrementalResolution(
            llm=MockLLM(responses=[""]),
            embedder=WordEmbedder(),
            candidate_retriever=boom,
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())
        assert len(res.nodes) == 1
        assert res.nodes[0].id == "solo__t"

    async def test_conflict_flag_survives_later_absorption(self):
        """A node flagged in stage 1 keeps its review flag if it is absorbed as a
        loser during stage-4 linking (Naseem #10 / CodeRabbit #7)."""
        batch = [
            GraphNode(
                id="bee__t", label="T", properties={"name": "Bee", "description": "unrelated"}
            ),
            GraphNode(
                id="node_a__t",
                label="T",
                properties={"name": "Node", "description": "d", "founded": "2001"},
            ),
            GraphNode(
                id="node_b__t",
                label="T",
                properties={"name": "Node", "description": "d", "founded": "1998"},
            ),
        ]
        # Stage 1 merges the two "Node" rows (flagging the founded conflict); stage 4
        # groups the survivor "Node" with "Bee" (Bee first → carrier), absorbing the
        # flagged node. The flag must propagate to Bee.
        decision = json.dumps(
            {
                "groups": [
                    {
                        "members": [1, 2],
                        "target": "new",
                        "canonical": "Merged",
                        "type": "T",
                        "description": "d",
                    },
                ]
            }
        )
        resolver = IncrementalResolution(
            llm=MockLLM(responses=[decision]),
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([GAL_SH]),  # both share a candidate → one pile
            immutable_props=("founded",),
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())
        carrier = next(n for n in res.nodes if n.id == "bee__t")
        assert carrier.properties.get("_needs_review") is True
        assert any("founded" in c for c in carrier.properties.get("_merge_conflicts", []))

    async def test_no_description_homograph_not_merged(self):
        """Same name, different type, NO descriptions → not auto-merged (Naseem #12).
        Description would fall back to the name (cosine 1.0); that isn't evidence."""
        batch = [
            GraphNode(id="paris__location", label="Location", properties={"name": "Paris"}),
            GraphNode(id="paris__person", label="Person", properties={"name": "Paris"}),
        ]
        resolver = IncrementalResolution(
            llm=MockLLM(responses=[""]),
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([]),
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())
        assert len(res.nodes) == 2, "no-evidence homographs must stay separate"

    async def test_large_pile_still_links_candidates(self):
        """A survivor pile larger than pile_cap is chunked, so candidates still
        reach the LLM and linking works (Naseem #9 / CodeRabbit #6)."""
        hub = GraphNode(
            id="hub__person", label="Person", properties={"name": "Hub", "description": "shared"}
        )
        batch = [
            GraphNode(
                id=f"e{i}__person",
                label="Person",
                properties={"name": f"Name{i}", "description": f"desc {i}"},
            )
            for i in range(5)
        ]
        # pile_cap=4, top_k=1 → survivor budget 3 → chunks [0,1,2] and [3,4].
        # chunk 1: refs 1-3 survivors, ref 4 = Hub → link ref1 into ref4.
        # chunk 2: refs 1-2 survivors, ref 3 = Hub → link ref1 into ref3.
        r1 = json.dumps(
            {
                "groups": [
                    {
                        "members": [1, 4],
                        "target": 4,
                        "canonical": "Hub",
                        "type": "Person",
                        "description": "d",
                    }
                ]
            }
        )
        r2 = json.dumps(
            {
                "groups": [
                    {
                        "members": [1, 3],
                        "target": 3,
                        "canonical": "Hub",
                        "type": "Person",
                        "description": "d",
                    }
                ]
            }
        )
        resolver = IncrementalResolution(
            llm=MockLLM(responses=[r1, r2]),
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([hub]),
            top_k=1,
            pile_cap=4,
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())
        # A survivor from each chunk linked into the existing hub — candidates were
        # NOT all evicted by the survivor count.
        assert res.remap.get("e0__person") == "hub__person"
        assert res.remap.get("e3__person") == "hub__person"
        # Two chunks both retargeted a carrier to hub — the output must not contain
        # the same id twice (else the store would MERGE it twice and clobber data).
        ids = [n.id for n in res.nodes]
        assert len(ids) == len(set(ids)), "resolved nodes must have unique ids"

    async def test_ragged_embedder_does_not_crash(self):
        """A malformed embedder returning ragged vectors degrades to no-merge
        rather than crashing the document (CodeRabbit #8)."""

        class RaggedEmbedder(Embedder):
            @property
            def model_name(self):
                return "ragged"

            def embed_query(self, text, **kw):
                return [1.0] * (len(str(text)) % 3 + 1)  # inconsistent lengths

        batch = [
            GraphNode(
                id="p__location",
                label="Location",
                properties={"name": "Paris", "description": "abc"},
            ),
            GraphNode(
                id="p__person", label="Person", properties={"name": "Paris", "description": "abcd"}
            ),
        ]
        resolver = IncrementalResolution(
            llm=MockLLM(responses=[""]),
            embedder=RaggedEmbedder(),
            candidate_retriever=make_retriever([]),
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())  # must not raise
        assert len(res.nodes) == 2

    async def test_full_description_sent_to_llm_not_truncated(self):
        """The merge prompt must send the FULL description, not a 180-char snippet
        (Naseem #3) — else repeated merges erode rich descriptions."""
        long_desc = "FalkorDB is a graph database. " + "detail " * 60  # > 180 chars
        batch = [
            GraphNode(
                id="acme__t", label="T", properties={"name": "Acme", "description": long_desc}
            ),
        ]
        candidate = GraphNode(
            id="acme_hub__t",
            label="T",
            properties={"name": "Acme", "description": "short existing"},
        )
        decision = json.dumps(
            {
                "groups": [
                    {
                        "members": [1, 2],
                        "target": 2,
                        "canonical": "Acme",
                        "type": "T",
                        "description": "merged",
                    },
                ]
            }
        )
        llm = CapturingLLM(decision)
        resolver = IncrementalResolution(
            llm=llm,
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([candidate]),
        )
        await resolver.resolve(GraphData(nodes=batch), _ctx())
        assert llm.prompts, "the LLM should have been asked to link the pile"
        assert len(long_desc) > 180
        assert long_desc in llm.prompts[0], "full description must be sent, not truncated"

    async def test_overlapping_llm_groups_are_dropped(self):
        """A ref reused across LLM groups must not double-merge unrelated entities
        (Naseem #2) — the second, overlapping group is dropped."""
        batch = [
            GraphNode(id="a__t", label="T", properties={"name": "Aye", "description": "d1"}),
            GraphNode(id="b__t", label="T", properties={"name": "Bee", "description": "d2"}),
        ]
        # refs 1=Aye, 2=Bee (new), 3=Gal Sh (graph candidate). Group 2 reuses ref 1.
        decision = json.dumps(
            {
                "groups": [
                    {
                        "members": [1, 2],
                        "target": "new",
                        "canonical": "AyeBee",
                        "type": "T",
                        "description": "d",
                    },
                    {
                        "members": [1, 3],
                        "target": 3,
                        "canonical": "X",
                        "type": "T",
                        "description": "d",
                    },
                ]
            }
        )
        resolver = IncrementalResolution(
            llm=MockLLM(responses=[decision]),
            embedder=WordEmbedder(),
            candidate_retriever=make_retriever([GAL_SH]),
        )
        res = await resolver.resolve(GraphData(nodes=batch), _ctx())
        # Group 1 applied (Bee → Aye); overlapping group 2 dropped, so nothing
        # links onto the graph candidate.
        assert res.remap.get("b__t") == "a__t"
        assert "gal_sh__person" not in res.remap.values()
        assert "gal_sh__person" not in {n.id for n in res.nodes}

    async def test_cross_type_merge_needs_higher_bar(self):
        """Same-name cross-type auto-merge requires cross_type_threshold, while
        same-type merges at the lower same_name_threshold (Naseem #5)."""
        # Two descriptions at cosine 0.85 — above same_name (0.80), below cross (0.90).
        table = {"desc-A": [1.0, 0.0], "desc-B": [0.85, 0.5268]}

        async def resolve_pair(label_b):
            batch = [
                GraphNode(
                    id="paris__location",
                    label="Location",
                    properties={"name": "Paris", "description": "desc-A"},
                ),
                GraphNode(
                    id="paris__other",
                    label=label_b,
                    properties={"name": "Paris", "description": "desc-B"},
                ),
            ]
            resolver = IncrementalResolution(
                llm=MockLLM(responses=[""]),
                embedder=ScriptedEmbedder(table),
                candidate_retriever=make_retriever([]),
            )
            return await resolver.resolve(GraphData(nodes=batch), _ctx())

        same_type = await resolve_pair("Location")  # 0.85 ≥ 0.80 → merge
        cross_type = await resolve_pair("Person")  # 0.85 < 0.90 → stay separate
        assert len(same_type.nodes) == 1, "same-type should merge at 0.85"
        assert len(cross_type.nodes) == 2, "cross-type needs 0.90, so 0.85 stays split"

    def test_normalize_name_folds_case_and_separators(self):
        assert normalize_name("GraphRAG-SDK") == "graphrag sdk"
        assert normalize_name("graphrag_sdk") == "graphrag sdk"
        assert normalize_name("  GraphRAG   SDK ") == "graphrag sdk"
