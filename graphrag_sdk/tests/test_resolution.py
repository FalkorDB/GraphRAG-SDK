"""Tests for ingestion/resolution_strategies/exact_match.py and base.py."""

from __future__ import annotations

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, GraphNode, GraphRelationship, ResolutionResult
from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution
from graphrag_sdk.ingestion.resolution_strategies.base import exact_match_merge

from .conftest import MockLLM


class TestExactMatchResolution:
    async def test_no_duplicates(self, ctx, sample_graph_data):
        """No duplicates — all nodes survive, merged_count = 0."""
        resolver = ExactMatchResolution()
        result = await resolver.resolve(sample_graph_data, ctx)
        assert result.merged_count == 0
        assert len(result.nodes) == 3
        assert len(result.relationships) == 2

    async def test_duplicate_by_name(self, ctx, sample_graph_data_with_duplicates):
        """alice-1 and alice-2 share name='Alice' so should merge."""
        resolver = ExactMatchResolution(resolve_property="name")
        result = await resolver.resolve(sample_graph_data_with_duplicates, ctx)
        assert result.merged_count == 1
        assert len(result.nodes) == 3  # Alice(merged), Bob, Acme

        # The survived Alice should have merged properties
        alice = next(n for n in result.nodes if n.properties.get("name") == "Alice")
        assert "role" in alice.properties or "age" in alice.properties

    async def test_duplicate_relationships_deduped(self, ctx):
        """When two nodes merge, their duplicate rels should collapse."""
        data = GraphData(
            nodes=[
                GraphNode(id="a1", label="X", properties={"name": "A"}),
                GraphNode(id="a2", label="X", properties={"name": "A"}),
                GraphNode(id="b", label="Y", properties={"name": "B"}),
            ],
            relationships=[
                GraphRelationship(start_node_id="a1", end_node_id="b", type="REL"),
                GraphRelationship(start_node_id="a2", end_node_id="b", type="REL"),
            ],
        )
        resolver = ExactMatchResolution(resolve_property="name")
        result = await resolver.resolve(data, ctx)
        assert result.merged_count == 1
        # Both rels point (a->b REL), should deduplicate
        assert len(result.relationships) == 1

    async def test_property_merging(self, ctx):
        """Survivor inherits properties from duplicates."""
        data = GraphData(
            nodes=[
                GraphNode(id="x1", label="T", properties={"name": "X", "color": "red"}),
                GraphNode(id="x2", label="T", properties={"name": "X", "size": "large"}),
            ],
            relationships=[],
        )
        resolver = ExactMatchResolution(resolve_property="name")
        result = await resolver.resolve(data, ctx)
        assert len(result.nodes) == 1
        merged = result.nodes[0]
        assert merged.properties["color"] == "red"
        assert merged.properties["size"] == "large"

    async def test_empty_input(self, ctx):
        data = GraphData()
        resolver = ExactMatchResolution()
        result = await resolver.resolve(data, ctx)
        assert result.merged_count == 0
        assert len(result.nodes) == 0
        assert len(result.relationships) == 0

    async def test_default_resolve_property_is_id(self, ctx):
        """Default resolves by 'id' in properties, falling back to node.id."""
        data = GraphData(
            nodes=[
                GraphNode(id="same", label="T", properties={"x": 1}),
                GraphNode(id="same", label="T", properties={"y": 2}),
            ],
            relationships=[],
        )
        resolver = ExactMatchResolution()  # resolve_property="id"
        result = await resolver.resolve(data, ctx)
        # Both have id="same" → should merge
        assert result.merged_count == 1
        assert len(result.nodes) == 1

    async def test_different_labels_not_merged(self, ctx):
        """Nodes with same name but different labels stay separate."""
        data = GraphData(
            nodes=[
                GraphNode(id="a1", label="Person", properties={"name": "Alice"}),
                GraphNode(id="a2", label="Company", properties={"name": "Alice"}),
            ],
            relationships=[],
        )
        resolver = ExactMatchResolution(resolve_property="name")
        result = await resolver.resolve(data, ctx)
        assert result.merged_count == 0
        assert len(result.nodes) == 2

    async def test_relationship_remapping(self, ctx):
        """Relationships pointing to merged nodes get remapped."""
        data = GraphData(
            nodes=[
                GraphNode(id="a1", label="T", properties={"name": "A"}),
                GraphNode(id="a2", label="T", properties={"name": "A"}),
                GraphNode(id="b", label="T", properties={"name": "B"}),
                GraphNode(id="c", label="T", properties={"name": "C"}),
            ],
            relationships=[
                GraphRelationship(start_node_id="b", end_node_id="a1", type="LINK"),
                GraphRelationship(start_node_id="c", end_node_id="a2", type="LINK"),
            ],
        )
        resolver = ExactMatchResolution(resolve_property="name")
        result = await resolver.resolve(data, ctx)
        # Both rels should now point to the survivor's id
        survivor_id = next(n.id for n in result.nodes if n.properties["name"] == "A")
        for rel in result.relationships:
            if rel.type == "LINK":
                assert rel.end_node_id == survivor_id


class TestCrossLabelMerge:
    """Bug 2: exact_match_merge with cross_label_merge=True groups by name
    only. Same-label duplicates within the group merge as usual; merges
    ACROSS different labels require LLM YES confirmation — without it
    (LLM unavailable, not enough descriptions, or LLM says NO), homograph
    nodes are preserved under their original labels."""

    async def test_cross_label_preserved_without_llm(self):
        """No LLM → fail-safe: cross-label nodes preserved (no silent merge)."""
        nodes = [
            GraphNode(
                id="fdb__tech",
                label="Technology",
                properties={"name": "FalkorDB", "description": "A graph database engine"},
            ),
            GraphNode(
                id="fdb__org",
                label="Organization",
                properties={"name": "FalkorDB", "description": "The company behind FalkorDB"},
            ),
        ]
        deduped, remap, count = await exact_match_merge(
            nodes,
            None,
            cross_label_merge=True,
        )
        assert len(deduped) == 2
        assert count == 0
        assert remap == {}
        labels = {n.label for n in deduped}
        assert labels == {"Technology", "Organization"}

    async def test_cross_label_preserved_below_threshold(self):
        """Below cross_label_min_descriptions -> fail-safe, no merge.

        Two nodes with one description each total 2, under the floor of 3.
        This stage has no vector filter, so a group it accepts costs an LLM
        call and rests entirely on that answer. Pairs below the floor are
        left to the cross-label PASS 2, which screens on cosine first.
        """
        nodes = [
            GraphNode(
                id="fdb__tech",
                label="Technology",
                properties={"name": "FalkorDB", "description": "A graph DB"},
            ),
            GraphNode(
                id="fdb__unk",
                label="Unknown",
                properties={"name": "FalkorDB", "description": "FalkorDB system"},
            ),
        ]
        llm = MockLLM(responses=["YES Technology\nsummary"])
        deduped, remap, count = await exact_match_merge(
            nodes,
            llm,
            cross_label_merge=True,
        )
        assert len(deduped) == 2
        assert count == 0

    async def test_cross_label_min_descriptions_can_lower_the_gate(self):
        """The floor is a knob: lowering it admits a group the default skips."""
        nodes = [
            GraphNode(
                id="fdb__tech",
                label="Technology",
                properties={"name": "FalkorDB", "description": "A graph DB"},
            ),
            GraphNode(
                id="fdb__org",
                label="Organization",
                properties={"name": "FalkorDB", "description": "The company behind it"},
            ),
        ]
        llm = MockLLM(responses=["YES Technology\nsummary"])
        deduped, remap, count = await exact_match_merge(
            nodes,
            llm,
            cross_label_merge=True,
            cross_label_min_descriptions=2,
        )
        assert len(deduped) == 1
        assert count == 1

    async def test_gate_is_independent_of_force_summary_threshold(self):
        """The two knobs answer different questions and must not be coupled.

        force_summary_threshold decides when the LLM *summarises*. Pushing it
        far above the description count must not suppress the merge, which is
        what happened while a single knob served both purposes.
        """
        nodes = [
            GraphNode(
                id="fdb__tech",
                label="Technology",
                properties={"name": "FalkorDB", "description": "A graph DB"},
            ),
            GraphNode(
                id="fdb__org",
                label="Organization",
                properties={"name": "FalkorDB", "description": "The company behind it"},
            ),
            GraphNode(
                id="fdb__prod",
                label="Product",
                properties={"name": "FalkorDB", "description": "A database product"},
            ),
        ]
        llm = MockLLM(responses=["YES Technology\nsummary"])
        deduped, remap, count = await exact_match_merge(
            nodes,
            llm,
            cross_label_merge=True,
            force_summary_threshold=99,
        )
        assert len(deduped) == 1
        assert count == 2

    async def test_cross_label_merge_disabled_by_default(self):
        """Without cross_label_merge=True, same-name cross-label nodes stay separate."""
        nodes = [
            GraphNode(
                id="fdb__tech",
                label="Technology",
                properties={"name": "FalkorDB", "description": "A graph database"},
            ),
            GraphNode(
                id="fdb__org",
                label="Organization",
                properties={"name": "FalkorDB", "description": "The company"},
            ),
        ]
        deduped, remap, count = await exact_match_merge(nodes, None)
        assert len(deduped) == 2
        assert count == 0

    async def test_cross_label_merge_three_labels_with_yes(self):
        """Three labels (3+ descriptions) → LLM says YES → merged under chosen type."""
        nodes = [
            GraphNode(
                id="fdb__tech",
                label="Technology",
                properties={"name": "FalkorDB", "description": "Graph database engine"},
            ),
            GraphNode(
                id="fdb__org",
                label="Organization",
                properties={"name": "FalkorDB", "description": "The company behind it"},
            ),
            GraphNode(
                id="fdb__unk",
                label="Unknown",
                properties={"name": "FalkorDB", "description": "FalkorDB system"},
            ),
        ]
        # LLM returns: 'YES <type>' on line 1, summary on line 2
        llm = MockLLM(responses=["YES Technology\nFalkorDB is a graph database engine."])
        deduped, remap, count = await exact_match_merge(
            nodes,
            llm,
            cross_label_merge=True,
        )
        assert len(deduped) == 1
        assert deduped[0].label == "Technology"
        assert count == 2

    async def test_cross_label_homograph_llm_says_no(self):
        """Nas's Paris case: LLM says NO → both homographs preserved."""
        nodes = [
            GraphNode(
                id="paris__loc",
                label="Location",
                properties={"name": "Paris", "description": "Capital city of France."},
            ),
            GraphNode(
                id="paris__per_1",
                label="Person",
                properties={"name": "Paris", "description": "American media personality."},
            ),
            GraphNode(
                id="paris__per_2",
                label="Person",
                properties={"name": "Paris", "description": "Hilton family member."},
            ),
        ]
        llm = MockLLM(responses=["NO\nThese are distinct real-world entities."])
        deduped, remap, count = await exact_match_merge(
            nodes,
            llm,
            cross_label_merge=True,
        )
        # Same-label Paris/Person duplicates still merge (Phase 1a).
        # Cross-label Paris/Location vs Paris/Person stays separate (fail-safe on NO).
        labels = sorted(n.label for n in deduped)
        assert labels == ["Location", "Person"]
        # The two Person duplicates collapsed into one survivor.
        assert count == 1

    async def test_cross_label_llm_error_fails_safe(self):
        """LLM call raises → fail-safe: preserve all nodes."""

        class FailingLLM(MockLLM):
            def invoke(self, prompt, **kwargs):
                raise RuntimeError("LLM unavailable")

        nodes = [
            GraphNode(
                id="fdb__tech",
                label="Technology",
                properties={"name": "FalkorDB", "description": "Graph database engine"},
            ),
            GraphNode(
                id="fdb__org",
                label="Organization",
                properties={"name": "FalkorDB", "description": "The company behind it"},
            ),
            GraphNode(
                id="fdb__unk",
                label="Unknown",
                properties={"name": "FalkorDB", "description": "FalkorDB system"},
            ),
        ]
        llm = FailingLLM(responses=[])
        deduped, remap, count = await exact_match_merge(
            nodes,
            llm,
            cross_label_merge=True,
        )
        assert len(deduped) == 3
        assert count == 0

    async def test_cross_label_all_in_one_group(self):
        """All same-name nodes land in one group regardless of label; LLM YES merges."""
        nodes = [
            GraphNode(
                id="fdb__tech_1",
                label="Technology",
                properties={"name": "FalkorDB", "description": "Graph DB v1"},
            ),
            GraphNode(
                id="fdb__tech_2",
                label="Technology",
                properties={"name": "FalkorDB", "description": "Graph DB v2"},
            ),
            GraphNode(
                id="fdb__org",
                label="Organization",
                properties={"name": "FalkorDB", "description": "The company"},
            ),
        ]
        llm = MockLLM(responses=["YES Technology\nFalkorDB is a graph database."])
        deduped, remap, count = await exact_match_merge(
            nodes,
            llm,
            cross_label_merge=True,
        )
        assert len(deduped) == 1
        assert count == 2
        survivor_id = deduped[0].id
        for node in nodes:
            if node.id != survivor_id:
                assert remap[node.id] == survivor_id

    async def test_cross_label_same_label_unaffected(self):
        """Same-label groups still merge normally with cross_label_merge=True."""
        nodes = [
            GraphNode(
                id="a1", label="Person", properties={"name": "Alice", "description": "Engineer"}
            ),
            GraphNode(
                id="a2", label="Person", properties={"name": "Alice", "description": "Developer"}
            ),
            GraphNode(
                id="b1", label="Person", properties={"name": "Bob", "description": "Manager"}
            ),
        ]
        deduped, remap, count = await exact_match_merge(
            nodes,
            None,
            cross_label_merge=True,
        )
        assert len(deduped) == 2  # Alice (merged) + Bob
        assert count == 1
        names = {n.properties["name"] for n in deduped}
        assert names == {"Alice", "Bob"}


class TestStage1PreservesAbsorbedLabels:
    """Stage 1's cross-label merge must not destroy the losers' labels.

    P3.29/P3.30 fixed this in the embedding stage and in PASS 2, but Stage 1
    has its own cross-label merge: it picks one survivor label and copies only
    the keys the survivor lacks, so a duplicate's differing label was dropped
    with no record. ``FalkorDB``/Organization absorbed into
    ``FalkorDB``/Technology left nothing saying the company existed, and
    ``MATCH (n:Organization)`` could never find it.
    """

    @staticmethod
    def _nodes():
        return [
            GraphNode(
                id="fdb__tech",
                label="Technology",
                properties={"name": "FalkorDB", "description": "A graph DB"},
            ),
            GraphNode(
                id="fdb__org",
                label="Organization",
                properties={"name": "FalkorDB", "description": "The company behind it"},
            ),
        ]

    async def test_absorbed_label_is_recorded(self):
        llm = MockLLM(responses=["YES Technology\nsummary"])
        deduped, _remap, count = await exact_match_merge(
            self._nodes(), llm, cross_label_merge=True, cross_label_min_descriptions=2
        )
        assert len(deduped) == 1 and count == 1
        assert deduped[0].label == "Technology"
        assert deduped[0].properties.get("merged_labels") == "Organization"

    async def test_same_label_merge_records_nothing(self):
        """Control: no label was lost, so no note should be written."""
        nodes = [
            GraphNode(
                id="a",
                label="Technology",
                properties={"name": "FalkorDB", "description": "one"},
            ),
            GraphNode(
                id="b",
                label="Technology",
                properties={"name": "FalkorDB", "description": "two"},
            ),
        ]
        deduped, _remap, _count = await exact_match_merge(
            nodes, MockLLM(responses=["ignored"]), cross_label_merge=True
        )
        assert len(deduped) == 1
        assert "merged_labels" not in deduped[0].properties

    async def test_several_absorbed_labels_are_all_recorded(self):
        nodes = self._nodes() + [
            GraphNode(
                id="fdb__prod",
                label="Product",
                properties={"name": "FalkorDB", "description": "A database product"},
            )
        ]
        llm = MockLLM(responses=["YES Technology\nsummary"])
        deduped, _remap, _count = await exact_match_merge(
            nodes, llm, cross_label_merge=True, cross_label_min_descriptions=2
        )
        assert len(deduped) == 1
        recorded = deduped[0].properties.get("merged_labels", "")
        assert "Organization" in recorded and "Product" in recorded
        assert "Technology" not in recorded, "the survivor's own label is not absorbed"


class TestExactMatchMatchesStage1:
    """``ExactMatchResolution`` must behave exactly like Phase 1 of
    ``LLMVerifiedResolution`` (``exact_match_merge``).

    It is not the default resolver, but a user who selects it explicitly is
    choosing "merge exact duplicates" -- and got a materially different and
    worse merge than the identically-named phase inside the default strategy.
    It grouped on the raw ``id`` property instead of the normalised name, and
    -- the damaging part -- it merged with "copy only keys the survivor
    lacks", so every duplicate's ``description`` and ``source_chunk_ids``
    were silently discarded. Same defect shape as P3.21/P3.29/P3.33.
    """

    async def test_grouping_is_on_normalised_name(self, ctx):
        """Case and surrounding whitespace must not defeat the match."""
        data = GraphData(
            nodes=[
                GraphNode(id="a1", label="Person", properties={"name": "Ada Lovelace"}),
                GraphNode(id="a2", label="Person", properties={"name": "  ada lovelace "}),
            ],
            relationships=[],
        )
        result = await ExactMatchResolution().resolve(data, ctx)
        assert result.merged_count == 1
        assert len(result.nodes) == 1

    async def test_descriptions_are_joined_not_discarded(self, ctx):
        """The loser's description must survive, joined with ' | '."""
        data = GraphData(
            nodes=[
                GraphNode(
                    id="a1",
                    label="Person",
                    properties={"name": "Ada", "description": "British mathematician"},
                ),
                GraphNode(
                    id="a2",
                    label="Person",
                    properties={"name": "Ada", "description": "wrote the first algorithm"},
                ),
            ],
            relationships=[],
        )
        result = await ExactMatchResolution().resolve(data, ctx)
        assert len(result.nodes) == 1
        desc = result.nodes[0].properties["description"]
        assert desc == "British mathematician | wrote the first algorithm"

    async def test_source_chunk_ids_are_unioned(self, ctx):
        """Provenance from every duplicate must be retained."""
        data = GraphData(
            nodes=[
                GraphNode(
                    id="a1",
                    label="Person",
                    properties={"name": "Ada", "source_chunk_ids": ["c1"]},
                ),
                GraphNode(
                    id="a2",
                    label="Person",
                    properties={"name": "Ada", "source_chunk_ids": ["c2"]},
                ),
            ],
            relationships=[],
        )
        result = await ExactMatchResolution().resolve(data, ctx)
        assert len(result.nodes) == 1
        assert sorted(result.nodes[0].properties["source_chunk_ids"]) == ["c1", "c2"]

    async def test_cross_label_merge_when_llm_supplied(self, ctx):
        """Same name under different labels gets an LLM verdict, as in Stage 1.

        Three descriptions total, to clear ``cross_label_min_descriptions``.
        """
        data = GraphData(
            nodes=[
                GraphNode(
                    id="f1",
                    label="Technology",
                    properties={"name": "FalkorDB", "description": "A graph database"},
                ),
                GraphNode(
                    id="f3",
                    label="Technology",
                    properties={"name": "FalkorDB", "description": "Redis-based graph engine"},
                ),
                GraphNode(
                    id="f2",
                    label="Organization",
                    properties={"name": "FalkorDB", "description": "The company that builds it"},
                ),
            ],
            relationships=[],
        )
        llm = MockLLM(responses=["YES Technology\nGraph database and its vendor"])
        result = await ExactMatchResolution(llm=llm).resolve(data, ctx)
        assert len(result.nodes) == 1
        assert result.merged_count == 2
        assert "Organization" in result.nodes[0].properties.get("merged_labels", "")

    async def test_llm_summarises_at_threshold(self, ctx):
        """Three or more descriptions are summarised, not concatenated."""
        data = GraphData(
            nodes=[
                GraphNode(id=f"a{i}", label="Person",
                          properties={"name": "Ada", "description": f"desc {i}"})
                for i in range(3)
            ],
            relationships=[],
        )
        llm = MockLLM(responses=["A concise summary"])
        result = await ExactMatchResolution(llm=llm).resolve(data, ctx)
        assert len(result.nodes) == 1
        assert result.nodes[0].properties["description"] == "A concise summary"

    async def test_identical_to_stage_1_on_the_same_input(self, ctx):
        """Equivalence check: same nodes through both paths, same outcome."""
        def build() -> list[GraphNode]:
            return [
                GraphNode(id="a1", label="Person",
                          properties={"name": "Ada", "description": "d1",
                                      "source_chunk_ids": ["c1"]}),
                GraphNode(id="a2", label="Person",
                          properties={"name": "ADA ", "description": "d2",
                                      "source_chunk_ids": ["c2"]}),
                GraphNode(id="b1", label="Person",
                          properties={"name": "Bob", "description": "d3"}),
            ]

        stage1_nodes, stage1_remap, stage1_count = await exact_match_merge(build(), None)
        result = await ExactMatchResolution().resolve(GraphData(nodes=build()), ctx)

        assert result.merged_count == stage1_count
        assert result.remap == stage1_remap
        assert [n.id for n in result.nodes] == [n.id for n in stage1_nodes]
        assert [n.properties.get("description") for n in result.nodes] == [
            n.properties.get("description") for n in stage1_nodes
        ]

    # ── controls: behaviour that must NOT change ──
    async def test_control_homographs_preserved_without_llm(self, ctx):
        """No LLM → cross-label nodes still must not merge."""
        data = GraphData(
            nodes=[
                GraphNode(id="p1", label="Person", properties={"name": "Paris"}),
                GraphNode(id="l1", label="Location", properties={"name": "Paris"}),
            ],
            relationships=[],
        )
        result = await ExactMatchResolution().resolve(data, ctx)
        assert result.merged_count == 0
        assert len(result.nodes) == 2

    async def test_control_relationships_still_remapped(self, ctx):
        data = GraphData(
            nodes=[
                GraphNode(id="a1", label="T", properties={"name": "A"}),
                GraphNode(id="a2", label="T", properties={"name": "A"}),
                GraphNode(id="b", label="T", properties={"name": "B"}),
            ],
            relationships=[
                GraphRelationship(start_node_id="b", end_node_id="a1", type="LINK"),
                GraphRelationship(start_node_id="b", end_node_id="a2", type="LINK"),
            ],
        )
        result = await ExactMatchResolution().resolve(data, ctx)
        assert len(result.relationships) == 1
        assert result.relationships[0].end_node_id == "a1"


class TestCrossLabelGroupingUsesTheSameKey:
    """The cross-label candidate scan must key nodes exactly as the
    same-label pass does.

    It used to read ``properties.get("name", "")`` while the same-label
    grouping read ``properties.get("name", node.id)``. Any node without a
    name therefore normalised to the empty string and every one of them
    landed in a single bucket -- so unrelated nameless nodes carrying
    different labels looked like a same-name homograph group and were sent
    to the LLM to be merged together.
    """

    async def test_nameless_nodes_are_not_bucketed_together(self):
        nodes = [
            GraphNode(id="alpha", label="Person",
                      properties={"description": "some person"}),
            GraphNode(id="beta", label="Location",
                      properties={"description": "somewhere else"}),
            GraphNode(id="gamma", label="Product",
                      properties={"description": "a third thing"}),
        ]
        llm = MockLLM(responses=["YES Person\nmerged summary"])
        deduped, remap, count = await exact_match_merge(
            nodes,
            llm,
            cross_label_merge=True,
        )
        assert count == 0, "unrelated nameless nodes must not merge"
        assert remap == {}
        assert len(deduped) == 3

    async def test_control_named_homographs_still_group(self):
        """Control: real same-name cross-label groups are still detected."""
        nodes = [
            GraphNode(id="a", label="Person",
                      properties={"name": "Ada", "description": "d1"}),
            GraphNode(id="b", label="Person",
                      properties={"name": "Ada", "description": "d2"}),
            GraphNode(id="c", label="Scientist",
                      properties={"name": "Ada", "description": "d3"}),
        ]
        llm = MockLLM(responses=["YES Person\nmerged summary"])
        deduped, _remap, count = await exact_match_merge(
            nodes,
            llm,
            cross_label_merge=True,
        )
        assert count == 2
        assert len(deduped) == 1
