"""Tests for the skills library (Phase 3.4)."""
from __future__ import annotations

from typing import Any

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import LLMResponse, SkillResult
from graphrag_sdk.skills import (
    SKILL_REGISTRY,
    ContradictionDetectionSkill,
    EntityComparisonSkill,
    GapAnalysisSkill,
    ImpactAnalysisSkill,
    TimelineReconstructionSkill,
    build_skill,
)


class FakeResult:
    def __init__(self, rows: list[list[Any]]):
        self.result_set = rows


class FakeGraphStore:
    """Routes Cypher queries to canned result sets by substring match."""

    def __init__(self, *, query_map=None, neighbors=None, pagerank=None):
        self._query_map = query_map or {}
        self._neighbors = neighbors or {}
        self._pagerank = pagerank or {}
        self.queries: list[str] = []

    async def query_raw(self, cypher: str, params: dict | None = None):
        self.queries.append(cypher)
        for needle, rows in self._query_map.items():
            if needle in cypher:
                return FakeResult(rows)
        return FakeResult([])

    async def weighted_neighbors(self, node_id: str, *, limit: int = 50):
        return self._neighbors.get(node_id, [])

    async def pagerank(self, **kwargs: Any):
        return self._pagerank


class FakeLLM:
    def __init__(self, content: str):
        self._content = content

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return LLMResponse(content=self._content)


class TestRegistry:
    def test_all_five_skills_registered(self):
        assert set(SKILL_REGISTRY) == {
            "entity_comparison",
            "impact_analysis",
            "contradiction_detection",
            "gap_analysis",
            "timeline_reconstruction",
        }

    def test_build_unknown_skill_raises(self):
        with pytest.raises(KeyError):
            build_skill("nope", FakeGraphStore())

    def test_build_known_skill(self):
        skill = build_skill("gap_analysis", FakeGraphStore())
        assert isinstance(skill, GapAnalysisSkill)


class TestEntityComparison:
    async def test_compares_neighbors_and_attributes(self, ctx: Context):
        store = FakeGraphStore(
            query_map={
                "properties(e)": [[{"name": "Alice", "role": "eng"}]],
            },
            neighbors={
                "alice": [("acme", 1.0, "WORKS_AT"), ("bob", 1.0, "KNOWS")],
                "carol": [("acme", 1.0, "WORKS_AT"), ("dan", 1.0, "KNOWS")],
            },
        )
        skill = EntityComparisonSkill(store)
        result = await skill.run(ctx, entity_a="alice", entity_b="carol")
        assert isinstance(result, SkillResult)
        assert "acme" in result.data["shared_neighbors"]
        assert "bob" in result.data["neighbors_only_a"]
        assert "dan" in result.data["neighbors_only_b"]

    async def test_requires_both_entities(self, ctx: Context):
        skill = EntityComparisonSkill(FakeGraphStore())
        with pytest.raises(ValueError):
            await skill.run(ctx, entity_a="alice")


class TestImpactAnalysis:
    async def test_ranks_impacted_by_distance(self, ctx: Context):
        store = FakeGraphStore(
            neighbors={
                "root": [("near", 1.0, "R")],
                "near": [("far", 1.0, "R")],
            }
        )
        skill = ImpactAnalysisSkill(store)
        result = await skill.run(ctx, entity="root", max_depth=3)
        impacted = {r["entity"]: r["distance"] for r in result.data["impacted"]}
        assert impacted.get("near") == 1
        assert impacted.get("far") == 2

    async def test_requires_entity(self, ctx: Context):
        skill = ImpactAnalysisSkill(FakeGraphStore())
        with pytest.raises(ValueError):
            await skill.run(ctx)


class TestContradictionDetection:
    async def test_parses_llm_contradictions(self, ctx: Context):
        store = FakeGraphStore(
            query_map={
                "-[r]-(m:__Entity__)": [["BORN_IN", "Paris", "born in Paris"],
                                        ["BORN_IN", "London", "born in London"]],
            }
        )
        llm = FakeLLM(
            '{"contradictions": [{"a": "Paris", "b": "London", '
            '"reason": "two birthplaces"}], "summary": "conflict found"}'
        )
        skill = ContradictionDetectionSkill(store, llm)
        result = await skill.run(ctx, entity="person")
        assert result.data["facts_examined"] == 2
        assert result.data["contradictions"][0]["reason"] == "two birthplaces"
        assert result.summary == "conflict found"

    async def test_no_llm_returns_facts_only(self, ctx: Context):
        store = FakeGraphStore(
            query_map={"-[r]-(m:__Entity__)": [["KNOWS", "bob", ""]]}
        )
        skill = ContradictionDetectionSkill(store, None)
        result = await skill.run(ctx, entity="alice")
        assert result.data["contradictions"] == []
        assert result.data["facts_examined"] == 1


class TestGapAnalysis:
    async def test_reports_isolated_and_sparse(self, ctx: Context):
        store = FakeGraphStore(
            query_map={
                "NOT (e)-[]-(:__Entity__)": [["lonely1"], ["lonely2"]],
                "count(*) AS n": [["Rare", 1], ["Common", 50]],
            }
        )
        skill = GapAnalysisSkill(store)
        result = await skill.run(ctx, min_instances=2)
        assert result.data["num_isolated"] == 2
        labels = {s["label"] for s in result.data["sparse_labels"]}
        assert "Rare" in labels
        assert "Common" not in labels


class TestTimelineReconstruction:
    async def test_orders_events_by_date(self, ctx: Context):
        store = FakeGraphStore(
            query_map={
                "properties(e)": [
                    ["e2", {"name": "Second", "date": "2020-05-01"}],
                    ["e1", {"name": "First", "year": "1999"}],
                    ["e3", {"name": "NoDate", "color": "red"}],
                ],
            }
        )
        skill = TimelineReconstructionSkill(store)
        result = await skill.run(ctx)
        order = [e["entity"] for e in result.data["timeline"]]
        assert order == ["e1", "e2"]
        assert result.data["num_events"] == 2
