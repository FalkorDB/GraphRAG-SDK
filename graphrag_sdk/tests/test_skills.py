"""Tests for the skills library (Phase 3.4)."""

from __future__ import annotations

from typing import Any

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import LLMResponse, SkillResult
from graphrag_sdk.retrieval.agentic import ToolContext, ToolRegistry, make_skill_tool
from graphrag_sdk.skills import (
    SKILL_REGISTRY,
    ContradictionDetectionSkill,
    EntityComparisonSkill,
    GapAnalysisSkill,
    ImpactAnalysisSkill,
    Skill,
    TimelineReconstructionSkill,
    build_skill,
    register_skill,
    skill_parameters,
)
from graphrag_sdk.skills.timeline_reconstruction import _extract_date, _is_date_key


class FakeResult:
    def __init__(self, rows: list[list[Any]]):
        self.result_set = rows


class FakeGraphStore:
    """Routes Cypher queries to canned result sets by substring match.

    Entity references resolve to themselves (``e.id = $ref``) when listed in
    ``entities``.
    """

    def __init__(self, *, query_map=None, neighbors=None, pagerank=None, entities=None):
        self._query_map = query_map or {}
        self._neighbors = neighbors or {}
        self._pagerank = pagerank or {}
        self._entities = entities or {}
        self.queries: list[tuple[str, dict]] = []

    async def query_raw(self, cypher: str, params: dict | None = None, **kwargs: Any):
        self.queries.append((cypher, params or {}))
        if "e.id = $ref OR e.name = $ref" in cypher:
            ref = (params or {}).get("ref")
            ids = [eid for eid, name in self._entities.items() if ref in (eid, name)]
            return FakeResult([[eid, self._entities[eid]] for eid in ids])
        if "toLower($ref)" in cypher:
            ref = str((params or {}).get("ref", "")).lower()
            ids = [eid for eid, name in self._entities.items() if name.lower() == ref]
            return FakeResult([[eid, self._entities[eid]] for eid in ids])
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
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(content=self._content)


PEOPLE = {"alice": "Alice", "carol": "Carol", "root": "Root", "person": "Person X"}


class TestRegistry:
    def test_all_five_skills_registered(self):
        assert {
            "entity_comparison",
            "impact_analysis",
            "contradiction_detection",
            "gap_analysis",
            "timeline_reconstruction",
        } <= set(SKILL_REGISTRY)

    def test_build_unknown_skill_raises(self):
        with pytest.raises(KeyError):
            build_skill("nope", FakeGraphStore())

    def test_build_known_skill(self):
        skill = build_skill("gap_analysis", FakeGraphStore())
        assert isinstance(skill, GapAnalysisSkill)

    def test_every_builtin_has_an_object_schema(self):
        for cls in SKILL_REGISTRY.values():
            assert cls.parameters["type"] == "object"
            assert cls.parameters.get("additionalProperties") is False

    def test_register_custom_skill(self):
        class OwnersSkill(Skill):
            name = "owners_test"
            description = "List owners."
            parameters = skill_parameters({"entity": {"type": "string"}}, ["entity"])

            async def run(self, ctx=None, **params):
                return SkillResult(skill=self.name, summary="owners: none")

        try:
            assert register_skill(OwnersSkill) is OwnersSkill
            assert build_skill("owners_test", FakeGraphStore()).name == "owners_test"
            with pytest.raises(ValueError, match="already registered"):

                class Other(OwnersSkill):
                    pass

                register_skill(Other)
            register_skill(Other, replace=True)
        finally:
            SKILL_REGISTRY.pop("owners_test", None)

    @pytest.mark.parametrize("name", ["skill", "has space", ""])
    def test_register_rejects_bad_names(self, name):
        cls = type("Bad", (GapAnalysisSkill,), {"name": name})
        with pytest.raises(ValueError):
            register_skill(cls)

    def test_register_rejects_non_skills(self):
        with pytest.raises(TypeError):
            register_skill(object)  # type: ignore[arg-type]


class TestEntityResolution:
    async def test_names_resolve_to_ids(self, ctx: Context):
        store = FakeGraphStore(entities={"alice_person": "Alice"})
        assert await GapAnalysisSkill(store)._resolve_entity("Alice") == "alice_person"
        assert await GapAnalysisSkill(store)._resolve_entity("ALICE") == "alice_person"

    async def test_unknown_and_ambiguous(self):
        store = FakeGraphStore(entities={"a1": "Apex", "a2": "Apex"})
        with pytest.raises(ValueError, match="several entities"):
            await GapAnalysisSkill(store)._resolve_entity("Apex")
        with pytest.raises(ValueError, match="no entity is called"):
            await GapAnalysisSkill(store)._resolve_entity("Zed")


class TestEntityComparison:
    async def test_compares_neighbors_and_attributes(self, ctx: Context):
        store = FakeGraphStore(
            entities=PEOPLE,
            query_map={
                "properties(e)": [
                    [{"name": "Alice", "role": "eng", "embedding": [0.1] * 8, "id": "x"}]
                ],
            },
            neighbors={
                "alice": [("acme", 1.0, "WORKS_AT"), ("bob", 1.0, "KNOWS")],
                "carol": [("acme", 1.0, "WORKS_AT"), ("dan", 1.0, "KNOWS")],
            },
        )
        skill = EntityComparisonSkill(store)
        result = await skill.run(ctx, entity_a="Alice", entity_b="carol")
        assert isinstance(result, SkillResult)
        assert "acme" in result.data["shared_neighbors"]
        assert "bob" in result.data["neighbors_only_a"]
        assert "dan" in result.data["neighbors_only_b"]

    async def test_embeddings_never_compared(self, ctx: Context):
        calls = {"n": 0}

        class Store(FakeGraphStore):
            async def query_raw(self, cypher, params=None, **kw):
                if "properties(e)" in cypher:
                    calls["n"] += 1
                    vec = [float(calls["n"])] * 4
                    return FakeResult(
                        [[{"name": f"E{calls['n']}", "embedding": vec, "name_embedding": vec}]]
                    )
                return await super().query_raw(cypher, params, **kw)

        llm = FakeLLM("comparison")
        skill = EntityComparisonSkill(Store(entities=PEOPLE), llm)
        result = await skill.run(ctx, entity_a="alice", entity_b="carol")
        assert set(result.data["differing_attributes"]) == {"name"}
        assert "embedding" not in llm.prompts[0]

    async def test_requires_both_entities(self, ctx: Context):
        skill = EntityComparisonSkill(FakeGraphStore())
        with pytest.raises(ValueError):
            await skill.run(ctx, entity_a="alice")


class TestImpactAnalysis:
    async def test_ranks_impacted_by_distance(self, ctx: Context):
        store = FakeGraphStore(
            entities=PEOPLE,
            neighbors={
                "root": [("near", 1.0, "R")],
                "near": [("far", 1.0, "R")],
            },
        )
        result = await ImpactAnalysisSkill(store).run(ctx, entity="Root", max_depth=3)
        impacted = {r["entity"]: r["distance"] for r in result.data["impacted"]}
        assert impacted.get("near") == 1
        assert impacted.get("far") == 2

    async def test_bounds_are_clamped(self, ctx: Context, monkeypatch):
        seen: dict[str, int] = {}

        class Walk:
            def __init__(self, fn, *, node_weights, beam_width, max_depth):
                seen.update(beam_width=beam_width, max_depth=max_depth)

            async def beam_search(self, start, ctx=None):
                return []

        monkeypatch.setattr("graphrag_sdk.skills.impact_analysis.DynamicGraphWalk", Walk)
        await ImpactAnalysisSkill(FakeGraphStore(entities=PEOPLE)).run(
            ctx, entity="root", max_depth=99, beam_width=10_000
        )
        assert seen == {"beam_width": 20, "max_depth": 5}

    async def test_bad_int_is_reported(self, ctx: Context):
        with pytest.raises(ValueError, match="'max_depth' must be an integer"):
            await ImpactAnalysisSkill(FakeGraphStore(entities=PEOPLE)).run(
                ctx, entity="root", max_depth="deep"
            )

    async def test_requires_entity(self, ctx: Context):
        with pytest.raises(ValueError):
            await ImpactAnalysisSkill(FakeGraphStore()).run(ctx)


class TestContradictionDetection:
    async def test_parses_llm_contradictions(self, ctx: Context):
        store = FakeGraphStore(
            entities=PEOPLE,
            query_map={
                "-[r:RELATES]-(m:__Entity__)": [
                    ["BORN_IN", "Paris", "born in Paris"],
                    ["BORN_IN", "London", "born in London"],
                ],
            },
        )
        llm = FakeLLM(
            '{"contradictions": [{"a": "Paris", "b": "London", '
            '"reason": "two birthplaces"}], "summary": "conflict found"}'
        )
        result = await ContradictionDetectionSkill(store, llm).run(ctx, entity="person")
        assert result.data["facts_examined"] == 2
        assert result.data["contradictions"][0]["reason"] == "two birthplaces"
        assert result.summary == "conflict found"

    async def test_no_llm_returns_facts_only(self, ctx: Context):
        store = FakeGraphStore(
            entities=PEOPLE, query_map={"-[r:RELATES]-(m:__Entity__)": [["KNOWS", "bob", ""]]}
        )
        result = await ContradictionDetectionSkill(store, None).run(ctx, entity="alice")
        assert result.data["contradictions"] == []
        assert result.data["facts_examined"] == 1

    async def test_limit_is_a_parameter_and_capped(self, ctx: Context):
        store = FakeGraphStore(entities=PEOPLE)
        await ContradictionDetectionSkill(store).run(ctx, entity="alice", limit=10_000)
        cypher, params = store.queries[-1]
        assert params["limit"] == 200 and "LIMIT $limit" in cypher
        with pytest.raises(ValueError, match=">= 1"):
            await ContradictionDetectionSkill(store).run(ctx, entity="alice", limit=0)


class TestGapAnalysis:
    async def test_reports_isolated_and_sparse(self, ctx: Context):
        store = FakeGraphStore(
            query_map={
                "NOT (e)-[:RELATES]-(:__Entity__)": [["lonely1"], ["lonely2"]],
                "count(*) AS n": [["Rare", 1], ["Common", 50]],
            }
        )
        result = await GapAnalysisSkill(store).run(ctx, min_instances=2)
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
        result = await TimelineReconstructionSkill(store).run(ctx)
        order = [e["entity"] for e in result.data["timeline"]]
        assert order == ["e1", "e2"]
        assert result.data["timeline"][0]["name"] == "First"
        assert result.data["num_events"] == 2

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("date", True),
            ("start_date", True),
            ("foundedYear", True),
            ("born_on", True),
            ("friend", False),
            ("legend", False),
            ("created_at", False),
            ("description", False),
            ("on", False),
        ],
    )
    def test_date_keys_match_whole_words(self, key, expected):
        assert _is_date_key(key) is expected

    def test_extract_date_ignores_non_years_and_bad_dates(self):
        assert _extract_date({"date": "unit 12345"}) is None
        assert _extract_date({"date": "2020-13-40"}) is None
        assert _extract_date({"year": "founded in 1887 by"})[0] == (1887, 0, 0)

    async def test_limit_applies_after_filtering(self, ctx: Context):
        rows = [[f"x{i}", {"name": f"N{i}"}] for i in range(10)] + [["d", {"date": "2001"}]]
        store = FakeGraphStore(query_map={"properties(e)": rows})
        result = await TimelineReconstructionSkill(store).run(ctx, limit=1)
        assert [e["entity"] for e in result.data["timeline"]] == ["d"]
        assert store.queries[-1][1]["scan"] == 5


class TestSkillAsTool:
    async def test_findings_and_refusals(self, ctx: Context):
        store = FakeGraphStore(entities=PEOPLE, neighbors={"root": [("near", 1.0, "R")]})
        registry = ToolRegistry()
        registry.register(make_skill_tool(ImpactAnalysisSkill(store)))
        spec = registry.specs()[0]
        assert spec.parameters["required"] == ["entity"]

        ok = await registry.run("impact_analysis", {"entity": "Root"}, ToolContext(ctx=ctx))
        assert ok.status == "ok" and 'Findings (impact_analysis): {"entity": "root"' in ok.content

        missing = await registry.run("impact_analysis", {"entity": "Nobody"}, ToolContext(ctx=ctx))
        assert missing.status == "refused" and "no entity is called 'Nobody'" in missing.content

        extra = await registry.run(
            "impact_analysis", {"entity": "Root", "colour": "red"}, ToolContext(ctx=ctx)
        )
        assert extra.status == "refused" and "unknown argument" in extra.content
