"""Tests for the skills library."""

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
    Skill,
    TimelineReconstructionSkill,
    build_skill,
    register_skill,
    skill_parameters,
)
from graphrag_sdk.skills.timeline_reconstruction import _extract_date, _is_date_key
from graphrag_sdk.tools import GraphRAGToolkit, ToolError


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

    async def test_flags_capped_neighbour_sets(self, ctx: Context):
        from graphrag_sdk.skills.entity_comparison import MAX_NEIGHBOURS

        limits: list[int] = []

        class Store(FakeGraphStore):
            async def weighted_neighbors(self, node_id, *, limit=50):
                limits.append(limit)
                count = MAX_NEIGHBOURS + 30 if node_id == "alice" else 3
                return [(f"{node_id}{i}", 1.0, "R") for i in range(count)][:limit]

        result = await EntityComparisonSkill(Store(entities=PEOPLE)).run(
            ctx, entity_a="alice", entity_b="carol"
        )
        assert limits == [MAX_NEIGHBOURS + 1, MAX_NEIGHBOURS + 1]
        assert len(result.data["neighbors_only_a"]) == MAX_NEIGHBOURS
        assert result.data["neighbors_truncated"] == {"entity_a": True, "entity_b": False}
        assert result.data["truncated"] is True

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

    async def test_bounds_are_clamped(self, ctx: Context):
        # A 10-hop chain: max_depth=99 is clamped to 5 hops.
        chain = {f"c{i}": [(f"c{i + 1}", 1.0, "R")] for i in range(10)}
        chain["root"] = [("c0", 1.0, "R")]
        result = await ImpactAnalysisSkill(FakeGraphStore(entities=PEOPLE, neighbors=chain)).run(
            ctx, entity="root", max_depth=99
        )
        assert max(r["distance"] for r in result.data["impacted"]) == 5
        # 30 direct neighbours, each with a child: beam_width=10_000 is
        # clamped to 20, so 20 of them are expanded to the second hop.
        wide = {"root": [(f"n{i:02d}", 1.0, "R") for i in range(30)]}
        wide.update({f"n{i:02d}": [(f"n{i:02d}_x", 1.0, "R")] for i in range(30)})
        result = await ImpactAnalysisSkill(FakeGraphStore(entities=PEOPLE, neighbors=wide)).run(
            ctx, entity="root", max_depth=2, beam_width=10_000
        )
        by_hop = [r["distance"] for r in result.data["impacted"]]
        assert by_hop.count(1) == 30 and by_hop.count(2) == 20

    async def test_every_direct_neighbour_is_reported(self, ctx: Context):
        wide = {"root": [(f"n{i:02d}", 1.0, "R") for i in range(30)]}
        wide.update({f"n{i:02d}": [(f"n{i:02d}_x", 1.0, "R")] for i in range(30)})
        result = await ImpactAnalysisSkill(FakeGraphStore(entities=PEOPLE, neighbors=wide)).run(
            ctx, entity="root", max_depth=2, beam_width=8
        )
        ranked = result.data["impacted"]
        assert [r["distance"] for r in ranked].count(1) == 30
        assert [r["distance"] for r in ranked].count(2) == 8
        # Closest first: every direct neighbour comes before any 2-hop entity.
        assert [r["distance"] for r in ranked] == sorted(r["distance"] for r in ranked)

    async def test_distance_score_and_via_come_from_one_path(self, ctx: Context):
        store = FakeGraphStore(
            entities=PEOPLE,
            neighbors={"root": [("a", 1.0, "R"), ("b", 0.1, "R")], "a": [("b", 1.0, "R")]},
        )
        result = await ImpactAnalysisSkill(store).run(ctx, entity="root", max_depth=3)
        b = next(r for r in result.data["impacted"] if r["entity"] == "b")
        assert b["distance"] == 1 and b["via"] == ["root", "b"]

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

    @pytest.mark.parametrize("reply", ["ok", "", '{"summary": "fine"}', '{"contradictions": 3}'])
    async def test_unparsed_reply_is_not_reported_as_checked(self, ctx: Context, reply):
        store = FakeGraphStore(
            entities=PEOPLE, query_map={"-[r:RELATES]-(m:__Entity__)": [["KNOWS", "bob", "x"]]}
        )
        result = await ContradictionDetectionSkill(store, FakeLLM(reply)).run(ctx, entity="alice")
        assert result.data["checked_by_llm"] is False
        assert "not the requested JSON" in result.data["llm_error"]
        assert result.data["facts"]

    async def test_failed_llm_call_is_not_reported_as_checked(self, ctx: Context):
        class BrokenLLM:
            async def ainvoke(self, prompt: str, **kwargs: Any):
                raise PermissionError("Incorrect API key provided: sk-secret123")

        store = FakeGraphStore(
            entities=PEOPLE, query_map={"-[r:RELATES]-(m:__Entity__)": [["KNOWS", "bob", "x"]]}
        )
        result = await ContradictionDetectionSkill(store, BrokenLLM()).run(ctx, entity="alice")
        assert result.data["checked_by_llm"] is False
        assert "PermissionError" in result.data["llm_error"]
        assert "sk-secret" not in result.data["llm_error"]

    async def test_facts_sent_to_the_llm_use_names(self, ctx: Context):
        store = FakeGraphStore(
            entities=PEOPLE,
            query_map={
                "-[r:RELATES]-(m:__Entity__)": [
                    ["BORN_IN", "city-17", "born there", "Paris", "Alice"],
                    ["KNOWS", "p-9", "", None, "Alice"],
                ]
            },
        )
        llm = FakeLLM('{"contradictions": [], "summary": ""}')
        result = await ContradictionDetectionSkill(store, llm).run(ctx, entity="alice")
        prompt = llm.prompts[0]
        assert "entity 'Alice'" in prompt and "- BORN_IN Paris: born there" in prompt
        assert "- KNOWS p-9" in prompt  # no name: the id stands in
        assert result.data["facts"][0]["other"] == "city-17"
        assert result.data["checked_by_llm"] is True and "llm_error" not in result.data

    async def test_no_llm_returns_facts_only(self, ctx: Context):
        store = FakeGraphStore(
            entities=PEOPLE, query_map={"-[r:RELATES]-(m:__Entity__)": [["KNOWS", "bob", ""]]}
        )
        result = await ContradictionDetectionSkill(store, None).run(ctx, entity="alice")
        assert result.data["contradictions"] == []
        assert result.data["facts_examined"] == 1
        assert result.data["truncated"] is False

    async def test_flags_facts_past_the_limit(self, ctx: Context):
        rows = [["KNOWS", f"p{i}", ""] for i in range(4)]
        store = FakeGraphStore(entities=PEOPLE, query_map={"-[r:RELATES]-(m:__Entity__)": rows})
        result = await ContradictionDetectionSkill(store, None).run(ctx, entity="alice", limit=3)
        assert result.data["facts_examined"] == 3 and result.data["truncated"] is True

    async def test_limit_is_a_parameter_and_capped(self, ctx: Context):
        store = FakeGraphStore(entities=PEOPLE)
        await ContradictionDetectionSkill(store).run(ctx, entity="alice", limit=10_000)
        cypher, params = store.queries[-1]
        # One row past the cap tells a full list from a cut one.
        assert params["limit"] == 201 and "LIMIT $limit" in cypher
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
        assert result.data["truncated"] is False

    async def test_reports_the_real_isolated_total_when_the_list_is_capped(self, ctx: Context):
        store = FakeGraphStore(
            query_map={
                "RETURN count(e) AS total": [[1234]],
                "NOT (e)-[:RELATES]-(:__Entity__)": [[f"lonely{i}"] for i in range(3)],
            }
        )
        result = await GapAnalysisSkill(store).run(ctx, limit=3)
        assert result.data["isolated_entities"] == ["lonely0", "lonely1", "lonely2"]
        assert result.data["num_isolated"] == 1234
        assert result.data["truncated"] is True


class TestTimelineReconstruction:
    async def test_orders_events_by_date(self, ctx: Context):
        store = FakeGraphStore(
            query_map={
                "AS dated": [
                    ["e2", "Second", [["date", "2020-05-01"]]],
                    ["e1", "First", [["year", "1999"]]],
                    # The query's name prefilter is broader than _is_date_key.
                    ["e3", "NoDate", [["friend", "red"]]],
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

    @pytest.mark.parametrize(
        ("props", "expected"),
        [
            ({"start_headcount": 1500}, None),
            ({"start_headcount": "1500"}, None),
            ({"end_balance": 2000.0}, None),
            ({"founded": 1887}, "1887"),
            ({"start_year": 1999}, "1999"),
            ({"birth_date": "1950"}, "1950"),
            ({"start_headcount": "2020-05-01"}, "2020-05-01"),
            ({"started": True}, None),
        ],
    )
    def test_numbers_are_years_only_under_a_year_like_name(self, props, expected):
        found = _extract_date(props)
        assert (found[1] if found else None) == expected

    def test_extract_date_ignores_non_years_and_bad_dates(self):
        assert _extract_date({"date": "unit 12345"}) is None
        assert _extract_date({"date": "2020-13-40"}) is None
        assert _extract_date({"year": "founded in 1887 by"})[0] == (1887, 0, 0)

    async def test_reports_truncated_when_the_scan_is_full(self, ctx: Context):
        rows = [[f"e{i}", f"E{i}", [["year", str(2000 + i)]]] for i in range(5)]
        store = FakeGraphStore(query_map={"AS dated": rows})
        result = await TimelineReconstructionSkill(store).run(ctx, limit=1)
        # Five candidates is the whole scan for limit=1: entities past it were
        # never read, so earlier events may exist.
        assert result.data["truncated"] is True
        assert result.data["timeline"][0]["entity"] == "e0"

    async def test_not_truncated_when_every_dated_entity_was_read(self, ctx: Context):
        store = FakeGraphStore(query_map={"AS dated": [["d", "D", [["date", "2001"]]]]})
        result = await TimelineReconstructionSkill(store).run(ctx, limit=10)
        assert result.data["truncated"] is False

    async def test_limit_applies_after_filtering(self, ctx: Context):
        store = FakeGraphStore(query_map={"AS dated": [["d", "D", [["date", "2001"]]]]})
        result = await TimelineReconstructionSkill(store).run(ctx, limit=1)
        assert [e["entity"] for e in result.data["timeline"]] == ["d"]
        cypher, params = store.queries[-1]
        assert params["scan"] == 5
        # Undated entities are filtered out in the query, before its LIMIT, so
        # they cannot use up the scan; embeddings are never read.
        assert cypher.index("WHERE size(dated) > 0") < cypher.index("LIMIT $scan")
        assert "properties(e)" not in cypher and "'embedding'" in cypher
        # "on" as a word of its own (published_on_day, onHold) passes the
        # prefilter; the letters around it are checked, not a suffix only.
        assert "toLower(substring(k, i, 2)) = 'on'" in cypher
        assert params["lower"] == list("abcdefghijklmnopqrstuvwxyz")


class _SkillRAG:
    """The parts of GraphRAG the toolkit's run_skill touches."""

    def __init__(self, store: FakeGraphStore):
        self._graph_store = store
        self.llm = None


class TestSkillAsTool:
    async def test_findings_and_refusals(self, ctx: Context):
        store = FakeGraphStore(entities=PEOPLE, neighbors={"root": [("near", 1.0, "R")]})
        toolkit = GraphRAGToolkit(_SkillRAG(store))  # type: ignore[arg-type]

        ok = await toolkit.run_skill("impact_analysis", {"entity": "Root"}, ctx=ctx)
        assert ok.skill == "impact_analysis" and ok.n == 1
        assert ok.data["entity"] == "root"
        assert toolkit.ledger.get(1).tool == "run_skill"

        with pytest.raises(ToolError, match="no entity is called 'Nobody'") as missing:
            await toolkit.run_skill("impact_analysis", {"entity": "Nobody"}, ctx=ctx)
        assert "lookup_entity" in (missing.value.hint or "")

        with pytest.raises(ToolError, match="colour") as extra:
            await toolkit.run_skill("impact_analysis", {"entity": "Root", "colour": "red"}, ctx=ctx)
        assert "entity" in (extra.value.hint or "")

        with pytest.raises(ToolError, match="Unknown skill"):
            await toolkit.run_skill("no_such_skill", {}, ctx=ctx)

    async def test_evidence_is_the_data_not_the_llm_summary(self, ctx: Context):
        store = FakeGraphStore(
            entities=PEOPLE,
            query_map={
                "-[r:RELATES]-(m:__Entity__)": [["FOUNDED_IN", "y", "founded in 1998", "1998"]],
            },
        )
        rag = _SkillRAG(store)
        rag.llm = FakeLLM('{"contradictions": [], "summary": "It was founded in 2005."}')
        toolkit = GraphRAGToolkit(rag)  # type: ignore[arg-type]
        out = await toolkit.run_skill("contradiction_detection", {"entity": "alice"}, ctx=ctx)
        item = toolkit.ledger.get(out.n)
        assert "founded in 1998" in item.content and "2005" not in item.content
        assert item.metadata["summary"] == "It was founded in 2005." == out.summary
        check = await toolkit.check_citations("Alice was founded in 2005 [1].")
        assert check.unsupported and "2005" in check.unsupported[0].missing[0]


# ── Impact analysis: wide hubs and minimal graph stores ─────────


class _NoSummary:
    async def __call__(self, ctx, prompt):
        return ""


def _impact_skill(store) -> ImpactAnalysisSkill:
    skill = ImpactAnalysisSkill(store)

    async def resolve(ref, param="entity"):
        return ref

    skill._resolve_entity = resolve  # type: ignore[method-assign]
    skill._summarize = _NoSummary()  # type: ignore[method-assign]
    return skill


class TestImpactNeighbourLimits:
    async def test_reports_more_than_fifty_direct_neighbours(self):
        from graphrag_sdk.skills.impact_analysis import MAX_DIRECT_NEIGHBOURS

        limits: list[int | None] = []

        class Store:
            async def weighted_neighbors(self, node_id, *, limit=50):
                limits.append(limit)
                hub = [(f"n{i:03d}", 1.0, "R") for i in range(120)]
                return hub[:limit] if node_id == "root" else []

            async def pagerank(self, **kwargs):
                return {}

        result = await _impact_skill(Store()).run(Context(), entity="root", max_depth=1)
        assert result.data["num_impacted"] == 120
        assert limits == [MAX_DIRECT_NEIGHBOURS + 1] and MAX_DIRECT_NEIGHBOURS >= 120
        assert result.data["truncated"] is False

    async def test_flags_capped_neighbour_lookups(self):
        from graphrag_sdk.skills.impact_analysis import MAX_DIRECT_NEIGHBOURS

        class Store:
            async def weighted_neighbors(self, node_id, *, limit=50):
                if node_id == "root":
                    return [(f"n{i:04d}", 1.0, "R") for i in range(MAX_DIRECT_NEIGHBOURS + 5)][
                        :limit
                    ]
                return []

            async def pagerank(self, **kwargs):
                return {}

        result = await _impact_skill(Store()).run(Context(), entity="root", max_depth=1)
        assert result.data["num_impacted"] == MAX_DIRECT_NEIGHBOURS
        assert result.data["truncated"] is True

    async def test_flags_a_capped_later_hop(self):
        from graphrag_sdk.skills.impact_analysis import HOP_NEIGHBOURS

        class Store:
            async def weighted_neighbors(self, node_id, *, limit=50):
                if node_id == "root":
                    return [("a", 1.0, "R")]
                if node_id == "a":
                    return [(f"b{i}", 1.0, "R") for i in range(HOP_NEIGHBOURS + 10)][:limit]
                return []

            async def pagerank(self, **kwargs):
                return {}

        result = await _impact_skill(Store()).run(Context(), entity="root", max_depth=2)
        assert result.data["num_impacted"] == 1 + HOP_NEIGHBOURS
        assert result.data["truncated"] is True

    async def test_works_with_a_store_without_a_limit_argument(self):
        class Store:
            async def weighted_neighbors(self, node_id):
                return [("a", 1.0, "R")] if node_id == "root" else []

            async def pagerank(self, **kwargs):
                return {}

        result = await _impact_skill(Store()).run(Context(), entity="root", max_depth=2)
        assert result.data["num_impacted"] == 1
        assert [(i["entity"], i["distance"]) for i in result.data["impacted"]] == [("a", 1)]
