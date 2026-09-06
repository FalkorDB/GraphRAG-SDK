"""``finalize()``: a resolver judges the whole graph.

Within one document, ``LLMVerifiedResolution`` decides that "Ms. Raman" and
"Priya Raman" are one person. Across sources it never got the chance: a table
row and a document mention arrive in different ``ingest`` calls, and the
deduplicator that does see both only merges spellings of the same name. The
pair fell between the two, and was reported as a probable duplicate.

``finalize`` closes the gap: by default with an ``LLMVerifiedResolution`` over
the instance's own model and embedder, or with whatever resolver it is handed,
or not at all with ``resolve=False``. The resolver decides identity exactly as
it does within a document; the deduplicator merges by its own rules — the
table's node survives, its signed values are kept, two rows of one table stay
two people. These tests pin that split of responsibilities.

Requires ``RUN_INTEGRATION=1``.
"""

from __future__ import annotations

import pytest

from graphrag_sdk import Column, Entity, ExactMatchResolution, Ontology, TableMapping
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, ResolutionResult
from graphrag_sdk.core.providers import Embedder
from graphrag_sdk.ingestion.resolution_strategies.base import (
    RESOLUTION_ASK_PAIRS,
    RESOLUTION_DISTINCT_IDS,
    RESOLUTION_REJECTED_PAIRS,
    RESOLUTION_SKIP_PAIRS,
    ResolutionStrategy,
)
from graphrag_sdk.ingestion.resolution_strategies.llm_verified_resolution import (
    LLMVerifiedResolution,
)
from graphrag_sdk.storage.deduplicator import _clusters

from .conftest import MockLLM

pytestmark = pytest.mark.integration

PRIYA = "priya_raman__person"
MS_RAMAN = "ms._raman__person"


def people(source: str = "employees.csv") -> TableMapping:
    return TableMapping(
        source=source,
        label="Person",
        key="employee_id",
        name="full_name",
        properties={"age": Column("age", "INTEGER"), "title": Column("title", "STRING")},
    )


def ontology() -> Ontology:
    return Ontology(entities=[Entity(label="Person")], tables=[people()])


class SaysTheyAreOne(ResolutionStrategy):
    """A resolver whose verdict is scripted, so the merge is what is under test."""

    def __init__(self, remap: dict[str, str], reject: set[frozenset[str]] | None = None) -> None:
        self.remap = remap
        self.reject = reject or set()
        self.seen: GraphData | None = None
        self.ctx: Context | None = None

    async def resolve(self, graph_data: GraphData, ctx: Context) -> ResolutionResult:
        self.seen = graph_data
        self.ctx = ctx
        ctx.metadata[RESOLUTION_REJECTED_PAIRS] = set(self.reject)
        return ResolutionResult(
            nodes=graph_data.nodes,
            relationships=graph_data.relationships,
            merged_count=len(self.remap),
            remap=dict(self.remap),
        )


class NearEmbedder(Embedder):
    """Two people's names embed close enough to ask about, and nobody else's do."""

    def __init__(self, close: set[str]) -> None:
        self.close = close

    @property
    def model_name(self) -> str:
        return "near-embedder"

    def embed_query(self, text: str, **kwargs) -> list[float]:
        if text in self.close:
            # Distinct vectors at cosine ~0.9: the LLM's zone, not the hard merge's.
            return [1.0, 0.0] if text == sorted(self.close)[0] else [0.9, 0.436]
        return [0.0, 1.0]


class OrthogonalEmbedder(Embedder):
    """Every distinct name embeds at similarity 0 to every other."""

    def __init__(self) -> None:
        self.seen: dict[str, int] = {}

    @property
    def model_name(self) -> str:
        return "orthogonal-embedder"

    def embed_query(self, text: str, **kwargs) -> list[float]:
        axis = self.seen.setdefault(text, len(self.seen))
        vector = [0.0] * 16
        vector[axis % 16] = 1.0
        return vector


async def _ingest_table_and_note(rag, tmp_path, note_llm_index: int = 0):
    path = tmp_path / "employees.csv"
    path.write_text("employee_id,full_name,age,title\nE-3,Priya Raman,39,Head of Regulatory\n")
    await rag.ingest(str(path))
    await rag.ingest(
        text="Ms. Raman told the regulator the tariff filing would be late.",
        document_id="market_note.txt",
    )


def _note_extraction() -> str:
    return (
        '{"entities": [{"name": "Ms. Raman", "type": "Person", '
        '"description": "Quoted in the market note on the tariff filing"}], '
        '"relationships": []}'
    )


class TestTheResolverDecidesAndTheTableRowSurvives:
    async def test_a_mention_the_resolver_recognises_folds_into_the_row(
        self, real_falkordb_rag_factory, tmp_path
    ):
        llm = MockLLM([_note_extraction()], strict=True)
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=ExactMatchResolution(resolve_property="name"), ontology=ontology()
        )
        await _ingest_table_and_note(rag, tmp_path)
        assert (await rag.query("MATCH (p:Person) RETURN count(p)")) == [[2]], (
            "different spellings, so two nodes until something decides"
        )

        resolver = SaysTheyAreOne({MS_RAMAN: PRIYA})
        result = await rag.finalize(resolver=resolver)

        assert result.resolved_duplicates == ["Person 'Ms. Raman' -> 'Priya Raman'"]
        assert result.entities_deduplicated == 1
        assert result.probable_duplicates == [], "decided, so no longer a guess to report"
        rows = await rag.query(
            "MATCH (p:Person) RETURN p.id, p.name, p.employees__age, p.employees__title, "
            "p.description, p.entity_key"
        )
        assert rows == [
            [
                PRIYA,
                "Priya Raman",
                39,
                "Head of Regulatory",
                "Quoted in the market note on the tariff filing",
                "E-3",
            ]
        ], "the row's node and its typed values, plus what the note knew about her"
        mentions = await rag.query(
            "MATCH (p:Person {id: $id})-[:MENTIONED_IN]->(c:Chunk)<-[:PART_OF]-(d:Document) "
            "RETURN collect(DISTINCT d.id)",
            {"id": PRIYA},
        )
        assert sorted(mentions[0][0]) == ["employees.csv", "market_note.txt"], (
            "the note's mention moved onto the row's node"
        )
        await rag.close()

    async def test_the_resolver_sees_the_table_values_as_the_rows_description(
        self, real_falkordb_rag_factory, tmp_path
    ):
        llm = MockLLM([_note_extraction()], strict=True)
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=ExactMatchResolution(resolve_property="name"), ontology=ontology()
        )
        await _ingest_table_and_note(rag, tmp_path)

        resolver = SaysTheyAreOne({})
        await rag.finalize(resolver=resolver)

        assert resolver.seen is not None
        shown = {node.id: node for node in resolver.seen.nodes}
        assert set(shown) == {PRIYA, MS_RAMAN}
        assert shown[PRIYA].label == "Person"
        assert shown[PRIYA].properties["description"] == (
            "employees: employee_id E-3, age 39, title Head of Regulatory"
        ), "a row has no prose, so its evidence is the values the table signed onto it"
        assert shown[MS_RAMAN].properties["description"].startswith("Quoted in the market note")
        await rag.close()

    async def test_two_rows_of_one_table_never_merge_whatever_the_resolver_says(
        self, real_falkordb_rag_factory, tmp_path
    ):
        rag = real_falkordb_rag_factory(
            llm=MockLLM(strict=True),
            resolver=ExactMatchResolution(resolve_property="name"),
            ontology=ontology(),
        )
        path = tmp_path / "employees.csv"
        path.write_text(
            "employee_id,full_name,age,title\n"
            "E-3,Priya Raman,39,Head of Regulatory\n"
            "E-4,P. Raman,52,Facilities\n"
        )
        await rag.ingest(str(path))

        resolver = SaysTheyAreOne({"p._raman__person": PRIYA})
        result = await rag.finalize(resolver=resolver)

        assert resolver.ctx.metadata[RESOLUTION_DISTINCT_IDS] == {PRIYA, "p._raman__person"}, (
            "told up front that keyed rows are pairwise distinct"
        )
        assert resolver.ctx.metadata[RESOLUTION_ASK_PAIRS] == set(), (
            "'P. Raman' ~ 'Priya Raman' is a near miss by name, but not one to ask about"
        )
        assert result.resolved_duplicates == []
        assert (await rag.query("MATCH (p:Person) RETURN count(p)")) == [[2]], (
            "a key declared them two people; a resolver cannot overrule the mapping"
        )
        await rag.close()

    async def test_without_a_resolver_the_pair_is_reported_as_before(
        self, real_falkordb_rag_factory, tmp_path
    ):
        llm = MockLLM([_note_extraction()], strict=True)
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=ExactMatchResolution(resolve_property="name"), ontology=ontology()
        )
        await _ingest_table_and_note(rag, tmp_path)

        result = await rag.finalize()

        assert result.resolved_duplicates == []
        assert (await rag.query("MATCH (p:Person) RETURN count(p)")) == [[2]]
        await rag.close()


class TestTheSameStrategyIngestUsesJudgesTheGraph:
    """``LLMVerifiedResolution`` end to end: embed, ask, answer, merge."""

    def _rag(self, real_falkordb_rag_factory, verdict: str):
        # Call 1 is the note's extraction; call 2 the resolver's YES/NO on the pair.
        llm = MockLLM([_note_extraction(), verdict], strict=True)
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=ExactMatchResolution(resolve_property="name"), ontology=ontology()
        )
        judge = LLMVerifiedResolution(llm=llm, embedder=NearEmbedder({"Priya Raman", "Ms. Raman"}))
        return rag, judge

    async def test_yes_merges_onto_the_row(self, real_falkordb_rag_factory, tmp_path):
        rag, judge = self._rag(real_falkordb_rag_factory, "YES — same person, Kestrel Grid")
        await _ingest_table_and_note(rag, tmp_path)

        result = await rag.finalize(resolver=judge)

        assert result.resolved_duplicates == ["Person 'Ms. Raman' -> 'Priya Raman'"]
        rows = await rag.query("MATCH (p:Person) RETURN p.id, p.employees__age, p.description")
        assert rows == [[PRIYA, 39, "Quoted in the market note on the tariff filing"]]
        await rag.close()

    async def test_no_leaves_two_and_reports_the_pair(self, real_falkordb_rag_factory, tmp_path):
        rag, judge = self._rag(real_falkordb_rag_factory, "NO — a different Raman")
        await _ingest_table_and_note(rag, tmp_path)

        result = await rag.finalize(resolver=judge)

        assert result.resolved_duplicates == []
        assert (await rag.query("MATCH (p:Person) RETURN count(p)")) == [[2]]
        await rag.close()


class TestANoIsRememberedAndANameRuleIsAsked:
    async def test_a_rejected_pair_is_recorded_skipped_next_time_and_not_reported(
        self, real_falkordb_rag_factory, tmp_path
    ):
        llm = MockLLM([_note_extraction()], strict=True)
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=ExactMatchResolution(resolve_property="name"), ontology=ontology()
        )
        await _ingest_table_and_note(rag, tmp_path)
        pair = frozenset((PRIYA, MS_RAMAN))

        first = SaysTheyAreOne({}, reject={pair})
        result = await rag.finalize(resolver=first)

        assert result.rejected_duplicates == ["Person 'Ms. Raman' | 'Priya Raman'"]
        assert first.ctx.metadata[RESOLUTION_SKIP_PAIRS] == set(), "nothing decided yet"
        edges = await rag.query(
            "MATCH (a:__Entity__)-[d:DISTINCT_FROM]->(b:__Entity__) RETURN a.id, b.id, d.decided_by"
        )
        assert edges == [[MS_RAMAN, PRIYA, "SaysTheyAreOne"]]

        second = SaysTheyAreOne({MS_RAMAN: PRIYA})
        result = await rag.finalize(resolver=second)

        assert second.ctx.metadata[RESOLUTION_SKIP_PAIRS] == {pair}, "told what was decided"
        assert result.rejected_duplicates == [], "not decided again"
        assert result.resolved_duplicates == ["Person 'Ms. Raman' -> 'Priya Raman'"], (
            "a strategy that ignores the hint is still obeyed — the hint is economy, not policy"
        )
        assert (await rag.query("MATCH ()-[d:DISTINCT_FROM]->() RETURN count(d)")) == [[0]], (
            "the memory went with the merged node"
        )
        await rag.close()

    async def test_a_decided_pair_leaves_the_probable_duplicates_report(
        self, real_falkordb_rag_factory, tmp_path
    ):
        # "M. Ellison" against "Maya Ellison" is a near miss by name rule.
        llm = MockLLM(
            [
                '{"entities": [{"name": "M. Ellison", "type": "Person", '
                '"description": "Presented the remediation plan"}], "relationships": []}'
            ],
            strict=True,
        )
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=ExactMatchResolution(resolve_property="name"), ontology=ontology()
        )
        path = tmp_path / "employees.csv"
        path.write_text("employee_id,full_name,age,title\nE-1,Maya Ellison,34,Engineer\n")
        await rag.ingest(str(path))
        await rag.ingest(text="M. Ellison presented the plan.", document_id="board.txt")
        maya, m = "maya_ellison__person", "m._ellison__person"

        before = await rag.finalize(resolve=False)
        assert [d.split(" — ")[0] for d in before.probable_duplicates] == [
            "Person 'M. Ellison' ~ 'Maya Ellison'"
        ] or [d.split(" — ")[0] for d in before.probable_duplicates] == [
            "Person 'Maya Ellison' ~ 'M. Ellison'"
        ]

        resolver = SaysTheyAreOne({}, reject={frozenset((maya, m))})
        after = await rag.finalize(resolver=resolver)

        assert resolver.ctx.metadata[RESOLUTION_ASK_PAIRS] == {frozenset((maya, m))}, (
            "the near miss was handed to the resolver as a pair to judge"
        )
        assert after.rejected_duplicates == ["Person 'M. Ellison' | 'Maya Ellison'"]
        assert after.probable_duplicates == [], "decided, so no longer a guess"
        assert (await rag.finalize(resolve=False)).probable_duplicates == [], "and it stays decided"
        await rag.close()

    async def test_llm_verified_asks_a_near_miss_below_its_threshold_and_skips_a_no(
        self, real_falkordb_rag_factory, tmp_path
    ):
        # Call 1: extraction. Call 2: the one YES/NO. A third call would fail loudly.
        llm = MockLLM(
            [
                '{"entities": [{"name": "M. Ellison", "type": "Person", '
                '"description": "Presented the remediation plan"}], "relationships": []}',
                "NO — not enough to say",
            ],
            strict=True,
        )
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=ExactMatchResolution(resolve_property="name"), ontology=ontology()
        )
        path = tmp_path / "employees.csv"
        path.write_text("employee_id,full_name,age,title\nE-1,Maya Ellison,34,Engineer\n")
        await rag.ingest(str(path))
        await rag.ingest(text="M. Ellison presented the plan.", document_id="board.txt")
        # Orthogonal vectors: similarity 0, far below any threshold. Only the
        # name rule can bring this pair in front of the model.
        judge = LLMVerifiedResolution(llm=llm, embedder=OrthogonalEmbedder())

        first = await rag.finalize(resolver=judge)
        assert first.rejected_duplicates == ["Person 'M. Ellison' | 'Maya Ellison'"]
        assert first.probable_duplicates == []

        second = await rag.finalize(resolver=judge)
        assert second.rejected_duplicates == [], "remembered; the model was not asked again"
        assert (await rag.query("MATCH (p:Person) RETURN count(p)")) == [[2]]
        await rag.close()

    async def test_llm_verified_never_asks_about_two_rows_of_one_table(
        self, real_falkordb_rag_factory, tmp_path
    ):
        # Strict with no responses: any LLM call fails the test.
        llm = MockLLM([], strict=True)
        rag = real_falkordb_rag_factory(
            llm=llm, resolver=ExactMatchResolution(resolve_property="name"), ontology=ontology()
        )
        path = tmp_path / "employees.csv"
        path.write_text(
            "employee_id,full_name,age,title\n"
            "E-3,Priya Raman,39,Head of Regulatory\n"
            "E-4,P. Raman,52,Facilities\n"
        )
        await rag.ingest(str(path))
        # Both names embed identically: a hard merge at any threshold, were it allowed.
        judge = LLMVerifiedResolution(llm=llm, embedder=NearEmbedder(set()))

        result = await rag.finalize(resolver=judge)

        assert result.resolved_duplicates == result.rejected_duplicates == []
        assert (await rag.query("MATCH (p:Person) RETURN count(p)")) == [[2]]
        await rag.close()


class TestFinalizeJudgesByDefault:
    """No resolver given: ``finalize`` builds one over the instance's llm and embedder."""

    def _rag(self, real_falkordb_rag_factory, llm):
        return real_falkordb_rag_factory(
            llm=llm, resolver=ExactMatchResolution(resolve_property="name"), ontology=ontology()
        )

    async def _ingest_row_and_initialled_mention(self, rag, tmp_path):
        path = tmp_path / "employees.csv"
        path.write_text("employee_id,full_name,age,title\nE-1,Maya Ellison,34,Engineer\n")
        await rag.ingest(str(path))
        await rag.ingest(text="M. Ellison presented the plan.", document_id="board.txt")

    def _mention(self) -> str:
        return (
            '{"entities": [{"name": "M. Ellison", "type": "Person", '
            '"description": "Presented the remediation plan"}], "relationships": []}'
        )

    async def test_the_default_asks_the_instances_model_and_merges_on_yes(
        self, real_falkordb_rag_factory, tmp_path
    ):
        # Call 1: extraction. Call 2: the resolver finalize() built for itself.
        llm = MockLLM([self._mention(), "YES — initial and surname match"], strict=True)
        rag = self._rag(real_falkordb_rag_factory, llm)
        await self._ingest_row_and_initialled_mention(rag, tmp_path)

        result = await rag.finalize()

        assert result.resolved_duplicates == ["Person 'M. Ellison' -> 'Maya Ellison'"]
        assert llm._call_index == 2, "the model was asked exactly once"
        rows = await rag.query("MATCH (p:Person) RETURN p.id, p.employees__age, p.description")
        assert rows == [["maya_ellison__person", 34, "Presented the remediation plan"]]
        await rag.close()

    async def test_the_default_remembers_a_no_under_its_own_name(
        self, real_falkordb_rag_factory, tmp_path
    ):
        llm = MockLLM([self._mention(), "NO — could be anyone"], strict=True)
        rag = self._rag(real_falkordb_rag_factory, llm)
        await self._ingest_row_and_initialled_mention(rag, tmp_path)

        first = await rag.finalize()
        second = await rag.finalize()

        assert first.rejected_duplicates == ["Person 'M. Ellison' | 'Maya Ellison'"]
        assert second.rejected_duplicates == [] and llm._call_index == 2
        assert (await rag.query("MATCH ()-[d:DISTINCT_FROM]->() RETURN d.decided_by")) == [
            ["LLMVerifiedResolution"]
        ]
        await rag.close()

    async def test_resolve_false_reports_and_calls_no_model(
        self, real_falkordb_rag_factory, tmp_path
    ):
        llm = MockLLM([self._mention()], strict=True)
        rag = self._rag(real_falkordb_rag_factory, llm)
        await self._ingest_row_and_initialled_mention(rag, tmp_path)

        result = await rag.finalize(resolve=False)

        assert llm._call_index == 1, "extraction only"
        assert result.resolved_duplicates == result.rejected_duplicates == []
        assert len(result.probable_duplicates) == 1, "reported, not decided"
        assert (await rag.query("MATCH (p:Person) RETURN count(p)")) == [[2]]
        await rag.close()

    async def test_the_default_is_tuned_for_names_that_differ_across_sources(
        self, real_falkordb_rag_factory
    ):
        rag = self._rag(real_falkordb_rag_factory, MockLLM())
        judge = rag._default_resolver()

        assert isinstance(judge, LLMVerifiedResolution)
        assert judge.llm is rag.llm and judge.embedder is rag.embedder
        assert judge.soft_threshold < LLMVerifiedResolution(rag.llm, rag.embedder).soft_threshold
        await rag.close()


class TestClusters:
    def test_chains_are_followed_and_labels_kept_apart(self):
        by_id = {
            "a": {"id": "a", "label": "Person"},
            "b": {"id": "b", "label": "Person"},
            "c": {"id": "c", "label": "Person"},
            "fruit": {"id": "fruit", "label": "Food"},
        }
        remap = {"a": "b", "b": "c", "fruit": "c", "ghost": "c"}

        groups = _clusters(remap, by_id)

        assert [sorted(e["id"] for e in group) for group in groups] == [["a", "b", "c"]], (
            "one group for the people; the fruit the resolver merged across labels is dropped,"
            " and an id the graph does not hold is ignored"
        )

    def test_a_cycle_terminates(self):
        by_id = {"a": {"id": "a", "label": "X"}, "b": {"id": "b", "label": "X"}}
        assert len(_clusters({"a": "b", "b": "a"}, by_id)) <= 1
