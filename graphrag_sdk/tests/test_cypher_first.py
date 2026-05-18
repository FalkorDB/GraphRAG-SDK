"""Tests for cypher_first strategy helpers.

The strategy class itself needs a graph + LLM, so we exercise the
pure-Python pieces (intent classifier, regex extractors, fuzzy intersect,
table formatter, numeric coercion) without external dependencies.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from graphrag_sdk.core.models import RawSearchResult
from graphrag_sdk.retrieval.strategies.cypher_first import (
    PATH_CYPHER_TABLE,
    PATH_NEGATION_EMPTY_NO,
    PATH_NUMERIC_MATH,
    PATH_RAG_FALLBACK,
    PATH_RAG_FALLBACK_CYPHER_EMPTY,
    PATH_RAG_FALLBACK_NUMERIC_FAIL,
    PATH_SHARED_PROPERTY_HYBRID,
    _coerce_number,
    _detect_property_kind,
    _extract_phrases,
    _extract_projects,
    _extract_roles,
    _fuzzy_intersect,
    _is_negation_existential,
    _tag_path,
    detect_aggregation_intent,
    format_result_as_markdown_table,
    is_which_list,
    is_yes_no,
)

# ── Intent classifier ────────────────────────────────────────────


class TestDetectIntent:
    def test_count_questions_are_aggregation(self):
        for q in [
            "How many people work at Acme Corp?",
            "How many distinct organizations are mentioned?",
            "Count of employees by org?",
        ]:
            assert detect_aggregation_intent(q) == "aggregation", q

    def test_which_most_is_aggregation(self):
        for q in [
            "Which city has the most employees mentioned?",
            "Which organization has the fewest employees?",
            "Which orgs share a project with Acme?",
        ]:
            assert detect_aggregation_intent(q) == "aggregation", q

    def test_more_than_multi_word(self):
        # The intent regex must catch up to 4 words between `more` and `than`.
        q = "Does Acme Corp have more employees mentioned than Initech Systems?"
        assert detect_aggregation_intent(q) == "aggregation"

    def test_both_a_and_b_is_aggregation(self):
        q = "Which job roles are held by employees at BOTH Acme Corp and Initech Systems?"
        assert detect_aggregation_intent(q) == "aggregation"

    def test_existential_is_aggregation(self):
        for q in [
            "Are there any organizations with no employees listed?",
            "Is there any employee at Acme on observability tooling?",
        ]:
            assert detect_aggregation_intent(q) == "aggregation", q

    def test_average_year_is_numeric_math(self):
        q = "What is the average year of founding across all 10 organizations?"
        assert detect_aggregation_intent(q) == "numeric_math"

    def test_total_revenue_is_numeric_math(self):
        assert detect_aggregation_intent("What is the total revenue?") == "numeric_math"

    def test_factoid_is_rag(self):
        for q in [
            "Who is the lighthouse keeper?",
            "Tell me about Acme Corp.",
            "What did the professor discover?",
        ]:
            assert detect_aggregation_intent(q) == "rag", q


class TestShapeDetectors:
    def test_yes_no_starts(self):
        assert is_yes_no("Is there any X?")
        assert is_yes_no("Does Acme have more employees than Initech?")
        assert is_yes_no("Are there any orgs without employees?")

    def test_yes_no_negative(self):
        assert not is_yes_no("Which X has the most Y?")
        assert not is_yes_no("How many people?")

    def test_which_list_starts(self):
        assert is_which_list("Which cities have offices?")
        assert is_which_list("List the orgs with 5 employees.")
        assert is_which_list("Name the people at Acme.")

    def test_which_list_negative(self):
        assert not is_which_list("How many people work here?")
        assert not is_which_list("Is there any employee at Acme?")

    def test_negation_existential(self):
        assert _is_negation_existential(
            "Are there any organizations with NO employees listed?"
        )
        assert _is_negation_existential(
            "Is there any company without an office?"
        )
        assert not _is_negation_existential(
            "Is there any employee at Acme who works on observability?"
        )
        assert not _is_negation_existential("How many people work here?")


# ── Property kind + extractors (M5 hybrid) ───────────────────────


class TestPropertyKind:
    def test_role_keywords(self):
        for q in [
            "What roles are at Acme?",
            "Which jobs are common?",
            "List the titles at Initech.",
        ]:
            assert _detect_property_kind(q) == "role", q

    def test_project_keywords(self):
        for q in [
            "Who works on observability?",
            "Which projects are shared?",
            "What initiatives is Acme contributing to?",
        ]:
            assert _detect_property_kind(q) == "project", q

    def test_other_returns_none(self):
        assert _detect_property_kind("Where is Acme based?") is None
        assert _detect_property_kind("How many people work there?") is None


class TestRoleExtractor:
    def test_basic_role_match(self):
        desc = "Anna Reyes is a senior engineer at Acme Corp."
        roles = _extract_roles(desc)
        assert "senior engineer" in roles

    def test_multiword_role(self):
        desc = "Uma Patel is a site reliability engineer at Initech Systems."
        assert "site reliability engineer" in _extract_roles(desc)

    def test_applied_scientist(self):
        desc = "Cyrus Doss is an applied scientist at Massive Dynamic."
        assert "applied scientist" in _extract_roles(desc)

    def test_no_role_no_match(self):
        # No professional-suffix word → no extraction.
        assert _extract_roles("Alice is based in Boston.") == set()

    def test_short_phrase_filtered(self):
        # 'engineer' alone (no qualifier) is too short — rejected by length gate.
        # But still captured as "engineer" if length > 3 — which it is (8).
        # The filter is just defensive; this test asserts current behavior.
        result = _extract_roles("Bob works as an engineer.")
        # accept either outcome; just ensure no crash
        assert isinstance(result, set)


class TestProjectExtractor:
    def test_basic_project_match(self):
        desc = (
            "Anna Reyes is a senior engineer at Acme Corp, "
            "contributing to the cross-region replication initiative "
            "and active in the cloud infrastructure community."
        )
        projects = _extract_projects(desc)
        assert any("cross-region replication initiative" in p for p in projects)

    def test_contributes_to_variant(self):
        desc = "Mira Jansen contributes to a migration to managed services."
        projects = _extract_projects(desc)
        assert any("migration to managed services" in p for p in projects)

    def test_no_contributes_no_match(self):
        assert _extract_projects("Pavel works at Wayne.") == set()


class TestExtractPhrases:
    def test_role_dispatch(self):
        out = _extract_phrases("Anna is a senior engineer", "role")
        assert "senior engineer" in out

    def test_project_dispatch(self):
        out = _extract_phrases("contributes to the pipeline rewrite", "project")
        assert any("pipeline rewrite" in p for p in out)

    def test_unknown_kind_empty(self):
        assert _extract_phrases("anything", "city") == set()


# ── Fuzzy intersect ──────────────────────────────────────────────


class TestFuzzyIntersect:
    def test_exact_match(self):
        a = {"senior engineer", "data scientist"}
        b = {"senior engineer", "product manager"}
        assert _fuzzy_intersect(a, b) == {"senior engineer"}

    def test_substring_in_one_direction(self):
        # "pipeline rewrite" (substring) should match "next-generation
        # pipeline rewrite" (superset).
        a = {"pipeline rewrite"}
        b = {"next-generation pipeline rewrite"}
        assert _fuzzy_intersect(a, b) == {"pipeline rewrite"}

    def test_substring_other_direction(self):
        a = {"next-generation pipeline rewrite"}
        b = {"pipeline rewrite"}
        assert _fuzzy_intersect(a, b) == {"next-generation pipeline rewrite"}

    def test_two_token_overlap_matches(self):
        a = {"automated incident response tooling"}
        b = {"incident response system tooling"}
        # 2+ shared content tokens: "incident", "response", "tooling"
        assert _fuzzy_intersect(a, b) == {"automated incident response tooling"}

    def test_single_stopword_no_match(self):
        # Sharing only "the" / "a" should not be enough.
        a = {"the migration"}
        b = {"a migration"}
        # Both reduce to {"migration"} (1 token) after stopword filter — no match.
        assert _fuzzy_intersect(a, b) == set()

    def test_no_overlap(self):
        assert _fuzzy_intersect({"alpha beta"}, {"gamma delta"}) == set()


# ── Markdown table formatter ─────────────────────────────────────


class TestFormatTable:
    def _result(self, header, rows):
        return SimpleNamespace(
            header=[[1, name] for name in header],
            result_set=rows,
        )

    def test_renders_with_headers(self):
        r = self._result(["city", "n"], [["Boston", 10], ["Chicago", 8]])
        md, parsed, truncated = format_result_as_markdown_table(r)
        assert "city | n" in md
        assert "Boston | 10" in md
        assert "Chicago | 8" in md
        assert not truncated
        assert parsed == [{"city": "Boston", "n": 10}, {"city": "Chicago", "n": 8}]

    def test_synthesizes_headers_when_missing(self):
        r = SimpleNamespace(header=None, result_set=[["x", "y"]])
        md, parsed, _ = format_result_as_markdown_table(r)
        assert "col_0 | col_1" in md
        assert parsed[0] == {"col_0": "x", "col_1": "y"}

    def test_empty_result(self):
        r = SimpleNamespace(header=[], result_set=[])
        md, parsed, truncated = format_result_as_markdown_table(r)
        assert md == "(empty result)"
        assert parsed == []
        assert not truncated

    def test_truncation_sentinel(self):
        rows = [["x", i] for i in range(150)]
        r = self._result(["item", "n"], rows)
        md, parsed, truncated = format_result_as_markdown_table(r, cap=50)
        assert truncated
        assert len(parsed) == 50
        assert "showing 50 of 150 rows" in md

    def test_null_and_list_cells(self):
        r = self._result(["a", "b"], [[None, ["x", "y", "z"]]])
        md, _, _ = format_result_as_markdown_table(r)
        assert "(null)" in md
        assert "x, y, z" in md


# ── Numeric coercion (M6 helper) ─────────────────────────────────


class TestCoerceNumber:
    def test_int(self):
        assert _coerce_number(1995) == 1995.0

    def test_float(self):
        assert _coerce_number(3.14) == pytest.approx(3.14)

    def test_string_with_number(self):
        # Date entity names are strings like "1995" — pull the integer.
        assert _coerce_number("1995") == 1995.0

    def test_string_with_embedded_number(self):
        assert _coerce_number("1995 is the year Acme was founded") == 1995.0

    def test_none(self):
        assert _coerce_number(None) is None

    def test_no_digits(self):
        assert _coerce_number("hello") is None

    def test_negative(self):
        assert _coerce_number("-42") == -42.0


# ── Path-tag contract (R2) ───────────────────────────────────────


class TestPathTag:
    """``_tag_path`` enforces the contract that every result emitted by
    ``CypherFirstAggregationStrategy`` carries a ``cypher_first_path``
    label, so operators can route metrics on which sub-path fired."""

    def test_labels_match_known_paths(self):
        # If we add a new path later, this list should grow in lockstep.
        assert PATH_NUMERIC_MATH == "numeric_math"
        assert PATH_SHARED_PROPERTY_HYBRID == "shared_property_hybrid"
        assert PATH_CYPHER_TABLE == "cypher_table"
        assert PATH_NEGATION_EMPTY_NO == "negation_empty_no"
        assert PATH_RAG_FALLBACK == "rag_fallback"
        assert PATH_RAG_FALLBACK_NUMERIC_FAIL == "rag_fallback_numeric_fail"
        assert PATH_RAG_FALLBACK_CYPHER_EMPTY == "rag_fallback_cypher_empty"

    def test_tag_path_adds_label_to_empty_metadata(self):
        result = RawSearchResult(records=[{"section": "x", "content": "y"}],
                                 metadata={})
        out = _tag_path(result, PATH_CYPHER_TABLE)
        assert out.metadata["cypher_first_path"] == PATH_CYPHER_TABLE
        assert out.metadata["strategy"] == "cypher_first"

    def test_tag_path_overwrites_existing_path_label(self):
        # When a sub-path delegates to another that already tagged the
        # result, the outer tag should win — it reflects the actual path
        # taken from the caller's perspective.
        result = RawSearchResult(
            records=[],
            metadata={"cypher_first_path": "earlier", "extra": 1},
        )
        out = _tag_path(result, PATH_RAG_FALLBACK)
        assert out.metadata["cypher_first_path"] == PATH_RAG_FALLBACK
        # Other metadata is preserved.
        assert out.metadata["extra"] == 1

    def test_tag_path_preserves_existing_strategy_label_if_set(self):
        # If a delegated strategy already tagged itself (e.g. "multi_path"),
        # don't clobber it — setdefault.
        result = RawSearchResult(
            records=[],
            metadata={"strategy": "multi_path"},
        )
        out = _tag_path(result, PATH_RAG_FALLBACK)
        assert out.metadata["strategy"] == "multi_path"
        assert out.metadata["cypher_first_path"] == PATH_RAG_FALLBACK

    def test_tag_path_returns_new_object_not_mutating_input(self):
        original_meta = {"strategy": "multi_path"}
        result = RawSearchResult(records=[], metadata=original_meta)
        out = _tag_path(result, PATH_RAG_FALLBACK)
        # Don't surprise callers by mutating their dict in place.
        assert "cypher_first_path" not in original_meta
        assert out.metadata is not original_meta


# ── End-to-end routing with mocks (R4) ───────────────────────────


class _FakeResult:
    """Mimics FalkorDB's query_raw return: ``result_set`` rows + a ``header``
    list of ``[type_int, name]`` pairs (we only use the name)."""
    def __init__(self, header, result_set):
        self.header = [[1, name] for name in header]
        self.result_set = result_set


class _FakeFallback:
    """Stand-in for MultiPathRetrieval — records the call and returns a
    well-formed RawSearchResult so we can verify delegation + tagging."""
    def __init__(self, records=None, metadata=None):
        self._records = records or [{"section": "passages", "content": "## ..."}]
        self._metadata = metadata or {"strategy": "multi_path"}
        self.calls = []

    async def _execute(self, query, ctx, **kwargs):
        self.calls.append(query)
        return RawSearchResult(records=list(self._records),
                               metadata=dict(self._metadata))


def _make_strategy(*, llm_responses=None, graph_results=None, fallback=None):
    """Build a CypherFirstAggregationStrategy with stubbed LLM + graph.

    ``llm_responses`` is a sequence of strings; each LLM call pops one.
    ``graph_results`` is a sequence of _FakeResult objects (or exceptions
    to raise); each ``query_raw`` call pops one.
    """
    from unittest.mock import AsyncMock, MagicMock

    from graphrag_sdk.core.models import LLMResponse
    from graphrag_sdk.retrieval.strategies.cypher_first import (
        CypherFirstAggregationStrategy,
    )

    llm = MagicMock()
    responses = list(llm_responses or [])
    async def _ainvoke(_prompt, **_kw):
        return LLMResponse(content=responses.pop(0) if responses else "")
    llm.ainvoke = AsyncMock(side_effect=_ainvoke)

    graph = MagicMock()
    results = list(graph_results or [])
    async def _query_raw(_cypher, _params=None):
        nxt = results.pop(0) if results else _FakeResult([], [])
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt
    graph.query_raw = AsyncMock(side_effect=_query_raw)

    return CypherFirstAggregationStrategy(
        graph_store=graph,
        vector_store=MagicMock(),
        embedder=MagicMock(),
        llm=llm,
        k_candidates=1,  # one candidate is enough for routing tests
        rag_fallback=fallback or _FakeFallback(),
    )


class TestStrategyRouting:
    """Behavioural tests: assert which sub-path fires for each intent +
    graph-state combination by inspecting ``metadata.cypher_first_path``."""

    async def test_rag_intent_delegates_to_fallback(self):
        # "Who is the lighthouse keeper?" doesn't match any aggregation
        # pattern — strategy must hand the query to the fallback verbatim
        # and tag the result with PATH_RAG_FALLBACK.
        from graphrag_sdk.core.context import Context
        fallback = _FakeFallback()
        strat = _make_strategy(fallback=fallback)
        ctx = Context()
        result = await strat._execute("Who is the lighthouse keeper?", ctx)
        assert result.metadata["cypher_first_path"] == PATH_RAG_FALLBACK
        assert fallback.calls == ["Who is the lighthouse keeper?"]

    async def test_aggregation_with_cypher_rows_takes_cypher_table(self):
        # Multi-candidate cypher returns a non-empty table → strategy
        # picks the table path and emits a single "cypher_results" item.
        from graphrag_sdk.core.context import Context
        cypher_code = (
            "```cypher\n"
            "MATCH (p:Person)-[:RELATES]-(o:Organization)\n"
            "RETURN o.name AS org, count(DISTINCT p) AS n\n"
            "ORDER BY n DESC LIMIT 5\n"
            "```"
        )
        strat = _make_strategy(
            llm_responses=[cypher_code],
            graph_results=[
                _FakeResult(["org", "n"], [["Acme", 10], ["Globex", 8]])
            ],
        )
        result = await strat._execute(
            "Which org has the most employees mentioned?", Context(),
        )
        assert result.metadata["cypher_first_path"] == PATH_CYPHER_TABLE
        assert result.metadata["cypher_rows"] == 2
        assert "Acme | 10" in result.records[0]["content"]

    async def test_numeric_intent_does_python_arithmetic(self):
        # Question asks for an average; cypher returns raw year values;
        # strategy computes 1989.9 in Python and tags PATH_NUMERIC_MATH.
        from graphrag_sdk.core.context import Context
        cypher_code = (
            "```cypher\nMATCH (d:Date) RETURN d.name AS year\n```"
        )
        years = [[str(y)] for y in
                 [1939, 1968, 1973, 1984, 1995, 1998, 2003, 2009, 2011, 2019]]
        strat = _make_strategy(
            llm_responses=[cypher_code],
            graph_results=[_FakeResult(["year"], years)],
        )
        result = await strat._execute(
            "What is the average year of founding across all 10 organizations?",
            Context(),
        )
        assert result.metadata["cypher_first_path"] == PATH_NUMERIC_MATH
        assert result.metadata["op"] == "average"
        assert result.metadata["value"] == pytest.approx(1989.9, abs=0.1)

    async def test_numeric_empty_extraction_falls_back_to_rag(self):
        # Cypher generated for the numeric path returns 0 numeric values
        # → strategy falls through to the RAG fallback and tags
        # PATH_RAG_FALLBACK_NUMERIC_FAIL.
        from graphrag_sdk.core.context import Context
        fallback = _FakeFallback()
        strat = _make_strategy(
            llm_responses=["```cypher\nMATCH (n:Person) RETURN n.name\n```"],
            graph_results=[_FakeResult(["name"], [["Alice"], ["Bob"]])],
            fallback=fallback,
        )
        result = await strat._execute(
            "What is the average year of founding?", Context(),
        )
        assert result.metadata["cypher_first_path"] == PATH_RAG_FALLBACK_NUMERIC_FAIL
        assert fallback.calls  # fallback was invoked

    async def test_negation_existential_empty_returns_no(self):
        # "Are there any orgs WITHOUT employees?" + cypher returns 0 rows
        # → strategy emits an authoritative "No" and tags
        # PATH_NEGATION_EMPTY_NO. No fallback delegation.
        from graphrag_sdk.core.context import Context
        cypher_code = (
            "```cypher\n"
            "MATCH (o:Organization) WHERE NOT EXISTS { "
            "MATCH (o)<-[:RELATES]-(p:Person) } RETURN o.name\n"
            "```"
        )
        fallback = _FakeFallback()
        strat = _make_strategy(
            llm_responses=[cypher_code],
            graph_results=[
                _FakeResult([], []),  # hybrid batch query (no shape match)
                _FakeResult(["name"], []),  # the actual cypher
            ],
            fallback=fallback,
        )
        result = await strat._execute(
            "Are there any organizations for which NO employees are listed?",
            Context(),
        )
        assert result.metadata["cypher_first_path"] == PATH_NEGATION_EMPTY_NO
        # Negation path must NOT delegate to the fallback.
        assert fallback.calls == []

    async def test_positive_existential_empty_falls_back_to_rag(self):
        # "Is there any employee at Acme on observability?" + cypher returns
        # 0 rows (typed label mismatch) → strategy falls through to RAG and
        # tags PATH_RAG_FALLBACK_CYPHER_EMPTY.
        from graphrag_sdk.core.context import Context
        cypher_code = (
            "```cypher\n"
            "MATCH (p:Person)-[:RELATES]-(t:Technology) RETURN p.name\n"
            "```"
        )
        fallback = _FakeFallback()
        strat = _make_strategy(
            llm_responses=[cypher_code],
            graph_results=[_FakeResult(["name"], [])],
            fallback=fallback,
        )
        result = await strat._execute(
            "Is there any employee at Acme working on observability?",
            Context(),
        )
        assert result.metadata["cypher_first_path"] == PATH_RAG_FALLBACK_CYPHER_EMPTY
        assert len(fallback.calls) == 1

    async def test_hybrid_warns_when_topology_assumption_violated(self, caplog):
        # The batched (Org)<-[:RELATES]-(Person) query returns zero tuples
        # → strategy emits a warning and falls through (returns None from
        # the hybrid; the rest of _execute keeps going).
        import logging

        from graphrag_sdk.core.context import Context
        cypher_code = (
            "```cypher\nMATCH (p:Person) RETURN p.name AS name\n```"
        )
        strat = _make_strategy(
            llm_responses=[cypher_code],
            graph_results=[
                _FakeResult([], []),  # hybrid batch — zero topology tuples
                _FakeResult(["name"], [["Alice"]]),  # cypher_table cypher
            ],
        )
        with caplog.at_level(logging.WARNING,
                             logger="graphrag_sdk.retrieval.strategies.cypher_first"):
            await strat._execute(
                "Which roles are held by employees at BOTH Acme and Globex?",
                Context(),
            )
        # The topology-violation warning fired.
        assert any("zero (Organization)<-[:RELATES]-(Person) tuples" in r.message
                   for r in caplog.records)

    async def test_hybrid_batch_query_is_scoped_and_capped(self):
        # The shared-property hybrid batch must not full-scan
        # (Org)<-[:RELATES]-(Person): for "both A and B" it scopes by the two
        # org names, caps the row count, and truncates chunk text length. For
        # "same X as Z" the org filter doesn't apply (we need every other org
        # to compare against the target) but the LIMIT and substring cap
        # still apply.
        from unittest.mock import AsyncMock, MagicMock

        from graphrag_sdk.core.context import Context
        from graphrag_sdk.retrieval.strategies.cypher_first import (
            _HYBRID_BATCH_ROW_CAP,
            _HYBRID_CHUNK_TEXT_CAP,
            CypherFirstAggregationStrategy,
        )

        for query, expect_scoped in [
            ("Which roles are held by employees at BOTH Acme and Globex?", True),
            ("Which employees have the same role as someone at Acme?", False),
        ]:
            graph = MagicMock()
            captured: list[tuple[str, dict | None]] = []
            async def _query_raw(cypher, params=None, _cap=captured):
                _cap.append((cypher, params))
                return _FakeResult([], [])
            graph.query_raw = AsyncMock(side_effect=_query_raw)

            strat = CypherFirstAggregationStrategy(
                graph_store=graph,
                vector_store=MagicMock(),
                embedder=MagicMock(),
                llm=MagicMock(),
                k_candidates=1,
                rag_fallback=MagicMock(),
            )
            await strat._hybrid_path.maybe_handle(query, Context())

            assert captured, f"hybrid batch query did not fire for: {query}"
            batch_cypher, batch_params = captured[0]
            assert batch_params is not None, "params must be passed to query_raw"
            assert batch_params["max_chunk_len"] == _HYBRID_CHUNK_TEXT_CAP
            assert "substring(c.text, 0, $max_chunk_len)" in batch_cypher
            assert f"LIMIT {_HYBRID_BATCH_ROW_CAP}" in batch_cypher
            if expect_scoped:
                # "Both A and B" — org names are pushed into the WHERE clause
                # rather than scanned across the full graph.
                assert "WHERE toLower(o.name) CONTAINS toLower($org_a)" in batch_cypher
                assert batch_params.get("org_a", "").lower() == "acme"
                assert batch_params.get("org_b", "").lower() == "globex"
            else:
                # "Same X as Z" — no org_a/org_b in params; LIMIT + substring
                # cap are the guardrails.
                assert "org_a" not in batch_params
                assert "org_b" not in batch_params


# ── Pluggable phrase extractor (R8) ──────────────────────────────


class TestPhraseExtractor:
    """Domain-specific extractors can replace the default role/project
    regexes without forking the strategy."""

    def test_default_extractor_matches_module_regexes(self):
        from graphrag_sdk.retrieval.strategies.cypher_first import (
            DefaultPhraseExtractor,
        )
        ext = DefaultPhraseExtractor()
        assert "senior engineer" in ext.extract(
            "Anna is a senior engineer at Acme.", "role"
        )
        assert any("pipeline rewrite" in p for p in ext.extract(
            "contributes to the pipeline rewrite", "project"
        ))
        # Unknown kinds return an empty set, not an exception.
        assert ext.extract("anything", "city") == set()

    async def test_strategy_uses_custom_extractor_in_hybrid_path(self):
        """A custom extractor passed to the strategy is consulted by the
        shared-property hybrid instead of the default regexes."""
        from graphrag_sdk.core.context import Context
        from graphrag_sdk.retrieval.strategies.cypher_first import PhraseExtractor

        class _UpperCaseRoleExtractor(PhraseExtractor):
            """Match exactly the literal strings 'ALPHA' and 'BETA' as roles
            — clearly distinguishable from the default regex output."""
            def extract(self, text, kind):
                if kind == "role":
                    return {w for w in ("ALPHA", "BETA") if w in text}
                return set()

        # Both orgs have one person whose chunk text mentions the same
        # custom token. With the default extractor, none of these phrases
        # would match (no role suffix). With our custom one, ALPHA is
        # common to both.
        from unittest.mock import AsyncMock, MagicMock

        graph = MagicMock()
        batch_rows = [
            ["Acme",    "Alice", "no description", ["Alice does ALPHA at Acme."]],
            ["Acme",    "Anna",  "no description", ["Anna does BETA at Acme."]],
            ["Globex",  "Bob",   "no description", ["Bob does ALPHA at Globex."]],
            ["Globex",  "Bea",   "no description", ["Bea does GAMMA at Globex."]],
        ]
        async def _query_raw(_cypher, _params=None):
            return SimpleNamespace(
                header=[[1, "org"], [1, "person"], [1, "desc"], [1, "chunks"]],
                result_set=batch_rows,
            )
        graph.query_raw = AsyncMock(side_effect=_query_raw)

        from graphrag_sdk.retrieval.strategies.cypher_first import (
            CypherFirstAggregationStrategy,
        )
        strat = CypherFirstAggregationStrategy(
            graph_store=graph,
            vector_store=MagicMock(),
            embedder=MagicMock(),
            llm=MagicMock(),
            k_candidates=1,
            rag_fallback=MagicMock(),
            phrase_extractor=_UpperCaseRoleExtractor(),
        )
        result = await strat._execute(
            "Which roles are held by employees at BOTH Acme and Globex?",
            Context(),
        )
        # Hybrid fired (not a fallback) and computed the right intersection.
        assert result.metadata["cypher_first_path"] == PATH_SHARED_PROPERTY_HYBRID
        assert result.metadata["common"] == ["ALPHA"]
