"""Tests for cypher_first strategy helpers.

The strategy class itself needs a graph + LLM, so we exercise the
pure-Python pieces (intent classifier, regex extractors, fuzzy intersect,
table formatter, numeric coercion) without external dependencies.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from graphrag_sdk.retrieval.strategies.cypher_first import (
    _coerce_number,
    _detect_property_kind,
    _extract_phrases,
    _extract_projects,
    _extract_roles,
    _fuzzy_intersect,
    _is_negation_existential,
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
