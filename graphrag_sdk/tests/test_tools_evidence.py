"""Tests for graphrag_sdk.tools.evidence: the evidence ledger and citation checks."""

from __future__ import annotations

import pytest

from graphrag_sdk.tools.evidence import (
    DEFAULT_LEDGER_CAPACITY,
    Evidence,
    EvidenceLedger,
    GroundedAnswer,
    ground_answer,
)


def ledger(n: int, capacity: int = DEFAULT_LEDGER_CAPACITY) -> EvidenceLedger:
    out = EvidenceLedger(capacity=capacity)
    for i in range(n):
        out.add(tool="retrieve", kind="fact", content=f"fact {i + 1}")
    return out


# ── EvidenceLedger ───────────────────────────────────────────────


class TestLedger:
    def test_numbers_run_from_one_across_tools(self):
        led = EvidenceLedger()
        a = led.add(tool="retrieve", kind="passage", content="A", source="hr.pdf")
        b = led.add(tool="cypher_query", kind="row", content="B", metadata={"row": 0})
        assert (a.n, b.n) == (1, 2)
        assert led.get(1) is a and led.get(2).tool == "cypher_query"
        assert led.get(3) is None
        assert [e.n for e in led.items] == [1, 2]
        assert len(led) == 2
        assert 1 in led and 3 not in led and "1" not in led

    def test_metadata_is_copied(self):
        meta = {"k": 1}
        ev = EvidenceLedger().add(tool="t", kind="k", content="c", metadata=meta)
        meta["k"] = 2
        assert ev.metadata == {"k": 1}

    def test_oldest_items_are_evicted_and_numbers_stay_stable(self):
        led = ledger(5, capacity=3)
        assert len(led) == 3
        assert [e.n for e in led.items] == [3, 4, 5]
        assert 1 not in led and led.get(1) is None
        assert led.get(5).content == "fact 5"
        assert led.add(tool="t", kind="k", content="next").n == 6
        assert [e.n for e in led.items] == [4, 5, 6]

    def test_citing_an_evicted_number_is_invalid(self):
        led = ledger(5, capacity=3)
        g = ground_answer("Old [1] and new [5].", led)
        assert g.grounded and g.cited == [5] and g.invalid == [1]
        assert g.answer == "Old and new [5]."

    @pytest.mark.parametrize("capacity", [0, -1])
    def test_capacity_must_be_positive(self, capacity):
        with pytest.raises(ValueError):
            EvidenceLedger(capacity=capacity)

    def test_default_capacity(self):
        assert DEFAULT_LEDGER_CAPACITY == 2000
        led = ledger(DEFAULT_LEDGER_CAPACITY + 1)
        assert len(led) == DEFAULT_LEDGER_CAPACITY and 1 not in led

    def test_evidence_to_dict_omits_empty_fields(self):
        assert Evidence(n=1, tool="t", kind="k", content="c").to_dict() == {
            "n": 1,
            "tool": "t",
            "kind": "k",
            "content": "c",
        }
        full = Evidence(n=2, tool="t", kind="k", content="c", source="s", metadata={"a": 1})
        assert full.to_dict()["source"] == "s" and full.to_dict()["metadata"] == {"a": 1}


# ── ground_answer ────────────────────────────────────────────────


class TestGroundAnswer:
    def test_valid_citations(self):
        led = ledger(2)
        g = ground_answer("Alice works at Acme [1] in Berlin [2].", led)
        assert isinstance(g, GroundedAnswer)
        assert g.grounded and g.reason == "cited"
        assert g.cited == [1, 2] and g.invalid == []
        assert g.answer == "Alice works at Acme [1] in Berlin [2]."
        assert [e.content for e in g.citations(led)] == ["fact 1", "fact 2"]

    def test_cited_is_in_first_use_order_without_repeats(self):
        g = ground_answer("B [2]. A [1]. B again [2, 1].", ledger(2))
        assert g.cited == [2, 1]

    def test_invalid_markers_are_removed(self):
        g = ground_answer("Acme [1, 9] is big [7].", ledger(2))
        assert g.grounded and g.reason == "cited"
        assert g.answer == "Acme [1] is big."
        assert g.invalid == [9, 7]

    def test_only_invented_citations_is_ungrounded(self):
        # The toolkit reports the verdict; the cleaned text is returned as is
        # (the caller's agent decides what to reply).
        g = ground_answer("Acme is big [7].", ledger(2))
        assert not g.grounded and g.reason == "invalid_citations"
        assert g.answer == "Acme is big."
        assert g.cited == [] and g.invalid == [7]

    def test_uncited_answer_is_ungrounded(self):
        g = ground_answer("Acme is big.", ledger(2))
        assert not g.grounded and g.reason == "no_citations"
        assert g.answer == "Acme is big."

    def test_no_evidence_means_no_citation_can_match(self):
        g = ground_answer("Acme was founded in 1999 [1].", EvidenceLedger())
        assert not g.grounded and g.reason == "invalid_citations"
        assert g.answer == "Acme was founded in 1999."

    def test_conversation_is_not_special_cased(self):
        # The SDK no longer answers; there is no "conversation" exemption.
        g = ground_answer("Hello! How can I help?", EvidenceLedger())
        assert not g.grounded and g.reason == "no_citations"
        assert g.answer == "Hello! How can I help?"

    @pytest.mark.parametrize("answer", ["", "   ", None])
    def test_empty_answer_is_ungrounded(self, answer):
        g = ground_answer(answer, ledger(1))
        assert not g.grounded and g.reason == "empty" and g.answer == ""
        assert g.cited == [] and g.invalid == []

    def test_years_and_code_indexes_are_not_citations(self):
        g = ground_answer("In [2024] use items[0] and f(x)[1] [1].", ledger(1))
        assert g.answer == "In [2024] use items[0] and f(x)[1] [1]."
        assert g.cited == [1] and g.invalid == []
        g = ground_answer("Use items[0] as the key.", EvidenceLedger())
        assert g.answer == "Use items[0] as the key." and g.reason == "no_citations"

    def test_glued_and_chained_markers_are_citations(self):
        g = ground_answer("Acme ships to the Berlin plant[1].", ledger(1))
        assert g.grounded and g.cited == [1]
        g = ground_answer("Acme ships to Berlin [1][9].", ledger(1))
        assert g.answer == "Acme ships to Berlin [1]." and g.invalid == [9]

    @pytest.mark.parametrize(
        "answer", ["Acme was founded in 1998[2].", "Acme's (CEO)[2] is Jane.", "Berlin [1][2]."]
    )
    def test_marker_after_digit_or_paren_counts(self, answer):
        g = ground_answer(answer, ledger(2))
        assert g.grounded and 2 in g.cited

    def test_comma_separated_markers(self):
        g = ground_answer("Both [1, 2].", ledger(2))
        assert g.cited == [1, 2] and g.answer == "Both [1, 2]."

    @pytest.mark.parametrize("marker", ["[1-3]", "[1 - 3]", "[1–3]"])
    def test_range_markers_cite_every_number(self, marker):
        g = ground_answer(f"All three {marker}.", ledger(3))
        assert g.grounded and g.cited == [1, 2, 3] and g.answer == f"All three {marker}."

    def test_a_range_past_the_ledger_keeps_only_real_numbers(self):
        g = ground_answer("Some [2-4].", ledger(3))
        assert g.cited == [2, 3] and g.invalid == [4] and g.answer == "Some [2, 3]."

    @pytest.mark.parametrize("text", ["Rated [3-1] here [1].", "Pages [1-200] [1]."])
    def test_backwards_or_huge_ranges_are_not_citations(self, text):
        g = ground_answer(text, ledger(1))
        assert g.cited == [1] and g.invalid == [] and g.answer == text

    def test_semicolon_separated_markers(self):
        g = ground_answer("Both [1; 2].", ledger(2))
        assert g.cited == [1, 2] and g.answer == "Both [1, 2]."

    def test_markers_inside_code_are_untouched(self):
        g = ground_answer("Use `arr[9]` here [1].\n```\nx = y[7]\n```", ledger(1))
        assert g.answer == "Use `arr[9]` here [1].\n```\nx = y[7]\n```"
        assert g.invalid == []

    def test_tilde_fenced_code_is_untouched(self):
        g = ground_answer("Example [1]:\n~~~\nx = y[9]\n~~~", ledger(1))
        assert g.answer == "Example [1]:\n~~~\nx = y[9]\n~~~" and g.invalid == []

    def test_unclosed_fence_runs_to_the_end(self):
        g = ground_answer("See [1].\n```\nx = y[9]", ledger(1))
        assert g.answer == "See [1].\n```\nx = y[9]" and g.invalid == []

    def test_cleanup_keeps_formatting_where_nothing_was_removed(self):
        answer = "Steps:\n- top [1]\n    - nested  item\n```\n    code\n```"
        assert ground_answer(answer, ledger(1)).answer == answer

    @pytest.mark.parametrize(
        ("answer", "shown"),
        [
            ("Acme [1] is big [9].", "Acme [1] is big."),
            ("word [9] [1].", "word [1]."),
            ("a [9] b [1].", "a b [1]."),
            ("plant[9] next [1].", "plant next [1]."),
            ("x [9]\nnext [1].", "x\nnext [1]."),
            ("Berlin [1][9].", "Berlin [1]."),
        ],
    )
    def test_cleanup_around_removed_markers(self, answer, shown):
        assert ground_answer(answer, ledger(1)).answer == shown

    def test_citations_skip_numbers_missing_from_the_ledger(self):
        g = GroundedAnswer("x", True, "cited", cited=[1, 9])
        assert [e.n for e in g.citations(ledger(1))] == [1]


# ── claim checks ─────────────────────────────────────────────────


def contract() -> EvidenceLedger:
    led = EvidenceLedger()
    led.add(
        tool="retrieve",
        kind="passage",
        content="Corvane pays 2% of the order value per week of delay, capped at 20%. "
        "Total 1,200,000 EUR.",
        source="corvane_contract.pdf",
    )
    led.add(tool="graph_walk", kind="path", content="Sofia Marquez -[SIGNED]-> HG-2025-118")
    return led


class TestClaimChecks:
    def test_supported_answer_has_no_warnings(self):
        g = ground_answer("The penalty is 2% per week, capped at 20% [1].", contract())
        assert g.grounded and g.uncited == [] and g.unsupported == []

    def test_figure_missing_from_the_cited_evidence(self):
        g = ground_answer("The penalty is 2% per week, capped at 25% [1].", contract())
        assert g.grounded and [u.missing for u in g.unsupported] == [["25"]]

    def test_figure_is_checked_against_the_cited_item_only(self):
        g = ground_answer("Sofia Marquez signed it, capped at 20% [2].", contract())
        assert [u.missing for u in g.unsupported] == [["20"]]

    def test_name_missing_from_the_cited_evidence(self):
        g = ground_answer("The Kestrel sensors are capped at 20% [1].", contract())
        assert [u.missing for u in g.unsupported] == [["Kestrel"]]

    def test_a_two_word_name_is_found_by_either_word(self):
        g = ground_answer("Corvane Systems pays 2% a week [1].", contract())
        assert g.unsupported == []

    def test_a_longer_name_needs_half_of_its_words(self):
        g = ground_answer("Sofia Marquez Rivera signed HG-2025-118 [2].", contract())
        assert g.unsupported == []
        g = ground_answer("Sofia Alvarez Rivera signed HG-2025-118 [2].", contract())
        assert [u.missing for u in g.unsupported] == [["Sofia Alvarez Rivera"]]

    @pytest.mark.parametrize("name", ["Ann", "Kest", "Order Kest Systems"])
    def test_names_match_whole_words_only(self, name):
        # "ann" is inside "planning"/"annual", "kest" inside "Kestrel": not a match.
        led = EvidenceLedger()
        led.add(tool="retrieve", kind="passage", content="Annual planning at Kestrel; order value.")
        g = ground_answer(f"Then {name} approved it [1].", led)
        assert [u.missing for u in g.unsupported] == [[name]]

    def test_whole_words_are_case_insensitive_and_unicode(self):
        led = EvidenceLedger()
        led.add(tool="retrieve", kind="passage", content="JÜRGEN MÜLLER runs the Köln office.")
        g = ground_answer("Then Jürgen Müller approved it in Köln [1].", led)
        assert g.unsupported == []

    @pytest.mark.parametrize("figure", ["1,200,000", "1200000"])
    def test_thousands_separators_are_ignored(self, figure):
        assert ground_answer(f"It is {figure} EUR in total [1].", contract()).unsupported == []

    def test_a_figure_inside_a_longer_number_does_not_count(self):
        g = ground_answer("Version 2.8 is mentioned in it [1].", contract())
        assert [u.missing for u in g.unsupported] == [["2.8"]]

    def test_claim_without_its_own_citation_is_listed(self):
        answer = "Sofia Marquez signed HG-2025-118 [2]. Riverside lost power for 4 hours."
        g = ground_answer(answer, contract())
        assert g.grounded and g.uncited == ["Riverside lost power for 4 hours."]

    @pytest.mark.parametrize(
        "answer",
        [
            "Sofia signed it.[2] Corvane pays 2% a week. [1]",
            "- **Penalty:** 2% per week [1]\n- Cap: 20% [1]\n\nWant more detail?",
            "The penalty is 2% per week [1]. The evidence does not say more.",
            "Here is the code:\n```\nTotal = 99\n```\nIt is 2% per week [1].",
        ],
    )
    def test_markers_lists_questions_and_plain_sentences(self, answer):
        g = ground_answer(answer, contract())
        assert g.uncited == [] and g.unsupported == []

    def test_ungrounded_answer_skips_claim_checks(self):
        g = ground_answer("Riverside lost power for 4 hours.", contract())
        assert not g.grounded and g.uncited == [] and g.unsupported == []

    def test_flagged_claims_are_capped(self):
        answer = "Fact [1]. " + " ".join(f"Item {i} costs {i + 100} EUR." for i in range(30))
        assert len(ground_answer(answer, contract()).uncited) == 10


class TestLargeNumbersAndAbbreviations:
    def test_four_digit_citations_count_when_the_ledger_has_them(self):
        led = ledger(1001)
        g = ground_answer("Big fact [1000]. More [1, 1000].", led)
        assert g.grounded and g.cited == [1000, 1] and g.invalid == []

    def test_a_year_in_brackets_is_not_a_citation(self):
        g = ground_answer("The war ended [1945].", ledger(3))
        assert not g.grounded and g.reason == "no_citations" and g.invalid == []
        assert g.answer == "The war ended [1945]."
        cited = ground_answer("The war ended [1945] [1].", ledger(1001))
        assert cited.cited == [1] and cited.answer == "The war ended [1945] [1]."

    @pytest.mark.parametrize(
        "answer",
        [
            "Sales at Corvane Inc. were 2.8 billion [1].",
            "Sales grew 40 percent under Mr. Stone [1].",
            "Sales grew 40 percent, e.g. at Corvane [1].",
            "Sales grew 40 percent in the U.S. under Stone [1].",
        ],
    )
    def test_abbreviations_do_not_end_a_sentence(self, answer):
        led = EvidenceLedger()
        led.add(
            tool="retrieve",
            kind="fact",
            content="Corvane Inc U.S. sales were 2.8 billion, up 40 percent under Stone",
        )
        g = ground_answer(answer, led)
        assert g.grounded and g.uncited == [] and g.unsupported == []

    def test_sentences_still_split_after_a_plain_full_stop(self):
        led = EvidenceLedger()
        led.add(tool="retrieve", kind="fact", content="Bob works at Acme")
        g = ground_answer("Acme grew 40 percent. It hired Bob Stone [1].", led)
        assert g.uncited == ["Acme grew 40 percent."]

    def test_a_marker_after_an_abbreviation_closes_the_sentence(self):
        led = EvidenceLedger()
        led.add(tool="retrieve", kind="fact", content="Acme bought Corvane Inc")
        g = ground_answer("Acme bought Corvane Inc. [1] Bob Stone runs Zeta Labs.", led)
        assert g.uncited == ["Bob Stone runs Zeta Labs."] and g.unsupported == []

    def test_mixed_marker_checks_each_number_against_the_ledger(self):
        led = ledger(1001, capacity=1)
        g = ground_answer("Both facts [1000, 1001].", led)
        assert g.grounded and g.cited == [1001] and g.invalid == [1000]
        assert g.answer == "Both facts [1001]."

    @pytest.mark.parametrize(
        "answer", ["Acme ships widgets [0] [1].", "Acme ships widgets, pages [1-200] [1]."]
    )
    def test_non_citation_brackets_are_not_claimed_figures(self, answer):
        led = EvidenceLedger()
        led.add(tool="retrieve", kind="fact", content="Acme ships widgets")
        g = ground_answer(answer, led)
        assert g.grounded and g.unsupported == []

    def test_a_bracketed_year_is_still_a_claimed_figure(self):
        g = ground_answer("The war ended [1945] [1].", ledger(1))
        assert [u.missing for u in g.unsupported] == [["1945"]]
