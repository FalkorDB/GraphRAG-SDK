"""Evidence cut from a chunk around spans (ingestion/evidence.py)."""

from graphrag_sdk.ingestion.evidence import (
    evidence_text,
    find_spans,
    flat_spans,
    is_whole_word,
)

TEXT = (
    "Northstar builds cabinets. Alice, CEO of Acme, announced the Q3 results. "
    "Revenue grew. She said Acme will hire.\n"
    "- Supplier: Apex Components Limited | GB | SUP-100\n\nEnd."
)


def test_each_mention_gives_its_sentence_and_far_apart_ones_are_joined():
    assert evidence_text(TEXT, find_spans(TEXT, "acme")) == (
        "Alice, CEO of Acme, announced the Q3 results. … She said Acme will hire."
    )


def test_a_table_row_is_cut_at_its_line():
    spans = find_spans(TEXT, "Apex Components Limited")
    assert evidence_text(TEXT, spans) == "- Supplier: Apex Components Limited | GB | SUP-100"


def test_spans_outside_the_text_give_nothing():
    assert evidence_text(TEXT, [(500, 510)]) == ""
    assert evidence_text(TEXT, []) == ""


def test_evidence_is_capped():
    text = "word " * 400
    out = evidence_text(text, [(0, 4)], max_chars=50)
    assert len(out) == 50 and out.endswith("…")


def test_flat_spans_is_one_list_of_ints():
    assert flat_spans([(1, 5), (9, 12)]) == [1, 5, 9, 12]
    assert flat_spans([]) == []


def test_without_a_sentence_boundary_the_window_starts_and_ends_on_a_word():
    # Header lines with no full stops: the window reaches its limit inside a
    # word on both sides, and moves to the nearest whole word.
    text = "Notification: " * 30 + "QN-2027 raised by Northstar Quality " + "Subject: " * 30
    out = evidence_text(text, find_spans(text, "QN-2027"))
    assert out.split()[0] == "Notification:"
    assert out.split()[-1] == "Subject:"


TABLE = (
    "Inspection results for the lot:\n\n"
    "| Step | Sampled | Result |\n"
    "|---|---|---|\n"
    "| Automated optical inspection | 5 of 24 | pass |\n"
    "| Sampling inspection | 8 of 8 | fail |\n"
)


def test_a_table_row_carries_its_header():
    spans = find_spans(TABLE, "Sampling inspection")
    assert evidence_text(TABLE, spans) == (
        "| Step | Sampled | Result | … | Sampling inspection | 8 of 8 | fail |"
    )


def test_the_header_row_is_not_given_twice():
    spans = find_spans(TABLE, "Step")
    assert evidence_text(TABLE, spans) == "| Step | Sampled | Result |"


def test_a_pipe_line_without_separator_is_not_a_table():
    text = "| just a line | with pipes |\n| another | line |\n"
    assert evidence_text(text, find_spans(text, "another")) == "| another | line |"


def test_a_line_wrapped_mid_sentence_is_one_sentence():
    text = (
        "Apex certifies that lot BAT-1 was manufactured in\n"
        "accordance with the specification and that sampling inspection\n"
        "was performed to the contracted plan. Sample size 5 boards.\n"
    )
    assert evidence_text(text, find_spans(text, "sampling inspection")) == (
        "Apex certifies that lot BAT-1 was manufactured in accordance with the "
        "specification and that sampling inspection was performed to the contracted plan."
    )


def test_a_list_item_ends_at_its_line():
    text = "Holds:\n- Lot A held at stores\n- Lot B released\n"
    assert evidence_text(text, find_spans(text, "Lot B")) == "- Lot B released"


def test_a_name_is_found_only_as_a_whole_name():
    text = "US exports rose. Ann announced business with Al. Algorithms helped the US."
    assert find_spans(text, "US") == [(0, 2), (71, 73)]
    assert find_spans(text, "Al") == [(45, 47)]
    assert find_spans(text, "Ann") == [(17, 20)]


def test_a_name_in_capitals_is_never_found_all_lowercase():
    assert find_spans("They told us to go. The US agreed.", "US") == [(24, 26)]
    assert find_spans("they told us to go.", "US") == []
    # Other cases still count: "NASA" written "Nasa".
    assert find_spans("Nasa launched it.", "NASA") == [(0, 4)]


def test_a_name_is_not_found_inside_a_longer_word_in_any_script():
    # A combining mark is part of the word: a vowel sign, an accent written apart.
    assert find_spans("भारतीय रेल", "भारत") == []
    assert find_spans("Cafe\u0301 de Flore", "Cafe") == []
    # A name that starts or ends in punctuation.
    assert find_spans("ASP.NET runs on .NET", ".NET") == [(16, 20)]
    assert find_spans("C++11 extends C++.", "C++") == [(14, 17)]
    # Turkish dotless ı is not I.
    assert find_spans("ıSTANBUL and Istanbul", "Istanbul") == [(13, 21)]


def test_a_korean_name_takes_particles_after_it_but_nothing_before():
    assert find_spans("서울에서 열렸다", "서울") == [(0, 2)]
    assert find_spans("대서울", "서울") == []


def test_occurrences_found_never_overlap():
    assert find_spans("哈哈哈", "哈哈") == [(0, 2)]
    # Of two that overlap, the one written as the name is kept.
    assert find_spans("a.A.A", "A.A") == [(2, 5)]


def test_a_capitalised_names_lowercase_occurrences_are_left_out():
    text = "Apple sells apple juice. APPLE stock rose."
    # "apple" the fruit is not the company; "APPLE" in a heading still is.
    assert find_spans(text, "Apple") == [(0, 5), (25, 30)]
    # Only lowercase ones: those are the name.
    assert find_spans("alice runs Acme.", "Alice") == [(0, 5)]
    # A lowercase name: every case.
    assert find_spans(text, "apple") == [(0, 5), (12, 17), (25, 30)]


def test_a_name_in_a_script_without_spaces_is_found_inside_a_run():
    assert find_spans("公司在杭州成立。", "杭州") == [(3, 5)]
    assert find_spans("東京都に本社がある", "東京") == [(0, 2)]
    assert find_spans("บริษัทในกรุงเทพฯ", "กรุงเทพ") == [(8, 15)]
    # A Latin name inside such a run is a whole word too.
    assert find_spans("在Apple公司工作", "Apple") == [(1, 6)]


def test_a_models_own_offsets_count_only_on_a_whole_word():
    text = "Our business is booming in the US."
    assert not is_whole_word(text, 5, 7)  # "us" inside "business"
    assert is_whole_word(text, 31, 33)
    assert not is_whole_word(text, 31, 99)  # past the end
    assert is_whole_word("公司在杭州成立", 3, 5)


def test_a_span_given_twice_is_kept_once():
    assert flat_spans([(1, 5), (1, 5), (9, 12)]) == [1, 5, 9, 12]


def test_a_zero_cap_gives_nothing():
    assert evidence_text("Alice runs Acme.", [(0, 5)], max_chars=0) == ""
