"""Evidence cut from a chunk around spans (ingestion/evidence.py)."""

from graphrag_sdk.ingestion.evidence import (
    evidence_text,
    find_spans,
    flat_spans,
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
