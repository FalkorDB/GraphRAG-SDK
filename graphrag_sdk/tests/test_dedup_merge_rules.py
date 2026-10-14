"""Merge rules every dedup phase shares.

A mention two keyed rows could equally own stays apart; a merge unions the
duplicate's provenance and keeps both descriptions.
"""

from __future__ import annotations

from graphrag_sdk.storage.deduplicator import (
    _mentions_two_rows_could_own,
    properties_to_carry,
    union_chunk_ids,
)


class TestMentionsTwoRowsCouldOwn:
    ROW_A = {"id": "e-1__person", "name": "John Smith", "label": "Person", "is_stub": False}
    ROW_B = {"id": "e-7__person", "name": "John Smith", "label": "Person", "is_stub": False}
    NOTE = {"id": "john_smith__person", "name": "John Smith", "label": "Person", "is_stub": None}

    def test_a_mention_beside_two_rows_is_paired_with_each_and_kept(self):
        found = _mentions_two_rows_could_own(self.ROW_A, [self.ROW_A, self.ROW_B, self.NOTE])

        assert sorted((m.id_a, m.id_b) for m in found) == [
            ("john_smith__person", "e-1__person"),
            ("john_smith__person", "e-7__person"),
        ]
        assert all(m.bridges_a_declared_source for m in found)
        assert "2 keyed rows" in found[0].reason

    def test_one_row_is_the_join_the_exact_phase_exists_for(self):
        assert _mentions_two_rows_could_own(self.ROW_A, [self.ROW_A, self.NOTE]) == []

    def test_a_placeholder_is_a_declared_identity_too(self):
        placeholder = {**self.ROW_B, "is_stub": True}
        found = _mentions_two_rows_could_own(self.ROW_A, [self.ROW_A, placeholder, self.NOTE])
        assert len(found) == 2

    def test_two_rows_and_no_mention_is_nothing_to_report(self):
        assert _mentions_two_rows_could_own(self.ROW_A, [self.ROW_A, self.ROW_B]) == []


class TestUnionChunkIds:
    def test_keeps_order_and_adds_only_what_is_missing(self):
        assert union_chunk_ids(["c1", "c2"], ["c2", "c3", "c1"]) == ["c1", "c2", "c3"]

    def test_anything_that_is_not_a_list_of_strings_counts_as_empty(self):
        assert union_chunk_ids(None, ["c1"]) == ["c1"]
        assert union_chunk_ids(["c1"], "c2") == ["c1"]
        assert union_chunk_ids(["c1", 7, None], [3]) == ["c1"]
        assert union_chunk_ids(None, None) == []


class TestWhatAMergeKeeps:
    """``properties_to_carry``: the survivor's value wins, except the description.

    The survivor is the keyed or better-connected node, which is usually the row —
    and a row's description is a template line where the prose entity's is the
    paragraph the reader wanted. The two are joined so neither is lost; nothing
    else changes.
    """

    NEVER = frozenset({"id", "embedding"})

    def test_both_descriptions_are_kept_whichever_node_had_which(self):
        keep = {"id": "a", "description": "An engineer.", "age": 34}
        dup = {"id": "b", "description": "An engineer who led the remediation plan in 2024."}
        carried = properties_to_carry(keep, dup, never=self.NEVER)
        assert carried == {"description": f"{keep['description']} | {dup['description']}"}

        carried = properties_to_carry(dup, keep, never=self.NEVER)
        assert carried == {
            "age": 34,
            "description": f"{dup['description']} | {keep['description']}",
        }, "the survivor's own text comes first"

    def test_an_identical_description_is_not_repeated(self):
        keep = {"id": "a", "description": "An engineer."}
        dup = {"id": "b", "description": "An engineer."}
        assert properties_to_carry(keep, dup, never=self.NEVER) == {}

    def test_every_other_value_on_the_survivor_still_wins(self):
        keep = {"id": "a", "name": "Maya Ellison", "age": 34, "embedding": [0.1]}
        dup = {"id": "b", "name": "M. Ellison", "age": 35, "embedding": [0.2], "role": "CTO"}
        assert properties_to_carry(keep, dup, never=self.NEVER) == {"role": "CTO"}

    def test_a_blank_description_on_the_survivor_is_filled(self):
        keep = {"id": "a", "description": ""}
        dup = {"id": "b", "description": "Something."}
        assert properties_to_carry(keep, dup, never=self.NEVER) == {"description": "Something."}
