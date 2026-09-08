"""Regression tests for the edge-remap queries used when merging duplicates.

These assert on query *shape* rather than behaviour, because the defect they
guard against is a planner choice, not a wrong answer. The old incoming-remap
query returned perfectly correct results; it just scanned every entity in the
graph to do it, so merge throughput fell from 61.9 to 10.3 merges/s between a
1K and a 50K node graph. No functional test can catch that.
"""

import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.storage.deduplicator import _REMAP_QUERIES, EntityDeduplicator


def _result(rows):
    r = MagicMock()
    r.result_set = rows
    return r


def _incoming_query() -> str:
    matches = [q for q in _REMAP_QUERIES if "MERGE (a)-[nr:RELATES]->" in q]
    assert len(matches) == 1, "expected exactly one incoming-RELATES remap query"
    return matches[0]


def test_incoming_remap_anchors_on_the_duplicate() -> None:
    """The duplicate must be matched (and thus indexed) before the traversal.

    Without the ``WITH dup`` barrier FalkorDB anchors the pattern on
    ``a:__Entity__`` and emits ``Node By Label Scan``. Reversing the arrow does
    not help -- only a separate MATCH ... WITH does.
    """
    q = _incoming_query()
    anchor = re.search(r"MATCH \(dup:__Entity__ \{id: \$dup_id\}\)\s*WITH dup", q)
    assert anchor, (
        "incoming remap must anchor on the indexed dup lookup before "
        "traversing, or every merge costs a full label scan:\n" + q
    )
    assert anchor.start() < q.index("MATCH (a:__Entity__)"), (
        "the dup anchor must come before the (a:__Entity__) traversal"
    )


@pytest.mark.parametrize("query", _REMAP_QUERIES)
def test_every_remap_query_looks_up_the_duplicate_by_indexed_id(query: str) -> None:
    """No remap query may start from a bare label scan over all entities."""
    first_match = query.index("MATCH ")
    opening = query[first_match:first_match + 60]
    assert "$dup_id" in opening or "$survivor_id" in opening, (
        "the first MATCH must be an id lookup so the index is used:\n" + query
    )


def test_relates_remaps_preserve_provenance() -> None:
    """Merging must union source_chunk_ids, never overwrite them.

    ``SET nr += properties(r)`` alone would replace the survivor edge's
    provenance with the duplicate's, which makes previously-rooted facts look
    unrooted to ``delete_stale_relationships``.
    """
    for q in _REMAP_QUERIES:
        if ":RELATES]->" not in q or "MERGE" not in q:
            continue
        assert "source_chunk_ids = old + [c IN contrib WHERE NOT c IN old]" in q, (
            "RELATES remap must union provenance rather than overwrite it:\n" + q
        )


# ── Name grouping ───────────────────────────────────────────────

from graphrag_sdk.storage.deduplicator import (  # noqa: E402
    is_acronym_of,
    normalize_entity_name,
)


class TestNormalizeEntityName:
    @pytest.mark.parametrize("a,b", [
        ("The University of Barcelona", "University of Barcelona"),
        ("Jardín Mecánico", "Jardin Mecanico"),
        ("Ayuntamiento de Calarosa.", "ayuntamiento de calarosa"),
        ("  Cape  Morrow   Light ", "Cape Morrow Light"),
    ])
    def test_variants_fold_together(self, a: str, b: str) -> None:
        assert normalize_entity_name(a) == normalize_entity_name(b)

    def test_generational_suffix_is_preserved(self) -> None:
        """A father and his son are different people, not a formatting variant."""
        assert (normalize_entity_name("Elias Whitford, Jr.")
                != normalize_entity_name("Elias Whitford"))

    def test_an_all_article_name_does_not_normalize_to_empty(self) -> None:
        assert normalize_entity_name("The") != ""

    @pytest.mark.parametrize("a,b", [
        ("東京", "大阪"),
        ("القاهرة", "بغداد"),
        ("東京", "Tokyo"),
    ])
    def test_non_latin_names_do_not_collapse_together(self, a: str, b: str) -> None:
        """ASCII folding leaves nothing of a CJK or Arabic name.

        An empty key would put every such entity of a label in one group and
        merge them all at finalize.
        """
        assert normalize_entity_name(a) != ""
        assert normalize_entity_name(a) != normalize_entity_name(b)

    def test_non_latin_name_still_folds_case_and_whitespace(self) -> None:
        assert normalize_entity_name("  東京  ") == normalize_entity_name("東京")


class TestIsAcronymOf:
    def test_matches_initials_skipping_stopwords(self) -> None:
        assert is_acronym_of("AIHS", "Ashford Island Historical Society")

    @pytest.mark.parametrize("short,long", [
        ("AIHS", "Ashford Island Historical Trust"),   # wrong final initial
        ("ABC", "Alpha"),                              # long form is one word
        ("A", "Alpha Beta"),                           # too short to be evidence
        ("VERYLONGACRONYM", "Very Long Acronym"),      # implausible length
        ("A1HS", "Ashford Island Historical Society"),  # not alphabetic
    ])
    def test_rejects_implausible_pairs(self, short: str, long: str) -> None:
        assert not is_acronym_of(short, long)


def _ent(eid: str, name: str, label: str, desc: str = "") -> dict:
    return {"id": eid, "name": name, "label": label, "description": desc}


class TestNameGrouping:
    def test_same_name_different_type_stays_separate(self) -> None:
        """Finalize-time exact match keys on (name, label).

        ``Paris``/Location and ``Paris``/Person are homographs, not duplicates.
        Deciding otherwise needs context, which is the ingest-time LLM
        resolver's job -- this pass deletes nodes without one.
        """
        groups = {
            ("paris", "location"): [_ent("1", "Paris", "Location")],
            ("paris", "person"): [_ent("2", "Paris", "Person")],
        }
        out = EntityDeduplicator._merge_acronym_groups(list(groups.values()))
        assert sorted(len(g) for g in out) == [1, 1]

    @pytest.mark.parametrize("a,b", [
        ("Los Angeles", "Angeles"),
        ("Le Mans", "Mans"),
        ("A.I.", "I"),
    ])
    def test_foreign_articles_and_dotted_acronyms_do_not_collide(
        self, a: str, b: str,
    ) -> None:
        assert normalize_entity_name(a) != normalize_entity_name(b)

    def test_dotted_acronym_folds_to_its_letters(self) -> None:
        assert normalize_entity_name("A.I.") == normalize_entity_name("AI")


class TestAcronymFold:
    def test_acronym_group_is_unioned(self) -> None:
        groups = [
            [_ent("1", "Ashford Island Historical Society", "Organization")],
            [_ent("2", "AIHS", "Organization")],
            [_ent("3", "Something Else", "Organization")],
        ]
        out = EntityDeduplicator._merge_acronym_groups(groups)
        sizes = sorted(len(g) for g in out)
        assert sizes == [1, 2], f"expected the acronym pair to union, got {sizes}"

    def test_result_is_order_independent(self) -> None:
        base = [
            [_ent("1", "Ashford Island Historical Society", "Organization")],
            [_ent("2", "AIHS", "Organization")],
            [_ent("3", "Bangor Maritime Trust", "Organization")],
            [_ent("4", "BMT", "Organization")],
        ]
        forward = EntityDeduplicator._merge_acronym_groups(base)
        reverse = EntityDeduplicator._merge_acronym_groups(list(reversed(base)))
        norm = lambda r: sorted(sorted(e["id"] for e in g) for g in r)  # noqa: E731
        assert norm(forward) == norm(reverse)

    def test_unrelated_short_names_do_not_chain(self) -> None:
        """The classic transitive-merge disaster: everything short collapsing."""
        groups = [[_ent(str(i), n, "Organization")] for i, n in
                  enumerate(["ABC", "XYZ", "QRS", "Alpha Beta Corp"])]
        out = EntityDeduplicator._merge_acronym_groups(groups)
        assert max(len(g) for g in out) <= 2

    def test_ambiguous_acronym_is_not_a_hub(self) -> None:
        """``US`` spells both United States and Universal Studios.

        Unioning through the short name would collapse the two long forms into
        one node and delete the other. An acronym with more than one plausible
        expansion is left alone for the LLM judge.
        """
        groups = [
            [_ent("1", "US", "Organization")],
            [_ent("2", "United States", "Organization")],
            [_ent("3", "Universal Studios", "Organization")],
        ]
        out = EntityDeduplicator._merge_acronym_groups(groups)
        assert sorted(len(g) for g in out) == [1, 1, 1]

    def test_acronym_requires_a_shared_label(self) -> None:
        groups = [
            [_ent("1", "AIHS", "Person")],
            [_ent("2", "Ashford Island Historical Society", "Organization")],
        ]
        out = EntityDeduplicator._merge_acronym_groups(groups)
        assert sorted(len(g) for g in out) == [1, 1]

    def test_acronym_shared_label_can_come_from_any_member(self) -> None:
        """A name-folded group carries every member's label."""
        groups = [
            [_ent("1", "AIHS", "Organization")],
            [_ent("2", "Ashford Island Historical Society", "Location"),
             _ent("3", "Ashford Island Historical Society", "Organization")],
        ]
        out = EntityDeduplicator._merge_acronym_groups(groups)
        assert sorted(len(g) for g in out) == [3]



class TestMergePreservesDescriptions:
    """NEW-1: merging must not delete the losing description.

    The survivor was chosen by longest description and the rest were
    ``DETACH DELETE``d, taking their text with them. Anything the loser said
    that the survivor did not was gone.
    """

    @staticmethod
    def _dedup(rows):
        graph = MagicMock()
        pages = [_result(rows), _result([])]

        async def query_raw(q, params=None):
            if "MATCH (e:__Entity__)" in q and "RETURN" in q and "SKIP" in q:
                return pages.pop(0) if pages else _result([])
            return _result([])

        graph.query_raw = AsyncMock(side_effect=query_raw)
        return EntityDeduplicator(graph, MagicMock()), graph

    @staticmethod
    def _description_writes(graph):
        return [
            c.args[1]["desc"]
            for c in graph.query_raw.call_args_list
            if len(c.args) > 1 and isinstance(c.args[1], dict) and "desc" in c.args[1]
        ]

    async def test_losing_description_is_preserved(self):
        dedup, graph = self._dedup([
            ["e1", "Cape Morrow Light", "a lighthouse on Cape Morrow", "Location"],
            ["e2", "cape morrow light", "first lit in 1871", "Location"],
        ])
        await dedup.deduplicate()
        writes = self._description_writes(graph)
        assert writes, "no description write was issued — the loser's text is gone"
        assert "first lit in 1871" in writes[0]
        assert "a lighthouse on Cape Morrow" in writes[0]

    async def test_same_name_different_label_is_not_merged(self):
        dedup, graph = self._dedup([
            ["e1", "Paris", "the capital of France", "Location"],
            ["e2", "Paris", "a prince of Troy", "Person"],
        ])
        merged = await dedup.deduplicate()
        assert merged == 0
        assert self._description_writes(graph) == []

    async def test_identical_descriptions_are_not_concatenated(self):
        dedup, graph = self._dedup([
            ["e1", "Alice", "an engineer", "Person"],
            ["e2", "alice", "an engineer", "Person"],
        ])
        await dedup.deduplicate()
        # The write still happens (it records the absorbed name as an alias),
        # but the description is not duplicated.
        for w in self._description_writes(graph):
            assert w == "an engineer"

    async def test_empty_description_does_not_produce_a_separator(self):
        dedup, graph = self._dedup([
            ["e1", "Bob", "a physicist", "Person"],
            ["e2", "bob", "", "Person"],
        ])
        await dedup.deduplicate()
        for w in self._description_writes(graph):
            assert not w.startswith(" | ") and not w.endswith(" | ")

