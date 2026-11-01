"""The exact phase's rules and the merges it writes, without a database.

Name normalisation and the acronym fold that group entities by name; merges
simulated through ``dedup_sim`` (descriptions, aliases, labels); the merge
query's shape, for what a functional test cannot see — the moves before the
delete, the fold and the delete under one guard; and how a merge whose reply
was lost reports what the earlier attempt did. The merge query itself runs on
a real FalkorDB in ``test_dedup_records``.
"""

import pytest

from graphrag_sdk.storage.deduplicator import (
    EntityDeduplicator,
    _acronym_key,
    _expansion_key,
    normalize_entity_name,
)
from graphrag_sdk.storage.merge_query import absorb_query

from .dedup_sim import AllSameJudge, simulate

# ── Name grouping ───────────────────────────────────────────────


def is_acronym_of(short: str, long: str) -> bool:
    """The rule the exact phase's acronym fold applies to one pair."""
    key = _acronym_key(normalize_entity_name(short))
    return key is not None and key == _expansion_key(normalize_entity_name(long))


class TestNormalizeEntityName:
    @pytest.mark.parametrize(
        "a,b",
        [
            ("The University of Barcelona", "University of Barcelona"),
            ("Jardín Mecánico", "Jardin Mecanico"),
            ("Ayuntamiento de Calarosa.", "ayuntamiento de calarosa"),
            ("  Cape  Morrow   Light ", "Cape Morrow Light"),
        ],
    )
    def test_variants_fold_together(self, a: str, b: str) -> None:
        assert normalize_entity_name(a) == normalize_entity_name(b)

    def test_generational_suffix_is_preserved(self) -> None:
        """A father and his son are different people, not a formatting variant."""
        assert normalize_entity_name("Elias Whitford, Jr.") != normalize_entity_name(
            "Elias Whitford"
        )

    def test_an_all_article_name_does_not_normalize_to_empty(self) -> None:
        assert normalize_entity_name("The") != ""

    @pytest.mark.parametrize(
        "a,b",
        [
            ("東京", "大阪"),
            ("القاهرة", "بغداد"),
            ("東京", "Tokyo"),
        ],
    )
    def test_non_latin_names_do_not_collapse_together(self, a: str, b: str) -> None:
        """ASCII folding leaves nothing of a CJK or Arabic name.

        An empty key would put every such entity of a label in one group and
        merge them all at finalize.
        """
        assert normalize_entity_name(a) != ""
        assert normalize_entity_name(a) != normalize_entity_name(b)

    def test_non_latin_name_still_folds_case_and_whitespace(self) -> None:
        assert normalize_entity_name("  東京  ") == normalize_entity_name("東京")

    @pytest.mark.parametrize(
        "a,b",
        [
            ("Отдел 5", "Кабинет 5"),  # ASCII fold left only "5"
            ("東京, Japan", "大阪, Japan"),  # ... only "japan"
            ("Школа №1", "Гимназия №1"),  # NFKD maps № -> "No"
            ("北京 University", "上海 University"),
        ],
    )
    def test_mixed_script_names_keep_their_non_latin_part(self, a: str, b: str) -> None:
        """One surviving ASCII token must not become the whole grouping key.

        Folding to ASCII dropped every non-Latin letter, so these same-label
        pairs collapsed onto their shared remnant and one of each was deleted.
        """
        assert normalize_entity_name(a) != normalize_entity_name(b)

    def test_accents_still_fold_after_keeping_other_scripts(self) -> None:
        assert normalize_entity_name("Отдел Пять") == normalize_entity_name("отдел  пять")
        assert normalize_entity_name("Café Müller") == normalize_entity_name("cafe muller")


class TestIsAcronymOf:
    def test_matches_initials_skipping_stopwords(self) -> None:
        assert is_acronym_of("AIHS", "Ashford Island Historical Society")

    @pytest.mark.parametrize(
        "short,long",
        [
            ("AIHS", "Ashford Island Historical Trust"),  # wrong final initial
            ("ABC", "Alpha"),  # long form is one word
            ("A", "Alpha Beta"),  # too short to be evidence
            ("VERYLONGACRONYM", "Very Long Acronym"),  # implausible length
            ("A1HS", "Ashford Island Historical Society"),  # not alphabetic
        ],
    )
    def test_rejects_implausible_pairs(self, short: str, long: str) -> None:
        assert not is_acronym_of(short, long)

    @pytest.mark.parametrize(
        "short,long",
        [
            ("Bo", "Ben Ottoson"),
            ("Al", "Anna Lindqvist"),
            ("Jo", "Jane Oliver"),
        ],
    )
    def test_two_letter_names_are_not_acronym_evidence(self, short: str, long: str) -> None:
        """Any two-letter name spells the initials of some two-word name.

        With a floor of 2 these Person pairs were merged -- one node deleted --
        with no LLM in the loop. Two letters carry too little signal.
        """
        assert not is_acronym_of(short, long)


def _ent(eid: str, name: str, label: str, desc: str = "") -> dict:
    return {"id": eid, "name": name, "label": label, "description": desc}


class TestNameGrouping:
    def test_same_name_different_type_stays_separate(self) -> None:
        """Finalize-time exact match keys on (name, label).

        ``Paris``/Location and ``Paris``/Person are homographs, not duplicates.
        Deciding otherwise needs context, which is the LLM judge's job at
        finalize() -- this pass deletes nodes without one.
        """
        groups = {
            ("paris", "location"): [_ent("1", "Paris", "Location")],
            ("paris", "person"): [_ent("2", "Paris", "Person")],
        }
        out = EntityDeduplicator._merge_acronym_groups(list(groups.values()))
        assert sorted(len(g) for g in out) == [1, 1]

    @pytest.mark.parametrize(
        "a,b",
        [
            ("Los Angeles", "Angeles"),
            ("Le Mans", "Mans"),
            ("A.I.", "I"),
        ],
    )
    def test_foreign_articles_and_dotted_acronyms_do_not_collide(
        self,
        a: str,
        b: str,
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
        groups = [
            [_ent(str(i), n, "Organization")]
            for i, n in enumerate(["ABC", "XYZ", "QRS", "Alpha Beta Corp"])
        ]
        out = EntityDeduplicator._merge_acronym_groups(groups)
        assert max(len(g) for g in out) <= 2

    def test_ambiguous_acronym_is_not_a_hub(self) -> None:
        """``ABC`` spells both American Broadcasting Company and Australian
        Broadcasting Corporation.

        Unioning through the short name would collapse the two long forms into
        one node and delete the other. An acronym with more than one plausible
        expansion is left alone for the LLM judge.
        """
        groups = [
            [_ent("1", "ABC", "Organization")],
            [_ent("2", "American Broadcasting Company", "Organization")],
            [_ent("3", "Australian Broadcasting Corporation", "Organization")],
        ]
        out = EntityDeduplicator._merge_acronym_groups(groups)
        assert sorted(len(g) for g in out) == [1, 1, 1]

    def test_two_letter_short_form_is_never_unioned(self) -> None:
        """Even with exactly one same-label candidate, ``Bo`` is not ``Ben Ottoson``."""
        groups = [
            [_ent("1", "Bo", "Person")],
            [_ent("2", "Ben Ottoson", "Person")],
        ]
        out = EntityDeduplicator._merge_acronym_groups(groups)
        assert sorted(len(g) for g in out) == [1, 1]

    def test_fold_is_linear_in_the_number_of_groups(self, monkeypatch) -> None:
        """Long forms are bucketed by initials; each short form is one lookup.

        The pairwise version normalised both names for every (short, long)
        pair -- quadratic inside ``finalize()``. Counting normalisation calls
        pins the linear shape without timing anything.
        """
        import graphrag_sdk.storage.deduplicator as mod

        calls = {"n": 0}
        real = mod.normalize_entity_name

        def counting(name: str) -> str:
            calls["n"] += 1
            return real(name)

        monkeypatch.setattr(mod, "normalize_entity_name", counting)
        shorts = [[_ent(f"s{i}", f"S{i:03d}", "Organization")] for i in range(200)]
        longs = [[_ent(f"l{i}", f"Long Form Number {i}", "Organization")] for i in range(200)]
        EntityDeduplicator._merge_acronym_groups(shorts + longs)
        assert calls["n"] <= 400, calls["n"]

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
            [
                _ent("2", "Ashford Island Historical Society", "Location"),
                _ent("3", "Ashford Island Historical Society", "Organization"),
            ],
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
    def _description_writes(sim):
        return [w.params["desc"] for w in sim.writes if "desc" in w.params]

    async def test_losing_description_is_preserved(self):
        sim = await simulate(
            [
                ["e1", "Cape Morrow Light", "a lighthouse on Cape Morrow", "Location"],
                ["e2", "cape morrow light", "first lit in 1871", "Location"],
            ],
            judge=AllSameJudge(),
        )
        assert sim.merged == 1
        writes = self._description_writes(sim)
        assert writes, "no description write was issued — the loser's text is gone"
        assert "first lit in 1871" in writes[0]
        assert "a lighthouse on Cape Morrow" in writes[0]

    async def test_same_name_different_label_is_not_merged_by_rule(self):
        sim = await simulate(
            [
                ["e1", "Paris", "the capital of France", "Location", None, False],
                ["e2", "Paris", "a prince of Troy", "Person"],
            ]
        )
        assert sim.merged == 0 and sim.ops == []

    async def test_empty_description_does_not_produce_a_separator(self):
        sim = await simulate(
            [
                ["e1", "Bob", "a physicist", "Person"],
                ["e2", "bob", "", "Person"],
            ],
            judge=AllSameJudge(),
        )
        assert self._description_writes(sim) == ["a physicist"]

    async def test_existing_segments_are_not_repeated(self):
        """Dedup compares members, not whole strings.

        Once a survivor holds ``["a lighthouse", "first lit in 1871"]``,
        absorbing a node described as ``"first lit in 1871"`` used to append
        it again. Across incremental finalize cycles that compounds on hub
        entities. The members are the survivor's stored ``descriptions``
        list, the shape every merge writes.
        """
        sim = await simulate(
            [
                ["e1", "Cape Morrow Light", "a lighthouse | first lit in 1871", "Location"],
                ["e2", "cape morrow light", "first lit in 1871", "Location"],
            ],
            stored={"e1": {"props": {"descriptions": ["a lighthouse", "first lit in 1871"]}}},
            judge=AllSameJudge(),
        )
        assert sim.merged == 1
        assert self._description_writes(sim) == []

    async def test_segments_accumulate_once_across_a_group(self):
        sim = await simulate(
            [
                ["e1", "Cape Morrow Light", "a lighthouse", "Location"],
                ["e2", "cape morrow light", "first lit in 1871", "Location"],
                ["e3", "Cape  Morrow  Light", "first lit in 1871 | automated in 1960", "Location"],
            ],
            stored={"e3": {"props": {"descriptions": ["first lit in 1871", "automated in 1960"]}}},
            judge=AllSameJudge(),
        )
        # e3 has the longest description and survives; the others fold in
        # once, each merge building on the one before.
        assert sim.nodes["e3"]["props"]["description"] == (
            "first lit in 1871 | automated in 1960 | a lighthouse"
        )
        assert set(sim.nodes) == {"e3"}

    def test_the_fold_and_the_delete_are_one_statement_after_every_edge_moved(self):
        """A failed description write must not leave the duplicate deleted,
        and no edge may be deleted with it: the fold, the delete and every
        edge move are one query, the moves before the delete, the fold and
        the delete under the same guard."""
        q = absorb_query([("RELATES", "in"), ("MENTIONED_IN", "out")], ["s.desc = $d"], [])
        assert (
            q.count("MERGE (b)-[nr:`RELATES`") == 1 and q.count("MERGE (s)-[nr:`MENTIONED_IN`") == 1
        )
        assert q.rindex("MERGE ") < q.index("SET s.desc = $d") < q.index("DETACH DELETE d")
        assert "CASE WHEN ok THEN [1] ELSE [] END | DETACH DELETE d" in q

    @pytest.mark.parametrize("direction", ["in", "out"])
    def test_relates_moves_union_relationship_table_ownership(self, direction):
        """The relationship tables owning an edge are provenance too: left to
        ``SET nr += properties(e)`` the survivor's edge kept only the
        duplicate's tables, so dropping one of those deleted an edge another,
        forgotten, table still lists."""
        q = absorb_query([("RELATES", direction)], [], [])
        assert "coalesce(ex.structured_sources, []) AS old_sources" in q
        assert (
            "ELSE old_sources + [x IN coalesce(e.structured_sources, []) "
            "WHERE NOT x IN old_sources] END"
        ) in q

    def test_node_provenance_is_unioned_before_deletion(self):
        """Entity nodes carry ``source_chunk_ids`` read back by
        ``CachedChunkExtraction``; deleting the duplicate outright dropped
        its half. The union is done by the merge query from the values at
        write time."""
        q = absorb_query([], [], [])
        assert "SET s.source_chunk_ids" in q
        assert "[c IN d.source_chunk_ids WHERE NOT c IN coalesce(s.source_chunk_ids, [])]" in q
        assert q.index("SET s.source_chunk_ids") < q.index("DETACH DELETE d")


class TestAcronymMerge:
    async def test_long_form_survives_regardless_of_description_length(self):
        """The *name* is what the entity is vector-indexed by. Keeping
        ``AIHS`` would leave the entity indexed as an opaque string."""
        sim = await simulate(
            [
                [
                    "a",
                    "AIHS",
                    "a much longer description that used to win the survivor slot",
                    "Organization",
                ],
                ["b", "Ashford Island Historical Society", "short", "Organization"],
            ],
            judge=AllSameJudge(),
        )
        assert sim.merged == 1
        (write,) = sim.writes
        assert (write.op.dup, write.op.survivor) == ("a", "b")

    async def test_absorbed_acronym_is_recorded_as_an_alias(self):
        sim = await simulate(
            [
                ["a", "AIHS", "", "Organization"],
                ["b", "Ashford Island Historical Society", "", "Organization"],
            ],
            judge=AllSameJudge(),
        )
        (write,) = sim.writes
        assert "s.aliases = $aliases" in write.sets
        assert write.params["aliases"] == ["AIHS"]

    async def test_spelling_variant_is_not_an_alias(self):
        sim = await simulate(
            [
                ["e1", "Cape Morrow Light", "", "Location"],
                ["e2", "cape morrow light", "", "Location"],
            ],
            judge=AllSameJudge(),
        )
        (write,) = sim.writes
        assert "aliases" not in write.params

    async def test_aliases_accumulate_without_duplicates(self):
        sim = await simulate(
            [
                ["b", "Ashford Island Historical Society", "", "Organization", ["A.I.H.S."]],
                ["a", "AIHS", "", "Organization", ["Ashford Society"]],
            ],
            judge=AllSameJudge(),
        )
        (write,) = sim.writes
        assert write.params["aliases"] == ["A.I.H.S.", "AIHS", "Ashford Society"]


def test_decimals_read_for_a_merge_are_compared_within_a_tolerance():
    """FalkorDB sends decimals to the client rounded to 15 digits, so a value
    read and compared back with ``=`` never matched and the merge was stale
    forever. Decimals, and lists holding one, go to their own checks."""
    from graphrag_sdk.storage.merge_query import MergeRead, snapshot_params

    read = MergeRead(
        keep={"score": 0.3, "rank": 2, "tags": ["a"]},
        dup={"weight": 0.3, "series": [0.3, 1], "label": "x"},
        keep_labels=[],
        dup_labels=[],
        kinds=[],
    )
    params = snapshot_params(
        read, {"weight": 0.3, "series": [0.3, 1], "label": "x"}, ["score", "rank", "tags"]
    )
    assert params["s_seen"] == {"rank": 2, "tags": ["a"]}
    assert params["s_near"] == {"score": 0.3} and params["s_near_list"] == {}
    assert params["d_seen"] == {"label": "x"}
    assert params["d_near"] == {"weight": 0.3}
    assert params["d_near_list"] == {"series": [0.3, 1]}
    q = absorb_query([], [], [])
    for side in ("s", "d"):
        assert f"WHERE coalesce(abs((CASE WHEN typeOf({side}[k]) IN" in q
        assert f"all(k IN keys(${side}_near_list) WHERE size(" in q
    # FalkorDB evaluates every CASE branch: no branch may do arithmetic.
    assert "THEN s[k] -" not in q and "THEN d[k] -" not in q


def test_an_operation_an_earlier_attempt_ticked_says_what_that_attempt_did():
    """A merge whose reply was lost comes back, through the connection's
    retry, as "not my turn". The plan page says whether the earlier attempt
    applied it or skipped it as changed."""
    from graphrag_sdk.storage.merge_query import (
        ALREADY_APPLIED,
        ALREADY_CHANGED,
        APPLIED,
        PlanStateError,
        outcome,
    )

    # owned, plan_found, next, my_turn, present, fresh, unhandled, was_changed
    assert outcome([True, True, 6, False, None, None, None, False], 5) == ALREADY_APPLIED
    assert outcome([True, True, 6, False, None, None, None, True], 5) == ALREADY_CHANGED
    assert outcome([True, True, 5, True, True, True, 0, False], 5) == APPLIED
    with pytest.raises(PlanStateError):
        outcome([True, True, 4, False, None, None, None, False], 5)


class TestChangedGroupsStayPending:
    """A merge skipped as changed leaves its group's checks incomplete. Members
    no pending record lists — every entity of a ``dedup_all=True`` run, a
    survivor checked in an earlier run — have to be listed, or they look
    checked and no ordinary run retries them."""

    def test_members_with_no_record_get_one(self):
        from graphrag_sdk.storage.deduplicator import _decide_records
        from graphrag_sdk.storage.system_store import Decisions

        updates, created, cleared = _decide_records(
            [], Decisions(clear=["s", "d"]), recheck={"s", "d"}
        )
        assert updates == [] and cleared == 0
        assert created.ids == ["d", "s"] and created.reasons == ["changed during the run"] * 2

    def test_a_member_already_listed_is_kept_and_the_others_join_its_record(self):
        from graphrag_sdk.storage.deduplicator import _decide_records
        from graphrag_sdk.storage.system_store import Decisions, PendingPage

        listed = PendingPage(0, ["d"], [1], [""])
        updates, created, _ = _decide_records(
            [listed], Decisions(clear=["s", "d"]), recheck={"s", "d"}
        )
        assert created is None
        assert [(u.ids, u.attempts, u.reasons) for u in updates] == [
            (["d", "s"], [1, 0], ["changed during the run", "changed during the run"])
        ]
