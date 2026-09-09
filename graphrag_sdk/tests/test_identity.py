"""Near-miss identity: finding the pairs exact matching cannot, and acting on them.

The join between a table row and a prose mention is exact string equality on the
display name. Real sources disagree about spelling, so the interesting failure is
not a wrong merge — it is two nodes that should have been one, with no warning.
"""

from __future__ import annotations

import logging

import pytest

from graphrag_sdk import Column, Entity, ExactMatchResolution, Ontology, TableMapping
from graphrag_sdk.storage.deduplicator import _survivor_rank
from graphrag_sdk.storage.identity import canonical_key, find_near_misses, why_same

# The mapping is part of the ontology now, declared once and found by ``ingest``
# on the file's basename — so the file each test writes must be named for it.
_HR = TableMapping(
    source="hr.csv",
    label="Person",
    key="employee_id",
    name="full_name",
    properties={"age": Column("age", "INTEGER")},
)
_ORGS = TableMapping(
    source="orgs.csv",
    label="Organization",
    key="org_id",
    name="org_name",
    properties={"employee_count": Column("employee_count", "INTEGER")},
)


@pytest.fixture
def resolver():
    return ExactMatchResolution(resolve_property="name")


class TestTheRules:
    """Tuned for recall: a missed pair is a silent wrong answer, a surfaced one
    costs a reader a line to dismiss. Nothing here merges, so that trade is free.
    """

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Maya Ellison", "M. Ellison"),
            ("Tomas Reyes", "T. Reyes"),
            ("J. R. R. Tolkien", "John Ronald Reuel Tolkien"),
            ("Northwind Energy", "Northwind Energy AS"),
            ("Kestrel Grid", "Kestrel Grid Ltd."),
            ("Globex Ltd", "Globex Limited"),
            ("Acme Corporation", "Acme Corp"),
            ("Jean-Luc Picard", "Jean Luc Picard"),
            ("Priya Raman", "Raman, Priya"),
        ],
    )
    def test_pairs_that_must_be_surfaced(self, a, b):
        assert why_same(a, b), f"{a!r} ~ {b!r} would have been missed"
        assert why_same(b, a), "the rule must not depend on argument order"

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Maya Ellison", "Maya Ellis"),
            ("John Smith", "Jane Smith"),
            ("Northwind Energy", "Southwind Energy"),
            ("Acme Corporation", "Acme Industries"),
            ("Tomas Reyes", "Tomas Reyna"),
            ("Kestrel Grid", "Kestrel Power"),
            ("Priya Raman", "Priya Ramesh"),
            ("Bank of Norway", "Bank of Sweden"),
            ("Maya Ellison", "Ellison"),
        ],
    )
    def test_pairs_that_must_not_be(self, a, b):
        assert why_same(a, b) is None, f"{a!r} ~ {b!r} was wrongly surfaced"

    def test_an_exact_match_is_not_a_near_miss(self):
        """Exact matching already merged it; reporting it again is noise."""
        assert why_same("Maya Ellison", "maya ellison") is None

    def test_a_legal_form_is_not_stripped_to_nothing(self):
        """'Group' is noise in 'Kestrel Group' and the whole name in 'Group'."""
        assert why_same("Group", "Holdings") is None

    def test_a_row_meeting_prose_is_ranked_first(self):
        """That pair is what this feature exists for, so it leads the report."""
        found = find_near_misses(
            [
                {"id": "a", "name": "Kestrel Grid", "label": "Organization", "is_stub": None},
                {"id": "b", "name": "Kestrel Grid Ltd", "label": "Organization", "is_stub": None},
                {"id": "e-1__person", "name": "Maya Ellison", "label": "Person", "is_stub": False},
                {"id": "m__person", "name": "M. Ellison", "label": "Person", "is_stub": None},
            ]
        )
        assert len(found) == 2
        assert found[0].bridges_a_declared_source
        assert found[0].label == "Person"

    def test_different_labels_are_not_compared(self):
        """Across labels is a different problem with its own report."""
        assert not find_near_misses(
            [
                {"id": "a", "name": "Acme Corp", "label": "Organization", "is_stub": None},
                {"id": "b", "name": "Acme Corporation", "label": "Product", "is_stub": None},
            ]
        )


class TestTheScanIsBlockedNotQuadratic:
    """``find_near_misses`` compares only names that share a comparable form.

    The all-against-all scan took 12 s at 1k entities under one label and gave
    up at 5k with a warning; ``finalize()`` ran it twice. Blocking must find the
    same pairs, so the check here is against :func:`why_same` applied to every
    pair — on a set built to exercise all three grounds it accepts on.
    """

    @staticmethod
    def _entities() -> list[dict]:
        import itertools
        import random

        rng = random.Random(7)
        first = ["Maya", "Marta", "Tomas", "Tobias", "Priya", "Jean-Luc", "J.", "John"]
        last = ["Ellison", "Reyes", "Raman", "Picard", "Holm", "Nguyen", "Tolkien"]
        orgs = ["Northwind Energy", "Kestrel Grid", "Globex", "Acme", "Initech"]
        forms = ["", " AS", " Ltd", " Ltd.", " Corp", " Corporation", " Limited", " LLC"]
        names = set()
        for a, b in itertools.product(first, last):
            names.add(f"{a} {b}")
            names.add(f"{b}, {a}")
            names.add(f"{a[0]}. {b}")
        for org, form in itertools.product(orgs, forms):
            names.add(org + form)
        for n in range(400):
            names.add(f"{rng.choice(first)} {rng.choice(last)} {n}")
        return [
            {"id": f"e{i}", "name": name, "label": "Thing", "is_stub": i % 5 == 0}
            for i, name in enumerate(sorted(names))
        ]

    def test_blocking_finds_every_pair_the_pairwise_scan_found(self):
        import itertools

        entities = self._entities()
        expected = {
            frozenset((a["id"], b["id"]))
            for a, b in itertools.combinations(entities, 2)
            if why_same(a["name"], b["name"]) is not None
        }
        assert len(expected) > 100, "the set must exercise the rules, not skip them"
        by_name = {e["name"]: e["id"] for e in entities}
        found = {
            frozenset((by_name[m.name_a], by_name[m.name_b]))
            for m in find_near_misses(entities, limit=len(entities) ** 2)
        }
        assert found == expected

    def test_an_overfull_block_is_skipped_with_a_warning_not_a_crash(self, caplog):
        from graphrag_sdk.storage import identity

        entities = [
            {"id": f"e{i}", "name": "Smith", "label": "Person", "is_stub": False}
            for i in range(identity._MAX_PER_LABEL + 1)
        ]
        entities.append({"id": "x", "name": "Maya Ellison", "label": "Person", "is_stub": False})
        entities.append({"id": "y", "name": "M. Ellison", "label": "Person", "is_stub": False})
        with caplog.at_level(logging.WARNING):
            found = find_near_misses(entities)
        assert [{m.name_a, m.name_b} for m in found] == [{"M. Ellison", "Maya Ellison"}]
        assert any("reduce to one comparable form (smith)" in m for m in caplog.messages)


class TestFinalizeReports:
    async def test_a_near_miss_is_reported_and_not_merged(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path, caplog
    ):
        rag = real_falkordb_rag_factory(
            llm=llm,
            resolver=resolver,
            ontology=Ontology(entities=[Entity(label="Person")], tables=[_HR]),
        )
        await rag.query(
            "CREATE (:__Entity__:Person {id:'m_ellison__person', name:'M. Ellison', "
            "description:'an engineer who led the remediation plan'})"
        )
        path = tmp_path / "hr.csv"
        path.write_text("employee_id,full_name,age\nE-1,Maya Ellison,34\n")
        await rag.ingest(str(path))

        with caplog.at_level(logging.WARNING):
            summary = await rag.finalize()

        assert any(
            "Maya Ellison" in line and "M. Ellison" in line for line in summary.probable_duplicates
        )
        assert any("did not merge" in message for message in caplog.messages)

        # Reported, not merged. Two nodes are recoverable; a merge is not, and
        # a merge would not even hold — the next extraction recreates the node
        # it deleted. The fix is the spelling in the source.
        people = await rag.query("MATCH (p:Person) RETURN count(p)")
        assert people[0][0] == 2
        await rag.close()

    async def test_a_clean_graph_reports_nothing(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(
            llm=llm,
            resolver=resolver,
            ontology=Ontology(entities=[Entity(label="Person")], tables=[_HR]),
        )
        path = tmp_path / "hr.csv"
        path.write_text("employee_id,full_name,age\nE-1,Maya Ellison,34\nE-2,Tomas Reyes,47\n")
        await rag.ingest(str(path))
        summary = await rag.finalize()
        assert summary.probable_duplicates == []
        await rag.close()


# Pairs that denote one thing and must end up as one node.
_ONE_THING = [
    ("Northwind Energy", "Northwind Energy AS"),
    ("Kestrel Grid", "Kestrel Grid Ltd."),
    ("Globex Ltd", "Globex Limited"),
    ("Acme Corporation", "Acme Corp"),
    ("Jean-Luc Picard", "Jean Luc Picard"),
    ("Priya Raman", "Raman, Priya"),
    ("Nordic Transmission AS", "Nordic Transmission"),
    ("Bank of Norway", "BANK OF NORWAY"),
    ("O'Brien Holdings", "OBrien Holdings"),
    ("Müller GmbH", "Müller"),
    ("Smith  &  Sons", "Smith and Sons"),
    ("Acme, Inc.", "Acme Inc"),
    ("Ellison, Maya", "Maya Ellison"),
    # "Inc" marks a legal form and nothing else, so stripping it is correct here:
    # Alphabet Inc. and Alphabet are one company.
    ("Alphabet Inc", "Alphabet"),
    # A symbol glued to a word is read aloud as a word.
    ("C#", "C Sharp"),
    ("C++", "C plus plus"),
    ("Smith + Wesson", "Smith & Wesson"),
]

# Pairs that are two things. A merge deletes a node and cannot be undone, so
# every one of these is a correctness requirement, not a quality target.
_TWO_THINGS = [
    ("Maya Ellison", "Maya Ellis"),
    ("John Smith", "Jane Smith"),
    ("Northwind Energy", "Southwind Energy"),
    ("Acme Corporation", "Acme Industries"),
    ("Tomas Reyes", "Tomas Reyna"),
    ("Kestrel Grid", "Kestrel Power"),
    ("Priya Raman", "Priya Ramesh"),
    ("Bank of Norway", "Bank of Sweden"),
    ("Maya Ellison", "Ellison"),
    ("Acme Holdings", "Acme Ventures"),
    ("Nordic Energy", "Nordic Gas"),
    ("A & B Ltd", "A and C Ltd"),
    # A symbol glued to a word is part of the name. Splitting on it made these
    # one key each and merged three programming languages into one node.
    ("C", "C#"),
    ("C", "C++"),
    ("C#", "C++"),
    ("F", "F#"),
    ("A", "A+"),
    ("James Morgan", "Morgan James"),
    ("Grace Newman", "Newman Grace"),
    ("Stanley Morgan", "Morgan Stanley"),
    # Distinct surnames that differ only in a diacritic. An ASCII-only tokenizer
    # treats the accented letter as a separator and gives both the key "m ller".
    ("Müller", "Möller"),
    ("Jörg Schmidt", "Jürg Schmidt"),
]

# A parent and its subsidiary, separated only by a word that also marks a legal
# form. Both appear together in any filings, supplier or org-chart corpus, and
# they are different legal entities. Stripping a RUN of suffixes from the full
# reporting list merged every one of these and deleted a node.
_PARENT_AND_SUBSIDIARY = [
    ("Sony Group Corporation", "Sony Corporation"),
    ("Vodafone Group Plc", "Vodafone Limited"),
    ("Hyundai Motor Group", "Hyundai Motor Company"),
    ("Berkshire Hathaway Inc", "Berkshire Hathaway Holdings"),
    ("Roche Holding AG", "Roche AG"),
    # Not companies at all — ordinary words that happen to sit in the legal-form
    # list. "Lucio Co" is a real surname.
    ("Four Seasons Spa", "Four Seasons"),
    ("Blue Man Group", "Blue Man"),
    ("Lucio Co", "Lucio"),
]

_TWO_THINGS = _TWO_THINGS + _PARENT_AND_SUBSIDIARY

# The subset of _TWO_THINGS that is a pure reordering: same tokens, different
# order, different things. Named explicitly rather than sliced off the end of
# _TWO_THINGS, which silently picks up whatever was appended last.
_REORDERINGS = [
    ("James Morgan", "Morgan James"),
    ("Grace Newman", "Newman Grace"),
    ("Stanley Morgan", "Morgan Stanley"),
]

# Same surname, one given name written as an initial. No canonical form reaches
# these without also merging every other person sharing that surname and letter.
_LEFT_TO_THE_REPORT = [
    ("Maya Ellison", "M. Ellison"),
    ("Tomas Reyes", "T. Reyes"),
    ("J. R. R. Tolkien", "John Ronald Reuel Tolkien"),
]


class TestTheCanonicalKey:
    """The grouping key the exact-match merge uses.

    Exact lowercase equality was the old key, and it left "Globex Ltd" and
    "Globex Limited" as two organizations — one holding the address, the other
    the revenue, and every question needing both answered from half the facts.
    """

    @pytest.mark.parametrize(("a", "b"), _ONE_THING)
    def test_two_spellings_of_one_name_reduce_to_one_key(self, a, b):
        assert canonical_key(a) == canonical_key(b)

    @pytest.mark.parametrize(("a", "b"), _TWO_THINGS)
    def test_two_different_things_never_share_a_key(self, a, b):
        assert canonical_key(a) != canonical_key(b)

    @pytest.mark.parametrize(("a", "b"), _LEFT_TO_THE_REPORT)
    def test_what_the_key_cannot_reach_is_still_reported(self, a, b):
        """Nothing that is one thing may be *silently* missed.

        The key does not join these, which is correct — merging them would merge
        every M-surname person too. The requirement is that they do not fall
        through: each is a near miss the user is shown.
        """
        assert canonical_key(a) != canonical_key(b)
        assert why_same(a, b)

    @pytest.mark.parametrize(("a", "b"), _REORDERINGS)
    def test_a_bare_reordering_is_not_evidence_of_one_thing(self, a, b):
        """Why the key preserves word order instead of sorting tokens.

        Sorting buys exactly one extra join over these fixtures — "Priya Raman"
        with "Raman, Priya", which the comma rule already handles — and costs
        three wrong merges: two pairs of people whose given name is the other's
        surname, and "Stanley Morgan" with "Morgan Stanley", a person and a bank.
        This test exists to fail if anyone reintroduces the sort.
        """
        assert canonical_key(a) != canonical_key(b)
        assert sorted(canonical_key(a).split()) == sorted(canonical_key(b).split())

    @pytest.mark.parametrize(("a", "b"), _PARENT_AND_SUBSIDIARY)
    def test_a_parent_and_its_subsidiary_are_not_merged(self, a, b):
        """The word that separates them is also a legal form. It must survive.

        Found by an independent audit, not by this file's original fixtures — they
        were all variations on one theme and never covered a holdco. The merge
        deletes a node, so each of these was silent data loss on the default
        finalize() path.

        They stay visible: why_same still reports every one of them, because
        over-reporting costs a reader a line and over-merging costs them a node.
        """
        assert canonical_key(a) != canonical_key(b)
        assert why_same(a, b), "declined for the merge, but must still be reported"

    def test_only_one_trailing_legal_form_is_removed(self):
        """Popping a run is what collapsed "Sony Group Corporation" to "sony"."""
        assert canonical_key("Sony Group Corporation") == "sony group"
        assert canonical_key("Kestrel Grid Ltd.") == "kestrel grid"

    def test_a_comma_before_a_legal_form_is_not_an_inversion(self):
        """``Acme, Inc.`` is not surname-first.

        Un-inverting it would move "inc" to the front, where the trailing-suffix
        rule cannot see it, and "Acme, Inc." would stop matching "Acme Inc".
        """
        assert canonical_key("Acme, Inc.") == canonical_key("Acme Inc")
        assert canonical_key("Acme, Inc.") == canonical_key("Acme")

    def test_the_key_is_stable_under_the_noise_sources_actually_differ_in(self):
        assert canonical_key("  ACME   Corp.  ") == canonical_key("acme corporation")

    @pytest.mark.parametrize("name", ["日本電力", "Газпром", "Ελλάδα", "Müller"])
    def test_a_name_with_no_latin_letters_still_gets_a_key_of_its_own(self, name):
        """Otherwise every such name shares the empty key and merges into one node.

        An ASCII-only word class treats every non-Latin character as a separator,
        so "日本電力" and "Газпром" both reduce to "" — one group, under any label
        they share, collapsing to a single entity.
        """
        assert canonical_key(name)
        assert canonical_key(name) != canonical_key("Ελλάδα") or name == "Ελλάδα"

    def test_a_name_that_is_only_punctuation_yields_an_empty_key(self):
        """The caller falls back to the raw name; it must not group everything.

        An empty key would put every such entity in one group under a label and
        merge them into a single node.
        """
        assert canonical_key("---") == ""
        assert canonical_key("") == ""


def _survivor(group):
    """What the exact-match pass keeps: highest rank wins, as it sorts."""
    return sorted(group, key=_survivor_rank, reverse=True)[0]


class TestWhichSpellingSurvives:
    """Canonical grouping makes the survivor's *name* a real choice.

    Under exact-equality grouping every name in a group was identical, so which
    node survived could not change the name on the graph. Grouping by canonical
    key puts "Globex Ltd" and "Globex Limited" in one group, and the survivor's
    spelling is the one every later query and answer shows.
    """

    def test_the_written_out_spelling_wins(self):
        group = [
            {"id": "a", "name": "Globex Ltd", "description": "", "is_stub": None},
            {"id": "b", "name": "Globex Limited", "description": "", "is_stub": None},
        ]
        assert _survivor(group)["name"] == "Globex Limited"

    def test_a_declared_key_still_outranks_a_longer_name(self):
        """Name length is a tiebreak, not a promotion.

        A mapped node's id is recomputed from the declared key on every ingest;
        an extracted node's is not. Letting a longer extracted name win would
        drop the reproducible id and re-create it as a second node next ingest.
        """
        group = [
            {"id": "e-1__org", "name": "Globex", "description": "", "is_stub": False},
            {"id": "x", "name": "Globex Limited Group", "description": "", "is_stub": None},
        ]
        assert _survivor(group)["id"] == "e-1__org"

    def test_the_survivor_does_not_depend_on_the_order_rows_came_back(self):
        """Entities are fetched with no ORDER BY and list.sort is stable.

        Without a total ordering the same data can settle on a different display
        name from one finalize to the next.
        """
        group = [
            {"id": "a", "name": "Acme Co", "description": "x", "is_stub": None},
            {"id": "b", "name": "ACME CO", "description": "x", "is_stub": None},
            {"id": "c", "name": "acme co", "description": "x", "is_stub": None},
        ]
        winners = {
            _survivor([group[i], group[j], group[k]])["id"]
            for i, j, k in ((0, 1, 2), (2, 1, 0), (1, 0, 2), (1, 2, 0))
        }
        assert len(winners) == 1


class TestTheJoinAgainstARealGraph:
    """What canonical grouping is for: a table row and a prose mention of one
    thing, spelled differently, becoming one node instead of two.
    """

    async def test_a_row_and_a_prose_mention_spelled_differently_become_one_node(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """Exact equality left these as two organizations.

        One held the address the CSV knew, the other the revenue the report
        described, and any question needing both was answered from half the
        facts while looking answered.
        """
        rag = real_falkordb_rag_factory(
            llm=llm,
            resolver=resolver,
            ontology=Ontology(entities=[Entity(label="Organization")], tables=[_ORGS]),
        )
        await rag.query(
            "CREATE (:__Entity__:Organization {id:'globex_limited__organization', "
            "name:'Globex Limited', description:'the grid operator named in the review'})"
        )
        path = tmp_path / "orgs.csv"
        path.write_text("org_id,org_name,employee_count\nG-1,Globex Ltd,240\n")
        await rag.ingest(str(path))

        await rag.finalize()

        rows = await rag.query("MATCH (o:Organization) RETURN o.id, o.name")
        assert len(rows) == 1, f"expected one organization, got {rows}"
        # The mapped node survives, under the id its own name gives it. Identity
        # is the name for both halves now, so this is also exactly the id a
        # re-ingest of the same row recomputes.
        assert rows[0][0] == "globex_ltd__organization"
        await rag.close()

    async def test_two_rows_of_one_table_stay_apart_however_alike_the_names(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        """The wider key must not overrule a declared key.

        A mapping that declares ``org_id`` is asserting that G-1 and G-2 are two
        organizations. That two of its rows happen to canonicalise together is
        not evidence against the thing the user declared.
        """
        rag = real_falkordb_rag_factory(
            llm=llm,
            resolver=resolver,
            ontology=Ontology(entities=[Entity(label="Organization")], tables=[_ORGS]),
        )
        path = tmp_path / "orgs.csv"
        path.write_text(
            "org_id,org_name,employee_count\nG-1,Globex Ltd,240\nG-2,Globex Limited,15\n"
        )
        await rag.ingest(str(path))

        await rag.finalize()

        rows = await rag.query("MATCH (o:Organization) RETURN o.id ORDER BY o.id")
        # Two names, two ids -- and the keys G-1 / G-2 say they are two things,
        # so the canonical-key merge leaves them alone.
        assert [r[0] for r in rows] == ["globex_limited__organization", "globex_ltd__organization"]
        await rag.close()
