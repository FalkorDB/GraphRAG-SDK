"""Near-miss identity: finding the pairs exact matching cannot, and acting on them.

The join between a table row and a prose mention is exact string equality on the
display name. Real sources disagree about spelling, so the interesting failure is
not a wrong merge — it is two nodes that should have been one, with no warning.
"""

from __future__ import annotations

import logging

import pytest

from graphrag_sdk import Column, ExactMatchResolution, Table
from graphrag_sdk.storage.identity import find_near_misses, why_same


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


class TestFinalizeReports:
    async def test_a_near_miss_is_reported_and_not_merged(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path, caplog
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.query(
            "CREATE (:__Entity__:Person {id:'m_ellison__person', name:'M. Ellison', "
            "description:'an engineer who led the remediation plan'})"
        )
        path = tmp_path / "hr.csv"
        path.write_text("employee_id,full_name,age\nE-1,Maya Ellison,34\n")
        await rag.ingest(
            str(path),
            mapping=Table("Person", key="employee_id", name="full_name",
                          age=Column("age", "INTEGER")),
        )

        with caplog.at_level(logging.WARNING):
            summary = await rag.finalize()

        assert any(
            "Maya Ellison" in line and "M. Ellison" in line
            for line in summary.probable_duplicates
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
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        path = tmp_path / "hr.csv"
        path.write_text(
            "employee_id,full_name,age\nE-1,Maya Ellison,34\nE-2,Tomas Reyes,47\n"
        )
        await rag.ingest(
            str(path),
            mapping=Table("Person", key="employee_id", name="full_name",
                          age=Column("age", "INTEGER")),
        )
        summary = await rag.finalize()
        assert summary.probable_duplicates == []
        await rag.close()
