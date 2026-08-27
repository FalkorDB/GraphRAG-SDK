"""Proposing a mapping: what is measured, and what is left to the model.

The design rule is that the model decides as little as possible. Everything a
reading of the data can settle — the key column, each column's type, which
columns are foreign keys — is settled by measurement, and only the semantic
choices are asked. These tests pin the measured half, which is the half that
must be right every time.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.core.models import DocumentInfo, Entity, Ontology
from graphrag_sdk.ingestion.loaders.record_loader import RecordBatch
from graphrag_sdk.ingestion.mapping import Column, Link, Table
from graphrag_sdk.ingestion.mapping_proposal import (
    _TABLE_PARAMETERS,
    MappingProposal,
    count_entities_per_label,
    find_foreign_keys,
    pick_key,
    profile_columns,
)


@pytest.fixture
def mock_conn():
    from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection

    conn = MagicMock(spec=FalkorDBConnection)
    result = MagicMock()
    result.result_set = []
    conn.query = AsyncMock(return_value=result)
    conn.config = ConnectionConfig()
    ontology_graph = MagicMock()
    ontology_graph.query = AsyncMock(return_value=result)
    conn._driver = MagicMock()
    conn._driver.select_graph = MagicMock(return_value=ontology_graph)
    conn._ensure_client = MagicMock()
    return conn


@pytest.fixture
def graphrag(mock_conn, embedder, llm):
    from graphrag_sdk.api.main import GraphRAG

    return GraphRAG(connection=mock_conn, llm=llm, embedder=embedder, embedding_dimension=8)


def batch_of(rows: list[dict[str, str]], columns: list[str] | None = None) -> RecordBatch:
    return RecordBatch(
        open_records=lambda: iter(rows),
        columns=columns or list(rows[0]),
        document_info=DocumentInfo(uid="people.csv", path="people.csv"),
    )


class TestTypesAreReadNotGuessed:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            (["1", "2", "3"], "INTEGER"),
            (["1.5", "2", "-0.5"], "FLOAT"),
            (["true", "no", "1"], "BOOLEAN"),
            (["2019-04-01", "2020-12-31"], "DATE"),
            (["Alice Smith", "Bob"], "STRING"),
        ],
    )
    def test_the_narrowest_type_every_value_parses(self, values, expected):
        rows = [{"c": v} for v in values]
        assert profile_columns(batch_of(rows))[0].inferred_type == expected

    def test_one_bad_value_widens_the_type(self):
        """A declared type is enforced at ingest, so a wrong one fails the load.
        Better to widen than to guess narrow from the first few rows."""
        rows = [{"c": "1"}, {"c": "2"}, {"c": "not a number"}]
        assert profile_columns(batch_of(rows))[0].inferred_type == "STRING"

    def test_an_empty_column_is_a_string(self):
        rows = [{"c": ""}, {"c": ""}]
        assert profile_columns(batch_of(rows))[0].inferred_type == "STRING"


class TestTheKeyIsMeasured:
    def test_a_unique_complete_column_is_the_key(self):
        rows = [{"id": "1", "name": "a"}, {"id": "2", "name": "a"}]
        assert pick_key(profile_columns(batch_of(rows))).name == "id"

    def test_a_column_with_a_gap_is_not_a_key(self):
        rows = [{"id": "1"}, {"id": ""}]
        assert pick_key(profile_columns(batch_of(rows))) is None

    def test_a_repeated_value_is_not_a_key(self):
        rows = [{"id": "1"}, {"id": "1"}]
        assert pick_key(profile_columns(batch_of(rows))) is None

    def test_the_leftmost_qualifying_column_wins(self):
        """Where several qualify, that is where an id column conventionally sits."""
        rows = [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]
        assert pick_key(profile_columns(batch_of(rows))).name == "a"


class TestForeignKeysAreLookedUpNotInferredFromNames:
    """A column is a reference when its values already resolve to entities, not
    when its name happens to end in ``_id``."""

    @staticmethod
    def _store(matched: int):
        result = MagicMock()
        result.result_set = [[matched]]
        store = MagicMock()
        store.query_raw = AsyncMock(return_value=result)
        return store

    async def test_a_column_whose_values_exist_is_offered(self):
        rows = [{"org_id": "ORG-42"}, {"org_id": "ORG-7"}]
        found = await find_foreign_keys(
            profile_columns(batch_of(rows)),
            Ontology(entities=[Entity(label="Organization")]),
            self._store(matched=2),
        )
        assert [(c.column, c.label) for c in found] == [("org_id", "Organization")]

    async def test_a_column_whose_values_do_not_exist_is_not_offered(self):
        rows = [{"org_id": "NOPE-1"}, {"org_id": "NOPE-2"}]
        found = await find_foreign_keys(
            profile_columns(batch_of(rows)),
            Ontology(entities=[Entity(label="Organization")]),
            self._store(matched=0),
        )
        assert found == []

    async def test_the_key_column_is_not_offered_as_a_foreign_key(self):
        rows = [{"id": "A"}, {"id": "B"}]
        found = await find_foreign_keys(
            profile_columns(batch_of(rows)),
            Ontology(entities=[Entity(label="Thing")]),
            self._store(matched=2),
            exclude="id",
        )
        assert found == []

    async def test_no_graph_means_no_candidates(self):
        rows = [{"org_id": "ORG-42"}]
        found = await find_foreign_keys(
            profile_columns(batch_of(rows)),
            Ontology(entities=[Entity(label="Organization")]),
            None,
        )
        assert found == []


class TestLabelsInUseAreCounted:
    """Being *in* the ontology is not evidence a label is the one in use.

    The built-in defaults seed a dozen labels into every graph, so a people table
    offered both ``Employee`` (declared and filled by an earlier source) and
    ``Person`` (a default holding nothing) would otherwise be told to pick and
    would reasonably pick the more obvious word — recreating exactly the split
    this module exists to prevent. Measured on a live graph before this: the
    proposal chose ``Person`` over an ``Employee`` that already held data.
    """

    async def test_counts_are_read_from_the_graph(self):
        result = MagicMock()
        result.result_set = [[7]]
        store = MagicMock()
        store.query_raw = AsyncMock(return_value=result)
        counts = await count_entities_per_label(
            Ontology(entities=[Entity(label="Employee"), Entity(label="Person")]), store
        )
        assert counts == {"Employee": 7, "Person": 7}

    async def test_no_graph_means_no_counts(self):
        counts = await count_entities_per_label(Ontology(entities=[Entity(label="Employee")]), None)
        assert counts == {}


class TestTheProposalIsReviewable:
    @staticmethod
    def _proposal() -> MappingProposal:
        table = Table(
            "Employee",
            key="person_id",
            name="full_name",
            age=Column("age", "INTEGER"),
            notes="notes",
            links=[Link("WORKS_AT", to="Organization", by="org_id")],
        )
        return MappingProposal(table=table, source="people.csv", evidence=["key is unique"])

    def test_it_emits_committable_code(self):
        """A proposal regenerated per run puts a model back in the ingest path.
        The output is meant to be committed, so it has to be valid Python."""
        code = self._proposal().as_code()
        assert 'Table("Employee"' in code
        assert 'key="person_id"' in code
        assert 'age=Column("age", "INTEGER")' in code
        assert 'notes="notes"' in code
        assert 'Link("WORKS_AT", to="Organization", by="org_id")' in code

        namespace: dict[str, object] = {"Table": Table, "Column": Column, "Link": Link}
        rebuilt = eval(code, namespace)  # noqa: S307 - the point is that it evaluates
        assert rebuilt.fingerprint == self._proposal().table.fingerprint

    def test_a_new_type_is_flagged_rather_than_applied(self):
        proposal = self._proposal()
        assert proposal.introduces_a_new_type is False
        proposal.requested_new_label = "MitigationPractice"
        assert proposal.introduces_a_new_type is True

    def test_the_summary_names_the_source_and_the_label(self):
        summary = self._proposal().summary()
        assert "people.csv" in summary and "Employee" in summary


class TestReservedColumnNames:
    def test_the_guarded_names_are_the_tables_own_parameters(self):
        """A column called "links" would collide with the ``links=`` argument and
        raise TypeError instead of mapping, so such a column is left unmapped."""
        assert _TABLE_PARAMETERS == {"node", "key", "name", "links", "description"}

    def test_the_collision_is_real(self):
        with pytest.raises(TypeError, match="multiple values"):
            Table("X", key="k", links=None, **{"links": Column("links")})


class TestATableIsNotReadAsProse:
    """Without a mapping a CSV goes down the text path, and that is silently the
    wrong thing: the whole file becomes one chunk with its commas intact and no
    column keeps its type. Measured on a two-row export: one entity written,
    ``age`` absent entirely, nothing raised.
    """

    @pytest.fixture
    def csv_file(self, tmp_path):
        path = tmp_path / "employees.csv"
        path.write_text("employee_id,full_name,age\nE-1,Alice Smith,34\n", encoding="utf-8")
        return str(path)

    async def test_a_csv_without_a_mapping_is_refused(self, graphrag, csv_file):
        with pytest.raises(ValueError, match="looks like a table"):
            await graphrag.ingest(csv_file)

    async def test_the_error_names_both_ways_out(self, graphrag, csv_file):
        with pytest.raises(ValueError) as caught:
            await graphrag.ingest(csv_file)
        message = str(caught.value)
        assert "mapping=Table(...)" in message
        assert "propose_mapping" in message
        assert "loader=TextLoader()" in message

    @pytest.mark.parametrize("suffix", [".csv", ".tsv", ".psv", ".tab", ".CSV"])
    async def test_every_tabular_suffix_is_covered(self, graphrag, tmp_path, suffix):
        path = tmp_path / f"data{suffix}"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="looks like a table"):
            await graphrag.ingest(str(path))

    async def test_an_explicit_loader_is_the_escape(self, graphrag, csv_file):
        """A table of support tickets or survey answers is prose that happens to
        live in columns, and saying so explicitly must keep working."""
        from graphrag_sdk import TextLoader

        graphrag._refuse_table_as_prose(csv_file, TextLoader())  # must not raise

    async def test_a_document_is_untouched(self, graphrag, tmp_path):
        graphrag._refuse_table_as_prose(str(tmp_path / "note.txt"), None)
        graphrag._refuse_table_as_prose(str(tmp_path / "paper.pdf"), None)

    async def test_a_mapping_bypasses_the_guard_entirely(self, graphrag, csv_file):
        """The guard exists to catch a *missing* mapping, so it must not fire on
        the ordinary structured path. It reached update() once and broke every
        structured re-sync before that was caught."""
        from graphrag_sdk import Column, Table

        mapping = Table("Person", key="employee_id", name="full_name", age=Column("age", "INTEGER"))
        result = await graphrag.ingest(csv_file, mapping=mapping)
        assert result.records == 1, "the structured path ran, so the guard stayed out of it"
