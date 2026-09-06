"""Reading a table: what is measured, and what is never guessed.

The design rule is that as little as possible is guessed. Everything a reading
of the data can settle — the key column, each column's type, which columns are
foreign keys — is settled by measurement, and everything a reading cannot settle
is left to the mapping the user declares. These tests pin the measured half,
which is the half that must be right every time.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.api.main import GraphRAG
from graphrag_sdk.core.models import DocumentInfo, Entity, Ontology
from graphrag_sdk.ingestion.loaders.record_loader import RecordBatch
from graphrag_sdk.ingestion.mapping import Column, TableMapping, record_mapping_for
from graphrag_sdk.ingestion.mapping_proposal import (
    count_entities_per_label,
    pick_key,
    profile_columns,
)
from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution


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


class TestLabelsInUseAreCounted:
    """Being *in* the ontology is not evidence a label is the one in use.

    The built-in defaults seed a dozen labels into every graph, so a people table
    faced with both ``Employee`` (declared and filled by an earlier source) and
    ``Person`` (a default holding nothing) would otherwise be settled on the more
    obvious word — recreating exactly the split this measurement exists to
    prevent. Measured on a live graph before this: ``Person`` was chosen over an
    ``Employee`` that already held data.
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


class TestAColumnNamedLikeAParameterIsNotLost:
    def test_a_column_called_links_is_mapped_as_a_property(self):
        """A column called "links" used to collide with the ``links=`` keyword
        argument that carried properties, raising TypeError instead of mapping,
        so such a column was left unmapped. Properties are a dict of their own
        now, so a column may be named after any part of the declaration.
        """
        mapping = TableMapping(
            source="x.csv",
            label="Thing",
            key="k",
            properties={"links": Column("links")},
            standalone=True,
        )
        assert record_mapping_for(mapping).anchor.typed_properties["links"] == Column("links")


class TestATableIsNotReadAsProse:
    """Left to the text path a CSV is silently the wrong thing: the whole file
    becomes one chunk with its commas intact and no column keeps its type.
    Measured on a two-row export: one entity written, ``age`` absent entirely,
    nothing raised.

    So the suffix decides the path, not the presence of a mapping. Every tabular
    suffix goes to the deterministic record path — with the declared mapping if
    the ontology holds one, with the measured reading of the file if it does not
    — and only an explicit loader opts out.
    """

    @pytest.fixture
    def employees_csv(self, tmp_path):
        path = tmp_path / "employees.csv"
        path.write_text("employee_id,full_name,age\nE-1,Alice Smith,34\n", encoding="utf-8")
        return str(path)

    @pytest.fixture
    def resolver(self):
        return ExactMatchResolution(resolve_property="name")

    @staticmethod
    async def _rows(rag, cypher):
        result = await rag._graph_store.query_raw(cypher)
        return result.result_set or []

    @pytest.mark.parametrize("suffix", [".csv", ".tsv", ".psv", ".tab", ".CSV"])
    def test_every_tabular_suffix_is_covered(self, suffix):
        assert GraphRAG._is_tabular(f"data{suffix}") is True

    def test_an_explicit_loader_is_the_escape(self):
        """A table of support tickets or survey answers is prose that happens to
        live in columns, and saying so explicitly must keep working."""
        from graphrag_sdk import TextLoader

        assert GraphRAG._is_tabular("tickets.csv", TextLoader()) is False

    def test_a_document_is_untouched(self):
        assert GraphRAG._is_tabular("note.txt") is False
        assert GraphRAG._is_tabular("paper.pdf") is False

    async def test_a_declared_mapping_is_found_by_basename_and_used(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        """The mapping lives in the ontology and is matched on the file's
        basename, so ``ingest`` takes no mapping argument and the two cannot
        disagree."""
        ontology = Ontology(
            entities=[Entity(label="Person")],
            tables=[
                TableMapping(
                    source="employees.csv",
                    label="Person",
                    key="employee_id",
                    name="full_name",
                    properties={"age": Column("age", "INTEGER")},
                )
            ],
        )
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        result = await rag.ingest(employees_csv)
        assert result.records == 1, "the structured path ran, so this is not prose"
        assert await self._rows(rag, "MATCH (p:Person) RETURN p.name, p.employees__age") == [
            ["Alice Smith", 34]
        ]

    async def test_a_csv_without_a_mapping_still_keeps_its_types(
        self, real_falkordb_rag_factory, llm, resolver, employees_csv
    ):
        """Nobody declared a mapping, so the label comes from the file name and
        the rows land unjoined — but they land as records with measured types,
        not as one chunk of prose with ``age`` absent entirely."""
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        result = await rag.ingest(employees_csv)
        assert result.records == 1
        assert await self._rows(rag, "MATCH (n:employees) RETURN n.employees__age") == [[34]]
