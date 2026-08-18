"""Reading a delimited source into records.

A loader parses and nothing else, so these tests are about the two things that
corrupt a structured ingest quietly: a stream that cannot be read twice, and a
header the mapping cannot address.
"""

from __future__ import annotations

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.ingestion.loaders.record_loader import CsvRecordLoader, RecordBatch


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "employees.csv"
    path.write_text(
        "employee_id,full_name,age\nE-1,Alice Smith,34\nE-2,Bob Jones,45\n",
        encoding="utf-8",
    )
    return path


class TestCsvRecordLoader:
    async def test_reads_header_and_rows(self, csv_file, ctx: Context):
        batch = await CsvRecordLoader().load_records(str(csv_file), ctx)
        assert batch.columns == ["employee_id", "full_name", "age"]
        assert batch.record_count == 2
        assert [r["full_name"] for r in batch] == ["Alice Smith", "Bob Jones"]

    async def test_the_stream_is_re_iterable(self, csv_file, ctx: Context):
        """Load bearing. The write path walks the records twice, once for chunks
        and once for the mapping. A one-shot iterator is silently empty the
        second time, which ingests zero rows and raises nothing.
        """
        batch = await CsvRecordLoader().load_records(str(csv_file), ctx)
        first = list(batch)
        second = list(batch)
        assert first == second
        assert len(second) == 2

    async def test_document_uid_defaults_to_the_file_name(self, csv_file, ctx: Context):
        """``update()`` and ``delete_document()`` address a source by this id."""
        batch = await CsvRecordLoader().load_records(str(csv_file), ctx)
        assert batch.document_info.uid == "employees.csv"

    async def test_document_id_can_be_overridden(self, csv_file, ctx: Context):
        loader = CsvRecordLoader(document_id="hr-export")
        batch = await loader.load_records(str(csv_file), ctx)
        assert batch.document_info.uid == "hr-export"

    async def test_tab_delimiter_is_sniffed(self, tmp_path, ctx: Context):
        path = tmp_path / "orgs.tsv"
        path.write_text("org_id\torg_name\nORG-42\tAcme Corp\n", encoding="utf-8")
        batch = await CsvRecordLoader().load_records(str(path), ctx)
        assert batch.columns == ["org_id", "org_name"]
        assert list(batch)[0]["org_name"] == "Acme Corp"

    async def test_explicit_delimiter_wins_over_sniffing(self, tmp_path, ctx: Context):
        path = tmp_path / "semi.csv"
        path.write_text("a;b\n1;2\n", encoding="utf-8")
        batch = await CsvRecordLoader(delimiter=";").load_records(str(path), ctx)
        assert batch.columns == ["a", "b"]

    async def test_short_row_yields_an_empty_cell_not_the_string_none(self, tmp_path, ctx: Context):
        """csv gives None for a missing trailing cell. Written through, it would
        become the literal text "None" on the node."""
        path = tmp_path / "ragged.csv"
        path.write_text("a,b,c\n1,2\n", encoding="utf-8")
        batch = await CsvRecordLoader().load_records(str(path), ctx)
        assert list(batch)[0]["c"] == ""

    async def test_missing_file_is_reported_as_such(self, tmp_path, ctx: Context):
        with pytest.raises(FileNotFoundError):
            await CsvRecordLoader().load_records(str(tmp_path / "nope.csv"), ctx)

    async def test_a_headerless_source_is_rejected(self, tmp_path, ctx: Context):
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="no header row"):
            await CsvRecordLoader().load_records(str(path), ctx)

    async def test_an_unnamed_column_is_rejected(self, tmp_path, ctx: Context):
        """A mapping addresses columns by name, so a blank one is unaddressable."""
        path = tmp_path / "blank.csv"
        path.write_text("a,,c\n1,2,3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unnamed columns"):
            await CsvRecordLoader().load_records(str(path), ctx)

    async def test_encoding_is_honoured(self, tmp_path, ctx: Context):
        path = tmp_path / "latin.csv"
        path.write_bytes("name\nZoë\n".encode("latin-1"))
        batch = await CsvRecordLoader(encoding="latin-1").load_records(str(path), ctx)
        assert list(batch)[0]["name"] == "Zoë"


class TestRecordBatch:
    def test_iterating_calls_the_factory_each_time(self):
        calls = []

        def factory():
            calls.append(1)
            return iter([{"a": "1"}])

        from graphrag_sdk.core.models import DocumentInfo

        batch = RecordBatch(
            open_records=factory,
            columns=["a"],
            document_info=DocumentInfo(uid="d", path="d"),
        )
        list(batch)
        list(batch)
        assert len(calls) == 2
