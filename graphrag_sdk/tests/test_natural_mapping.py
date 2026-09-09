"""The obvious reading of a table nobody declared a mapping for.

``ingest("employees.csv")`` without a mapping does not fail: every column becomes
a typed property and the label comes from the file name. What it will not do is
guess which column identifies the thing a row is *about* — types are measured,
identity is not, and a wrong identity guess silently attaches rows to the wrong
entities. So the rows land queryable and unjoined, and the report says so.

No database and no model needed.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.tables import TableMapping
from graphrag_sdk.ingestion.loaders.record_loader import CsvRecordLoader
from graphrag_sdk.ingestion.mapping import MappingError
from graphrag_sdk.ingestion.mapping_proposal import natural_mapping


async def _batch(text: str, name: str = "employees.csv"):
    path = pathlib.Path(tempfile.mkdtemp()) / name
    path.write_text(text)
    return await CsvRecordLoader().load_records(str(path), Context())


class TestWhatItDerives:
    async def test_every_column_becomes_a_typed_property(self):
        batch = await _batch(
            "employee_id,full_name,age,revenue_musd,joined\n"
            "E-1,Maya Ellison,34,880.5,2019-04-01\n"
            "E-2,Tomas Reyes,47,2450.0,2015-11-01\n"
        )
        mapping, _ = natural_mapping(batch, "employees.csv")

        assert mapping.label == "employees"
        assert mapping.key == "employee_id"
        types = {name: column.type for name, column in mapping.typed_properties.items()}
        assert types == {
            "full_name": "STRING",
            "age": "INTEGER",
            "revenue_musd": "FLOAT",
            "joined": "DATE",
        }

    async def test_the_key_column_is_not_also_a_property(self):
        batch = await _batch("employee_id,age\nE-1,34\n")
        mapping, _ = natural_mapping(batch, "employees.csv")
        assert "employee_id" not in mapping.typed_properties

    async def test_an_awkward_header_is_stored_under_a_usable_name(self):
        batch = await _batch("employee_id,HQ Country\nE-1,Norway\n")
        mapping, notes = natural_mapping(batch, "employees.csv")
        assert mapping.typed_properties["col_hq_country"].name == "HQ Country"
        assert any("col_hq_country" in note for note in notes)

    async def test_two_headers_reducing_to_one_name_are_both_kept(self):
        """``HQ Country`` and ``hq-country`` both store as ``col_hq_country``,
        and the second silently replaced the first: a column of the export was
        gone from the mapping with nothing said. Suffixed in header order."""
        batch = await _batch("employee_id,HQ Country,hq-country\nE-1,Norway,NO\n")
        mapping, notes = natural_mapping(batch, "employees.csv")

        stored = {name: column.name for name, column in mapping.typed_properties.items()}
        assert stored == {"col_hq_country": "HQ Country", "col_hq_country_2": "hq-country"}
        assert any("'hq-country' stored as 'col_hq_country_2'" in note for note in notes)

    @pytest.mark.parametrize(
        ("source", "label"),
        [
            ("employees.csv", "employees"),
            ("/data/2026-08/employees.csv", "employees"),
            ("hr-final(1).csv", "hr_final_1"),
            ("2024_hr.csv", "t_2024_hr"),
        ],
    )
    async def test_the_label_comes_from_the_file_name(self, source, label):
        batch = await _batch("employee_id,age\nE-1,34\n")
        assert natural_mapping(batch, source)[0].label == label


class TestItNeverGuessesIdentity:
    async def test_no_name_column_is_declared(self):
        """A display name is what joins a row to a mention in prose.

        Choosing one is a guess, and a wrong guess attaches rows to the wrong
        entities without saying so — the failure class this whole path avoids.
        """
        batch = await _batch("employee_id,full_name,age\nE-1,Maya Ellison,34\nE-2,Tomas Reyes,47\n")
        mapping, notes = natural_mapping(batch, "employees.csv")

        assert mapping.name is None, "a name column was guessed"
        assert mapping.standalone is True, "unjoined by design, so say so"
        assert any("not joined" in note for note in notes)
        assert any("name=<column>" in note for note in notes), "the report must name the fix"

    async def test_a_wordy_column_is_still_not_taken_as_a_name(self):
        """``ColumnProfile.looks_like_a_name`` could pick this. It is not used."""
        batch = await _batch(
            "row_id,description\n"
            "1,Northwind Energy reported a shortfall\n"
            "2,Kestrel Grid renewed the agreement\n"
        )
        mapping, _ = natural_mapping(batch, "notes.csv")
        assert mapping.name is None


class TestACodeIsNotANumber:
    @pytest.mark.parametrize("values", [["02134", "10001", "00501"], ["0044", "0049"]])
    async def test_zero_padded_values_stay_strings(self, values):
        """``02134`` parses as INTEGER and comes back as ``2134``. Zip codes,
        phone prefixes and padded account codes would lose digits silently."""
        rows = "\n".join(f"R-{index},{value}" for index, value in enumerate(values))
        batch = await _batch(f"row_id,code\n{rows}\n")
        mapping, _ = natural_mapping(batch, "codes.csv")
        assert mapping.properties["code"].type == "STRING"

    async def test_a_plain_zero_is_still_a_number(self):
        batch = await _batch("row_id,count\nR-1,0\nR-2,12\nR-3,0.5\n")
        mapping, _ = natural_mapping(batch, "counts.csv")
        assert mapping.properties["count"].type == "FLOAT"


class TestTypesAreMeasuredOverTheWholeFile:
    async def test_a_late_unparseable_value_widens_the_type(self):
        """A 500-row sample would declare INTEGER and then fail on row 601.

        The declared type is enforced at ingest, so getting it from a sample means
        describing the file in a way the file itself violates.
        """
        rows = "\n".join(f"E-{index},{index}" for index in range(600))
        batch = await _batch(f"employee_id,age\n{rows}\nE-999,N/A\n")
        mapping, _ = natural_mapping(batch, "employees.csv")
        assert mapping.typed_properties["age"].type == "STRING"

    async def test_a_clean_column_keeps_its_narrow_type(self):
        rows = "\n".join(f"E-{index},{index}" for index in range(600))
        batch = await _batch(f"employee_id,age\n{rows}\n")
        mapping, _ = natural_mapping(batch, "employees.csv")
        assert mapping.typed_properties["age"].type == "INTEGER"


class TestWhatItRefuses:
    async def test_no_unique_column_is_refused(self):
        """Keying on the row ordinal would rebind every row the moment the export
        is re-sorted, so a table with no identity is refused rather than invented."""
        batch = await _batch("region,amount\nNorth,10\nNorth,10\nSouth,20\n", "dupes.csv")
        with pytest.raises(MappingError, match="unique and complete"):
            natural_mapping(batch, "dupes.csv")

    async def test_the_chosen_key_is_always_reported(self):
        """The leftmost unique column wins, which can be a measure column when it
        happens to be distinct. Named in the report rather than guessed better."""
        batch = await _batch("region,amount\nNorth,10\nSouth,20\n", "sales.csv")
        mapping, notes = natural_mapping(batch, "sales.csv")
        assert mapping.key == "region"
        assert f"key {mapping.key!r}" in notes[0]


class TestTheTypeNarrowingWarningIsMeasuredNotSampled:
    """A pure check over a RecordBatch — no database, so it runs in CI.

    ``properties={"age": "age"}`` means STRING, and the SDK says so at ingest when
    the column holds only numbers. It once profiled a 500-row sample, so on a file
    clean for 500 rows and ``"N/A"`` on row 501 it advised INTEGER, and taking that
    advice made the next load raise. The integration copies of these tests need
    RUN_INTEGRATION; this pair is what CI actually enforces.
    """

    @staticmethod
    def _mapping():
        from graphrag_sdk.ingestion.mapping import record_mapping_for

        return record_mapping_for(
            TableMapping(
                source="hr.csv",
                label="Person",
                key="employee_id",
                name="full_name",
                properties={"age": "age"},
            )
        )

    async def test_a_uniformly_numeric_string_column_is_flagged(self, caplog):
        import logging

        from graphrag_sdk.ingestion.structured_pipeline import (
            _warn_about_a_column_typed_narrower_than_declared,
        )

        batch = await _batch(
            "employee_id,full_name,age\n"
            + "".join(f"E-{i},P{i},{20 + i % 40}\n" for i in range(1, 501)),
            name="hr.csv",
        )
        with caplog.at_level(logging.WARNING):
            _warn_about_a_column_typed_narrower_than_declared(batch, self._mapping(), "hr.csv")
        assert any("declared STRING hold only one narrower type" in m for m in caplog.messages)
        assert any("Column('age', 'INTEGER')" in m for m in caplog.messages)

    async def test_one_bad_value_past_row_500_silences_it(self, caplog):
        import logging

        from graphrag_sdk.ingestion.structured_pipeline import (
            _warn_about_a_column_typed_narrower_than_declared,
        )

        batch = await _batch(
            "employee_id,full_name,age\n"
            + "".join(f"E-{i},P{i},{20 + i % 40}\n" for i in range(1, 501))
            + "E-501,P501,N/A\n",
            name="hr.csv",
        )
        with caplog.at_level(logging.WARNING):
            _warn_about_a_column_typed_narrower_than_declared(batch, self._mapping(), "hr.csv")
        assert not any(
            "declared STRING hold only one narrower type" in m for m in caplog.messages
        ), "advised a type the whole file contradicts"
