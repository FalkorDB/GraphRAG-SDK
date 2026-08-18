# GraphRAG SDK — Ingestion: Record Loader Strategy
# Pattern: Strategy — every structured source adapter implements this interface.
#
# Sits beside LoaderStrategy rather than replacing it. A prose loader returns
# text; a record loader returns a re-openable stream of flat records.

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import DocumentInfo


@dataclass
class RecordBatch:
    """A re-openable stream of records, plus what the source is.

    ``open_records`` is a *factory*, not an iterable, and that is load bearing.
    The write path walks the records twice, once to build a chunk per record and
    once to map records onto nodes and edges. A one-shot iterator is silently
    empty the second time, which produces a zero-row ingest with no error. A
    factory is re-iterable by construction.

    A loader that genuinely cannot reopen its source spools once and closes over
    the buffer, which puts the memory cost at the loader where it is visible
    instead of corrupting the write.

    Args:
        open_records: Called with no arguments, returns a fresh iterator of flat
            ``{column: value}`` records.
        columns: The source's column names, in source order.
        document_info: Identity of the source, so its records hang off one
            Document node.
        record_count: Set when the loader knows it cheaply, ``None`` when
            streaming.
    """

    open_records: Callable[[], Iterator[dict[str, Any]]]
    columns: list[str]
    document_info: DocumentInfo
    record_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self.open_records()


class RecordLoaderStrategy(ABC):
    """Abstract base class for structured source loaders.

    A loader parses and nothing else. It writes no graph, applies no mapping and
    calls no model.

    Example::

        class MyRecordLoader(RecordLoaderStrategy):
            async def load_records(self, source: str, ctx: Context) -> RecordBatch:
                return RecordBatch(
                    open_records=lambda: iter(read_my_source(source)),
                    columns=["a", "b"],
                    document_info=DocumentInfo(uid=source, path=source),
                )
    """

    @abstractmethod
    async def load_records(self, source: str, ctx: Context) -> RecordBatch:
        """Read a structured source into a re-openable record stream.

        Args:
            source: Path, URL or identifier for the source.
            ctx: Execution context.

        Returns:
            A :class:`RecordBatch`.
        """
        ...


class CsvRecordLoader(RecordLoaderStrategy):
    """Loads a delimited text file as one record per row.

    Reopens the file on each iteration, so nothing is held in memory beyond the
    current row.

    Args:
        delimiter: Field separator. ``None`` sniffs it from the first kilobyte,
            which handles comma and tab without the caller choosing.
        encoding: File encoding.
        document_id: Overrides the Document node id. Defaults to the file name,
            which is the stable handle ``update()`` and ``delete_document()`` use.
    """

    def __init__(
        self,
        *,
        delimiter: str | None = None,
        encoding: str = "utf-8",
        document_id: str | None = None,
    ) -> None:
        self._delimiter = delimiter
        self._encoding = encoding
        self._document_id = document_id

    def _sniff(self, path: Path) -> str:
        if self._delimiter:
            return self._delimiter
        with path.open("r", encoding=self._encoding, newline="") as handle:
            sample = handle.read(1024)
        if not sample:
            return ","
        try:
            return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            return ","

    async def load_records(self, source: str, ctx: Context) -> RecordBatch:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"structured source not found: {source}")
        delimiter = self._sniff(path)

        with path.open("r", encoding=self._encoding, newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            columns = list(reader.fieldnames or [])
            record_count = sum(1 for _ in reader)
        if not columns:
            raise ValueError(f"{source} has no header row, so no columns to map")
        blank = [i for i, c in enumerate(columns) if not (c or "").strip()]
        if blank:
            raise ValueError(
                f"{source} has unnamed columns at positions {blank}; every column "
                "a mapping might read needs a name"
            )

        def open_records() -> Iterator[dict[str, Any]]:
            with path.open("r", encoding=self._encoding, newline="") as fresh:
                for row in csv.DictReader(fresh, delimiter=delimiter):
                    # csv gives None for short rows; normalise so a mapping sees
                    # a missing cell rather than the string "None".
                    yield {k: ("" if v is None else v) for k, v in row.items() if k}

        ctx.log(
            f"Loaded {record_count} records from {path.name} "
            f"({len(columns)} columns, delimiter {delimiter!r})"
        )
        return RecordBatch(
            open_records=open_records,
            columns=columns,
            document_info=DocumentInfo(
                uid=self._document_id or path.name,
                path=str(path),
                metadata={"kind": "structured", "delimiter": delimiter},
            ),
            record_count=record_count,
        )
