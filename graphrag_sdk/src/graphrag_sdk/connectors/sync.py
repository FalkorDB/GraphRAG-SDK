# GraphRAG SDK — Connectors: keeping the graph in step with an external source
#
# Fetching belongs to the frameworks: dlt for databases, file stores and REST
# APIs, LlamaIndex readers for apps dlt has no source for (see sources.py). What
# lives here is the part every source shares, and the part a framework cannot
# do for us because it depends on how this SDK stores documents: one stable id
# per document, skipping what has not changed, and removing what the source no
# longer has.

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import DocumentInfo, DocumentOutput
from graphrag_sdk.core.tables import TableMapping
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy
from graphrag_sdk.ingestion.loaders.record_loader import RecordBatch, RecordLoaderStrategy
from graphrag_sdk.ingestion.structured_pipeline import StructuredIngestionResult

if TYPE_CHECKING:
    from graphrag_sdk.api.main import GraphRAG

logger = logging.getLogger(__name__)

# Properties the SDK keeps on a Document node for its own bookkeeping. Source
# metadata that used one of these names would overwrite the content hash the
# no-op check reads, or the kind an update is refused on.
_RESERVED_PROPERTIES = frozenset({"id", "path", "content_hash", "kind", "table", "ready_to_commit"})
_RESERVED_PREFIXES = ("cleanup_",)
_PENDING_MARKER = "__pending__"


@dataclass
class SourceDocument:
    """One document as its source has it.

    Args:
        id: The source's own id for the document: a Notion page id, a Drive file
            id, an object URL. It must stay the same for as long as the document
            exists, because it is what tells a changed document from a new one.
        text: The document's text.
        metadata: Provenance such as the source URL, author or modified time.
            Stored on the Document node. Values that are not strings, numbers,
            booleans, dates or lists of those are dropped.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncReport:
    """What one sync of a document source did, by graph document id.

    In a dry run nothing is written and the lists say what a real run would do.
    """

    source: str
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False
    prune_skipped: str | None = None
    """Why removed documents were not deleted, when pruning was asked for but
    did not run."""

    def summary(self) -> str:
        """One line, e.g. ``notion: 2 added, 1 updated, 40 unchanged, 0 deleted, 0 failed``."""
        text = (
            f"{self.source}: {len(self.added)} added, {len(self.updated)} updated, "
            f"{len(self.unchanged)} unchanged, {len(self.deleted)} deleted, "
            f"{len(self.failed)} failed"
        )
        return f"{text} (dry run)" if self.dry_run else text


@dataclass
class SyncResult:
    """What :func:`sync` did, per dlt resource."""

    documents: dict[str, SyncReport] = field(default_factory=dict)
    tables: dict[str, StructuredIngestionResult] = field(default_factory=dict)
    table_rows: dict[str, int] = field(default_factory=dict)
    """Rows read per table. Filled in a dry run too, where nothing is written."""


def document_id_for(source: str, source_id: str) -> str:
    """The graph document id for a document of ``source``: ``<source>:<source_id>``.

    The prefix keeps two sources' ids apart and is what pruning lists by, so a
    sync of one source can never delete another source's documents.
    """
    return f"{source}:{source_id}"


async def sync_documents(
    rag: GraphRAG,
    documents: Iterable[SourceDocument] | AsyncIterable[SourceDocument],
    *,
    source: str,
    prune: bool = True,
    dry_run: bool = False,
    cache_unchanged_chunks: bool = True,
    chunker: ChunkingStrategy | None = None,
    extractor: ExtractionStrategy | None = None,
    ctx: Context | None = None,
) -> SyncReport:
    """Bring the graph's documents for ``source`` in line with ``documents``.

    Each document is written through :meth:`GraphRAG.update` with
    ``if_missing="ingest"``. A document whose text has not changed is a no-op
    with no model calls, and a changed one re-extracts only its changed chunks
    when ``cache_unchanged_chunks`` is on. With ``prune``, documents of this
    source that ``documents`` did not include are deleted.

    Pruning assumes ``documents`` is the whole source. Pass ``prune=False`` when
    it is a subset, such as only the pages edited since the last run, or every
    document the subset left out would be deleted.

    Does not call :meth:`GraphRAG.finalize`; call it once after syncing.

    Args:
        rag: The graph to write to.
        documents: Every document the source has now. A plain iterable is read
            in a worker thread, since fetching it is usually network I/O.
        source: A short name for the source, such as ``notion`` or ``drive``.
            Each document's graph id is ``<source>:<id>``.
        prune: Delete this source's documents that ``documents`` no longer has.
            Skipped, with the reason in ``prune_skipped``, when the source
            returned nothing at all: an empty answer is more often a revoked
            token or a wrong folder than a source whose every document was
            deleted.
        dry_run: Report what would happen without writing anything.
        cache_unchanged_chunks: Reuse the extraction of chunks whose text did not
            change when a document is updated.
        chunker: Chunking strategy for the documents, as for :meth:`GraphRAG.update`.
        extractor: Extraction strategy for the documents, as for :meth:`GraphRAG.update`.
        ctx: Execution context.

    Returns:
        A :class:`SyncReport`.
    """
    _check_source_name(source)
    existing = await _existing_documents(rag, f"{source}:")
    report = SyncReport(source=source, dry_run=dry_run)
    seen: set[str] = set()

    async for doc in _aiter(documents):
        if not isinstance(doc, SourceDocument):
            raise TypeError(
                f"sync_documents() takes SourceDocument items, got {type(doc).__name__}"
            )
        doc_id = document_id_for(source, str(doc.id))
        if doc_id in seen:
            report.failed[doc_id] = "the source returned this id twice; kept the first"
            continue
        seen.add(doc_id)

        if _PENDING_MARKER in doc_id:
            report.failed[doc_id] = f"ids may not contain {_PENDING_MARKER!r}"
            continue
        if dry_run:
            if doc_id not in existing:
                report.added.append(doc_id)
            elif existing[doc_id] == _text_hash(doc.text):
                report.unchanged.append(doc_id)
            else:
                report.updated.append(doc_id)
            continue

        try:
            result = await rag.update(
                doc_id,
                document_id=doc_id,
                loader=_PrefetchedLoader(doc),
                if_missing="ingest",
                cache_unchanged_chunks=cache_unchanged_chunks,
                chunker=chunker,
                extractor=extractor,
                ctx=ctx,
            )
        except Exception as exc:
            logger.warning("sync of %s failed: %s", doc_id, exc)
            report.failed[doc_id] = f"{type(exc).__name__}: {exc}"
            continue
        if result.no_op:
            report.unchanged.append(doc_id)
        elif result.replaced_existing:
            report.updated.append(doc_id)
        else:
            report.added.append(doc_id)

    if prune:
        stale = sorted(set(existing) - seen)
        if stale and not seen:
            report.prune_skipped = (
                f"the source returned no documents, so its {len(stale)} documents in "
                "the graph were kept; delete them with delete_document() if the "
                "source really is empty"
            )
            logger.warning("%s: %s", source, report.prune_skipped)
        else:
            for doc_id in stale:
                if dry_run:
                    report.deleted.append(doc_id)
                    continue
                try:
                    await rag.delete_document(doc_id, if_missing="ignore")
                except Exception as exc:
                    logger.warning("delete of %s failed: %s", doc_id, exc)
                    report.failed[doc_id] = f"{type(exc).__name__}: {exc}"
                    continue
                report.deleted.append(doc_id)

    logger.info(report.summary())
    return report


async def sync_table(
    rag: GraphRAG,
    rows: Iterable[dict[str, Any]] | AsyncIterable[dict[str, Any]],
    *,
    table: str,
    ctx: Context | None = None,
) -> StructuredIngestionResult:
    """Write ``rows`` as the current state of ``table``.

    A table is written as a snapshot, the way the SDK writes any table: rows
    that are new or changed are written, and rows that disappeared from the
    source are removed. ``rows`` must therefore be the whole table, not only the
    rows changed since the last run. An unchanged table is a no-op. No model is
    called.

    The table needs a :class:`~graphrag_sdk.core.tables.TableMapping` in the
    ontology whose ``source`` is the table's name (``users`` or ``users.csv``).

    Args:
        rag: The graph to write to.
        rows: Every row of the table, as ``{column: value}`` dicts.
        table: The table's name, as a dlt resource or database calls it.
        ctx: Execution context.

    Returns:
        The :class:`StructuredIngestionResult` of the write.

    Raises:
        ValueError: No mapping in the ontology names ``table``.
    """
    mapping = await _mapping_named(rag, table)
    records = [row async for row in _aiter(rows)]
    loader = _RowsLoader(records)
    result = await rag.ingest(mapping.source, record_loader=loader, ctx=ctx)
    assert isinstance(result, StructuredIngestionResult)  # a record_loader routes to tables
    return result


async def sync(
    rag: GraphRAG,
    data: Any,
    *,
    prune: bool = True,
    dry_run: bool = False,
    cache_unchanged_chunks: bool = True,
    chunker: ChunkingStrategy | None = None,
    extractor: ExtractionStrategy | None = None,
    ctx: Context | None = None,
) -> SyncResult:
    """Sync one or more dlt sources or resources into the graph.

    Each resource is read in full and handled by what it yields:

    - :class:`SourceDocument` items, as from
      :func:`~graphrag_sdk.connectors.sources.llamaindex_documents` and
      :func:`~graphrag_sdk.connectors.sources.file_documents`, are synced with
      :func:`sync_documents`. The resource's name is the source name, so it must
      be unique per source: two Drive folders need two names.
    - Rows (dicts), as from dlt's ``sql_database`` or ``rest_api`` sources, are
      written with :func:`sync_table`, matched to a ``TableMapping`` by the
      resource's name.

    Every run reads every resource from the start. dlt's incremental cursors are
    not used, because removed documents and rows are found by comparing a full
    read with the graph. A resource given a cursor with
    ``apply_hints(incremental=...)`` is refused while ``prune`` is on, since its
    partial read would delete everything it skipped. A cursor declared as the
    resource function's default argument is not detected, so do not prune such
    a resource.

    Requires the ``dlt`` extra: ``pip install graphrag-sdk[dlt]``.

    Example::

        from dlt.sources.sql_database import sql_database
        from llama_index.readers.notion import NotionPageReader

        from graphrag_sdk.connectors import llamaindex_documents, sync

        notion = llamaindex_documents(
            NotionPageReader(integration_token=token), name="notion", page_ids=ids
        )
        result = await sync(rag, [notion, sql_database(credentials=pg_url)])
        await rag.finalize()

    Args:
        rag: The graph to write to.
        data: A dlt source, a dlt resource, or a list of them.
        prune: Delete documents a source no longer has. Tables always drop rows
            that left the source, since a table is written as a snapshot.
        dry_run: Read everything and report, without writing. Tables report
            their row counts only.
        cache_unchanged_chunks: See :func:`sync_documents`.
        chunker: See :func:`sync_documents`. Tables are not chunked.
        extractor: See :func:`sync_documents`. No model extracts from tables.
        ctx: Execution context.

    Returns:
        A :class:`SyncResult`.
    """
    result = SyncResult()
    for resource in _resources(data):
        name = resource.name
        if prune and _has_cursor(resource):
            raise ValueError(
                f"Resource '{name}' has an incremental cursor, so it returns only part "
                "of the source and pruning would delete the rest. Remove the cursor, "
                "or sync with prune=False."
            )
        items = _flatten(await asyncio.to_thread(list, resource))
        if not items:
            logger.warning("%s returned nothing; skipped", name)
            continue
        if all(isinstance(item, SourceDocument) for item in items):
            result.documents[name] = await sync_documents(
                rag,
                items,
                source=name,
                prune=prune,
                dry_run=dry_run,
                cache_unchanged_chunks=cache_unchanged_chunks,
                chunker=chunker,
                extractor=extractor,
                ctx=ctx,
            )
        elif all(isinstance(item, dict) for item in items):
            result.table_rows[name] = len(items)
            if not dry_run:
                result.tables[name] = await sync_table(rag, items, table=name, ctx=ctx)
        else:
            kinds = sorted({type(item).__name__ for item in items})
            raise TypeError(
                f"Resource '{name}' yielded {', '.join(kinds)}. A resource yields "
                "either SourceDocument items or row dicts, not a mix."
            )
    return result


# ── Helpers ──────────────────────────────────────────────────────


class _PrefetchedLoader(LoaderStrategy):
    """Hands :meth:`GraphRAG.update` a document that was already fetched.

    Going through a loader rather than ``text=`` is what carries the source's
    metadata onto the Document node; text mode has no way to pass it.
    """

    def __init__(self, doc: SourceDocument) -> None:
        self._doc = doc

    async def load(self, source: str, ctx: Context) -> DocumentOutput:
        return DocumentOutput(
            text=self._doc.text,
            document_info=DocumentInfo(
                uid=source, path=source, metadata=graph_safe_metadata(self._doc.metadata)
            ),
        )


class _RowsLoader(RecordLoaderStrategy):
    """Hands the structured write path rows that are already in memory."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        columns: dict[str, None] = {}
        for record in records:
            columns.update(dict.fromkeys(record))
        self._columns = list(columns)

    async def load_records(self, source: str, ctx: Context) -> RecordBatch:
        return RecordBatch(
            open_records=lambda: iter(self._records),
            columns=self._columns,
            document_info=DocumentInfo(uid=source, path=source),
            record_count=len(self._records),
        )


def graph_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """The part of ``metadata`` a graph property can hold.

    Strings, finite numbers and booleans are kept, dates become ISO strings, and
    lists keep their scalar items. Anything else is dropped, as are names the SDK
    uses for its own bookkeeping on the Document node.
    """
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        name = str(key)
        if name in _RESERVED_PROPERTIES or name.startswith(_RESERVED_PREFIXES):
            continue
        if isinstance(value, (list, tuple)):
            items = [v for v in (_scalar(item) for item in value) if v is not None]
            if items:
                safe[name] = items
            continue
        scalar = _scalar(value)
        if scalar is not None:
            safe[name] = scalar
    return safe


def _scalar(value: Any) -> str | int | float | bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return None


def _text_hash(text: str) -> str:
    # The same hash update() compares, so a dry run predicts its no-ops exactly.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_source_name(source: str) -> None:
    if not source or not source.strip():
        raise ValueError("'source' must be a non-empty name, such as 'notion'")
    if _PENDING_MARKER in source:
        raise ValueError(f"'source' may not contain {_PENDING_MARKER!r}")


async def _existing_documents(rag: GraphRAG, prefix: str) -> dict[str, str | None]:
    """``{document id: content hash}`` for the graph's documents under ``prefix``."""
    rows = await rag.query(
        "MATCH (d:Document) WHERE d.id STARTS WITH $prefix "
        "AND NOT d.id CONTAINS $pending RETURN d.id, d.content_hash",
        {"prefix": prefix, "pending": _PENDING_MARKER},
    )
    return {row[0]: row[1] for row in rows}


async def _mapping_named(rag: GraphRAG, table: str) -> TableMapping:
    ontology = await rag.get_ontology()
    for mapping in ontology.tables:
        base = os.path.basename(os.path.normpath(mapping.source))
        if table in (base, os.path.splitext(base)[0]):
            return mapping
    declared = ", ".join(m.source for m in ontology.tables) or "none"
    raise ValueError(
        f"No TableMapping in the ontology is for table '{table}' (declared: "
        f"{declared}). Declare TableMapping(source={table!r}, label=..., name=...) "
        "so its rows know which entity they are."
    )


def _resources(data: Any) -> list[Any]:
    try:
        from dlt.extract import DltResource, DltSource
    except ImportError as exc:
        raise ImportError(
            "sync() needs dlt. Install it with: pip install graphrag-sdk[dlt]"
        ) from exc
    resources: list[Any] = []
    for item in data if isinstance(data, (list, tuple)) else [data]:
        if isinstance(item, DltSource):
            resources.extend(item.selected_resources.values())
        elif isinstance(item, DltResource):
            resources.append(item)
        else:
            raise TypeError(
                f"sync() takes dlt sources and resources, got {type(item).__name__}. "
                "Use sync_documents() or sync_table() for plain iterables."
            )
    names = [r.name for r in resources]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"Two resources are named {', '.join(duplicates)}. The name is the source "
            "name documents are kept under, so each needs its own."
        )
    return resources


def _has_cursor(resource: Any) -> bool:
    from dlt.sources import incremental

    # A cursor applied to a plain resource is the resource's ``incremental``
    # itself. sql_database instead gives every table a wrapper, cursor or not,
    # whose own ``incremental`` is set only once a cursor is applied.
    value = getattr(resource, "incremental", None)
    if isinstance(value, incremental):
        return True
    return getattr(value, "incremental", None) is not None


def _flatten(items: list[Any]) -> list[Any]:
    """Rows from a resource, with Arrow tables and data frames opened up into dicts."""
    flat: list[Any] = []
    for item in items:
        if hasattr(item, "to_pylist"):  # pyarrow Table / RecordBatch
            flat.extend(item.to_pylist())
        elif hasattr(item, "to_dict") and hasattr(item, "columns"):  # pandas DataFrame
            flat.extend(item.to_dict("records"))
        else:
            flat.append(item)
    return flat


async def _aiter(items: Iterable[Any] | AsyncIterable[Any]) -> AsyncIterator[Any]:
    if isinstance(items, AsyncIterable):
        async for item in items:
            yield item
        return
    if isinstance(items, (list, tuple)):
        for item in items:
            yield item
        return
    iterator: Iterator[Any] = iter(items)
    done = object()
    while True:
        item = await asyncio.to_thread(next, iterator, done)
        if item is done:
            return
        yield item
