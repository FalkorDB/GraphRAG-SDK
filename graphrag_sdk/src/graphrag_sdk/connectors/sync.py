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
import re
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Literal
from weakref import WeakKeyDictionary

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import DocumentInfo, DocumentOutput, Ontology
from graphrag_sdk.core.tables import TableMapping
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy
from graphrag_sdk.ingestion.loaders.record_loader import RecordBatch, RecordLoaderStrategy
from graphrag_sdk.ingestion.mapping import record_mapping_for
from graphrag_sdk.ingestion.structured_pipeline import StructuredIngestionResult

if TYPE_CHECKING:
    from graphrag_sdk.api.main import GraphRAG

logger = logging.getLogger(__name__)

# Properties the SDK keeps on a Document node for its own bookkeeping. Source
# metadata that used one of these names would overwrite the content hash the
# no-op check reads, the kind an update is refused on, or the marker crash
# recovery deletes a document on.
_VERSION_PROPERTY = "source_version"
_RESERVED_PROPERTIES = frozenset(
    {
        "id",
        "path",
        "content_hash",
        "kind",
        "table",
        "ready_to_commit",
        "pending_delete",
        _VERSION_PROPERTY,
    }
)
_RESERVED_PREFIXES = ("cleanup_",)
_PENDING_MARKER = "__pending__"


@dataclass
class _SourceLock:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0
    waiters: list[object] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _SourceLockPlanner:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


@dataclass
class _SourceLockReservation:
    rag: GraphRAG
    loop: asyncio.AbstractEventLoop
    loops: WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, _SourceLock]]
    locks: dict[str, _SourceLock]
    entries: list[tuple[str, _SourceLock]]
    token: object
    held: list[_SourceLock] = field(default_factory=list)
    released: bool = False

    async def __aenter__(self) -> None:
        try:
            for _, entry in self.entries:
                while entry.waiters[0] is not self.token:
                    changed = entry.changed
                    await changed.wait()
                await entry.lock.acquire()
                self.held.append(entry)
                entry.waiters.pop(0)
                changed = entry.changed
                entry.changed = asyncio.Event()
                changed.set()
        except BaseException:
            self._release()
            raise

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> Literal[False]:
        self._release()
        return False

    def _release(self) -> None:
        if self.released:
            return
        self.released = True
        for entry in reversed(self.held):
            entry.lock.release()
        for name, entry in self.entries:
            if self.token in entry.waiters:
                entry.waiters.remove(self.token)
                changed = entry.changed
                entry.changed = asyncio.Event()
                changed.set()
            entry.users -= 1
            if not entry.users:
                del self.locks[name]
        if not self.locks:
            self.loops.pop(self.loop, None)
        if not self.loops:
            _SOURCE_LOCKS.pop(self.rag, None)


_SOURCE_LOCKS: WeakKeyDictionary[
    GraphRAG, WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, _SourceLock]]
] = WeakKeyDictionary()
_SOURCE_LOCK_PLANNERS: WeakKeyDictionary[
    GraphRAG, WeakKeyDictionary[asyncio.AbstractEventLoop, _SourceLockPlanner]
] = WeakKeyDictionary()


@dataclass
class SourceDocument:
    """One document as its source has it.

    Args:
        id: The source's own id for the document: a Notion page id, a Drive file
            id, an object URL. It must stay the same for as long as the document
            exists, because it is what tells a changed document from a new one.
        text: The document's text, or a function that downloads it. A function
            is only called when the document has to be read.
        metadata: Provenance such as the source URL, author or modified time.
            Stored on the Document node. Values that are not strings, numbers,
            booleans, dates or lists of those are dropped.
        version: What the source says about the document's revision, such as
            its modified time or ETag. When it matches the version stored with
            the graph's copy, the document is reported unchanged without its
            text being read, so pass ``text`` as a function to skip the download.
    """

    id: str
    text: str | Callable[[], str]
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str | None = None

    def load_text(self) -> str:
        """The text, downloading it if ``text`` is a function."""
        text = self.text() if callable(self.text) else self.text
        if not isinstance(text, str):
            raise TypeError(f"document {self.id!r} has no text ({type(text).__name__})")
        return text


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
    document the subset left out would be deleted. Calls on the same GraphRAG
    instance and event loop are serialized per source. Across instances or
    processes, the caller must serialize jobs to prevent conflicting pruning.

    Does not call :meth:`GraphRAG.finalize`; call it once after syncing.

    Args:
        rag: The graph to write to.
        documents: Every document the source has now. A plain iterable is read
            in a worker thread, since fetching it is usually network I/O.
        source: A short name for the source, such as ``notion`` or ``drive``.
            Each document's graph id is ``<source>:<id>``. It may not contain
            ``:``, which would let one source's prefix cover another's.
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
    async with _source_locks(rag, [source]):
        return await _sync_documents(
            rag,
            documents,
            source=source,
            prune=prune,
            dry_run=dry_run,
            cache_unchanged_chunks=cache_unchanged_chunks,
            chunker=chunker,
            extractor=extractor,
            ctx=ctx,
        )


@asynccontextmanager
async def _source_locks(rag: GraphRAG, names: Iterable[str]) -> AsyncIterator[None]:
    """Hold the per-source locks for ``names`` on this instance and event loop.

    Reserve every source before waiting, in sorted order, so overlapping calls
    cannot overtake one another or deadlock.
    """
    async with _source_lock_planner(rag):
        reservation = _reserve_source_locks(rag, names)
    async with reservation:
        yield


@asynccontextmanager
async def _source_lock_planner(rag: GraphRAG) -> AsyncIterator[None]:
    """Order mapping preflight and lock reservation across sync entry points."""
    loop = asyncio.get_running_loop()
    loops = _SOURCE_LOCK_PLANNERS.setdefault(rag, WeakKeyDictionary())
    planner = loops.setdefault(loop, _SourceLockPlanner())
    planner.users += 1
    try:
        async with planner.lock:
            yield
    finally:
        planner.users -= 1
        if not planner.users and loops.get(loop) is planner:
            del loops[loop]
            if not loops:
                _SOURCE_LOCK_PLANNERS.pop(rag, None)


def _reserve_source_locks(rag: GraphRAG, names: Iterable[str]) -> _SourceLockReservation:
    loop = asyncio.get_running_loop()
    loops = _SOURCE_LOCKS.setdefault(rag, WeakKeyDictionary())
    locks = loops.setdefault(loop, {})
    entries = [(name, locks.setdefault(name, _SourceLock())) for name in sorted(set(names))]
    reservation = object()
    for _, entry in entries:
        entry.users += 1
        entry.waiters.append(reservation)
    return _SourceLockReservation(rag, loop, loops, locks, entries, reservation)


async def _sync_documents(
    rag: GraphRAG,
    documents: Iterable[SourceDocument] | AsyncIterable[SourceDocument],
    *,
    source: str,
    prune: bool,
    dry_run: bool,
    cache_unchanged_chunks: bool,
    chunker: ChunkingStrategy | None,
    extractor: ExtractionStrategy | None,
    ctx: Context | None,
) -> SyncReport:
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
        known_hash, known_version, known_props, pending = existing.get(
            doc_id, (None, None, {}, False)
        )
        # A missing hash means the last write was incomplete, and a recovery marker
        # an interrupted delete or cleanup; update() repairs both, so neither skips.
        if (
            known_hash is not None
            and doc.version is not None
            and doc.version == known_version
            and not pending
            and not _needs_recovery(known_props)
        ):
            if not dry_run:
                try:
                    await _refresh_metadata(rag, doc_id, doc, known_props)
                except Exception as exc:
                    logger.warning("metadata refresh of %s failed: %s", doc_id, exc)
                    report.failed[doc_id] = f"{type(exc).__name__}: {exc}"
                    continue
            report.unchanged.append(doc_id)
            continue
        try:
            text = await asyncio.to_thread(doc.load_text) if callable(doc.text) else doc.load_text()
        except Exception as exc:
            logger.warning("reading %s failed: %s", doc_id, exc)
            report.failed[doc_id] = f"{type(exc).__name__}: {exc}"
            continue
        if dry_run:
            if doc_id not in existing:
                report.added.append(doc_id)
            elif known_hash == _text_hash(text):
                report.unchanged.append(doc_id)
            else:
                report.updated.append(doc_id)
            continue

        try:
            result = await rag.update(
                doc_id,
                document_id=doc_id,
                loader=_PrefetchedLoader(doc, text),
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
        incomplete = (result.metadata or {}).get("incomplete_writes")
        if incomplete:
            # The pipeline reports failed chunks and short writes here instead of
            # raising. No content hash was recorded, so the next sync retries it.
            report.failed[doc_id] = f"incomplete writes: {incomplete}"
        elif result.no_op:
            # update() returns on a matching hash before writing anything, so a
            # moved page or a new author would otherwise never reach the graph.
            try:
                current_props = known_props
                if pending:
                    latest = await _existing_documents(rag, f"{source}:")
                    current = latest.get(doc_id)
                    if current is None:
                        raise RuntimeError(
                            f"document {doc_id!r} disappeared during metadata refresh"
                        )
                    current_props = current[2]
                await _refresh_metadata(rag, doc_id, doc, current_props)
            except Exception as exc:
                logger.warning("metadata refresh of %s failed: %s", doc_id, exc)
                report.failed[doc_id] = f"{type(exc).__name__}: {exc}"
                continue
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
    called. An empty ``rows`` removes every row the table had. Calls on the same
    GraphRAG instance and event loop are serialized per table.

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
    async with _source_lock_planner(rag):
        ontology = await rag.get_ontology(initialize=False)
        mapping = _mapping_named(ontology, table)
        reservation = _reserve_source_locks(rag, [mapping.source])
    async with reservation:
        return await _sync_table(rag, rows, table=table, ctx=ctx, mapping=mapping)


async def _sync_table(
    rag: GraphRAG,
    rows: Iterable[dict[str, Any]] | AsyncIterable[dict[str, Any]],
    *,
    table: str,
    ctx: Context | None,
    mapping: TableMapping,
) -> StructuredIngestionResult:
    records = [row async for row in _aiter(rows)]
    # An empty table still has a header. Without one the mapping's key and name
    # columns read as missing and the snapshot that removes every row is refused.
    header = [] if records else sorted(record_mapping_for(mapping).columns)
    loader = _RowsLoader(records, header)
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

    Every run reads every resource from the start. Cursors, item types, table
    mappings and the columns each mapping reads are checked for every resource
    before anything is written, so those problems fail the whole call rather
    than leaving it half applied. A cell that does not parse as its declared
    type is only found while its table is written.

    Calls on the same GraphRAG instance and event loop are serialized per
    resource name, from the first read to the last write, so overlapping calls
    apply their snapshots in the order they were made.

    dlt's incremental cursors are not supported. ``sync()`` keeps no dlt state,
    so a cursor would not advance between runs, and a partial read is wrong for
    both kinds of resource: pruning would delete the documents it skipped, and a
    table snapshot would remove the rows it skipped. A resource given a cursor
    with ``apply_hints(incremental=...)`` is refused. A cursor declared as the
    resource function's default argument is not detected, so do not pass such a
    resource.

    An empty resource is handled by what it would have been. One whose name has a
    ``TableMapping`` is an empty table, and its rows are removed. Any other is an
    empty document source, which deletes nothing (see :func:`sync_documents`).

    Requires the ``dlt`` extra: ``pip install graphrag-sdk[dlt]``.

    Example::

        from dlt.sources.sql_database import sql_database
        from llama_index.readers.notion import NotionPageReader

        from graphrag_sdk.connectors import llamaindex_documents, sync

        notion = llamaindex_documents(
            NotionPageReader(integration_token=token),
            name="notion", id_from=lambda doc: doc.id_, page_ids=ids
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
    resources = _resources(data)
    for resource in resources:
        if _has_cursor(resource):
            raise ValueError(
                f"Resource '{resource.name}' has an incremental cursor. sync() reads "
                "every source in full and keeps no dlt state, so a cursor would not "
                "advance between runs, and its partial read would delete the documents "
                "or table rows it skipped. Remove the cursor."
            )
    async with _source_lock_planner(rag):
        ontology = await rag.get_ontology(initialize=False)
        mappings = {resource.name: _find_mapping(ontology, resource.name) for resource in resources}
        lock_names = []
        for resource in resources:
            lock_names.append(resource.name)
            mapping = mappings[resource.name]
            if mapping is not None:
                lock_names.append(mapping.source)
        reservation = _reserve_source_locks(rag, lock_names)
    # Reserved before reading, so overlapping calls apply their snapshots in order.
    async with reservation:
        return await _sync_resources(
            rag,
            resources,
            ontology=ontology,
            mappings=mappings,
            prune=prune,
            dry_run=dry_run,
            cache_unchanged_chunks=cache_unchanged_chunks,
            chunker=chunker,
            extractor=extractor,
            ctx=ctx,
        )


async def _sync_resources(
    rag: GraphRAG,
    resources: list[Any],
    *,
    ontology: Ontology,
    mappings: dict[str, TableMapping | None],
    prune: bool,
    dry_run: bool,
    cache_unchanged_chunks: bool,
    chunker: ChunkingStrategy | None,
    extractor: ExtractionStrategy | None,
    ctx: Context | None,
) -> SyncResult:
    table_names = {name for name, mapping in mappings.items() if mapping is not None}

    # Read and classify everything before the first write.
    documents: dict[str, list[SourceDocument]] = {}
    tables: dict[str, list[dict[str, Any]]] = {}
    for resource in resources:
        name = resource.name
        items = _flatten(await asyncio.to_thread(list, resource))
        if all(isinstance(item, SourceDocument) for item in items) and (
            items or name not in table_names
        ):
            documents[name] = items
        elif all(isinstance(item, dict) for item in items):
            mapping = mappings[name] or _mapping_named(ontology, name)
            _check_columns(mapping, items, name)
            tables[name] = items
        else:
            kinds = sorted({type(item).__name__ for item in items})
            raise TypeError(
                f"Resource '{name}' yielded {', '.join(kinds)}. A resource yields "
                "either SourceDocument items or row dicts, not a mix."
            )
    for name in documents:
        _check_source_name(name)

    result = SyncResult()
    for name, docs in documents.items():
        result.documents[name] = await _sync_documents(
            rag,
            docs,
            source=name,
            prune=prune,
            dry_run=dry_run,
            cache_unchanged_chunks=cache_unchanged_chunks,
            chunker=chunker,
            extractor=extractor,
            ctx=ctx,
        )
    for name, rows in tables.items():
        result.table_rows[name] = len(rows)
        if not dry_run:
            mapping = mappings[name] or _mapping_named(ontology, name)
            result.tables[name] = await _sync_table(rag, rows, table=name, ctx=ctx, mapping=mapping)
    return result


# ── Helpers ──────────────────────────────────────────────────────


class _PrefetchedLoader(LoaderStrategy):
    """Hands :meth:`GraphRAG.update` a document that was already fetched.

    Going through a loader rather than ``text=`` is what carries the source's
    metadata onto the Document node; text mode has no way to pass it.
    """

    def __init__(self, doc: SourceDocument, text: str) -> None:
        self._doc = doc
        self._text = text

    async def load(self, source: str, ctx: Context) -> DocumentOutput:
        return DocumentOutput(
            text=self._text,
            document_info=DocumentInfo(
                uid=source, path=source, metadata=_document_properties(self._doc)
            ),
        )


def _document_properties(doc: SourceDocument) -> dict[str, Any]:
    props = graph_safe_metadata(doc.metadata)
    if doc.version is not None:
        props[_VERSION_PROPERTY] = str(doc.version)
    return props


class _RowsLoader(RecordLoaderStrategy):
    """Hands the structured write path rows that are already in memory."""

    def __init__(self, records: list[dict[str, Any]], header: list[str]) -> None:
        self._records = records
        columns: dict[str, None] = dict.fromkeys(header)
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
    uses for its own bookkeeping on the Document node. Keys must be bare ASCII
    identifiers, since the client's parameter serializer does not quote map keys.
    """
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        name = str(key)
        if (
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
            or name in _RESERVED_PROPERTIES
            or name.startswith(_RESERVED_PREFIXES)
        ):
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
    if ":" in source:
        # Ids are "<source>:<id>" and pruning lists by prefix, so a source named
        # "wiki" would own, and prune, every "wiki:team:<id>" document.
        raise ValueError(f"'source' may not contain ':', got {source!r}")
    if _PENDING_MARKER in source:
        raise ValueError(f"'source' may not contain {_PENDING_MARKER!r}")


async def _refresh_metadata(
    rag: GraphRAG, doc_id: str, doc: SourceDocument, stored: dict[str, Any]
) -> None:
    """Write what changed in ``doc``'s metadata, and remove keys the source dropped."""
    current = _document_properties(doc)
    changed = {key: value for key, value in current.items() if stored.get(key) != value}
    # Keys are bare identifiers (checked here), so they are safe to name in Cypher.
    removed = sorted(key for key in stored if key not in current and _is_source_key(key))
    if not changed and not removed:
        return
    cypher = "MATCH (d:Document {id: $id}) SET d += $props"
    if removed:
        cypher += " REMOVE " + ", ".join(f"d.`{key}`" for key in removed)
    await rag.query(cypher, {"id": doc_id, "props": changed})


def _is_source_key(key: str) -> bool:
    return (
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is not None
        and key not in _RESERVED_PROPERTIES
        and not key.startswith(_RESERVED_PREFIXES)
    )


def _needs_recovery(stored: dict[str, Any]) -> bool:
    return bool(stored.get("pending_delete")) or any(
        key.startswith(_RESERVED_PREFIXES) for key in stored
    )


async def _existing_documents(
    rag: GraphRAG, prefix: str
) -> dict[str, tuple[str | None, str | None, dict[str, Any], bool]]:
    """``{logical document id: (hash, version, properties, has pending copy)}``."""
    rows = await rag.query(
        "MATCH (d:Document) WHERE d.id STARTS WITH $prefix "
        f"RETURN d.id, d.content_hash, d.{_VERSION_PROPERTY}, properties(d)",
        {"prefix": prefix},
    )
    existing: dict[str, tuple[str | None, str | None, dict[str, Any], bool]] = {}
    pending_priority: dict[str, int] = {}
    for row in rows:
        node_id = row[0]
        logical_id, marker, suffix = node_id.partition(_PENDING_MARKER)
        is_pending = bool(
            marker and len(suffix) == 8 and all(c in "0123456789abcdef" for c in suffix)
        )
        if not is_pending:
            logical_id = node_id
        props = row[3] or {}
        if is_pending:
            priority = 2 if props.get("ready_to_commit") else 1
            if pending_priority.get(logical_id, 0) > priority:
                continue
            pending_priority[logical_id] = priority
            existing[logical_id] = (None, None, props, True)
        elif logical_id not in pending_priority:
            existing[logical_id] = (row[1], row[2], props, False)
    return existing


def _find_mapping(ontology: Ontology, table: str) -> TableMapping | None:
    for mapping in ontology.tables:
        base = os.path.basename(os.path.normpath(mapping.source))
        if table in (base, os.path.splitext(base)[0]):
            return mapping
    return None


def _mapping_named(ontology: Ontology, table: str) -> TableMapping:
    mapping = _find_mapping(ontology, table)
    if mapping is not None:
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


def _check_columns(mapping: TableMapping, rows: list[dict[str, Any]], table: str) -> None:
    """Refuse a table whose rows lack a column its mapping reads, before any write."""
    if not rows:
        return  # sync_table() gives an empty table the mapping's own header
    header: dict[str, None] = {}
    for row in rows:
        header.update(dict.fromkeys(row))
    problems = record_mapping_for(mapping).validate_against(list(header))
    if problems:
        raise ValueError(
            f"Table '{table}' does not fit its mapping, so nothing was written:\n  "
            + "\n  ".join(problems)
        )


def _has_cursor(resource: Any) -> bool:
    """Whether the resource, or any resource it reads from, has a cursor.

    The parents matter: ``file_documents()`` wraps a filesystem listing in a
    transformer, and a cursor on the listing makes the transformer's read
    partial too.
    """
    from dlt.sources import incremental

    pipe = getattr(resource, "_pipe", None)
    if pipe is None:
        return _is_cursor(getattr(resource, "incremental", None), incremental)
    while pipe is not None:
        if any(_is_cursor(step, incremental) for step in pipe.steps):
            return True
        pipe = pipe.parent
    return False


def _is_cursor(value: Any, incremental: type) -> bool:
    # A cursor applied to a plain resource is an ``incremental`` step. sql_database
    # instead gives every table a wrapper, cursor or not, whose own
    # ``incremental`` is set only once a cursor is applied.
    if isinstance(value, incremental):
        return True
    return type(value).__name__ == "IncrementalResourceWrapper" and (
        getattr(value, "incremental", None) is not None
    )


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
    worker = asyncio.create_task(asyncio.to_thread(list, items))
    try:
        materialized = await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
        worker.result()
        raise
    for item in materialized:
        yield item
