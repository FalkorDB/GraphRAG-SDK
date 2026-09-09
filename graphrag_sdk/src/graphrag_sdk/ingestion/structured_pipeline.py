# GraphRAG SDK — Ingestion: Structured Pipeline
# The deterministic write path for records: no model is called anywhere in here.
#
# Deliberately generic. It never names a source, a column or a label, so a new
# source costs a declaration and not code.

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    DocumentInfo,
    GraphNode,
    GraphRelationship,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import compute_entity_id
from graphrag_sdk.ingestion.lexical_graph import LexicalGraphWriter
from graphrag_sdk.ingestion.loaders.record_loader import (
    RecordBatch,
    RecordLoaderStrategy,
    cell_text,
)
from graphrag_sdk.ingestion.mapping import (
    Column,
    MappingError,
    NodeMapping,
    RecordMapping,
    safe_property_name,
)
from graphrag_sdk.ingestion.mapping_proposal import _WHOLE_FILE, profile_columns
from graphrag_sdk.storage.graph_store import ReferenceNode

logger = logging.getLogger(__name__)

# Chunk properties the writer owns. A column named one of these, or whose header
# is not a usable identifier, is stored under a ``col_`` name by
# ``safe_property_name`` rather than overwriting the chunk's identity or reaching
# the driver as something it cannot serialise.
_CHUNK_RESERVED = frozenset({"id", "text", "index", "kind", "record_key", "embedding"})


def record_chunk_uid(document_uid: str, record_key: str, occurrence: int = 0) -> str:
    """Deterministic chunk id for one record.

    Keyed on the *effective* document uid, never on a canonical one. Keying a
    record chunk on a canonical document id makes a pending update MERGE onto the
    live document's chunks, which the cutover then deletes: measured as three
    chunks before and zero after, with no exception raised.

    ``occurrence`` distinguishes rows that share a key. A chunk identifies a
    *row*; only the entity identifies by key. Without this, a source whose key
    column is not unique silently loses rows: two rows keyed K1 produced one
    chunk holding the first row's cells, while the ingest reported two. The
    default reproduces the original digest exactly, so chunk ids already in a
    graph do not move.
    """
    suffix = "" if occurrence == 0 else f"::{occurrence}"
    digest = hashlib.sha256(f"{document_uid}::{record_key}{suffix}".encode()).hexdigest()
    return f"rec_{digest[:24]}"


def _new_record_digest(mapping: RecordMapping, columns: list[str]) -> hashlib._Hash:
    """Seed a digest with everything about a source other than its rows.

    The mapping is folded in because identical rows under a changed declaration
    produce a different graph, so a re-declared mapping must not look like
    unchanged data to ``update()``'s no-op short circuit.
    """
    digest = hashlib.sha256(mapping.fingerprint.encode("utf-8"))
    digest.update(b"\x00columns\x00" + "\x1f".join(columns).encode("utf-8"))
    return digest


def _feed_record(digest: hashlib._Hash, columns: list[str], record: dict[str, Any]) -> None:
    """Fold one record into a digest, reading columns in declared order."""
    digest.update(b"\x00row\x00")
    for column in columns:
        digest.update(f"{column}={record.get(column, '')}\x1f".encode())


def records_content_hash(batch: RecordBatch, mapping: RecordMapping) -> str:
    """Digest a source's records, for the ``update()`` no-op short circuit.

    Costs one pass over the records. ``update()`` needs the hash *before* it
    decides whether to write anything, which is earlier than the ingest path
    happens to compute the same value, so this walks the stream itself rather
    than duplicating the algorithm.
    """
    digest = _new_record_digest(mapping, batch.columns)
    for record in batch:
        _feed_record(digest, batch.columns, record)
    return digest.hexdigest()


def render_record(record: dict[str, Any]) -> str:
    """Render a record as a sentence.

    ``text`` is the only field chunk embedding and full-text search read, and
    embedders are trained on language. Key-value soup separated by punctuation
    embeds poorly and loses to real prose in vector search, so a record that is
    never retrieved would be a record that may as well not be a chunk.
    """
    parts = [
        f"{key.replace('_', ' ')} {value}"
        for key, value in record.items()
        if value not in (None, "")
    ]
    return ", ".join(parts) + "." if parts else ""


def _ambiguous_names(batch: RecordBatch, mapping: RecordMapping) -> set[tuple[str, str]]:
    """``(label, lowercased name)`` for every name the batch gives two different keys.

    A name is the entity's identity, so two rows sharing one would become one
    entity — two John Smiths collapsing into a single Person. Those rows keep
    their key as identity instead, which also means a prose mention of "John
    Smith" joins neither, correctly: it is ambiguous. Reported, so a table with
    many such names is noticed rather than silently half-joined.

    Counted per label over rows *and* references, by distinct key. A reference
    names its target too (``manager_id`` plus ``manager_name``), and two managers
    both called John used to derive one id from the name: the edge remap then
    held one entry for the two of them, and Alice, whose row said ``E1``, was
    linked to ``E2``. A row and a reference sharing a name under two keys is the
    same collision. The same key under one name twice is one entity named twice,
    not an ambiguity.
    """
    keys: dict[tuple[str, str], set[str]] = {}
    # (label, key column, name column) for nodes that name what they write; the
    # filter is what lets mypy see the name column is a str below.
    named = [(node.label, node.key, node.name) for node in mapping.nodes if node.name]
    if not named:
        return set()
    for record in batch:
        for label, key_column, name_column in named:
            display = cell_text(record, name_column).lower()
            key = record.get(key_column)
            key = "" if key is None else str(key).strip()
            if display and key:
                keys.setdefault((label, display), set()).add(key)
    ambiguous = {name for name, seen in keys.items() if len(seen) > 1}
    if ambiguous:
        sample = ", ".join(repr(name) for _, name in sorted(ambiguous)[:5])
        logger.warning(
            "%d name(s) appear on more than one row (e.g. %s). Entity identity is the "
            "name, so those rows are keyed on their declared key instead and a document "
            "mention of that name will not join any of them — it is ambiguous.",
            len(ambiguous),
            sample,
        )
    return ambiguous


def _unique_references(references: list[ReferenceNode]) -> list[ReferenceNode]:
    """First occurrence of each referenced id, in order.

    Order matters: a reference names a node only ON CREATE, so the first entry is
    the one whose name lands on a node that does not exist yet.
    """
    seen: set[str] = set()
    unique: list[ReferenceNode] = []
    for reference in references:
        if reference.id in seen:
            continue
        seen.add(reference.id)
        unique.append(reference)
    return unique


def _reference_key(reference: ReferenceNode) -> tuple[str, str]:
    return reference.label, str(reference.properties.get("entity_key", ""))


def _collapse_same_key_references(
    references: list[ReferenceNode],
    edges: list[GraphRelationship],
    keyed: dict[tuple[str, str], list[tuple[str, str]]],
) -> list[ReferenceNode]:
    """One node per ``(label, key)`` within a batch, however each row named it.

    A reference's id comes from the name the row supplies for it, or from the
    key when it supplies none. So a citations table whose ``citing`` column
    carries only an id while its ``cited`` column carries id plus title derives
    two different ids for the same paper — and a self-referential table
    (``manager_id`` pointing at rows of the same file) derives one for the row
    and another for the reference. The lookup against the graph cannot catch
    either: nothing has been written yet. Without this, a single file raised
    two placeholders for one key, which the design rules out.

    The row wins when this batch declares the entity, otherwise the reference
    that names it, otherwise the first bare one. Edges follow the survivor.
    """
    canonical: dict[tuple[str, str], str] = {}
    for (label, _signed_key), rows in keyed.items():
        for entity_key, node_id in rows:
            canonical.setdefault((label, entity_key), node_id)
    for reference in references:
        key = _reference_key(reference)
        if reference.name != key[1]:
            canonical.setdefault(key, reference.id)
    for reference in references:
        canonical.setdefault(_reference_key(reference), reference.id)

    remap = {
        reference.id: canonical[_reference_key(reference)]
        for reference in references
        if canonical[_reference_key(reference)] != reference.id
    }
    if not remap:
        return references
    for edge in edges:
        edge.start_node_id = remap.get(edge.start_node_id, edge.start_node_id)
        edge.end_node_id = remap.get(edge.end_node_id, edge.end_node_id)
    return [reference for reference in references if reference.id not in remap]


def _walk_records(
    batch: RecordBatch,
    mapping: RecordMapping,
    document_uid: str,
    skipped: list[int] | None = None,
) -> Iterator[tuple[int, dict[str, Any], str, str]]:
    """Yield ``(index, record, record_key, chunk_uid)`` for every usable row.

    The write path walks the records twice, once for chunks and once for the
    mapping, and both need the *same* chunk id per row. Deriving that in two
    places is how they would drift, so it is derived here and shared. Rows
    without a key are skipped by both passes for the same reason: with no key a
    row has no stable identity, so it could never be updated or deleted later.

    ``skipped`` collects the 1-based source line of every row dropped that way,
    for a caller that wants to report it. A drop used to be invisible: a four-row
    file with one blank key cell reported ``records: 3`` and no warning, so the
    numbers agreed with each other and with nothing else.
    """
    anchor = mapping.anchor
    occurrences: dict[str, int] = {}
    for index, record in enumerate(batch):
        record_key = cell_text(record, anchor.key)
        if not record_key:
            if skipped is not None:
                skipped.append(index + 1)
            continue
        occurrence = occurrences.get(record_key, 0)
        occurrences[record_key] = occurrence + 1
        yield index, record, record_key, record_chunk_uid(document_uid, record_key, occurrence)


def record_cells(record: dict[str, Any]) -> dict[str, Any]:
    """The record's cells verbatim, to sit on the chunk beside the rendered text.

    The typed projection lives on the entity, where queries and aggregation read
    it. This is the faithful record of what the source said, so the original row
    is recoverable from the graph without parsing the rendered sentence.
    """
    cells: dict[str, Any] = {}
    for key, value in record.items():
        if value in (None, ""):
            continue
        cells[safe_property_name(key, _CHUNK_RESERVED)] = value
    return cells


class StructuredIngestionResult:
    """Counts from one structured ingest.

    ``chunks_deleted`` / ``entities_deleted`` are non-zero when the source was
    already in the graph and the write was a re-sync: rows that disappeared from
    the source take their chunks with them, and entities nothing else mentions
    any more go too. A caller watching those numbers is watching for exactly the
    thing that used to be silent.
    """

    __slots__ = (
        "records",
        "chunks",
        "entities",
        "references",
        "edges",
        "document_id",
        "content_hash",
        "chunks_deleted",
        "entities_deleted",
        "replaced_existing",
        "no_op",
        "rows_skipped",
        "rows_in_source",
        "entities_moved",
        "identity_moved",
        "references_ambiguous",
    )

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        self.records = 0
        self.chunks = 0
        self.entities = 0
        self.references = 0
        self.content_hash = ""
        self.edges = 0
        self.chunks_deleted = 0
        self.entities_deleted = 0
        self.replaced_existing = False
        self.no_op = False
        self.rows_skipped = 0
        self.rows_in_source = 0
        # Nodes the write moved to the id their name now gives them: a renamed
        # row, or a placeholder a row filled in. The map is old id -> new id,
        # for the caller whose bookkeeping still holds the old ones.
        self.entities_moved = 0
        self.identity_moved: dict[str, str] = {}
        # Foreign keys more than one existing node answers to. Each was left on
        # a placeholder carrying the key rather than attached to whichever node
        # the graph listed last; the (label, key) pairs are what to look at.
        self.references_ambiguous: list[tuple[str, str]] = []

    def as_dict(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "document_id": self.document_id,
            "records": self.records,
            "chunks": self.chunks,
            "entities": self.entities,
            "references": self.references,
            "edges": self.edges,
        }
        # Only when there is something to say. A caller reading a clean load
        # should not have to check a zero, and a caller reading a lossy one
        # should not have to know to look.
        if self.rows_skipped:
            summary["rows_skipped"] = self.rows_skipped
            summary["rows_in_source"] = self.rows_in_source
        if self.entities_moved:
            summary["entities_moved"] = self.entities_moved
            summary["identity_moved"] = dict(self.identity_moved)
        if self.references_ambiguous:
            summary["references_ambiguous"] = [
                f"{label}:{key}" for label, key in self.references_ambiguous
            ]
        if self.replaced_existing:
            summary["replaced_existing"] = True
            summary["chunks_deleted"] = self.chunks_deleted
            summary["entities_deleted"] = self.entities_deleted
            summary["no_op"] = self.no_op
        return summary

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StructuredIngestionResult({self.as_dict()})"


def _warn_about_a_column_typed_narrower_than_declared(
    batch: RecordBatch, mapping: RecordMapping, source: str
) -> None:
    """Say something when a column declared STRING holds only numbers or dates.

    ``properties={"age": "age"}`` is the documented shorthand, and it means
    STRING — so the column that motivated this whole feature ("run a CSV through
    the prose path and ``age`` becomes the string 34, so nothing can be
    averaged") lands as a string anyway, from a declaration that looks right.
    The load succeeds, finalize reports nothing, and the truth arrives much later
    as a Cypher type error on an aggregate.

    Declared types are deliberately not inferred: a column that holds integers
    for five hundred rows and "N/A" on the next one would be silently retyped and
    then fail the load it was meant to describe. But the file is open right here,
    so the mismatch is measurable, and saying so costs one sampled pass.
    """
    declared_strings: dict[str, str] = {}
    for node in mapping.nodes:
        for prop, column in node.typed_properties.items():
            if column.type == "STRING":
                declared_strings[column.name] = prop
    for edge in mapping.edges:
        for prop, column in edge.typed_properties.items():
            if column.type == "STRING":
                declared_strings[column.name] = prop
    if not declared_strings:
        return

    try:
        # The WHOLE file, not a sample -- the same reason natural_mapping does.
        # Sampled, this recommends INTEGER for a column that is clean for 500 rows
        # and holds "N/A" on row 501, and following that advice makes the next
        # load raise. Advice that breaks the thing it advises about is worse than
        # silence. The file is already walked twice here, so this costs nothing.
        profiles = profile_columns(batch, sample_rows=batch.record_count or _WHOLE_FILE)
    except Exception:  # a warning is never worth failing an ingest over
        logger.debug("Column profiling for the type-narrowing check failed", exc_info=True)
        return

    narrower = [
        (profile.name, declared_strings[profile.name], profile.inferred_type)
        for profile in profiles
        if profile.name in declared_strings and profile.inferred_type != "STRING"
    ]
    if not narrower:
        return
    detail = "; ".join(
        f"{column!r} (declared as {prop!r}) reads as {measured}"
        for column, prop, measured in narrower
    )
    logger.warning(
        "%s: %d column(s) declared STRING hold only one narrower type: %s. A STRING "
        "column cannot be averaged, compared numerically or filtered by date, and "
        "text-to-Cypher will not try. Declare the type to get that: "
        "properties={%r: Column(%r, %r)}. Leave it as it is if the column really is "
        "text that happens to look numeric, such as a zero-padded code.",
        source,
        len(narrower),
        detail,
        narrower[0][1],
        narrower[0][0],
        narrower[0][2],
    )


class StructuredIngestionPipeline(LexicalGraphWriter):
    """Writes a structured source into the graph, deterministically.

    Executes a fixed sequence, none of which consults a model:

    1. **Load** — parse the source into a re-openable record stream
    2. **Validate** — check the mapping against the source's real header
    3. **Lexical graph** — a Document, and one Chunk per record
    4. **Map** — declared columns become typed entity nodes and RELATES edges
    5. **Write** — entities, then references (keyed always, named ON CREATE),
       then edges

    The same input always produces the same graph, because identity comes from a
    declared key and every property type is declared rather than inferred.

    Args:
        loader: How to read the source into records.
        graph_store: Storage layer.
    """

    def __init__(self, loader: RecordLoaderStrategy, graph_store: Any) -> None:
        self.loader = loader
        self.graph_store = graph_store

    async def run(
        self,
        source: str,
        mapping: RecordMapping,
        ctx: Context | None = None,
        *,
        document_id: str | None = None,
        strict: bool = False,
        link_sequential: bool = False,
    ) -> StructuredIngestionResult:
        """Ingest one structured source.

        Args:
            source: Path or identifier for the source.
            mapping: The declaration.
            ctx: Execution context.
            document_id: Overrides the Document node id.
            strict: Fail when the source has a column the mapping never reads.
                Off by default because ignoring a column is a legitimate choice.
            link_sequential: Whether to chain NEXT_CHUNK between records. False,
                and that is not cosmetic: records have no reading order, and
                text-to-Cypher is told NEXT_CHUNK means "the next sequential
                Chunk", so chaining unrelated rows asserts a sequence that does
                not exist.

        Raises:
            MappingError: If the mapping does not fit the source. Nothing is
                written in that case.
        """
        ctx = ctx or Context()
        batch = await self.loader.load_records(source, ctx)
        doc_info = batch.document_info
        # The Document remembers how it was written, and update() refuses to
        # re-read a table as prose on the strength of it. Stamped here so a
        # custom loader that says nothing about it still produces a Document
        # that can be re-synced. It also remembers *which* table, by the
        # signature the table's columns are stored under, so a Document loaded
        # under a caller's own id is still found by the table's cleanup and by
        # drop_table(): matching the id to the table's filename found neither.
        stamped: dict[str, Any] = {**dict(doc_info.metadata), "kind": "structured"}
        if mapping.anchor.signature:
            stamped["table"] = mapping.anchor.signature
        doc_info = DocumentInfo(
            uid=document_id or doc_info.uid, path=doc_info.path, metadata=stamped
        )

        problems = mapping.validate_against(batch.columns, strict=strict)
        if problems:
            raise MappingError(f"mapping does not fit {source}:\n  " + "\n  ".join(problems))

        _warn_about_a_column_typed_narrower_than_declared(batch, mapping, source)

        result = StructuredIngestionResult(doc_info.uid)

        # Steps 3 to 5, all in memory. Both passes complete before anything is
        # written, because a declared type that does not hold raises from the
        # second one: `column 'age' declares INTEGER but holds 'N/A'` in row
        # 30,000 of an export. Writing as we went left the Document, a chunk for
        # every row and the content hash committed with no entities at all — and
        # because the document record then existed, the retry after fixing the
        # cell routed through update() and compared hashes instead of writing,
        # so the graph stayed broken with nothing raised. The docs promise a
        # failure "leaves the graph untouched"; this is what makes that true.
        chunks = self._record_chunks(batch, mapping, doc_info, result)
        if not chunks.chunks:
            # Still a state of the table — the one with no rows. The Document is
            # written with its hash so a re-sync to it completes its cutover and
            # takes the previous rows out, instead of failing on a pending
            # Document that was never written.
            ctx.log(f"{source} produced no records; Document written, nothing else")
            await self._build_lexical_graph(
                doc_info, chunks, ctx, content_hash=result.content_hash, link_sequential=False
            )
            return result
        nodes, references, edges, keyed = self._map_records(batch, mapping, doc_info, result)

        # A foreign key arriving AFTER its target's source must attach to the real
        # node, not raise a placeholder beside it. Look every reference up by its
        # key; where the target already exists, point the edges there instead —
        # and the reference too, so its claim still lands: the signed key column
        # is how the graph knows this table points at that node, and what a
        # re-sync or a drop reads to decide whose row the node still is.
        #
        # A reference to a row this batch writes is left out of the lookup. The
        # graph still holds that row under last export's id, and the lookup would
        # point the reference -- and its edges -- back at it, just before the
        # reconciliation below moves the row to its new id: measured as Bob
        # reporting to a recreated placeholder of Alice's old name beside the
        # renamed Alice. The collapse in _map_records already sent such a
        # reference to the row's id; the row is where it must stay.
        owned_here = {(label, key) for (label, _), rows in keyed.items() for key, _ in rows}
        resolved: dict[str, str] = {}
        by_label: dict[str, list[ReferenceNode]] = {}
        for reference in references:
            if _reference_key(reference) in owned_here:
                continue
            by_label.setdefault(reference.label, []).append(reference)
        for label, refs in by_label.items():
            found = await self.graph_store.resolve_by_entity_key(
                label, [str(r.properties.get("entity_key", "")) for r in refs]
            )
            for reference in refs:
                key = str(reference.properties.get("entity_key", ""))
                candidates = found.get(key, [])
                if len(candidates) > 1:
                    # Two tables numbered this label from 1, and this key means a
                    # different row in each. The edge stays on a placeholder that
                    # says only "the one keyed thus"; picking a node would pick one
                    # table's row for the other's link.
                    if (label, key) not in result.references_ambiguous:
                        result.references_ambiguous.append((label, key))
                    continue
                if candidates and candidates[0] != reference.id:
                    resolved[reference.id] = candidates[0]
        if result.references_ambiguous:
            logger.warning(
                "%s: %d foreign key(s) match more than one existing node and were left "
                "on placeholders: %s",
                source,
                len(result.references_ambiguous),
                ", ".join(f"{label}:{key}" for label, key in result.references_ambiguous),
            )
        if resolved:
            references = [r._replace(id=resolved.get(r.id, r.id)) for r in references]
            for edge in edges:
                edge.start_node_id = resolved.get(edge.start_node_id, edge.start_node_id)
                edge.end_node_id = resolved.get(edge.end_node_id, edge.end_node_id)

        # The other direction: a placeholder this source's rows now name, or a row
        # whose name changed since the last export. Both carry the key; both are
        # moved to the id the name gives them before anything is written.
        for (label, signed_key), rows in keyed.items():
            result.identity_moved.update(
                await self.graph_store.reconcile_keyed_identity(label, signed_key, rows)
            )
        result.entities_moved = len(result.identity_moved)

        await self._build_lexical_graph(
            doc_info,
            chunks,
            ctx,
            content_hash=result.content_hash,
            link_sequential=link_sequential,
        )
        await self.graph_store.upsert_nodes(nodes)
        await self._retract_blank_cells(nodes)
        # One entry per row arrives here, but a foreign key repeats: 25k rows
        # pointing at 50 organizations produced 25k MERGEs for 50 nodes. They are
        # idempotent, so every repeat after the first is pure waste.
        await self.graph_store.upsert_reference_nodes(_unique_references(references))
        await self.graph_store.upsert_relationships(edges)

        ctx.log(
            f"Structured ingest of {doc_info.uid}: {result.records} records, "
            f"{result.entities} entities, {result.references} references, "
            f"{result.edges} edges"
        )
        return result

    async def _retract_blank_cells(self, nodes: list[GraphNode]) -> int:
        """Take a column off the rows whose cell is empty in this export.

        ``upsert_nodes`` adds what a row carries and leaves the rest alone, so a
        cell that held a value last time and is blank now would keep the old
        value on the node. The table is the only writer of its signed columns;
        an empty cell means the column is not there, and after a re-sync the
        node has to say so too. Grouped by column, so a batch costs one query
        per column that has a blank in it.
        """
        blank: dict[tuple[str, str], list[str]] = {}
        for node in nodes:
            for prop, value in node.properties.items():
                if value is None:
                    blank.setdefault((node.label, prop), []).append(node.id)
        removed = 0
        for (label, prop), ids in blank.items():
            removed += await self.graph_store.drop_node_property(label, prop, ids=ids)
        return removed

    # ── internals ───────────────────────────────────────────────

    def _record_chunks(
        self,
        batch: RecordBatch,
        mapping: RecordMapping,
        doc_info: DocumentInfo,
        result: StructuredIngestionResult,
    ) -> TextChunks:
        """One TextChunk per record, carrying the cells alongside the text.

        Digests the records on the way past, so the content hash costs no extra
        pass over the source. The digest covers the mapping too: identical rows
        under a changed declaration produce a different graph, so treating them
        as unchanged would skip an update that was needed.
        """
        digest = _new_record_digest(mapping, batch.columns)
        for record in batch:
            _feed_record(digest, batch.columns, record)
        result.content_hash = digest.hexdigest()

        chunks: list[TextChunk] = []
        key_counts: dict[str, int] = {}
        skipped: list[int] = []
        rows_seen = 0
        for index, record, record_key, chunk_uid in _walk_records(
            batch, mapping, doc_info.uid, skipped
        ):
            rows_seen += 1
            key_counts[record_key] = key_counts.get(record_key, 0) + 1
            chunks.append(
                TextChunk(
                    uid=chunk_uid,
                    text=render_record(record),
                    index=index,
                    metadata={
                        "kind": "record",
                        "record_key": record_key,
                        **record_cells(record),
                    },
                )
            )
        result.records = result.chunks = len(chunks)
        result.rows_skipped = len(skipped)
        result.rows_in_source = rows_seen + len(skipped)

        if skipped:
            # The counts alone cannot carry this: "records: 3" for a four-row
            # file is a true statement about what was written and a false one
            # about what the file said, and nothing in the result distinguished
            # the two. A dropped row is not recoverable later, because the next
            # re-sync reads the same blank cell and drops it again.
            shown = ", ".join(str(line) for line in skipped[:10])
            more = f" and {len(skipped) - 10} more" if len(skipped) > 10 else ""
            logger.warning(
                "%s: %d of %d rows have no value in the declared key column %r "
                "and were not loaded (row %s%s). A row without a key has no "
                "stable identity, so it cannot be updated or deleted later. "
                "Give those rows a key, or declare a column that is always "
                "present.",
                doc_info.uid,
                len(skipped),
                result.rows_in_source,
                mapping.anchor.key,
                shown,
                more,
            )

        repeated = {key: count for key, count in key_counts.items() if count > 1}
        if repeated:
            # Each row still gets its own chunk, so no cells are lost. But the
            # key identifies the *entity*, so repeated keys mean several rows
            # describe one entity and only one row's values survive on it. Which
            # one is not worth promising: within a write batch FalkorDB keeps the
            # first, across batches the later one wins. Usually a repeated key
            # means the wrong column was declared, which is worth saying out loud
            # rather than leaving to be discovered.
            sample = ", ".join(f"{key!r} x{count}" for key, count in sorted(repeated.items())[:5])
            logger.warning(
                "%s: column %r is the declared key but is not unique (%d repeated "
                "values, e.g. %s). Every row is kept as its own chunk, but rows "
                "sharing a key describe one entity, so only one row's values "
                "survive on it.",
                doc_info.uid,
                mapping.anchor.key,
                len(repeated),
                sample,
            )
        return TextChunks(chunks=chunks)

    def _map_records(
        self,
        batch: RecordBatch,
        mapping: RecordMapping,
        doc_info: DocumentInfo,
        result: StructuredIngestionResult,
    ) -> tuple[
        list[GraphNode],
        list[ReferenceNode],
        list[GraphRelationship],
        dict[tuple[str, str], list[tuple[str, str]]],
    ]:
        """Second pass: declared columns become nodes and edges.

        This is why RecordBatch hands over a factory. A one-shot iterator is
        empty by the time this pass runs, and nothing raises.
        """
        # Normalised once per source rather than once per record: for a large
        # table the difference is millions of redundant dict builds.
        node_columns = {node.handle: node.typed_properties for node in mapping.nodes}
        edge_columns = {
            (edge.type, edge.source, edge.target): edge.typed_properties for edge in mapping.edges
        }
        nodes: list[GraphNode] = []
        references: list[ReferenceNode] = []
        edges: list[GraphRelationship] = []
        # (label, signed key property) -> [(entity_key, node_id)] for every real
        # node, so run() can point placeholders and renamed rows at the id the
        # name now gives them — and only rows this table wrote.
        keyed: dict[tuple[str, str], list[tuple[str, str]]] = {}

        # An entity's id comes from its NAME — the same derivation a prose mention
        # gets — so a row and a document describing one thing land on one node
        # with no merge step. Two rows sharing a name would then collapse into
        # one entity, so those fall back to the key: a mention of an ambiguous
        # name should not auto-join either row, and this way it cannot.
        ambiguous = _ambiguous_names(batch, mapping)

        for _index, record, _record_key, chunk_uid in _walk_records(batch, mapping, doc_info.uid):
            ids: dict[str, str] = {}

            for node in mapping.nodes:
                # Stripped, as the row key already is in _walk_records: the id
                # derivation ignores surrounding whitespace, so a key stored
                # with it would derive the same node as one stored without yet
                # never match it by entity_key.
                raw_key = cell_text(record, node.key)
                if not raw_key:
                    continue
                display = cell_text(record, node.name)
                if display and (node.label, display.lower()) not in ambiguous:
                    node_id = compute_entity_id(display, node.label)
                else:
                    node_id = compute_entity_id(raw_key, node.label)
                if not node_id:
                    continue
                ids[node.handle] = node_id
                if node.reference:
                    # A record that denormalises the referenced entity's name
                    # ("org_id" plus "org_name") can label the stub properly.
                    # Falling back to the raw key would name a node "ORG-42",
                    # which then resolves against nothing and reads as a real
                    # name to whoever queries it.
                    fallback = raw_key
                    if node.name:
                        fallback = cell_text(record, node.name) or raw_key
                    references.append(
                        ReferenceNode(
                            id=node_id,
                            label=node.label,
                            name=fallback,
                            properties={
                                node.signed(node.key_property): raw_key,
                                # Unsigned and SDK-owned, like is_stub: the value
                                # this row is keyed by, whoever wrote it. It is
                                # how the owning source finds this placeholder.
                                "entity_key": raw_key,
                                "is_stub": True,
                            },
                        )
                    )
                    result.references += 1
                else:
                    keyed.setdefault((node.label, node.signed(node.key_property)), []).append(
                        (raw_key, node_id)
                    )
                    nodes.append(
                        GraphNode(
                            id=node_id,
                            label=node.label,
                            properties=self._node_properties(
                                node, record, raw_key, node_columns[node.handle]
                            ),
                        )
                    )
                    result.entities += 1
                edges.append(
                    GraphRelationship(
                        start_node_id=node_id,
                        end_node_id=chunk_uid,
                        type="MENTIONED_IN",
                    )
                )

            for edge in mapping.edges:
                start, end = ids.get(edge.source), ids.get(edge.target)
                if not start or not end or start == end:
                    # A self loop means both aliases resolved to the same entity,
                    # which is a fact about the data, not an edge worth writing.
                    continue
                properties: dict[str, Any] = {
                    "rel_type": edge.type,
                    "fact": f"({edge.source}, {edge.type}, {edge.target})",
                    "source_chunk_ids": [chunk_uid],
                }
                for prop, column in edge_columns[(edge.type, edge.source, edge.target)].items():
                    value = column.cast(record.get(column.name))
                    if value is not None:
                        # Signed, exactly as node properties are. Two tables
                        # declaring one edge property used to overwrite each
                        # other here, silently and irreversibly.
                        properties[edge.signed(prop)] = value
                edges.append(
                    GraphRelationship(
                        start_node_id=start,
                        end_node_id=end,
                        type="RELATES",
                        properties=properties,
                    )
                )
                result.edges += 1

        references = _collapse_same_key_references(references, edges, keyed)
        return nodes, references, edges, keyed

    @staticmethod
    def _node_properties(
        node: NodeMapping,
        record: dict[str, Any],
        raw_key: str,
        columns: dict[str, Column],
    ) -> dict[str, Any]:
        """Typed properties for one entity, plus the alias that lets it resolve."""
        # Signed with the declaring source, so two tables describing one entity
        # write hr__age and finance__age rather than racing on age. `is_stub`,
        # `entity_key` and `name` stay unsigned: the SDK owns them, and an
        # extracted property is unsigned by construction, which is what makes a
        # collision between prose and a table impossible rather than guarded.
        properties: dict[str, Any] = {
            node.signed(node.key_property): raw_key,
            "entity_key": raw_key,
            "is_stub": False,
        }
        if node.name:
            display = record.get(node.name)
            if display not in (None, ""):
                properties["name"] = str(display)
        for prop, column in columns.items():
            # An empty cell is carried as ``None``: the write skips it, and
            # ``_retract_blank_cells`` takes the column off the node, so a cell
            # cleared between two exports does not keep its old value.
            properties[node.signed(prop)] = column.cast(record.get(column.name))
        return properties
