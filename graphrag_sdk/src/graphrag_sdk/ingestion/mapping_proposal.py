# GraphRAG SDK — Ingestion: reading a table nobody declared a mapping for
#
# A mapping is written by hand, because choosing which real-world thing a row
# describes is a judgement and getting it wrong splits one entity across two
# labels. Nothing here guesses at that.
#
# What is here is *measurement*, for the case where no mapping was declared at
# all. A column's type and which column identifies a row are both measurable —
# over the whole file, not a sample, because a column that holds integers for
# five hundred rows and "N/A" on row five hundred and one would be declared
# INTEGER from a sample and then fail the load it was meant to describe.
#
# The result is deliberately unjoined: no name column is guessed, so the rows
# land queryable and connected to nothing, and finalize() says so. A wrong
# guess about identity would instead attach rows to the wrong entities and look
# like it worked.

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from graphrag_sdk.core.models import Ontology
from graphrag_sdk.ingestion.loaders.record_loader import RecordBatch
from graphrag_sdk.ingestion.mapping import (
    Column,
    MappingError,
    TableMapping,
    safe_property_name,
)
from graphrag_sdk.utils.cypher import sanitize_cypher_label

logger = logging.getLogger(__name__)

# How many rows to read for profiling. Enough to see whether a column is unique
# and what it holds; small enough that proposing costs nothing on a large table.
_DEFAULT_SAMPLE_ROWS = 500

# A column is only offered as a foreign key when most of its values already

_TYPE_ORDER = ("INTEGER", "FLOAT", "BOOLEAN", "DATE", "STRING")


@dataclass
class ColumnProfile:
    """What reading the sample says about one column. No model involved."""

    name: str
    inferred_type: str
    filled: int
    total: int
    distinct: int
    samples: list[str] = field(default_factory=list)

    @property
    def is_unique(self) -> bool:
        """Every filled value distinct, and nothing missing."""
        return self.total > 0 and self.filled == self.total and self.distinct == self.total

    @property
    def looks_like_a_name(self) -> bool:
        """Mostly multi-word text, which is what a display name looks like."""
        if self.inferred_type != "STRING" or not self.samples:
            return False
        wordy = sum(1 for value in self.samples if " " in value.strip())
        return wordy >= max(1, len(self.samples) // 2)

    def describe(self) -> str:
        shown = ", ".join(repr(value) for value in self.samples[:3])
        note = " unique, no gaps" if self.is_unique else f" {self.filled}/{self.total} filled"
        return f"{self.name} ({self.inferred_type}{note}) e.g. {shown}"


def _infer_type(values: list[str]) -> str:
    """The narrowest declared type every value parses as.

    Read from the data rather than guessed, and deliberately conservative: one
    unparseable value in the sample drops the column to the next widest type,
    because a declared type is enforced at ingest and a wrong one fails the load.
    """
    filled = [value for value in values if value not in (None, "")]
    if not filled:
        return "STRING"
    for candidate in _TYPE_ORDER:
        if candidate == "STRING":
            break
        probe = Column("probe", candidate)
        try:
            for value in filled:
                probe.cast(value)
        except Exception:
            continue
        return candidate
    return "STRING"


def profile_columns(
    batch: RecordBatch, sample_rows: int = _DEFAULT_SAMPLE_ROWS
) -> list[ColumnProfile]:
    """Read a sample and describe every column. Deterministic, no model."""
    seen: dict[str, list[str]] = {column: [] for column in batch.columns}
    total = 0
    for record in batch:
        if total >= sample_rows:
            break
        total += 1
        for column in batch.columns:
            seen[column].append(str(record.get(column, "") or ""))

    profiles = []
    for column, values in seen.items():
        filled = [value for value in values if value.strip()]
        profiles.append(
            ColumnProfile(
                name=column,
                inferred_type=_infer_type(values),
                filled=len(filled),
                total=total,
                distinct=len(set(filled)),
                samples=[value for value in filled[:5]],
            )
        )
    return profiles


def natural_mapping(batch: RecordBatch, source: str) -> tuple[TableMapping, list[str]]:
    """The obvious reading of a table nobody declared a mapping for.

    ``ingest("employees.csv")`` with no mapping does not fail. Every column
    becomes a typed property, the label comes from the file name, and the key is
    the leftmost column that identifies a row. Returns the mapping and the notes
    worth reporting about how it was arrived at.

    **No name column, so no join.** The types here are *measured* — a column
    either parses as an integer across the whole file or it does not — but which
    column identifies the thing a row is *about* is a guess, and a wrong guess
    silently attaches rows to the wrong entities. That is the failure class this
    path exists to avoid, so the rows land queryable and unjoined, and the report
    names declaring a mapping as the way to connect them.

    Profiled over the **whole file**, not a sample: a column that holds integers
    for five hundred rows and ``"N/A"`` on row five hundred and one would be
    declared INTEGER from a sample and then fail the load it was meant to
    describe.
    """
    profiles = profile_columns(batch, sample_rows=batch.record_count or _WHOLE_FILE)
    notes: list[str] = []

    label = _label_from_source(source)
    key_profile = pick_key(profiles)
    if key_profile is None:
        # Nothing identifies a row, so a row cannot be updated or deleted later.
        # Refused rather than keyed on the ordinal, which would silently rebind
        # every row to a different entity the moment the export is re-sorted.
        raise MappingError(
            f"{source} has no column that is unique and complete, so no row can "
            f"be given a stable identity. Declare a mapping naming the key, or "
            f"add an id column to the export."
        )
    notes.append(
        f"key {key_profile.name!r} — unique and complete across {key_profile.total} row(s)"
    )

    properties: dict[str, Column | str] = {}
    for profile in profiles:
        if profile.name == key_profile.name:
            continue
        property_name = safe_property_name(profile.name)
        properties[property_name] = Column(profile.name, profile.inferred_type)
        if property_name != profile.name:
            notes.append(f"column {profile.name!r} stored as {property_name!r}")
    typed = [
        f"{name} {column.type}"
        for name, column in properties.items()
        if isinstance(column, Column) and column.type != "STRING"
    ]
    if typed:
        notes.append("measured types: " + ", ".join(sorted(typed)))
    notes.append(
        "no name column was declared, so these rows are not joined to anything "
        "in your documents. Declare a mapping with name=<column> to connect them."
    )
    return (
        TableMapping(
            source=source,
            label=label,
            key=key_profile.name,
            properties=properties,
            standalone=True,
            derived=True,
            description=f"Derived from {source}; no mapping was declared.",
        ),
        notes,
    )


def _label_from_source(source: str) -> str:
    """A label from the file name — ``employees.csv`` becomes ``employees``.

    Deliberately not title-cased into ``Employees``: that reads like a declared
    type someone chose, and this one was derived. Keeping the file's own casing
    makes it obvious in the browser which labels nobody declared.
    """
    stem = source.replace("\\", "/").rsplit("/", 1)[-1]
    if "." in stem:
        stem = stem[: stem.rindex(".")]
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"t_{cleaned}" if cleaned else "table"
    return cleaned


#: profile_columns takes a row count; this is "all of them" for a batch that did
#: not report one.
_WHOLE_FILE = 10_000_000


async def count_entities_per_label(ontology: Ontology, graph_store: Any) -> dict[str, int]:
    """How many entities each ontology label actually holds.

    Being *in* the ontology is not evidence that a label is the one in use. The
    built-in defaults seed a dozen labels into every graph, so a people table
    offered both ``Employee`` (which a previous source declared and filled) and
    ``Person`` (a default holding nothing) will be told to pick, and will
    reasonably pick the more obvious word — recreating the split this whole
    module exists to prevent.

    A count settles it without asking: prefer the label already carrying data.
    """
    counts: dict[str, int] = {}
    if graph_store is None:
        return counts
    for entity in ontology.entities:
        safe_label = sanitize_cypher_label(entity.label)
        try:
            result = await graph_store.query_raw(f"MATCH (n:`{safe_label}`) RETURN count(n)")
        except Exception:
            logger.debug("could not count entities for %s", entity.label)
            continue
        rows = getattr(result, "result_set", None) or []
        counts[entity.label] = int(rows[0][0]) if rows and rows[0] else 0
    return counts


def pick_key(profiles: list[ColumnProfile]) -> ColumnProfile | None:
    """The column that identifies a row, by measurement.

    Unique and complete across the sample. Where several qualify the leftmost
    wins, which is where an id column conventionally sits.
    """
    for profile in profiles:
        if profile.is_unique:
            return profile
    return None
