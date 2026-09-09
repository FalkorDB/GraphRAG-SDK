# GraphRAG SDK — Ingestion: reading a table nobody declared a mapping for
#
# A mapping is normally written by hand, because choosing which real-world thing
# a row describes is a judgement and getting it wrong splits one entity across
# two labels. When nobody wrote one, the judgement is asked of the model —
# once per table, never per row — and everything the model says is checked
# against the file before it is used: the key it picks must actually be unique,
# the types it picks must actually parse, the columns it names must exist. The
# model chooses; the data has the last word.
#
# What is measured rather than asked: a column's type and which columns identify
# a row. Both are read over the whole file, not a sample, because a column that
# holds integers for five hundred rows and "N/A" on row five hundred and one
# would be declared INTEGER from a sample and then fail the load it describes.
#
# Without a model, or when the model cannot produce an acceptable answer, the
# file is read as-is: every column a typed property, the label from the file
# name, no name column and so no join. Deliberately unjoined, because a wrong
# guess about identity attaches rows to the wrong entities and looks like it
# worked; finalize() reports the table either way, so the user sees what the
# graph is running on and can declare a mapping to change it.

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from graphrag_sdk.core.models import Ontology
from graphrag_sdk.core.providers.base import LLMInterface
from graphrag_sdk.core.tables import RESERVED_PROPERTY_NAMES, Link
from graphrag_sdk.ingestion.loaders.record_loader import RecordBatch, cell_text
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
    parses_as: frozenset[str] = frozenset({"STRING"})
    """Every declared type all filled values read as. STRING always; LIST always,
    since any cell is a list of one. What a proposed type is checked against."""

    @property
    def is_unique(self) -> bool:
        """Every filled value distinct, and nothing missing."""
        return self.total > 0 and self.filled == self.total and self.distinct == self.total


# A leading zero followed by a digit: "02134", "00501". A number never keeps
# one, so the value is a code, and reading it as a number would drop the zero.
_ZERO_PADDED = re.compile(r"^\s*[-+]?0\d")


def _types_holding(values: list[str]) -> frozenset[str]:
    """Every declared type all of ``values`` parse as.

    Read from the data rather than guessed, and deliberately strict: one
    unparseable value drops the type, because a declared type is enforced at
    ingest and a wrong one fails the load. STRING and LIST always hold.
    """
    filled = [value for value in values if value not in (None, "")]
    holding = {"STRING", "LIST"}
    if not filled:
        return frozenset(holding)
    # Zip codes, phone numbers and padded account codes parse as numbers and
    # lose their leading zeros on the way. That is a code, not a quantity.
    padded = any(_ZERO_PADDED.match(str(value)) for value in filled)
    for candidate in _TYPE_ORDER:
        if candidate == "STRING" or (padded and candidate in ("INTEGER", "FLOAT")):
            continue
        probe = Column("probe", candidate)
        try:
            for value in filled:
                probe.cast(value)
        except Exception:
            continue
        holding.add(candidate)
    return frozenset(holding)


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
            seen[column].append(cell_text(record, column))

    profiles = []
    for column, values in seen.items():
        filled = [value for value in values if value.strip()]
        holding = _types_holding(values)
        profiles.append(
            ColumnProfile(
                name=column,
                inferred_type=next(t for t in _TYPE_ORDER if t in holding),
                filled=len(filled),
                total=total,
                distinct=len(set(filled)),
                samples=[value for value in filled[:5]],
                parses_as=holding,
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
        property_name = _unclaimed(safe_property_name(profile.name), properties)
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
            source=table_name(source),
            label=label,
            key=key_profile.name,
            properties=properties,
            standalone=True,
            derived=True,
            description=f"Read as-is from {source}; no mapping was declared.",
        ),
        notes,
    )


def _unclaimed(property_name: str, taken: dict[str, Column | str]) -> str:
    """``property_name``, or the first ``property_name_2``, ``_3``... not in ``taken``.

    ``safe_property_name`` is not injective: ``HQ Country`` and ``hq-country``
    both store as ``col_hq_country``, and the dict assignment silently kept the
    later column and lost the earlier one. Suffixed in header order so the same
    export always proposes the same names.
    """
    if property_name not in taken:
        return property_name
    ordinal = 2
    while f"{property_name}_{ordinal}" in taken:
        ordinal += 1
    return f"{property_name}_{ordinal}"


def table_name(source: str) -> str:
    """The name the ontology knows a table by: the file's basename.

    A derived mapping is stored under this rather than the path it was read
    from, so that the declaration a user later writes — ``TableMapping(
    source="employees.csv", ...)``, the way every example spells it — lands on
    the same stored mapping and replaces it, instead of sitting beside it as a
    second table with the same signature.
    """
    return os.path.basename(os.path.normpath(source))


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


# ── Asking the model ────────────────────────────────────────────────────────
#
# One call per table. The model is shown what was measured — every column with
# its type, fill and uniqueness, the first rows, and what the graph already
# holds — and asked the one thing measurement cannot settle: what a row is
# *about*. Its answer is structured output, validated by building the actual
# TableMapping from it, and any error is fed back for a retry.

_TYPE_NAMES = Literal["STRING", "INTEGER", "FLOAT", "BOOLEAN", "DATE", "LIST"]

# Rows shown verbatim in the prompt, on top of the per-column profile.
_PROMPT_ROWS = 5


class ProposedProperty(BaseModel):
    """One column kept as a typed property of the row's entity."""

    column: str = Field(description="The column, exactly as it appears in the header.")
    property: str | None = Field(
        default=None,
        description="Property name to store it under. Omit to use the column name.",
    )
    type: _TYPE_NAMES = Field(default="STRING", description="The value type.")


class ProposedLink(BaseModel):
    """One column that points at another entity, and the edge to it."""

    column: str = Field(description="The column holding the other entity's key or name.")
    type: str = Field(description="Relationship type, UPPER_SNAKE_CASE, e.g. WORKS_AT.")
    to: str = Field(description="Label of the entity the column points at.")
    name_column: str | None = Field(
        default=None,
        description="Column holding the target's display name, if the row carries one.",
    )


class MappingProposal(BaseModel):
    """What the model says a table is. Shaped exactly like a TableMapping."""

    label: str = Field(description="The kind of thing each row describes, e.g. Person.")
    name: str | None = Field(
        default=None,
        description=(
            "The column holding the display name of the thing a row is about — the "
            "name a document would mention it by. null when no column is one."
        ),
    )
    key: str | None = Field(
        default=None,
        description=(
            "The column that uniquely identifies a row, if it differs from name. "
            "null when the name is unique on its own."
        ),
    )
    properties: list[ProposedProperty] = Field(default_factory=list)
    links: list[ProposedLink] = Field(default_factory=list)
    reasoning: str = Field(default="", description="One sentence on why this reading.")


_SYSTEM_PROMPT = """\
You map a table onto a knowledge graph. Every row of the table describes one \
thing. Decide what kind of thing (its label), which column holds that thing's \
name, which column identifies the row, what type each remaining column holds, and \
which columns point at other things in the graph.

Rules:
- Prefer a label that is already in the graph when the rows describe the same \
kind of thing. Reuse the existing label exactly; do not invent Employee when Person \
exists and holds people.
- `name` is the column a document would call the thing by (a person's full name, \
a company's name). It is what joins a row to a mention in prose. Set it whenever \
such a column exists; leave it null when no column is a name.
- `key` is a column that is unique and filled on every row. Leave it null when the \
name column is unique on its own, or when there is no name.
- A column that holds another table's key, or another entity's name, is a link, \
not a property. Its `to` must be a label already in the graph, one of the mapped \
tables' labels, or this table's own label. Do not link to a label that does not exist.
- Name a link's relationship type for what the column means from this row's side: a \
project's lead column is LED_BY, an order's customer column is PLACED_BY. Reuse a \
relationship type already in the graph only when it means exactly that; otherwise \
name a new one.
- Every other column is a property. Keep the column's meaning; use the measured type \
unless it is clearly wrong.
- Do not list the name column, the key column or a link column under properties.
- Reply with one JSON object and nothing else — no prose, no code fences.\
"""


def _proposal_prompt(
    source: str,
    profiles: list[ColumnProfile],
    rows: list[dict[str, Any]],
    ontology: Ontology,
    entity_counts: dict[str, int],
) -> str:
    lines = [f"Table: {table_name(source)}", "", "Columns, measured over the whole file:"]
    for profile in profiles:
        if profile.is_unique:
            fill = "unique, filled on every row"
        else:
            fill = f"{profile.filled}/{profile.total} filled, {profile.distinct} distinct"
        shown = ", ".join(repr(value) for value in profile.samples[:3])
        lines.append(f"- {profile.name}: {profile.inferred_type}; {fill}; e.g. {shown}")

    if rows:
        columns = [profile.name for profile in profiles]
        lines += ["", "First rows:", ",".join(columns)]
        for row in rows:
            lines.append(",".join(cell_text(row, column) for column in columns))

    entities = sorted(ontology.entities, key=lambda entity: -entity_counts.get(entity.label, 0))
    if entities:
        lines += ["", "Labels already in the graph (entity count, then properties):"]
        for entity in entities:
            props = ", ".join(prop.name for prop in entity.properties if not prop.structured)
            count = entity_counts.get(entity.label, 0)
            # The description is what lets the model judge "the same kind of thing":
            # a grants table is not an Experiment just because both have a title.
            about = f" — {entity.description}" if entity.description else ""
            lines.append(f"- {entity.label} ({count}): {props or 'no properties'}{about}")
    if ontology.relations:
        lines += ["", "Relationship types already in the graph:"]
        for relation in ontology.relations:
            patterns = ", ".join(f"{src} -> {tgt}" for src, tgt in relation.patterns) or "any"
            lines.append(f"- {relation.label}: {patterns}")
    if ontology.tables:
        lines += ["", "Tables already mapped (a column holding one of these keys is a link):"]
        for mapping in ontology.tables:
            lines.append(
                f"- {mapping.source} -> {mapping.label}, key column {mapping.key_column!r}"
                + (f", name column {mapping.name!r}" if mapping.name else "")
            )

    schema = MappingProposal.model_json_schema()
    lines += ["", "Reply with JSON matching this schema:", str(schema)]
    return "\n".join(lines)


def _mapping_from_proposal(
    proposal: MappingProposal,
    *,
    source: str,
    profiles: list[ColumnProfile],
    known_labels: set[str],
    model_name: str,
) -> tuple[TableMapping, list[str]]:
    """Build the TableMapping a proposal describes, or raise MappingError saying why not.

    Everything the model asserted about the *data* is checked against the
    profile: the columns exist, the key is unique, the types parse. A type the
    model narrowed past what the file holds is widened back to the measured one
    rather than refused — that is a fact about the file, not a judgement.
    """
    by_name = {profile.name: profile for profile in profiles}
    notes: list[str] = []

    def column(kind: str, value: str) -> ColumnProfile:
        profile = by_name.get(value)
        if profile is None:
            raise MappingError(
                f"{kind} names column {value!r}, which is not in the header. "
                f"Columns are: {', '.join(by_name)}"
            )
        return profile

    name = proposal.name or None
    key = proposal.key or None
    if name is not None:
        column("name", name)
    if key is not None:
        key_profile = column("key", key)
        if not key_profile.is_unique:
            raise MappingError(
                f"key {key!r} is not unique and complete: {key_profile.filled}/"
                f"{key_profile.total} filled, {key_profile.distinct} distinct. Pick a "
                f"column that identifies every row, or null if the name does."
            )
    if key is None and name is not None and not by_name[name].is_unique:
        raise MappingError(
            f"name {name!r} repeats or has gaps ({by_name[name].distinct} distinct in "
            f"{by_name[name].total} rows), so it cannot identify a row on its own. "
            f"Name a unique key column as well."
        )
    if key is None and name is None:
        raise MappingError(
            "neither name nor key is set, so no row can be given a stable identity. "
            "Set name to the column holding the row's name, or key to a unique column."
        )

    taken: dict[str, str] = {}
    for kind, value in (("name", name), ("key", key)):
        if value is not None:
            taken.setdefault(value, kind)

    links: list[Link] = []
    for link in proposal.links:
        column("link", link.column)
        if link.column in taken:
            raise MappingError(
                f"column {link.column!r} is used as both {taken[link.column]} and a link."
            )
        taken[link.column] = "link"
        if link.to not in known_labels and link.to != proposal.label:
            raise MappingError(
                f"link {link.type} points at {link.to!r}, which is not a label in the "
                f"graph. Known labels: {', '.join(sorted(known_labels)) or 'none'}. "
                f"Point at one of these, or keep the column as a property."
            )
        if link.name_column is not None:
            column("link name", link.name_column)
            taken.setdefault(link.name_column, "link name")
        links.append(Link(link.type, to=link.to, by=link.column, name=link.name_column))

    properties: dict[str, Column | str] = {}
    for prop in proposal.properties:
        profile = column("property", prop.column)
        if prop.column in taken:
            raise MappingError(
                f"column {prop.column!r} is listed as a property but is also the "
                f"{taken[prop.column]}; drop it from properties."
            )
        taken[prop.column] = "property"
        property_name = safe_property_name(prop.property or prop.column)
        if property_name in properties:
            raise MappingError(f"two columns are stored as property {property_name!r}")
        chosen = prop.type
        if chosen not in profile.parses_as:
            notes.append(
                f"{prop.column} kept as {profile.inferred_type}, not {chosen}: not every "
                f"value parses as {chosen}"
            )
            chosen = profile.inferred_type
        properties[property_name] = Column(prop.column, chosen)

    # A column the model left out is data the graph would silently lose. Kept,
    # at its measured type, and said so.
    for profile in profiles:
        if profile.name in taken:
            continue
        property_name = safe_property_name(profile.name)
        if property_name in properties or property_name in RESERVED_PROPERTY_NAMES:
            property_name = safe_property_name(f"col {profile.name}")
        properties[property_name] = Column(profile.name, profile.inferred_type)
        notes.append(f"{profile.name} was not mentioned; kept as {profile.inferred_type}")

    why = proposal.reasoning.strip()
    mapping = TableMapping(
        source=table_name(source),
        label=proposal.label,
        key=key,
        name=name,
        properties=properties,
        links=links,
        standalone=not links,
        derived=True,
        description=f"Proposed by {model_name} from {table_name(source)}"
        + (f": {why}" if why else "."),
    )
    return mapping, notes


async def propose_mapping(
    batch: RecordBatch,
    source: str,
    *,
    llm: LLMInterface,
    ontology: Ontology,
    entity_counts: dict[str, int] | None = None,
    max_retries: int = 2,
) -> tuple[TableMapping, list[str]]:
    """Ask the model what a table is, and hold it to the data.

    The model is shown the measured profile of every column, the first rows and
    the ontology as it stands, and asked for a :class:`MappingProposal`. The
    proposal is turned into a real :class:`TableMapping` — which is where every
    claim about the data is checked — and a proposal that does not survive that
    is sent back with the exact reason, up to ``max_retries`` times.

    Raises :class:`MappingError` when the model never produces an acceptable
    proposal, so the caller can fall back to :func:`natural_mapping`. The
    returned mapping is ``derived=True``: finalize() reports the table as
    running on a proposal, and a ``TableMapping`` the user later declares for
    the same source replaces it.
    """
    from graphrag_sdk.discovery.instructor import extract_with_retry
    from graphrag_sdk.discovery.proposal import OntologyDiscoveryError

    profiles = profile_columns(batch, sample_rows=batch.record_count or _WHOLE_FILE)
    rows: list[dict[str, Any]] = []
    for record in batch:
        rows.append(record)
        if len(rows) >= _PROMPT_ROWS:
            break

    known_labels = {entity.label for entity in ontology.entities}
    known_labels |= {mapping.label for mapping in ontology.tables}
    counts = entity_counts or {}
    outcome: dict[str, Any] = {}

    def check(proposal: MappingProposal) -> list[str]:
        try:
            outcome["mapping"], outcome["notes"] = _mapping_from_proposal(
                proposal,
                source=source,
                profiles=profiles,
                known_labels=known_labels,
                model_name=llm.model_name,
            )
        except MappingError as exc:
            return [str(exc)]
        return []

    try:
        await extract_with_retry(
            llm,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=_proposal_prompt(source, profiles, rows, ontology, counts),
            response_model=MappingProposal,
            extra_validate=check,
            max_retries=max_retries,
            chunk_id=f"table:{table_name(source)}",
        )
    except OntologyDiscoveryError as exc:
        raise MappingError(
            f"{llm.model_name} did not produce an acceptable mapping for {source} in "
            f"{exc.attempts} attempt(s): {exc.last_error}"
        ) from exc

    mapping: TableMapping = outcome["mapping"]
    notes: list[str] = list(outcome["notes"])
    notes.insert(
        0,
        f"label {mapping.label!r}"
        + (f", name {mapping.name!r}" if mapping.name else ", no name column")
        + (f", key {mapping.key!r}" if mapping.key != mapping.name else "")
        + (
            ", links " + ", ".join(f"{link.type}->{link.to} by {link.by}" for link in mapping.links)
            if mapping.links
            else ""
        ),
    )
    return mapping, notes
