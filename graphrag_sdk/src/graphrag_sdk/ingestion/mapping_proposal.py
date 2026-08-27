# GraphRAG SDK — Ingestion: proposing a mapping for a structured source
#
# Writing a mapping by hand means choosing a label by hand, and a label chosen
# by hand can invent `Person` while the graph already uses `Employee`. Nothing
# catches it, and the result is one human held as two nodes with the facts split
# between them.
#
# So a proposal is constrained rather than free: the label must come from the
# ontology that already exists. A genuinely new type can be requested, but it is
# surfaced for approval instead of appearing on its own.
#
# The model is asked to decide as little as possible. A key column, a column's
# type and a foreign key are all *measurable*, and measurement beats judgement
# wherever it reaches — an LLM guessing INTEGER is strictly worse than parsing
# the values. What is left for the model is the part no measurement answers:
# which existing concept this table describes, and what to call its
# relationships.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import Ontology
from graphrag_sdk.core.providers import LLMInterface
from graphrag_sdk.discovery.instructor import extract_with_retry
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import compute_entity_id
from graphrag_sdk.ingestion.loaders.record_loader import RecordBatch
from graphrag_sdk.ingestion.mapping import (
    COLUMN_TYPES,
    RESERVED_PROPERTY_NAMES,
    Column,
    Link,
    Table,
)
from graphrag_sdk.utils.cypher import sanitize_cypher_label

logger = logging.getLogger(__name__)

# How many rows to read for profiling. Enough to see whether a column is unique
# and what it holds; small enough that proposing costs nothing on a large table.
_DEFAULT_SAMPLE_ROWS = 500

# A column is only offered as a foreign key when most of its values already
# resolve to an entity of the candidate label. Below this it is coincidence.
_FK_MATCH_FLOOR = 0.6

_TYPE_ORDER = ("INTEGER", "FLOAT", "BOOLEAN", "DATE", "STRING")

# ``Table`` takes its properties as keyword arguments, so a column sharing a name
# with one of its own parameters would collide: a source with a column literally
# called "links" would raise TypeError rather than mapping. Such a column is left
# unmapped and reported, so it can be given a property name by hand.
_TABLE_PARAMETERS = frozenset({"node", "key", "name", "links", "description"})


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


@dataclass
class ForeignKeyCandidate:
    """A column whose values already resolve to entities of a known label."""

    column: str
    label: str
    matched: int
    checked: int

    @property
    def ratio(self) -> float:
        return self.matched / self.checked if self.checked else 0.0

    def describe(self) -> str:
        return (
            f"{self.column} → {self.label} "
            f"({self.matched}/{self.checked} sampled values already exist)"
        )


class _ProposedLink(BaseModel):
    """One relationship the model proposes for a foreign-key column."""

    type: str = Field(description="Relationship name in CAPS_WITH_UNDERSCORES, e.g. WORKS_AT")
    to: str = Field(description="The label being pointed at. Must be one offered.")
    by: str = Field(description="The column holding that entity's key. Must be one offered.")


class _ProposedMapping(BaseModel):
    """The narrow set of choices only judgement can make."""

    label: str = Field(description="Which existing ontology label this table's rows describe")
    name_column: str | None = Field(
        default=None, description="Column holding the display name, or null if there is none"
    )
    links: list[_ProposedLink] = Field(default_factory=list)
    new_label_reason: str | None = Field(
        default=None,
        description=(
            "Only set when no existing label fits. Explain what the rows are, and put the "
            "proposed new label in `label`."
        ),
    )


@dataclass
class MappingProposal:
    """A reviewable mapping for a structured source. Nothing is applied.

    ``table`` is usable as-is, but the point of the object is the rest of it:
    the evidence behind each mechanical choice, and whether the model had to ask
    for a type the ontology does not have.

    Attributes:
        table: The proposed mapping.
        source: The file it was proposed for.
        evidence: Why each part was chosen — measured facts, not assertions.
        warnings: Things a reviewer should look at, e.g. unmapped columns.
        requested_new_label: Set when no existing label fitted. The mapping is
            still returned, but applying it introduces a type to the ontology,
            so it wants a decision rather than a nod.
    """

    table: Table
    source: str
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    requested_new_label: str | None = None
    reason_for_new_label: str | None = None

    @property
    def introduces_a_new_type(self) -> bool:
        return self.requested_new_label is not None

    def as_code(self) -> str:
        """The mapping as Python, to commit rather than regenerate.

        A proposal regenerated on every run is a model in the ingest path, and
        the whole point of a declared mapping is that no model runs there. Commit
        the output and the load stays deterministic.
        """
        anchor = self.table.anchor
        lines = [f'Table("{anchor.label}",', f'      key="{anchor.key}",']
        if anchor.name:
            lines.append(f'      name="{anchor.name}",')
        for prop, column in anchor.typed_properties.items():
            if column.type == "STRING":
                lines.append(f'      {prop}="{column.name}",')
            else:
                lines.append(f'      {prop}=Column("{column.name}", "{column.type}"),')
        if self.table.edges:
            rendered = ", ".join(
                f'Link("{edge.type}", to="{target.label}", by="{target.key}")'
                for edge in self.table.edges
                for target in [self._node_for(edge.target)]
                if target is not None
            )
            lines.append(f"      links=[{rendered}],")
        lines.append("      )")
        return "\n".join(lines)

    def _node_for(self, handle: str) -> Any:
        for node in self.table.nodes:
            if node.handle == handle:
                return node
        return None

    def summary(self) -> str:
        """A short report for a human deciding whether to accept this."""
        out = [f"{self.source} → {self.table.anchor.label}"]
        out += [f"  {line}" for line in self.evidence]
        if self.requested_new_label:
            out.append(
                f"  ! proposes a NEW type {self.requested_new_label!r}: "
                f"{self.reason_for_new_label or 'no reason given'}"
            )
        out += [f"  ? {line}" for line in self.warnings]
        return "\n".join(out)


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


async def find_foreign_keys(
    profiles: list[ColumnProfile],
    ontology: Ontology,
    graph_store: Any,
    *,
    exclude: str | None = None,
) -> list[ForeignKeyCandidate]:
    """Columns whose values already resolve to entities in the graph.

    A foreign key is measurable, not a matter of opinion: an entity's id is
    derived from its key and its label, so the expected id for a value can be
    computed and looked up. That is an indexed point lookup rather than a guess,
    and it distinguishes a real reference from a column that merely looks like
    one because of its name.
    """
    labels = [entity.label for entity in ontology.entities]
    if not labels or graph_store is None:
        return []

    candidates: list[ForeignKeyCandidate] = []
    for profile in profiles:
        if profile.name == exclude or profile.inferred_type != "STRING":
            continue
        probes = [value for value in dict.fromkeys(profile.samples) if value.strip()][:5]
        if not probes:
            continue
        for label in labels:
            expected = [compute_entity_id(value, label) for value in probes]
            expected = [candidate_id for candidate_id in expected if candidate_id]
            if not expected:
                continue
            try:
                result = await graph_store.query_raw(
                    "UNWIND $ids AS wanted MATCH (n:__Entity__ {id: wanted}) RETURN count(n)",
                    {"ids": expected},
                )
            except Exception:
                logger.debug("foreign-key probe failed for %s → %s", profile.name, label)
                continue
            rows = getattr(result, "result_set", None) or []
            matched = int(rows[0][0]) if rows and rows[0] else 0
            if matched and matched / len(expected) >= _FK_MATCH_FLOOR:
                candidates.append(
                    ForeignKeyCandidate(
                        column=profile.name,
                        label=label,
                        matched=matched,
                        checked=len(expected),
                    )
                )
    return candidates


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


_SYSTEM_PROMPT = """You map a table onto an existing knowledge graph.

You are given a table's columns with real sample values, and the labels that
already exist in the graph's ontology. Decide only these things:

1. `label` — which EXISTING label the rows describe. Choose from the offered
   labels. Each is shown with how many entities it already holds, and a label
   already in use almost always beats an unused one that sounds more natural: if
   the graph already calls these things `Employee`, answer `Employee`, not
   `Person`, even though `Person` is offered and reads better. Answering the
   unused word creates a second set of entities for the same real-world things,
   which is the one outcome to avoid. Prefer an unused label only when no label
   in use plausibly describes these rows.
2. `name_column` — the column holding a human-readable display name, or null.
   This is what lets a row join a mention of the same thing in a document, so it
   should be the column a document would spell out, not an internal code.
3. `links` — for each foreign-key column offered to you, the relationship name
   in CAPS_WITH_UNDERSCORES, read in the direction row → target.

Only if NO offered label plausibly fits: put your suggested new label in
`label` and explain in `new_label_reason`. Prefer an existing label.

Do not choose the key column or any column types. Those are measured for you."""


def _user_prompt(
    source: str,
    profiles: list[ColumnProfile],
    ontology: Ontology,
    key: ColumnProfile | None,
    foreign_keys: list[ForeignKeyCandidate],
    label_counts: dict[str, int] | None = None,
) -> str:
    counts = label_counts or {}
    # Ordered by how much data each label holds, so the ones in use are read
    # first and the unused defaults trail behind them.
    ordered = sorted(ontology.entities, key=lambda e: (-counts.get(e.label, 0), e.label))
    label_lines = []
    for entity in ordered:
        described = f" — {entity.description}" if entity.description else ""
        held = counts.get(entity.label)
        in_use = f" [{held} entities already]" if held else " [unused]" if held == 0 else ""
        label_lines.append(f"  {entity.label}{in_use}{described}")

    fk_lines = [f"  {candidate.describe()}" for candidate in foreign_keys] or ["  (none detected)"]
    return (
        f"Table: {source}\n\n"
        f"Columns:\n" + "\n".join(f"  {profile.describe()}" for profile in profiles) + "\n\n"
        f"Measured key column: {key.name if key else 'none found'}\n\n"
        f"Foreign-key columns detected in the graph:\n" + "\n".join(fk_lines) + "\n\n"
        "Labels already in the ontology:\n" + "\n".join(label_lines) + "\n"
    )


async def propose_mapping(
    batch: RecordBatch,
    ontology: Ontology,
    llm: LLMInterface,
    *,
    graph_store: Any = None,
    source: str | None = None,
    sample_rows: int = _DEFAULT_SAMPLE_ROWS,
    max_retries: int = 3,
    ctx: Context | None = None,
) -> MappingProposal:
    """Propose a mapping for a structured source, against the ontology it must fit.

    One model call, at authoring time. Nothing is written and nothing is applied:
    the result is a :class:`MappingProposal` to review. Commit its ``as_code()``
    output and the ingest itself stays free of any model.
    """
    ctx = ctx or Context()
    source = source or batch.document_info.uid

    profiles = profile_columns(batch, sample_rows=sample_rows)
    key = pick_key(profiles)
    foreign_keys = await find_foreign_keys(
        profiles, ontology, graph_store, exclude=key.name if key else None
    )
    label_counts = await count_entities_per_label(ontology, graph_store)
    known_labels = {entity.label for entity in ontology.entities}

    def _check(choice: _ProposedMapping) -> list[str]:
        """Semantic checks the model must satisfy, fed back on failure."""
        problems: list[str] = []
        column_names = {profile.name for profile in profiles}
        if choice.label not in known_labels and not choice.new_label_reason:
            problems.append(
                f"label {choice.label!r} is not in the ontology. Choose one of "
                f"{sorted(known_labels)}, or set new_label_reason to explain why none fit."
            )
        if choice.name_column and choice.name_column not in column_names:
            problems.append(f"name_column {choice.name_column!r} is not a column of this table")
        offered = {(candidate.column, candidate.label) for candidate in foreign_keys}
        for link in choice.links:
            if (link.by, link.to) not in offered:
                problems.append(
                    f"link {link.type!r} uses by={link.by!r} to={link.to!r}, which was not "
                    f"offered. Offered: {sorted(offered) or 'none'}"
                )
        return problems

    if key is None:
        raise MappingProposalError(
            f"{source}: no column is unique and complete across the sample, so no column "
            "identifies a row. A structured source needs one; add an id column, or pass a "
            "mapping by hand naming the key you intend."
        )

    ctx.log(f"Proposing a mapping for {source}: {len(profiles)} columns, key {key.name!r}")
    choice = await extract_with_retry(
        llm,
        _SYSTEM_PROMPT,
        _user_prompt(source, profiles, ontology, key, foreign_keys, label_counts),
        _ProposedMapping,
        extra_validate=_check,
        max_retries=max_retries,
    )

    # Properties: every column that is not the key, the name, or a foreign key,
    # typed from the data rather than from the model.
    link_columns = {link.by for link in choice.links}
    properties: dict[str, Column | str] = {}
    warnings: list[str] = []
    for profile in profiles:
        if profile.name in {key.name, choice.name_column} or profile.name in link_columns:
            continue
        if profile.name in RESERVED_PROPERTY_NAMES or profile.name in _TABLE_PARAMETERS:
            warnings.append(
                f"column {profile.name!r} clashes with a name the mapping API reserves and "
                "was left unmapped; give it a property name by hand if you need it, e.g. "
                f'{profile.name}_value=Column("{profile.name}")'
            )
            continue
        if profile.inferred_type not in COLUMN_TYPES:  # pragma: no cover - defensive
            continue
        properties[profile.name] = Column(profile.name, profile.inferred_type)

    links = [Link(link.type, to=link.to, by=link.by) for link in choice.links]

    # Assembled as one mapping rather than as keyword arguments, because the
    # property names come from a file and only the guard above keeps them from
    # colliding with the parameters themselves.
    arguments: dict[str, Any] = {"key": key.name, "name": choice.name_column}
    if links:
        arguments["links"] = links
    arguments.update(properties)
    table = Table(choice.label, **arguments)

    problems = table.validate_against(batch.columns)
    if problems:  # pragma: no cover - the validator above should prevent this
        raise MappingProposalError(
            f"{source}: the proposed mapping does not fit the source:\n  " + "\n  ".join(problems)
        )

    evidence = [
        f"key {key.name!r} — unique and complete across {key.total} sampled rows",
        f"label {choice.label!r} — chosen from {len(known_labels)} in the ontology"
        + (
            f", which already holds {label_counts[choice.label]} entities"
            if label_counts.get(choice.label)
            else ", which held nothing before this"
        ),
    ]
    if choice.name_column:
        evidence.append(f"name {choice.name_column!r} — joins rows to mentions in documents")
    for candidate in foreign_keys:
        used = any(link.by == candidate.column and link.to == candidate.label for link in links)
        evidence.append(f"{candidate.describe()}{'' if used else ' — offered but not linked'}")
    for prop, column in table.anchor.typed_properties.items():
        evidence.append(f"{prop} {column.type} — every sampled value parses")

    unmapped = sorted(set(batch.columns) - table.columns)
    if unmapped:
        warnings.append("columns not mapped anywhere: " + ", ".join(unmapped))

    return MappingProposal(
        table=table,
        source=source,
        evidence=evidence,
        warnings=warnings,
        requested_new_label=(choice.label if choice.label not in known_labels else None),
        reason_for_new_label=choice.new_label_reason,
    )


class MappingProposalError(RuntimeError):
    """Raised when a source cannot be mapped without a human deciding something."""
