# GraphRAG SDK — Tools: result models + LLM-text rendering
# Typed results of the toolkit tools. Every model is pydantic v2, carries the
# evidence numbers [n] to cite, and renders bounded text via to_llm_text().

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from graphrag_sdk.storage.graph_store import GraphStore

_ELLIPSIS = "…"
_SNIPPET_CHARS = 240


def _clean(value: Any) -> str:
    """Sanitize a value for LLM output: strip control chars (reuses the
    ingestion sanitizer) and collapse all whitespace runs to single spaces."""
    return " ".join(GraphStore._sanitize_string(str(value)).split())


def _snippet(value: Any, limit: int = _SNIPPET_CHARS) -> str:
    """A cleaned, length-capped one-line excerpt ending with an ellipsis."""
    text = _clean(value)
    return text if len(text) <= limit else text[: limit - 1] + _ELLIPSIS


def _render(
    preamble: list[str],
    sections: list[tuple[str, list[str]]],
    *,
    max_chars: int,
) -> str:
    """Assemble preamble lines + (header, items) sections into text ≤ max_chars.

    Truncation happens only at item boundaries; a dropped tail is marked
    with the exact marker ``…(N more)``. Deterministic for equal inputs.
    """
    if max_chars < 1:
        return ""
    lines: list[str] = []
    used = 0

    def try_add(line: str) -> bool:
        nonlocal used
        cost = len(line) + (1 if lines else 0)
        if used + cost > max_chars:
            return False
        lines.append(line)
        used += cost
        return True

    for line in preamble:
        if not try_add(line):
            # Cut the line to the space left rather than dropping it, so a
            # long summary still shows its start.
            room = max_chars - used - (1 if lines else 0)
            if not lines or room > 40:
                lines.append(line[: max(room, 1) - 1] + _ELLIPSIS)
            return "\n".join(lines)

    for header, items in sections:
        if not items:
            continue
        header_line = f"{header} ({len(items)}):"
        full_marker = f"  {_ELLIPSIS}({len(items)} more)"
        # Only start a section if the header plus at least a drop-marker fit —
        # otherwise a bare header would dangle with nothing under it. Marker
        # strings never grow as the remaining count shrinks, so this reserve
        # guarantees every emitted header is followed by an item or a marker.
        needed = used + (1 if lines else 0) + len(header_line) + 1 + len(full_marker)
        if needed > max_chars:
            try_add(f"{header}: {_ELLIPSIS}({len(items)} items)")
            continue
        try_add(header_line)
        for idx, item in enumerate(items):
            marker = f"  {_ELLIPSIS}({len(items) - idx} more)"
            reserve = (len(marker) + 1) if idx < len(items) - 1 else 0
            if used + len(item) + 1 + reserve <= max_chars:
                try_add(item)
            else:
                try_add(marker)
                break
    return "\n".join(lines)


class ToolResult(BaseModel):
    """Base class for toolkit results: strict fields + LLM-text rendering."""

    model_config = ConfigDict(extra="forbid")

    def to_llm_text(self, *, max_chars: int = 4000) -> str:
        """Render a compact, deterministic plain-text form bounded by max_chars."""
        return _render(self._preamble(), self._sections(), max_chars=max_chars)

    def _preamble(self) -> list[str]:  # pragma: no cover - overridden
        return []

    def _sections(self) -> list[tuple[str, list[str]]]:  # pragma: no cover
        return []


def _n(n: int | None) -> str:
    return f"[{n}] " if n is not None else ""


class DocumentRef(BaseModel):
    """A source document reference."""

    model_config = ConfigDict(extra="forbid")
    document_id: str
    document_path: str = ""


class EntityCard(BaseModel):
    """A knowledge-graph entity with its user-facing properties."""

    model_config = ConfigDict(extra="forbid")
    n: int | None = None
    id: str = ""
    name: str
    label: str = ""
    description: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class RelationTriple(BaseModel):
    """A relation ``source -[type]-> target`` with the fact it states."""

    model_config = ConfigDict(extra="forbid")
    n: int | None = None
    source: str
    type: str
    target: str
    fact: str | None = None


class Fact(BaseModel):
    """A fact statement found by retrieval."""

    model_config = ConfigDict(extra="forbid")
    n: int | None = None
    text: str


class ChunkRef(BaseModel):
    """A retrieved source passage with provenance ids."""

    model_config = ConfigDict(extra="forbid")
    n: int | None = None
    chunk_id: str
    document_id: str = ""
    #: Omitted when it equals ``document_id``.
    document_path: str | None = None
    text: str


class EntityTypeInfo(BaseModel):
    """An entity label with its description, declared properties and live count."""

    model_config = ConfigDict(extra="forbid")
    label: str
    description: str | None = None
    count: int = 0
    properties: list[str] = Field(default_factory=list)


class RelationTypeInfo(BaseModel):
    """A relation type with its description, patterns and live count."""

    model_config = ConfigDict(extra="forbid")
    label: str
    description: str | None = None
    patterns: list[tuple[str, str]] = Field(default_factory=list)
    count: int = 0


def _entity_line(e: EntityCard) -> str:
    line = f"- {_n(e.n)}{_clean(e.name)}"
    if e.label:
        line += f" [{_clean(e.label)}]"
    if e.description:
        line += f": {_snippet(e.description, 160)}"
    return line


def _relation_line(r: RelationTriple) -> str:
    line = f"- {_n(r.n)}{_clean(r.source)} -[{_clean(r.type)}]-> {_clean(r.target)}"
    if r.fact:
        line += f": {_snippet(r.fact, 160)}"
    return line


class StatusResult(ToolResult):
    """What the graph holds."""

    entities: int = 0
    relations: int = 0
    documents: int = 0
    chunks: int = 0
    entity_types: list[EntityTypeInfo] = Field(default_factory=list)
    relation_types: list[RelationTypeInfo] = Field(default_factory=list)
    hint: str | None = None

    def _preamble(self) -> list[str]:
        lines = [
            f"Entities: {self.entities} | Relations: {self.relations} | "
            f"Documents: {self.documents} | Chunks: {self.chunks}"
        ]
        if self.hint:
            lines.append(f"Hint: {_clean(self.hint)}")
        return lines

    def _sections(self) -> list[tuple[str, list[str]]]:
        return [
            ("Entity types", [f"- {_clean(t.label)}: {t.count}" for t in self.entity_types]),
            ("Relation types", [f"- {_clean(t.label)}: {t.count}" for t in self.relation_types]),
        ]


class OntologyResult(ToolResult):
    """The declared ontology (with live counts)."""

    entity_types: list[EntityTypeInfo] = Field(default_factory=list)
    relation_types: list[RelationTypeInfo] = Field(default_factory=list)

    def _sections(self) -> list[tuple[str, list[str]]]:
        ent_lines = []
        for e in self.entity_types:
            line = f"- {_clean(e.label)}: {e.count}"
            if e.description:
                line += f" — {_snippet(e.description, 100)}"
            if e.properties:
                line += f" (props: {', '.join(_clean(p) for p in e.properties)})"
            ent_lines.append(line)
        rel_lines = []
        for r in self.relation_types:
            line = f"- {_clean(r.label)}: {r.count}"
            if r.patterns:
                pats = ", ".join(f"{_clean(a)}->{_clean(b)}" for a, b in r.patterns)
                line += f" [{pats}]"
            if r.description:
                line += f" — {_snippet(r.description, 100)}"
            rel_lines.append(line)
        return [("Entity types", ent_lines), ("Relation types", rel_lines)]


class EntityMatch(BaseModel):
    """An entity whose name matched a lookup."""

    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    label: str = ""
    description: str = ""


class LookupResult(ToolResult):
    """Entities matching a name, best first."""

    query: str
    matches: list[EntityMatch] = Field(default_factory=list)
    hint: str | None = None

    def _preamble(self) -> list[str]:
        lines = [f"Lookup: {_clean(self.query)}"]
        if self.hint:
            lines.append(f"Hint: {_clean(self.hint)}")
        return lines

    def _sections(self) -> list[tuple[str, list[str]]]:
        def line(m: EntityMatch) -> str:
            label = f"{_clean(m.label)}, " if m.label else ""
            desc = f": {_snippet(m.description, 160)}" if m.description else ""
            return f'- "{_clean(m.name)}" ({label}id {_clean(m.id)}){desc}'

        return [("Matches", [line(m) for m in self.matches])]


class TraverseNode(BaseModel):
    """An entity a walk reached."""

    model_config = ConfigDict(extra="forbid")
    n: int | None = None
    id: str
    name: str
    depth: int
    via: str = ""


class TraverseResult(ToolResult):
    """What a bounded walk from one entity reached."""

    start: str
    start_id: str
    via: list[str] | None = None
    nodes: list[TraverseNode] = Field(default_factory=list)
    edges: list[RelationTriple] = Field(default_factory=list)
    stopped: str = "complete"
    truncated: bool = False
    hint: str | None = None

    def _preamble(self) -> list[str]:
        path = f" via {' -> '.join(self.via)}" if self.via else ""
        lines = [f"From {_clean(self.start)}{path}: stopped={self.stopped}"]
        if self.hint:
            lines.append(f"Hint: {_clean(self.hint)}")
        return lines

    def _sections(self) -> list[tuple[str, list[str]]]:
        return [
            ("Edges", [_relation_line(e) for e in self.edges]),
            (
                "Reached",
                [f"- {_n(x.n)}{_clean(x.name)} (depth {x.depth})" for x in self.nodes],
            ),
        ]


class WalkPath(BaseModel):
    """One scored path."""

    model_config = ConfigDict(extra="forbid")
    n: int | None = None
    nodes: list[str] = Field(default_factory=list)
    node_ids: list[str] = Field(default_factory=list)
    edges: list[str] = Field(default_factory=list)
    score: float = 0.0


def _path_text(p: WalkPath) -> str:
    parts = [_clean(p.nodes[0])] if p.nodes else []
    for i, name in enumerate(p.nodes[1:]):
        rel = p.edges[i] if i < len(p.edges) else "?"
        parts.append(f"-[{_clean(rel)}]-> {_clean(name)}")
    return " ".join(parts)


class WalkResult(ToolResult):
    """The strongest paths a weighted walk found."""

    start: str
    goal: str | None = None
    paths: list[WalkPath] = Field(default_factory=list)
    hint: str | None = None

    def _preamble(self) -> list[str]:
        target = f" to {_clean(self.goal)}" if self.goal else ""
        lines = [f"Paths from {_clean(self.start)}{target}"]
        if self.hint:
            lines.append(f"Hint: {_clean(self.hint)}")
        return lines

    def _sections(self) -> list[tuple[str, list[str]]]:
        return [
            (
                "Paths",
                [f"- {_n(p.n)}{_path_text(p)} (score {p.score:.3f})" for p in self.paths],
            )
        ]


class CypherRow(BaseModel):
    """One numbered result row."""

    model_config = ConfigDict(extra="forbid")
    n: int | None = None
    values: list[Any] = Field(default_factory=list)


class CypherResult(ToolResult):
    """Rows returned by a guarded read-only Cypher query."""

    columns: list[str] = Field(default_factory=list)
    rows: list[CypherRow] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    hint: str | None = None

    def _preamble(self) -> list[str]:
        suffix = " (truncated)" if self.truncated else ""
        lines = [
            f"Columns: {', '.join(_clean(c) for c in self.columns)}",
            f"Rows: {self.row_count}{suffix}",
        ]
        if self.hint:
            lines.append(f"Hint: {_clean(self.hint)}")
        return lines

    def _sections(self) -> list[tuple[str, list[str]]]:
        return [
            (
                "Rows",
                [
                    f"- {_n(r.n)}{_clean(json.dumps(r.values, ensure_ascii=False, default=str))}"
                    for r in self.rows
                ],
            )
        ]


class RetrieveResult(ToolResult):
    """Ranked retrieval context for a question (no answer generated)."""

    question: str
    entities: list[EntityCard] = Field(default_factory=list)
    relations: list[RelationTriple] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    chunks: list[ChunkRef] = Field(default_factory=list)
    documents: list[DocumentRef] = Field(default_factory=list)
    hint: str | None = None

    def _preamble(self) -> list[str]:
        lines = [f"Question: {_clean(self.question)}"]
        if self.hint:
            lines.append(f"Hint: {_clean(self.hint)}")
        return lines

    def _sections(self) -> list[tuple[str, list[str]]]:
        return [
            ("Entities", [_entity_line(e) for e in self.entities]),
            ("Relations", [_relation_line(r) for r in self.relations]),
            ("Facts", [f"- {_n(f.n)}{_snippet(f.text)}" for f in self.facts]),
            (
                "Passages",
                [
                    f"- {_n(c.n)}[{_clean(c.document_path or c.document_id)}] {_snippet(c.text)}"
                    for c in self.chunks
                ],
            ),
        ]


class SkillRunResult(ToolResult):
    """A skill's findings, cited as one piece of evidence."""

    skill: str
    n: int | None = None
    summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)

    def _preamble(self) -> list[str]:
        lines = [f"{_n(self.n)}Skill {_clean(self.skill)}"]
        if self.summary:
            lines.append(_snippet(self.summary, 2000))
        return lines

    def _sections(self) -> list[tuple[str, list[str]]]:
        return [
            (
                "Data",
                [
                    f"- {_clean(k)}: {_snippet(json.dumps(v, ensure_ascii=False, default=str))}"
                    for k, v in self.data.items()
                ],
            )
        ]


class EvidenceRef(BaseModel):
    """A piece of evidence an answer cites."""

    model_config = ConfigDict(extra="forbid")
    n: int
    tool: str
    kind: str
    content: str
    source: str = ""


class CitationCheckResult(ToolResult):
    """An answer's citations checked against the session's evidence."""

    grounded: bool
    reason: str
    cited: list[int] = Field(default_factory=list)
    invalid: list[int] = Field(default_factory=list)
    answer: str = ""
    sources: list[EvidenceRef] = Field(default_factory=list)
    hint: str | None = None

    def _preamble(self) -> list[str]:
        lines = [f"Grounded: {self.grounded} ({self.reason})"]
        if self.invalid:
            lines.append(f"Invalid citations removed: {self.invalid}")
        if self.hint:
            lines.append(f"Hint: {_clean(self.hint)}")
        return lines

    def _sections(self) -> list[tuple[str, list[str]]]:
        return [
            (
                "Sources",
                [f"- [{s.n}] ({_clean(s.tool)}) {_snippet(s.content, 200)}" for s in self.sources],
            )
        ]


class DocSample(BaseModel):
    """One file under the files root, with a sample of its text."""

    model_config = ConfigDict(extra="forbid")
    path: str
    type: str
    size_bytes: int = 0
    chars: int = 0
    sample: str = ""
    error: str | None = None


class SampleDocsResult(ToolResult):
    """A page of the files under the files root, one sample each."""

    pattern: str
    total: int
    offset: int = 0
    next_offset: int | None = None
    types: dict[str, int] = Field(default_factory=dict)
    files: list[DocSample] = Field(default_factory=list)
    hint: str | None = None

    def _preamble(self) -> list[str]:
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(self.types.items())) or "none"
        lines = [
            f"{self.total} files match '{_clean(self.pattern)}' ({kinds}); "
            f"showing {self.offset + 1 if self.files else 0}-{self.offset + len(self.files)}."
        ]
        if self.next_offset is not None:
            lines.append(f"More: offset={self.next_offset}.")
        if self.hint:
            lines.append(f"Hint: {_clean(self.hint)}")
        return lines

    def _sections(self) -> list[tuple[str, list[str]]]:
        items = []
        for f in self.files:
            body = f"error: {_clean(f.error)}" if f.error else _snippet(f.sample, 600)
            items.append(f"- {_clean(f.path)} ({f.chars} chars): {body}")
        return [("Files", items)]


class PrunedTypes(BaseModel):
    """What extraction found but the ontology does not declare: dropped, or kept as Unknown."""

    model_config = ConfigDict(extra="forbid")
    entities: dict[str, int] = Field(
        default_factory=dict, description="Undeclared entity type -> entities dropped."
    )
    relations: dict[str, int] = Field(
        default_factory=dict, description="Undeclared relation type -> relations dropped."
    )
    pattern_mismatches: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Declared relation -> 'Source->Target' pair it is not declared for -> count.",
    )
    pattern_examples: dict[str, dict[str, str]] = Field(
        default_factory=dict,
        description=(
            "Declared relation -> 'Source->Target' pair -> one dropped relation "
            "('source name -> target name: fact'), to judge whether the pair fits."
        ),
    )
    dangling: int = Field(default=0, description="Relations whose endpoint was dropped.")
    unknown: int = Field(
        default=0,
        description="Entities kept with label Unknown: their type is not in the ontology.",
    )

    @classmethod
    def from_report(cls, report: Any) -> PrunedTypes | None:
        """From a pipeline ``metadata["pruned"]`` dict; ``None`` when empty."""
        if not isinstance(report, dict) or not report:
            return None
        return cls(
            entities={str(k): int(v) for k, v in (report.get("entities") or {}).items()},
            relations={str(k): int(v) for k, v in (report.get("relations") or {}).items()},
            pattern_mismatches={
                str(rel): {str(pair): int(n) for pair, n in pairs.items()}
                for rel, pairs in (report.get("pattern_mismatches") or {}).items()
            },
            pattern_examples={
                str(rel): {str(pair): str(text) for pair, text in pairs.items()}
                for rel, pairs in (report.get("pattern_examples") or {}).items()
            },
            dangling=int(report.get("dangling") or 0),
            unknown=int(report.get("unknown") or 0),
        )

    def add(self, other: PrunedTypes) -> None:
        for key, n in other.entities.items():
            self.entities[key] = self.entities.get(key, 0) + n
        for key, n in other.relations.items():
            self.relations[key] = self.relations.get(key, 0) + n
        for rel, pairs in other.pattern_mismatches.items():
            mine = self.pattern_mismatches.setdefault(rel, {})
            for pair, n in pairs.items():
                mine[pair] = mine.get(pair, 0) + n
        for rel, examples in other.pattern_examples.items():
            kept = self.pattern_examples.setdefault(rel, {})
            for pair, text in examples.items():
                kept.setdefault(pair, text)
        self.dangling += other.dangling
        self.unknown += other.unknown

    @property
    def missing_types(self) -> bool:
        """Whether an undeclared type or pattern showed up (not only dangling edges)."""
        return bool(self.entities or self.relations or self.pattern_mismatches or self.unknown)

    def summary(self, limit: int = 8) -> str:
        def top(counts: dict[str, int]) -> str:
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            shown = ", ".join(f"{_clean(k)} ({n})" for k, n in ranked[:limit])
            return shown + (f" (+{len(ranked) - limit} more)" if len(ranked) > limit else "")

        parts = []
        if self.entities:
            parts.append(f"entity types {top(self.entities)}")
        if self.relations:
            parts.append(f"relation types {top(self.relations)}")
        if self.pattern_mismatches:
            flat = {
                f"{rel} {pair}": n
                for rel, pairs in self.pattern_mismatches.items()
                for pair, n in pairs.items()
            }
            parts.append(f"undeclared relation patterns {top(flat)}")
        if self.unknown:
            parts.append(f"{self.unknown} entities of no declared type (stored as Unknown)")
        if self.dangling:
            parts.append(f"{self.dangling} relations to dropped entities")
        return "; ".join(parts)


class IngestedDocument(BaseModel):
    """The outcome for one file of an ingest call."""

    model_config = ConfigDict(extra="forbid")
    path: str
    document_id: str | None = None
    status: Literal["ingested", "unchanged", "reextracted", "failed"]
    chunks_indexed: int = 0
    nodes_created: int = 0
    relationships_created: int = 0
    plan: dict[str, Any] | None = Field(
        default=None, description="Strategies used: chunker, extractor, resolver, params, reason."
    )
    pruned: PrunedTypes | None = None
    error: str | None = None


class IngestResult(ToolResult):
    """Outcome of adding text or files to the graph.

    For text, ``document_id`` names the stored document; for files,
    ``documents`` has one entry per file. ``pruned`` totals what the
    ontology did not declare and was dropped.
    """

    document_id: str | None = None
    status: Literal["ingested", "unchanged", "reextracted"] | None = Field(
        default=None, description="For text: whether it was extracted, or already stored as is."
    )
    chunks_indexed: int = 0
    nodes_created: int = 0
    relationships_created: int = 0
    documents: list[IngestedDocument] = Field(default_factory=list)
    failed: int = 0
    pruned: PrunedTypes | None = None
    finalized: bool = False
    hint: str | None = None

    def _preamble(self) -> list[str]:
        totals = (
            f"{self.chunks_indexed} chunks, {self.nodes_created} nodes, "
            f"{self.relationships_created} relations"
        )
        if self.document_id is not None:
            doc = _clean(self.document_id)
            if self.status == "unchanged":
                lines = [f"Document '{doc}' is already stored unchanged; nothing was extracted."]
            elif self.status == "reextracted":
                lines = [f"Re-extracted document '{doc}' ({totals})."]
            else:
                lines = [f"Stored document '{doc}' ({totals})."]
        else:
            counts: dict[str, int] = {}
            for doc in self.documents:
                counts[doc.status] = counts.get(doc.status, 0) + 1
            status = ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) or "none"
            lines = [f"Files: {status} ({totals})."]
        if self.pruned is not None:
            lines.append(f"Dropped (not in the ontology): {self.pruned.summary()}.")
        lines.append("Finalized." if self.finalized else "Not finalized yet.")
        if self.hint:
            lines.append(f"Hint: {_clean(self.hint)}")
        return lines

    def _sections(self) -> list[tuple[str, list[str]]]:
        items = []
        for doc in self.documents:
            line = f"- {_clean(doc.path)}: {doc.status}"
            if doc.error:
                line += f" — {_snippet(doc.error, 160)}"
            else:
                line += f" ({doc.chunks_indexed} chunks, {doc.nodes_created} nodes)"
            if doc.plan:
                p = doc.plan
                line += f" [{p.get('chunker')}/{p.get('extractor')}/{p.get('resolver')}]"
            items.append(line)
        return [("Documents", items)]


class FinalizeReport(ToolResult):
    """Outcome of finalize."""

    finalized: bool = True
    entities_deduplicated: int = 0
    entities_embedded: int = 0
    relationships_embedded: int = 0
    null_stubs_removed: int = 0
    hint: str | None = None

    def _preamble(self) -> list[str]:
        lines = [
            f"Finalized: {self.entities_deduplicated} duplicates merged, "
            f"{self.entities_embedded} entities and {self.relationships_embedded} relations "
            f"embedded, {self.null_stubs_removed} empty stubs removed."
            if self.finalized
            else "Not finalized."
        ]
        if self.hint:
            lines.append(f"Hint: {_clean(self.hint)}")
        return lines
