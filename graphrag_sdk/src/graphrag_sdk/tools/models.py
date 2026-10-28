# GraphRAG SDK — Tools: result models + LLM-text rendering
# Typed results of the toolkit tools. Every model is pydantic v2, carries the
# evidence numbers [n] to cite, and renders bounded text via to_llm_text().

from __future__ import annotations

import json
from typing import Any

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
    document_path: str = ""
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


class IngestResult(ToolResult):
    """Outcome of adding text to the graph."""

    document_id: str
    chunks_indexed: int = 0
    nodes_created: int = 0
    relationships_created: int = 0
    finalized: bool = False
    hint: str | None = None

    def _preamble(self) -> list[str]:
        lines = [
            f"Stored document '{_clean(self.document_id)}' "
            f"({self.chunks_indexed} chunks, {self.nodes_created} nodes, "
            f"{self.relationships_created} relations)."
        ]
        lines.append("Finalized." if self.finalized else "Not finalized yet.")
        if self.hint:
            lines.append(f"Hint: {_clean(self.hint)}")
        return lines


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
