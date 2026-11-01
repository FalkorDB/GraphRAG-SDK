# GraphRAG SDK — Tools: GraphRAGToolkit
# The graph tools an outside agent calls (through MCP or its own framework).
# The agent plans, calls tools and writes the answer; the toolkit runs each
# call bounded and read-only, numbers the evidence it returns and checks the
# agent's citations against it. There is no agent loop in the SDK.

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import logging
import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ValidationError

from graphrag_sdk.core.connection import ConnectionConfig
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import (
    ChunkingError,
    ConfigError,
    LoaderError,
    ReadOnlyViolation,
)
from graphrag_sdk.core.models import (
    RESERVED_NODE_LABELS,
    DocumentOutput,
    Entity,
    Ontology,
    Relation,
)
from graphrag_sdk.core.providers import Embedder, LLMInterface
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval
from graphrag_sdk.tools import files as fs
from graphrag_sdk.tools import graph_ops as ops
from graphrag_sdk.tools.errors import ToolError
from graphrag_sdk.tools.evidence import (
    DEFAULT_LEDGER_CAPACITY,
    EvidenceLedger,
    ground_answer,
    states_a_gap,
)
from graphrag_sdk.tools.models import (
    ChunkRef,
    CitationCheckResult,
    ClaimCheck,
    DocSample,
    DocumentRef,
    EntityCard,
    EntityTypeInfo,
    EvidenceRef,
    Fact,
    FinalizeReport,
    IngestedDocument,
    IngestResult,
    OntologyResult,
    PreviewFilesResult,
    PrunedTypes,
    RelationTriple,
    RelationTypeInfo,
    RetrieveResult,
    StatusResult,
    properties_text,
)
from graphrag_sdk.tools.specs import (
    RETRIEVE_PATHS,
    TOOL_REGISTRY,
    CheckCitationsInput,
    IngestInput,
    PlanRuleInput,
    PreviewFilesInput,
    RetrieveInput,
    SetOntologyInput,
    ToolSpec,
    build_tool_specs,
    tool_enabled,
)
from graphrag_sdk.utils.cypher import sanitize_cypher_label

if TYPE_CHECKING:
    from graphrag_sdk.api.main import GraphRAG

FinalizePolicy = Literal["manual", "on_write", "never"]
_POLICIES = ("manual", "on_write", "never")
_TENANT_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
logger = logging.getLogger(__name__)

#: Most names listed in a hint.
_HINT_NAMES = 30
#: Total characters preview_files returns in one call (split across the files).
SAMPLE_BUDGET_CHARS = 30_000
#: Most characters of one preview page as JSON (paths, counts and samples), so
#: the MCP server's 40k result cap never has to cut files from it.
_PAGE_BUDGET_CHARS = 36_000
#: Files loaded or ingested at once by preview_files / ingest(files=...).
_FILE_CONCURRENCY = 3
# Files whose last plan is remembered for reextract (oldest forgotten first).
_PLAN_MEMORY = 10_000
_NO_PLAN = (
    " (no plan remembered for this file: plans are kept in memory only, so after"
    " a restart pass plan=[...] to repeat a custom plan)"
)
#: retrieve's search paths and the MultiPath path behind each. ``links`` has
#: none: retrieve walks the relations itself, so MultiPath's own expansion,
#: whose output it never reads, is not run.
_MULTIPATH_PATH = {
    "facts": "relates",
    "exact_names": "entity_cypher",
    "fuzzy_names": "entity_fulltext",
    "passages": "chunks",
}
_ALL_MULTIPATH_PATHS = frozenset(_MULTIPATH_PATH.values())
#: Paths that find the entities links walk from.
_SEED_PATHS = frozenset({"facts", "exact_names", "fuzzy_names"})


class _FixedPlan:
    """A MultiPath path router that always runs the same paths."""

    def __init__(self, paths: frozenset[str]) -> None:
        self._paths = sorted(paths)

    async def plan(self, query: str, ctx: Context) -> list[str]:
        return list(self._paths)


class _ConfinedLoader(LoaderStrategy):
    """The default loader for a file, reading it only from inside the root.

    ``source`` is the file's path relative to the root (an absolute path
    under the root is accepted too). Paths are validated when an ingest call
    starts; the file is then opened without following any symlink and read
    from that open file (see :func:`files.open_confined`), so a symlink
    swapped in meanwhile cannot lead outside the root.
    """

    def __init__(self, root: Path, max_bytes: int | None) -> None:
        self._root = root
        self._max_bytes = max_bytes

    async def load(self, source: str, ctx: Context) -> DocumentOutput:
        rel = source
        if Path(source).is_absolute():
            try:
                rel = Path(source).relative_to(self._root).as_posix()
            except ValueError:
                raise ToolError(
                    f"{Path(source).name!r} is not under the files root.",
                    hint="Pass paths relative to the files root.",
                ) from None
        data = await asyncio.to_thread(fs.read_confined, self._root, rel, self._max_bytes)
        # Parsed as its name says (a.md -> Markdown), like GraphRAG's default loaders.
        doc = await asyncio.to_thread(fs.document_from_bytes, rel, data)
        info = doc.document_info.model_copy(update={"path": source})
        return doc.model_copy(update={"document_info": info})


class _Loaded(LoaderStrategy):
    """Hands back a document already loaded (and planned from), so it is not read twice."""

    def __init__(self, document: DocumentOutput) -> None:
        self._document = document

    async def load(self, source: str, ctx: Context) -> DocumentOutput:
        return self._document


class _SamePlan:
    """An ingestion planner that returns one fixed plan, under another planner's limits."""

    def __init__(self, plan: Any, max_chunks: int | None) -> None:
        self._plan = plan
        self.max_chunks = max_chunks

    async def plan(self, text: str | None, *, source: str | None = None, ctx: Any = None) -> Any:
        return self._plan


#: Entity types the SDK writes itself: Unknown holds entities whose type was not resolved.
_RESERVED_ENTITY_TYPES = RESERVED_NODE_LABELS | {"Unknown"}
_LABEL_HINT = (
    "Use letters, digits, spaces, '-' or '_'; no '|', ':', backticks, line breaks "
    "or other control characters, and no __X__ names (the SDK's own)."
)


def _label_error(label: str, *, entity: bool) -> ToolError | None:
    """Why ``label`` cannot be a type name, as a ToolError, or ``None``."""
    try:
        clean = sanitize_cypher_label(label) == label
    except ValueError:  # blank
        clean = False
    if (
        not clean
        or "|" in label
        or ":" in label
        or any(unicodedata.category(c).startswith("C") for c in label)
        or (len(label) > 4 and label.startswith("__") and label.endswith("__"))
    ):
        return ToolError(f"{label!r} is not a usable type name.", hint=_LABEL_HINT)
    if entity and label in _RESERVED_ENTITY_TYPES:
        return ToolError(
            f"{label!r} is reserved by the SDK.",
            hint=f"{label!r} is reserved by the SDK; choose another name "
            "(e.g. 'Publication' for Document, 'Passage' for Chunk).",
        )
    return None


_FACT_RE = re.compile(r"^(.+?) —\[([^\]]+)\]→ (.+?)(?:: (.*))?$", re.S)


def _short_fact(raw: str) -> tuple[str, tuple[str, str, str] | None]:
    """A retrieval fact line without the repeated triple, and its triple key.

    Retrieval writes ``A —[T]→ B: (A, T, B): description``; the bracketed
    triple only repeats ``A —[T]→ B``. Lines in other shapes pass through.
    """
    raw = raw.strip()
    m = _FACT_RE.match(raw)
    if not m:
        return raw, None
    src, rel_type, tgt, rest = m.group(1), m.group(2), m.group(3), m.group(4) or ""
    body = ops.fact_body(rest, rel_type)
    text = f"{src} —[{rel_type}]→ {tgt}" + (f": {body}" if body else "")
    return text, (src.lower(), rel_type, tgt.lower())


_PATTERN_HINT_RELATIONS = 6
_PATTERN_HINT_EXAMPLES = 8


def _pattern_hint(pruned: PrunedTypes, redo: str) -> str:
    """Say how many relations an undeclared pattern dropped, and how to keep them."""
    mismatches = pruned.pattern_mismatches
    dropped = sum(n for pairs in mismatches.values() for n in pairs.values())
    ranked = sorted(mismatches.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
    add = []
    examples = []
    for rel, pairs in ranked[:_PATTERN_HINT_RELATIONS]:
        ordered = sorted(pairs, key=lambda p: (-pairs[p], p))
        ends = [pair.split("->", 1) for pair in ordered]
        add.append({"label": rel, "patterns": [e for e in ends if len(e) == 2]})
        for pair in ordered:
            example = pruned.pattern_examples.get(rel, {}).get(pair)
            if example and len(examples) < _PATTERN_HINT_EXAMPLES:
                examples.append(f"{rel} {pair}: {example}")
    more = " (and more, see pruned)" if len(ranked) > _PATTERN_HINT_RELATIONS else ""
    seen = f" Dropped, e.g.: {'; '.join(examples)}." if examples else ""
    return (
        f"{dropped} extracted relation(s) were dropped only because their [source, target] "
        f"pair is not declared.{seen} Keep only the pairs whose facts the relation's meaning "
        "fits; for the others, declare a better relation type. To keep them all: "
        f"set_ontology(relations={json.dumps(add)}){more}, then {redo}, then finalize."
    )


def _names_hint(names: Sequence[str]) -> str:
    shown = list(names)[:_HINT_NAMES]
    more = f" (+{len(names) - len(shown)} more)" if len(names) > len(shown) else ""
    return ", ".join(shown) + more if shown else "none"


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _may_have_written(exc: BaseException) -> bool:
    """Whether a file's failure may have come after the pipeline began writing.

    Loading and chunking (the chunk budget included) run before anything is
    written; any later failure can leave the file's nodes in the graph.
    """
    return not isinstance(exc, (ToolError, LoaderError, ChunkingError))


def _file_error(exc: BaseException) -> str:
    """One file's error for a tool result.

    Only messages the SDK writes about the file itself are forwarded (masked):
    a ToolError, a loader's read failure and the chunk budget's refusal. Any
    other exception (a model provider's, say) is named by type only, as the
    MCP server does, because its message can quote a secret that ``redact``
    does not recognise; the log has it.
    """
    from graphrag_sdk.ingestion.ingestion_planner import ChunkBudgetExceededError
    from graphrag_sdk.mcp.server import redact  # lazy: the server imports this module

    if isinstance(exc, (ToolError, LoaderError, ChunkBudgetExceededError)):
        return _short(redact(str(exc)), 300)
    logger.warning("A file failed (%s); details withheld.", type(exc).__name__)
    return (
        f"Failed with {type(exc).__name__}; the message is withheld "
        "(details are in the server log)."
    )


def _as_dict(value: Any) -> Any:
    return value.model_dump() if isinstance(value, BaseModel) else value


def _ingested(path: str, res: Any, status: str, plan: dict[str, Any] | None) -> IngestedDocument:
    meta = getattr(res, "metadata", None) or {}
    return IngestedDocument(
        path=path,
        document_id=res.document_info.uid,
        status=status,
        chunks_indexed=res.chunks_indexed,
        nodes_created=res.nodes_created,
        relationships_created=res.relationships_created,
        plan=plan,
        pruned=PrunedTypes.from_report(meta.get("pruned")),
    )


class GraphRAGToolkit:
    """The graph tools for an outside agent, over one :class:`GraphRAG`.

    Read tools: ``status``, ``get_ontology``, ``retrieve``,
    ``check_citations``. Ingestion tools:
    ``preview_files`` (with a ``files_root``), ``set_ontology``, ``ingest``,
    ``finalize``. Each is an async method; :meth:`tool_specs` describes them
    for an agent framework and :meth:`call` runs one by name with validated
    arguments.

    Every tool result an answer may rely on carries an evidence number
    ``[n]``. Numbers run across all calls of one session (one toolkit, or one
    :meth:`new_session` copy), and :meth:`check_citations` verifies an
    answer's ``[n]`` against them.

    Args:
        rag: The GraphRAG instance to wrap.
        finalize_policy: ``"manual"`` (default: call :meth:`finalize`
            yourself after a batch of ingests), ``"on_write"`` (finalize
            after every ingest; for demos, finalize cost grows with the
            graph) or ``"never"`` (you call ``rag.finalize()``).
        read_only: Drop the ingestion tools and refuse them.
        include: Optional subset of tool names to advertise and accept in
            :meth:`call`. Direct method calls are not affected.
        tenant_id: Stamped into the Context of every operation.
        owns_rag: When True, :meth:`aclose` closes the wrapped GraphRAG.
        ledger_capacity: Evidence items kept per session.
        files_root: Folder ``preview_files`` and ``ingest(files=...)`` read
            from; nothing outside it is read. ``None`` (default) disables
            file ingestion.
        plan_limits: The strategies an agent's ingest plan may use, as
            :class:`~graphrag_sdk.ingestion.ingestion_planner.RuleBasedIngestionPlanner`
            keyword arguments (``chunkers``, ``extractors``, ``resolvers``,
            ``max_llm_pairs``, ``tune_params``). Default
            :data:`~graphrag_sdk.ingestion.ingestion_planner.CHEAP_PLAN_LIMITS`:
            no strategy that adds LLM calls.
        max_file_bytes: Largest file ``ingest(files=...)`` reads (default
            50 MiB); a larger one is refused with a hint. ``None``: no cap.
            ``preview_files`` reads only part of each file, whatever its size.
    """

    def __init__(
        self,
        rag: GraphRAG,
        *,
        finalize_policy: FinalizePolicy = "manual",
        read_only: bool = False,
        include: Sequence[str] | None = None,
        tenant_id: str = "default",
        owns_rag: bool = False,
        ledger_capacity: int = DEFAULT_LEDGER_CAPACITY,
        files_root: str | os.PathLike[str] | None = None,
        plan_limits: Mapping[str, Any] | None = None,
        max_file_bytes: int | None = fs.MAX_FILE_BYTES,
    ) -> None:
        from graphrag_sdk.ingestion.ingestion_planner import (
            CHEAP_PLAN_LIMITS,
            RuleBasedIngestionPlanner,
        )

        if finalize_policy not in _POLICIES:
            raise ValueError(f"finalize_policy must be one of {_POLICIES}, got {finalize_policy!r}")
        valid = {td.name for td in TOOL_REGISTRY}
        if include is not None:
            unknown = sorted(set(include) - valid)
            if unknown:
                raise ValueError(f"Unknown tool names in include={unknown}; valid: {sorted(valid)}")
        self._rag = rag
        self._finalize_policy: FinalizePolicy = finalize_policy
        self._read_only = read_only
        self._include = frozenset(include) if include is not None else None
        self._tenant_id = tenant_id
        self._owns_rag = owns_rag
        self._ledger_capacity = ledger_capacity
        self._ledger = EvidenceLedger(ledger_capacity)
        self._files_root: Path | None = (
            fs.resolve_root(files_root) if files_root is not None else None
        )
        if max_file_bytes is not None and (
            isinstance(max_file_bytes, bool)
            or not isinstance(max_file_bytes, int)
            or max_file_bytes <= 0
        ):
            raise ValueError("max_file_bytes must be a positive int or None")
        self._max_file_bytes = max_file_bytes
        limits = dict(CHEAP_PLAN_LIMITS if plan_limits is None else plan_limits)
        RuleBasedIngestionPlanner((), **limits)  # validate now, not at the first ingest
        self._plan_limits: dict[str, Any] = limits
        # Shared by every session copy of this toolkit.
        self._shared: dict[str, Any] = {}

    # ── Lifecycle ────────────────────────────────────────────────

    @property
    def rag(self) -> GraphRAG:
        """The wrapped GraphRAG instance."""
        return self._rag

    @property
    def files_root(self) -> Path | None:
        """The folder file ingestion reads from (``None``: disabled)."""
        return self._files_root

    @property
    def ledger(self) -> EvidenceLedger:
        """This session's numbered evidence."""
        return self._ledger

    def new_session(self) -> GraphRAGToolkit:
        """A toolkit on the same graph with its own, empty evidence ledger.

        Use one per conversation (MCP does this per client session) so
        evidence numbers from different conversations never mix. The copy
        never closes the wrapped GraphRAG.
        """
        clone = copy.copy(self)
        clone._ledger = EvidenceLedger(self._ledger_capacity)
        clone._owns_rag = False
        return clone

    @classmethod
    def for_tenant(
        cls,
        base_config: ConnectionConfig,
        tenant_id: str,
        *,
        llm: LLMInterface,
        embedder: Embedder,
        ontology: Ontology | None = None,
        embedding_dimension: int = 256,
        **toolkit_kwargs: Any,
    ) -> GraphRAGToolkit:
        """Build a toolkit bound to a tenant-scoped graph.

        Derives ``graph_name = f"{base_config.graph_name}__{tenant_id}"`` and
        constructs a dedicated GraphRAG the toolkit owns (closed by
        :meth:`aclose` / ``async with``).
        """
        if not _TENANT_RE.fullmatch(tenant_id):
            raise ValueError(
                "tenant_id must match ^[A-Za-z0-9_-]{1,64}$ (it becomes part "
                f"of the graph name); got {tenant_id!r}"
            )
        from graphrag_sdk.api.main import GraphRAG  # local import: avoid cycle

        config = dataclasses.replace(
            base_config, graph_name=f"{base_config.graph_name}__{tenant_id}"
        )
        rag = GraphRAG(
            connection=config,
            llm=llm,
            embedder=embedder,
            ontology=ontology,
            embedding_dimension=embedding_dimension,
        )
        return cls(rag, tenant_id=tenant_id, owns_rag=True, **toolkit_kwargs)

    async def aclose(self) -> None:
        """Close the wrapped GraphRAG if this toolkit owns it."""
        if self._owns_rag:
            await self._rag.close()

    async def __aenter__(self) -> GraphRAGToolkit:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ── Specs & dispatch ─────────────────────────────────────────

    def tool_specs(self) -> list[ToolSpec]:
        """Tool definitions for this toolkit configuration."""
        return build_tool_specs(
            read_only=self._read_only,
            finalize_policy=self._finalize_policy,
            include=self._include,
            files=self._files_root is not None,
            plan_limits=self._plan_limits,
        )

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> BaseModel:
        """Validate ``arguments`` for tool ``name`` and run it.

        The entry point for adapters. Applies the same ``read_only`` /
        ``include`` / finalize-policy gates as :meth:`tool_specs`. Bad
        arguments and unknown or disabled tools raise :class:`ToolError`
        with a hint; a write on a read-only toolkit raises
        :class:`ReadOnlyViolation`.
        """
        by_name = {td.name: td for td in TOOL_REGISTRY}
        td = by_name.get(name)
        if td is None:
            enabled = [t.name for t in TOOL_REGISTRY if self._enabled(t)]
            raise ToolError(f"Unknown tool {name!r}.", hint=f"Tools: {', '.join(enabled)}.")
        if self._read_only and (not td.read_only or td.ingest_only):
            raise ReadOnlyViolation(f"{name} is disabled: the toolkit is read-only.")
        if not self._enabled(td):
            enabled = [t.name for t in TOOL_REGISTRY if self._enabled(t)]
            hint = f"Tools: {', '.join(enabled)}."
            if td.needs_files and self._files_root is None:
                hint = "No files root is configured; ingest text instead. " + hint
            raise ToolError(f"Tool {name!r} is not enabled.", hint=hint)
        try:
            model = td.input_model(**(arguments or {}))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
                for err in exc.errors()
            )
            raise ToolError(
                f"Invalid arguments for {name}: {problems}.",
                hint="Fix the arguments to match the tool's input schema.",
            ) from None
        return await getattr(self, name)(**model.model_dump())

    def _enabled(self, td: Any) -> bool:
        return tool_enabled(
            td,
            read_only=self._read_only,
            finalize_policy=self._finalize_policy,
            include=self._include,
            files=self._files_root is not None,
        )

    def _new_ctx(self) -> Context:
        return Context(tenant_id=self._tenant_id)

    @property
    def _store(self) -> Any:
        return self._rag._graph_store

    def _evidence(
        self, tool: str, kind: str, content: str, source: str = "", **meta: Any
    ) -> int | None:
        # The full text the agent was shown, so check_citations checks the
        # same text the agent cited from. Nothing to cite gets no number, so
        # an empty passage cannot make an answer grounded.
        if not content.strip():
            return None
        return self._ledger.add(
            tool=tool, kind=kind, content=content, source=source, metadata=meta
        ).n

    @property
    def _graph_name(self) -> str:
        config = getattr(self._rag._conn, "config", None)
        return str(getattr(config, "graph_name", None) or "default")

    def _missing_graph(self) -> ToolError:
        return ToolError(
            f"Graph '{self._graph_name}' does not exist.",
            hint=(
                "Ingest documents (and finalize) first, or check the graph name "
                "the server was started with."
            ),
        )

    async def _require_graph(self) -> None:
        """Raise a ToolError unless the graph exists and can be reached.

        Checked read-only on every call (asking never creates the graph), so
        a graph deleted after an earlier call is reported missing.
        """
        if not await self._graph_exists():
            raise self._missing_graph()

    async def _graph_exists(self) -> bool:
        """Whether the graph exists; a ToolError if the store cannot be reached."""
        exists = await ops.graph_exists(self._store)
        if exists is None:
            raise ToolError(
                f"Could not reach the graph store to read graph '{self._graph_name}'.",
                hint="Check that FalkorDB is running and reachable, then call the tool again.",
            )
        return exists

    # ── Read tools ───────────────────────────────────────────────

    async def status(self) -> StatusResult:
        """What the graph holds: counts, entity labels and relation types."""
        exists = await self._graph_exists()
        counts = await ops.graph_counts(self._store)
        label_counts, rel_counts = await ops.schema_counts(self._store)
        hint = None
        if counts["entities"] == 0 and not exists:
            hint = f"{self._missing_graph()}"
        elif counts["entities"] == 0:
            hint = "The graph is empty: ingest documents (and finalize) before querying."
        elif counts["relations"] == 0:
            hint = "The graph has entities but no relations."
        result = StatusResult(
            entities=counts["entities"],
            relations=counts["relations"],
            documents=counts["documents"],
            chunks=counts["chunks"],
            entity_types=[EntityTypeInfo(label=k, count=v) for k, v in label_counts.items()],
            relation_types=[RelationTypeInfo(label=k, count=v) for k, v in rel_counts.items()],
        )
        # The counts are evidence too: an answer about them can cite [n].
        result.n = self._evidence("status", "status", result.to_llm_text(max_chars=100_000))
        result.hint = hint
        return result

    async def _declared_ontology(self) -> Ontology:
        """The persisted ontology, read without registering anything.

        ``GraphRAG.get_ontology()`` registers the default entity types into
        an empty ontology graph; a read tool must not write, and those types
        would then be part of every ontology declared afterwards. Before
        anything is persisted this is the ontology the GraphRAG was built with.
        """
        store = getattr(self._rag, "_ontology_store", None)
        if store is None:
            return await self._rag.get_ontology()
        loaded = await store.load()
        if loaded.entities or loaded.relations:
            return loaded
        return getattr(self._rag, "ontology", None) or Ontology()

    async def get_ontology(self) -> OntologyResult:
        """The declared ontology with live counts (undeclared stored types too)."""
        await self._require_graph()
        return await self._ontology_result()

    async def _ontology_result(self) -> OntologyResult:
        # No existence check: set_ontology may run before the first ingest.
        ontology = await self._declared_ontology()
        label_counts, rel_counts = await ops.schema_counts(self._store)
        declared_e = {e.label: e for e in ontology.entities}
        declared_r = {r.label: r for r in ontology.relations}
        entity_types = [
            EntityTypeInfo(
                label=label,
                description=declared_e[label].description if label in declared_e else None,
                count=count,
                properties=[a.name for a in declared_e[label].properties]
                if label in declared_e
                else [],
            )
            for label, count in sorted({**{e: 0 for e in declared_e}, **label_counts}.items())
        ]
        relation_types = [
            RelationTypeInfo(
                label=label,
                description=declared_r[label].description if label in declared_r else None,
                patterns=list(declared_r[label].patterns) if label in declared_r else [],
                count=count,
            )
            for label, count in sorted({**{r: 0 for r in declared_r}, **rel_counts}.items())
        ]
        result = OntologyResult(entity_types=entity_types, relation_types=relation_types)
        result.n = self._evidence("get_ontology", "ontology", result.to_llm_text(max_chars=100_000))
        return result

    def _make_strategy(
        self,
        top_k: int,
        ontology: Ontology | None = None,
        plan: frozenset[str] | None = None,
    ) -> MultiPathRetrieval:
        """A MultiPathRetrieval sized to ``top_k`` (its constructor only assigns).

        ``plan`` names the MultiPath paths to run (default: every path
        retrieve reads).
        """
        return MultiPathRetrieval(
            graph_store=self._rag._graph_store,
            vector_store=self._rag._vector_store,
            embedder=self._rag.embedder,
            llm=self._rag.llm,
            chunk_top_k=top_k,
            rel_top_k=top_k,
            max_entities=max(2 * top_k, 10),
            router=_FixedPlan(plan or _ALL_MULTIPATH_PATHS),
            ontology=ontology or self._rag._global_ontology,
        )

    async def _search(
        self, question: str, top_k: int, ctx: Context, plan: frozenset[str] | None = None
    ) -> Any:
        """Run MultiPath retrieval without writing to the graph.

        ``GraphRAG.retrieve`` first initialises the ontology, which on a graph
        nothing was declared on yet registers the default entity types: a
        write, from a read tool. Until the GraphRAG has done that itself, run
        the strategy directly instead. MultiPath reads the ontology only for
        text-to-Cypher, which retrieve never enables, so the persisted one is
        not loaded for it.
        """
        if getattr(self._rag, "_ontology_initialized", True):
            return await self._rag.retrieve(
                question, strategy=self._make_strategy(top_k, plan=plan), ctx=ctx
            )
        await self._rag._validate_graph_config(ctx=ctx)
        ctx.ensure_budget("retrieval strategy search")
        return await self._make_strategy(top_k, plan=plan).search(question, ctx)

    async def retrieve(
        self,
        question: str,
        *,
        top_k: int = 8,
        expand_hops: int = 1,
        include_chunks: bool = True,
        paths: list[str] | None = None,
        ctx: Context | None = None,
    ) -> RetrieveResult:
        """Hybrid retrieval for ``question``: entities, relations, facts, passages.

        Runs the SDK's MultiPath retrieval (vector, full-text and graph) and
        returns the ranked context, without generating an answer. Each
        entity, relation, fact and passage is a piece of evidence. ``paths``
        runs only the named search paths (see ``RETRIEVE_PATHS``); by default
        every path runs.
        """
        inp = RetrieveInput(
            question=question,
            top_k=top_k,
            expand_hops=expand_hops,
            include_chunks=include_chunks,
            paths=paths,  # type: ignore[arg-type]
        )
        wanted = set(inp.paths or RETRIEVE_PATHS)
        if not inp.include_chunks:
            wanted.discard("passages")
        if not wanted:
            raise ToolError(
                "paths names only passages, but include_chunks is false.",
                hint="Drop include_chunks, or add other paths.",
            )
        if "links" in wanted and not wanted & _SEED_PATHS:
            wanted.add("exact_names")  # links walk from found entities
        plan = frozenset(_MULTIPATH_PATH[p] for p in wanted if p in _MULTIPATH_PATH)
        ctx = ctx or self._new_ctx()
        await self._require_graph()
        store = self._store
        rr = await self._search(inp.question, inp.top_k, ctx, plan)
        prov: dict[str, Any] = rr.metadata.get("provenance") or {}
        found = list(dict.fromkeys(e["id"] for e in (prov.get("entities") or []) if e.get("id")))
        cards = await ops.enrich_entities(store, found)
        # Only stored entities: a vector store that returns no endpoint ids
        # leaves the facts path naming entities by a made-up id (the name,
        # lower-cased) that matches no node.
        seed_ids = [eid for eid in found if eid in cards][: inp.top_k]
        # Every query runs before any evidence is numbered, so a failing
        # query leaves nothing citable that the agent was never shown.
        triples = (
            await ops.expand_triples(
                store, seed_ids, hops=inp.expand_hops, cap=min(4 * inp.top_k, 40)
            )
            if "links" in wanted
            else []
        )
        prov_chunks = (
            [c for c in (prov.get("chunks") or []) if c.get("id")][: inp.top_k]
            if inp.include_chunks
            else []
        )
        doc_map = (
            await ops.chunk_documents(store, [c["id"] for c in prov_chunks]) if prov_chunks else {}
        )

        entities = []
        for seed_id in seed_ids:
            card = cards[seed_id]
            text = card["name"] + (f" ({card['label']})" if card["label"] else "")
            if card.get("description"):
                text += f": {card['description']}"
            if props := properties_text(card.get("properties") or {}):
                text += f" ({props})"
            entities.append(
                EntityCard(
                    n=self._evidence("retrieve", "entity", text, entity_id=card["id"]), **card
                )
            )

        relations = []
        shown: set[tuple[str, str, str]] = set()
        for t in triples:
            text = f"{t['source']} -[{t['type']}]-> {t['target']}"
            if t["fact"]:
                text += f": {t['fact']}"
            shown.add((t["source"].lower(), t["type"], t["target"].lower()))
            relations.append(
                RelationTriple(
                    n=self._evidence("retrieve", "relation", text),
                    source=t["source"],
                    type=t["type"],
                    target=t["target"],
                    fact=t["fact"] or None,
                )
            )

        facts = []
        for raw in (prov.get("facts") or [])[: 2 * inp.top_k]:
            text, key = _short_fact(str(raw))
            if not text or (key and key in shown):
                continue
            facts.append(Fact(n=self._evidence("retrieve", "fact", text), text=text))

        chunks: list[ChunkRef] = []
        documents: list[DocumentRef] = []
        if inp.include_chunks:
            seen_docs: set[str] = set()
            for c in prov_chunks:
                doc_id, doc_path = doc_map.get(c["id"], ("", c.get("document_path", "")))
                text = str(c.get("text", ""))
                chunks.append(
                    ChunkRef(
                        n=self._evidence(
                            "retrieve",
                            "passage",
                            text,
                            source=doc_path or doc_id,
                            chunk_id=c["id"],
                        ),
                        chunk_id=c["id"],
                        document_id=doc_id,
                        document_path=doc_path if doc_path != doc_id else None,
                        text=text,
                    )
                )
                if doc_id and doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    documents.append(DocumentRef(document_id=doc_id, document_path=doc_path))
        hint = None
        if not (entities or relations or facts or chunks):
            hint = (
                "Nothing relevant was found on any path. Rephrase the question with "
                "other words, or call status to check the graph holds data."
            )
            if inp.paths:
                hint = (
                    "Nothing relevant was found on the paths you chose. Run retrieve "
                    "again without paths."
                )
        return RetrieveResult(
            question=inp.question,
            entities=entities,
            relations=relations,
            facts=facts,
            chunks=chunks,
            documents=documents,
            hint=hint,
        )

    async def check_citations(self, answer: str) -> CitationCheckResult:
        """Check ``answer``'s ``[n]`` citations against this session's evidence.

        Markers that match no evidence are removed from the returned answer.
        Grounded means at least one real piece of evidence is cited. On a
        grounded answer it also lists claims (sentences stating a figure or a
        name) with no [n] of their own, and cited claims whose figures or names
        the cited evidence does not contain; neither changes ``grounded``. A
        sentence that says the evidence lacks something ("The graph does not
        say who founded Globex.") is not a claim and needs no [n].
        """
        inp = CheckCitationsInput(answer=answer)
        checked = ground_answer(inp.answer, self._ledger)
        sources = [
            EvidenceRef(n=ev.n, tool=ev.tool, kind=ev.kind, content=ev.content, source=ev.source)
            for ev in checked.citations(self._ledger)
        ]
        hints: list[str] = []
        if checked.reason == "no_citations" and states_a_gap(checked.answer):
            hints.append(
                "Saying the evidence does not answer needs no citation; cite [n] only "
                "for the claims the answer makes."
            )
        elif checked.reason == "no_citations":
            hints.append("Cite the evidence numbers [n] from tool results after each claim.")
        elif checked.reason == "invalid_citations":
            hints.append(
                "None of those numbers were returned in this session; cite only [n] "
                "from tool results."
            )
        elif checked.invalid:
            hints.append("Some numbers were not returned in this session and were removed.")
        if checked.uncited:
            hints.append("Give each listed claim the [n] that states it, or drop it.")
        if checked.unsupported:
            hints.append(
                "The cited evidence does not contain the missing figures or names: cite "
                "the [n] that states them, or correct the claim."
            )
        hint = " ".join(hints) or None
        return CitationCheckResult(
            grounded=checked.grounded,
            reason=checked.reason,
            cited=checked.cited,
            invalid=checked.invalid,
            uncited=checked.uncited,
            unsupported=[
                ClaimCheck(sentence=c.sentence, missing=c.missing) for c in checked.unsupported
            ],
            answer=checked.answer,
            sources=sources,
            hint=hint,
        )

    # ── Write tools ──────────────────────────────────────────────

    def _ensure_writable(self, tool: str) -> None:
        if self._read_only:
            raise ReadOnlyViolation(f"{tool} is disabled: the toolkit is read-only.")

    def _require_root(self, tool: str) -> Path:
        if self._files_root is None:
            raise ToolError(
                f"{tool} needs a files root, and none is configured.",
                hint="Ingest text instead, or start the toolkit with files_root=...",
            )
        return self._files_root

    async def preview_files(
        self,
        pattern: str = "**/*",
        *,
        offset: int = 0,
        limit: int = 20,
        max_chars: int = 1000,
        ctx: Context | None = None,
    ) -> PreviewFilesResult:
        """A page of the files under the files root, with a sample of each.

        Each sample is taken from the middle of the file's text, where the
        subject matter is rather than the title page or the references. Page
        through every file (``next_offset``) to see the whole corpus before
        designing an ontology. Samples are not evidence: nothing here is
        stored, and they get no ``[n]``.
        """
        self._ensure_writable("preview_files")
        root = self._require_root("preview_files")
        inp = PreviewFilesInput(pattern=pattern, offset=offset, limit=limit, max_chars=max_chars)
        try:
            rels = await asyncio.to_thread(fs.list_files, root, inp.pattern)
        except ValueError as exc:
            raise ToolError(str(exc), hint="Use a glob relative to the files root.") from None
        types: dict[str, int] = {}
        for rel in rels:
            kind = Path(rel).suffix.lower().lstrip(".")
            types[kind] = types.get(kind, 0) + 1
        page = rels[inp.offset : inp.offset + inp.limit]
        per_file = min(inp.max_chars, SAMPLE_BUDGET_CHARS // max(len(page), 1))
        gate = asyncio.Semaphore(_FILE_CONCURRENCY)

        async def sample(rel: str) -> DocSample:
            kind = Path(rel).suffix.lower().lstrip(".")
            size = 0
            async with gate:
                try:
                    real = await asyncio.to_thread(fs.canonical_file, root, rel)
                    text, chars, estimated, size = await asyncio.to_thread(
                        fs.sample_confined, root, real, per_file, self._max_file_bytes
                    )
                except Exception as exc:  # one unreadable file must not fail the page
                    return DocSample(path=rel, type=kind, size_bytes=size, error=_file_error(exc))
            return DocSample(
                path=rel,
                type=kind,
                size_bytes=size,
                chars=chars,
                chars_estimated=estimated,
                sample=text,
            )

        samples = list(await asyncio.gather(*(sample(rel) for rel in page)))
        # Keep the page within the result cap here, so next_offset names the
        # first file left out instead of a cut further on dropping files unseen.
        room = _PAGE_BUDGET_CHARS - len(inp.pattern) - len(json.dumps(types)) - 500
        kept = 0
        for item in samples:
            room -= len(json.dumps(item.model_dump(exclude_none=True), ensure_ascii=False)) + 2
            if kept and room < 0:
                break
            kept += 1
        samples = samples[:kept]
        end = inp.offset + kept
        hint = None
        if not rels:
            hint = (
                f"No supported files match {inp.pattern!r} "
                f"({', '.join(fs.SUPPORTED_SUFFIXES)}). Try pattern='**/*'."
            )
        elif end < len(rels):
            hint = "Keep paging until next_offset is null, then design the ontology."
        return PreviewFilesResult(
            pattern=inp.pattern,
            total=len(rels),
            offset=inp.offset,
            next_offset=end if end < len(rels) else None,
            types=types,
            files=samples,
            hint=hint,
        )

    async def set_ontology(
        self,
        entities: Sequence[Mapping[str, Any] | BaseModel] = (),
        relations: Sequence[Mapping[str, Any] | BaseModel] = (),
    ) -> OntologyResult:
        """Declare entity and relation types; extraction keeps only these.

        Adds to what is declared: new types are added, a given description
        replaces the old one, and new ``[source, target]`` patterns are added
        to a relation. Nothing is removed or renamed. Call it before the first
        ingest; types added later apply to files ingested (or re-extracted)
        afterwards.
        """
        self._ensure_writable("set_ontology")
        inp = SetOntologyInput(
            entities=[_as_dict(e) for e in entities],
            relations=[_as_dict(r) for r in relations],
        )
        if not inp.entities and not inp.relations:
            raise ToolError(
                "set_ontology needs at least one entity or relation type.",
                hint="Pass entities=[{label, description}] and/or relations=[...].",
            )
        labels = [(e.label, True) for e in inp.entities] + [(r.label, False) for r in inp.relations]
        for label, entity in labels:
            error = _label_error(label, entity=entity)
            if error is not None:
                raise error

        # Load, merge and write under one lock shared by every session, so a
        # concurrent call cannot write back a stale copy over this one.
        async with self._lock("ontology"):
            rag = self._rag
            pending = getattr(rag, "ontology", None) or Ontology()
            if not rag._ontology_initialized and (pending.entities or pending.relations):
                # An ontology given to GraphRAG(...) and not registered yet.
                await rag._ensure_ontology_initialized()
            stored = await rag._ontology_store.load()
            persisted = bool(stored.entities or stored.relations)
            base = stored if persisted else pending
            entities_by = {e.label: e for e in base.entities}
            relations_by = {r.label: r for r in base.relations}
            known = set(entities_by) | {e.label for e in inp.entities}
            undeclared = sorted(
                {end for r in inp.relations for pair in r.patterns for end in pair} - known
            )
            if undeclared:
                raise ToolError(
                    f"Relation patterns use undeclared entity types: {', '.join(undeclared)}.",
                    hint="Declare them in entities (in this call or an earlier one).",
                )

            new_entities: list[Entity] = []
            new_relations: list[Relation] = []
            descriptions: list[tuple[str, str, str]] = []  # (kind, label, description)
            patterns: list[tuple[str, str, str]] = []  # (relation, source, target)
            for e in inp.entities:
                old = entities_by.get(e.label)
                if old is None:
                    entity = Entity(label=e.label, description=e.description)
                    entities_by[e.label] = entity
                    new_entities.append(entity)
                elif e.description and e.description != old.description:
                    descriptions.append(("entity", e.label, e.description))
            for r in inp.relations:
                pairs = list(dict.fromkeys((src, tgt) for src, tgt in r.patterns))
                old_rel = relations_by.get(r.label)
                if old_rel is None:
                    relation = Relation(label=r.label, description=r.description, patterns=pairs)
                    relations_by[r.label] = relation
                    new_relations.append(relation)
                    continue
                if r.description and r.description != old_rel.description:
                    descriptions.append(("relation", r.label, r.description))
                have = {tuple(p) for p in old_rel.patterns}
                patterns.extend((r.label, s, t) for s, t in pairs if (s, t) not in have)

            if not persisted:
                # Nothing stored yet: fold every change into one declaration.
                for kind, label, text in descriptions:
                    if kind == "entity":
                        entities_by[label] = entities_by[label].model_copy(
                            update={"description": text}
                        )
                    else:
                        relations_by[label] = relations_by[label].model_copy(
                            update={"description": text}
                        )
                for rel, src, tgt in patterns:
                    old_rel = relations_by[rel]
                    relations_by[rel] = old_rel.model_copy(
                        update={"patterns": [*old_rel.patterns, (src, tgt)]}
                    )
                descriptions, patterns = [], []
            try:
                if new_entities or new_relations or not persisted:
                    merged = base.model_copy(
                        update={
                            "entities": list(entities_by.values()),
                            "relations": list(relations_by.values()),
                        }
                    )
                    await self._rag.set_ontology(merged)
                for kind, label, text in descriptions:
                    if kind == "entity":
                        await self._rag.set_entity_description(label, text)
                    else:
                        await self._rag.set_relation_description(label, text)
                for rel, src, tgt in patterns:
                    await self._rag.add_relation_pattern(rel, src, tgt)
            except ValueError as exc:
                raise ToolError(
                    f"The ontology was not accepted: {_short(str(exc), 500)}",
                    hint="Existing types cannot change their properties; add new types instead.",
                ) from None
            finally:
                # What was written, also after a partial failure, for the extractor.
                await rag._refresh_global_ontology()
        return await self._ontology_result()

    async def ingest(
        self,
        text: str | None = None,
        *,
        document_id: str | None = None,
        files: Sequence[str] | None = None,
        plan: Sequence[Mapping[str, Any] | BaseModel] | None = None,
        reextract: bool = False,
        ctx: Context | None = None,
    ) -> IngestResult:
        """Extract ``text``, or ``files`` under the files root, into the graph.

        ``files`` takes paths and globs relative to the files root (at most
        50 files per call). ``plan`` is a list of rules, each a glob
        (``match``) with the chunker / extractor / resolver for the files it
        matches; the first matching rule wins, and the allowed options are
        those of ``plan_limits``. A file whose content is unchanged is
        skipped unless ``reextract`` is set, which re-extracts it in place
        (after the ontology has grown, for instance); without a ``plan``, a
        file re-extracted this way keeps the plan of its last ingest by
        this toolkit. That memory is not persisted: after a restart the
        planner plans the file afresh, so pass ``plan`` to repeat a custom one.

        Under ``finalize_policy="on_write"`` this also finalizes (unless
        nothing changed), whose cost grows with the graph; prefer "manual"
        and one :meth:`finalize` per batch.
        """
        self._ensure_writable("ingest")
        inp = IngestInput(
            text=text,
            document_id=document_id,
            files=list(files) if files is not None else None,
            plan=[_as_dict(r) for r in plan] if plan is not None else None,
            reextract=reextract,
        )
        ctx = ctx or self._new_ctx()
        failed_after_write = False
        if inp.text is not None:
            if inp.reextract:
                await self._refuse_file_document(inp.document_id)
                async with self._lock("update"):
                    result = await self._rag.update(
                        text=inp.text,
                        document_id=inp.document_id,
                        force=True,
                        if_missing="ingest",
                        ctx=ctx,
                    )
                status = "reextracted" if result.replaced_existing else "ingested"
            else:
                result = await self._rag.ingest(text=inp.text, document_id=inp.document_id, ctx=ctx)
                unchanged = (result.metadata or {}).get("skipped_unchanged")
                status = "unchanged" if unchanged else "ingested"
            out = IngestResult(
                document_id=result.document_info.uid,
                status=status,
                chunks_indexed=result.chunks_indexed,
                nodes_created=result.nodes_created,
                relationships_created=result.relationships_created,
                pruned=PrunedTypes.from_report(result.metadata.get("pruned")),
            )
        else:
            out, failed_after_write = await self._ingest_files(
                inp.files or [], inp.plan or [], inp.reextract, ctx
            )
        # Only a document written needs a finalize: an unchanged one or a file
        # refused before writing does not, but a file that failed later may
        # have left nodes behind.
        written = ("ingested", "reextracted", "updated")
        wrote = out.status in written or any(d.status in written for d in out.documents)
        wrote = wrote or failed_after_write
        finalize_error = None
        if self._finalize_policy == "on_write" and wrote:
            try:
                await self._rag.finalize()
                out.finalized = True
            except Exception as exc:  # the documents are written: keep their result
                logger.warning("finalize after ingest failed (%s)", type(exc).__name__)
                finalize_error = (
                    f"The documents were written, but finalize failed ({type(exc).__name__}); "
                    "call finalize to retry."
                )
        out.hint = self._ingest_hint(out)
        if finalize_error:
            out.hint = f"{finalize_error} {out.hint}" if out.hint else finalize_error
        return out

    async def _refuse_file_document(self, document_id: str | None) -> None:
        """Refuse to re-extract text over a document that was read from a file.

        ``reextract`` replaces the stored document with the text given, so
        without this check it could overwrite any file the operator ingested.
        """
        if document_id is None:
            return
        record = await self._rag._graph_store.get_document_record(document_id)
        if record is None:
            return
        from_file = (
            record.kind == "structured"
            or (record.path is not None and record.path != document_id)
            or os.path.isabs(record.path or "")
        )
        if not from_file and self._files_root is not None:
            under = self._files_root / document_id
            from_file = await asyncio.to_thread(
                lambda: under.is_file() and fs._inside(self._files_root, under)
            )
        if from_file:
            raise ToolError(
                f"Document {document_id!r} was read from a file; text cannot replace it.",
                hint="To extract a file again, use ingest(files=[...], reextract=true).",
            )

    def _lock(self, name: str) -> asyncio.Lock:
        """A lock shared by every session copy of this toolkit.

        Remade when the running loop changes, as a lock that once waited on one
        loop raises when acquired from another.
        """
        loop = asyncio.get_running_loop()
        held = self._shared.get(name)
        if held is None or held[0] is not loop:
            held = (loop, asyncio.Lock())
            self._shared[name] = held
        return held[1]

    def _remember_plan(self, path: str, plan: Mapping[str, Any]) -> None:
        plans: dict[str, dict[str, Any]] = self._shared.setdefault("file_plans", {})
        plans.pop(path, None)
        plans[path] = dict(plan)
        while len(plans) > _PLAN_MEMORY:
            del plans[next(iter(plans))]

    def _last_plan(self, path: str) -> Any:
        """The plan ``path`` was last ingested with by this toolkit, or ``None``."""
        from graphrag_sdk.ingestion.ingestion_planner import IngestionPlan

        used = self._shared.get("file_plans", {}).get(path)
        if not used:
            return None
        try:
            return IngestionPlan(
                chunker=used["chunker"],
                extractor=used["extractor"],
                resolver=used["resolver"],
                chunker_params=dict(used.get("chunker_params") or {}),
                extractor_params=dict(used.get("extractor_params") or {}),
                resolver_params=dict(used.get("resolver_params") or {}),
                reason="the plan of this file's last ingest",
            )
        except (KeyError, ValueError):
            return None

    async def _ingest_files(
        self,
        entries: list[str],
        rules: list[PlanRuleInput],
        reextract: bool,
        ctx: Context,
    ) -> tuple[IngestResult, bool]:
        """The files' result, and whether a file failed after it may have written."""
        from graphrag_sdk.ingestion.ingestion_planner import (
            PlanRule,
            RuleBasedIngestionPlanner,
        )

        root = self._require_root("ingest(files=...)")
        try:
            rels = await asyncio.to_thread(fs.resolve_files, root, entries)
            # Sources (and so document ids) are paths relative to the root.
            planner = RuleBasedIngestionPlanner(
                [PlanRule.from_dict(r.model_dump(exclude_none=True)) for r in rules],
                **self._plan_limits,
            )
        except ValueError as exc:
            raise ToolError(str(exc), hint="Fix the plan rules.") from None
        await asyncio.to_thread(fs.check_sizes, root, rels, self._max_file_bytes)
        loader = _ConfinedLoader(root, self._max_file_bytes)
        errors: list[BaseException] = []
        found: dict[str, IngestedDocument] = {}
        # A file already in the graph goes through update(): re-ingesting an
        # edited file would otherwise add its new chunks beside the old ones.
        if reextract:
            stored = list(rels)
        else:
            records = await asyncio.gather(
                *(self._rag._graph_store.get_document_record(rel) for rel in rels)
            )
            stored = [rel for rel, rec in zip(rels, records) if rec is not None]
        fresh = [rel for rel in rels if rel not in frozenset(stored)]
        if fresh:
            results = await self._rag.ingest(
                fresh,
                planner=planner,
                loader=loader,
                max_concurrency=_FILE_CONCURRENCY,
                ctx=ctx,
            )
            for rel, res in zip(fresh, results):
                if isinstance(res, BaseException):
                    errors.append(res)
                    found[rel] = IngestedDocument(path=rel, status="failed", error=_file_error(res))
                    continue
                meta = res.metadata or {}
                used = (meta.get("plan") or {}).get("plan")
                if used:
                    self._remember_plan(rel, used)
                found[rel] = _ingested(
                    rel,
                    res,
                    "unchanged" if meta.get("skipped_unchanged") else "ingested",
                    used,
                )
        gate = asyncio.Semaphore(_FILE_CONCURRENCY)
        # update() runs one at a time across every session, as apply_changes()
        # does: one document's orphan cleanup can delete an entity another
        # document's update has just written but not yet linked.
        writer = self._lock("update")

        async def redo(rel: str) -> IngestedDocument:
            try:
                async with gate:
                    # Planned like a first ingest (the same Markdown parsing
                    # of a .txt, per-slot fallback and chunk budget), with
                    # the file's last plan unless rules were passed.
                    document = await loader.load(rel, ctx)
                    last = None if rules else self._last_plan(rel)
                    active = planner if last is None else _SamePlan(last, planner.max_chunks)
                    report: dict[str, Any] = {}
                    (
                        chunker,
                        extractor,
                        resolver,
                        document,
                    ) = await self._rag._plan_ingestion_strategies(
                        text=None,
                        source=rel,
                        chunker=None,
                        extractor=None,
                        resolver=None,
                        planner=active,
                        ctx=ctx,
                        report=report,
                        loaded_document=document,
                    )
                async with writer:
                    res = await self._rag.update(
                        rel,
                        force=reextract,
                        if_missing="ingest",
                        loader=_Loaded(document) if document is not None else loader,
                        chunker=chunker,
                        extractor=extractor,
                        resolver=resolver,
                        ctx=ctx,
                    )
            except Exception as exc:  # one bad file must not fail the batch
                errors.append(exc)
                return IngestedDocument(path=rel, status="failed", error=_file_error(exc))
            if res.no_op:
                return _ingested(rel, res, "unchanged", None)
            plan_info = report.get("plan")
            if plan_info:
                self._remember_plan(rel, plan_info)
                if last is None and not rules:
                    plan_info = {**plan_info, "reason": f"{plan_info['reason']}{_NO_PLAN}"}
            if not res.replaced_existing:
                status = "ingested"
            else:
                status = "reextracted" if reextract else "updated"
            return _ingested(rel, res, status, plan_info)

        for rel, doc in zip(stored, await asyncio.gather(*(redo(rel) for rel in stored))):
            found[rel] = doc
        docs = [found[rel] for rel in rels]

        total = PrunedTypes()
        for doc in docs:
            if doc.pruned is not None:
                total.add(doc.pruned)
        out = IngestResult(
            chunks_indexed=sum(d.chunks_indexed for d in docs),
            nodes_created=sum(d.nodes_created for d in docs),
            relationships_created=sum(d.relationships_created for d in docs),
            documents=docs,
            failed=sum(d.status == "failed" for d in docs),
            pruned=total if (total.missing_types or total.dangling) else None,
        )
        return out, any(_may_have_written(e) for e in errors)

    def _ingest_hint(self, out: IngestResult) -> str | None:
        hints: list[str] = []
        if out.failed:
            hints.append(f"{out.failed} file(s) failed; see each document's error.")
        if out.pruned is not None and out.pruned.missing_types:
            affected = [
                d.path for d in out.documents if d.pruned is not None and d.pruned.missing_types
            ]
            redo = (
                f"ingest(files={affected[:10]}{' ...' if len(affected) > 10 else ''}, "
                "reextract=true)"
                if affected
                else f"ingest(text=..., document_id={out.document_id!r}, reextract=true)"
            )
            pruned = out.pruned
            if pruned.pattern_mismatches:
                hints.append(_pattern_hint(pruned, redo))
            if pruned.unknown_endpoints:
                n = sum(pruned.unknown_endpoints.values())
                hints.append(
                    f"{n} relation(s) ({_names_hint(sorted(pruned.unknown_endpoints))}) were "
                    "dropped because an endpoint's type was not resolved (stored as Unknown); "
                    "a pattern cannot name Unknown. Unknown can also be a declared type the "
                    "extractor classified with low confidence. Only if those entities are "
                    "of a type the ontology lacks, declare it with "
                    "set_ontology(entities=[...]) and add it to the relation's patterns, "
                    f"then {redo}."
                )
            if pruned.entities or pruned.relations or pruned.unknown:
                found = (
                    "Extraction found types the ontology does not declare (see pruned). "
                    if pruned.entities or pruned.relations
                    else ""
                )
                unresolved = (
                    f"{pruned.unknown} entities have an unresolved type (stored as Unknown): "
                    "a type the ontology lacks, or a declared type the extractor classified "
                    "with low confidence; status shows the total. "
                    if pruned.unknown
                    else ""
                )
                hints.append(
                    f"{found}{unresolved}If new types matter, add them with set_ontology, "
                    f"then {redo}."
                )
        if out.documents and all(d.status == "unchanged" for d in out.documents):
            hints.append("Every file was unchanged; pass reextract=true to extract them again.")
        if out.status == "unchanged":
            hints.append(
                "This text is already stored; to extract it again, pass "
                f"document_id={out.document_id!r} and reextract=true."
            )
        if self._finalize_policy == "manual" and not out.finalized:
            hints.append("Call finalize once after this batch of ingests.")
        return " ".join(hints) or None

    async def finalize(self) -> FinalizeReport:
        """Run ``GraphRAG.finalize()`` (dedup, embeddings, indexes).

        Under "on_write" each ingest already finalized, so this does
        nothing; under "never" it raises :class:`ConfigError`.
        """
        self._ensure_writable("finalize")
        if self._finalize_policy == "on_write":
            return FinalizeReport(finalized=True, hint="Each ingest already finalized.")
        if self._finalize_policy == "never":
            raise ConfigError(
                'finalize_policy="never": the toolkit never finalizes; call '
                "GraphRAG.finalize() yourself."
            )
        result = await self._rag.finalize()
        return FinalizeReport(
            finalized=True,
            entities_deduplicated=int(getattr(result, "entities_deduplicated", 0) or 0),
            entities_embedded=int(getattr(result, "entities_embedded", 0) or 0),
            relationships_embedded=int(getattr(result, "relationships_embedded", 0) or 0),
            null_stubs_removed=int(getattr(result, "null_stubs_removed", 0) or 0),
        )
