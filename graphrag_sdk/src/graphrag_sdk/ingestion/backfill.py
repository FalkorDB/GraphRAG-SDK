"""LLM-driven backfill executor for ontology evolution.

When the ontology gains structure that already-ingested data lacks (a new
attribute on an existing entity type, a brand-new entity type, a brand-new
relation pattern), we need to re-scan the stored chunks and ask the LLM to
fill the new shape in.

The :py:class:`BackfillExecutor` owns the concurrency and per-chunk
idempotency-marker bookkeeping. (Per-call retries are delegated to
``LLMInterface.ainvoke`` — the executor itself does not retry; failed
chunks land in ``BackfillResult.failed_chunks`` for caller-driven
retry.) The caller supplies four operation-specific callbacks:

- a scope iterator that yields :py:class:`ChunkContext` rows;
- a ``prompt_builder`` that produces a focused per-chunk prompt;
- a ``parse_fn`` that validates the LLM JSON;
- a ``merge_fn`` that writes the parsed payload into the data graph.

Markers live on the chunk node (``c.extracted_ops: list[string]``).
``op_id`` is deterministic from the operation signature, so re-running the
same backfill skips already-processed chunks naturally — see the module
docstring of :py:mod:`graphrag_sdk.api.main` for the full evolution model.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from graphrag_sdk.core.models import Ontology
from graphrag_sdk.core.providers import LLMInterface
from graphrag_sdk.storage.graph_store import GraphStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkContext:
    """Payload handed to ``prompt_builder`` / ``parse_fn`` / ``merge_fn``.

    Carries the chunk identity, raw text, the in-scope entities (per the
    scope query that produced this row), and a snapshot of the ontology
    so prompt rendering doesn't have to round-trip to storage.
    """

    chunk_id: str
    chunk_text: str
    payload: dict[str, Any]
    ontology: Ontology


@dataclass(frozen=True)
class BackfillMergeStats:
    """Counts returned by ``merge_fn`` after writing one chunk's payload."""

    values_filled: int = 0
    values_skipped: int = 0
    dropped_for_coercion: int = 0


@dataclass
class BackfillResult:
    """Post-run summary of a backfill operation.

    Returned by the opportunistic-discovery methods
    (``GraphRAG.backfill_entity`` and ``GraphRAG.backfill_relation_pattern``).
    ``failed_chunks`` carries chunk ids that raised so the caller can retry
    just those; these methods do not raise on per-chunk failures.

    ``chunks_skipped`` counts chunks already marked with the same
    ``op_id`` from a prior run — i.e. work this run did NOT have to redo.
    Read it to confirm idempotency on reruns.

    ``chunks_in_scope`` is the total chunks that match the scope query
    before the marker filter — populated even on a dry run so callers
    can preview LLM cost before incurring it.

    ``estimated_cost_usd`` is ``None`` in v1 — :py:class:`LLMInterface` does
    not expose usage stats. A future provider-specific hook can populate it.
    """

    operation_id: str
    target_nodes: int = 0
    chunks_in_scope: int = 0
    chunks_scanned: int = 0
    chunks_skipped: int = 0
    llm_calls: int = 0
    values_filled: int = 0
    values_skipped: int = 0
    dropped_for_coercion: int = 0
    failed_chunks: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    estimated_cost_usd: float | None = None


@dataclass
class EvolutionResult:
    """Return value of an atomic ontology-evolution call.

    Carries the refreshed ontology plus the observability counters from
    the internal LLM backfill. Returned by ``GraphRAG.add_attribute``
    and any future atomic-evolve methods. On a successful return:

    - The ontology graph has been updated.
    - Every chunk in scope was processed (LLM call + merge) or skipped
      because a prior idempotent run already marked it.
    - No chunk failures remain — hard failures would have raised
      :py:class:`OntologyEvolutionError` instead, which carries the
      failing chunk ids on ``OntologyEvolutionError.failed_chunks``.

    ``chunks_in_scope`` is the total chunks the scope query matched
    before the marker filter. Populated even on a dry run so callers
    can preview LLM cost before incurring it.

    On a ``dry_run=True`` call ``chunks_in_scope`` is the only
    meaningful field — no LLM was invoked, no ontology was written.
    """

    ontology: Any  # graphrag_sdk.core.models.Ontology (avoid circular import)
    chunks_in_scope: int = 0
    chunks_scanned: int = 0
    chunks_skipped: int = 0
    llm_calls: int = 0
    values_filled: int = 0
    values_skipped: int = 0  # LLM returned null
    elapsed_s: float = 0.0


class OntologyEvolutionError(RuntimeError):
    """Raised when an atomic evolution call cannot complete its data migration.

    The ontology graph is NOT updated when this is raised — the data graph
    has been partially mutated but the schema still reflects the pre-call
    state. Re-running the same evolution call is safe and idempotent
    (chunk markers ensure already-processed chunks are skipped).

    Inspect :py:attr:`failed_chunks` to identify the chunks that hard-failed
    (LLM error, parse error, etc.) so the underlying cause can be addressed
    before retry.
    """

    def __init__(
        self,
        message: str,
        *,
        failed_chunks: list[str],
        chunks_scanned: int,
    ) -> None:
        super().__init__(message)
        self.failed_chunks = list(failed_chunks)
        self.chunks_scanned = chunks_scanned


PromptBuilder = Callable[[ChunkContext], str]
ParseFn = Callable[[str, ChunkContext], Any]
MergeFn = Callable[[Any, ChunkContext], Awaitable[BackfillMergeStats]]


class BackfillExecutor:
    """Drive an LLM-backed re-scan over already-ingested chunks.

    Concurrency-bounded (``asyncio.Semaphore``), per-chunk-idempotent
    (``GraphStore.mark_chunk_extracted``), and resilient — a chunk that
    raises during parse/merge lands in ``BackfillResult.failed_chunks``
    and the run continues.
    """

    def __init__(
        self,
        llm: LLMInterface,
        graph_store: GraphStore,
        *,
        concurrency: int = 4,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self._llm = llm
        self._graph_store = graph_store
        self._concurrency = concurrency

    async def run(
        self,
        *,
        op_id: str,
        chunks: AsyncIterator[ChunkContext],
        prompt_builder: PromptBuilder,
        parse_fn: ParseFn,
        merge_fn: MergeFn,
    ) -> BackfillResult:
        """Execute the backfill. See module docstring for callback contracts.

        ``chunks`` is consumed once. ``op_id`` is the deterministic
        operation signature used both for the chunk marker and the
        ``BackfillResult.operation_id`` echo.

        Concurrency is bounded by a worker-pool pattern: ``concurrency``
        long-lived worker tasks consume ``ChunkContext`` instances from a
        bounded :py:class:`asyncio.Queue`. Live ``asyncio.Task`` count is
        therefore O(concurrency) regardless of corpus size — without this
        pattern, a 50k-chunk backfill would create 50k pending tasks up
        front and hold them all in memory.
        """
        result = BackfillResult(operation_id=op_id)
        start = time.monotonic()
        # Queue size = 2x concurrency to keep workers warm without
        # backpressuring the producer too eagerly.
        queue: asyncio.Queue[ChunkContext | None] = asyncio.Queue(maxsize=self._concurrency * 2)

        async def _worker() -> None:
            while True:
                ctx = await queue.get()
                try:
                    if ctx is None:
                        return
                    try:
                        prompt = prompt_builder(ctx)
                        response = await self._llm.ainvoke(prompt)
                        result.llm_calls += 1
                        parsed = parse_fn(response.content, ctx)
                        stats = await merge_fn(parsed, ctx)
                        result.values_filled += stats.values_filled
                        result.values_skipped += stats.values_skipped
                        result.dropped_for_coercion += stats.dropped_for_coercion
                        await self._graph_store.mark_chunk_extracted(ctx.chunk_id, op_id)
                        result.chunks_scanned += 1
                    except Exception as exc:
                        logger.warning(
                            "Backfill chunk %s failed for op '%s': %s",
                            ctx.chunk_id,
                            op_id,
                            exc,
                        )
                        logger.debug("Backfill chunk failure details", exc_info=True)
                        result.failed_chunks.append(ctx.chunk_id)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(_worker()) for _ in range(self._concurrency)]
        async for ctx in chunks:
            await queue.put(ctx)
        # One sentinel per worker — graceful stop after the queue drains.
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

        result.elapsed_s = time.monotonic() - start
        logger.info(
            "Backfill op '%s' done — scanned=%d failed=%d filled=%d skipped=%d elapsed=%.2fs",
            op_id,
            result.chunks_scanned,
            len(result.failed_chunks),
            result.values_filled,
            result.values_skipped,
            result.elapsed_s,
        )
        return result
