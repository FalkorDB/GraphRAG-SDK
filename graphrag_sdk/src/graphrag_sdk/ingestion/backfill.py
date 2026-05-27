"""LLM-driven backfill executor for ontology evolution.

When the ontology gains structure that already-ingested data lacks (a new
attribute on an existing entity type, a brand-new entity type, a brand-new
relation pattern), we need to re-scan the stored chunks and ask the LLM to
fill the new shape in.

The :py:class:`BackfillExecutor` owns the concurrency, retries, and
per-chunk idempotency-marker bookkeeping. The caller supplies four
operation-specific callbacks:

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

    Returned by every Group-3 ``GraphRAG`` method. ``failed_chunks`` carries
    the chunk ids that raised so the caller can retry just those; the
    operation as a whole does not raise on per-chunk failures.

    ``estimated_cost_usd`` is ``None`` in v1 — :py:class:`LLMInterface` does
    not expose usage stats. A future provider-specific hook can populate it.
    """

    operation_id: str
    target_nodes: int = 0
    chunks_scanned: int = 0
    chunks_skipped: int = 0
    llm_calls: int = 0
    values_filled: int = 0
    values_skipped: int = 0
    dropped_for_coercion: int = 0
    failed_chunks: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    estimated_cost_usd: float | None = None


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
        """
        result = BackfillResult(operation_id=op_id)
        start = time.monotonic()
        sem = asyncio.Semaphore(self._concurrency)

        async def _process_one(ctx: ChunkContext) -> None:
            async with sem:
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
                        ctx.chunk_id, op_id, exc,
                    )
                    logger.debug("Backfill chunk failure details", exc_info=True)
                    result.failed_chunks.append(ctx.chunk_id)

        tasks: list[asyncio.Task[None]] = []
        async for ctx in chunks:
            tasks.append(asyncio.create_task(_process_one(ctx)))
        if tasks:
            await asyncio.gather(*tasks)

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
