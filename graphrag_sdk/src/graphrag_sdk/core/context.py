# GraphRAG SDK — Core: Context Object
# Threaded through every strategy call for tracing, tenancy, and budgeting.
# Origin: User design (TenantID, TraceID, Latency Budget) + Neo4j RunContext (simplified).

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from graphrag_sdk.core.models import TokenUsage

logger = logging.getLogger(__name__)


@dataclass
class Context:
    """Execution context passed to every strategy and pipeline step.

    Carries:
    - **tenant_id**: Multi-tenant isolation key
    - **trace_id**: Unique request ID for distributed tracing
    - **latency_budget_ms**: Optional hard latency cap (strategies can check remaining time)
    - **tracer**: Reference to the telemetry tracer (if configured)
    - **metadata**: Arbitrary key-value bag for custom data

    Unlike Neo4j's dual ``run()`` / ``run_with_context()`` pattern,
    Context is always present — there is no execution without it.
    """

    tenant_id: str = "default"
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    latency_budget_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: TokenUsage = field(default_factory=TokenUsage)
    _start_time: float = field(default_factory=time.monotonic, repr=False)

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds elapsed since context creation."""
        return (time.monotonic() - self._start_time) * 1000

    @property
    def remaining_budget_ms(self) -> float | None:
        """Remaining latency budget in ms, or None if no budget set."""
        if self.latency_budget_ms is None:
            return None
        return max(0.0, self.latency_budget_ms - self.elapsed_ms)

    @property
    def budget_exceeded(self) -> bool:
        """True if the latency budget has been exceeded."""
        remaining = self.remaining_budget_ms
        return remaining is not None and remaining <= 0

    def child(self, **overrides: Any) -> Context:
        """Create a child context inheriting tenant/trace but with optional overrides.

        Useful for per-step contexts within a pipeline.

        Note: ``usage`` is **not** inherited — the child starts with zero counters.
        Token usage recorded in a child context is NOT propagated back to the parent.
        For full usage tracking, pass the parent context directly to all callees.
        """
        return Context(
            tenant_id=overrides.get("tenant_id", self.tenant_id),
            trace_id=overrides.get("trace_id", self.trace_id),
            latency_budget_ms=overrides.get("latency_budget_ms", self.remaining_budget_ms),
            metadata={**self.metadata, **overrides.get("metadata", {})},
        )

    def record_usage(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        embedding_tokens: int = 0,
    ) -> None:
        """Accumulate token counts from a single LLM or embedding call.

        Called by provider implementations after every successful API
        response.  Totals are available on :attr:`usage` at the end of
        the operation.  Safe to call with all-zero values (no-op).
        """
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens
        self.usage.embedding_tokens += embedding_tokens

    def log(self, message: str, level: int = logging.INFO) -> None:
        """Log with context prefix for traceability."""
        prefix = f"[tenant={self.tenant_id} trace={self.trace_id[:8]}]"
        logger.log(level, f"{prefix} {message}")
