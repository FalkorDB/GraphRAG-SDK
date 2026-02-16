# GraphRAG SDK 2.0 — Core: Context Object
# Threaded through every strategy call for tracing, tenancy, and budgeting.
# Origin: User design (TenantID, TraceID, Latency Budget) + Neo4j RunContext (simplified).

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

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
    latency_budget_ms: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _start_time: float = field(default_factory=time.monotonic, repr=False)

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds elapsed since context creation."""
        return (time.monotonic() - self._start_time) * 1000

    @property
    def remaining_budget_ms(self) -> Optional[float]:
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
        """
        return Context(
            tenant_id=overrides.get("tenant_id", self.tenant_id),
            trace_id=overrides.get("trace_id", self.trace_id),
            latency_budget_ms=overrides.get("latency_budget_ms", self.remaining_budget_ms),
            metadata={**self.metadata, **overrides.get("metadata", {})},
        )

    def log(self, message: str, level: int = logging.INFO) -> None:
        """Log with context prefix for traceability."""
        prefix = f"[tenant={self.tenant_id} trace={self.trace_id[:8]}]"
        logger.log(level, f"{prefix} {message}")
