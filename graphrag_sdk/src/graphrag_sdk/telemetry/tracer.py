# GraphRAG SDK 2.0 — Telemetry: Tracer
# OpenTelemetry-compatible span tracking for enterprise observability.
# Origin: User design — first-class telemetry module (replaces Neo4j's custom EventNotifier).

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """A performance tracking span.

    Records the name, start time, end time, and metadata of an operation.
    Compatible with OpenTelemetry span interface for future integration.
    """

    name: str
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)

    def end(self) -> None:
        """Mark this span as complete."""
        self.end_time = time.monotonic()

    @property
    def duration_ms(self) -> float | None:
        """Duration in milliseconds, or None if not yet ended."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Serialise span for logging or export."""
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }


class Tracer:
    """Lightweight tracing system for SDK operations.

    Provides span-based performance tracking that is compatible with
    OpenTelemetry's interface. In v1 this is a simple in-process tracer;
    in v2+ it can wrap a real OpenTelemetry TracerProvider.

    Example::

        tracer = Tracer("graphrag-sdk")

        with tracer.span("ingestion.pipeline") as s:
            s.metadata["source"] = "doc.pdf"
            # do work...

        # After execution
        for span in tracer.completed_spans:
            print(f"{span.name}: {span.duration_ms:.1f}ms")
    """

    def __init__(self, service_name: str = "graphrag-sdk") -> None:
        self.service_name = service_name
        self._spans: list[Span] = []
        self._active_span: Span | None = None

    @contextmanager
    def span(self, name: str, **metadata: Any) -> Generator[Span, None, None]:
        """Create a performance span as a context manager.

        Args:
            name: Name of the operation being traced.
            **metadata: Arbitrary metadata to attach to the span.

        Yields:
            The active Span object.
        """
        s = Span(name=name, metadata=metadata)

        parent = self._active_span
        if parent is not None:
            parent.children.append(s)

        self._active_span = s

        try:
            yield s
        finally:
            s.end()
            self._active_span = parent

            if parent is None:
                self._spans.append(s)

            logger.debug(f"Span [{name}]: {s.duration_ms:.1f}ms")

    def start_span(self, name: str, **metadata: Any) -> Span:
        """Manually start a span (remember to call ``span.end()``).

        Prefer the context manager ``span()`` when possible.
        """
        s = Span(name=name, metadata=metadata)
        self._spans.append(s)
        return s

    @property
    def completed_spans(self) -> list[Span]:
        """All completed root-level spans."""
        return [s for s in self._spans if s.end_time is not None]

    def summary(self) -> str:
        """Human-readable summary of all completed spans."""
        lines = [f"=== Trace Summary ({self.service_name}) ==="]
        for s in self.completed_spans:
            self._format_span(s, lines, indent=0)
        return "\n".join(lines)

    def _format_span(self, span: Span, lines: list[str], indent: int) -> None:
        prefix = "  " * indent
        duration = f"{span.duration_ms:.1f}ms" if span.duration_ms else "in-progress"
        lines.append(f"{prefix}{span.name}: {duration}")
        for child in span.children:
            self._format_span(child, lines, indent + 1)

    def reset(self) -> None:
        """Clear all recorded spans."""
        self._spans.clear()
        self._active_span = None
