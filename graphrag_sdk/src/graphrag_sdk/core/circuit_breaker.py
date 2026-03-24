# GraphRAG SDK 2.0 — Core: Circuit Breaker
# Async-safe circuit breaker for database connections.

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing — reject immediately
    HALF_OPEN = "half_open"  # Probing with one request


@dataclass
class CircuitBreaker:
    """Async-safe circuit breaker for database connections.

    Tracks consecutive failures. After ``failure_threshold`` failures,
    the circuit opens and rejects requests immediately for
    ``recovery_timeout`` seconds before allowing a probe request.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    _failure_count: int = field(default=0, init=False, repr=False)
    _last_failure_time: float = field(default=0.0, init=False, repr=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @property
    def state(self) -> CircuitState:
        """Current circuit state (computed, no side effects)."""
        if (
            self._state == CircuitState.OPEN
            and time.monotonic() - self._last_failure_time >= self.recovery_timeout
        ):
            return CircuitState.HALF_OPEN
        return self._state

    async def record_success(self) -> None:
        """Record a successful request — resets failure count and closes circuit."""
        async with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    async def record_failure(self) -> None:
        """Record a failed request — opens circuit after threshold."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN

    async def allow_request(self) -> bool:
        """Whether a request should be allowed through.

        Atomic: in HALF_OPEN state, only the first caller gets permission
        (transitions to HALF_OPEN internally so subsequent callers are
        rejected until ``record_success()`` or ``record_failure()``).
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if (
                self._state == CircuitState.OPEN
                and time.monotonic() - self._last_failure_time >= self.recovery_timeout
            ):
                # Transition to HALF_OPEN — only this caller gets through
                self._state = CircuitState.HALF_OPEN
                return True
            return False
