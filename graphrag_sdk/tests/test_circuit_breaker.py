"""Tests for core/circuit_breaker.py — async-safe circuit breaker."""
from __future__ import annotations

import asyncio

from graphrag_sdk.core.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    async def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert await cb.allow_request() is True

    async def test_stays_closed_on_success(self):
        cb = CircuitBreaker(failure_threshold=3)
        await cb.record_success()
        assert cb.state == CircuitState.CLOSED

    async def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            await cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert await cb.allow_request() is False

    async def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        await cb.record_failure()
        await cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert await cb.allow_request() is True

    async def test_rejects_when_open(self):
        cb = CircuitBreaker(failure_threshold=1)
        await cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert await cb.allow_request() is False

    async def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        await cb.record_failure()
        assert cb.state == CircuitState.OPEN

        await asyncio.sleep(0.15)
        # First caller gets through (transitions to HALF_OPEN)
        assert await cb.allow_request() is True
        # State is now HALF_OPEN — subsequent callers are rejected
        assert cb.state == CircuitState.HALF_OPEN
        assert await cb.allow_request() is False

    async def test_closes_on_success_in_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        await cb.record_failure()
        await asyncio.sleep(0.15)
        # Trigger HALF_OPEN transition via allow_request
        assert await cb.allow_request() is True

        await cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert await cb.allow_request() is True

    async def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        await cb.record_failure()
        await cb.record_failure()
        await cb.record_success()
        # After reset, need 3 more failures to open
        await cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    async def test_concurrent_failures_are_safe(self):
        cb = CircuitBreaker(failure_threshold=5)
        # Fire 10 concurrent failures — should not corrupt state
        await asyncio.gather(*[cb.record_failure() for _ in range(10)])
        assert cb.state == CircuitState.OPEN

    async def test_half_open_allows_only_one_probe(self):
        """Only one concurrent caller should get through in HALF_OPEN."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        await cb.record_failure()
        await asyncio.sleep(0.15)

        # Fire 5 concurrent allow_request calls — only 1 should succeed
        results = await asyncio.gather(*[cb.allow_request() for _ in range(5)])
        assert results.count(True) == 1
        assert results.count(False) == 4
