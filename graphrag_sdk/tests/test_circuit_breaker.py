"""Tests for core/circuit_breaker.py — async-safe circuit breaker."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from graphrag_sdk.core.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request is True

    async def test_stays_closed_on_success(self):
        cb = CircuitBreaker(failure_threshold=3)
        await cb.record_success()
        assert cb.state == CircuitState.CLOSED

    async def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            await cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request is False

    async def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        await cb.record_failure()
        await cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request is True

    async def test_rejects_when_open(self):
        cb = CircuitBreaker(failure_threshold=1)
        await cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request is False

    async def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        await cb.record_failure()
        assert cb.state == CircuitState.OPEN

        await asyncio.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request is True

    async def test_closes_on_success_in_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        await cb.record_failure()
        await asyncio.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        await cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request is True

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
