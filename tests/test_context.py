"""Tests for core/context.py — execution context with tenancy and latency budgeting."""
from __future__ import annotations

import time

import pytest

from graphrag_sdk.core.context import Context


class TestContextCreation:
    def test_defaults(self):
        ctx = Context()
        assert ctx.tenant_id == "default"
        assert ctx.trace_id  # auto UUID
        assert ctx.latency_budget_ms is None
        assert ctx.metadata == {}

    def test_custom_values(self):
        ctx = Context(
            tenant_id="acme",
            trace_id="trace-123",
            latency_budget_ms=1000.0,
            metadata={"env": "test"},
        )
        assert ctx.tenant_id == "acme"
        assert ctx.trace_id == "trace-123"
        assert ctx.latency_budget_ms == 1000.0
        assert ctx.metadata["env"] == "test"

    def test_unique_trace_ids(self):
        ctx1 = Context()
        ctx2 = Context()
        assert ctx1.trace_id != ctx2.trace_id


class TestElapsedTime:
    def test_elapsed_increases(self):
        ctx = Context()
        t1 = ctx.elapsed_ms
        time.sleep(0.01)
        t2 = ctx.elapsed_ms
        assert t2 > t1

    def test_elapsed_is_positive(self):
        ctx = Context()
        assert ctx.elapsed_ms >= 0


class TestLatencyBudget:
    def test_no_budget(self):
        ctx = Context()
        assert ctx.remaining_budget_ms is None
        assert ctx.budget_exceeded is False

    def test_budget_remaining(self):
        ctx = Context(latency_budget_ms=10000.0)
        remaining = ctx.remaining_budget_ms
        assert remaining is not None
        assert remaining > 0
        assert ctx.budget_exceeded is False

    def test_budget_exceeded(self):
        ctx = Context(latency_budget_ms=0.0)
        time.sleep(0.001)
        assert ctx.budget_exceeded is True
        assert ctx.remaining_budget_ms == 0.0


class TestChildContext:
    def test_inherits_parent(self):
        parent = Context(tenant_id="acme", latency_budget_ms=5000.0, metadata={"env": "prod"})
        child = parent.child()
        assert child.tenant_id == "acme"
        assert child.trace_id == parent.trace_id
        assert child.metadata["env"] == "prod"

    def test_override_tenant(self):
        parent = Context(tenant_id="acme")
        child = parent.child(tenant_id="beta")
        assert child.tenant_id == "beta"
        assert parent.tenant_id == "acme"  # parent unchanged

    def test_child_metadata_merged(self):
        parent = Context(metadata={"env": "prod", "version": "1.0"})
        child = parent.child(metadata={"version": "2.0", "extra": "yes"})
        assert child.metadata["env"] == "prod"  # inherited
        assert child.metadata["version"] == "2.0"  # overridden
        assert child.metadata["extra"] == "yes"  # added

    def test_child_budget_is_remaining(self):
        parent = Context(latency_budget_ms=5000.0)
        time.sleep(0.01)
        child = parent.child()
        assert child.latency_budget_ms is not None
        assert child.latency_budget_ms < 5000.0

    def test_child_budget_override(self):
        parent = Context(latency_budget_ms=5000.0)
        child = parent.child(latency_budget_ms=1000.0)
        assert child.latency_budget_ms == 1000.0


class TestContextLogging:
    def test_log_does_not_raise(self, caplog):
        ctx = Context(tenant_id="test-t", trace_id="aaaa-bbbb-cccc-dddd")
        import logging

        with caplog.at_level(logging.INFO):
            ctx.log("Hello from test")
        assert "test-t" in caplog.text
        assert "Hello from test" in caplog.text
