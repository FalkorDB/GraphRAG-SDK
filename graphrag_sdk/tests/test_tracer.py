"""Tests for telemetry/tracer.py — Span-based tracing."""
from __future__ import annotations

import time

import pytest

from graphrag_sdk.telemetry.tracer import Span, Tracer


class TestSpan:
    def test_creation(self):
        s = Span(name="test-op")
        assert s.name == "test-op"
        assert s.end_time is None
        assert s.duration_ms is None

    def test_end(self):
        s = Span(name="op")
        time.sleep(0.01)
        s.end()
        assert s.end_time is not None
        assert s.duration_ms is not None
        assert s.duration_ms > 0

    def test_metadata(self):
        s = Span(name="op", metadata={"key": "value"})
        assert s.metadata["key"] == "value"

    def test_children(self):
        parent = Span(name="parent")
        child = Span(name="child")
        parent.children.append(child)
        assert len(parent.children) == 1

    def test_to_dict(self):
        s = Span(name="test")
        s.end()
        d = s.to_dict()
        assert d["name"] == "test"
        assert "duration_ms" in d
        assert "children" in d
        assert isinstance(d["children"], list)

    def test_to_dict_with_children(self):
        parent = Span(name="parent")
        child = Span(name="child")
        child.end()
        parent.children.append(child)
        parent.end()
        d = parent.to_dict()
        assert len(d["children"]) == 1
        assert d["children"][0]["name"] == "child"


class TestTracer:
    def test_creation(self):
        t = Tracer(service_name="test-svc")
        assert t.service_name == "test-svc"
        assert t.completed_spans == []

    def test_span_context_manager(self):
        t = Tracer()
        with t.span("operation") as s:
            time.sleep(0.01)
        assert len(t.completed_spans) == 1
        assert t.completed_spans[0].name == "operation"
        assert t.completed_spans[0].duration_ms > 0

    def test_nested_spans(self):
        t = Tracer()
        with t.span("outer") as outer:
            with t.span("inner") as inner:
                time.sleep(0.005)
        # Only outer is a root span
        assert len(t.completed_spans) == 1
        assert t.completed_spans[0].name == "outer"
        assert len(t.completed_spans[0].children) == 1
        assert t.completed_spans[0].children[0].name == "inner"

    def test_span_metadata_kwargs(self):
        t = Tracer()
        with t.span("op", key="value", count=3) as s:
            pass
        assert s.metadata["key"] == "value"
        assert s.metadata["count"] == 3

    def test_start_span_manual(self):
        t = Tracer()
        s = t.start_span("manual-op")
        assert s.end_time is None
        s.end()
        assert s in t.completed_spans

    def test_summary(self):
        t = Tracer(service_name="my-service")
        with t.span("op1"):
            with t.span("sub1"):
                pass
        with t.span("op2"):
            pass
        summary = t.summary()
        assert "my-service" in summary
        assert "op1" in summary
        assert "sub1" in summary
        assert "op2" in summary

    def test_reset(self):
        t = Tracer()
        with t.span("something"):
            pass
        assert len(t.completed_spans) == 1
        t.reset()
        assert len(t.completed_spans) == 0

    def test_multiple_root_spans(self):
        t = Tracer()
        with t.span("a"):
            pass
        with t.span("b"):
            pass
        with t.span("c"):
            pass
        assert len(t.completed_spans) == 3

    def test_span_exception_still_records(self):
        t = Tracer()
        with pytest.raises(ValueError):
            with t.span("failing"):
                raise ValueError("oops")
        assert len(t.completed_spans) == 1
        assert t.completed_spans[0].duration_ms is not None

    def test_deeply_nested_spans(self):
        t = Tracer()
        with t.span("L1"):
            with t.span("L2"):
                with t.span("L3"):
                    pass
        root = t.completed_spans[0]
        assert root.name == "L1"
        assert root.children[0].name == "L2"
        assert root.children[0].children[0].name == "L3"
