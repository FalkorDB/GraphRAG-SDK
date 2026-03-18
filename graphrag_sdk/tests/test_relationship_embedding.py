"""Tests for GraphRelationship.to_fact_text() helper."""
from __future__ import annotations

import pytest

from graphrag_sdk.core.models import GraphRelationship


# ── Tests ──────────────────────────────────────────────────────


class TestToFactText:
    def test_with_description(self):
        """to_fact_text() should include description when present."""
        rel = GraphRelationship(
            start_node_id="alice",
            end_node_id="acme",
            type="WORKS_AT",
            properties={
                "src_name": "Alice",
                "tgt_name": "Acme Corp",
                "description": "employed as engineer",
            },
        )
        assert rel.to_fact_text() == "(Alice, WORKS_AT, Acme Corp): employed as engineer"

    def test_without_description(self):
        """to_fact_text() should work without description."""
        rel = GraphRelationship(
            start_node_id="alice",
            end_node_id="acme",
            type="WORKS_AT",
            properties={
                "src_name": "Alice",
                "tgt_name": "Acme Corp",
            },
        )
        assert rel.to_fact_text() == "(Alice, WORKS_AT, Acme Corp)"

    def test_falls_back_to_node_ids(self):
        """to_fact_text() should fall back to node IDs when no src_name/tgt_name."""
        rel = GraphRelationship(
            start_node_id="alice",
            end_node_id="acme",
            type="WORKS_AT",
            properties={},
        )
        assert rel.to_fact_text() == "(alice, WORKS_AT, acme)"
