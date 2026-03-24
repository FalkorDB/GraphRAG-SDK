"""Tests for cypher_generation module."""
from __future__ import annotations

import pytest

from graphrag_sdk.retrieval.strategies.cypher_generation import (
    extract_cypher,
    validate_cypher,
    _sanitize_cypher,
)


# ── extract_cypher ────────────────────────────────────────────────


class TestExtractCypher:
    def test_extracts_from_labeled_code_block(self):
        text = "```cypher\nMATCH (n) RETURN n\n```"
        assert extract_cypher(text) == "MATCH (n) RETURN n"

    def test_extracts_from_unlabeled_code_block(self):
        text = "```\nMATCH (n) RETURN n\n```"
        assert extract_cypher(text) == "MATCH (n) RETURN n"

    def test_extracts_raw_cypher(self):
        text = "MATCH (n:Person) RETURN n.name LIMIT 10"
        assert extract_cypher(text) == text

    def test_returns_empty_for_non_cypher(self):
        assert extract_cypher("Just some plain text") == ""

    def test_returns_empty_for_empty_input(self):
        assert extract_cypher("") == ""
        assert extract_cypher(None) == ""

    def test_extracts_from_block_with_surrounding_text(self):
        text = "Here is the query:\n```cypher\nMATCH (p:Person) RETURN p.name\n```\nDone."
        assert extract_cypher(text) == "MATCH (p:Person) RETURN p.name"


# ── validate_cypher ───────────────────────────────────────────────


class TestValidateCypher:
    def test_valid_query_passes(self):
        assert validate_cypher("MATCH (n:Person) RETURN n.name LIMIT 10") == []

    def test_rejects_empty(self):
        errors = validate_cypher("")
        assert any("Empty" in e for e in errors)

    def test_rejects_write_operations(self):
        for keyword in ["CREATE", "DELETE", "SET", "MERGE", "REMOVE"]:
            cypher = f"MATCH (n) {keyword} (m) RETURN n"
            errors = validate_cypher(cypher)
            assert any("Write" in e or "read-only" in e.lower() for e in errors), \
                f"{keyword} should be rejected"

    def test_rejects_missing_return(self):
        errors = validate_cypher("MATCH (n:Person) WHERE n.name = 'Alice'")
        assert any("RETURN" in e for e in errors)

    def test_rejects_unknown_labels(self):
        errors = validate_cypher("MATCH (n:Spaceship) RETURN n.name")
        assert any("Unknown label: Spaceship" in e for e in errors)

    def test_accepts_known_labels(self):
        for label in ["Person", "Organization", "Technology", "Location",
                       "__Entity__", "Chunk", "Document"]:
            assert validate_cypher(f"MATCH (n:{label}) RETURN n.name") == []

    def test_rejects_call_procedures(self):
        errors = validate_cypher("CALL db.labels() YIELD label RETURN label")
        assert any("CALL" in e for e in errors)

    def test_rejects_load_csv(self):
        errors = validate_cypher("LOAD CSV FROM 'file:///data.csv' AS row RETURN row")
        assert any("LOAD CSV" in e for e in errors)

    def test_rejects_multi_statement(self):
        errors = validate_cypher("MATCH (n) RETURN n; MATCH (m) DELETE m")
        assert any("Multiple" in e or "statement" in e.lower() for e in errors)

    def test_must_start_with_read_keyword(self):
        errors = validate_cypher("CREATE (n:Person) RETURN n")
        assert any("must start with" in e.lower() for e in errors)

    def test_strips_comments_before_validation(self):
        # Write keyword hidden in a comment should not trigger rejection
        cypher = "// CREATE is not allowed\nMATCH (n:Person) RETURN n.name"
        assert validate_cypher(cypher) == []

    def test_optional_match_allowed(self):
        cypher = "OPTIONAL MATCH (n:Person)-[:RELATES]-(m) RETURN n.name, m.name"
        assert validate_cypher(cypher) == []

    def test_unwind_allowed(self):
        cypher = "UNWIND [1,2,3] AS x MATCH (n) WHERE id(n)=x RETURN n"
        assert validate_cypher(cypher) == []

    def test_with_allowed(self):
        cypher = "WITH 'Alice' AS name MATCH (n:Person) WHERE n.name = name RETURN n"
        assert validate_cypher(cypher) == []


# ── _sanitize_cypher ──────────────────────────────────────────────


class TestSanitizeCypher:
    def test_adds_limit_when_missing(self):
        result = _sanitize_cypher("MATCH (n) RETURN n")
        assert "LIMIT" in result

    def test_keeps_existing_limit(self):
        cypher = "MATCH (n) RETURN n LIMIT 10"
        result = _sanitize_cypher(cypher)
        assert result.count("LIMIT") == 1

    def test_removes_shortest_path(self):
        cypher = "MATCH path = shortestPath((a)-[*]-(b)) RETURN path"
        result = _sanitize_cypher(cypher)
        assert "shortestPath" not in result

    def test_removes_all_shortest_paths(self):
        cypher = "MATCH p = allShortestPaths((a)-[*]-(b)) RETURN p"
        result = _sanitize_cypher(cypher)
        assert "allShortestPaths" not in result

    def test_removes_path_assignment(self):
        cypher = "MATCH path = (a)-[:RELATES]->(b) RETURN a, b"
        result = _sanitize_cypher(cypher)
        assert "path =" not in result and "path=" not in result


# ── execute_cypher_retrieval ──────────────────────────────────────


class TestExecuteCypherRetrieval:
    async def test_returns_empty_on_generation_failure(self):
        """When LLM returns garbage, should return empty results."""
        from unittest.mock import AsyncMock, MagicMock
        from graphrag_sdk.core.models import LLMResponse
        from graphrag_sdk.retrieval.strategies.cypher_generation import (
            execute_cypher_retrieval,
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=LLMResponse(content="I don't know"))
        mock_graph = MagicMock()

        facts, entities = await execute_cypher_retrieval(mock_graph, mock_llm, "test?")
        assert facts == []
        assert entities == {}

    async def test_returns_empty_on_execution_error(self):
        """When Cypher execution fails, should return empty results."""
        from unittest.mock import AsyncMock, MagicMock
        from graphrag_sdk.core.models import LLMResponse
        from graphrag_sdk.retrieval.strategies.cypher_generation import (
            execute_cypher_retrieval,
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=LLMResponse(
                content="```cypher\nMATCH (n:Person) RETURN n.name\n```"
            )
        )
        mock_graph = MagicMock()
        mock_graph.query_raw = AsyncMock(side_effect=Exception("connection error"))

        facts, entities = await execute_cypher_retrieval(mock_graph, mock_llm, "test?")
        assert facts == []
        assert entities == {}

    async def test_parses_result_rows(self):
        """Successful execution should parse rows into facts and entities."""
        from unittest.mock import AsyncMock, MagicMock
        from graphrag_sdk.core.models import LLMResponse
        from graphrag_sdk.retrieval.strategies.cypher_generation import (
            execute_cypher_retrieval,
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=LLMResponse(
                content="```cypher\nMATCH (n:Person) RETURN n.name\n```"
            )
        )
        result_mock = MagicMock()
        result_mock.result_set = [["Alice"], ["Bob"]]
        mock_graph = MagicMock()
        mock_graph.query_raw = AsyncMock(return_value=result_mock)

        facts, entities = await execute_cypher_retrieval(mock_graph, mock_llm, "test?")
        assert len(facts) == 2
        assert "Alice" in facts[0]
        assert "alice" in entities
        assert "bob" in entities
