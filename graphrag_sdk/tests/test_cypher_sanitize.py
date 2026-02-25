# Tests for graphrag_sdk.utils.cypher.sanitize_cypher_label

import pytest

from graphrag_sdk.utils.cypher import sanitize_cypher_label


class TestSanitizeCypherLabel:
    def test_normal_label_unchanged(self):
        assert sanitize_cypher_label("Person") == "Person"

    def test_strips_whitespace(self):
        assert sanitize_cypher_label("  Person  ") == "Person"

    def test_strips_backticks(self):
        assert sanitize_cypher_label("Per`son") == "Person"

    def test_injection_attempt(self):
        """Backtick injection should be neutralized."""
        result = sanitize_cypher_label("Label`) DETACH DELETE n //")
        assert "`" not in result
        assert result == "Label) DETACH DELETE n //"

    def test_empty_after_cleaning_raises(self):
        with pytest.raises(ValueError, match="Invalid Cypher label"):
            sanitize_cypher_label("```")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="Invalid Cypher label"):
            sanitize_cypher_label("   ")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid Cypher label"):
            sanitize_cypher_label("")

    def test_underscore_labels(self):
        assert sanitize_cypher_label("__Entity__") == "__Entity__"

    def test_label_with_spaces(self):
        assert sanitize_cypher_label("Some Label") == "Some Label"
