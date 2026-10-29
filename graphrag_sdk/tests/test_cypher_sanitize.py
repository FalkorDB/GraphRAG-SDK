# Tests for graphrag_sdk.utils.cypher.sanitize_cypher_label and merge_key_value

import pytest

from graphrag_sdk.utils.cypher import merge_key_value, sanitize_cypher_label


class TestMergeKeyValue:
    def test_dict_key_ignores_key_order_and_mixed_key_types(self):
        a = {"label": "X", 1: "alt", "nested": {2: "x", "y": 3}}
        b = {"nested": {"y": 3, 2: "x"}, 1: "alt", "label": "X"}
        assert merge_key_value(a) == merge_key_value(b)

    def test_dict_never_matches_a_list_of_its_pairs(self):
        assert merge_key_value({"a": 1}) != merge_key_value([[["str", "a"], 1]])

    @pytest.mark.parametrize("a, b", [(5, 5.0), ("x", "x")])
    def test_equal_values_share_a_key(self, a, b):
        assert merge_key_value(a) == merge_key_value(b)

    @pytest.mark.parametrize("a, b", [(True, 1), ("5", 5), ({1: "a"}, {"1": "a"})])
    def test_different_kinds_never_share_a_key(self, a, b):
        assert merge_key_value(a) != merge_key_value(b)


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
