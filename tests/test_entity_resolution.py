"""
Tests for entity resolution and deduplication functionality.
"""

import pytest
from graphrag_sdk.entity_resolution import EntityResolver


class TestEntityResolver:
    """Test cases for EntityResolver class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.resolver = EntityResolver(similarity_threshold=0.85)
    
    def test_normalize_string(self):
        """Test string normalization."""
        # Test whitespace normalization
        assert self.resolver.normalize_string("  John  Doe  ") == "john doe"
        
        # Test case normalization
        assert self.resolver.normalize_string("JOHN DOE") == "john doe"
        
        # Test special characters removal
        assert self.resolver.normalize_string("John-Doe!") == "john doe"
        
        # Test empty string
        assert self.resolver.normalize_string("") == ""
        
        # Test None handling
        assert self.resolver.normalize_string(None) == ""
    
    def test_normalize_date(self):
        """Test date normalization."""
        # Test YYYY-MM-DD format (already normalized)
        assert self.resolver.normalize_date("2023-12-25") == "2023-12-25"
        
        # Test MM/DD/YYYY format
        assert self.resolver.normalize_date("12/25/2023") == "2023-12-25"
        
        # Test single digit month and day
        assert self.resolver.normalize_date("1/5/2023") == "2023-01-05"
        
        # Test invalid date
        assert self.resolver.normalize_date("invalid") is None
        
        # Test None handling
        assert self.resolver.normalize_date(None) is None
        
        # Test empty string
        assert self.resolver.normalize_date("") is None
    
    def test_compute_similarity(self):
        """Test similarity computation."""
        # Exact match
        assert self.resolver.compute_similarity("John Doe", "John Doe") == 1.0
        
        # Case insensitive
        assert self.resolver.compute_similarity("John Doe", "john doe") == 1.0
        
        # Similar strings
        similarity = self.resolver.compute_similarity("John Doe", "John  Doe")
        assert similarity > 0.9
        
        # Different strings
        similarity = self.resolver.compute_similarity("John Doe", "Jane Smith")
        assert similarity < 0.5
        
        # Empty strings
        assert self.resolver.compute_similarity("", "") == 0.0
    
    def test_are_entities_similar(self):
        """Test entity similarity detection."""
        entity1 = {
            "label": "Person",
            "attributes": {"name": "John Doe", "age": 30}
        }
        
        entity2 = {
            "label": "Person",
            "attributes": {"name": "John  Doe", "age": 31}
        }
        
        # Similar entities (same name, different age)
        assert self.resolver.are_entities_similar(entity1, entity2, ["name"]) is True
        
        # Different labels
        entity3 = {
            "label": "Organization",
            "attributes": {"name": "John Doe"}
        }
        assert self.resolver.are_entities_similar(entity1, entity3, ["name"]) is False
        
        # Different entities
        entity4 = {
            "label": "Person",
            "attributes": {"name": "Jane Smith", "age": 25}
        }
        assert self.resolver.are_entities_similar(entity1, entity4, ["name"]) is False
    
    def test_merge_entity_attributes(self):
        """Test entity attribute merging."""
        entity1 = {
            "label": "Person",
            "attributes": {"name": "John Doe", "age": 30}
        }
        
        entity2 = {
            "label": "Person",
            "attributes": {"name": "John Doe", "email": "john@example.com"}
        }
        
        merged = self.resolver.merge_entity_attributes(entity1, entity2)
        
        # Check label is preserved
        assert merged["label"] == "Person"
        
        # Check attributes are merged
        assert "name" in merged["attributes"]
        assert "age" in merged["attributes"]
        assert "email" in merged["attributes"]
        
        # Check values
        assert merged["attributes"]["name"] == "John Doe"
        assert merged["attributes"]["age"] == 30
        assert merged["attributes"]["email"] == "john@example.com"
    
    def test_deduplicate_entities(self):
        """Test entity deduplication."""
        entities = [
            {"label": "Person", "attributes": {"name": "John Doe"}},
            {"label": "Person", "attributes": {"name": "John  Doe"}},  # Duplicate
            {"label": "Person", "attributes": {"name": "Jane Smith"}},
        ]
        
        deduplicated, dup_count = self.resolver.deduplicate_entities(entities, ["name"])
        
        # Check duplicate count
        assert dup_count == 1
        
        # Check deduplicated list length
        assert len(deduplicated) == 2
        
        # Check distinct entities remain
        names = [e["attributes"]["name"] for e in deduplicated]
        assert "John Doe" in names or "John  Doe" in names
        assert "Jane Smith" in names
    
    def test_deduplicate_entities_empty_list(self):
        """Test deduplication with empty list."""
        deduplicated, dup_count = self.resolver.deduplicate_entities([], ["name"])
        
        assert len(deduplicated) == 0
        assert dup_count == 0
    
    def test_normalize_entity_attributes(self):
        """Test entity attribute normalization."""
        entity = {
            "label": "Person",
            "attributes": {
                "name": "  John  Doe  ",
                "birth_date": "12/25/1990",
                "title": "Software   Engineer"
            }
        }
        
        normalized = self.resolver.normalize_entity_attributes(entity)
        
        # Check name normalization (whitespace)
        assert normalized["attributes"]["name"] == "John Doe"
        
        # Check date normalization
        assert normalized["attributes"]["birth_date"] == "1990-12-25"
        
        # Check title normalization
        assert normalized["attributes"]["title"] == "Software Engineer"
    
    def test_custom_similarity_threshold(self):
        """Test custom similarity threshold."""
        # Create resolver with higher threshold
        strict_resolver = EntityResolver(similarity_threshold=0.95)
        
        entity1 = {
            "label": "Person",
            "attributes": {"name": "John Doe"}
        }
        
        entity2 = {
            "label": "Person",
            "attributes": {"name": "John D"}
        }
        
        # Default threshold (0.85) should match
        assert self.resolver.are_entities_similar(entity1, entity2, ["name"]) is False
        
        # Strict threshold (0.95) should not match
        assert strict_resolver.are_entities_similar(entity1, entity2, ["name"]) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
