"""
Tests for extraction validation functionality.
"""

import pytest
from graphrag_sdk import Ontology, Entity, Relation, Attribute, AttributeType
from graphrag_sdk.extraction_validator import ExtractionValidator


class TestExtractionValidator:
    """Test cases for ExtractionValidator class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create a simple ontology for testing
        person_entity = Entity(
            label="Person",
            attributes=[
                Attribute(name="name", type=AttributeType.STRING, unique=True, required=True),
                Attribute(name="age", type=AttributeType.NUMBER, unique=False, required=False),
                Attribute(name="email", type=AttributeType.STRING, unique=False, required=False),
            ]
        )
        
        company_entity = Entity(
            label="Company",
            attributes=[
                Attribute(name="name", type=AttributeType.STRING, unique=True, required=True),
                Attribute(name="founded_year", type=AttributeType.NUMBER, unique=False, required=False),
            ]
        )
        
        works_at_relation = Relation(
            label="WORKS_AT",
            source=person_entity,
            target=company_entity,
            attributes=[
                Attribute(name="since", type=AttributeType.NUMBER, unique=False, required=False),
            ]
        )
        
        self.ontology = Ontology(
            entities=[person_entity, company_entity],
            relations=[works_at_relation]
        )
        
        self.validator = ExtractionValidator(self.ontology, strict_mode=False)
        self.strict_validator = ExtractionValidator(self.ontology, strict_mode=True)
    
    def test_validate_entity_valid(self):
        """Test validation of valid entity."""
        entity = {
            "label": "Person",
            "attributes": {
                "name": "John Doe",
                "age": 30,
                "email": "john@example.com"
            }
        }
        
        is_valid, errors, quality_score = self.validator.validate_entity(entity)
        
        assert is_valid is True
        assert len(errors) == 0
        assert quality_score == 1.0
    
    def test_validate_entity_missing_label(self):
        """Test validation of entity missing label."""
        entity = {
            "attributes": {"name": "John Doe"}
        }
        
        is_valid, errors, quality_score = self.validator.validate_entity(entity)
        
        assert is_valid is False
        assert len(errors) > 0
        assert quality_score == 0.0
    
    def test_validate_entity_invalid_label(self):
        """Test validation of entity with invalid label."""
        entity = {
            "label": "InvalidEntity",
            "attributes": {"name": "John Doe"}
        }
        
        # Non-strict mode should allow with reduced quality
        is_valid, errors, quality_score = self.validator.validate_entity(entity)
        assert quality_score < 1.0
        
        # Strict mode should reject
        is_valid_strict, errors_strict, quality_score_strict = self.strict_validator.validate_entity(entity)
        assert is_valid_strict is False
    
    def test_validate_entity_missing_required_attribute(self):
        """Test validation of entity missing required attribute."""
        entity = {
            "label": "Person",
            "attributes": {
                "age": 30,  # Missing required 'name'
            }
        }
        
        is_valid, errors, quality_score = self.validator.validate_entity(entity)
        
        # Should have errors about missing required attribute
        assert any("required" in error.lower() or "name" in error.lower() for error in errors)
        assert quality_score < 1.0
    
    def test_validate_entity_missing_unique_attribute(self):
        """Test validation of entity missing unique attribute."""
        entity = {
            "label": "Person",
            "attributes": {
                "age": 30,  # Missing unique 'name'
            }
        }
        
        is_valid, errors, quality_score = self.validator.validate_entity(entity)
        
        # Should have errors about missing unique attribute
        assert any("unique" in error.lower() for error in errors)
        assert quality_score < 1.0
    
    def test_validate_entity_wrong_attribute_type(self):
        """Test validation of entity with wrong attribute type."""
        entity = {
            "label": "Person",
            "attributes": {
                "name": "John Doe",
                "age": "thirty",  # Should be number, not string
            }
        }
        
        is_valid, errors, quality_score = self.validator.validate_entity(entity)
        
        # Should have type error
        assert any("age" in error.lower() and "number" in error.lower() for error in errors)
        assert quality_score < 1.0
    
    def test_validate_relation_valid(self):
        """Test validation of valid relation."""
        relation = {
            "label": "WORKS_AT",
            "source": {
                "label": "Person",
                "attributes": {"name": "John Doe"}
            },
            "target": {
                "label": "Company",
                "attributes": {"name": "Acme Inc"}
            },
            "attributes": {
                "since": 2020
            }
        }
        
        is_valid, errors, quality_score = self.validator.validate_relation(relation)
        
        assert is_valid is True
        assert len(errors) == 0
        assert quality_score == 1.0
    
    def test_validate_relation_missing_fields(self):
        """Test validation of relation missing required fields."""
        relation = {
            "label": "WORKS_AT",
            "source": {"label": "Person"}
            # Missing target
        }
        
        is_valid, errors, quality_score = self.validator.validate_relation(relation)
        
        assert is_valid is False
        assert len(errors) > 0
        assert quality_score == 0.0
    
    def test_validate_relation_invalid_combination(self):
        """Test validation of relation with invalid entity combination."""
        relation = {
            "label": "WORKS_AT",
            "source": {
                "label": "Company",  # Wrong: should be Person
                "attributes": {"name": "Acme Inc"}
            },
            "target": {
                "label": "Person",  # Wrong: should be Company
                "attributes": {"name": "John Doe"}
            }
        }
        
        is_valid, errors, quality_score = self.validator.validate_relation(relation)
        
        # Should have error about invalid combination
        assert any("invalid relation" in error.lower() for error in errors)
        assert quality_score < 1.0
    
    def test_validate_extraction_complete(self):
        """Test validation of complete extraction."""
        data = {
            "entities": [
                {
                    "label": "Person",
                    "attributes": {"name": "John Doe", "age": 30}
                },
                {
                    "label": "Company",
                    "attributes": {"name": "Acme Inc"}
                },
                {
                    "label": "InvalidEntity",  # Invalid
                    "attributes": {"name": "Test"}
                }
            ],
            "relations": [
                {
                    "label": "WORKS_AT",
                    "source": {"label": "Person", "attributes": {"name": "John Doe"}},
                    "target": {"label": "Company", "attributes": {"name": "Acme Inc"}}
                },
                {
                    "label": "INVALID_RELATION",  # Invalid
                    "source": {"label": "Person"},
                    "target": {"label": "Company"}
                }
            ]
        }
        
        validated_data, report = self.validator.validate_extraction(data)
        
        # Check report statistics
        assert report["total_entities"] == 3
        assert report["valid_entities"] >= 2  # At least 2 valid entities
        assert report["total_relations"] == 2
        
        # Check validated data
        assert len(validated_data["entities"]) >= 2
        assert len(validated_data["relations"]) >= 1
        
        # Check quality scores
        assert 0.0 <= report["entity_quality_avg"] <= 1.0
        assert 0.0 <= report["relation_quality_avg"] <= 1.0
    
    def test_validate_extraction_empty(self):
        """Test validation of empty extraction."""
        data = {
            "entities": [],
            "relations": []
        }
        
        validated_data, report = self.validator.validate_extraction(data)
        
        assert report["total_entities"] == 0
        assert report["valid_entities"] == 0
        assert report["total_relations"] == 0
        assert report["valid_relations"] == 0
    
    def test_validate_extraction_all_valid(self):
        """Test validation where all extractions are valid."""
        data = {
            "entities": [
                {
                    "label": "Person",
                    "attributes": {"name": "John Doe", "age": 30}
                },
                {
                    "label": "Company",
                    "attributes": {"name": "Acme Inc", "founded_year": 2000}
                }
            ],
            "relations": [
                {
                    "label": "WORKS_AT",
                    "source": {"label": "Person", "attributes": {"name": "John Doe"}},
                    "target": {"label": "Company", "attributes": {"name": "Acme Inc"}},
                    "attributes": {"since": 2020}
                }
            ]
        }
        
        validated_data, report = self.validator.validate_extraction(data)
        
        # All should be valid
        assert report["valid_entities"] == report["total_entities"]
        assert report["valid_relations"] == report["total_relations"]
        
        # Quality should be high
        assert report["entity_quality_avg"] >= 0.9
        assert report["relation_quality_avg"] >= 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
