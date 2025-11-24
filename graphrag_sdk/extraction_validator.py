"""
Extraction Validator Module

This module provides validation and quality scoring for extracted entities and relations
to improve the accuracy and reliability of the knowledge graph.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from graphrag_sdk.ontology import Ontology

logger = logging.getLogger(__name__)


class ExtractionValidator:
    """
    Validates extracted entities and relations against the ontology and quality criteria.
    """
    
    def __init__(self, ontology: Ontology, strict_mode: bool = False):
        """
        Initialize the ExtractionValidator.
        
        Args:
            ontology (Ontology): The ontology to validate against.
            strict_mode (bool): If True, reject extractions that don't perfectly match ontology.
                               If False, attempt to fix common issues. Default is False.
        """
        self.ontology = ontology
        self.strict_mode = strict_mode
        
    def validate_entity(self, entity: Dict[str, Any]) -> Tuple[bool, List[str], float]:
        """
        Validate an extracted entity against the ontology.
        
        Args:
            entity (Dict): Entity to validate with 'label' and 'attributes' keys.
            
        Returns:
            Tuple[bool, List[str], float]: 
                - is_valid: Whether the entity is valid
                - errors: List of validation errors
                - quality_score: Quality score from 0.0 to 1.0
        """
        errors = []
        quality_score = 1.0
        
        # Check if entity has required fields
        if "label" not in entity:
            errors.append("Entity missing 'label' field")
            return False, errors, 0.0
        
        if "attributes" not in entity:
            errors.append("Entity missing 'attributes' field")
            quality_score -= 0.2
        
        # Check if entity label exists in ontology
        ontology_entity = self.ontology.get_entity_with_label(entity["label"])
        if not ontology_entity:
            errors.append(f"Entity label '{entity['label']}' not found in ontology")
            if self.strict_mode:
                return False, errors, 0.0
            quality_score -= 0.3
        
        # Validate attributes if entity exists in ontology
        if ontology_entity and "attributes" in entity:
            attr_errors, attr_score = self._validate_entity_attributes(
                entity["attributes"], 
                ontology_entity.attributes
            )
            errors.extend(attr_errors)
            quality_score *= attr_score
        
        is_valid = len(errors) == 0 or (not self.strict_mode and quality_score > 0.3)
        return is_valid, errors, quality_score
    
    def _validate_entity_attributes(
        self, 
        attributes: Dict[str, Any], 
        ontology_attributes: List
    ) -> Tuple[List[str], float]:
        """
        Validate entity attributes against ontology schema.
        
        Args:
            attributes (Dict): Extracted attributes.
            ontology_attributes (List): Expected attributes from ontology.
            
        Returns:
            Tuple[List[str], float]: List of errors and quality score.
        """
        errors = []
        quality_score = 1.0
        
        # Create a map of ontology attributes
        ontology_attr_map = {attr.name: attr for attr in ontology_attributes}
        
        # Check for required attributes
        required_attrs = [attr for attr in ontology_attributes if attr.required]
        for attr in required_attrs:
            if attr.name not in attributes or not attributes[attr.name]:
                errors.append(f"Required attribute '{attr.name}' is missing or empty")
                quality_score -= 0.2
        
        # Check for unique attributes (at least one should be present)
        unique_attrs = [attr for attr in ontology_attributes if attr.unique]
        if unique_attrs:
            has_unique = any(
                attr.name in attributes and attributes[attr.name] 
                for attr in unique_attrs
            )
            if not has_unique:
                errors.append("Entity missing unique identifying attributes")
                quality_score -= 0.3
        
        # Validate attribute types
        for attr_name, attr_value in attributes.items():
            if attr_name in ontology_attr_map:
                expected_type = ontology_attr_map[attr_name].type
                type_valid, type_error = self._validate_attribute_type(
                    attr_name, attr_value, expected_type
                )
                if not type_valid:
                    errors.append(type_error)
                    quality_score -= 0.1
        
        # Penalize for having too few attributes
        if len(attributes) < len(ontology_attributes) * 0.5:
            quality_score -= 0.1
        
        return errors, max(0.0, quality_score)
    
    def _validate_attribute_type(
        self, 
        attr_name: str, 
        attr_value: any, 
        expected_type
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that an attribute value matches the expected type.
        
        Args:
            attr_name (str): Attribute name.
            attr_value (any): Attribute value.
            expected_type: Expected type from ontology.
            
        Returns:
            Tuple[bool, Optional[str]]: Is valid and optional error message.
        """
        if attr_value is None:
            return True, None  # Allow None values
        
        # Import AttributeType here to avoid circular imports
        from graphrag_sdk.attribute import AttributeType
        
        # Check type compatibility
        if expected_type == AttributeType.STRING:
            if not isinstance(attr_value, str):
                return False, f"Attribute '{attr_name}' should be a string, got {type(attr_value).__name__}"
        elif expected_type == AttributeType.NUMBER:
            if not isinstance(attr_value, (int, float)):
                return False, f"Attribute '{attr_name}' should be a number, got {type(attr_value).__name__}"
        elif expected_type == AttributeType.BOOLEAN:
            if not isinstance(attr_value, bool):
                return False, f"Attribute '{attr_name}' should be a boolean, got {type(attr_value).__name__}"
        
        return True, None
    
    def validate_relation(self, relation: Dict[str, Any]) -> Tuple[bool, List[str], float]:
        """
        Validate an extracted relation against the ontology.
        
        Args:
            relation (Dict): Relation to validate.
            
        Returns:
            Tuple[bool, List[str], float]: 
                - is_valid: Whether the relation is valid
                - errors: List of validation errors
                - quality_score: Quality score from 0.0 to 1.0
        """
        errors = []
        quality_score = 1.0
        
        # Check required fields
        if "label" not in relation:
            errors.append("Relation missing 'label' field")
            return False, errors, 0.0
        
        if "source" not in relation or "target" not in relation:
            errors.append("Relation missing 'source' or 'target' field")
            return False, errors, 0.0
        
        # Validate source and target entities
        source = relation.get("source", {})
        target = relation.get("target", {})
        
        if "label" not in source or "label" not in target:
            errors.append("Relation source or target missing 'label' field")
            quality_score -= 0.3
        
        # Check if relation exists in ontology
        ontology_relations = self.ontology.get_relations_with_label(relation["label"])
        if not ontology_relations:
            errors.append(f"Relation label '{relation['label']}' not found in ontology")
            if self.strict_mode:
                return False, errors, 0.0
            quality_score -= 0.3
        
        # Check if the specific source->target combination is valid
        if ontology_relations and "label" in source and "label" in target:
            valid_combination = False
            for ont_rel in ontology_relations:
                if (ont_rel.source.label == source["label"] and 
                    ont_rel.target.label == target["label"]):
                    valid_combination = True
                    break
            
            if not valid_combination:
                errors.append(
                    f"Invalid relation: {source['label']}-[{relation['label']}]->{target['label']}"
                )
                quality_score -= 0.4
        
        is_valid = len(errors) == 0 or (not self.strict_mode and quality_score > 0.3)
        return is_valid, errors, quality_score
    
    def validate_extraction(
        self, 
        data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Validate a complete extraction (entities and relations).
        
        Args:
            data (Dict): Extraction data with 'entities' and 'relations' keys.
            
        Returns:
            Tuple[Dict, Dict]: 
                - validated_data: Filtered data containing only valid extractions
                - validation_report: Report with statistics and issues
        """
        validated_data = {
            "entities": [],
            "relations": []
        }
        
        validation_report = {
            "total_entities": 0,
            "valid_entities": 0,
            "invalid_entities": 0,
            "total_relations": 0,
            "valid_relations": 0,
            "invalid_relations": 0,
            "entity_quality_avg": 0.0,
            "relation_quality_avg": 0.0,
            "errors": []
        }
        
        # Validate entities
        entity_quality_scores = []
        if "entities" in data:
            validation_report["total_entities"] = len(data["entities"])
            
            for entity in data["entities"]:
                is_valid, errors, quality_score = self.validate_entity(entity)
                entity_quality_scores.append(quality_score)
                
                if is_valid:
                    validated_data["entities"].append(entity)
                    validation_report["valid_entities"] += 1
                else:
                    validation_report["invalid_entities"] += 1
                    validation_report["errors"].extend([
                        f"Entity {entity.get('label', 'Unknown')}: {error}" 
                        for error in errors
                    ])
                    logger.warning(f"Invalid entity filtered: {entity.get('label')}, errors: {errors}")
        
        # Calculate average entity quality
        if entity_quality_scores:
            validation_report["entity_quality_avg"] = sum(entity_quality_scores) / len(entity_quality_scores)
        
        # Validate relations
        relation_quality_scores = []
        if "relations" in data:
            validation_report["total_relations"] = len(data["relations"])
            
            for relation in data["relations"]:
                is_valid, errors, quality_score = self.validate_relation(relation)
                relation_quality_scores.append(quality_score)
                
                if is_valid:
                    validated_data["relations"].append(relation)
                    validation_report["valid_relations"] += 1
                else:
                    validation_report["invalid_relations"] += 1
                    validation_report["errors"].extend([
                        f"Relation {relation.get('label', 'Unknown')}: {error}" 
                        for error in errors
                    ])
                    logger.warning(f"Invalid relation filtered: {relation.get('label')}, errors: {errors}")
        
        # Calculate average relation quality
        if relation_quality_scores:
            validation_report["relation_quality_avg"] = sum(relation_quality_scores) / len(relation_quality_scores)
        
        logger.info(
            f"Validation complete: {validation_report['valid_entities']}/{validation_report['total_entities']} "
            f"entities valid, {validation_report['valid_relations']}/{validation_report['total_relations']} "
            f"relations valid"
        )
        
        return validated_data, validation_report
