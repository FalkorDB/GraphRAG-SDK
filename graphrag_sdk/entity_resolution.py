"""
Entity Resolution Module

This module provides functionality for entity deduplication, normalization,
and resolution to improve knowledge graph quality and reduce redundancy.
"""

import re
import logging
from typing import Optional, Dict, List, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class EntityResolver:
    """
    Handles entity resolution, deduplication, and normalization.
    
    This class implements various strategies to identify and merge duplicate entities,
    normalize entity attributes, and maintain consistency across the knowledge graph.
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize the EntityResolver.
        
        Args:
            similarity_threshold (float): Threshold for considering entities as duplicates (0.0-1.0).
                                         Default is 0.85.
        """
        self.similarity_threshold = similarity_threshold
        
    def normalize_string(self, text: str) -> str:
        """
        Normalize a string by removing extra whitespace, converting to lowercase,
        and removing special characters for comparison.
        
        Args:
            text (str): The text to normalize.
            
        Returns:
            str: The normalized text.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove punctuation and special characters for comparison
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def normalize_date(self, date_str: str) -> Optional[str]:
        """
        Normalize date strings to a consistent format (YYYY-MM-DD).
        
        Args:
            date_str (str): The date string to normalize.
            
        Returns:
            Optional[str]: The normalized date in YYYY-MM-DD format, or None if parsing fails.
        """
        if not date_str or not isinstance(date_str, str):
            return None
        
        # Common date patterns
        patterns = [
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', r'\1-\2-\3'),  # YYYY-M-D or YYYY-MM-DD
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\3-\1-\2'),  # M/D/YYYY or MM/DD/YYYY
            (r'(\d{1,2})-(\d{1,2})-(\d{4})', r'\3-\1-\2'),  # D-M-YYYY or DD-MM-YYYY
        ]
        
        for pattern, replacement in patterns:
            match = re.match(pattern, date_str)
            if match:
                try:
                    year, month, day = match.groups() if 'YYYY' in pattern else (match.group(3), match.group(1), match.group(2))
                    # Ensure proper formatting with leading zeros
                    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
                except (ValueError, AttributeError):
                    continue
        
        return None
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity score between two text strings using sequence matching.
        
        Args:
            text1 (str): First text string.
            text2 (str): Second text string.
            
        Returns:
            float: Similarity score between 0.0 and 1.0.
        """
        norm1 = self.normalize_string(text1)
        norm2 = self.normalize_string(text2)
        
        if not norm1 or not norm2:
            return 0.0
        
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def are_entities_similar(
        self, 
        entity1: Dict[str, any], 
        entity2: Dict[str, any],
        unique_attributes: List[str]
    ) -> bool:
        """
        Determine if two entities are similar enough to be considered duplicates.
        
        Args:
            entity1 (Dict): First entity with label and attributes.
            entity2 (Dict): Second entity with label and attributes.
            unique_attributes (List[str]): List of attribute names that uniquely identify entities.
            
        Returns:
            bool: True if entities are similar enough to be considered duplicates.
        """
        # Entities must have the same label
        if entity1.get("label") != entity2.get("label"):
            return False
        
        attr1 = entity1.get("attributes", {})
        attr2 = entity2.get("attributes", {})
        
        # Check similarity for each unique attribute
        similarities = []
        for attr_name in unique_attributes:
            val1 = str(attr1.get(attr_name, ""))
            val2 = str(attr2.get(attr_name, ""))
            
            if not val1 or not val2:
                continue
                
            similarity = self.compute_similarity(val1, val2)
            similarities.append(similarity)
        
        # If we have similarity scores, check if average exceeds threshold
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            return avg_similarity >= self.similarity_threshold
        
        return False
    
    def merge_entity_attributes(
        self, 
        entity1: Dict[str, any], 
        entity2: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Merge attributes from two similar entities, preferring non-empty values
        and the most complete information.
        
        Args:
            entity1 (Dict): First entity.
            entity2 (Dict): Second entity to merge into the first.
            
        Returns:
            Dict: Merged entity with combined attributes.
        """
        merged = {
            "label": entity1.get("label"),
            "attributes": {}
        }
        
        attr1 = entity1.get("attributes", {})
        attr2 = entity2.get("attributes", {})
        
        # Merge attributes, preferring longer/more complete values
        all_keys = set(attr1.keys()) | set(attr2.keys())
        
        for key in all_keys:
            val1 = attr1.get(key, "")
            val2 = attr2.get(key, "")
            
            # Prefer non-empty values
            if val1 and not val2:
                merged["attributes"][key] = val1
            elif val2 and not val1:
                merged["attributes"][key] = val2
            elif val1 and val2:
                # Prefer longer value (more complete information)
                merged["attributes"][key] = val1 if len(str(val1)) >= len(str(val2)) else val2
        
        return merged
    
    def deduplicate_entities(
        self, 
        entities: List[Dict[str, any]], 
        unique_attributes: List[str]
    ) -> Tuple[List[Dict[str, any]], int]:
        """
        Deduplicate a list of entities based on similarity of unique attributes.
        
        Args:
            entities (List[Dict]): List of entities to deduplicate.
            unique_attributes (List[str]): List of attribute names used for uniqueness.
            
        Returns:
            Tuple[List[Dict], int]: Deduplicated list of entities and count of removed duplicates.
        """
        if not entities:
            return [], 0
        
        deduplicated = []
        duplicate_count = 0
        
        for entity in entities:
            # Check if this entity is similar to any already in deduplicated list
            found_duplicate = False
            
            for i, existing in enumerate(deduplicated):
                if self.are_entities_similar(entity, existing, unique_attributes):
                    # Merge the duplicate into the existing entity
                    deduplicated[i] = self.merge_entity_attributes(existing, entity)
                    found_duplicate = True
                    duplicate_count += 1
                    logger.debug(f"Merged duplicate entity: {entity.get('label')}")
                    break
            
            if not found_duplicate:
                deduplicated.append(entity)
        
        logger.info(f"Deduplicated {duplicate_count} entities from {len(entities)} total")
        return deduplicated, duplicate_count
    
    def normalize_entity_attributes(self, entity: Dict[str, any]) -> Dict[str, any]:
        """
        Normalize attributes of an entity for consistency.
        
        Args:
            entity (Dict): Entity with attributes to normalize.
            
        Returns:
            Dict: Entity with normalized attributes.
        """
        if "attributes" not in entity:
            return entity
        
        normalized_attrs = {}
        
        for key, value in entity["attributes"].items():
            if not value:
                continue
            
            # Normalize based on attribute name patterns
            key_lower = key.lower()
            
            if "date" in key_lower or "time" in key_lower:
                # Try to normalize dates
                normalized = self.normalize_date(str(value))
                normalized_attrs[key] = normalized if normalized else value
            elif "name" in key_lower or "title" in key_lower:
                # Normalize names and titles (proper spacing, consistent format)
                normalized_attrs[key] = " ".join(str(value).split())
            elif isinstance(value, str):
                # General string normalization (remove extra spaces)
                normalized_attrs[key] = " ".join(value.split())
            else:
                normalized_attrs[key] = value
        
        entity["attributes"] = normalized_attrs
        return entity
    
    def resolve_coreferences(self, text: str, entities: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Resolve coreferences in entity names using simple heuristics.
        This is a basic implementation that can be enhanced with NLP models.
        
        Args:
            text (str): Original text for context.
            entities (List[Dict]): List of extracted entities.
            
        Returns:
            List[Dict]: Entities with resolved coreferences.
        """
        # This is a placeholder for more sophisticated coreference resolution
        # In a production system, you would use NLP models like spaCy or AllenNLP
        
        # Simple heuristic: track full names and replace abbreviated versions
        full_names = {}
        
        for entity in entities:
            if entity.get("label") == "Person":
                name_attr = None
                for attr_key in ["name", "full_name", "person_name"]:
                    if attr_key in entity.get("attributes", {}):
                        name_attr = attr_key
                        break
                
                if name_attr:
                    name = entity["attributes"][name_attr]
                    # Store full names (assuming they have spaces)
                    if " " in name and len(name.split()) > 1:
                        # Map first name to full name
                        first_name = name.split()[0].lower()
                        if first_name not in full_names or len(name) > len(full_names[first_name]):
                            full_names[first_name] = name
        
        # Replace abbreviated names with full names
        for entity in entities:
            if entity.get("label") == "Person":
                name_attr = None
                for attr_key in ["name", "full_name", "person_name"]:
                    if attr_key in entity.get("attributes", {}):
                        name_attr = attr_key
                        break
                
                if name_attr:
                    name = entity["attributes"][name_attr]
                    # If it's a single name, try to expand it
                    if " " not in name:
                        name_lower = name.lower()
                        if name_lower in full_names:
                            entity["attributes"][name_attr] = full_names[name_lower]
                            logger.debug(f"Resolved coreference: {name} -> {full_names[name_lower]}")
        
        return entities
