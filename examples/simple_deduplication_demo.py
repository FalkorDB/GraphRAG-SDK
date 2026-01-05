"""
Simple standalone demo of entity deduplication improvements.
This script can run without installing the full graphrag_sdk package.
"""

import re
from difflib import SequenceMatcher
from typing import Optional, List, Dict, Tuple


class SimpleEntityResolver:
    """Simplified entity resolver for demonstration."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
    
    def normalize_string(self, text: str) -> str:
        """Normalize a string for comparison."""
        if not text or not isinstance(text, str):
            return ""
        text = text.lower()
        text = " ".join(text.split())
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize date to YYYY-MM-DD format."""
        if not date_str or not isinstance(date_str, str):
            return None
        
        patterns = [
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', lambda m: (m.group(1), m.group(2), m.group(3))),
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: (m.group(3), m.group(1), m.group(2))),
        ]
        
        for pattern, extract in patterns:
            match = re.match(pattern, date_str)
            if match:
                try:
                    year, month, day = extract(match)
                    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
                except (ValueError, AttributeError):
                    continue
        return None
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity score between two strings."""
        norm1 = self.normalize_string(text1)
        norm2 = self.normalize_string(text2)
        if not norm1 or not norm2:
            return 0.0
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def are_entities_similar(
        self, 
        entity1: Dict, 
        entity2: Dict,
        unique_attributes: List[str]
    ) -> bool:
        """Check if two entities are similar."""
        if entity1.get("label") != entity2.get("label"):
            return False
        
        attr1 = entity1.get("attributes", {})
        attr2 = entity2.get("attributes", {})
        
        similarities = []
        for attr_name in unique_attributes:
            val1 = str(attr1.get(attr_name, ""))
            val2 = str(attr2.get(attr_name, ""))
            
            if not val1 or not val2:
                continue
            
            similarity = self.compute_similarity(val1, val2)
            similarities.append(similarity)
        
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            return avg_similarity >= self.similarity_threshold
        
        return False
    
    def merge_entity_attributes(self, entity1: Dict, entity2: Dict) -> Dict:
        """Merge attributes from two entities."""
        merged = {
            "label": entity1.get("label"),
            "attributes": {}
        }
        
        attr1 = entity1.get("attributes", {})
        attr2 = entity2.get("attributes", {})
        
        all_keys = set(attr1.keys()) | set(attr2.keys())
        
        for key in all_keys:
            val1 = attr1.get(key, "")
            val2 = attr2.get(key, "")
            
            if val1 and not val2:
                merged["attributes"][key] = val1
            elif val2 and not val1:
                merged["attributes"][key] = val2
            elif val1 and val2:
                merged["attributes"][key] = val1 if len(str(val1)) >= len(str(val2)) else val2
        
        return merged
    
    def deduplicate_entities(
        self, 
        entities: List[Dict], 
        unique_attributes: List[str]
    ) -> Tuple[List[Dict], int]:
        """Deduplicate a list of entities."""
        if not entities:
            return [], 0
        
        deduplicated = []
        duplicate_count = 0
        
        for entity in entities:
            found_duplicate = False
            
            for i, existing in enumerate(deduplicated):
                if self.are_entities_similar(entity, existing, unique_attributes):
                    deduplicated[i] = self.merge_entity_attributes(existing, entity)
                    found_duplicate = True
                    duplicate_count += 1
                    break
            
            if not found_duplicate:
                deduplicated.append(entity)
        
        return deduplicated, duplicate_count
    
    def normalize_entity_attributes(self, entity: Dict) -> Dict:
        """Normalize entity attributes."""
        if "attributes" not in entity:
            return entity
        
        normalized_attrs = {}
        
        for key, value in entity["attributes"].items():
            if not value:
                continue
            
            key_lower = key.lower()
            
            if "date" in key_lower or "time" in key_lower:
                normalized = self.normalize_date(str(value))
                normalized_attrs[key] = normalized if normalized else value
            elif "name" in key_lower or "title" in key_lower:
                normalized_attrs[key] = " ".join(str(value).split())
            elif isinstance(value, str):
                normalized_attrs[key] = " ".join(value.split())
            else:
                normalized_attrs[key] = value
        
        entity["attributes"] = normalized_attrs
        return entity


def demo():
    """Run demonstration."""
    print("=" * 70)
    print(" GraphRAG-SDK Entity Deduplication Demo")
    print("=" * 70)
    
    resolver = SimpleEntityResolver(similarity_threshold=0.85)
    
    # Sample entities with duplicates
    entities = [
        {
            "label": "Person",
            "attributes": {
                "name": "John Doe",
                "birth_date": "12/25/1990",
                "email": "john@example.com"
            }
        },
        {
            "label": "Person",
            "attributes": {
                "name": "John  Doe",  # Extra spaces - duplicate!
                "birth_date": "1990-12-25",
                "email": "john.doe@example.com"
            }
        },
        {
            "label": "Person",
            "attributes": {
                "name": "Jane Smith",
                "birth_date": "1/15/1985",
            }
        },
        {
            "label": "Person",
            "attributes": {
                "name": "Jane   Smith",  # Extra spaces - duplicate!
                "birth_date": "01/15/1985",
            }
        },
        {
            "label": "Organization",
            "attributes": {
                "name": "Acme Corp",
                "founded": "2000"
            }
        },
    ]
    
    print(f"\n📊 Original entities: {len(entities)}")
    print("-" * 70)
    for i, entity in enumerate(entities, 1):
        attrs = entity['attributes']
        name = attrs.get('name', 'N/A')
        date = attrs.get('birth_date', attrs.get('founded', 'N/A'))
        print(f"  {i}. {entity['label']}: {name} (date: {date})")
    
    # Step 1: Normalize
    print(f"\n🔧 Step 1: Normalizing attributes...")
    print("-" * 70)
    normalized = [resolver.normalize_entity_attributes(e) for e in entities]
    for i, entity in enumerate(normalized, 1):
        attrs = entity['attributes']
        name = attrs.get('name', 'N/A')
        date = attrs.get('birth_date', attrs.get('founded', 'N/A'))
        print(f"  {i}. {entity['label']}: {name} (date: {date})")
    
    # Step 2: Deduplicate
    print(f"\n🔍 Step 2: Deduplicating entities...")
    print("-" * 70)
    deduplicated, dup_count = resolver.deduplicate_entities(normalized, ["name"])
    
    print(f"  ✓ Removed {dup_count} duplicate(s)")
    print(f"  ✓ Unique entities: {len(deduplicated)}")
    
    print(f"\n📈 Final entities: {len(deduplicated)}")
    print("-" * 70)
    for i, entity in enumerate(deduplicated, 1):
        attrs = entity['attributes']
        name = attrs.get('name', 'N/A')
        date = attrs.get('birth_date', attrs.get('founded', 'N/A'))
        email = attrs.get('email', 'N/A')
        print(f"  {i}. {entity['label']}: {name}")
        print(f"     Date: {date}, Email: {email}")
    
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    print(f"""
Before: {len(entities)} entities (with duplicates)
After:  {len(deduplicated)} unique entities
Improvement: {dup_count} duplicates removed ({dup_count/len(entities)*100:.1f}% reduction)

✅ Entity deduplication successfully demonstrated!
✅ Date normalization: Multiple formats → YYYY-MM-DD
✅ Name normalization: Whitespace and formatting standardized
✅ Fuzzy matching: Similar entities identified and merged
""")


if __name__ == "__main__":
    demo()
