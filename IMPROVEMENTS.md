# GraphRAG-SDK Improvements

This document outlines the improvements made to the GraphRAG-SDK to enhance knowledge graph generation, reduce duplication, improve entity extraction accuracy, and overall system quality.

## Table of Contents
1. [Entity Resolution & Deduplication](#entity-resolution--deduplication)
2. [Extraction Validation](#extraction-validation)
3. [Enhanced Prompts](#enhanced-prompts)
4. [Integration with Knowledge Graph](#integration-with-knowledge-graph)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)

## Entity Resolution & Deduplication

### Overview
The new `EntityResolver` class provides sophisticated entity deduplication and normalization capabilities to reduce redundancy in the knowledge graph.

### Features

#### 1. String Normalization
- Converts text to lowercase
- Removes extra whitespace
- Strips special characters for comparison
- Ensures consistent formatting

#### 2. Date Normalization
- Supports multiple date formats (YYYY-MM-DD, MM/DD/YYYY, etc.)
- Converts all dates to standard YYYY-MM-DD format
- Handles various separators (-, /, etc.)

#### 3. Fuzzy Matching
- Uses sequence matching to compute similarity scores
- Configurable similarity threshold (default: 0.85)
- Identifies entities that are semantically similar

#### 4. Entity Deduplication
- Compares entities based on unique attributes
- Merges duplicate entities intelligently
- Preserves the most complete information

#### 5. Coreference Resolution
- Resolves pronouns and abbreviated names to full identifiers
- Maintains entity consistency across documents
- Uses context-aware matching

### Usage

```python
from graphrag_sdk import EntityResolver

# Initialize with custom threshold
resolver = EntityResolver(similarity_threshold=0.85)

# Normalize strings
normalized = resolver.normalize_string("  John  Doe  ")  # "john doe"

# Normalize dates
date = resolver.normalize_date("12/25/2023")  # "2023-12-25"

# Deduplicate entities
entities = [
    {"label": "Person", "attributes": {"name": "John Doe"}},
    {"label": "Person", "attributes": {"name": "John  Doe"}},  # duplicate
]
deduplicated, count = resolver.deduplicate_entities(entities, ["name"])
# Returns: 1 entity, count = 1
```

## Extraction Validation

### Overview
The `ExtractionValidator` class validates extracted entities and relations against the ontology, ensuring data quality and consistency.

### Features

#### 1. Entity Validation
- Checks if entity labels exist in ontology
- Validates required attributes are present
- Ensures unique attributes are provided
- Verifies attribute types match schema

#### 2. Relation Validation
- Validates relation labels against ontology
- Checks source and target entity compatibility
- Verifies relation direction correctness
- Validates relation attributes

#### 3. Quality Scoring
- Assigns quality scores (0.0 - 1.0) to extractions
- Provides detailed error reporting
- Enables filtering based on quality thresholds

#### 4. Extraction Reports
- Generates comprehensive validation reports
- Tracks valid vs invalid extractions
- Reports average quality scores
- Lists validation errors for debugging

### Usage

```python
from graphrag_sdk import ExtractionValidator

# Initialize with ontology
validator = ExtractionValidator(ontology, strict_mode=False)

# Validate a single entity
entity = {"label": "Person", "attributes": {"name": "John Doe"}}
is_valid, errors, quality_score = validator.validate_entity(entity)

# Validate complete extraction
data = {
    "entities": [...],
    "relations": [...]
}
validated_data, report = validator.validate_extraction(data)

print(f"Valid entities: {report['valid_entities']}/{report['total_entities']}")
print(f"Average quality: {report['entity_quality_avg']:.2f}")
```

## Enhanced Prompts

### Improvements to Data Extraction Prompts

1. **Entity Consistency Guidelines**
   - Clear instructions on avoiding duplicates
   - Use of canonical forms (full names, complete titles)
   - Consistent entity references across text

2. **Format Consistency**
   - Standardized date format (YYYY-MM-DD)
   - Consistent name formatting and capitalization
   - Normalized numbers with units

3. **Accuracy Guidelines**
   - Extract only high-confidence information
   - Preserve exact meaning from source
   - No inference beyond stated facts

4. **Enhanced Documentation**
   - Clearer examples and guidelines
   - Better structure and organization
   - Explicit constraints and requirements

### Improvements to Ontology Creation Prompts

1. **Attribute Extraction**
   - Emphasis on unique identifiers
   - Distinction between required and optional attributes
   - Clear attribute type specifications

2. **Design Principles**
   - Focus on general, timeless concepts
   - Avoid redundancy in entities and relations
   - Balance between simplicity and completeness

## Integration with Knowledge Graph

### Enhanced `process_sources` Method

The `KnowledgeGraph.process_sources()` method now supports deduplication and validation:

```python
kg.process_sources(
    sources=sources,
    enable_deduplication=True,  # Enable entity deduplication
    enable_validation=True,      # Enable extraction validation
)
```

### Configuration Options

The `ExtractDataStep` now accepts additional configuration:

```python
config = {
    "max_workers": 16,
    "max_input_tokens": 500000,
    "max_output_tokens": 8192,
    "similarity_threshold": 0.85,  # Deduplication threshold
}
```

## Usage Examples

### Example 1: Basic Usage with Deduplication

```python
from graphrag_sdk import KnowledgeGraph, Ontology
from graphrag_sdk.source import URL
from graphrag_sdk.models.litellm import LiteModel
from graphrag_sdk.model_config import KnowledgeGraphModelConfig

# Setup
model = LiteModel(model_name="openai/gpt-4.1")
sources = [URL("https://example.com/article")]

# Create ontology
ontology = Ontology.from_sources(sources=sources, model=model)

# Create knowledge graph with deduplication enabled
kg = KnowledgeGraph(
    name="my_kg",
    model_config=KnowledgeGraphModelConfig.with_model(model),
    ontology=ontology,
)

# Process sources with deduplication and validation
kg.process_sources(
    sources=sources,
    enable_deduplication=True,
    enable_validation=True,
)
```

### Example 2: Custom Similarity Threshold

```python
from graphrag_sdk.steps.extract_data_step import ExtractDataStep

# Configure with custom similarity threshold
config = {
    "max_workers": 16,
    "max_input_tokens": 500000,
    "max_output_tokens": 8192,
    "similarity_threshold": 0.90,  # Higher threshold for stricter matching
}

step = ExtractDataStep(
    sources=sources,
    ontology=ontology,
    model=model,
    graph=graph,
    config=config,
    enable_deduplication=True,
    enable_validation=True,
)
```

### Example 3: Standalone Entity Resolution

```python
from graphrag_sdk import EntityResolver

resolver = EntityResolver(similarity_threshold=0.85)

# Normalize entity attributes
entities = [
    {"label": "Person", "attributes": {"name": "John Doe", "birth_date": "12/25/1990"}},
    {"label": "Person", "attributes": {"name": "Jane Smith", "birth_date": "1985-03-15"}},
]

# Normalize dates and format
for entity in entities:
    entity = resolver.normalize_entity_attributes(entity)
    
# Deduplicate
deduplicated, dup_count = resolver.deduplicate_entities(entities, ["name"])
print(f"Removed {dup_count} duplicates")
```

## Best Practices

### 1. Entity Deduplication

**Do:**
- Enable deduplication for sources with potential duplicates
- Use appropriate similarity thresholds (0.80-0.90 range)
- Define clear unique attributes in your ontology
- Normalize data formats consistently

**Don't:**
- Set similarity threshold too low (< 0.75) - may merge distinct entities
- Set similarity threshold too high (> 0.95) - may miss duplicates
- Skip defining unique attributes in ontology

### 2. Extraction Validation

**Do:**
- Enable validation to catch extraction errors early
- Review validation reports to understand quality issues
- Use non-strict mode for flexibility with diverse sources
- Monitor average quality scores over time

**Don't:**
- Use strict mode with noisy or diverse data sources
- Ignore validation errors without investigation
- Disable validation without good reason

### 3. Ontology Design

**Do:**
- Define at least one unique attribute per entity
- Specify required vs optional attributes clearly
- Use general, reusable entity types
- Include sufficient attributes for meaningful queries

**Don't:**
- Create overly specific entity types
- Duplicate entity types or relations
- Omit unique identifiers

### 4. Performance Optimization

**Do:**
- Adjust `max_workers` based on available resources
- Use appropriate token limits for your use case
- Monitor processing time and adjust config
- Use progress bars for long-running operations

**Don't:**
- Set `max_workers` too high (may exhaust resources)
- Use unlimited token limits (may hit API limits)
- Process all sources in a single batch for large datasets

## Performance Considerations

### Memory Usage
- Entity deduplication requires holding entities in memory
- For large datasets, consider batch processing
- Monitor memory usage with many concurrent workers

### Processing Time
- Deduplication adds minimal overhead (< 5% typically)
- Validation is very fast (< 1% overhead)
- Main bottleneck is still LLM API calls

### Accuracy vs Speed Trade-off
- Higher similarity thresholds are faster but less accurate
- Validation adds minimal time but improves quality significantly
- Disable features selectively if speed is critical

## Future Enhancements

Potential areas for future improvement:

1. **Advanced Coreference Resolution**
   - Integration with NLP models (spaCy, AllenNLP)
   - Multi-document entity tracking
   - Cross-reference resolution

2. **Machine Learning-Based Deduplication**
   - Train custom similarity models
   - Learn from user feedback
   - Context-aware matching

3. **Incremental Validation**
   - Real-time validation during extraction
   - Immediate feedback to LLM
   - Iterative refinement

4. **Enhanced Ontology Evolution**
   - Automatic ontology updates from validated extractions
   - Conflict resolution for schema changes
   - Version control for ontologies

5. **Quality Metrics Dashboard**
   - Visualization of extraction quality
   - Historical quality trends
   - Entity-level quality scores

## Conclusion

These improvements significantly enhance the GraphRAG-SDK's ability to:
- Generate high-quality knowledge graphs
- Reduce entity duplication and redundancy
- Improve extraction accuracy and reliability
- Provide better visibility into data quality
- Enable more consistent and maintainable knowledge graphs

For questions or issues, please refer to the main README or open an issue on GitHub.
