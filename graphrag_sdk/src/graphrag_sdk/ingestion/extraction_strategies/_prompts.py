# GraphRAG SDK 2.0 — Extraction: Prompt Templates
# Benchmark-winning 2-step LLM prompts for entity extraction
# and relationship verification/extraction.

NER_PROMPT = (
    "You are an expert named entity recognition system.\n"
    "Extract all entities from the text below.\n\n"
    "## Entity Types\n"
    "Only extract entities of these types: {entity_types}\n\n"
    "## Text\n"
    "{text}\n\n"
    "## Instructions\n"
    "- Extract ALL named entities present in the text.\n"
    "- Entity names MUST be specific, named references — proper nouns, named places, "
    "titled works, specific concepts, or named objects.\n"
    "- Do NOT extract pronouns (he, she, they, it, him, her, his, them, who, whom, "
    "I, we, you, one).\n"
    "- Do NOT extract generic references (narrator, the narrator, author, reader, "
    "the man, the woman, people, person, someone, story, chapter, book, text).\n"
    "- If a pronoun refers to a named entity, use the named entity's actual name.\n"
    "- For each entity, provide:\n"
    "  - name: the exact text span as it appears in the text\n"
    "  - type: one of the entity types above\n"
    "  - description: a brief description\n"
    "  - confidence: a float 0-1 indicating how confident you are\n"
    "  - start: the character offset where the entity starts in the text\n"
    "  - end: the character offset where the entity ends in the text\n\n"
    "Return ONLY a JSON array of objects:\n"
    '[{{"name": "<entity_name>", "type": "<entity_type>", '
    '"description": "<brief description>", "confidence": 0.95, '
    '"start": 0, "end": 5}}]\n\n'
    "Return ONLY valid JSON, nothing else."
)

VERIFY_EXTRACT_RELS_PROMPT = (
    "You are an expert knowledge graph builder.\n"
    "Given the text and pre-extracted entities below, do two things:\n"
    "1. VERIFY the entities: remove any that are not actually in the text, "
    "fix any naming errors, and add any missed entities.\n"
    "2. EXTRACT all relationships between the verified entities.\n\n"
    "## Entity Types\n"
    "{entity_types}\n\n"
    "## Pre-extracted Entities\n"
    "{entities_json}\n\n"
    "## Text\n"
    "{text}\n\n"
    "## Instructions\n\n"
    "### Entities\n"
    "- For each verified entity provide a rich description that captures "
    "key attributes, roles, and context from the text. This description is "
    "used for search and retrieval — make it informative.\n"
    "- span_start: the character offset in the text where the entity name "
    "first appears.\n"
    "- span_end: the character offset where the entity name ends.\n\n"
    "### Relationships\n"
    "- Extract ALL factual connections stated or implied in the text.\n"
    "- source and target must be entity names from the verified entity list.\n"
    "- type: a descriptive relationship label in UPPER_SNAKE_CASE "
    "(e.g. WORKS_AT, MARRIED_TO, LOCATED_IN, CREATED_BY, PART_OF).\n"
    "- description: a concise sentence describing the relationship as a "
    "standalone fact. This is embedded for semantic search — it must be "
    "self-contained and understandable without the original text.\n"
    "- keywords: comma-separated terms that characterize this relationship "
    "(e.g. 'employment, career' or 'family, parentage'). Used for "
    "fulltext search.\n"
    "- weight: a float 0-1 indicating confidence (1.0 = explicitly stated, "
    "0.5 = implied, 0.2 = weak inference).\n"
    "- span_start: the character offset in the text where the evidence "
    "sentence for this relationship starts.\n"
    "- span_end: the character offset where the evidence sentence ends.\n\n"
    "Return ONLY a JSON object with two arrays:\n"
    '{{"entities": [{{"name": "...", "type": "...", "description": "...", '
    '"span_start": 0, "span_end": 5}}], '
    '"relationships": [{{"source": "...", "target": "...", "type": "...", '
    '"description": "...", "keywords": "...", "weight": 0.9, '
    '"span_start": 0, "span_end": 50}}]}}\n\n'
    "Return ONLY valid JSON, nothing else."
)
