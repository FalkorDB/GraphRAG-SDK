import re
import logging
from graphrag_sdk import Ontology
from typing import Union, Optional
from fix_busted_json import repair_json


logger = logging.getLogger(__name__)

def extract_json(text: Union[str, dict], skip_repair: Optional[bool] = False) -> str:
    """
    Extracts JSON from a string or dictionary, optionally skipping JSON repair.
    
    Args:
        text (Union[str, dict]): The input text or dictionary.
        skip_repair (Optional[bool]): Flag to skip JSON repair. Defaults to False.
        
    Returns:
        str: The extracted JSON as a string.
    """
    if not isinstance(text, str):
        text = str(text)
    regex = r"(?:```)?(?:json)?([^`]*)(?:\\n)?(?:```)?"
    matches = re.findall(regex, text, re.DOTALL)

    try:
        return repair_json("".join(matches)) if not skip_repair else "".join(matches)
    except Exception as e:
        logger.error(f"Failed to repair JSON: {e} - {text}")
        return "".join(matches)


def map_dict_to_cypher_properties(d: dict) -> str:
    """
    Maps a dictionary to Cypher query properties.
    
    Args:
        d (dict): The dictionary to map.
        
    Returns:
        str: A Cypher-formatted string of properties.
    """
    cypher = "{"
    if isinstance(d, list):
        if len(d) == 0:
            return "{}"
        for i, item in enumerate(d):
            cypher += f"{i}: {item}, "
        cypher = (cypher[:-2] if len(cypher) > 1 else cypher) + "}"
        return cypher
    for key, value in d.items():
        # Check value type
        if isinstance(value, str):
            # Find unescaped quotes
            reg = r"((?<!\\)(\"))|((?<!\\)(\'))"
            search = re.search(reg, value)
            if search:
                i = 0
                for match in re.finditer(reg, value):
                    value = (
                        value[: match.start() + i] + "\\" + value[match.start() + i :]
                    )
                    i += 1
            value = f'"{value}"' if f"{value}" != "None" else '""'
        else:
            value = str(value) if f"{value}" != "None" else '""'
        cypher += f"{key}: {value}, "
    cypher = (cypher[:-2] if len(cypher) > 1 else cypher) + "}"
    return cypher


def stringify_falkordb_response(response: Union[list, str]) -> str:
    """
    Converts FalkorDB response to a string.
    
    Args:
        response (Union[list, str]): The response to stringify.
        
    Returns:
        str: The stringified response.
    """
    if not isinstance(response, list) or len(response) == 0:
        data = str(response).strip()
    elif not isinstance(response[0], list):
        data = str(response).strip()
    else:
        for l, _ in enumerate(response):
            if not isinstance(response[l], list):
                response[l] = str(response[l])
            else:
                for i, __ in enumerate(response[l]):
                    response[l][i] = str(response[l][i])
        data = str(response).strip()

    return data


def extract_cypher(text: str) -> str:
    """
    Extracts Cypher query from a text block.
    
    Args:
        text (str): The text containing a Cypher query.
        
    Returns:
        str: The extracted Cypher query.
    """

    if not text.startswith("```"):
        return text

    regex = r"```(?:cypher)?(.*?)```"
    matches = re.findall(regex, text, re.DOTALL)

    return "".join(matches)


def validate_cypher(
    cypher: str, ontology: Ontology
) -> Optional[list[str]]:
    """
    Validates a Cypher query against the ontology.
    
    Args:
        cypher (str): The Cypher query.
        ontology (Ontology): The ontology to validate against.
        
    Returns:
        Optional[list[str]]: A list of validation errors, or None if valid.
    """
    try:
        if not cypher or len(cypher) == 0:
            return ["Cypher statement is empty"]

        errors = []

        # Check if entities exist in ontology
        errors.extend(validate_cypher_entities_exist(cypher, ontology))

        # Check if relations exist in ontology
        errors.extend(validate_cypher_relations_exist(cypher, ontology))

        # Check if relation directions are correct
        errors.extend(validate_cypher_relation_directions(cypher, ontology))

        if len(errors) > 0:
            return errors

        return None
    except Exception as e:
        print(f"Failed to verify cypher labels: {e}")
        return None


def validate_cypher_entities_exist(cypher: str, ontology: Ontology) -> list[str]:
    """
    Validates whether entities in the Cypher query exist in the ontology.
    
    Args:
        cypher (str): The Cypher query.
        ontology (Ontology): The ontology to validate against.
        
    Returns:
        list[str]: A list of errors if entities are not found.
    """
    # Check if entities exist in ontology
    not_found_entity_labels = []
    entity_labels = re.findall(r"\(:(.*?)\)", cypher)
    for label in entity_labels:
        label = label.split(":")[1] if ":" in label else label
        label = label.split("{")[0].strip() if "{" in label else label
        if label not in [entity.label for entity in ontology.entities]:
            not_found_entity_labels.append(label)

    return [
        f"Entity {label} not found in ontology" for label in not_found_entity_labels
    ]


def validate_cypher_relations_exist(cypher: str, ontology: Ontology) -> list[str]:
    """
    Validates whether relations in the Cypher query exist in the ontology.
    
    Args:
        cypher (str): The Cypher query.
        ontology (Ontology): The ontology to validate against.
        
    Returns:
        list[str]: A list of errors if relations are not found.
    """
    # Check if relations exist in ontology
    not_found_relation_labels = []
    relation_labels = re.findall(r"\[:(.*?)\]", cypher)
    for relation in relation_labels:
        for label in relation.split("|"):
            max_idx = min(
                    label.index("*") if "*" in label else len(label),
                    label.index("{") if "{" in label else len(label),
                    label.index("]") if "]" in label else len(label),
                    )
            label = label[:max_idx]
            if label not in [relation.label for relation in ontology.relations]:
                not_found_relation_labels.append(label)

    return [
        f"Relation {label} not found in ontology" for label in not_found_relation_labels
    ]


def validate_cypher_relation_directions(
    cypher: str, ontology: Ontology
) -> list[str]:
    """
    Validates relation directions in a Cypher query.
    
    Args:
        cypher (str): The Cypher query.
        ontology (Ontology): The ontology to validate against.
        
    Returns:
        list[str]: A list of errors if relation directions are incorrect.
    """
    errors = []
    
    # Pattern to match complete relationship patterns: (node)-[rel]->(node) or (node)<-[rel]-(node)
    # This handles both directions and various node/relationship syntaxes
    relationship_pattern = r'\(([^)]*)\)\s*(<?)--?\[\s*([^]]*)\s*\]\s*--?(>?)\s*\(([^)]*)\)'
    
    matches = re.finditer(relationship_pattern, cypher, re.IGNORECASE)
    
    for match in matches:
        try:
            source_node = match.group(1).strip()
            left_arrow = match.group(2)  # '<' if present
            relation_content = match.group(3).strip()
            right_arrow = match.group(4)  # '>' if present
            target_node = match.group(5).strip()
            
            # Skip if no relationship label found
            if not relation_content or relation_content == '':
                continue
                
            # Extract relationship label from content like "r:RELATIONSHIP_TYPE" or ":RELATIONSHIP_TYPE"
            relation_label_match = re.search(r':\s*([A-Za-z_][A-Za-z0-9_]*)', relation_content)
            if not relation_label_match:
                continue
                
            relation_label = relation_label_match.group(1).strip()
            
            # Determine direction: <- means reverse, -> means forward, -- means undirected
            is_directed_left = bool(left_arrow)  # <-
            is_directed_right = bool(right_arrow)  # ->
            
            # Skip undirected relationships (--) as they don't have direction constraints
            if not is_directed_left and not is_directed_right:
                continue
            
            # Extract node labels from source and target
            source_label = _extract_node_label(source_node)
            target_label = _extract_node_label(target_node)
            
            # Skip if we can't extract labels (e.g., variables without labels)
            if not source_label or not target_label:
                continue
            
            # If direction is left (<-), swap source and target
            if is_directed_left and not is_directed_right:
                source_label, target_label = target_label, source_label
            
            # Get ontology relations with this label
            ontology_relations = ontology.get_relations_with_label(relation_label)
            
            if len(ontology_relations) == 0:
                # This error is already handled by validate_cypher_relations_exist
                continue
            
            # Check if any ontology relation matches the direction
            found_relation = False
            for ontology_relation in ontology_relations:
                if (
                    ontology_relation.source.label == source_label
                    and ontology_relation.target.label == target_label
                ):
                    found_relation = True
                    break
            
            if not found_relation:
                errors.append(
                    f"Relation '{relation_label}' does not connect {source_label} to {target_label}. "
                    f"Make sure the relation direction is correct. "
                    f"Valid relations: {', '.join([str(r) for r in ontology_relations])}"
                )
                
        except Exception as e:
            # Skip problematic patterns rather than failing
            continue
    
    return errors

def _extract_node_label(node_content: str) -> str:
    """
    Extract node label from node content like 'n:Person' or ':Person' or 'person:Person'.
    Returns the label or empty string if not found.
    """
    if not node_content:
        return ""
    
    # Look for pattern like ":Label" in the node content
    label_match = re.search(r':\s*([A-Za-z_][A-Za-z0-9_]*)', node_content)
    if label_match:
        return label_match.group(1).strip()
    
    return ""
