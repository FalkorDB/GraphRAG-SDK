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
    relations = list(re.finditer(r"\[.*?\]", cypher))
    i = 0
    for relation in relations:
        try:
            relation_label = (
                re.search(r"(?:\[)(?:\w)*(?:\:)([^\d\{\]\*\.\:]+)", relation.group(0))
                .group(1)
                .strip()
            )
            prev_relation = relations[i - 1] if i > 0 else None
            next_relation = relations[i + 1] if i < len(relations) - 1 else None
            before = (
                cypher[prev_relation.end() : relation.start()]
                if prev_relation
                else cypher[: relation.start()]
            )
            if "," in before:
                before = before.split(",")[-1]
            rel_before = re.search(r"([^\)\],]+)", before[::-1]).group(0)[::-1]
            after = (
                cypher[relation.end() : next_relation.start()]
                if next_relation
                else cypher[relation.end() :]
            )
            rel_after = re.search(r"([^\(\[,]+)", after).group(0)
            entity_before = re.search(r"\(.+:(.*?)\)", before).group(0)
            entity_after = re.search(r"\(([^\),]+)(\)?)", after).group(0)
            if rel_before == "-" and rel_after == "->":
                source = entity_before
                target = entity_after
            elif rel_before == "<-" and rel_after == "-":
                source = entity_after
                target = entity_before
            else:
                continue

            source_label = re.search(r"(?:\:)([^\)\{]+)", source).group(1).strip()
            target_label = re.search(r"(?:\:)([^\)\{]+)", target).group(1).strip()

            ontology_relations = ontology.get_relations_with_label(relation_label)

            if len(ontology_relations) == 0:
                errors.append(f"Relation {relation_label} not found in ontology")

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
                    """
                    Relation {relation_label} does not connect {source_label} to {target_label}. Make sure the relation direction is correct. 
                    Valid relations: 
                    {valid_relations}
""".format(
                        relation_label=relation_label,
                        source_label=source_label,
                        target_label=target_label,
                        valid_relations="\n".join([str(e) for e in ontology_relations]),
                    )
                )

            i += 1
        except Exception:
            continue

    return errors
