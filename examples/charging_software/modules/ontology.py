from graphrag_sdk import Ontology
from graphrag_sdk.source import AbstractSource
from graphrag_sdk.models import GenerativeModel
import json
import os

boundaries = """
    Extract only the most relevant information about charging software.
    Focus on the protocol for communication.
    Avoid creating entities for details that can be expressed as attributes.
"""

def generate_ontology(sources: list[AbstractSource], model: GenerativeModel, filename):
    """Generate ontology from documents"""
    ontology = Ontology.from_sources(
        sources=sources,
        model=model,
        boundaries=boundaries
    )

    # Save the ontology to the disk as a json file.
    output_dir = "examples/charging_software/05_ontologies"  # todo: change directory
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as file:
        file.write(json.dumps(ontology.to_json(), indent=2))


def merge_ontologies(main_ontology: Ontology, additional_path: str):
    """Merge with existing ontology"""
    with open(additional_path) as f:
        existing = Ontology.from_json(json.load(f))
    return main_ontology.merge_with(existing)
