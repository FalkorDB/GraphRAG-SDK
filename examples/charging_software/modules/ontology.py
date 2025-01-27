from graphrag_sdk import Ontology
from graphrag_sdk.source import AbstractSource
from graphrag_sdk.models import GenerativeModel
import json

def generate_ontology(sources: list[AbstractSource], model: GenerativeModel):
    """Generate ontology from documents"""
    ontology = Ontology.from_sources(
        sources=sources,
        model=model,
        boundaries="EV charging software domain"
    )
    return ontology

def merge_ontologies(main_ontology: Ontology, additional_path: str):
    """Merge with existing ontology"""
    with open(additional_path) as f:
        existing = Ontology.from_json(json.load(f))
    return main_ontology.merge_with(existing)
