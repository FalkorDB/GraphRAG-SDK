import logging
import os
import json
from modules.ontology import generate_ontology
from modules.kg_processing import build_knowledge_graph
from graphrag_sdk.models.litellm import LiteModel
from graphrag_sdk.models.ollama import OllamaGenerativeModel
from graphrag_sdk.models.gemini import GeminiGenerativeModel
from graphrag_sdk.model_config import KnowledgeGraphModelConfig
from graphrag_sdk.source import Source
from graphrag_sdk import KnowledgeGraph, Ontology

logging.basicConfig(level=logging.INFO)

folder = "shellrecharge"  # "everest-core"
path_in = "examples/charging_software/03_data_in/code_repos"
# path_in = "examples/charging_software/03_data_in/code_repos"

MODEL_TYPE = "gemini"  # Switch to "ollama", "litellm" 


def merge_ontology_directory(directory: str) -> Ontology:
    """Merge all ontology JSON files in a directory using the SDK's merge_with method."""
    merged_ontology = Ontology()

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                logging.info(f"Merging {filename}")

            with open(filepath, "r") as f:
                try:
                    current_ontology = Ontology.from_json(json.load(f))
                    merged_ontology.merge_with(current_ontology)
                except Exception as e:
                    logging.error(f"Failed to merge {filename}: {str(e)}")

    # # Post-merge cleanup and validation
    # merged_ontology.discard_entities_without_relations()
    # merged_ontology.discard_relations_without_entities()

    if not merged_ontology.validate_entities():
        logging.warning("Merged ontology contains entities without unique attributes")

    return merged_ontology


if __name__ == "__main__":
    # Source configuration: Data folder.
    src_files = path_in+"/"+folder
    sources = []

    # For each file in the source directory and its subdirectories, create a new Source object.
    for root, dirs, files in os.walk(src_files):
        for file in files:
            sources.append(Source(os.path.join(root, file)))
        # Model configuration
    if MODEL_TYPE == "litellm":
        model = LiteModel(model_name="deepseek/deepseek-chat")
        # model = LiteModel(model_name="deepseek/deepseek-reasoner")
    elif MODEL_TYPE == "ollama":
        model = OllamaGenerativeModel(model_name="deepseek-r1:14b")
        # "deepseek-r1:14b"
    elif MODEL_TYPE == "gemini":
        model = GeminiGenerativeModel(model_name="models/gemini-2.0-flash-exp")
    else:
        logging.info("Specify LLM to be used.")
    
    # # Generate ontology
    f_name = "ontology_"+folder+".json"

    generate_ontology(sources, model, f_name)

    # ontologies_dir = "examples"+"/"+"charging_software"+"/"+"05_ontologies"
    # target_dir = "examples"+"/"+"charging_software"

    # merged = merge_ontology_directory(ontologies_dir)

    # # Save merged ontology
    # with open(os.path.join(target_dir, "merged_ontology.json"), "w") as f:
    #     json.dump(merged.to_json(), f, indent=2)

    # with open(os.path.join(target_dir, "merged_ontology.json"), "r") as f:
    #     ontology = Ontology.from_json(json.loads(f.read()))
    
    # # Build knowledge graph with unified config
    # kg = KnowledgeGraph(
    #     name="charging-demo",
    #     model_config=KnowledgeGraphModelConfig.with_model(model),
    #     ontology=ontology,
    #     host="localhost",
    #     port=6379
    # )
    
    # kg.process_sources(sources)

    # # Example usage
    # graph = build_knowledge_graph("redis://localhost:6379", ontology)

    # # Add chat interface from quickstart
    # def query_kg(question: str):
    #     chat = kg.chat_session()
    #     return chat.send_message(question)
    
    # print(query_kg("Explain key EV charging concepts in this knowledge base"))
