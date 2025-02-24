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

folder = "otelzap"  # "everest-core"
path_in = "examples/charging_software/03_data_in/code_repos"
# path_in = "examples/charging_software/03_data_in/code_repos"

MODEL_TYPE = "gemini"  # Switch to "ollama", "litellm" 

if __name__ == "__main__":
    ###################### SELECT LLM MODEL#### ########################
    if MODEL_TYPE == "litellm":
        model = LiteModel(model_name="deepseek/deepseek-chat")
        # model = LiteModel(model_name="deepseek/deepseek-reasoner")
    elif MODEL_TYPE == "groq":
        model = LiteModel(model_name="groq/deepseek-r1-distill-llama-70b")
    elif MODEL_TYPE == "openrouter":
        model = LiteModel(model_name="openrouter/deepseek/deepseek-r1:free")
    elif MODEL_TYPE == "ollama":
        model = OllamaGenerativeModel(model_name="deepseek-r1:14b")
    elif MODEL_TYPE == "gemini":
        model = GeminiGenerativeModel(model_name="models/gemini-2.0-flash-exp")
    else:
        logging.info("Specify LLM to be used.")
    
    ################### CREATE ONTOLOGIES FROM SOURCES ########################
    # Source configuration: Data folder.
    src_files = path_in+"/"+folder
    sources = []

    # For each file in the source directory and its subdirectories, create a new Source object.
    for root, dirs, files in os.walk(src_files):
        for file in files:
            sources.append(Source(os.path.join(root, file)))

    # Generate ontology
    f_name = "ontology_"+folder+".json"

    generate_ontology(sources, model, f_name)

    ######################### MERGE JSON ONTOLOGIES ###########################

    # ontologies_dir = "examples"+"/"+"charging_software"+"/"+"04_ontologies"
    # target_dir = "examples"+"/"+"charging_software"+"/"+"05_merged_ontologies"

    # merged = merge_ontology_directory(ontologies_dir)

    # # Save merged ontology
    # with open(os.path.join(target_dir, "merged_ontology.json"), "w") as f:
    #     json.dump(merged.to_json(), f, indent=2)

    ########### CREATE KNOWLEDGE GRAPH FROM (MERGED) ONTOLOGIES ###############

    # ontologies_dir = "examples"+"/"+"charging_software"+"/"+"04_ontologies"
    # name_onto = "ontology_"+folder
    # fname_onto = name_onto+".json"

    # with open(os.path.join(ontologies_dir, fname_onto), "r") as f:
    #     ontology = Ontology.from_json(json.loads(f.read()))
    
    # Build knowledge graph with unified config
    kg = KnowledgeGraph(
        name=folder,
        model_config=KnowledgeGraphModelConfig.with_model(model),
        ontology=ontology,
        host="localhost",
        port=6379
    )
    
    # # kg.process_sources(sources)  # Only if directly from sources to kg.

    ########### INTERACT WITH KNOWLEDGE GRAPH, SAVE IT AS .rdb file ###########

    # Add chat interface from quickstart
    def query_kg(question: str):
        chat = kg.chat_session()
        return chat.send_message(question)
    
    print(query_kg("Explain key EV charging concepts in this knowledge base"))

    logging.info("This is the end.")
