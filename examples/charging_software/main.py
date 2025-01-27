import logging
import os
import json
from modules.ontology import generate_ontology
from modules.kg_processing import build_knowledge_graph
from graphrag_sdk.models.litellm import LiteModel
from graphrag_sdk.model_config import KnowledgeGraphModelConfig
from graphrag_sdk.source import Source
from graphrag_sdk import KnowledgeGraph, Ontology

logging.basicConfig(level=logging.INFO)

MODEL_TYPE = "litellm"  # Switch to "litellm" for DeepSeek

if __name__ == "__main__":
    # Source configuration: Data folder.
    src_files = "examples/charging_software/data"
    sources = []

    # For each file in the source directory and its subdirectories, create a new Source object.
    for root, dirs, files in os.walk(src_files):
        for file in files:
            sources.append(Source(os.path.join(root, file)))
        # Model configuration
    if MODEL_TYPE != "litellm":
        logging.info("Specify LLM to be used.")
    elif MODEL_TYPE == "litellm":
        model = LiteModel(model_name="deepseek/deepseek-chat")
    

    # # Generate ontology
    generate_ontology(sources, model)

    with open("examples/charging_software/ontologies/ontology.json", "r") as f:
        ontology = Ontology.from_json(json.loads(f.read()))
    
    # Build knowledge graph with unified config
    kg = KnowledgeGraph(
        name="charging-demo",
        model_config=KnowledgeGraphModelConfig.with_model(model),
        ontology=ontology,
        host="localhost",
        port=6379
    )
    
    kg.process_sources(sources)

    # Example usage
    graph = build_knowledge_graph("redis://localhost:6379", ontology)

    # # Add chat interface from quickstart
    # def query_kg(question: str):
    #     chat = kg.chat_session()
    #     return chat.send_message(question)
    
    # print(query_kg("Explain key EV charging concepts in this knowledge base"))
