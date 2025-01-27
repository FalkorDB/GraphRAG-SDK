import logging
from config import settings
from modules.ontology import generate_ontology, merge_ontologies
from modules.kg_processing import build_knowledge_graph, query_graph
from graphrag_sdk.models.azure_openai import AzureOpenAiGenerativeModel
from graphrag_sdk.models.litellm import LiteModel
from graphrag_sdk.model_config import KnowledgeGraphModelConfig
from graphrag_sdk.source import URL
from graphrag_sdk import KnowledgeGraph

logging.basicConfig(level=logging.INFO)

MODEL_TYPE = "azure"  # Switch to "litellm" for DeepSeek

if __name__ == "__main__":
    # Model configuration
    if MODEL_TYPE == "azure":
        model = AzureOpenAiGenerativeModel(
            api_key="your-key-here",
            model_name="gpt-4"
        )
    elif MODEL_TYPE == "litellm":
        model = LiteModel(model_name="deepseek/deepseek-reasoner")
    
    # Source configuration
    sources = [URL("https://en.wikipedia.org/wiki/EV_charging")]  # More relevant source
    
    # Generate ontology
    ontology = generate_ontology(sources, model)
    
    # Build knowledge graph with unified config
    kg = KnowledgeGraph(
        name="charging-demo",
        model_config=KnowledgeGraphModelConfig.with_model(model),
        ontology=ontology,
        host="localhost",
        port=6379
    )
    
    # Add chat interface from quickstart
    def query_kg(question: str):
        chat = kg.chat_session()
        return chat.send_message(question)
    
    # Example usage
    graph = build_knowledge_graph("redis://localhost:6379", ontology)
    print(query_kg("Explain key EV charging concepts in this knowledge base"))
