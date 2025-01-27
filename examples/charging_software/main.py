import logging
from config import settings
from modules.ontology import generate_ontology, merge_ontologies
from modules.kg_processing import build_knowledge_graph, query_graph
from graphrag_sdk.models.azure_openai import AzureOpenAiGenerativeModel

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Initialize model
    model = AzureOpenAiGenerativeModel(
        api_key="your-key-here",
        model_name="gpt-4"
    )
    
    # Generate ontology
    ontology = generate_ontology([], model)  # Add document sources
    
    # Example usage
    graph = build_knowledge_graph("redis://localhost:6379", ontology)
    results = query_graph(graph, "MATCH (n) RETURN n LIMIT 10")
    print(f"Found {len(results)} nodes")
