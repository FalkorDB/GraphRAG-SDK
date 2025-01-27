"""
GraphRAG SDK Quickstart with DeepSeek via LiteLLM
"""

from dotenv import load_dotenv
from graphrag_sdk import KnowledgeGraph, Ontology
from graphrag_sdk.models.litellm import LiteModel
from graphrag_sdk.model_config import KnowledgeGraphModelConfig
from graphrag_sdk.source import URL

load_dotenv()

# Initialize DeepSeek model via LiteLLM
model = LiteModel(
    model_name="deepseek/channel",  # Verify exact model name format
    api_base="https://api.deepseek.com/v1"  # Verify endpoint URL
)

# Create ontology from sample source
sources = [URL("https://en.wikipedia.org/wiki/Your_Topic")]  # Start with a focused data source
ontology = Ontology.from_sources(
    sources=sources,
    model=model,
    extraction_params={"chunk_size": 1000}  # Adjust based on content
)

# Build Knowledge Graph
kg = KnowledgeGraph(
    name="deepseek-demo",
    model_config=KnowledgeGraphModelConfig.with_model(model),
    ontology=ontology,
    host="localhost",
    port=6379
)

kg.process_sources(sources)  # Monitor FalkorDB logs during processing

def query_kg(question: str):
    """Query interface for the knowledge graph"""
    chat = kg.chat_session()
    response = chat.send_message(question)
    return response

# Example usage
if __name__ == "__main__":
    print(query_kg("Explain the key concepts in this knowledge graph"))
