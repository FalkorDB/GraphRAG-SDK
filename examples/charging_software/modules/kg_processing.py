from falkordb import Graph
from graphrag_sdk import Ontology

def build_knowledge_graph(redis_url: str, ontology: Ontology):
    """Build and populate knowledge graph"""
    graph = Graph(redis_url, "charging-demo")
    ontology.save_to_graph(graph)
    return graph

def query_graph(graph: Graph, cypher_query: str):
    """Execute Cypher query on knowledge graph"""
    return graph.query(cypher_query).result_set
