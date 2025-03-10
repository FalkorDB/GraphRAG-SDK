import os
import pytest
import logging
import numpy as np
from dotenv import load_dotenv
from graphrag_sdk.entity import Entity
from deepeval.test_case import LLMTestCase
from graphrag_sdk.relation import Relation
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.models.litellm import LiteModel
from graphrag_sdk.source import Source_FromRawText
from graphrag_sdk.attribute import Attribute, AttributeType
# Custom metrics of DeepEval
from graphrag_sdk.graph_metrics import GraphContextualRelevancy, GraphContextualRecall
from graphrag_sdk import KnowledgeGraph, KnowledgeGraphModelConfig, GenerativeModelConfig


load_dotenv()
TEST_FILE = "tests/data/madoff.txt"

# Test queries and expected results
BENCHMARK = [
    {
        "query": "How many actors acted in a movie?",
        "expected": "Over than 10 actors acted in a movie."
    },
    {
        "query": "Which actors acted in a movie?",
        "expected": "Joseph Scotto, Melony Feliciano, and Donna Pastorello acted in a movie."
    },
    {
        "query": "What is the role of Joseph Scotto in a movie?",
        "expected": "Joseph Scotto played the role of Bernie Madoff in a movie."
    },
    {
        "query": "Did Donna Pastorello act in a movie?",
        "expected": "Yes, Donna Pastorello acted in a movie as Eleanor Squillari."
    },
    {
        "query": "Who played the role of Mark Madoff in a movie?",
        "expected": "Alex Olson played the role of Mark Madoff in a movie."
    },
    {
        "query": "Did Melony Feliciano have a named role in a movie?",
        "expected": "No, Melony Feliciano acted as a background extra in a movie."
    },]

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture
def movie_actor_ontology():
    """
    Create and return an ontology for movies and actors with the ACTED_IN relationship.
    """
    ontology = Ontology()

    # Add Actor entity with name attribute
    ontology.add_entity(
        Entity(
            label="Actor",
            attributes=[
                Attribute(
                    name="name",
                    attr_type=AttributeType.STRING,
                    unique=True,
                    required=True,
                ),
            ],
        )
    )
    
    # Add Movie entity with title attribute
    ontology.add_entity(
        Entity(
            label="Movie",
            attributes=[
                Attribute(
                    name="title",
                    attr_type=AttributeType.STRING,
                    unique=True,
                    required=True,
                ),
            ],
        )
    )
    
    # Add ACTED_IN relation between Actor and Movie with role attribute
    ontology.add_relation(
        Relation(
            label="ACTED_IN",
            source="Actor",
            target="Movie",
            attributes=[
                Attribute(
                    name="role",
                    attr_type=AttributeType.STRING,
                    unique=False,
                    required=True,
                ),
            ],
        )
    )
    
    return ontology


@pytest.fixture
def knowledge_graph_setup(movie_actor_ontology):
    """
    Sets up the knowledge graph using the provided ontology.
    """
    # Get the model name from the environment variable, with a default
    model_name = os.getenv("TEST_MODEL", "gemini/gemini-2.0-flash")
    graph_name = model_name.split("/")[0]
    
    # Configure the model with zero temperature for deterministic outputs
    model = LiteModel(
        model_name=model_name, 
        generation_config=GenerativeModelConfig(temperature=0)
    )

    # Initialize the knowledge graph with the model and ontology
    kg = KnowledgeGraph(
        name=graph_name,
        ontology=movie_actor_ontology,
        model_config=KnowledgeGraphModelConfig.with_model(model),
    )
    
    # URL for a movie on IMDB
    with open(TEST_FILE) as f:
        string = f.read()
        
    sources = [Source_FromRawText(string)]

    # Process the source to populate the knowledge graph
    kg.process_sources(sources)
    
    return kg


@pytest.fixture
def delete_kg():
    """
    Returns a function that deletes a given knowledge graph.
    """
    def cleanup(kg):
        logger.info("Cleaning up test graph...")
        kg.delete()

    return cleanup


class TestGraphRAGPipeline:
    """
    Test Knowledge Graph using LiteLLM for query processing and evaluation.
    Tests the ability of the graph to answer questions based on processed sources.
    """
    
    def test_movie_actor_queries(self, knowledge_graph_setup, delete_kg):
        """
        Test the knowledge graph's ability to answer questions about actors and movies.
        """
        kg = knowledge_graph_setup
        
        # Configure evaluation metrics
        relevancy_metric = GraphContextualRelevancy(threshold=0.5)
        recall_metric = GraphContextualRecall(threshold=0.5)
        scores = []

        # Run each test case and evaluate metrics
        for case in BENCHMARK:
            chat = kg.chat_session()
            answer = chat.send_message(case["query"])

            # Create a test case for evaluation
            test_case = LLMTestCase(
                input=case["query"],
                actual_output=answer["response"],
                retrieval_context=[answer["context"]],
                context=[answer["context"]],
                name="knowledge_graph_test",
                expected_output=case["expected"],
                additional_metadata=answer["cypher"],
            )
            
            # Measure metrics
            relevancy_score = relevancy_metric.measure(test_case)
            recall_score = recall_metric.measure(test_case)
            
            # Calculate and store average score
            combined_score = (relevancy_score + recall_score) / 2
            scores.append(combined_score)
            
            # Log results for debugging
            logger.info(f"Query: {case['query']}")
            logger.info(f"Response: {answer['response']}")
            logger.info(f"Combined Score: {combined_score}")

        # Clean up by deleting the graph
        delete_kg(kg)
        
        # Verify that the average score meets the threshold
        average_score = np.mean(scores)
        logger.info(f"Average Score: {average_score}")
        assert average_score >= 0.5, f"Average score {average_score} is below threshold of 0.5"