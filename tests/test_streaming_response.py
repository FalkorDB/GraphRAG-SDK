import os
import pytest
import logging
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

# Test case
USECASE = {
        "query": "How many actors acted in a movie?\nPlease give me full details of the actors with a long output.",
        "expected": "More than 10 actors acted in a movie."
    }

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


class TestStreamingResponse:
    """
    Test the knowledge graph's ability to answer questions about actors and movies.
    Test the streaming response of the knowledge graph.
    """
    
    def test_streaming(self, knowledge_graph_setup, delete_kg):
        """
        Test the knowledge graph's ability to answer questions about actors and movies with streaming response.
        """
        kg = knowledge_graph_setup
        
        # Configure evaluation metrics
        relevancy_metric = GraphContextualRelevancy(threshold=0.5)
        recall_metric = GraphContextualRecall(threshold=0.5)

        chat = kg.chat_session()
        # Track that we received multiple chunks to verify streaming
        received_chunks = []
        for chunk in chat.send_message_stream(USECASE["query"]):
            logger.info(chunk)
            received_chunks.append(chunk)

        # Verify that streaming actually occurred (received multiple chunks)
        assert len(received_chunks) > 1, "Expected multiple chunks in streaming response"

        # Get the last complete response for the cypher and context
        response_dict = chat.last_complete_response
        answer = ' '.join(received_chunks)

        assert answer.strip() == response_dict["response"].strip(), "Combined chunks (using join) should match last complete response"

        # Create a test case for evaluation
        test_case = LLMTestCase(
            input=USECASE["query"],
            actual_output=answer,
            retrieval_context=[response_dict["context"]],
            context=[response_dict["context"]],
            name="streaming_test",
            expected_output=USECASE["expected"],
            additional_metadata=response_dict["cypher"],
        )

        # Measure metrics
        relevancy_score = relevancy_metric.measure(test_case)
        recall_score = recall_metric.measure(test_case)

        # Calculate and store average score
        combined_score = (relevancy_score + recall_score) / 2
        
        # Log results for debugging
        logger.info(f"Query: {USECASE['query']}")
        logger.info(f"Response: {answer['response']}")
        logger.info(f"Combined Score: {combined_score}")

        # Clean up by deleting the graph
        delete_kg(kg)
        
        assert combined_score >= 0.5, f"Average score {combined_score} is below threshold of 0.5"