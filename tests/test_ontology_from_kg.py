import pytest
import logging
from json import loads
from falkordb import FalkorDB
from dotenv import load_dotenv
from graphrag_sdk.entity import Entity
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.relation import Relation
from graphrag_sdk.attribute import Attribute, AttributeType
from graphrag_sdk.models.gemini import GeminiGenerativeModel
from graphrag_sdk import KnowledgeGraph, KnowledgeGraphModelConfig

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def ontology_kg_setup():
    """
    Sets up an ontology, initializes the KnowledgeGraph, and imports data.
    """
    # Build up the ontology.
    ontology = Ontology()
    ontology.add_entity(
        Entity(
            label="City",
            attributes=[
                Attribute("name", AttributeType.STRING, required=False, unique=False),
                Attribute("population", AttributeType.NUMBER, required=False, unique=False),
                Attribute("weather", AttributeType.STRING, required=False, unique=False),
            ],
        )
    )
    ontology.add_entity(
        Entity(
            label="Country",
            attributes=[
                Attribute("name", AttributeType.STRING, required=False, unique=False),
            ],
        )
    )
    ontology.add_entity(
        Entity(
            label="Restaurant",
            attributes=[
                Attribute("description", AttributeType.STRING, required=False, unique=False),
                Attribute("food_type", AttributeType.STRING, required=False, unique=False),
                Attribute("name", AttributeType.STRING, required=False, unique=False),
                Attribute("rating", AttributeType.NUMBER, required=False, unique=False),
            ],
        )
    )
    ontology.add_relation(Relation(label="IN_COUNTRY", source="City", target="Country"))
    ontology.add_relation(Relation(label="IN_CITY", source="Restaurant", target="City"))

    # Create a model and a knowledge graph.
    model = GeminiGenerativeModel("gemini-2.0-flash")
    kg = KnowledgeGraph(
        name="test_ontology",
        ontology=ontology,
        model_config=KnowledgeGraphModelConfig.with_model(model),
    )

    # Import test data.
    with open("tests/data/cities.json") as f:
        cities = loads(f.read())
    with open("tests/data/restaurants.json") as f:
        restaurants = loads(f.read())

    for city in cities:
        kg.add_node(
            "City",
            {
                "name": city["name"],
                "weather": city["weather"],
                "population": city["population"],
            },
        )
        kg.add_node("Country", {"name": city["country"]})
        kg.add_edge(
            "IN_COUNTRY",
            "City",
            "Country",
            {"name": city["name"]},
            {"name": city["country"]},
        )

    for restaurant in restaurants:
        kg.add_node(
            "Restaurant",
            {
                "name": restaurant["name"],
                "description": restaurant["description"],
                "rating": restaurant["rating"],
                "food_type": restaurant["food_type"],
            },
        )
        kg.add_edge(
            "IN_CITY",
            "Restaurant",
            "City",
            {"name": restaurant["name"]},
            {"name": restaurant["city"]},
        )

    return ontology, kg


@pytest.fixture
def delete_kg():
    """
    Returns a function that deletes a given knowledge graph.
    """
    def cleanup(kg):
        logger.info("Cleaning up test graph...")
        kg.delete()

    return cleanup


class TestOntologyFromKG:
    def test_ontology_serialization(self, ontology_kg_setup, delete_kg):
        """
        Tests serializing the Ontology from the knowledge graph.
        """
        ontology, kg = ontology_kg_setup
        logger.info("Testing ontology serialization...")

        db = FalkorDB()
        graph = db.select_graph("test_ontology")
        loaded_ontology = Ontology.from_kg_graph(graph=graph)

        assert loaded_ontology.to_json() == ontology.to_json()

        # Now clean up the KG by calling the function from delete_kg fixture
        delete_kg(kg)