
import logging
import unittest
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


class TestOntologyFromKG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.ontology = Ontology()
        cls.ontology.add_entity(
            Entity(
                label="City",
                attributes=[
                    Attribute(
                        name="name",
                        attr_type=AttributeType.STRING,
                        required=False,
                        unique=False,
                    ),
                    Attribute(
                        name="population",
                        attr_type=AttributeType.NUMBER,
                        required=False,
                        unique=False,
                    ),
                    Attribute(
                        name="weather",
                        attr_type=AttributeType.STRING,
                        required=False,
                        unique=False,
                    ),
                ],
            )
        )
        cls.ontology.add_entity(
            Entity(
                label="Country",
                attributes=[
                    Attribute(
                        name="name",
                        attr_type=AttributeType.STRING,
                        required=False,
                        unique=False,
                    ),
                ],
            )
        )
        cls.ontology.add_entity(
            Entity(
                label="Restaurant",
                attributes=[
                    Attribute(
                        name="description",
                        attr_type=AttributeType.STRING,
                        required=False,
                        unique=False,
                    ),
                    Attribute(
                        name="food_type",
                        attr_type=AttributeType.STRING,
                        required=False,
                        unique=False,
                    ),
                    Attribute(
                        name="name",
                        attr_type=AttributeType.STRING,
                        required=False,
                        unique=False,
                    ),
                    Attribute(
                        name="rating",
                        attr_type=AttributeType.NUMBER,
                        required=False,
                        unique=False,
                    ),
                ],
            )
        )
        cls.ontology.add_relation(
            Relation(
                label="IN_COUNTRY",
                source="City",
                target="Country",
            )
        )
        cls.ontology.add_relation(
            Relation(
                label="IN_CITY",
                source="Restaurant",
                target="City",
            )
        )
        cls.model = GeminiGenerativeModel("gemini-1.5-flash-001")
        cls.kg = KnowledgeGraph(
            name="test_ontology",
            ontology=cls.ontology,
            model_config=KnowledgeGraphModelConfig.with_model(cls.model),
        )
        cls.import_data(cls.kg)

    @classmethod
    def import_data(
        self,
        kg: KnowledgeGraph,
    ):
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

    # Delete graph after tests
    @classmethod
    def tearDownClass(cls):
        logger.info("Cleaning up test graph...")
        cls.kg.delete()

    def test_ontology_serialization(self):
        logger.info("Testing ontology serialization...")
        db = FalkorDB()
        graph = db.select_graph("test_ontology")
        ontology = Ontology.from_kg_graph(graph=graph)
        self.assertEqual(ontology.to_json(), self.ontology.to_json())