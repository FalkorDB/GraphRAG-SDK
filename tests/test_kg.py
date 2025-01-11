from dotenv import load_dotenv

load_dotenv()
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.entity import Entity
from graphrag_sdk.relation import Relation
from graphrag_sdk.attribute import Attribute, AttributeType
import unittest
from graphrag_sdk.models.gemini import GeminiGenerativeModel
from graphrag_sdk import KnowledgeGraph, KnowledgeGraphModelConfig
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestKG(unittest.TestCase):
    """
    Test Knowledge Graph
    """

    @classmethod
    def setUpClass(cls):
        cls.ontology = Ontology([], [])

        cls.ontology.add_entity(
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
        cls.ontology.add_entity(
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
        cls.ontology.add_relation(
            Relation(
                label="ACTED_IN",
                source="Actor",
                target="Movie",
                attributes=[
                    Attribute(
                        name="role",
                        attr_type=AttributeType.STRING,
                        unique=False,
                        required=False,
                    ),
                ],
            )
        )

        cls.graph_name = "test_kg"

        model = GeminiGenerativeModel(model_name="gemini-1.5-flash-001")
        cls.kg = KnowledgeGraph(
            name=cls.graph_name,
            ontology=cls.ontology,
            model_config=KnowledgeGraphModelConfig.with_model(model),
        )

    # Delete graph after tests
    @classmethod
    def tearDownClass(cls):
        cls.kg.delete()

    def test_add_node_valid(self):
        self.kg.add_node("Actor", {"name": "Tom Hanks"})

    def test_add_node_invalid(self):
        with self.assertRaises(Exception):
            self.kg.add_node("Actor", {"title": "Tom Hanks"})

    def test_add_relation_valid(self):
        self.kg.add_edge(
            "ACTED_IN",
            "Actor",
            "Movie",
            {"name": "Tom Hanks"},
            {"title": "Forrest Gump"},
            {"role": "Forrest Gump"},
        )

    def test_add_relation_invalid(self):
        with self.assertRaises(Exception):
            self.kg.add_edge(
                "ACTED_IN",
                "Actor",
                "Movie",
                {"title": "Tom Hanks"},
                {"title": "Forrest Gump"},
                {"role": "Forrest Gump"},
            )

        with self.assertRaises(Exception):
            self.kg.add_edge(
                "ACTED_IN",
                "Actor",
                "Movie",
                {"name": "Tom Hanks"},
                {"name": "Forrest Gump"},
                {},
            )
