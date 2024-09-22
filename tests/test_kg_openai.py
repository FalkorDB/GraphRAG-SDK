from dotenv import load_dotenv

load_dotenv()
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.entity import Entity
from graphrag_sdk.relation import Relation
from graphrag_sdk.attribute import Attribute, AttributeType
import unittest
from graphrag_sdk.source import Source
from graphrag_sdk.models.openai import OpenAiGenerativeModel
from graphrag_sdk import KnowledgeGraph, KnowledgeGraphModelConfig
import logging
from falkordb import FalkorDB

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



ontology = Ontology([], [])

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
                required=False,
            ),
        ],
    )
)

file_path = "tests/data/madoff.txt"

sources = [Source(file_path)]

for i in range(10):
    try:
        graph_name = "IMDB_openai_" + str(i)
        model = OpenAiGenerativeModel(model_name="gpt-3.5-turbo-0125")
        kg = KnowledgeGraph(
            name=graph_name,
            ontology=ontology,
            model_config=KnowledgeGraphModelConfig.with_model(model),
        )

        kg.process_sources(sources)
        kg.delete()
    except Exception as e:
        print(e)

answer = kg.ask("List a few actors")

logger.info(f"Answer: {answer}")

assert "Joseph Scotto" in answer[0], "Joseph Scotto not found in answer"

class TestKGOpenAI(unittest.TestCase):
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
        cls.graph_name = "IMDB_openai"
        model = OpenAiGenerativeModel(model_name="gpt-3.5-turbo-0125")
        cls.kg = KnowledgeGraph(
            name=cls.graph_name,
            ontology=cls.ontology,
            model_config=KnowledgeGraphModelConfig.with_model(model),
        )

    def test_kg_creation(self):

        file_path = "tests/data/madoff.txt"

        sources = [Source(file_path)]

        self.kg.process_sources(sources)

        answer = self.kg.ask("List a few actors")

        logger.info(f"Answer: {answer}")

        assert "Joseph Scotto" in answer[0], "Joseph Scotto not found in answer"

    def test_kg_delete(self):

        self.kg.delete()

        db = FalkorDB()
        graphs = db.list_graphs()
        self.assertNotIn(self.graph_name, graphs)
