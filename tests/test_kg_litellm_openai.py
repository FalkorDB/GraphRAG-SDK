import re
import logging
import unittest
from falkordb import FalkorDB
from dotenv import load_dotenv
from graphrag_sdk.entity import Entity
from graphrag_sdk.source import Source_FromRawText
from graphrag_sdk.relation import Relation
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.attribute import Attribute, AttributeType
from graphrag_sdk.models.litellm import LiteModel
from graphrag_sdk import KnowledgeGraph, KnowledgeGraphModelConfig, GenerativeModelConfig

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestKGLiteLLM(unittest.TestCase):
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
        model = LiteModel(model_name="gpt-4o", generation_config=GenerativeModelConfig(temperature=0))
        cls.kg = KnowledgeGraph(
            name=cls.graph_name,
            ontology=cls.ontology,
            model_config=KnowledgeGraphModelConfig.with_model(model),
        )

    def test_kg_creation(self):

        file_path = "tests/data/madoff.txt"
        with open(file_path) as f:
            string = f.read()
            
        sources = [Source_FromRawText(string)]

        self.kg.process_sources(sources)

        chat = self.kg.chat_session()
        answer = chat.send_message("How many actors acted in a movie?")
        answer = answer['response']

        logger.info(f"Answer: {answer}")

        actors_count = re.findall(r'\d+', answer)
        num_actors = 0 if len(actors_count) == 0 else int(actors_count[0])

        assert num_actors > 10, "The number of actors found should be greater than 10"