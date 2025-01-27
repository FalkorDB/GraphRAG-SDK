import re
import os
import logging
import unittest
from graphrag_sdk.source import URL
from falkordb import FalkorDB
from dotenv import load_dotenv
from graphrag_sdk.entity import Entity
from graphrag_sdk.source import Source
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.relation import Relation
from graphrag_sdk.attribute import Attribute, AttributeType
from graphrag_sdk.models.litellm import LiteModel
from graphrag_sdk import KnowledgeGraph, KnowledgeGraphModelConfig

load_dotenv()

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

graph_name = "IMDB_embed"

model = LiteModel(model_name="gemini/gemini-1.5-flash-001")

kg = KnowledgeGraph(
    name=graph_name,
    ontology=ontology,
    model_config=KnowledgeGraphModelConfig.with_model(model),
)
sources = []
for file in os.listdir():
    if file.endswith(".pdf"):
        sources.append(Source(file))
urls = ["https://www.rottentomatoes.com/m/side_by_side_2012",
"https://www.rottentomatoes.com/m/matrix",
# "https://www.rottentomatoes.com/m/matrix_revolutions",
# "https://www.rottentomatoes.com/m/matrix_reloaded",
# "https://www.rottentomatoes.com/m/speed_1994",
"https://www.rottentomatoes.com/m/john_wick_chapter_4"]

sources = [URL(url) for url in urls]

failed_chunks = kg.process_sources(sources)

chat = kg.chat_session()
answer = chat.send_message("How many actors acted in a movie?")
answer = answer['response']

logger.info(f"Answer: {answer}")

actors_count = re.findall(r'\d+', answer)
num_actors = 0 if len(actors_count) == 0 else int(actors_count[0])

class TestKGGemini(unittest.TestCase):
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

        cls.graph_name = "IMDB_gemini"

        model = GeminiGenerativeModel(model_name="gemini-1.5-flash-001")
        cls.kg = KnowledgeGraph(
            name=cls.graph_name,
            ontology=cls.ontology,
            model_config=KnowledgeGraphModelConfig.with_model(model),
        )

    def test_kg_creation(self):
        file_path = "tests/data/madoff.txt"

        sources = [Source(file_path)]

        self.kg.process_sources(sources)
        
        chat = self.kg.chat_session()
        answer = chat.send_message("How many actors acted in a movie?")
        answer = answer['response']

        logger.info(f"Answer: {answer}")

        actors_count = re.findall(r'\d+', answer)
        num_actors = 0 if len(actors_count) == 0 else int(actors_count[0])

        assert num_actors > 10, "The number of actors found should be greater than 10"

    def test_kg_delete(self):
        self.kg.delete()

        db = FalkorDB()
        graphs = db.list_graphs()
        self.assertNotIn(self.graph_name, graphs)
