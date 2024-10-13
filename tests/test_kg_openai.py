import re
import logging
import unittest
from falkordb import FalkorDB
from dotenv import load_dotenv
from graphrag_sdk.entity import Entity
from graphrag_sdk.source import Source
from graphrag_sdk.relation import Relation
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.attribute import Attribute, AttributeType
from graphrag_sdk.models.openai import OpenAiGenerativeModel
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
ontology.add_entity(
    Entity(
        label="Director",
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

ontology.add_relation(
    Relation(
        label="DIRECTED_IN",
        source="Director",
        target="Movie",
        attributes=[
        ],
    )
)
graph_name = "IMDB_openai"
model = OpenAiGenerativeModel(model_name="gpt-3.5-turbo-0125")
kg = KnowledgeGraph(
    name=graph_name,
    ontology=ontology,
    model_config=KnowledgeGraphModelConfig.with_model(model),
    indexing=True
)
urls = ["https://www.rottentomatoes.com/m/side_by_side_2012",
"https://www.rottentomatoes.com/m/matrix",
"https://www.rottentomatoes.com/m/matrix_revolutions",
"https://www.rottentomatoes.com/m/matrix_reloaded",
"https://www.rottentomatoes.com/m/speed_1994",
"https://www.rottentomatoes.com/m/john_wick_chapter_4"]

sources = [Source(url) for url in urls]
# kg.process_sources(sources)

chat = kg.chat_session()

print(chat.send_message("Who is the director of the movie matrix?"))
print(chat.send_message("How this director connected to Reeves?"))
print(chat.send_message("Who is the director of the movie side by side?"))
print(chat.send_message("Order the directors that you mentioned in all of our conversation by lexical order."))

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

        answer = self.kg.ask("How many actors acted in a movie?")

        logger.info(f"Answer: {answer}")

        actors_count = re.findall(r'\d+', answer[0])
        num_actors = 0 if len(actors_count) == 0 else int(actors_count[0])

        assert num_actors > 10, "The number of actors found should be greater than 10"

    def test_kg_delete(self):
        self.kg.delete()

        db = FalkorDB()
        graphs = db.list_graphs()
        self.assertNotIn(self.graph_name, graphs)
