import re
import os
import logging
import unittest
from falkordb import FalkorDB
from dotenv import load_dotenv
from graphrag_sdk.entity import Entity
from graphrag_sdk.source import Source_FromRawText
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.relation import Relation
from graphrag_sdk.attribute import Attribute, AttributeType
from graphrag_sdk.models.gemini import GeminiGenerativeModel
from graphrag_sdk.embeddings.litellm import LiteModelEmbeddings
from graphrag_sdk import KnowledgeGraph, KnowledgeGraphModelConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
        model_embed = LiteModelEmbeddings(model_name="gemini/text-embedding-004")
        cls.kg = KnowledgeGraph(
            name=cls.graph_name,
            ontology=cls.ontology,
            model_config=KnowledgeGraphModelConfig.with_model(model),
            model_embedding=model_embed,
        )

    def test_kg_creation(self):
        
        file_path = "tests/data/madoff.txt"
        with open(file_path) as f:
            string = f.read()
            
        sources = [Source_FromRawText(string)]
        chunking_processor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        self.kg.process_sources(sources, chunking_processor=chunking_processor)
        
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
