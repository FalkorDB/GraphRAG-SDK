from dotenv import load_dotenv

load_dotenv()
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.entity import Entity
from graphrag_sdk.relation import Relation
from graphrag_sdk.attribute import Attribute, AttributeType
import unittest
from graphrag_sdk.source import Source
from graphrag_sdk.models.ollama import OllamaGenerativeModel
from graphrag_sdk import KnowledgeGraph, KnowledgeGraphModelConfig
import os
import logging
from falkordb import FalkorDB
import pytest


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

graph_name = "IMDB_ollama"

model = OllamaGenerativeModel(model_name="gemma2:9b")
kg = KnowledgeGraph(
    name=graph_name,
    ontology=ontology,
    model_config=KnowledgeGraphModelConfig.with_model(model),
)


file_path = "tests/data/madoff.txt"

sources = [Source(file_path)]

kg.process_sources(sources)

answer = kg.ask("List a few actors")

logger.info(f"Answer: {answer}")

assert "Joseph Scotto" in answer[0], "Joseph Scotto not found in answer"


class TestKGOllama(unittest.TestCase):
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

        cls.graph_name = "IMDB_ollama"

        model = OllamaGenerativeModel(model_name="gemma2:2b")
        cls.kg = KnowledgeGraph(
            name=cls.graph_name,
            ontology=cls.ontology,
            model_config=KnowledgeGraphModelConfig.with_model(model),
        )

    @pytest.mark.skipif(condition=True, reason="Not ready for testing")
    def test_kg_creation(self):

        file_path = "tests/data/madoff.txt"

        sources = [Source(file_path)]

        self.kg.process_sources(sources)

        answer = self.kg.ask("List a few actors")

        logger.info(f"Answer: {answer}")

        assert "Joseph Scotto" in answer[0], "Joseph Scotto not found in answer"

    @pytest.mark.skipif(condition=True, reason="Not ready for testing")
    def test_kg_delete(self):

        self.kg.delete()

        db = FalkorDB()
        graphs = db.list_graphs()
        self.assertNotIn(self.graph_name, graphs)
