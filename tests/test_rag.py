import os
import logging
import unittest
import numpy as np
from dotenv import load_dotenv
from graphrag_sdk.entity import Entity
from graphrag_sdk.source import Source
from deepeval.test_case import LLMTestCase
from graphrag_sdk.relation import Relation
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.models.litellm import LiteModel
from graphrag_sdk.attribute import Attribute, AttributeType
from graphrag_sdk import KnowledgeGraph, KnowledgeGraphModelConfig

from graphrag_sdk.test_metrics import CombineMetrics
os.environ["DEEPEVAL_ENABLE_TELEMETRY"] = "NO"
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestKGLiteLLM(unittest.TestCase):
    """
    Test Knowledge Graph
    """

    @classmethod
    def setUpClass(cls):
        # Get the model name from the environment variable
        model_name = os.getenv("TEST_MODEL", "gemini/gemini-2.0-flash-exp")

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
        cls.graph_name = "IMDB_deep"

        # Use the model name from the environment variable
        model = LiteModel(model_name=model_name)

        cls.kg = KnowledgeGraph(
            name=cls.graph_name,
            ontology=cls.ontology,
            model_config=KnowledgeGraphModelConfig.with_model(model),
        )

    def test_llm(self):
        file_path = "tests/data/madoff.txt"

        sources = [Source(file_path)]

        self.kg.process_sources(sources)
        inputs = [
            "How many actors acted in a movie?",
            "Which actors acted in the movie Madoff: The Monster of Wall Street?",
            "What is the role of Joseph Scotto in Madoff: The Monster of Wall Street?",
            "Did Donna Pastorello act in Madoff: The Monster of Wall Street?",
            "Who played the role of Mark Madoff in Madoff: The Monster of Wall Street?",
            "Did Melony Feliciano have a named role in Madoff: The Monster of Wall Street?",
        ]

        expected_outputs = [
            "Over than 10 actors acted in a movie.",
            "Joseph Scotto, Melony Feliciano, and Donna Pastorello acted in the movie Madoff: The Monster of Wall Street.",
            "Joseph Scotto played the role of Bernie Madoff in Madoff: The Monster of Wall Street.",
            "Yes, Donna Pastorello acted in Madoff: The Monster of Wall Street as Eleanor Squillari.",
            "Alex Olson played the role of Mark Madoff in Madoff: The Monster of Wall Street.",
            "No, Melony Feliciano acted as a background extra in Madoff: The Monster of Wall Street.",
        ]

        answer_combined_metric = CombineMetrics(threshold=0.5)
        scores = []

        for input_text, expected_output in zip(inputs, expected_outputs):
            chat = self.kg.chat_session()
            answer = chat.send_message(input_text)

            test_case = LLMTestCase(
                input=input_text,
                actual_output=answer["response"],
                retrieval_context=["Cypher Query: " + answer["cypher"] + " Output: " + answer["context"]],
                context=["Cypher Query: " + answer["cypher"] + " Output: " + answer["context"]],
                name="kg_test",
                expected_output=expected_output,
                additional_metadata=None,
            )

            score = answer_combined_metric.measure(test_case)
            scores.append(score)

        assert np.mean(scores) >= 0.8
