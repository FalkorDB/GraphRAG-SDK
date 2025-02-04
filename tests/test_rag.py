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
from graphrag_sdk import KnowledgeGraph, KnowledgeGraphModelConfig, GenerativeModelConfig

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

        cls.ontology = Ontology()

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
                        required=True,
                    ),
                ],
            )
        )
        cls.graph_name = model_name.split("/")[0]

        # Use the model name from the environment variable
        model = LiteModel(model_name=model_name, generation_config=GenerativeModelConfig(temperature=0))


        cls.kg = KnowledgeGraph(
            name=cls.graph_name,
            ontology=cls.ontology,
            model_config=KnowledgeGraphModelConfig.with_model(model),
        )

    def test_llm(self):
        url = "https://www.imdb.com/title/tt23732458/"

        sources = [Source(url)]

        self.kg.process_sources(sources)
        inputs = [
            "How many actors acted in a movie?",
            "Which actors acted in a movie?",
            "What is the role of Joseph Scotto in a movie?",
            "Did Donna Pastorello act in a movie?",
            "Who played the role of Mark Madoff in a movie?",
            "Did Melony Feliciano have a named role in a movie?",
        ]

        expected_outputs = [
            "Over than 10 actors acted in a movie.",
            "Joseph Scotto, Melony Feliciano, and Donna Pastorello acted in a movie",
            "Joseph Scotto played the role of Bernie Madoff in a movie.",
            "Yes, Donna Pastorello acted in a movie as Eleanor Squillari.",
            "Alex Olson played the role of Mark Madoff in a movie.",
            "No, Melony Feliciano acted as a background extra in a movie.",
        ]
        answer_combined_metric = CombineMetrics(threshold=0.5)
        scores = []

        for input_text, expected_output in zip(inputs, expected_outputs):
            chat = self.kg.chat_session()
            answer = chat.send_message(input_text)

            test_case = LLMTestCase(
            input=input_text,
            actual_output=answer["response"],
            retrieval_context=[answer["context"]],
            context=[answer["context"]],
            name="kg_rag_test",
            expected_output=expected_output,
            additional_metadata=answer["cypher"],
            )
            score = answer_combined_metric.measure(test_case)
            scores.append(score)

        self.kg.delete()
        assert np.mean(scores) >= 0.5
