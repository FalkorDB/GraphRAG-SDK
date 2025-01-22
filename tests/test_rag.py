import logging
import unittest
from dotenv import load_dotenv
from deepeval import assert_test
from graphrag_sdk.entity import Entity
from graphrag_sdk.source import Source
from deepeval.test_case import LLMTestCase
from graphrag_sdk.relation import Relation
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.models.litellm import LiteModel
from graphrag_sdk.attribute import Attribute, AttributeType
from graphrag_sdk import KnowledgeGraph, KnowledgeGraphModelConfig
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from graphrag_sdk.custom_metric import FaithfulRelevancyGraphContextualMetric

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
ontology = Ontology()

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
graph_name = "IMDB_deep"
model = LiteModel(model_name="gemini/gemini-2.0-flash-exp")

kg_gemini = KnowledgeGraph(
    name=graph_name,
    ontology=ontology,
    model_config=KnowledgeGraphModelConfig.with_model(model),
)
file_path = "tests/data/madoff.txt"

sources = [Source(file_path)]

kg_gemini.process_sources(sources)

answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
answer_cp_metric = ContextualPrecisionMetric(threshold=0.5)
answer_crecall_metric = ContextualRecallMetric(threshold=0.5)
answer_crelevancy_metric = ContextualRelevancyMetric(threshold=0.5)

inputs = ["How many actors acted in a movie?",
          "Which actors acted in the movie Madoff: The Monster of Wall Street?",
          "What is the role of Joseph Scotto in Madoff: The Monster of Wall Street?",
          "Did Donna Pastorello act in Madoff: The Monster of Wall Street?",
          "Who played the role of Mark Madoff in Madoff: The Monster of Wall Street?",
          "Did Melony Feliciano have a named role in Madoff: The Monster of Wall Street?"]

expected_outputs = ["Over than 10 actors acted in a movie.",
                    "Joseph Scotto, Melony Feliciano, and Donna Pastorello acted in the movie Madoff: The Monster of Wall Street.",
                    "Joseph Scotto played the role of Bernie Madoff in Madoff: The Monster of Wall Street.",
                    "Yes, Donna Pastorello acted in Madoff: The Monster of Wall Street as Eleanor Squillari.",
                    "Alex Olson played the role of Mark Madoff in Madoff: The Monster of Wall Street.",
                    "No, Melony Feliciano acted as a background extra in Madoff: The Monster of Wall Street."]

answer_faithful_relevancy_metric = FaithfulRelevancyGraphContextualMetric(threshold=0.5)
scores = []
for input, expected_output in zip(inputs, expected_outputs):
    chat = kg_gemini.chat_session()
    answer = chat.send_message(input)

    test_case = LLMTestCase(
            input=input,
            actual_output=answer['response'],
            retrieval_context=[answer['context']],
            context=[answer['context']],
            name="gemini",
            expected_output=expected_output,
            additional_metadata={"cypher_query": answer['cypher']}
        )

    score = answer_faithful_relevancy_metric.measure(test_case)
    scores.append(score)

assert_test(test_case, [answer_relevancy_metric, answer_cp_metric, answer_crelevancy_metric])

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
        cls.graph_name = "IMDB_deep"
        model = LiteModel(model_name="gpt-4o")
        
        cls.kg_gemini = KnowledgeGraph(
            name=cls.graph_name,
            ontology=cls.ontology,
            model_config=KnowledgeGraphModelConfig.with_model(model),
        )


    def test_kg_creation(self):

        file_path = "tests/data/madoff.txt"

        sources = [Source(file_path)]

        self.kg_gemini.process_sources(sources)
        input = "How many actors acted in a movie?"

        chat = self.kg_gemini.chat_session()
        answer = chat.send_message(input)
        answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
        answer_cp_metric = ContextualPrecisionMetric(threshold=0.5)
        answer_crecall_metric = ContextualRecallMetric(threshold=0.5)
        answer_crelevancy_metric = ContextualRelevancyMetric(threshold=0.5)



        test_case = LLMTestCase(
                input=input,
                actual_output=answer['response'],
                retrieval_context=[answer['context']],
                context=[answer['context']],
                name="gemini",
                expected_output="Over than 10 actors acted in a movie."
            )
        assert_test(test_case, [answer_relevancy_metric, answer_cp_metric, answer_crelevancy_metric])