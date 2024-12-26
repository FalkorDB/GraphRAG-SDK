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
        cls.graph_name = "IMDB"
        model = LiteModel(model_name="gpt-4o")
        
        cls.kg_openai = KnowledgeGraph(
            name=cls.graph_name,
            ontology=cls.ontology,
            model_config=KnowledgeGraphModelConfig.with_model(model),
        )
        
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
        
        chat = self.kg_openai.chat_session()
        answer = chat.send_message(input)
        test_case = LLMTestCase(
                input=input,
                actual_output=answer['response'],
                retrieval_context=[answer['context']],
                context=[answer['context']],
                name="openai",
                expected_output="Over than 10 actors acted in a movie."
            )

        assert_test(test_case, [answer_relevancy_metric,answer_cp_metric, answer_crelevancy_metric])