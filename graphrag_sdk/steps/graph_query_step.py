import logging
from falkordb import Graph
from graphrag_sdk.steps.Step import Step
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.models import (
    GenerativeModelChatSession,
)
from graphrag_sdk.helpers import (
    extract_cypher,
    validate_cypher,
    stringify_falkordb_response,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GraphQueryGenerationStep(Step):
    """
    Graph Query Step
    """

    def __init__(
        self,
        graph: Graph,
        ontology: Ontology,
        chat_session: GenerativeModelChatSession,
        config: dict = None,
        last_answer: str = None,
        cypher_prompt: str = None,
        cypher_prompt_with_history: str = None,
    ) -> None:
        self.ontology = ontology
        self.config = config or {}
        self.graph = graph
        self.chat_session = chat_session
        self.last_answer = last_answer
        self.cypher_prompt = cypher_prompt
        self.cypher_prompt_with_history = cypher_prompt_with_history

    def run(self, question: str, retries: int = 10):
        cypher = ""
        for i in range(retries):
            try:
                cypher_prompt = (
                    (self.cypher_prompt.format(question=question) 
                    if self.last_answer is None
                    else self.cypher_prompt_with_history.format(question=question, last_answer=self.last_answer))
                )   
                logger.debug(f"Cypher Prompt: {cypher_prompt}")
                cypher_statement_response = self.chat_session.send_message(
                    cypher_prompt,
                )
                logger.debug(f"Cypher Statement Response: {cypher_statement_response}")
                cypher = extract_cypher(cypher_statement_response.text)
                logger.debug(f"Cypher: {cypher}")

                if not cypher or len(cypher) == 0:
                    return (None, None)

                validation_errors = validate_cypher(cypher, self.ontology)
                if validation_errors is not None:
                    raise Exception("\n".join(validation_errors))

                if cypher is not None:
                    result_set = self.graph.query(cypher).result_set
                    context = stringify_falkordb_response(result_set)
                    logger.debug(f"Context: {context}")
                    logger.debug(f"Context size: {len(result_set)}")
                    logger.debug(f"Context characters: {len(str(context))}")

                return (context, cypher)
            except Exception as e:
                logger.debug(f"Error: {e}")
                error = e
                self.chat_session.delete_last_message()

        raise Exception("Failed to generate Cypher query: " + str(error))
