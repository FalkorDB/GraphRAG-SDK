from graphrag_sdk.steps.Step import Step
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.models import (
    GenerativeModelChatSession,
)
from graphrag_sdk.fixtures.prompts import (
    CYPHER_GEN_SYSTEM,
    CYPHER_GEN_PROMPT,
    CYPHER_GEN_PROMPT_WITH_ERROR,
    CYPHER_GEN_PROMPT_WITH_HISTORY,
)
import logging
from graphrag_sdk.helpers import (
    extract_cypher,
    validate_cypher,
    stringify_falkordb_response,
)
from falkordb import Graph
import re

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
        model_embedding: str = None,
    ) -> None:
        self.ontology = ontology
        self.config = config or {}
        self.graph = graph
        self.chat_session = chat_session
        self.last_answer = last_answer
        self.model_embedding = model_embedding

    def run(self, question: str, retries: int = 5):
        error = False

        cypher = ""
        while error is not None and retries > 0:
            try:
                cypher_prompt = (
                    (CYPHER_GEN_PROMPT.format(question=question) 
                    if self.last_answer is None
                    else CYPHER_GEN_PROMPT_WITH_HISTORY.format(question=question, last_answer=self.last_answer))
                    if error is False
                    else CYPHER_GEN_PROMPT_WITH_ERROR.format(
                        question=question, error=error
                    )
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
                    cypher = self.rephrase_entities(cypher)
                    result_set = self.graph.query(cypher).result_set
                    context = stringify_falkordb_response(result_set)
                    logger.debug(f"Context: {context}")
                    logger.debug(f"Context size: {len(result_set)}")
                    logger.debug(f"Context characters: {len(str(context))}")

                return (context, cypher)
            except Exception as e:
                logger.debug(f"Error: {e}")
                error = e
                retries -= 1

        raise Exception("Failed to generate Cypher query: " + str(error))

    
    def rephrase_entities(self, cypher_query: str):
        """
        Rephrase entities in the cypher query to make them more human-readable.
        """
        
        triplets = self.extract_triplets(cypher_query)
        for triplet in triplets:
            res = self.graph.query(f"CALL db.idx.fulltext.queryNodes('{triplet[1]}', '{triplet[0]}') YIELD node, score RETURN node.{triplet[2]}, score")
            print(triplet, res.result_set)
            if res.result_set is not None:
                cypher_query = cypher_query.replace(f"'{triplet[0]}'", f"'{res.result_set[0][0]}'")
        return cypher_query
        # results = self.graph.query(query, params=params)
        # print(results.result_set)
        
        
    def extract_triplets(self, cypher_query: str):
        """
        Extract triplets from the cypher query.
        """
        string_matches = re.findall(r"'(.*?)'", cypher_query)
        label_matches = re.findall(r"\((\w+):(\w+)(?:\s*{([^}]*)})?\)", cypher_query)
        attribute_matches = re.findall(r"(\w+)\.(\w+)", cypher_query)

        var_to_label = {var: label for var, label, _ in label_matches}
        var_to_properties = {
            var: dict(pair.split(':') for pair in (props or '').split(',') if ':' in pair)
            for var, _, props in label_matches
        }

        results = {(val.strip().strip("'"), var_to_label[var], key) 
                for var, props in var_to_properties.items() 
                for key, val in props.items()}

        results.update((string, var_to_label[var], attr) 
                    for var, attr in attribute_matches 
                    for string in string_matches if string in cypher_query)
        
        return list(results)
