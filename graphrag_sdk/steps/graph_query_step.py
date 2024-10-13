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
        indexing: bool = False,
    ) -> None:
        self.ontology = ontology
        self.config = config or {}
        self.graph = graph
        self.chat_session = chat_session
        self.last_answer = last_answer
        if indexing:
            self.indexes = self.graph.query("call db.indexes()").result_set
        else:
            self.indexes = None

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
                    if self.indexes is not None:
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

    
    def rephrase_entities(self, cypher_query: str) -> str:
        """
        Rephrase entities in the cypher query to make them more human-readable.
        """
        triplets = self.extract_triplets(cypher_query)
        
        for entity, label, property_name, var in triplets:
            if (index := next((idx for idx in self.indexes if label in idx and property_name in idx[1]), None)) is None:
                # Create a new index if it does not exist
                self.graph.query(f"CALL db.idx.fulltext.createNodeIndex('{label}', '{property_name}')")
                self.indexes.append((label, [property_name]))
            else:
                # Query the fulltext index
                res = self.graph.query(f"CALL db.idx.fulltext.queryNodes('{label}', '{entity}') YIELD node, score RETURN node.{property_name}, score")
                if res.result_set:
                    matched_value = [i[0] for i in res.result_set]
                    if entity not in matched_value:
                        print('Changed from: ', entity, '->',matched_value[0])
                        cypher_query = cypher_query.replace(f"'{entity}'", f"'{matched_value[0]}'")
        
        return cypher_query
        
        
    def extract_triplets(self, cypher_query: str):
        """
        Extract triplets from the cypher query.
        """
        string_matches = re.findall(r"'(.*?)'", cypher_query)
        label_matches = re.findall(r"\((\w+):(\w+)(?:\s*{([^}]*)})?\)", cypher_query)
        attribute_conditions = re.findall(r"(\w+)\.(\w+)\s*CONTAINS\s*'(.+?)'", cypher_query)

        var_to_label = {var: label for var, label, _ in label_matches}
        var_to_properties = {
            var: dict(pair.split(':') for pair in (props or '').split(',') if ':' in pair)
            for var, _, props in label_matches
        }

        results = set()
        for var, props in var_to_properties.items():
            for key, val in props.items():
                results.add((val.strip().strip("'"), var_to_label[var], key, var))


        for var, attr, value in attribute_conditions:
            if var in var_to_label and value in string_matches:
                results.add((value, var_to_label[var], attr, var))
        
        return list(results)
