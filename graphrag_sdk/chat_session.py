from falkordb import Graph
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.steps.qa_step import QAStep
from graphrag_sdk.model_config import KnowledgeGraphModelConfig
from graphrag_sdk.steps.graph_query_step import GraphQueryGenerationStep


class ChatSession:
    """
    Represents a chat session with a Knowledge Graph.

    Args:
        model_config (KnowledgeGraphModelConfig): The model configuration to use.
        ontology (Ontology): The ontology to use.
        graph (Graph): The graph to query.

    Examples:
        >>> from graphrag_sdk import KnowledgeGraph, Orchestrator
        >>> from graphrag_sdk.ontology import Ontology
        >>> from graphrag_sdk.model_config import KnowledgeGraphModelConfig
        >>> model_config = KnowledgeGraphModelConfig.with_model(model)
        >>> kg = KnowledgeGraph("test_kg", model_config, ontology)
        >>> chat_session = kg.start_chat()
        >>> chat_session.send_message("What is the capital of France?")
    """

    def __init__(self, model_config: KnowledgeGraphModelConfig, ontology: Ontology, graph: Graph,
                cypher_system_instruction: str, qa_system_instruction: str,
                cypher_gen_prompt: str, qa_prompt: str, cypher_gen_prompt_history: str):
        """
        Initializes a new ChatSession object.

        Args:
            model_config (KnowledgeGraphModelConfig): The model configuration.
            ontology (Ontology): The ontology object.
            graph (Graph): The graph object.

        Attributes:
            model_config (KnowledgeGraphModelConfig): The model configuration.
            ontology (Ontology): The ontology object.
            graph (Graph): The graph object.
            cypher_chat_session (CypherChatSession): The Cypher chat session object.
            qa_chat_session (QAChatSession): The QA chat session object.
        """
        self.model_config = model_config
        self.graph = graph
        self.ontology = ontology
        cypher_system_instruction = cypher_system_instruction.format(ontology=str(ontology.to_json()))

        
        self.cypher_prompt = cypher_gen_prompt
        self.qa_prompt = qa_prompt
        self.cypher_prompt_with_history = cypher_gen_prompt_history
        
        self.cypher_chat_session = (
            model_config.cypher_generation.with_system_instruction(
                cypher_system_instruction
            ).start_chat()
        )
        self.qa_chat_session = model_config.qa.with_system_instruction(
                qa_system_instruction
            ).start_chat()
        self.last_answer = None

    def send_message(self, message: str):
        """
        Sends a message to the chat session.

        Args:
            message (str): The message to send.

        Returns:
            dict: The response to the message in the following format:
                    {"question": message, 
                    "response": answer, 
                    "context": context, 
                    "cypher": cypher}
        """
        cypher_step = GraphQueryGenerationStep(
            graph=self.graph,
            chat_session=self.cypher_chat_session,
            ontology=self.ontology,
            last_answer=self.last_answer,
            cypher_prompt=self.cypher_prompt,
            cypher_prompt_with_history=self.cypher_prompt_with_history
        )

        (context, cypher) = cypher_step.run(message)

        if not cypher or len(cypher) == 0:
            return {
                "question": message,
                "response": "I am sorry, I could not find the answer to your question",
                "context": None,
                "cypher": None
                }

        qa_step = QAStep(
            chat_session=self.qa_chat_session,
            qa_prompt=self.qa_prompt,
        )

        answer = qa_step.run(message, cypher, context)
        self.last_answer = answer
        
        return {
            "question": message, 
            "response": answer, 
            "context": context, 
            "cypher": cypher
            }