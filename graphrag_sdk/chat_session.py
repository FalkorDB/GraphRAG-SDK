import json
from falkordb import Graph
from typing import Iterator
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.steps.qa_step import QAStep
from graphrag_sdk.steps.stream_qa_step import StreamingQAStep
from graphrag_sdk.model_config import KnowledgeGraphModelConfig
from graphrag_sdk.steps.graph_query_step import GraphQueryGenerationStep

CYPHER_ERROR_RES = "Sorry, I could not find the answer to your question"

class ResponseDict(dict):
    """
    A dictionary that also provides property access to response components.
    Maintains backward compatibility while adding new property-based access.
    """
    
    def __init__(self, response: 'ChatResponse'):
        self._response = response
        super().__init__({
            "question": response.query,
            "response": response.answer,
            "context": response.context,
            "cypher": response.cypher
        })
    
    @property
    def query(self) -> str:
        """The original question/query."""
        return self._response.query
    
    @property
    def cypher(self) -> str:
        """The generated Cypher query."""
        return self._response.cypher
    
    @property
    def context(self) -> str:
        """The extracted context from the graph."""
        return self._response.context
    
    @property
    def answer(self) -> str:
        """The final QA answer."""
        return self._response.answer
    
    @property
    def execution_time(self) -> float:
        """Query execution time in seconds."""
        return self._response.execution_time
    
    @property
    def error(self) -> str:
        """Error message if any step failed."""
        return self._response.error

class ChatResponse:
    """
    Represents a response from a chat session with access to different pipeline stages.
    """
    
    def __init__(self, question: str, context: str = None, cypher: str = None, 
                 answer: str = None, execution_time: float = None, error: str = None):
        self._question = question
        self._context = context
        self._cypher = cypher
        self._answer = answer
        self._execution_time = execution_time
        self._error = error
    
    @property
    def query(self) -> str:
        """The original question/query."""
        return self._question
    
    @property
    def cypher(self) -> str:
        """The generated Cypher query."""
        return self._cypher
    
    @property
    def context(self) -> str:
        """The extracted context from the graph."""
        return self._context
    
    @property
    def answer(self) -> str:
        """The final QA answer."""
        return self._answer
    
    @property
    def execution_time(self) -> float:
        """Query execution time in seconds."""
        return self._execution_time
    
    @property
    def error(self) -> str:
        """Error message if any step failed."""
        return self._error
    
    def to_dict(self) -> dict:
        """Convert response to dictionary format for backward compatibility."""
        return {
            "question": self._question,
            "response": self._answer,
            "context": self._context,
            "cypher": self._cypher
        }

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
        >>> session = kg.start_chat()
        >>> response = session.send_message("What is the capital of France?")
        >>> # Backward compatible dict access:
        >>> print(response["question"])  # "What is the capital of France?"
        >>> print(response["response"])  # "Paris"
        >>> # New property access:
        >>> print(response.query)        # "What is the capital of France?"
        >>> print(response.cypher)       # "MATCH (c:City)..."
        >>> print(response.context)      # Retrieved context
        >>> print(response.answer)       # "Paris"
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
            cypher_system_instruction (str): System instruction for cypher generation.
            qa_system_instruction (str): System instruction for QA.
            cypher_gen_prompt (str): Prompt template for cypher generation.
            qa_prompt (str): Prompt template for QA.
            cypher_gen_prompt_history (str): Prompt template for cypher generation with history.

        Attributes:
            model_config (KnowledgeGraphModelConfig): The model configuration.
            ontology (Ontology): The ontology object.
            graph (Graph): The graph object.
            cypher_chat_session: The Cypher chat session object.
            qa_chat_session: The QA chat session object.
        """
        self.model_config = model_config
        self.graph = graph
        self.ontology = ontology
        
        # Filter the ontology to remove unique and required attributes that are not needed for Q&A. 
        ontology_prompt = clean_ontology_for_prompt(ontology)
                
        cypher_system_instruction = cypher_system_instruction.format(ontology=ontology_prompt)
        
        self.cypher_prompt = cypher_gen_prompt
        self.qa_prompt = qa_prompt
        self.cypher_prompt_with_history = cypher_gen_prompt_history
        
        self.cypher_chat_session = model_config.cypher_generation.start_chat(
                cypher_system_instruction
            )
        self.qa_chat_session = model_config.qa.start_chat(
                qa_system_instruction
            )
        
        self.last_complete_response = None
        
        # Metadata to store additional information about the chat session (currently only last query execution time)
        self.metadata = {"last_query_execution_time": None}
        
    def _generate_cypher_query(self, message: str) -> tuple:
        """
        Generate a Cypher query for the given message.
        
        Args:
            message (str): The message to generate a query for.
            
        Returns:
            tuple: A tuple containing (context, cypher)
        """
        last_answer = self.last_complete_response.answer if self.last_complete_response else None
        
        cypher_step = GraphQueryGenerationStep(
            graph=self.graph,
            chat_session=self.cypher_chat_session,
            ontology=self.ontology,
            last_answer=last_answer,
            cypher_prompt=self.cypher_prompt,
            cypher_prompt_with_history=self.cypher_prompt_with_history
        )

        (context, cypher, query_execution_time) = cypher_step.run(message)
        self.metadata["last_query_execution_time"] = query_execution_time
        
        return (context, cypher)

    def send_message(self, message: str) -> ResponseDict:
        """
        Sends a message to the chat session.

        Args:
            message (str): The message to send.

        Returns:
            ResponseDict: A dict-like object with backward compatibility that also provides property access:
                - dict access: result["question"], result["response"], result["context"], result["cypher"]
                - property access: result.query, result.answer, result.context, result.cypher, result.execution_time
        """
        response = self._send_message_internal(message)
        return ResponseDict(response)
    
    def _send_message_internal(self, message: str) -> ChatResponse:
        """
        Internal method that returns the full ChatResponse object.
        """
        (context, cypher) = self._generate_cypher_query(message)
        execution_time = self.metadata.get("last_query_execution_time")

        # If the cypher is empty, return an error response
        if not cypher or len(cypher) == 0:
            response = ChatResponse(
                question=message,
                context=None,
                cypher=None,
                answer=CYPHER_ERROR_RES,
                execution_time=execution_time,
                error="Could not generate valid cypher query"
            )
            self.last_complete_response = response
            return response
        
        qa_step = QAStep(
            chat_session=self.qa_chat_session,
            qa_prompt=self.qa_prompt,
        )

        answer = qa_step.run(message, cypher, context)

        response = ChatResponse(
            question=message,
            context=context,
            cypher=cypher,
            answer=answer,
            execution_time=execution_time
        )
        
        self.last_complete_response = response
        return response
    
    def send_message_stream(self, message: str) -> Iterator[str]:
        """
        Sends a message to the chat session and streams the response.

        Args:
            message (str): The message to send.

        Yields:
            str: Chunks of the response as they're generated.
        """
        (context, cypher) = self._generate_cypher_query(message)
        execution_time = self.metadata.get("last_query_execution_time")

        if not cypher or len(cypher) == 0:
            # Stream the error message for consistency with successful responses
            yield CYPHER_ERROR_RES
            
            self.last_complete_response = ChatResponse(
                question=message,
                context=None,
                cypher=None,
                answer=CYPHER_ERROR_RES,
                execution_time=execution_time,
                error="Could not generate valid cypher query"
            )
            return

        qa_step = StreamingQAStep(
            chat_session=self.qa_chat_session,
            qa_prompt=self.qa_prompt,
        )

        # Yield chunks of the response as they're generated
        for chunk in qa_step.run(message, cypher, context):
            yield chunk

        # Set the last answer using chat history to ensure we have the complete response
        final_answer = qa_step.chat_session.get_chat_history()[-1]['content']
        
        self.last_complete_response = ChatResponse(
            question=message,
            context=context,
            cypher=cypher,
            answer=final_answer,
            execution_time=execution_time
        )

    def search(self, message: str) -> ChatResponse:
        """
        Searches the knowledge graph by generating a Cypher query and extracting relevant context.
        This method only performs cypher generation and context extraction without Q&A.

        Args:
            message (str): The search query or question.

        Returns:
            ChatResponse: Search results with cypher and context, but no answer.
        """
        context, cypher = self._generate_cypher_query(message)
        execution_time = self.metadata.get("last_query_execution_time")
        
        # Handle query generation failure
        if not cypher:
            response = ChatResponse(
                question=message,
                context=None,
                cypher=None,
                answer=None,
                execution_time=execution_time,
                error="Could not generate valid cypher query"
            )
        else:
            response = ChatResponse(
                question=message,
                context=context,
                cypher=cypher,
                answer=None,  # No QA step for search
                execution_time=execution_time
            )
        
        # Update session state
        self.last_complete_response = response
        return response

def clean_ontology_for_prompt(ontology: dict) -> str:
    """
    Cleans the ontology by removing 'unique' and 'required' keys and prepares it for use in a prompt.

    Args:
        ontology (dict): The ontology to clean and transform.

    Returns:
        str: The cleaned ontology as a JSON string.
    """
    # Convert the ontology object to a JSON.
    ontology = ontology.to_json()
    
    # Remove unique and required attributes from the ontology.
    for entity in ontology["entities"]:
        for attribute in entity["attributes"]:
            del attribute['unique']
            del attribute['required']
    
    for relation in ontology["relations"]:
        for attribute in relation["attributes"]:
            del attribute['unique']
            del attribute['required']
    
    # Return the transformed ontology as a JSON string
    return json.dumps(ontology)