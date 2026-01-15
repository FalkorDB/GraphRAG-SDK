import logging
from typing import Optional, Iterator
from graphrag_sdk.steps.Step import Step
from graphrag_sdk.models import GenerativeModelChatSession


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class StreamingQAStep(Step):
    """
    QA Step that supports streaming responses
    """

    def __init__(
        self,
        chat_session: GenerativeModelChatSession,
        config: Optional[dict] = None,
        qa_prompt: Optional[str] = None,
    ) -> None:
        """
        Initialize the QA Step.
        
        Args:
            chat_session (GenerativeModelChatSession): The chat session for handling the QA.
            config (Optional[dict]): Optional configuration for the step.
            qa_prompt (Optional[str]): The prompt template for question answering.
        """
        self.config = config or {}
        self.chat_session = chat_session
        self.qa_prompt = qa_prompt

    def run(self, question: str, cypher: str, context: str) -> Iterator[str]:
        """
        Run the QA step and stream the response chunks.
        
        Args:
            question (str): The question being asked.
            cypher (str): The Cypher query to run.
            context (str): Context for the QA.
            
        Returns:
            Iterator[str]: A generator that yields response chunks.
        """
        qa_prompt = self.qa_prompt.format(
            context=context, cypher=cypher, question=question
        )
        logger.debug(f"QA Prompt: {qa_prompt}")
        # Send the message and stream the response
        for chunk in self.chat_session.send_message_stream(qa_prompt):
            yield chunk