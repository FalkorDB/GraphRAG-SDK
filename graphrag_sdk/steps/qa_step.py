import logging
from typing import Optional
from graphrag_sdk.steps.Step import Step
from graphrag_sdk.models import GenerativeModelChatSession


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class QAStep(Step):
    """
    QA Step
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

    def run(self, question: str, cypher: str, context: str) -> str:
        """
        Run the QA step.
        
        Args:
            question (str): The question being asked.
            cypher (str): The Cypher query to run.
            context (str): Context for the QA.
            
        Returns:
            str: The response from the QA session.
        """
        qa_prompt = self.qa_prompt.format(
            context=context, cypher=cypher, question=question
        )

        logger.debug(f"QA Prompt: {qa_prompt}")
        qa_response = self.chat_session.send_message(qa_prompt)

        return qa_response.text
