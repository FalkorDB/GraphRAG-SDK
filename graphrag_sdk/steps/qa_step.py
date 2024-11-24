import logging
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
        config: dict = None,
        qa_prompt: str = None,
    ) -> None:
        self.config = config or {}
        self.chat_session = chat_session
        self.qa_prompt = qa_prompt

    def run(self, question: str, cypher: str, context: str):
        qa_prompt = self.qa_prompt.format(
            context=context, cypher=cypher, question=question
        )

        logger.debug(f"QA Prompt: {qa_prompt}")
        qa_response = self.chat_session.send_message(qa_prompt)

        return qa_response.text
