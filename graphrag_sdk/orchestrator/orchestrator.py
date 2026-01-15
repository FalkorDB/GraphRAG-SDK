import logging
from typing import Optional
from graphrag_sdk.agents import Agent
from graphrag_sdk.helpers import extract_json
from graphrag_sdk.models import GenerativeModel, GenerativeModelChatSession
from .orchestrator_runner import OrchestratorRunner, OrchestratorResult
from .execution_plan import (
    ExecutionPlan,
)
from graphrag_sdk.fixtures.prompts import (
    ORCHESTRATOR_SYSTEM,
    ORCHESTRATOR_EXECUTION_PLAN_PROMPT,
)


logger = logging.getLogger(__name__)

class Orchestrator:
    """
    The Orchestrator class is responsible for managing agents and generating an execution plan 
    based on the user's input question. It interacts with a generative model to produce the plan 
    and assigns agents to execute tasks.
    """

    def __init__(self, model: GenerativeModel, backstory: Optional[str] = ""):
        """
        Initialize the Orchestrator with a generative model and an optional backstory.
        
        Args:
            model (GenerativeModel): The model that powers the orchestration process.
            backstory (Optional[str]): Optional backstory or context to be included in the orchestration system.
        """
        self._model = model
        self._backstory = backstory
        self._agents = []
        self._chat = None
        
    def _get_chat(self) -> GenerativeModelChatSession:
        """
        Internal method to get or initialize a chat session with the model.
        
        Returns:
            GenerativeModelChatSession: The chat session used for communication with the model.
        """
        if self._chat is None:
            self._chat = self._model.start_chat(
                ORCHESTRATOR_SYSTEM.replace("#BACKSTORY", self._backstory).replace(
                    "#AGENTS",
                    ",".join([str(agent) for agent in self._agents]))
            )

        return self._chat

    def register_agent(self, agent: Agent):
        self._agents.append(agent)

    def ask(self, question: str) -> OrchestratorResult:
        """
        Ask the orchestrator a question and run the corresponding execution plan.
        
        Args:
            question (str): The user's question.
            
        Returns:
            OrchestratorResult: The result of executing the plan.
        """
        return self.runner(question).run()

    def runner(self, question: str) -> OrchestratorRunner:
        """
        Create an OrchestratorRunner to execute the plan based on the user's question.
        
        Args:
            question (str): The user's input question.
            
        Returns:
            OrchestratorRunner: A runner that will handle the execution of the plan.
        """
        plan = self._create_execution_plan(question)

        return OrchestratorRunner(
            self._get_chat(), self._agents, plan, user_question=question
        )

    def _create_execution_plan(self, question: str) -> ExecutionPlan:
        """
        Generate an execution plan based on the user's question by interacting with the model.
        
        Args:
            question (str): The question or prompt for which the execution plan is generated.
            
        Returns:
            ExecutionPlan: The generated execution plan.
            
        Raises:
            Exception: If the plan generation fails.
        """
        try:
            response = self._get_chat().send_message(
                ORCHESTRATOR_EXECUTION_PLAN_PROMPT.replace("#QUESTION", question)
            )

            logger.debug(f"Execution plan response: {response.text}")

            plan = ExecutionPlan.from_json(
                extract_json(response.text, skip_repair=True)
            )

            logger.debug(f"Execution plan: {plan}")

            return plan
        except Exception as e:
            logger.error(f"Failed to create plan: {e}")
            raise e
