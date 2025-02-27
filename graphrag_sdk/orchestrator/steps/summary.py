import logging
import graphrag_sdk.orchestrator.step
from graphrag_sdk.orchestrator.step_result import StepResult
from graphrag_sdk.orchestrator.orchestrator_runner import OrchestratorRunner
from graphrag_sdk.fixtures.prompts import ORCHESTRATOR_SUMMARY_PROMPT

logger = logging.getLogger(__name__)


class SummaryResult(StepResult):
    """
    Represents the result of a summary step.
    """

    def __init__(self, output: str):
        """
        Initializes a new SummaryResult object.
        
        Args:
            output (str): The summary output.
        """
        self._output = output

    def to_json(self) -> dict:
        """
        Convert the summary result to a JSON-serializable dictionary.
        
        Returns:
            dict: A dictionary representation of the summary result.
        """
        return {
            "output": self._output,
        }

    @staticmethod
    def from_json(json: dict) -> "SummaryResult":
        """
        Create a SummaryResult instance from a JSON dictionary.
        
        Args:
            json (dict): The input JSON dictionary.
            
        Returns:
            SummaryResult: An instance of SummaryResult.
        """
        
        return SummaryResult(
            json["output"],
        )

    def __str__(self) -> str:
        return f"SummaryResult(output={self._output})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def output(self) -> str:
        return self._output


class SummaryStep(graphrag_sdk.orchestrator.step.PlanStep):
    """
    Represents a step that generates a summary.
    """

    def __init__(self, id: str, properties: any):
        """
        Initializes a new SummaryStep object.
        
        Args:
            id (str): The identifier for the step.
            properties (any): The properties of the summary step.
        """
        self._id = id
        self._properties = {}

    @property
    def id(self) -> str:
        return self._id

    @property
    def block(self) -> graphrag_sdk.orchestrator.step.StepBlockType:
        return graphrag_sdk.orchestrator.step.StepBlockType.SUMMARY

    @property
    def properties(self) -> any:
        return self._properties

    def to_json(self) -> dict:
        """
        Convert the summary step to a JSON-serializable dictionary.
        
        Returns:
            dict: A dictionary representation of the summary step.
        """
        return {
            "id": self.id,
            "block": self.block,
            "properties": {},
        }

    @staticmethod
    def from_json(json: dict) -> "SummaryStep":
        """
        Create a SummaryStep from a JSON dictionary.
        
        Args:
            json (dict): The input JSON dictionary.
            
        Returns:
            SummaryStep: An instance of SummaryStep.
        """
        return SummaryStep(json["id"], {})

    def __str__(self) -> str:
        return f"SummaryStep(id={self.id}, properties={self.properties})"

    def __repr__(self) -> str:
        return str(self)

    def run(
        self,
        runner: OrchestratorRunner,
    ) -> SummaryResult:
        """
        Run the summary step, generating a summary based on execution logs.
        
        Args:
            runner (OrchestratorRunner): The orchestrator runner instance.
            
        Returns:
            SummaryResult: The result of the summary step.
        """
        response = runner.chat.send_message(
            ORCHESTRATOR_SUMMARY_PROMPT.replace(
                "#USER_QUESTION", str(runner.user_question)
            ).replace("#EXECUTION_LOG", str(runner.runner_log))
        )

        return SummaryResult(response.text)
