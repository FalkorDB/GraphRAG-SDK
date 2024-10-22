import graphrag_sdk.orchestrator.step
from graphrag_sdk.orchestrator.step_result import StepResult
from concurrent.futures import ThreadPoolExecutor, wait
from graphrag_sdk.orchestrator.orchestrator_runner import OrchestratorRunner
from graphrag_sdk.fixtures.prompts import ORCHESTRATOR_SUMMARY_PROMPT

import logging

logger = logging.getLogger(__name__)


class SummaryResult(StepResult):

    def __init__(self, output: str):
        self._output = output

    def to_json(self) -> dict:
        return {
            "output": self._output,
        }

    @staticmethod
    def from_json(json: dict) -> "SummaryResult":
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

    def __init__(self, id: str, properties: any):
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
        return {
            "id": self.id,
            "block": self.block,
            "properties": {},
        }

    @staticmethod
    def from_json(json: dict) -> "SummaryStep":
        return SummaryStep(json["id"], {})

    def __str__(self) -> str:
        return f"SummaryStep(id={self.id}, properties={self.properties})"

    def __repr__(self) -> str:
        return str(self)

    def run(
        self,
        runner: OrchestratorRunner,
    ) -> SummaryResult:
        response = runner.chat.send_message(
            ORCHESTRATOR_SUMMARY_PROMPT.replace(
                "#USER_QUESTION", str(runner.user_question)
            ).replace("#EXECUTION_LOG", str(runner.runner_log))
        )

        return SummaryResult(response.text)
