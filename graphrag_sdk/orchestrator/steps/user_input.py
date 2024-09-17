from graphrag_sdk.orchestrator.step import PlanStep, StepBlockType
from graphrag_sdk.orchestrator.step_result import StepResult
from graphrag_sdk.orchestrator.orchestrator_runner import OrchestratorRunner
import logging

logger = logging.getLogger(__name__)


class UserInputResult(StepResult):

    def __init__(self, output: str):
        self._output = output

    def to_json(self) -> dict:
        return {
            "output": self._output,
        }

    @staticmethod
    def from_json(json: dict) -> "UserInputResult":
        return UserInputResult(
            json["output"],
        )

    def __str__(self) -> str:
        return f"UserInputResult(output={self._output})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def output(self) -> str:
        return self._output


class UserInputProperties:

    def __init__(self, question: str):
        self.question = question

    @staticmethod
    def from_json(json: dict) -> "UserInputProperties":
        return UserInputProperties(
            json["question"],
        )

    def to_json(self) -> dict:
        return {
            "question": self.question,
        }
    
    def __str__(self) -> str:
        return f"UserInputProperties(question={self.question})"
    
    def __repr__(self) -> str:
        return str(self)


class UserInputStep(PlanStep):

    def __init__(self, id: str, properties: UserInputProperties):
        self._id = id
        self._properties = properties

    @property
    def id(self) -> str:
        return self._id

    @property
    def block(self) -> StepBlockType:
        return StepBlockType.USER_INPUT

    @property
    def properties(self) -> UserInputProperties:
        return self._properties

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "block": self.block,
            "properties": self.properties.to_json(),
        }

    @staticmethod
    def from_json(json: dict) -> "UserInputStep":
        return UserInputStep(
            json["id"], UserInputProperties.from_json(json["properties"])
        )

    def __str__(self) -> str:
        return f"UserInputStep(id={self.id}, properties={self.properties})"

    def __repr__(self) -> str:
        return str(self)

    def run(
        self,
        runner: OrchestratorRunner,
        config: dict = None,
    ) -> UserInputResult:
        logger.info(f"Running user input step: {self.id}")
        return UserInputResult(runner.get_user_input(self.properties.question))
