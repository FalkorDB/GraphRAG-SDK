from abc import ABC, abstractmethod
from .step_result import StepResult
from json import loads

class StepBlockType:
    PARALLEL = "parallel"
    AGENT = "agent"
    SUMMARY = "summary"
    USER_INPUT = "user_input"

    @staticmethod
    def from_str(text: str) -> "StepBlockType":
        if text == StepBlockType.PARALLEL:
            return StepBlockType.PARALLEL
        elif text == StepBlockType.AGENT:
            return StepBlockType.AGENT
        elif text == StepBlockType.SUMMARY:
            return StepBlockType.SUMMARY
        elif text == StepBlockType.USER_INPUT:
            return StepBlockType.USER_INPUT
        else:
            raise ValueError(f"Unknown step block type: {text}")


class PlanStep(ABC):

    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @property
    @abstractmethod
    def block(self) -> StepBlockType:
        pass

    @property
    @abstractmethod
    def properties(self) -> any:
        pass

    @staticmethod
    def from_json(json: dict| str) -> "PlanStep":
        json =  json if isinstance(json, dict) else loads(json)
        from graphrag_sdk.orchestrator.steps import PLAN_STEP_TYPE_MAP
        block = StepBlockType.from_str(json["block"])
        step_type = PLAN_STEP_TYPE_MAP[block]

        if step_type is None:
            raise ValueError(f"Unknown step block type: {block}")
 
        return step_type.from_json(json)

    @abstractmethod
    def run(
        self,
        runner: any,
    ) -> StepResult:
        pass
