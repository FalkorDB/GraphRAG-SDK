from json import loads
from graphrag_sdk.orchestrator.step import PlanStep, StepBlockType


class ExecutionPlan:

    steps: list[PlanStep] = []

    def __init__(self, steps: list[PlanStep]):
        self.steps = steps

    @staticmethod
    def from_json(json: str | dict) -> "ExecutionPlan":
        if isinstance(json, str):
            json = loads(json)
        return ExecutionPlan([PlanStep.from_json(step) for step in json])

    def to_json(self) -> dict:
        return {"steps": [step.to_json() for step in self.steps]}

    def __str__(self) -> str:
        return f"ExecutionPlan(steps={self.to_json()})"
