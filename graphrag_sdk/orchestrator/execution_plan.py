from json import loads
from typing import Union
<<<<<<< HEAD
from graphrag_sdk.orchestrator.step import PlanStep, StepBlockType
=======
from graphrag_sdk.orchestrator.step import PlanStep
>>>>>>> b2aa07fc70e298ca25ae07c67c1e8af35dd2953b


class ExecutionPlan:
    """
    Represents an execution plan consisting of a sequence of steps.
    """
    
    steps: list[PlanStep] = []

    def __init__(self, steps: list[PlanStep]):
        """
        Args:
            steps (List[PlanStep]): The list of steps in the execution plan.
        """
        self.steps = steps

    @staticmethod
    def from_json(json: Union[str, dict]) -> "ExecutionPlan":
<<<<<<< HEAD
=======
        """
        Create an ExecutionPlan instance from a JSON dictionary or string.
        
        Args:
            json (Union[str, dict]): The input JSON string or dictionary.
            
        Returns:
            ExecutionPlan: An instance of ExecutionPlan.
        """
>>>>>>> b2aa07fc70e298ca25ae07c67c1e8af35dd2953b
        if isinstance(json, str):
            json = loads(json)
        return ExecutionPlan([PlanStep.from_json(step) for step in json])

    def to_json(self) -> dict:
        """
        Convert the execution plan to a JSON-serializable dictionary.
        
        Returns:
            dict: A dictionary representation of the execution plan.
        """
        return {"steps": [step.to_json() for step in self.steps]}

    def __str__(self) -> str:
        return f"ExecutionPlan(steps={self.to_json()})"
