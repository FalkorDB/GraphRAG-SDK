from json import loads
from typing import Union
from .step_result import StepResult
from abc import ABC, abstractmethod


class StepBlockType:
    """
    Enum representing different types of step blocks in a plan.
    """
    PARALLEL = "parallel"
    AGENT = "agent"
    SUMMARY = "summary"
    USER_INPUT = "user_input"

    @staticmethod
    def from_str(text: str) -> "StepBlockType":
        """
        Convert a string to a corresponding StepBlockType enum value.
        
        Args:
            text (str): The string representation of the step block type.
            
        Returns:
            StepBlockType: The corresponding block type.
            
        Raises:
            ValueError: If the input string does not match any known step block type.
        """
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
    """
    Abstract base class for a PlanStep, which defines a step in the execution plan.
    Each subclass must implement the following properties and methods.
    """

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
    def from_json(json: Union[dict, str]) -> "PlanStep":
        """
        Factory method to create a PlanStep instance from a JSON object or string.
        
        Args:
            json (Union[dict, str]): The JSON representation of the step.
            
        Returns:
            PlanStep: The corresponding step instance.
            
        Raises:
            ValueError: If the block type is unknown or if step_type is None.
        """
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
