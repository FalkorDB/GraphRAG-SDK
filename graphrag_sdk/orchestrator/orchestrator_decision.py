from json import loads
from typing import Union, Optional
from graphrag_sdk.orchestrator.step import PlanStep


class OrchestratorDecisionCode:
    
    CONTINUE = "continue"
    END = "end"
    UPDATE_STEP = "update_step"

    @staticmethod
    def from_str(code: str) -> str:
        """
        Convert a string representation of a decision code to the corresponding constant.
        
        Args:
            code (str): The string representation of a decision code.
            
        Returns:
            str: The corresponding decision code.
            
        Raises:
            ValueError: If the code is unknown.
        """
        if code == OrchestratorDecisionCode.CONTINUE:
            return OrchestratorDecisionCode.CONTINUE
        elif code == OrchestratorDecisionCode.END:
            return OrchestratorDecisionCode.END
        elif code == OrchestratorDecisionCode.UPDATE_STEP:
            return OrchestratorDecisionCode.UPDATE_STEP
        else:
            raise ValueError(f"Unknown code: {code}")


class OrchestratorDecision:
    """
    Represents a decision made by the orchestrator.
    """
    
    def __init__(
        self, code: OrchestratorDecisionCode, new_step: Optional[PlanStep] = None
    ):
        """
        Initialize a new OrchestratorDecision object.
        
            Args:
            code (OrchestratorDecisionCode): The decision code.
            new_step (Optional[PlanStep]): The new step to execute, if applicable.
        """
        self.code = code
        self.new_step = new_step

    def to_json(self) -> dict:
        """
        Convert the orchestrator decision to a JSON-serializable dictionary.
        
        Returns:
            dict: A dictionary representation of the orchestrator decision.
        """
        return {
            "code": self.code,
            "new_step": self.new_step.to_json() if self.new_step else None,
        }

    @staticmethod
    def from_json(json: Union[dict, str]) -> "OrchestratorDecision":
        """
        Create an OrchestratorDecision instance from a JSON dictionary or string.
        
        Args:
            json (Union[dict, str]): The input dictionary or string containing decision data.
            
        Returns:
            OrchestratorDecision: An instance of OrchestratorDecision.
        """
        json = json if isinstance(json, dict) else loads(json)
        return OrchestratorDecision(
            OrchestratorDecisionCode.from_str(json["code"]),
            (
                PlanStep.from_json(json["new_step"])
                if "new_step" in json and json["new_step"]
                else None
            ),
        )

    def __str__(self) -> str:
        return f"OrchestratorDecision(code={self.code}, new_step={self.new_step})"

    def __repr__(self) -> str:
        return str(self)
