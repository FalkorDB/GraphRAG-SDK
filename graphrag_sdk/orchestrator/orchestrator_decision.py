from graphrag_sdk.orchestrator.step import PlanStep
from json import loads


class OrchestratorDecisionCode:
    CONTINUE = "continue"
    END = "end"
    UPDATE_STEP = "update_step"

    @staticmethod
    def from_str(code: str) -> str:
        if code == OrchestratorDecisionCode.CONTINUE:
            return OrchestratorDecisionCode.CONTINUE
        elif code == OrchestratorDecisionCode.END:
            return OrchestratorDecisionCode.END
        elif code == OrchestratorDecisionCode.UPDATE_STEP:
            return OrchestratorDecisionCode.UPDATE_STEP
        else:
            raise ValueError(f"Unknown code: {code}")


class OrchestratorDecision:
    def __init__(
        self, code: OrchestratorDecisionCode, new_step: PlanStep | None = None
    ):
        self.code = code
        self.new_step = new_step

    def to_json(self) -> dict:
        return {
            "code": self.code,
            "new_step": self.new_step.to_json() if self.new_step else None,
        }

    @staticmethod
    def from_json(json: dict | str) -> "OrchestratorDecision":
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
