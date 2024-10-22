from graphrag_sdk.agents import Agent
from graphrag_sdk.models import GenerativeModelChatSession
from .execution_plan import ExecutionPlan
from .step import StepResult, PlanStep, StepBlockType
from .orchestrator_decision import OrchestratorDecision, OrchestratorDecisionCode
from graphrag_sdk.fixtures.prompts import ORCHESTRATOR_DECISION_PROMPT
from graphrag_sdk.helpers import extract_json
import logging

logger = logging.getLogger(__name__)


class OrchestratorResult(StepResult):

    def __init__(self, output: str):
        self._output = output

    def to_json(self) -> dict:
        return {
            "output": self._output,
        }

    @staticmethod
    def from_json(json: dict) -> "OrchestratorResult":
        return OrchestratorResult(
            json["output"],
        )

    def __str__(self) -> str:
        return f"OrchestratorResult(output={self._output})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def output(self) -> str:
        return self._output


class OrchestratorRunner:
    _runner_log: list[tuple[PlanStep, StepResult]] = []
    _agent_sessions: dict[str, GenerativeModelChatSession] = {}

    def __init__(
        self,
        chat: GenerativeModelChatSession,
        agents: list[Agent],
        plan: ExecutionPlan,
        user_question: str = "",
        config: dict = None,
    ):
        self._chat = chat
        self._agents = agents
        self._plan = plan
        self._user_question = user_question
        self._config = config or {"parallel_max_workers": 16}
        self._runner_log = []
        self._agent_sessions = {}

    @property
    def plan(self) -> ExecutionPlan:
        return self._plan

    @property
    def chat(self) -> GenerativeModelChatSession:
        return self._chat

    @property
    def runner_log(self) -> list[tuple[PlanStep, StepResult]]:
        return self._runner_log

    @property
    def user_question(self) -> str:
        return self._user_question

    def get_agent(self, agent_id: str) -> Agent:
        return next(agent for agent in self._agents if agent.agent_id == agent_id)

    def get_session(self, session_id: str) -> GenerativeModelChatSession | None:
        return self._agent_sessions.get(session_id, None)

    def set_session(self, session_id: str, session: GenerativeModelChatSession):
        self._agent_sessions[session_id] = session

    def get_user_input(self, question: str) -> str:
        return input(question)

    def run(self) -> OrchestratorResult:

        first_step = self._plan.steps[0] if len(self._plan.steps) > 0 else None

        if first_step is None:
            return OrchestratorResult("No steps to run")

        first_step_result = first_step.run(self)

        self._runner_log.append((first_step, first_step_result))

        loop_response = self._run_loop(self._plan.steps[1:])

        logger.info(f"Execution log: {self._runner_log}")
        logger.info(f"Execution result: {loop_response}")

        return loop_response

    def _run_loop(self, steps: list[PlanStep]) -> StepResult:
        decision = self._get_orchestrator_decision(
            steps[0] if len(steps) > 0 else None,
        )

        if decision.code == OrchestratorDecisionCode.END:
            return self._handle_end_decision()

        if decision.code == OrchestratorDecisionCode.CONTINUE:
            return self._handle_continue_decision(steps)

        if decision.code == OrchestratorDecisionCode.UPDATE_STEP:
            return self._handle_update_step_decision(decision.new_step)

    def _handle_end_decision(self) -> StepResult:
        last_step = self._runner_log[-1][0] if len(self._runner_log) > 0 else None
        if last_step is None:
            return OrchestratorResult("No steps to run")
        if last_step.block != StepBlockType.SUMMARY:
            return self._call_summary_step()
        last_result = self._runner_log[-1][1] if len(self._runner_log) > 0 else None

        return (
            OrchestratorResult(last_result.output)
            if last_result
            else OrchestratorResult("No steps to run")
        )

    def _handle_continue_decision(self, steps: list[PlanStep]) -> StepResult:
        if len(steps) == 0:
            return self._handle_end_decision()

        next_step = steps[0]
        next_step_result = next_step.run(self)
        self._runner_log.append((next_step, next_step_result))
        return self._run_loop(steps[1:])

    def _handle_update_step_decision(self, new_step: PlanStep) -> StepResult:
        next_step_result = new_step.run(self)
        self._runner_log.append((new_step, next_step_result))
        return self._run_loop([])

    def _call_summary_step(self):
        return self._run_loop(
            [
                PlanStep.from_json(
                    {
                        "block": StepBlockType.SUMMARY,
                        "id": "summary",
                        "properties": {},
                    }
                )
            ]
        )

    def _get_orchestrator_decision(
        self,
        next_step: PlanStep | None = None,
    ) -> OrchestratorDecision:

        response = self.chat.send_message(
            ORCHESTRATOR_DECISION_PROMPT.replace(
                "#LOG_HISTORY",
                str(self._runner_log),
            ).replace(
                "#NEXT_STEP",
                str(next_step),
            )
        )

        logger.debug(f"Orchestrator decision response: {response.text}")

        return OrchestratorDecision.from_json(extract_json(response.text))
