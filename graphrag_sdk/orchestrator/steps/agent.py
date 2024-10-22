from graphrag_sdk.agents.agent import AgentResponseCode
from graphrag_sdk.orchestrator.step_result import StepResult
from graphrag_sdk.orchestrator.step import PlanStep, StepBlockType
from graphrag_sdk.orchestrator.orchestrator_runner import OrchestratorRunner
import logging

logger = logging.getLogger(__name__)


class AgentStepResult(StepResult):

    def __init__(self, response_code: AgentResponseCode, payload: dict):
        self.response_code = response_code
        self.payload = payload

    def to_json(self) -> dict:
        return {
            "response_code": self.response_code,
            "payload": self.payload,
        }

    @staticmethod
    def from_json(json: dict) -> "AgentStepResult":
        return AgentStepResult(
            AgentResponseCode.from_str(json["response_code"]),
            json["payload"],
        )

    def __str__(self) -> str:
        return f"AgentStepResult(response_code={self.response_code}, payload={self.payload})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def output(self) -> str:
        return self.payload.get("output", str(self))


class AgentProperties:

    def __init__(self, agent_id: str, session_id: str = None, payload: dict = None):
        self.agent_id = agent_id
        self.session_id = session_id
        self.payload = payload

    @staticmethod
    def from_json(json: dict) -> "AgentProperties":
        return AgentProperties(
            json["agent_id"],
            json.get("session_id", None),
            json.get("payload", None),
        )

    def to_json(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "payload": self.payload,
        }

    def __str__(self) -> str:
        return f"AgentProperties(agent_id={self.agent_id}, session_id={self.session_id}, payload={self.payload})"

    def __repr__(self) -> str:
        return str(self)


class AgentStep(PlanStep):

    def __init__(self, id: str, properties: AgentProperties):
        self._id = id
        self._properties = properties

    @property
    def id(self) -> str:
        return self._id

    @property
    def block(self) -> StepBlockType:
        return StepBlockType.AGENT

    @property
    def properties(self) -> AgentProperties:
        return self._properties

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "block": self.block,
            "properties": self.properties.to_json(),
        }

    @staticmethod
    def from_json(json: dict) -> "AgentStep":
        return AgentStep(json["id"], AgentProperties.from_json(json["properties"]))

    def __str__(self) -> str:
        return f"AgentStep(id={self.id}, properties={self.properties})"

    def __repr__(self) -> str:
        return str(self)

    def run(
        self,
        runner: "OrchestratorRunner",
    ) -> AgentStepResult:
        logger.info(f"Running agent {self.properties.agent_id}, step: {self.id}, payload: {self.properties.payload}")

        agent = runner.get_agent(self.properties.agent_id)
        if agent is None:
            raise ValueError(f"Agent with id {self.properties.agent_id} not found")

        response = agent.run(self.properties.payload)
        logger.debug(f"Agent response: {response}")
        return AgentStepResult(AgentResponseCode.AGENT_RESPONSE, {"output": response})
