from abc import ABC, abstractmethod

class AgentResponseCode:
    """
    Represents the response codes for an agent.
    """

    AGENT_RESPONSE = "agent_response"
    AGENT_ERROR = "agent_error"
    AGENT_REQUEST_INPUT = "agent_request_input"

    @staticmethod
    def from_str(text: str) -> "AgentResponseCode":
        """
        Converts a string to an AgentResponseCode object.

        Args:
            text (str): The string representation of the response code.

        Returns:
            AgentResponseCode: The corresponding AgentResponseCode object.

        Raises:
            ValueError: If the input string does not match any known response code.
        """
        if text == AgentResponseCode.AGENT_RESPONSE:
            return AgentResponseCode.AGENT_RESPONSE
        elif text == AgentResponseCode.AGENT_ERROR:
            return AgentResponseCode.AGENT_ERROR
        elif text == AgentResponseCode.AGENT_REQUEST_INPUT:
            return AgentResponseCode.AGENT_REQUEST_INPUT
        else:
            raise ValueError(f"Unknown agent response code: {text}")


class AgentResponse:
    """
    Represents a response from an agent.

    Attributes:
        response_code (AgentResponseCode): The response code.
        payload (dict): The payload of the response.
    """

    def __init__(self, response_code: AgentResponseCode, payload: dict):
        """
        Initializes a new Agent object.

        Args:
            response_code (AgentResponseCode): The response code of the agent.
            payload (dict): The payload associated with the agent.

        """
        self.response_code = response_code
        self.payload = payload

    def to_json(self) -> dict:
        """
        Converts the AgentResponse object to a JSON-compatible dictionary.

        Returns:
            dict: The JSON representation of the AgentResponse object.
        """
        return {
            "response_code": self.response_code,
            "payload": self.payload,
        }

    @staticmethod
    def from_json(json: dict) -> "AgentResponse":
        """
        Creates an AgentResponse object from a JSON-compatible dictionary.

        Args:
            json (dict): The JSON representation of the AgentResponse object.

        Returns:
            AgentResponse: The created AgentResponse object.
        """
        return AgentResponse(
            AgentResponseCode.from_str(json["response_code"]),
            json["payload"],
        )

    def __str__(self) -> str:
        """
        Returns a string representation of the AgentResponse object.

        Returns:
            str: The string representation of the AgentResponse object.
        """
        return (
            f"AgentResponse(response_code={self.response_code}, payload={self.payload})"
        )

    def __repr__(self) -> str:
        """
        Returns a string representation of the AgentResponse object.

        Returns:
            str: The string representation of the AgentResponse object.
        """
        return str(self)


class Agent(ABC):
    """
    Abstract base class for agents in the system.
    """

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """
        Get the unique identifier of the agent.

        Returns:
            str: The agent's identifier.
        """
        pass

    @property
    @abstractmethod
    def introduction(self) -> str:
        """
        Get the introduction of the agent.

        Returns:
            str: The agent's introduction.
        """
        pass

    @property
    @abstractmethod
    def interface(self) -> list[dict]:
        """
        Get the interface of the agent.

        Returns:
            List[Dict]: The agent's interface.
        """
        pass

    @abstractmethod
    def run(self, params: dict) -> str:
        """
        Run the agent with the given parameters and chat session.

        Args:
            params (Dict): The parameters for the agent.

        Returns:
            str: The agent's response.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """
        Get a string representation of the agent.

        Returns:
            str: The string representation of the agent.
        """
        pass
