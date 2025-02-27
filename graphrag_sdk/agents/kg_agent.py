from .agent import Agent
from graphrag_sdk.kg import KnowledgeGraph


class KGAgent(Agent):
    """Represents an Agent for a FalkorDB Knowledge Graph.

    Args:
        agent_id (str): The ID of the agent.
        kg (KnowledgeGraph): The knowledge graph to query.
        introduction (str): The introduction to the agent.

    Examples:
        >>> from graphrag_sdk import KnowledgeGraph, Orchestrator
        >>> from graphrag_sdk.agents.kg_agent import KGAgent
        >>> orchestrator = Orchestrator(model)
        >>> kg = KnowledgeGraph("test_kg", ontology, model)
        >>> agent = KGAgent("test_agent", kg, "This is a test agent.")
        >>> orchestrator.register_agent(agent)

    """

    _interface = [
        {
            "name": "prompt",
            "type": "string",
            "required": True,
            "description": "The prompt to ask the agent.",
        }
    ]

    def __init__(self, agent_id: str, kg: KnowledgeGraph, introduction: str):
        """
        Initializes a KGAgent object.

        Args:
            agent_id (str): The ID of the agent.
            kg (KnowledgeGraph): The knowledge graph associated with the agent.
            introduction (str): The introduction of the agent.
        """
        super().__init__()
        self.agent_id = agent_id
        self.introduction = introduction
        self.kg = kg
        self.chat_session = self._kg.chat_session()

    @property
    def agent_id(self) -> str:
        """
        Returns the ID of the agent.

        Returns:
            str: The ID of the agent.
        """
        return self._agent_id

    @agent_id.setter
    def agent_id(self, value) -> None:
        """
        Sets the agent ID.

        Args:
            value (str): The ID of the agent.
        """
        self._agent_id = value

    @property
    def introduction(self) -> str:
        """
        Returns the introduction of the agent.

        Returns:
            str: The introduction of the agent.
        """
        return self._introduction

    @introduction.setter
    def introduction(self, value) -> None:
        """
        Sets the introduction of the agent.

        Args:
            value (str): The introduction of the agent.
        """
        self._introduction = value

    @property
    def interface(self) -> list[dict]:
        """
        Returns the interface of the KG agent.

        Returns:
            list[dict]: The interface of the KG agent.
        """
        return self._interface

    @property
    def kg(self) -> KnowledgeGraph:
        """
        Returns the KnowledgeGraph object associated with this agent.

        Returns:
            KnowledgeGraph: The KnowledgeGraph object.
        """
        return self._kg

    @kg.setter
    def kg(self, value: KnowledgeGraph) -> None:
        """
        Sets the knowledge graph for the agent.

        Args:
            value (KnowledgeGraph): The knowledge graph to be set.
        """
        self._kg = value

    def run(self, params: dict) -> str:
        """
        Ask the agent a question.

        Args:
            params (dict): The parameters for the agent.

        Returns:
            str: The agent's response.

        """
        output = self.chat_session.send_message(params["prompt"])
        return output['response']

    def __repr__(self) -> str:
        """
        Returns a string representation of the KGAgent object.

        The string representation includes the Agent ID, Knowledge Graph Name,
        Interface, and Introduction.

        Returns:
            str: A string representation of the KGAgent object.
        """
        return f"""
---
Agent ID: {self.agent_id}
Knowledge Graph Name: {self._kg.name}
Interface: {self.interface}

Introduction: {self.introduction}
"""
