import logging
from graphrag_sdk.orchestrator.step import PlanStep, StepBlockType
from graphrag_sdk.orchestrator.step_result import StepResult
from graphrag_sdk.orchestrator.orchestrator_runner import OrchestratorRunner


logger = logging.getLogger(__name__)

class UserInputResult(StepResult):
    """
    Represents the result of a user input step.
    """

    def __init__(self, output: str):
        """
        Initializes a new UserInputResult object.
        
        Args:
            output (str): The user's input.
        """
        self._output = output

    def to_json(self) -> dict:
        """
        Convert the user input result to a JSON-serializable dictionary.
        
        Returns:
            Dict: A dictionary representation of the user input result.
        """
        return {
            "output": self._output,
        }

    @staticmethod
    def from_json(json: dict) -> "UserInputResult":
        """
        Create a UserInputResult instance from a JSON dictionary.
        
        Args:
            json (Dict): The input JSON dictionary.
            
        Returns:
            UserInputResult: An instance of UserInputResult.
        """
        return UserInputResult(
            json["output"],
        )

    def __str__(self) -> str:
        return f"UserInputResult(output={self._output})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def output(self) -> str:
        return self._output


class UserInputProperties:
    """
    Represents the properties required for a user input step.
    """

    def __init__(self, question: str):
        """
        Initialize a new UserInputProperties object.
        
        Args:
            question (str): The question to prompt the user.
        """
        self.question = question

    @staticmethod
    def from_json(json: dict) -> "UserInputProperties":
        """
        Create UserInputProperties from a JSON dictionary.
        
        Args:
            json (Dict): The input JSON dictionary.
            
        Returns:
            UserInputProperties: An instance of UserInputProperties.
        """
        return UserInputProperties(
            json["question"],
        )

    def to_json(self) -> dict:
        """
        Convert the user input properties to a JSON-serializable dictionary.
        
        Returns:
            Dict: A dictionary representation of the user input properties.
        """
        return {
            "question": self.question,
        }
    
    def __str__(self) -> str:
        return f"UserInputProperties(question={self.question})"
    
    def __repr__(self) -> str:
        return str(self)


class UserInputStep(PlanStep):
    """
    Represents a step that requires user input.
    """

    def __init__(self, id: str, properties: UserInputProperties):
        """
        Initializes a new UserInputStep object.
        
        Args:
            id (str): The identifier for the step.
            properties (UserInputProperties): The properties of the user input step.
        """
        self._id = id
        self._properties = properties

    @property
    def id(self) -> str:
        return self._id

    @property
    def block(self) -> StepBlockType:
        return StepBlockType.USER_INPUT

    @property
    def properties(self) -> UserInputProperties:
        return self._properties

    def to_json(self) -> dict:
        """
        Convert the user input step to a JSON-serializable dictionary.
        
        Returns:
            dict: A dictionary representation of the user input step.
        """
        return {
            "id": self.id,
            "block": self.block,
            "properties": self.properties.to_json(),
        }

    @staticmethod
    def from_json(json: dict) -> "UserInputStep":
        """
        Create a UserInputStep from a JSON dictionary.
        
        Args:
            json (dict): The input JSON dictionary.
            
        Returns:
            UserInputStep: An instance of UserInputStep.
        """
        return UserInputStep(
            json["id"], UserInputProperties.from_json(json["properties"])
        )

    def __str__(self) -> str:
        return f"UserInputStep(id={self.id}, properties={self.properties})"

    def __repr__(self) -> str:
        return str(self)

    def run(
        self,
        runner: OrchestratorRunner,
    ) -> UserInputResult:
        """
        Run the user input step, prompting the user for input.
        
        Args:
            runner (OrchestratorRunner): The orchestrator runner instance.
            
        Returns:
            UserInputResult: The result of the user input step.
        """
        logger.info(f"Running user input step: {self.id}")
        return UserInputResult(runner.get_user_input(self.properties.question))
