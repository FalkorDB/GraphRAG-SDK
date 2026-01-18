from typing import Optional
import graphrag_sdk.orchestrator.step
from graphrag_sdk.orchestrator.step import PlanStep
from concurrent.futures import ThreadPoolExecutor, wait
from graphrag_sdk.orchestrator.step_result import StepResult
from graphrag_sdk.orchestrator.orchestrator_runner import OrchestratorRunner


class ParallelStepResult(StepResult):
    """
    Represents the result of executing parallel steps.
    """
    
    results: list[StepResult]

    def __init__(self, results: list[StepResult]):
        """
        Initializes a new ParallelStepResult object.
        
        Args:
            results (list[StepResult]): The results of the parallel steps.
        """
        self.results = results

    def to_json(self) -> dict:
        """
        Convert the parallel step result to a JSON-serializable dictionary.
        
        Returns:
            dict: A dictionary representation of the parallel step result.
        """
        return {"results": [result.to_json() for result in self.results]}

    @staticmethod
    def from_json(json: dict) -> "ParallelStepResult":
        """
        Create a ParallelStepResult instance from a JSON dictionary.
        
        Args:
            json (dict): The input JSON dictionary.
            
        Returns:
            ParallelStepResult: An instance of ParallelStepResult.
        """
        return ParallelStepResult(
            [
                StepResult.from_json(result)
                for result in json["results"]
            ]
        )

    def __str__(self) -> str:
        return f"ParallelStepResult(results={self.results})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def output(self) -> str:
        return "\n".join([result.output for result in self.results])


class ParallelProperties:
    """
    Represents properties for parallel execution of steps.
    """
    
    steps: list["PlanStep"]

    def __init__(self, steps: list["PlanStep"]):
        """
        Initializes a new ParallelProperties object.
        
        Args:
            steps (list[PlanStep]): The list of steps to execute in parallel.
        """
        self.steps = steps

    @staticmethod
    def from_json(json: dict) -> "ParallelProperties":
        """
        Create ParallelProperties from a JSON dictionary.
        
        Args:
            json (dict): The input JSON dictionary.
            
        Returns:
            ParallelProperties: An instance of ParallelProperties.
        """
        return ParallelProperties(
            [
                graphrag_sdk.orchestrator.step.PlanStep.from_json(step)
                for step in (json if isinstance(json, list) else json["steps"])
            ]
        )

    def to_json(self) -> dict:
        """
        Convert the parallel properties to a JSON-serializable dictionary.
        
        Returns:
            dict: A dictionary representation of the parallel properties.
        """
        return {"steps": [step.to_json() for step in self.steps]}
    
    def __str__(self) -> str:
        return f"ParallelProperties(steps={self.steps})"
    
    def __repr__(self) -> str:
        return str(self)


class ParallelStep(graphrag_sdk.orchestrator.step.PlanStep):
    """
    Represents a step that executes multiple sub-steps in parallel.
    """

    def __init__(self, id: str, properties: ParallelProperties):
        """
        Initializes a new ParallelStep object.
        
        Args:
            id (str): The identifier for the step.
            properties (ParallelProperties): The properties of the parallel step.
        """
        self._id = id
        self._properties = properties

    @property
    def id(self) -> str:
        return self._id

    @property
    def block(self) -> graphrag_sdk.orchestrator.step.StepBlockType:
        return graphrag_sdk.orchestrator.step.StepBlockType.PARALLEL

    @property
    def properties(self) -> ParallelProperties:
        return self._properties

    def to_json(self) -> dict:
        """
        Convert the parallel step to a JSON-serializable dictionary.
        
        Returns:
            dict: A dictionary representation of the parallel step.
        """
        return {
            "id": self.id,
            "block": self.block,
            "properties": self.properties.to_json(),
        }

    @staticmethod
    def from_json(json: dict) -> "ParallelStep":
        """
        Create a ParallelStep from a JSON dictionary.
        
        Args:
            json (dict): The input JSON dictionary.
            
        Returns:
            ParallelStep: An instance of ParallelStep.
        """
        return ParallelStep(
            json["id"], ParallelProperties.from_json(json["properties"])
        )

    def __str__(self) -> str:
        return f"ParallelStep(id={self.id}, properties={self.properties})"

    def __repr__(self) -> str:
        return str(self)

    def run(
        self, runner: OrchestratorRunner, config: Optional[dict] = None
    ) -> ParallelStepResult:
        """
        Run the parallel step, executing sub-steps concurrently.
        
        Args:
            runner (OrchestratorRunner): The orchestrator runner instance.
            config (Optional[dict]): Configuration options. Defaults to None.
            
        Returns:
            ParallelStepResult: The result of the parallel step execution.
        """
        config = config or {"parallel_max_workers": 16}
        tasks = []
        with ThreadPoolExecutor(
            max_workers=min(config["parallel_max_workers"], len(self.properties.steps))
        ) as executor:
            for sub_step in self.properties.steps:
                tasks.append(executor.submit(sub_step.run, runner))

        wait(tasks)

        return ParallelStepResult([task.result() for task in tasks])
