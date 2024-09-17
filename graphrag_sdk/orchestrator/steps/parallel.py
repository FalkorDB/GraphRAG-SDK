import graphrag_sdk.orchestrator.step
from concurrent.futures import ThreadPoolExecutor, wait
from graphrag_sdk.orchestrator.step_result import StepResult
from graphrag_sdk.orchestrator.orchestrator_runner import OrchestratorRunner


class ParallelStepResult(StepResult):
    results: list[StepResult]

    def __init__(self, results: list[StepResult]):
        self.results = results

    def to_json(self) -> dict:
        return {"results": [result.to_json() for result in self.results]}

    @staticmethod
    def from_json(json: dict) -> "ParallelStepResult":
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
    steps: list["PlanStep"]

    def __init__(self, steps: list["PlanStep"]):
        self.steps = steps

    @staticmethod
    def from_json(json: dict) -> "ParallelProperties":
        return ParallelProperties(
            [
                graphrag_sdk.orchestrator.step.PlanStep.from_json(step)
                for step in (json if isinstance(json, list) else json["steps"])
            ]
        )

    def to_json(self) -> dict:
        return {"steps": [step.to_json() for step in self.steps]}
    
    def __str__(self) -> str:
        return f"ParallelProperties(steps={self.steps})"
    
    def __repr__(self) -> str:
        return str(self)


class ParallelStep(graphrag_sdk.orchestrator.step.PlanStep):

    def __init__(self, id: str, properties: ParallelProperties):
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
        return {
            "id": self.id,
            "block": self.block,
            "properties": self.properties.to_json(),
        }

    @staticmethod
    def from_json(json: dict) -> "ParallelStep":
        return ParallelStep(
            json["id"], ParallelProperties.from_json(json["properties"])
        )

    def __str__(self) -> str:
        return f"ParallelStep(id={self.id}, properties={self.properties})"

    def __repr__(self) -> str:
        return str(self)

    def run(
        self, runner: OrchestratorRunner, config: dict = None
    ) -> ParallelStepResult:
        config = config or {"parallel_max_workers": 16}
        tasks = []
        with ThreadPoolExecutor(
            max_workers=min(config["parallel_max_workers"], len(self.properties.steps))
        ) as executor:
            for sub_step in self.properties.steps:
                tasks.append(executor.submit(sub_step.run, runner))

        wait(tasks)

        return ParallelStepResult([task.result() for task in tasks])
