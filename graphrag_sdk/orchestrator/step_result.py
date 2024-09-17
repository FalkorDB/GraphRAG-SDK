from abc import ABC, abstractmethod

class StepResult(ABC):

    @property
    @abstractmethod
    def output(self) -> str:
        pass

    @abstractmethod
    def to_json(self) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def from_json(json: dict) -> "StepResult":
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass
