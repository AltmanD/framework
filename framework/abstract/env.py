from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Env(ABC):
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def step(self, action: Any, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def get_action_space(self) -> Any:
        pass

    @abstractmethod
    def get_observation_space(self) -> Any:
        pass

    @abstractmethod
    def calc_reward(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def render(self, *args, **kwargs) -> None:
        pass
