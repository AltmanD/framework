from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .model import Model


class Agent(ABC):
    def __init__(self, model: Model, *args, **kwargs) -> None:
        """Set model, algorithm and other configurations"""
        self.model = model

    @abstractmethod
    def preprocess(self, state: Any, *args, **kwargs) -> Any:
        """Preprocess the game state"""
        pass

    @abstractmethod
    def save(self, *args, **kwargs) -> None:
        """Save the checkpoint file"""
        pass

    @abstractmethod
    def load(self, *args, **kwargs) -> None:
        """Load the checkpoint file"""
        pass

    @abstractmethod
    def learn(self, *args, **kwargs) -> None:
        """Train the agent"""
        pass

    def to_json(self):
        # TODO
        pass

    def predict(self, state: Any, *args, **kwargs):
        """Get the action distribution at specific state"""
        return self.model.forward(state, *args, **kwargs)

    def policy(self, state: Any, *args, **kwargs) -> Any:
        """Choose action during exploitation"""
        return np.argmax(self.predict(state, *args, **kwargs)[0])

    def sample(self, state: Any, *args, **kwargs) -> Any:
        """Choose action during exploration/sampling"""
        p = self.model.forward(state, *args, **kwargs)[0]
        return np.random.choice(len(p), p=p)
