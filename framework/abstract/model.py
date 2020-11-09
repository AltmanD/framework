from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import json

import tensorflow as tf


class Model(ABC):
    def __init__(self, model_id: str, config: dict, *args, **kwargs) -> None:
        """
        1. Define layers and tensors
        2. Set other configurations
        :param model_id: The identifier of the model
        :param config: The configuration of hyper-parameters
        """
        self.model_id = model_id
        self.config = config

    @abstractmethod
    def build(self, *args, **kwargs) -> None:
        """Build the computational graph"""
        pass

    @abstractmethod
    def set_weights(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def get_weights(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def forward(self, states: Any, *args, **kwargs) -> Any:
        pass

    def to_json(self, save_path: str = '.') -> None:
        """Convert hyper-parameters to JSON file"""
        with open(Path(save_path) / f'config-{self.model_id}.json') as f:
            json.dump(self.config, f)

    def __call__(self, *args, **kwargs) -> Any:
        with tf.get_default_session() as sess:
            return self.forward(sess, *args, **kwargs)
