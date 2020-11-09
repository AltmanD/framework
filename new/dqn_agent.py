from typing import Any

import numpy as np
from tensorflow.keras.optimizers import RMSprop

from framework import Agent
from .cnn_model import CNNModel
from .replay_buffer import ReplayBuffer


class DQNAgent(Agent):
    def __init__(self, observation_space, action_space, batch_size, model_cls, model_cfg=None, epsilon=1,
                 epsilon_min=0.01, gamma=0.99, buffer_size=5000):
        self.observation_space = observation_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size

        self.memory = ReplayBuffer(buffer_size)
        self.policy_model: CNNModel = model_cls(observation_space, action_space)
        self.target_model: CNNModel = model_cls(observation_space, action_space)

        self.update_target_model()

        # Compile model
        opt = RMSprop(learning_rate=0.0001)
        self.policy_model.model.compile(loss='huber_loss', optimizer=opt)

        super(DQNAgent, self).__init__(self.policy_model)

    def learn(self, *args, **kwargs) -> None:
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        next_action = np.argmax(self.policy_model.forward(next_states), axis=-1)
        target = rewards + (1 - dones) * self.gamma * self.target_model.forward(next_states)[
            np.arange(self.batch_size), next_action]
        target_f = self.policy_model.forward(states)
        target_f[np.arange(self.batch_size), actions] = target
        self.policy_model.fit(states, target_f, epochs=1, verbose=1)

    def sample(self, state, *args, **kwargs):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_space)
        else:
            act_values = self.model.forward(state[np.newaxis])
            return np.argmax(act_values[0])

    def memorize(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def preprocess(self, state: Any, *args, **kwargs) -> Any:
        raise NotImplemented

    def update_target_model(self):
        self.target_model.set_weights(self.policy_model.get_weights())

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def adjust_ep(self, step):
        fraction = min(1.0, float(step) / self.time_steps)
        self.epsilon = 1 + fraction * (self.epsilon_min - 1)

    def save(self, name):
        self.policy_model.save_weights('save/{}'.format(name))

    def load(self, weight, filename):
        filename = '{}/{}'.format(self.save_dir, filename)
        with open(filename, 'wb') as f:
            f.write(weight)
        self.model.load_weights(filename)
