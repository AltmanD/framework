import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from framework import Model


class CNNModel(Model):
    def __init__(self, observation_space, action_space, conv_config=None, *args, **kwargs):
        self.conv_1 = Conv2D(16, 8, 4, activation='relu', input_shape=observation_space)
        self.conv_2 = Conv2D(32, 4, 2, activation='relu')
        self.flatten = Flatten()
        self.dense_1 = Dense(256, activation='relu')
        self.dense_2 = Dense(action_space, activation='linear')

        super(CNNModel, self).__init__('0', {}, *args, **kwargs)  # TODO: Config

        self.model = None

        self.build()

    def build(self):
        self.model = Sequential()
        self.model.add(self.conv_1)
        self.model.add(self.conv_2)
        self.model.add(self.flatten)
        self.model.add(self.dense_1)
        self.model.add(self.dense_2)

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)

    def set_weights(self, weights, *args, **kwargs):
        self.model.set_weights(weights)

    def get_weights(self, *args, **kwargs):
        return self.model.get_weights()

    def forward(self, states, *args, **kwargs):
        return self.model.predict(states)
