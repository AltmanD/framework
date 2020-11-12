import sys 
sys.path.append("..") 

import numpy as np
import zmq

from new.atari import AtariEnv
from new.cnn_model import CNNModel
from new.dqn_agent import DQNAgent
from new.protobuf.data import Data, arr2bytes

if __name__ == '__main__':

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5000")

    env = AtariEnv('PongNoFrameskip-v4', 4)

    dqn_agent = DQNAgent(
        env.get_observation_space(),
        env.get_action_space(),
        32,
        CNNModel
    )

    num_steps = 1000000
    episode_rewards = [0.0]

    state = env.reset()
    for step in range(num_steps):

        weights = socket.recv()
        if len(weights):
            dqn_agent.set_weights(weights)

        # Adjust Epsilon
        fraction = min(1.0, float(step) / num_steps)
        dqn_agent.epsilon = 1 + fraction * (dqn_agent.epsilon_min - 1)

        action = dqn_agent.sample(state)
        next_state, reward, done, info = env.step(action)

        data = Data(state=arr2bytes(state), next_state=arr2bytes(next_state), action=int(action),
                    reward=reward, done=done, epoch=step)
        socket.send(data.SerializeToString())

        state = next_state
        episode_rewards[-1] += reward

        if done:
            num_episodes = len(episode_rewards)
            mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 2)

            print('episode: {}, step: {}/{}, mean reward: {}, epsilon: {:.3f}'.format(
                num_episodes, step + 1, num_steps, mean_100ep_reward, dqn_agent.epsilon))

            state = env.reset()
            episode_rewards.append(0.0)
