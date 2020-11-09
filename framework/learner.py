import zmq
from tensorflow.keras.optimizers import RMSprop

from new.dqn_agent import DQNAgent
from new.cnn_model import CNNModel
from new.protobuf.data import Data, bytes2arr
from new.atari import AtariEnv

if __name__ == '__main__':
    env = AtariEnv('PongNoFrameskip-v4', 4)

    dqn_agent = DQNAgent(
        env.get_observation_space(),
        env.get_action_space(),
        32,
        CNNModel
    )

    opt = RMSprop(learning_rate=0.0001)
    dqn_agent.policy_model.model.compile(loss='huber_loss', optimizer=opt)
    
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5000")

    batch_size = 32
    num_steps = 1000000
    start_steps = 10000
    update_freq = 1000

    weight = b''
    for step in range(num_steps):

        socket.send(weight)
        weight = b''

        data = Data()
        data.ParseFromString(socket.recv())
        state, next_state = bytes2arr(data.state), bytes2arr(data.next_state)
        dqn_agent.memorize(state, data.action, data.reward, next_state, data.done)

        if step > start_steps:
            dqn_agent.learn(batch_size)

            if step % update_freq == 0:
                dqn_agent.update_target_model()
