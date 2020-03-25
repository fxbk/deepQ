from collections import deque

import numpy as np
import tensorflow as tf
import gym

def dense(x, weights, bias, activation=tf.identity, **activation_kwargs):
    """Dense layer."""
    try:
        x = x.astype('float32')
    except:
        pass
    z = tf.matmul(x, weights) + bias

    return activation(z) # , **activationn_kwargs)


def init_weights(shape, initializer):
    """Initialize weights for tensorflow layer."""
    weights = tf.Variable(
        initializer(shape),
        trainable=True,
        dtype=tf.float32
    )

    return weights


class Network(object):
    """Q-function approximator."""

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size=[24, 24],
                 weights_initializer=tf.initializers.glorot_uniform(),
                 bias_initializer=tf.initializers.zeros(),
                 optimizer=tf.keras.optimizers.Adam,
                 **optimizer_kwargs):
        """Initialize weights and hyperparameters."""
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        np.random.seed(41)

        self.initialize_weights(weights_initializer, bias_initializer)
        self.optimizer = optimizer(**optimizer_kwargs)

    def initialize_weights(self, weights_initializer, bias_initializer):
        """Initialize and store weights."""
        wshapes = [
            [self.input_size, self.hidden_size[0]],
            [self.hidden_size[0], self.hidden_size[1]],
            [self.hidden_size[1], self.output_size]
        ]

        bshapes = [
            [1, self.hidden_size[0]],
            [1, self.hidden_size[1]],
            [1, self.output_size]
        ]

        self.weights = [init_weights(s, weights_initializer) for s in wshapes]
        self.biases = [init_weights(s, bias_initializer) for s in bshapes]

        self.trainable_variables = self.weights + self.biases
        self.loss = None

    def model(self, inputs):
        """Given a state vector, return the Q values of actions."""
        h1 = dense(inputs, self.weights[0], self.biases[0], tf.nn.relu)
        h2 = dense(h1, self.weights[1], self.biases[1], tf.nn.relu)

        out = dense(h2, self.weights[2], self.biases[2])

        return out

    def train_step(self, inputs, targets, actions_one_hot):
        """Update weights."""
        with tf.GradientTape() as tape:
            qvalues = tf.squeeze(self.model(inputs))
            preds = tf.reduce_sum(qvalues * actions_one_hot, axis=1)
            self.loss = tf.losses.mean_squared_error(targets, preds)

        grads = tape.gradient(self.loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))


class Memory(object):
    """Memory buffer for Experience Replay."""

    def __init__(self, max_size):
        """Initialize a buffer containing max_size experiences."""
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size=batch_size,
            replace=False
        )

        return [self.buffer[i] for i in index]

    def __len__(self):
        """Interface to access buffer length."""
        return len(self.buffer)


class Agent(object):
    """Deep Q-learning agent."""

    def __init__(self,
                 state_space_size,
                 action_space_size,
                 target_update_freq=50,
                 discount=0.99,
                 batch_size=5,
                 max_explore=1,
                 min_explore=0.05,
                 anneal_rate=(1 / 10000),
                 replay_memory_size=100,
                 replay_start_size=10):
        """Set parameters, initialize network."""
        self.action_space_size = action_space_size

        self.online_network = Network(state_space_size, action_space_size)
        self.target_network = Network(state_space_size, action_space_size)

        self.update_target_network()

        # training parameters
        self.target_update_freq = target_update_freq
        self.discount = discount
        self.batch_size = batch_size

        # policy during learning
        self.max_explore = max_explore + (anneal_rate * replay_start_size)
        self.min_explore = min_explore
        self.anneal_rate = anneal_rate
        self.steps = 0

        # replay memory
        self.memory = Memory(replay_memory_size)
        self.replay_start_size = replay_start_size
        self.experience_replay = Memory(replay_memory_size)

    def handle_episode_start(self):
        self.last_state, self.last_action = None, None

    def step(self, observation, training=True):
        """Observe state and rewards, select action.
        It is assumed that `observation` will be an object with
        a `state` vector and a `reward` float or integer. The reward
        corresponds to the action taken in the previous step.
        """
        last_state, last_action = self.last_state, self.last_action
        last_reward = observation.reward
        state = observation.state

        action = self.policy(state, training)

        if training:
            self.steps += 1

            if last_state is not None:
                experience = {
                    "state": last_state,
                    "action": last_action,
                    "reward": last_reward,
                    "next_state": state
                }

                self.memory.add(experience)

            if self.steps > self.replay_start_size:
                self.train_network()

                if self.steps % self.target_update_freq == 0:
                    self.update_target_network()

        self.last_state = state
        self.last_action = action

        return action

    def policy(self, state, training):
        """Epsilon-greedy policy for training, greedy policy otherwise."""
        explore_prob = self.max_explore - (self.steps * self.anneal_rate)
        explore = max(explore_prob, self.min_explore) > np.random.rand()

        if training and explore:
            action = np.random.randint(self.action_space_size)
        else:
            inputs = np.expand_dims(state, 0)
            qvalues = self.online_network.model(inputs)
            action = np.squeeze(np.argmax(qvalues, axis=-1))

        return action

    def update_target_network(self):
        """Update target network weights with current online network values."""
        variables = self.online_network.trainable_variables
        variables_copy = [tf.Variable(v) for v in variables]
        self.target_network.trainable_variables = variables_copy

    def train_network(self):
        """Update online network weights."""
        batch = self.memory.sample(self.batch_size)
        inputs = np.array([b["state"] for b in batch])
        actions = np.array([b["action"] for b in batch])
        rewards = np.array([b["reward"] for b in batch])
        next_inputs = np.array([b["next_state"] for b in batch])

        actions_one_hot = np.eye(self.action_space_size)[actions]

        next_qvalues = np.squeeze(self.target_network.model(next_inputs))
        targets = rewards + self.discount * np.amax(next_qvalues, axis=-1)

        self.online_network.train_step(inputs, targets, actions_one_hot)


class observation(object):
    def __init__(self, state, reward):
        self.state = state
        self.reward = reward


if __name__ == '__main__':
    environemt = 'CartPole-v1'
    env = gym.make(environemt)

    agent = Agent(4, env.action_space.n)

    episods = 1000
    retrain = True
    total_loss = []
    total_reward_list = []
    from tqdm import tqdm

    if retrain:
        for i in tqdm(range(episods)):
            state = env.reset()
            reward, penalties, epochs = 0, 0, 0
            total_reward = 0
            done = False
            obs = observation(state, reward)
            agent.handle_episode_start()
            while not done:
                # env.render()
                action = agent.step(obs)
                new_state, reward, done, info = env.step(action)

                total_reward += reward
                obs.state = new_state
                obs.reward = reward if not done else -reward
                epochs += 1

            if agent.online_network.loss != None:
                total_loss.append(agent.online_network.loss)
            total_reward_list.append(total_reward)

            if (i+1) % 100 == 0:
                np.savetxt(f'models/loss.csv', np.array(total_loss), delimiter=',')
                np.savetxt(f'models/reward.csv', np.array(total_reward_list), delimiter=',')
            if (i+1) % 1000 == 0 and i != 0:
                trained_weights = agent.online_network.weights
                for nlayer in range(len(trained_weights)):
                    np.savetxt(f'models/episode_{i+1}_weights_layer_{nlayer}.csv',
                               agent.online_network.weights[nlayer].numpy(),
                               delimiter=',')
                    np.savetxt(f'models/episode_{i+1}_biases_layer_{nlayer}.csv',
                               agent.online_network.biases[nlayer].numpy(),
                               delimiter=',')
        env.close()

    for nlayer in range(3):
        agent.online_network.weights[nlayer] = np.genfromtxt(f'models/episode_{episods}_weights_layer_{nlayer}.csv',
                                                             delimiter=',').astype('float32')
        agent.online_network.biases[nlayer] = np.genfromtxt(f'models/episode_{episods}_biases_layer_{nlayer}.csv',
                                                             delimiter=',').astype('float32')
    import time
    for t in range(10):
        state = env.reset()
        done = False
        while not done:
            env.render()
            state = np.matrix([state])
            action = agent.online_network.model(state)
            action = np.argmax(action.numpy())
            print(action)
            state, reward, done, info = env.step(action)

    env.close()
