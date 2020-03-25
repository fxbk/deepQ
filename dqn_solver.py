import numpy as np
import tensorflow as tf
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque

class dqn_solver(object):
    def __init__(self, env, model, memory_size, start_training_steps, batch_size, episodes, epsilon_max, epsilon_min,
                 gamma):
        self.memory_size = memory_size
        self.start_training_steps = start_training_steps
        self.batch_size = batch_size
        self.episodes = episodes
        self.episodes = episodes
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        self.head_memory = 0
        self.memory = deque(maxlen=self.memory_size)
        self.model = model

        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.steps = 0
        self.reward_list = []

    def remember(self, state_t_minus_1, action_t_minus_1, reward_t_minus_1, state_t, done):
        self.memory.append((state_t_minus_1, action_t_minus_1, reward_t_minus_1, state_t, done))
        self.head_memory += 1

    def experience_replay(self):
        if self.head_memory < self.batch_size:
            return
        index = np.random.choice(np.arange(min(self.head_memory - 1, self.memory_size)), size=self.batch_size,
                                 replace=False)
        return [self.memory[i] for i in index]

    def policy(self, state, episode):
        self.epsilon_max = self.epsilon_max * 0.95
        if np.random.rand() < max(self.epsilon_max, self.epsilon_min):
            action = np.random.randint(self.action_space)
        else:
            action = np.argmax(self.model.predict(state)[0])

        return action

    def train(self):
        for episode in tqdm(range(self.episodes)):
            state = env.reset()
            state = np.reshape(state, [1, self.observation_space])
            done = False
            total_reward = 0
            while not done:
                # env.render()
                action = self.policy(state, episode)
                new_state, reward, done, info = env.step(action)
                reward = reward if not done else -reward
                new_state = np.reshape(new_state, [1, self.observation_space])
                total_reward += reward
                self.remember(state, action, reward, new_state, done)

                if self.steps > self.start_training_steps:
                    for state_t_minus_1, action_t_minus_1, reward_t_minus_1, state_t, done_t in self.experience_replay():
                        q_update = reward_t_minus_1
                        if not done_t:
                            q_update = (reward_t_minus_1 + self.gamma * np.amax(self.model.predict(state_t)[0]))
                        q_values = self.model.predict(state_t_minus_1)
                        q_values[0][action_t_minus_1] = q_update
                        self.model.fit(state_t_minus_1, q_values, verbose=0)

                state = new_state
                self.steps += 1
            self.reward_list.append(total_reward)
            if episode % 5 == 0:
                reward_array = np.array(self.reward_list)
                np.savetxt(f'reward.csv', reward_array, delimiter=',')
                plt.figure()
                plt.plot(reward_array)
                plt.savefig(f'reward.png')
                plt.close()
            if episode != 0 and episode % 50 == 0:
                self.model.save(f'carpole_epsidode{episode}')


if __name__ == '__main__':
    model = tf.keras.Sequential([tf.keras.layers.Dense(24, input_shape=(4,), activation='relu'),
                                 tf.keras.layers.Dense(24, activation='relu'),
                                 tf.keras.layers.Dense(2, activation='linear')
                                 ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))

    environemt = 'CartPole-v1'
    env = gym.make(environemt)
    DQG_SOLVER = dqn_solver(env, model, memory_size=1000000, start_training_steps=100, batch_size=20, episodes=150,
                            epsilon_max=1, epsilon_min=0.01, gamma=0.95)
    DQG_SOLVER.train()
