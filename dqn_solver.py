import numpy as np
import tensorflow as tf
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque

class dqn_solver(object):
    def __init__(self, env, enviroment, model, memory_size, start_training_steps, batch_size, episodes, epsilon_max, epsilon_min,
                 gamma, max_steps, number, train_freq, aneal_rate):
        self.memory_size = memory_size
        self.start_training_steps = start_training_steps
        self.batch_size = batch_size
        self.episodes = episodes
        self.episodes = episodes
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.enviroment = environemt
        self.max_steps = max_steps
        self.head_memory = 0
        self.memory = deque(maxlen=self.memory_size)
        self.model = model
        self.number = number
        self.train_freq = train_freq
        self.aneal_rate = aneal_rate

        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
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
        self.epsilon_max = self.epsilon_max * self.aneal_rate
        if np.random.rand() < max(self.epsilon_max, self.epsilon_min):
            action = env.action_space.sample()
        else:
            action = self.model.predict(state)[0]# np.argmax(self.model.predict(state)[0])

        return action

    def train(self):
        done_list = []
        steps_until_done_list = []
        for episode in tqdm(range(self.episodes)):
            state = env.reset()
            state = np.reshape(state, [1, self.observation_space])
            done = False
            total_reward = 0
            for i in range(self.max_steps):
                # env.render()
                action = self.policy(state, episode)
                new_state, reward, done, info = env.step(action)
                # reward = reward if not done else -reward
                new_state = np.reshape(new_state, [1, self.observation_space])
                total_reward += reward
                self.remember(state, action, reward, new_state, done)

                if self.steps > self.start_training_steps and self.steps % self.train_freq == 0:
                    for state_t_minus_1, action_t_minus_1, reward_t_minus_1, state_t, done_t in self.experience_replay():
                        q_update = reward_t_minus_1
                        if not done_t:
                            q_update = (reward_t_minus_1 + self.gamma * np.amax(self.model.predict(state_t)[0]))
                        q_values = self.model.predict(state_t_minus_1)
                        q_values[0] = q_update
                        self.model.fit(state_t_minus_1, q_values, verbose=0)
                if done:
                    break

                state = new_state
                self.steps += 1
            steps_until_done_list.append(i)
            self.reward_list.append(total_reward)
            done_list.append(1 if done else 0)
            if episode % 2 == 0:
                done_array = np.array(done_list)
                np.savetxt(f'{self.enviroment}_{self.number}_done_count.csv', done_array, delimiter=',')
                plt.figure()
                plt.plot(done_array)
                plt.savefig(f'{self.enviroment}_{self.number}_done_count.png')
                plt.close()

                array = np.array(steps_until_done_list)
                np.savetxt(f'{self.enviroment}_{self.number}_steps_per_episode.csv', array, delimiter=',')
                plt.figure()
                plt.plot(array)
                plt.savefig(f'{self.enviroment}_{self.number}_steps_per_episode.png')
                plt.close()

                reward_array = np.array(self.reward_list)
                np.savetxt(f'{self.enviroment}_reward_{self.number}.csv', reward_array, delimiter=',')
                plt.figure()
                plt.plot(reward_array)
                plt.savefig(f'{self.enviroment}_reward_{self.number}.png')
                plt.close()
            if episode != 0 and episode % 5 == 0 or episode == self.episodes:
                self.model.save(f'{self.enviroment}_{self.number}_epsidode{episode}')


if __name__ == '__main__':
    environemt = 'MountainCarContinuous-v0'
    env = gym.make(environemt)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    model = tf.keras.Sequential([tf.keras.layers.Dense(24, input_shape=(observation_space,), activation='relu'),
                                 tf.keras.layers.Dense(24, activation='relu'),
                                 tf.keras.layers.Dense(action_space, activation='linear')
                                 ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))

    # DQG_SOLVER = dqn_solver(env, environemt, model, memory_size=1000000, start_training_steps=500, batch_size=20,
    #                         episodes=20, epsilon_max=1, epsilon_min=0.01, gamma=0.95, max_steps=500, train_freq=50,
    #                         aneal_rate=0.999, number=1)
    # DQG_SOLVER.train()

    # DQG_SOLVER = dqn_solver(env, environemt, model, memory_size=1000000, start_training_steps=10000, batch_size=20,
    #                         episodes=20, epsilon_max=1, epsilon_min=0.01, gamma=0.95, max_steps=1000, train_freq=20,
    #                         aneal_rate=0.9995, number=2)
    # DQG_SOLVER.train()

    DQG_SOLVER = dqn_solver(env, environemt, model, memory_size=1000000, start_training_steps=10000, batch_size=20,
                            episodes=30, epsilon_max=1, epsilon_min=0.01, gamma=0.95, max_steps=1000, train_freq=20,
                            aneal_rate=0.99999, number=3)
    DQG_SOLVER.train()

    DQG_SOLVER = dqn_solver(env, environemt, model, memory_size=1000000, start_training_steps=500000, batch_size=20,
                            episodes=30, epsilon_max=1, epsilon_min=0.01, gamma=0.95, max_steps=1000, train_freq=20,
                            aneal_rate=0.999999, number=4)
    DQG_SOLVER.train()

    DQG_SOLVER = dqn_solver(env, environemt, model, memory_size=10000000, start_training_steps=500000, batch_size=20,
                            episodes=30, epsilon_max=1, epsilon_min=0.01, gamma=0.90, max_steps=1000, train_freq=20,
                            aneal_rate=0.999999, number=5)
    DQG_SOLVER.train()

    DQG_SOLVER = dqn_solver(env, environemt, model, memory_size=1000000, start_training_steps=1000, batch_size=20,
                            episodes=50, epsilon_max=1, epsilon_min=0.01, gamma=0.95, max_steps=1000, train_freq=5,
                            aneal_rate=0.9995, number=6)
    DQG_SOLVER.train()

    DQG_SOLVER = dqn_solver(env, environemt, model, memory_size=1000000, start_training_steps=1000, batch_size=20,
                            episodes=20, epsilon_max=1, epsilon_min=0.01, gamma=0.95, max_steps=10000, train_freq=10,
                            aneal_rate=0.9995, number=7)
    DQG_SOLVER.train()

    DQG_SOLVER = dqn_solver(env, environemt, model, memory_size=1000000, start_training_steps=1000, batch_size=20,
                            episodes=20, epsilon_max=1, epsilon_min=0.01, gamma=0.99, max_steps=10000, train_freq=10,
                            aneal_rate=0.9995, number=8)
    DQG_SOLVER.train()

    DQG_SOLVER = dqn_solver(env, environemt, model, memory_size=1000000, start_training_steps=1000, batch_size=20,
                            episodes=20, epsilon_max=1, epsilon_min=0.01, gamma=0.99, max_steps=10000, train_freq=10,
                            aneal_rate=0.9995, number=9)
    DQG_SOLVER.train()


