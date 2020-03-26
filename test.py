import numpy as np
import gym
import os

enviroment = 'MountainCar-v0'
env = gym.make(enviroment)
Q = np.zeros((1, 1))
states = []

for i_episode in range(200):
    observation = env.reset()
    reward = 0
    done = False
    steps = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if not any((observation == x).all() for x in states):
            states.append(observation)

        os.system('clear')
        print(f'Reward: {reward}')
        print(f'Episode: {i_episode}')
        print(f'Done: {done}')
        print(f'Steps: {steps}')
        steps += 1

print(f"Number of states: {len(states)}")
print()

env.close()
