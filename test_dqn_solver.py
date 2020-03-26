import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    environment = 'MountainCarContinuous-v0'
    env = gym.make(environment)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    model = tf.keras.models.load_model('MountainCarContinuous-v0_1_epsidode1025')
    total_reward_list = []

    for i in range(100):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            env.render()
            state = np.reshape(state, [1, observation_space])
            if steps in []: # [12, 13, 14, 15, 16] or steps in [30, 31, 32, 33, 34, 35]:
                action = 1
            elif steps in []: # [50, 51, 52, 53, 54] or steps in [70, 71, 72, 73, 74]:
                action = 0
            else:
                action = model.predict(state)[0] # np.argmax(model.predict(state)[0])
            state, reward, done, info = env.step(action)
            total_reward += reward
            os.system('clear')
            print(f'Reward: {reward}')
            print(f'Total_Reward: {total_reward}')
            print(f'Episode: {i}')
            print(f'Done: {done}')
            print(f'Steps: {steps}')
            steps += 1
        total_reward_list.append(total_reward)

    plt.figure()
    plt.plot(total_reward_list)
    plt.savefig(f'{environment}_eval_reward.png')
    plt.close()
    print(np.mean(total_reward_list))

