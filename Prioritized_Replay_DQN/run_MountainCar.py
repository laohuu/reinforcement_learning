from Prioritized_Replay_DQN import DQNPrioritizedReplay
from gym import make
import numpy as np

MEMORY_SIZE = 10000
ACTION_SPACE = 11

env = make('MountainCar-v0')
env = env.unwrapped
env.seed(21)

print("env.action_space :", env.action_space)
print("env.observation_space :", env.observation_space)
print("env.observation_space :", env.observation_space.high)
print("env.observation_space :", env.observation_space.low)

RL = DQNPrioritizedReplay(env.action_space.n, env.observation_space.shape[0],
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=200,
                          memory_size=MEMORY_SIZE,
                          e_greedy_increment=0.001
                          )

total_steps = 0
for i in range(500):
    observation = env.reset()
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        if done: reward = 10

        RL.store_transition(observation, action, reward, observation_)

        if RL.memory_counter > RL.memory_size:
            RL.learn()

        if done:
            break
        observation = observation_
        total_steps += 1

env.close()
RL.plot_cost()
