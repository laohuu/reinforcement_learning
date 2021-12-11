from Dueling_DQN import Dueling_DQN
from gym import make
import numpy as np

MEMORY_SIZE = 3000
ACTION_SPACE = 11

env = make('Pendulum-v1')
env = env.unwrapped
print("env.action_space :", env.action_space)
print("env.observation_space :", env.observation_space)
print("env.observation_space :", env.observation_space.high)
print("env.observation_space :", env.observation_space.low)

RL = Dueling_DQN(ACTION_SPACE, env.observation_space.shape[0],
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=200,
                 memory_size=MEMORY_SIZE,
                 e_greedy_increment=0.001
                 )

total_steps = 0

observation = env.reset()
while True:
    env.render()

    action = RL.choose_action(observation)

    f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # convert to [-2 ~ 2] float actions
    observation_, reward, done, info = env.step(np.array([f_action]))
    reward /= 10  # normalize to a range of (-1, 0). r = 0 when get upright
    # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
    # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.

    RL.store_transition(observation, action, reward, observation_)

    if RL.memory_counter > RL.memory_size:
        RL.learn()

    if done:
        break

    if total_steps - MEMORY_SIZE > 20000:  # stop game
        break

    observation = observation_
    total_steps += 1
env.close()
RL.plot_cost()
