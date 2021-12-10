from RL_brain import DeepQNetwork
from gym import make

if __name__ == "__main__":
    env = make("CartPole-v1")
    # env = env.unwrapped
    print("env.action_space :", env.action_space)
    print("env.observation_space :", env.observation_space)
    print("env.observation_space :", env.observation_space.high)
    print("env.observation_space :", env.observation_space.low)

    RL = DeepQNetwork(env.action_space.n, env.observation_space.shape[0],
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      e_greedy_increment=0.001
                      )
    total_steps = 0
    for i in range(500):
        observation = env.reset()
        while True:
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, _ = env.step(action)

            # the smaller theta and closer to center the better
            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            RL.store_transition(observation, action, reward, observation_)

            if (RL.memory_counter > RL.memory_size) and (total_steps % 5 == 0):
                RL.learn()

            if done:
                break
            observation = observation_
            total_steps += 1
    env.close()
    RL.plot_cost()
