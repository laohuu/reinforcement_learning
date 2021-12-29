import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
gamma = 0.0002
learning_rate = 0.98
MAX_EPISODE = 10000
RENDER = True

env = gym.make('CartPole-v1')
env = env.unwrapped
env.seed(1)
torch.manual_seed(1)

print("env.action_space :", env.action_space)
print("env.observation_space :", env.observation_space)
# print("env.action_space.high :", env.action_space.high)
# print("env.action_space.low :", env.action_space.low)

n_features = env.observation_space.shape[0]
n_actions = env.action_space.n


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.episode = []

        hidden_units = 10
        self.fc_layer = nn.Sequential(nn.Linear(n_features, hidden_units),
                                      nn.Sigmoid(),
                                      nn.Linear(hidden_units, n_actions),
                                      nn.Softmax(dim=-1))
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.fc_layer(x)
        return x

    def put_data(self, item):
        self.episode.append(item)

    def train_net(self):
        reward = 0
        self.optimizer.zero_grad()
        for r, prob in self.episode[::-1]:
            reward = r + gamma * reward
            loss = -torch.log(prob) * reward
            loss.backward()

        self.optimizer.step()

        self.episode = []

    def choose_action(self, observation):
        prob_weights = self.forward(torch.from_numpy(observation))
        m = Categorical(prob_weights)
        action_idx = m.sample()
        return action_idx, prob_weights


def main():
    policy = Policy()
    score = 0.0
    print_interval = 20

    for n_epi in range(MAX_EPISODE):
        s = env.reset()
        done = False

        while not done:  # CartPole-v1 forced to terminates at 500 step.
            if RENDER:
                env.render()
            action, prob_weights = policy.choose_action(s)
            s_, r, done, info = env.step(action.item())
            policy.put_data((r, prob_weights[action]))
            s = s_
            score += r

        policy.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {}".format(n_epi, score / print_interval))
            score = 0.0
    env.close()


if __name__ == '__main__':
    main()
