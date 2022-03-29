import gym
import random
import collections

import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
lr_mu = 0.0005
lr_q = 0.001
gamma = 0.99
batch_size = 32
buffer_limit = 50000
tau = 0.005  # for target network soft update

MAX_EPISODE = 10000
RENDER = True

env = gym.make('Pendulum-v1')
# env = env.unwrapped
env.seed(1)
torch.manual_seed(1)

print("env.action_space :", env.action_space)
print("env.observation_space :", env.observation_space)

n_features = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]


# class NormalizedActions(gym.ActionWrapper):
#     def action(self, action):
#         low_bound = self.action_space.low
#         upper_bound = self.action_space.high
#
#         action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
#         # 将经过tanh输出的值重新映射回环境的真实值内
#         action = np.clip(action, low_bound, upper_bound)
#
#         return action
#
#     def reverse_action(self, action):
#         low_bound = self.action_space.low
#         upper_bound = self.action_space.high
#
#         # 因为激活函数使用的是tanh，这里将环境输出的动作正则化到（-1，1）
#
#         action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
#         action = np.clip(action, low_bound, upper_bound)
#
#         return action


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)  # 初始化buffer容量

    def put(self, transition):
        self.buffer.append(transition)  # 存入一个transition

    def sample(self, n):  # 取样
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_next_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_next_lst.append(s_)
            done_mask_lst.append([done_mask])

        return torch.tensor(numpy.array(s_lst), dtype=torch.float), torch.tensor(numpy.array(a_lst), dtype=torch.float), \
               torch.tensor(numpy.array(r_lst)), torch.tensor(numpy.array(s_next_lst), dtype=torch.float), \
               torch.tensor(numpy.array(done_mask_lst))

    def size(self):
        return len(self.buffer)


class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()

        hidden_dims = 128
        hidden_dims_2 = 64
        self.fc1 = nn.Linear(n_features, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims_2)
        self.fc_mu = nn.Linear(hidden_dims_2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * 2  # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        return mu


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        hidden_dims = 64
        hidden_dims_2 = 32
        self.fc_s = nn.Linear(n_features, hidden_dims)
        self.fc_a = nn.Linear(1, hidden_dims)
        self.fc_q = nn.Linear(hidden_dims * 2, hidden_dims_2)
        self.fc_out = nn.Linear(hidden_dims_2, n_actions)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s, a, r, s_next, done_mask = memory.sample(batch_size)

    target = r + gamma * q_target(s_next, mu_target(s_next)) * done_mask
    q_loss = F.smooth_l1_loss(q(s, a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    mu_loss = -q(s, mu(s)).mean()  # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()


def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param.data * tau + (1.0 - tau) * param_target.data)


def main():
    memory = ReplayBuffer()

    q, q_target = QNet(), QNet()
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    print_interval = 20

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    for n_epi in range(MAX_EPISODE):
        s = env.reset()
        done = False

        while not done:
            if RENDER:
                env.render()
            a = mu(torch.from_numpy(s).float())
            a = a.item() + ou_noise()[0]
            s_next, r, done, info = env.step([a])
            memory.put((s, a, r / 100.0, s_next, done))
            score += r
            s = s_next

        if memory.size() > 2000:
            for i in range(10):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q, q_target)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()
