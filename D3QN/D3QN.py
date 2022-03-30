import gym
import collections
import random

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
MAX_EPISODE = 10000
RENDER = True

env = gym.make('CartPole-v1')
env = env.unwrapped
env.seed(1)
torch.manual_seed(1)

print("env.action_space :", env.action_space)
print("env.observation_space :", env.observation_space)

n_features = env.observation_space.shape[0]
n_actions = env.action_space.n


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_next_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_next_lst.append(s_)
            done_mask_lst.append([done_mask])

        return torch.tensor(numpy.array(s_lst), dtype=torch.float), torch.tensor(numpy.array(a_lst)), \
               torch.tensor(numpy.array(r_lst)), torch.tensor(numpy.array(s_next_lst), dtype=torch.float), \
               torch.tensor(numpy.array(done_mask_lst))

    def size(self):
        return len(self.buffer)


class DQNDuelingNet(nn.Module):
    def __init__(self):
        super(DQNDuelingNet, self).__init__()
        hidden_dims = 128
        self.feature_layer = nn.Sequential(nn.Linear(n_features, hidden_dims),
                                           nn.ReLU())
        self.value_layer = nn.Linear(hidden_dims, 1)
        self.advantage_layer = nn.Linear(hidden_dims, n_actions)

    def forward(self, x):
        feature = self.feature_layer(x)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        avg_advantage = torch.mean(input=advantage, dim=-1, keepdim=True)
        q_values = value + (advantage - avg_advantage)
        return q_values


# Epsilon_Greedy_Exploration
# MAX_Greedy_Update
class Dueling_DQN:
    def __init__(self):
        # [target_net, evaluate_net]
        self.evaluate_net = DQNDuelingNet()
        self.target_net = type(self.evaluate_net)()
        self.target_net.load_state_dict(self.evaluate_net.state_dict())  # copy weights and stuff

        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(),
                                          learning_rate)
        self.memory = ReplayBuffer()

    def train(self):
        s, a, r, s_, done_mask = self.memory.sample(batch_size)

        q_out = self.evaluate_net(s)
        q_a = q_out.gather(1, a)

        # 与Dueling DQN的不同之处
        # max_q_prime = torch.max(self.target_net(s_), dim=1, keepdim=True).values
        #  target = r + gamma * max_q_prime * done_mask
        q_target_next = self.target_net(s_).detach()
        q_eval_next = self.evaluate_net(s_).detach()
        q_next = q_target_next.gather(1, q_eval_next.argmax(axis=1).reshape(-1, 1))
        target = r + gamma * q_next * done_mask

        loss = F.smooth_l1_loss(q_a, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample_action(self, obs, epsilon):
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            out = self.evaluate_net(obs)
            return out.argmax().item()


def main():
    trainer = Dueling_DQN()

    print_interval = 20
    score = 0.0

    for n_epi in range(MAX_EPISODE):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            if RENDER:
                env.render()
            a = trainer.sample_action(torch.from_numpy(s).float(), epsilon)
            s_, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            trainer.memory.put((s, a, r / 100.0, s_, done_mask))
            s = s_

            score += r
            if done:
                break

        if trainer.memory.size() > 2000:
            trainer.train()

        if n_epi % print_interval == 0 and n_epi != 0:
            trainer.target_net.load_state_dict(trainer.evaluate_net.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score / print_interval, trainer.memory.size(), epsilon * 100))
            score = 0.0
    env.close()


if __name__ == '__main__':
    main()
