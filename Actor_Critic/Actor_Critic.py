import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
n_rollout = 10
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


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []

        hidden_dims = 256
        self.feature_layer = nn.Sequential(nn.Linear(n_features, hidden_dims),
                                           nn.ReLU())

        self.fc_pi = nn.Linear(hidden_dims, n_actions)
        self.fc_v = nn.Linear(hidden_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x):
        x = self.feature_layer(x)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=-1)
        return prob

    def v(self, x):
        x = self.feature_layer(x)
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_next_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_next_lst.append(s_)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_next_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(
            a_lst), torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_next_lst, dtype=torch.float), torch.tensor(
            done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_next_batch, done_batch

    def train_net(self):
        s, a, r, s_, done = self.make_batch()
        td_target = r + gamma * self.v(s_) * done
        delta = td_target - self.v(s)

        pi = self.pi(s)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


def main():
    model = ActorCritic()
    print_interval = 20
    score = 0.0

    for n_epi in range(MAX_EPISODE):
        done = False
        s = env.reset()
        while not done:
            for t in range(n_rollout):
                if RENDER:
                    env.render()
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_next, r, done, info = env.step(a)
                model.put_data((s, a, r, s_next, done))

                s = s_next
                score += r

                if done:
                    break

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0
    env.close()


if __name__ == '__main__':
    main()
