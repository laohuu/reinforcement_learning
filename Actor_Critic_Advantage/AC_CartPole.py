import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from gym import make

np.random.seed(1)
torch.manual_seed(1)

# Superparameters
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000  # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

env = make('CartPole-v1')
env.seed(1)
env = env.unwrapped

print("env.action_space :", env.action_space)
print("env.observation_space :", env.observation_space)
print("env.observation_space.high :", env.observation_space.high)
print("env.observation_space.low :", env.observation_space.low)

N_F = env.observation_space.shape[0]
N_A = env.action_space.n


class PolicyNet(nn.Module):
    def __init__(self, n_actions, n_features):
        super(PolicyNet, self).__init__()
        hidden_units = 20
        self.fc_layer = nn.Sequential(nn.Linear(n_features, hidden_units),
                                      nn.Sigmoid(),
                                      nn.Linear(hidden_units, n_actions),
                                      nn.Softmax(dim=-1))

    def forward(self, x):
        output = self.fc_layer(x)
        return output


class CriticNet(nn.Module):
    def __init__(self, n_features):
        super(CriticNet, self).__init__()
        hidden_units = 20
        self.fc_layer = nn.Sequential(nn.Linear(n_features, hidden_units),
                                      nn.ReLU(),
                                      nn.Linear(hidden_units, 1))

    def forward(self, x):
        output = self.fc_layer(x)
        return output


class Actor(object):
    def __init__(self, n_features, n_actions, lr=0.001):
        self.actor_net = PolicyNet(n_actions, n_features)
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr

        self.optimizer = torch.optim.Adam(self.actor_net.parameters(),
                                          self.lr)
        self.cost_his = []

    def learn(self, s, a, td):
        state = torch.Tensor(s[np.newaxis, :])
        torch_acts = torch.as_tensor(a)
        torch_acts_one_hot = F.one_hot(torch_acts, num_classes=self.n_actions)
        torch_td_error = torch.Tensor(td).reshape(-1, 1).detach()
        all_act_prob = self.actor_net(state)

        exp_v = torch.log(all_act_prob) * torch_acts_one_hot * torch_td_error
        loss = torch.mean(-exp_v)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cost_his.append(loss.data.numpy())
        return exp_v

    def choose_action(self, observation):
        state = torch.Tensor(observation[np.newaxis, :])
        prob_weights = self.actor_net(state)
        action_idx = prob_weights.reshape(-1, ).multinomial(num_samples=1).numpy()[0]
        return action_idx


class Critic(object):
    def __init__(self, n_features, lr=0.01):
        self.critic_net = CriticNet(n_features)
        self.n_features = n_features
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.critic_net.parameters(),
                                          self.lr)
        self.cost_his = []
        self.loss_function = torch.nn.MSELoss()

    def learn(self, s, r, s_):
        s, s_ = torch.Tensor(s[np.newaxis, :]), torch.Tensor(s_[np.newaxis, :])

        v = self.critic_net(s)
        v_ = self.critic_net(s_).detach()
        td_error = r + GAMMA * v_ - v
        loss = self.loss_function(v, r + GAMMA * v_)
        # loss = torch.mean(torch.square(td_error))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cost_his.append(loss.data.numpy())

        return td_error


actor = Actor(n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(n_features=N_F,
                lr=LR_C)  # we need a good teacher, so the teacher should learn faster than the actor

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER:
            env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done:
            r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break
