# import numpy as np
# import torch
# from torch import nn
# import torch.nn.functional as F
# from gym import make
#
# np.random.seed(1)
# torch.manual_seed(1)
#
#
# class PolicyNet(nn.Module):
#     def __init__(self, n_features):
#         super(PolicyNet, self).__init__()
#         hidden_units = 30
#         self.feature_layer = nn.Sequential(nn.Linear(n_features, hidden_units),
#                                            nn.ReLU())
#         self.mu_layer = nn.Linear(hidden_units, 1)
#         self.sigma_layer = nn.Sequential(nn.Linear(hidden_units, 1),
#                                          nn.ReLU())
#
#     def forward(self, x):
#         feature = self.feature_layer(x)
#         mu = self.mu_layer(feature)
#         sigma = self.sigma_layer(feature)
#         return mu, sigma
#
#
# class CriticNet(nn.Module):
#     def __init__(self, n_features):
#         super(CriticNet, self).__init__()
#         hidden_units = 20
#         self.fc_layer = nn.Sequential(nn.Linear(n_features, hidden_units),
#                                       nn.ReLU(),
#                                       nn.Linear(hidden_units, 1))
#
#     def forward(self, x):
#         output = self.fc_layer(x)
#         return output
#
#
# class Actor(object):
#     def __init__(self, n_features, action_bound, lr=0.0001):
#         self.actor_net = PolicyNet(n_features)
#         self.n_features = n_features
#         self.action_bound = action_bound
#         self.lr = lr
#
#         self.optimizer = torch.optim.Adam(self.actor_net.parameters(),
#                                           self.lr)
#
#     def learn(self, s, a, td):
#         state = torch.Tensor(s[np.newaxis, :])
#         torch_acts = torch.as_tensor(a)
#         torch_td_error = torch.Tensor(td).reshape(-1, 1).detach()
#
#         mu, sigma = self.actor_net(state)
#         mu, sigma = torch.squeeze(mu * 2), torch.squeeze(sigma + 0.001)
#         normal_dist = torch.distributions.Normal(mu, sigma)
#
#         log_prob = normal_dist.log_prob(torch_acts)
#         exp_v = log_prob * torch_td_error
#         exp_v += 0.01 * normal_dist.entropy()
#
#         loss = torch.mean(-exp_v)
#         self.optimizer.zero_grad()
#
#         loss.backward()
#         self.optimizer.step()
#         return exp_v
#
#     def choose_action(self, s):
#         state = torch.Tensor(s[np.newaxis, :])
#         mu, sigma = self.actor_net(state)
#         mu, sigma = torch.squeeze(mu * 2), torch.squeeze(sigma + 0.001)
#
#         normal_dist = torch.distributions.Normal(mu, sigma)
#         action = torch.clamp(normal_dist.sample(), torch.Tensor(self.action_bound[0]),
#                              torch.Tensor(self.action_bound[1]))
#
#         return action
#
#
# class Critic(object):
#     def __init__(self, n_features, lr=0.01):
#         self.critic_net = CriticNet(n_features)
#         self.n_features = n_features
#         self.lr = lr
#         self.optimizer = torch.optim.Adam(self.critic_net.parameters(),
#                                           self.lr)
#
#         self.loss_function = torch.nn.MSELoss()
#
#     def learn(self, s, r, s_):
#         s, s_ = torch.Tensor(s[np.newaxis, :]), torch.Tensor(s_[np.newaxis, :])
#
#         v = self.critic_net(s)
#         v_ = self.critic_net(s_).detach()
#         td_error = r + GAMMA * v_ - v
#         # loss = self.loss_function(v, r + GAMMA * v_)
#         loss = torch.mean(torch.square(td_error))
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         return td_error
#
#
# MAX_EPISODE = 1000
# MAX_EP_STEPS = 200
# DISPLAY_REWARD_THRESHOLD = -100  # renders environment if total episode reward is greater then this threshold
# RENDER = False  # rendering wastes time
# GAMMA = 0.9
# LR_A = 0.001  # learning rate for actor
# LR_C = 0.01  # learning rate for critic
#
# env = make('Pendulum-v1')
# env.seed(1)  # reproducible
# env = env.unwrapped
#
# N_S = env.observation_space.shape[0]
# A_BOUND = env.action_space.high
#
# actor = Actor(n_features=N_S, lr=LR_A, action_bound=[-A_BOUND, A_BOUND])
# critic = Critic(n_features=N_S, lr=LR_C)
#
# for i_episode in range(MAX_EPISODE):
#     s = env.reset()
#     t = 0
#     ep_rs = []
#     while True:
#         # if RENDER:
#         env.render()
#         a = actor.choose_action(s)
#
#         s_, r, done, info = env.step(a)
#         r /= 10
#
#         td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
#         actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
#
#         s = s_
#         t += 1
#         ep_rs.append(r)
#         if t > MAX_EP_STEPS:
#             ep_rs_sum = sum(ep_rs)
#             if 'running_reward' not in globals():
#                 running_reward = ep_rs_sum
#             else:
#                 running_reward = running_reward * 0.9 + ep_rs_sum * 0.1
#             if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
#             print("episode:", i_episode, "  reward:", int(running_reward))
#             break
